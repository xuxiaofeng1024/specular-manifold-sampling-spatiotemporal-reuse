#include <thread>
#include <mutex>

#include <enoki/morton.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/spiral.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <mutex>
#include <fstream>
#include <sstream>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fresolver.h>

NAMESPACE_BEGIN(mitsuba)

//
class BlobWriter {
public:
    BlobWriter(const std::string& filename)
        : f(filename, std::ios::out | std::ios::binary) {
    }

    template <typename Type>
    typename std::enable_if<std::is_standard_layout<Type>::value, BlobWriter&>::type
        operator << (Type Element) {
        Write(&Element, 1);
        return *this;
    }

    // CAUTION: This function may break down on big-endian architectures.
    //          The ordering of bytes has to be reverted then.
    template <typename T>
    void Write(T* Src, size_t Size) {
        f.write(reinterpret_cast<const char*>(Src), Size * sizeof(T));
    }
    std::ofstream f;
};


class BlobReader {
public:
    BlobReader(const std::string& filename) : f(filename, std::ios::in | std::ios::binary) {}

    template <typename Type>
    typename std::enable_if<std::is_standard_layout<Type>::value, BlobReader&>::type
        operator >> (Type& Element) {
        Read(&Element, 1);
        return *this;
    }

    // CAUTION: This function may break down on big-endian architectures.
    //          The ordering of bytes has to be reverted then.
    template <typename T>
    void Read(T* Dest, size_t Size) {
        f.read(reinterpret_cast<char*>(Dest), Size * sizeof(T));
    }

    bool isValid() const {
        return (bool)(f);
    }

    std::ifstream f;
};

template <class Ptr>
void print(Ptr begptr, Ptr endptr) {
    for (Ptr ptr = begptr; ptr != endptr; ++ptr) {
        auto value = *ptr;
        std::cout << value << std::endl;
    }
}


// -----------------------------------------------------------------------------

MTS_VARIANT SamplingIntegrator<Float, Spectrum>::SamplingIntegrator(const Properties &props)
    : Base(props) {

        m_block_size = (uint32_t) props.size_("block_size", 0);
        uint32_t block_size = math::round_to_power_of_two(m_block_size);
        if (m_block_size > 0 && block_size != m_block_size) {
            Log(Warn, "Setting block size from %i to next higher power of two: %i", m_block_size,
                block_size);
            m_block_size = block_size;
        }

        m_samples_per_pass = (uint32_t) props.size_("samples_per_pass", (size_t) -1);
        m_timeout = props.float_("timeout", -1.f);

        /// Disable direct visibility of emitters if needed
        m_hide_emitters = props.bool_("hide_emitters", false);

        m_glint_diff_scale_factor_clamp = props.float_("glint_diff_scale_factor_clamp", 0.f);

        m_reuse = props.bool_("reuse", false);
        m_spatial_reuse_size = (uint32_t)props.size_("spatial_reuse_size", 4);
        m_spatial_reuse_unique = props.bool_("spatial_reuse_unique", false);
        m_pixel_reuse_unique = props.bool_("pixel_reuse_unique", false);
        m_spatial_merge_threshold = 1e-4f;
        m_pixel_merge_threshold = 1e-5f;

        auto fs = Thread::thread()->file_resolver();
        fs::path current_file_path = fs->resolve(props.string("current_file_path_reuse"));
        m_current_file_path_reuse = current_file_path.filename().string();

        fs::path temporal_file_path = fs->resolve(props.string("temporal_file_path_reuse"));
        m_temporal_file_path_reuse = temporal_file_path.filename().string();

        m_temporal_reuse =  props.bool_("temporal_reuse", false);

    }

MTS_VARIANT SamplingIntegrator<Float, Spectrum>::~SamplingIntegrator() { }

MTS_VARIANT void SamplingIntegrator<Float, Spectrum>::cancel() {
    m_stop = true;
}

MTS_VARIANT std::vector<std::string> SamplingIntegrator<Float, Spectrum>::aov_names() const {
    return { };
}

MTS_VARIANT bool SamplingIntegrator<Float, Spectrum>::render(Scene *scene, Sensor *sensor) {
    ScopedPhase sp(ProfilerPhase::Render);
    m_stop = false;

    ref<Film> film = sensor->film();
    ScalarVector2i film_size = film->crop_size();
    size_t total_spp = sensor->sampler()->sample_count();
    size_t samples_per_pass = (m_samples_per_pass == (size_t) -1)
                               ? total_spp : std::min((size_t) m_samples_per_pass, total_spp);
    if ((total_spp % samples_per_pass) != 0)
        Throw("sample_count (%d) must be a multiple of samples_per_pass (%d).",
              total_spp, samples_per_pass);

    size_t n_passes = (total_spp + samples_per_pass - 1) / samples_per_pass;

    std::vector<std::string> channels = aov_names();
    bool has_aovs = !channels.empty();

    // Insert default channels and set up the film
    for (size_t i = 0; i < 5; ++i)
        channels.insert(channels.begin() + i, std::string(1, "XYZAW"[i]));
    film->prepare(channels);

    if constexpr (!is_cuda_array_v<Float>) {
        /// Render on the CPU using a spiral pattern
        size_t n_threads = __global_thread_count;
        Log(Info, "Starting render job (%ix%i, %i sample%s,%s %i thread%s)",
            film_size.x(), film_size.y(),
            total_spp, total_spp == 1 ? "" : "s",
            n_passes > 1 ? tfm::format(" %d passes,", n_passes) : "",
            n_threads, n_threads == 1 ? "" : "s");

        if (m_timeout > 0.f)
            Log(Info, "Timeout specified: %.2f seconds.", m_timeout);

        // Find a good block size to use for splitting up the total workload.
        if (m_block_size == 0) {
            uint32_t block_size = MTS_BLOCK_SIZE;
            while (true) {
                if (block_size == 1 || hprod((film_size + block_size - 1) / block_size) >= n_threads)
                    break;
                block_size /= 2;
            }
            m_block_size = block_size;
        }

        Spiral spiral(film, m_block_size, n_passes);

        ThreadEnvironment env;
        ref<ProgressReporter> progress = new ProgressReporter("Rendering");
        std::mutex mutex;

        // Total number of blocks to be handled, including multiple passes.
        size_t total_blocks = spiral.block_count() * n_passes,
               blocks_done = 0;

        m_render_timer.reset();
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, total_blocks, 1),
            [&](const tbb::blocked_range<size_t> &range) {
                ScopedSetThreadEnvironment set_env(env);
                ref<Sampler> sampler = sensor->sampler()->clone();
                ref<ImageBlock> block = new ImageBlock(m_block_size, channels.size(),
                                                       film->reconstruction_filter(),
                                                       !has_aovs);
                scoped_flush_denormals flush_denormals(true);
                std::unique_ptr<Float[]> aovs(new Float[channels.size()]);


                /* add vector structure to store PixelCacheInfo records within a block*/
                std::vector<PixelCacheInfo> block_records_w;

                /* add vector structure to load PixelCacheInfo records within a block*/
                std::vector<PixelCacheInfo> block_records_r;
                //Temporal reuse
                std::vector<PixelCacheInfo> block_records_r_temporal;

                // For each block
                for (auto i = range.begin(); i != range.end() && !should_stop(); ++i) {
                    auto [offset, size, block_id] = spiral.next_block();
                    Assert(hprod(size) != 0);
                    block->set_size(size);
                    block->set_offset(offset);

                    // Ensure that the sample generation is fully deterministic
                    sampler->seed(block_id);

/*
                    render_block(scene, sensor, sampler, block,
                                 aovs.get(), samples_per_pass);
*/
                    block_records_w.clear();
                    size_t block_records_w_size = size[0] * size[1];
                    block_records_w.reserve(block_records_w_size); 
                    block_records_r.clear();
                    block_records_r_temporal.clear();
                    //for Spatial Reuse 
                    if(m_reuse) {                   
                        std::ostringstream filename_r;
                        filename_r << m_current_file_path_reuse <<"/block_records_w_" << offset[0] << "_" << offset[1] << ".txt";
                        load(filename_r.str(), block_records_r);

                        //for Temporal reuse
                        if(m_temporal_reuse){
                            std::ostringstream filename_r_temporal;
                            filename_r_temporal << m_temporal_file_path_reuse <<"/block_records_w_" << offset[0] << "_" << offset[1] << ".txt";

                            load(filename_r_temporal.str(), block_records_r_temporal);

                            for (size_t i = 0; i < block_records_r.size(); ++i) {
                                block_records_r[i].merge_unique(m_pixel_merge_threshold, block_records_r_temporal[i]);
                            }
                        }

                    }

                    render_block_reuse(scene, sensor, sampler, block,
                                       aovs.get(), block_records_w, block_records_r, samples_per_pass); 

                    if(!m_reuse){
                        std::ostringstream filename_w;
                        filename_w << m_current_file_path_reuse << "/block_records_w_" << offset[0] << "_" << offset[1] << ".txt";
                        dump(filename_w.str(), block_records_w);
                    }

                    film->put(block);

                    /* Critical section: update progress bar */ {
                        std::lock_guard<std::mutex> lock(mutex);
                        blocks_done++;
                        progress->update(blocks_done / (ScalarFloat) total_blocks);
                    }
                }
            }
        );
        if (!m_stop) {
            if (m_timeout > 0) {
                Float spp_f = blocks_done;
                spp_f /= spiral.block_count();
                int spp = int(floor(spp_f));

                Log(Info, "Rendering finished. Computed %d spp and took %s.",
                    spp, util::time_string(m_render_timer.value(), true));
            } else {
                Log(Info, "Rendering finished. (took %s)",
                    util::time_string(m_render_timer.value(), true));
            }
        }

    } else {
        ref<Sampler> sampler = sensor->sampler();

        ScalarFloat diff_scale_factor = rsqrt((ScalarFloat) sampler->sample_count());
        diff_scale_factor = max(diff_scale_factor, m_glint_diff_scale_factor_clamp);
        ScalarUInt32 total_sample_count = hprod(film_size) * (uint32_t) samples_per_pass;
        if (sampler->wavefront_size() != total_sample_count)
            sampler->seed(arange<UInt64>(total_sample_count));

        UInt32 idx = arange<UInt32>(total_sample_count);
        if (samples_per_pass != 1)
            idx /= (uint32_t) samples_per_pass;

        ref<ImageBlock> block = new ImageBlock(film_size, channels.size(),
                                               film->reconstruction_filter(),
                                               !has_aovs);
        block->clear();
        Vector2f pos = Vector2f(Float(idx % uint32_t(film_size[0])),
                                Float(idx / uint32_t(film_size[0])));
        std::vector<Float> aovs(channels.size());

        for (size_t i = 0; i < n_passes; i++)
            render_sample(scene, sensor, sampler, block, aovs.data(),
                          pos, diff_scale_factor);

        film->put(block);

        if (!m_stop)
            Log(Info, "Rendering finished. (took %s)",
                util::time_string(m_render_timer.value(), true));
    }

    return !m_stop;
}

MTS_VARIANT void SamplingIntegrator<Float, Spectrum>::render_block(const Scene *scene,
                                                                   const Sensor *sensor,
                                                                   Sampler *sampler,
                                                                   ImageBlock *block,
                                                                   Float *aovs,
                                                                   size_t sample_count_) const {
    block->clear();
    uint32_t pixel_count  = (uint32_t)(m_block_size * m_block_size),
             sample_count = (uint32_t)(sample_count_ == (size_t) -1
                                           ? sampler->sample_count()
                                           : sample_count_);

    ScalarFloat diff_scale_factor = rsqrt((ScalarFloat) sampler->sample_count());
    diff_scale_factor = max(diff_scale_factor, m_glint_diff_scale_factor_clamp);

    if constexpr (!is_array_v<Float>) {
        for (uint32_t i = 0; i < pixel_count && !should_stop(); ++i) {
            ScalarPoint2u pos = enoki::morton_decode<ScalarPoint2u>(i);
            if (any(pos >= block->size()))
                continue;

            pos += block->offset();
            for (uint32_t j = 0; j < sample_count && !should_stop(); ++j) {
                render_sample(scene, sensor, sampler, block, aovs,
                              pos, diff_scale_factor);
            }
        }
    } else if constexpr (is_array_v<Float> && !is_cuda_array_v<Float>) {
        for (auto [index, active] : range<UInt32>(pixel_count * sample_count)) {
            if (should_stop())
                break;
            Point2u pos = enoki::morton_decode<Point2u>(index / UInt32(sample_count));
            active &= !any(pos >= block->size());
            pos += block->offset();
            render_sample(scene, sensor, sampler, block, aovs, pos, diff_scale_factor, active);
        }
    } else {
        ENOKI_MARK_USED(scene);
        ENOKI_MARK_USED(sensor);
        ENOKI_MARK_USED(aovs);
        ENOKI_MARK_USED(diff_scale_factor);
        ENOKI_MARK_USED(pixel_count);
        ENOKI_MARK_USED(sample_count);
        Throw("Not implemented for CUDA arrays.");
    }
}

// for spatial reuse, merging based on Morton code characteristics (reuse_size 0,4,16,64,256,1024) 
MTS_VARIANT void
SamplingIntegrator<Float, Spectrum>::spatial_merge_info(std::vector<PixelCacheInfo> &block_records_r, size_t reuse_size) {
    if (block_records_r.size() < reuse_size || reuse_size <= 0) {
        return;
    }
    for (size_t i = 0; i <= block_records_r.size() - reuse_size; i += reuse_size) {
        PixelCacheInfo spatial_merge = block_records_r[i];
        for (size_t j = 1; j < reuse_size; j++) {
            spatial_merge.merge(block_records_r[i + j]);
        }
        for (size_t j = 0; j < reuse_size; j++) {
            if (m_spatial_reuse_unique)
                block_records_r[i + j].merge_unique(m_spatial_merge_threshold, spatial_merge);
            else
                block_records_r[i + j] = spatial_merge;
        }
    }
}

MTS_VARIANT void SamplingIntegrator<Float, Spectrum>::render_block_reuse(const Scene *scene,
                                                                         const Sensor *sensor,
                                                                         Sampler *sampler,
                                                                         ImageBlock *block,
                                                                         Float *aovs,
                                                                         std::vector<PixelCacheInfo> &block_records_w,
                                                                         std::vector<PixelCacheInfo> &block_records_r,
                                                                         size_t sample_count_){
    block->clear();
    uint32_t pixel_count  = (uint32_t)(m_block_size * m_block_size),
             sample_count = (uint32_t)(sample_count_ == (size_t) -1
                                           ? sampler->sample_count()
                                           : sample_count_);

    ScalarFloat diff_scale_factor = rsqrt((ScalarFloat) sampler->sample_count());
    diff_scale_factor = max(diff_scale_factor, m_glint_diff_scale_factor_clamp);

    //Spatial reuse, expand the set of solutions
    if (m_reuse && m_spatial_reuse_size > 0)
        spatial_merge_info(block_records_r, m_spatial_reuse_size); 

    if constexpr (!is_array_v<Float>){
        for (uint32_t i = 0; i < pixel_count && !should_stop(); ++i) {
            ScalarPoint2u pos = enoki::morton_decode<ScalarPoint2u>(i);
            PixelCacheInfo pixel_records_w;
            if (any(pos >= block->size())){
                block_records_w.push_back(pixel_records_w);
                continue;
            }

            pos += block->offset();
            for (uint32_t j = 0; j < sample_count && !should_stop(); ++j) {
                if (!m_reuse && j==0) 
                    render_sample_reuse(scene, sensor, sampler, block, aovs,
                                        pos, diff_scale_factor, pixel_records_w, block_records_r[i], true);
                if (!m_reuse && j > 0){
                    PixelCacheInfo pixel_records_w_temp;
                    render_sample_reuse(scene, sensor, sampler, block, aovs,
                                        pos, diff_scale_factor, pixel_records_w_temp, block_records_r[i], false);

                    //de-duplicate
                    if (m_pixel_reuse_unique)
                        pixel_records_w.merge_unique(m_pixel_merge_threshold, pixel_records_w_temp);
                    //don't de-duplicate
                    if (!m_pixel_reuse_unique)
                        pixel_records_w.merge(pixel_records_w_temp);
                }
                if (m_reuse) 
                    render_sample_reuse(scene, sensor, sampler, block, aovs,
                                        pos, diff_scale_factor, pixel_records_w, block_records_r[i], false);
            }
            block_records_w.push_back(pixel_records_w);
        }
    } else if constexpr (is_array_v<Float> && !is_cuda_array_v<Float>) {
        for (auto [index, active] : range<UInt32>(pixel_count * sample_count)) {
            if (should_stop())
                break;
            Point2u pos = enoki::morton_decode<Point2u>(index / UInt32(sample_count));
            active &= !any(pos >= block->size());
            pos += block->offset();
            render_sample(scene, sensor, sampler, block, aovs, pos, diff_scale_factor, active);
        }
    } else {
        ENOKI_MARK_USED(scene);
        ENOKI_MARK_USED(sensor);
        ENOKI_MARK_USED(aovs);
        ENOKI_MARK_USED(diff_scale_factor);
        ENOKI_MARK_USED(pixel_count);
        ENOKI_MARK_USED(sample_count);
        Throw("Not implemented for CUDA arrays.");
    }
}
MTS_VARIANT void SamplingIntegrator<Float, Spectrum>::render_sample(
    const Scene *scene, const Sensor *sensor, Sampler *sampler, ImageBlock *block,
    Float *aovs, const Vector2f &pos, ScalarFloat diff_scale_factor, Mask active) const {
    Vector2f position_sample = pos + sampler->next_2d(active);

    Point2f aperture_sample(.5f);
    if (sensor->needs_aperture_sample())
        aperture_sample = sampler->next_2d(active);

    Float time = sensor->shutter_open();
    if (sensor->shutter_open_time() > 0.f)
        time += sampler->next_1d(active) * sensor->shutter_open_time();

    Float wavelength_sample = sampler->next_1d(active);

    Vector2f adjusted_position =
        (position_sample - sensor->film()->crop_offset()) /
        sensor->film()->crop_size();

    auto [ray, ray_weight] = sensor->sample_ray_differential(
        time, wavelength_sample, adjusted_position, aperture_sample);

    ray.scale_differential(diff_scale_factor);

    const Medium *medium = sensor->medium();
    std::pair<Spectrum, Mask> result = sample(scene, sampler, ray, medium, aovs + 5, active);
    result.first = ray_weight * result.first;

    UnpolarizedSpectrum spec_u = depolarize(result.first);

    Color3f xyz;
    if constexpr (is_monochromatic_v<Spectrum>) {
        xyz = spec_u.x();
    } else if constexpr (is_rgb_v<Spectrum>) {
        xyz = srgb_to_xyz(spec_u, active);
    } else {
        static_assert(is_spectral_v<Spectrum>);
        xyz = spectrum_to_xyz(spec_u, ray.wavelengths, active);
    }

    aovs[0] = xyz.x();
    aovs[1] = xyz.y();
    aovs[2] = xyz.z();
    aovs[3] = select(result.second, Float(1.f), Float(0.f));
    aovs[4] = 1.f;

    block->put(position_sample, aovs, active);
}

MTS_VARIANT void SamplingIntegrator<Float, Spectrum>::render_sample_reuse(
    const Scene *scene, const Sensor *sensor, Sampler *sampler, ImageBlock *block,
    Float *aovs, const Vector2f &pos, ScalarFloat diff_scale_factor, 
    PixelCacheInfo &pixel_records_w,
    PixelCacheInfo &pixel_records_r,
    bool first_not_reuse,
    Mask active){
    Vector2f position_sample;
    //For the first time, emitting light from the center of the pixel.
    if (first_not_reuse){
        position_sample = pos + (0.5f);
    }
    else { 
        position_sample = pos + sampler->next_2d(active);
    }

    Point2f aperture_sample(.5f);
    if (sensor->needs_aperture_sample())
        aperture_sample = sampler->next_2d(active);

    Float time = sensor->shutter_open();
    if (sensor->shutter_open_time() > 0.f)
        time += sampler->next_1d(active) * sensor->shutter_open_time();

    Float wavelength_sample = sampler->next_1d(active);

    Vector2f adjusted_position =
        (position_sample - sensor->film()->crop_offset()) /
        sensor->film()->crop_size();

    auto [ray, ray_weight] = sensor->sample_ray_differential(
        time, wavelength_sample, adjusted_position, aperture_sample);

    ray.scale_differential(diff_scale_factor);

    const Medium *medium = sensor->medium();
    std::pair<Spectrum, Mask> result = sample_reuse(scene, sampler, ray, pixel_records_w, pixel_records_r, medium, aovs + 5, active);
    result.first = ray_weight * result.first;

    UnpolarizedSpectrum spec_u = depolarize(result.first);

    Color3f xyz;
    if constexpr (is_monochromatic_v<Spectrum>) {
        xyz = spec_u.x();
    } else if constexpr (is_rgb_v<Spectrum>) {
        xyz = srgb_to_xyz(spec_u, active);
    } else {
        static_assert(is_spectral_v<Spectrum>);
        xyz = spectrum_to_xyz(spec_u, ray.wavelengths, active);
    }

    aovs[0] = xyz.x();
    aovs[1] = xyz.y();
    aovs[2] = xyz.z();
    aovs[3] = select(result.second, Float(1.f), Float(0.f));
    aovs[4] = 1.f;

    block->put(position_sample, aovs, active);
}

MTS_VARIANT std::pair<Spectrum, typename SamplingIntegrator<Float, Spectrum>::Mask>
SamplingIntegrator<Float, Spectrum>::sample(const Scene * /* scene */,
                                            Sampler * /* sampler */,
                                            const RayDifferential3f & /* ray */,
                                            const Medium * /* medium */,
                                            Float * /* aovs */,
                                            Mask /* active */) const {
    NotImplementedError("sample");
}


MTS_VARIANT std::pair<Spectrum, typename SamplingIntegrator<Float, Spectrum>::Mask>
SamplingIntegrator<Float, Spectrum>::sample_reuse(const Scene * /* scene */,
                                            Sampler * /* sampler */,
                                            const RayDifferential3f & /* ray */,
                                            PixelCacheInfo &pixel_records_w,
                                            PixelCacheInfo &pixel_records_r,
                                            const Medium * /* medium */,
                                            Float * /* aovs */,
                                            Mask /* active */){
    NotImplementedError("sample_reuse");
}

//Write the solutions to file
MTS_VARIANT void 
SamplingIntegrator<Float, Spectrum>::dump(std::string filename_w, std::vector<PixelCacheInfo> &block_records_w){
    BlobWriter saver(filename_w);
    size_t block_records_w_realsize = block_records_w.size();
    saver << block_records_w_realsize;
    for(size_t i = 0; i < block_records_w_realsize; ++i){

        saver 
            << block_records_w[i].si_point[0] << block_records_w[i].si_point[1] << block_records_w[i].si_point[2] << block_records_w[i].si_point[3] << block_records_w[i].si_point[4];
        saver 
            << block_records_w[i].m[0] << block_records_w[i].m[1] << block_records_w[i].m[2] << block_records_w[i].m[3] << block_records_w[i].m[4];
        if(block_records_w[i].m[0] != 0) {
            saver.f.write(reinterpret_cast<const char*>(block_records_w[i].points0.data()), block_records_w[i].m[0] * sizeof(Point3f));
        }
        if(block_records_w[i].m[1] != 0) {
            saver.f.write(reinterpret_cast<const char*>(block_records_w[i].points1.data()), block_records_w[i].m[1] * sizeof(Point3f));
        }
        if(block_records_w[i].m[2] != 0) {
            saver.f.write(reinterpret_cast<const char*>(block_records_w[i].points2.data()), block_records_w[i].m[2] * sizeof(Point3f));
        }
        if(block_records_w[i].m[3] != 0) {
            saver.f.write(reinterpret_cast<const char*>(block_records_w[i].points3.data()), block_records_w[i].m[3] * sizeof(Point3f));
        }
        if(block_records_w[i].m[4] != 0) {
            saver.f.write(reinterpret_cast<const char*>(block_records_w[i].points4.data()), block_records_w[i].m[4] * sizeof(Point3f));
        }
    }
}

//Load all solution information from file
MTS_VARIANT void 
SamplingIntegrator<Float, Spectrum>::load(std::string filename_r, std::vector<PixelCacheInfo> &block_records_r){
    BlobReader reader(filename_r);

    size_t block_records_r_size;
    reader >> block_records_r_size;
    block_records_r.reserve(block_records_r_size);
    for (size_t i = 0; i < block_records_r_size; ++i){
        PixelCacheInfo pixel_records_r;
        reader 
            >> pixel_records_r.si_point[0] >> pixel_records_r.si_point[1] >> pixel_records_r.si_point[2] >> pixel_records_r.si_point[3] >> pixel_records_r.si_point[4];
        reader 
            >> pixel_records_r.m[0] >> pixel_records_r.m[1] >> pixel_records_r.m[2] >> pixel_records_r.m[3] >> pixel_records_r.m[4];
        if(pixel_records_r.m[0] != 0) { 
            pixel_records_r.points0.resize(pixel_records_r.m[0]);
            reader.f.read(reinterpret_cast<char*>(pixel_records_r.points0.data()), pixel_records_r.m[0] * sizeof(Point3f));
        }
        if(pixel_records_r.m[1] != 0) { 
            pixel_records_r.points1.resize(pixel_records_r.m[1]);
            reader.f.read(reinterpret_cast<char*>(pixel_records_r.points1.data()), pixel_records_r.m[1] * sizeof(Point3f));
        }
        if(pixel_records_r.m[2] != 0) { 
            pixel_records_r.points2.resize(pixel_records_r.m[2]);
            reader.f.read(reinterpret_cast<char*>(pixel_records_r.points2.data()), pixel_records_r.m[2] * sizeof(Point3f));
        }
        if(pixel_records_r.m[3] != 0) { 
            pixel_records_r.points3.resize(pixel_records_r.m[3]);
            reader.f.read(reinterpret_cast<char*>(pixel_records_r.points3.data()), pixel_records_r.m[3] * sizeof(Point3f));
        }
        if(pixel_records_r.m[4] != 0) { 
            pixel_records_r.points4.resize(pixel_records_r.m[4]);
            reader.f.read(reinterpret_cast<char*>(pixel_records_r.points4.data()), pixel_records_r.m[4] * sizeof(Point3f));
        }
        block_records_r.push_back(pixel_records_r);
    }
}

// -----------------------------------------------------------------------------

MTS_VARIANT MonteCarloIntegrator<Float, Spectrum>::MonteCarloIntegrator(const Properties &props)
    : Base(props) {
    /// Depth to begin using russian roulette
    m_rr_depth = props.int_("rr_depth", 5);
    if (m_rr_depth <= 0)
        Throw("\"rr_depth\" must be set to a value greater than zero!");

    /*  Longest visualized path depth (``-1 = infinite``). A value of \c 1 will
        visualize only directly visible light sources. \c 2 will lead to
        single-bounce (direct-only) illumination, and so on. */
    m_max_depth = props.int_("max_depth", -1);
    if (m_max_depth < 0 && m_max_depth != -1)
        Throw("\"max_depth\" must be set to -1 (infinite) or a value >= 0");
}

MTS_VARIANT MonteCarloIntegrator<Float, Spectrum>::~MonteCarloIntegrator() { }

MTS_IMPLEMENT_CLASS_VARIANT(Integrator, Object, "integrator")
MTS_IMPLEMENT_CLASS_VARIANT(SamplingIntegrator, Integrator)
MTS_IMPLEMENT_CLASS_VARIANT(MonteCarloIntegrator, SamplingIntegrator)

MTS_INSTANTIATE_CLASS(Integrator)
MTS_INSTANTIATE_CLASS(SamplingIntegrator)
MTS_INSTANTIATE_CLASS(MonteCarloIntegrator)

NAMESPACE_END(mitsuba)
