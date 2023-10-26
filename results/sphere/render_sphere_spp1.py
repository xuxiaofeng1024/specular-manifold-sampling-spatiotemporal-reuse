import os
from subprocess import PIPE, run

for angle in range(60, 61):
    print(angle)
    directory ='results_sphere_angle_{}'.format(angle)
    print(directory)

    try:
        os.mkdir(directory)
    except:
        pass

    def run_cmd(command, name):
        print("Render {} ..".format(name))
        result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
        log_str = result.stdout
        with open('{}/{}_log.txt'.format(directory, name), 'w') as file:
            file.write(log_str)



    crop_s = 1080
    crop_x = 420
    crop_y = 0

    spatial_reuse_size = 0
    spp=2
    M = 8

    name = "sphere_sms_angle{}".format(angle)
    cmd = "mitsuba "
    cmd += "sphere_sms.xml "
    cmd += "-Dlight_angle={} ".format(angle)
    cmd += "-o {}/{}.exr ".format(directory, name)
    cmd += "-t24 "
    cmd += "-Dspp={} ".format(spp)
    #cmd += "-Dsamples_per_pass=2 "
    cmd += "-Dsamples_per_pass={} ".format(spp)
    #cmd += "-Dtimeout={} ".format(timeout)
    cmd += "-Dcrop_offset_x={} ".format(crop_x)
    cmd += "-Dcrop_offset_y={} ".format(crop_y)
    cmd += "-Dcrop_width={} ".format(crop_s)
    cmd += "-Dcrop_height={} ".format(crop_s)
    cmd += "-Dcaustics_twostage=true "
    cmd += "-Dcaustics_biased=true "
    
    cmd += "-Dcaustics_reuse=false "
    cmd += "-Dcaustics_spatial_reuse_size={} ".format(spatial_reuse_size)
    cmd += "-Dcaustics_spatial_reuse_unique=false "
    cmd += "-Dcaustics_pixel_reuse_unique=true "
    cmd += "-Dcaustics_current_path_reuse={} ".format(directory)
    cmd += "-Dcaustics_temporal_reuse=false "
    
    cmd += "-Dcaustics_max_trials={} ".format(M)
    run_cmd(cmd, name)

    print("done.")
