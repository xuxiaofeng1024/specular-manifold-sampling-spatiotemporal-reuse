import os
from subprocess import PIPE, run


#angle = 10
for angle in range(59, 60):
    print(angle)
    #directory ='results_sphere_reuse_angle_{}'.format(angle)
    directory ='results_sphere_reuse'
    print(directory)
    
    try:
        os.mkdir(directory)
    except:
        pass

    def run_cmd(command, name):1
        print("Render {} ..".format(name))
        result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
        log_str = result.stdout
        with open('{}/{}_log.txt'.format(directory, name), 'w') as file:
            file.write(log_str)


    timeout = 54

    crop_s = 1080
    crop_x = 420
    crop_y = 0

    M = 8
    spatial_reuse_size = 4
 
    name = "sphere_reuse_sms_spatial_unique_{:02d}_angle{}".format(spatial_reuse_size, angle)
    cmd = "mitsuba "
    cmd += "sphere_sms.xml "
    cmd += "-Dlight_angle={} ".format(angle)
    cmd += "-o {}/{}.exr ".format(directory, name)
    cmd += "-t24 "
    cmd += "-Dspp=99999999 "
    cmd += "-Dsamples_per_pass=1 "
    cmd += "-Dtimeout={} ".format(timeout)
    cmd += "-Dcrop_offset_x={} ".format(crop_x)
    cmd += "-Dcrop_offset_y={} ".format(crop_y)
    cmd += "-Dcrop_width={} ".format(crop_s)
    cmd += "-Dcrop_height={} ".format(crop_s)
    cmd += "-Dcaustics_twostage=false "
    cmd += "-Dcaustics_biased=true "
    
    cmd += "-Dcaustics_reuse=true "
    cmd += "-Dcaustics_spatial_reuse_size={} ".format(spatial_reuse_size)
    cmd += "-Dcaustics_spatial_reuse_unique=true "
    cmd += "-Dcaustics_pixel_reuse_unique=true "
    
    cmd += "-Dcaustics_current_path_reuse=results_sphere_angle_{} ".format(angle)
    
    cmd += "-Dcaustics_temporal_reuse=false "
    
    cmd += "-Dcaustics_max_trials={} ".format(M)
    run_cmd(cmd, name)

    print("done.")
