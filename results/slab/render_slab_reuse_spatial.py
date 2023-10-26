import os
from subprocess import PIPE, run

try:
    os.mkdir('results_slab')
except:
    pass

def run_cmd(command, name):
    print("Render {} ..".format(name))
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    log_str = result.stdout
    with open('results_slab/{}_log.txt'.format(name), 'w') as file:
        file.write(log_str)

# 5 min for all methods that we want to compare
timeout = 34

angle = 0

crop_s = 1080
crop_x = 420
crop_y = 0


#spatial_reuse_size = 4
M = 8

#for spatial_reuse_size in[0, 4, 16,64,256]:
for spatial_reuse_size in[4]:
    name = "slab_reuse_sms_angle{}_spatial_unique_{:02d}".format(angle, spatial_reuse_size)
    cmd = "mitsuba "
    cmd += "slab_sms.xml "
    cmd += "-Dlight_angle={} ".format(angle)
    cmd += "-t24 "
    cmd += "-o results_slab/{}.exr ".format(name)
    cmd += "-Dspp=999999999 "
    #cmd += "-Dspp={} ".format(spp)
    cmd += "-Dsamples_per_pass=1 "
    cmd += "-Dtimeout={} ".format(timeout)
    cmd += "-Dcrop_offset_x={} ".format(crop_x)
    cmd += "-Dcrop_offset_y={} ".format(crop_y)
    cmd += "-Dcrop_width={} ".format(crop_s)
    cmd += "-Dcrop_height={} ".format(crop_s)
    cmd += "-Dcaustics_biased=true "
    
    cmd += "-Dcaustics_reuse=true "
    cmd += "-Dcaustics_spatial_reuse_size={} ".format(spatial_reuse_size)
    cmd += "-Dcaustics_spatial_reuse_unique=true "
    cmd += "-Dcaustics_pixel_reuse_unique=true "
    
    cmd += "-Dcaustics_current_path_reuse=results_slab "
    
    cmd += "-Dcaustics_temporal_reuse=false "
    
    cmd += "-Dcaustics_max_trials={} ".format(M)
    run_cmd(cmd, name)

print("done.")
