from intern.remote.boss import BossRemote
from intern.resource.boss.resource import ChannelResource
'''
col = 'collman'
exp = 'M247514_Rorb_1_Site3Align2_EM'
channel = 'm247514_Site3Annotation_MN_global'
dtype = 'uint8'
host,token = get_host_token()
x_range = [0,14020]
y_range = [0,14723]
z_range = [0,49]

remote = BossRemote('./neurodata.cfg')
voxel_size = '3 3 50'
voxel_unit = 'nanometers'

channel_resource = ChannelResource(channel, col, exp, 'annotation', '', 0, dtype, 0)
project = remote.get_project(channel_resource)
for z in range(z_range[0],z_range[1]):
    print(z)
    remote.create_cutout(channel_resource, 0, (x_range[0],x_range[1]), (y_range[0],y_range[1]), (z,z+1), data[z].reshape(-1,data[z].shape[0],data[z].shape[1]))
print('done!')
'''

'''
config_dict:
{
    "token": token,
    "host": "api.boss.neurodata.io",
    "protocol": "https"
}
'''
def create_boss_remote(config_dict):
    remote = BossRemote(config_dict)
    return remote
    
def boss_push(token, 
              col, 
              exp, 
              z_range,
              y_range,
              x_range,
              data_dict):
    print(z_range)
    dtype = "uint8"
    config_dict = {"token": token, "host": "api.boss.neurodata.io" , "protocol": "https"}
    remote = create_boss_remote(config_dict)
    
    for key, data in data_dict.items():
        channel = key
        print(data.shape)
        z, y, x = data.shape
    
        channel_resource = ChannelResource(channel, col, exp, 'annotation', '', 0, dtype, 0)
        print("Pushing to BOSS...")
        
        for z in range(z_range[0],z_range[1]):
            print(z)
            try:    
                old_channel = remote.get_project(channel_resource)
                remote.create_cutout(old_channel, 0, (x_range[0],x_range[1]), (y_range[0],y_range[1]), (z,z+1), data[z-z_range[0]].reshape(-1,data[z-z_range[0]].shape[0],data[z-z_range[0]].shape[1]))
            except:
                channel_resource = ChannelResource(channel, col, exp, 'annotation', '', 0, dtype, 0, sources = ["em_clahe"])
                new_channel = remote.create_project(channel_resource)
                remote.create_cutout(new_channel, 0, (x_range[0],x_range[1]), (y_range[0],y_range[1]), (z,z+1), data[z-z_range[0]].reshape(-1,data[z-z_range[0]].shape[0],data[z-z_range[0]].shape[1]))


        print("Pushed {} to Boss".format(channel))
    
    