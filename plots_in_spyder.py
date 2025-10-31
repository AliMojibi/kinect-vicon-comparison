import numpy as np
import matplotlib.pyplot as plt
import dataprocessor

dict_kinect, bones_kinect = dataprocessor.kinect_dict_data("G3-Kinect-ELK-P1T1-Unknown-C-0.txt")
dict_vicon, bones_vicon = dataprocessor.vicon_dict_data("G3-Vicon-ELK-P1T1-Unknown-C-0.txt")

kinect_rh = dict_kinect['HandRight']
vicon_rh = dict_vicon['Right Hand']
kinect_lh = dict_kinect['HandLeft']
vicon_lh = dict_vicon['Left Hand']

kinect_rs = dict_kinect['ShoulderRight']
vicon_rs = dict_vicon['Right Shoulder']
kinect_ls = dict_kinect['ShoulderLeft']
vicon_ls = dict_vicon['Left Shoulder']
kinect_c = dict_kinect['SpineBase']
vicon_c = dict_vicon['Hips']
kinect_h = dict_kinect['Head']
vicon_h = dict_vicon['Head']
kinect_rf = dict_kinect['FootRight']
vicon_rf = dict_vicon['Right Foot']
kinect_lf = dict_kinect['FootLeft']
vicon_lf = dict_vicon['Left Foot']

kinect_ss = dict_kinect['SpineShoulder']
vicon_ss = dict_vicon['Chest']
torso_back = (kinect_ss - kinect_c).copy()



kinect_data = [kinect_rs, kinect_ls, kinect_c, kinect_h, kinect_rf, kinect_lf, kinect_rh, kinect_lh, kinect_ss]
vicon_data = [vicon_rs, vicon_ls, vicon_c, vicon_h, vicon_rf, vicon_lf, vicon_rh, vicon_lh, vicon_ss]

cut_off, sampling, order = 2, 30, 4


for kinect in kinect_data:
    for column in range(kinect.shape[1]):
        kinect[:, column] = dataprocessor.apply_butterworth_lowpass_filter(kinect[:, column], cut_off, sampling, order)

for column in range(torso_back.shape[1]):
    torso_back[:, column] = dataprocessor.apply_butterworth_lowpass_filter(torso_back[:, column], cut_off, sampling, order)


for vicon in vicon_data:
    for column in range(vicon.shape[1]):
        vicon[:, column] = dataprocessor.apply_butterworth_lowpass_filter(vicon[:, column], cut_off, sampling, order)


for index, kinect in enumerate(kinect_data):
    kinect_resampled = dataprocessor.resample_signal(kinect, original_sr=30, target_sr=60)
    kinect_data[index], vicon_data[index] = dataprocessor.make_length_similar(kinect_resampled, vicon_data[index])

for column in range(torso_back.shape[1]):
    torso_back[:, column] = dataprocessor.apply_butterworth_lowpass_filter(torso_back[:, column], cut_off, sampling, order)

# for index, kinect in enumerate(kinect_data):
torso_back = dataprocessor.resample_signal(torso_back, original_sr=30, target_sr=60)
# torso_back[index], vicon_data[index] = dataprocessor.make_length_similar(kinect_resampled, vicon_data[index])


plt.subplot(121)
plt.plot(kinect_rs[:, 0:3])
plt.title('kinect')

plt.subplot(122)
plt.plot(vicon_rs[:, 0:3])
plt.title('vicon')

plt.show()

kinect_trans = kinect_data[:-1].copy()
vicon_trans = vicon_data[:-1].copy()

kinect_stacked = np.concatenate(kinect_trans, axis=0)
vicon_stacked = np.concatenate(vicon_trans, axis=0)
kinect_stacked.shape, vicon_stacked.shape 
R, t = dataprocessor.rigid_transform_3D(kinect_stacked[:, 0:3].T, vicon_stacked[:, 0:3].T)


for index, item in enumerate(kinect_data):
    in_ref_frame = R@item[:, :3].T+t
    kinect_data[index] = in_ref_frame.copy()
    
torso_back = R@torso_back[:, :3].T

print(torso_back.shape)
plt.plot(np.linalg.norm(torso_back, axis=0))
np.linalg.norm(torso_back).shape
torso_back[:,0].shape = (3,1)
vicon_data[2] = (vicon_data[0][:, :3] + vicon_data[1][:, :3])/2+torso_back[:,2]


for c, item in enumerate(vicon_data):
    vicon_data[c] = item[:, :3].T.copy()
    

lbls_kin = ['x_kinect', 'y_kinect', 'z_kinect']
lbls_vic = ['x_vicon', 'y_vicon', 'z_vicon']
plt.figure(figsize=(15,10))
plt.subplot(333)
i = 1
offsets = np.zeros((3,3))
for index, item in enumerate(kinect_data[:3]):
    j = 0
    for row in range(3):
        plt.subplot(3,3,i)
        plt.plot(item[row, :], label = lbls_kin[row])
        plt.plot(vicon_data[index][row, :], label = lbls_vic[row])
        plt.legend()
        offsets[index, j] = item[row, 0] - vicon_data[index][row, 0]
        i += 1    
        j+=1
plt.show()

lbls_kin = ['x_kinect', 'y_kinect', 'z_kinect']
lbls_vic = ['x_vicon', 'y_vicon', 'z_vicon']
plt.figure(figsize=(15,10))
plt.subplot(333)
i = 1
for index, item in enumerate(kinect_data[:3]):
    j = 0
    for row in range(3):
        plt.subplot(3,3,i)
        item[row, :] -=  offsets[index, j]
        plt.plot(item[row, :], label = lbls_kin[row])
        plt.plot(vicon_data[index][row, :], label = lbls_vic[row])
        plt.legend()
        offsets[index, j] = item[row, 0] - vicon_data[index][row, 0]
        i += 1    
        j+=1
plt.show()


u = kinect_data[0] - kinect_data[2] # rs - c
i_hat = u/np.linalg.norm(u, axis=0)
v = kinect_data[1] - kinect_data[2] # ls - c
cps = np.cross(i_hat, v, axisa=0, axisb=0, axisc=0)
j_hat = cps/np.linalg.norm(cps, axis=0)

k_hat = np.cross(i_hat, j_hat, axisa=0, axisb=0, axisc=0)
# u.shape, v.shape
 
s_hat = kinect_data[1] - kinect_data[0]
s_hat /= np.linalg.norm(s_hat, axis=0)

f_front_hat = np.cross(s_hat, j_hat, axisa=0, axisb=0, axisc=0)

f_front_hat = f_front_hat[:, 0].copy()


#########

for key in dict_kinect:
    for column in range(dict_kinect[key].shape[1]):
        dict_kinect[key][:, column] = dataprocessor.apply_butterworth_lowpass_filter(dict_kinect[key][:, column], cut_off, sampling, order)

for key in dict_vicon:
    for column in range(dict_vicon[key].shape[1]):
        dict_vicon[key][:, column] = dataprocessor.apply_butterworth_lowpass_filter(dict_vicon[key][:, column], cut_off, sampling, order)

for index, key in enumerate(dict_kinect):
    kinect_resampled = dataprocessor.resample_signal(dict_kinect[key], original_sr=30, target_sr=60)
    dict_kinect[key], _ = dataprocessor.make_length_similar(kinect_resampled, dict_vicon['Chest'])

keys_vic = list(dict_vicon.keys())
keys_kin = list(dict_kinect.keys())
time = 100
fig = plt.figure(figsize = (10, 7))

ax = plt.axes(projection ="3d")

# ax.scatter(vicon_data[2][0,0], 
#            vicon_data[2][0,1],
#            vicon_data[2][0,2],
#            c='red', marker='*', s=1000)

# ax.set_xlim(0, 1)
# ax.set_ylim(-0.4, 0.8)
# ax.set_zlim(0.2, 1.4)
for time in range(time,time+1):
    for bone in bones_vicon:
        key_start = keys_vic[bone[0]]
        key_end = keys_vic[bone[1]]
        coords_start = dict_vicon[key_start][time, 0:3]
        coords_end = dict_vicon[key_end][time, 0:3]
        coords = np.vstack((coords_start, coords_end))
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], '.-r',color='blue')
        ax.axis('square')
        # ax.set_xlim(0.5, 1)
        # ax.set_ylim(-0.4, 0.8)
        # ax.set_zlim(0.2, 1.4)

# fig = plt.figure(figsize = (10, 7))

# ax = plt.axes(projection ="3d")        
# for time in range(time,time+1):    
    for bone in bones_kinect:
        key_start = keys_kin[bone[0]]
        key_end = keys_kin[bone[1]]
        coords_start = dict_kinect[key_start][time, 0:3]
        coords_end = dict_kinect[key_end][time, 0:3]
        coords = np.vstack((coords_start, coords_end))
        coords = (R@coords.T + t).T # want to each point be a row
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], '.-m')
        ax.axis('square')
        # ax.set_xlim(0.5, 1)
        # ax.set_ylim(-0.4, 0.8)
        # ax.set_zlim(0.2, 1.4)
        

        ax.view_init(0, 0)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')


plt.show()
