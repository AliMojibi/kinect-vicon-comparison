import numpy as np
import matplotlib.pyplot as plt
import dataprocessor

dict_kinect, bones_kinect = dataprocessor.kinect_dict_data("G3-Kinect-CTK-P1T2-Unknown-E2B1-0.txt")
dict_vicon, bones_vicon = dataprocessor.vicon_dict_data("G3-Vicon-CTK-P1T2-Unknown-E2B1-0.txt")


spine_mid_kinect = dict_kinect['SpineMid']
mid_shoulder_kinect = dict_kinect['SpineShoulder']

trunk_kin = np.delete(mid_shoulder_kinect - spine_mid_kinect,list(range(3, 7)), axis=1)
trunk_kin_norm = np.linalg.norm(trunk_kin, axis=1)
f_kin = 30 # Hz
stop_time = dict_kinect['SpineMid'].shape[0]/f_kin
time_kin = np.arange(start=0, stop=stop_time, step = 1/f_kin)
plt.plot(time_kin,trunk_kin_norm, '--', color='red')
len_trunk_kin=np.max(trunk_kin_norm) - np.min(trunk_kin_norm)
print(100*len_trunk_kin/np.mean(trunk_kin_norm))


spine_mid_vicon = dict_vicon['Chest']
mid_shoulder_vicon = (dict_vicon['Right Shoulder'] + dict_vicon['Left Shoulder'])/2

trunk_vic = np.delete(mid_shoulder_vicon - spine_mid_vicon,list(range(3, 7)), axis=1)
trunk_vic_norm = np.linalg.norm(trunk_vic, axis=1)
f_vic = 60 # Hz
stop_time = dict_vicon['Chest'].shape[0]/f_vic
time_vic = np.arange(start=0, stop=stop_time, step = 1/f_vic)
# plt.figure()
plt.plot(time_vic,trunk_vic_norm, '--', color='blue')
len_trunk_vic=np.max(trunk_vic_norm) - np.min(trunk_vic_norm)
print(100*len_trunk_vic/np.mean(trunk_vic_norm))


from scipy import signal
order, cutt_off, sampling = 2, 3, 30
sos = signal.butter(order, cutt_off,fs=sampling, btype='lowpass', output='sos')
filtered = signal.sosfilt(sos, trunk_kin_norm)
plt.figure()
plt.plot(time_kin,trunk_kin_norm, '-', color='red', label='no filter')
plt.plot(time_kin,filtered, '-', color='blue', label = 'filtered')

plt.legend()

plt.figure()
plt.specgram(trunk_kin_norm, Fs=sampling, cmap='jet_r')
plt.colorbar()

plt.figure()
plt.plot(time_kin,filtered, '-', color='blue', label = 'filtered-kin')
plt.plot(time_vic,trunk_vic_norm, '-', color='red', label = 'vicon')
xx = (np.mean(trunk_vic_norm) - np.mean(filtered))/np.mean(trunk_vic_norm)
print(xx)