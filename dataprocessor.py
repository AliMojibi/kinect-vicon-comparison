import numpy as np
import pandas as pd

# a function wich returns a dict data from txt files for Kinect and Vicon
def return_dict_data(key_names, N, exercise_data):
    dict_data = {}
    for key in key_names:
        dict_data[key] = None

    for i in range(len(key_names)):
        key = key_names[i]
        start_index = i*N 
        dict_data[key] = exercise_data[:, start_index:start_index+N]
        
    return dict_data


def vicon_dict_data(directory:str, drop_quat=True):
    # df_vic = pd.read_csv("G3-Vicon-CTK-P1T2-Unknown-E2B1-0.txt", sep = " ", header=None)
    df_vic = pd.read_csv(directory, sep = " ", header=None)/1000 # Converting mm to m


    # vicon data preprocessing
    df_vic = df_vic.drop(7*17, axis = 1)
    np_vic = df_vic.to_numpy() 

    keys_vic = ['Right Forearm'
    ,'Left Forearm'
    ,'Right Arm'
    ,'Left Arm'
    ,'Chest'
    ,'Right Thigh'
    ,'Left Thigh'
    ,'Right Shoulder'
    ,'Left Shoulder'
    ,'Right Hand'
    ,'Left Hand'
    ,'Right Foot'
    ,'Left Foot'
    ,'Hips'
    ,'Head'
    ,'Right Tibia'
    ,'Left Tibia']

    N = 7

    dict_vic = return_dict_data(keys_vic, N, np_vic)

    # Vicon skeleton plot
    bone_list_vic = [(8,7),(7,2),(8,3),(0,2),(1,3),(0,9),(1,10),
                (14,4),(4,13),(13,5),(13,6),(5,15),(6,16),(11,15),(12,16)]

    
    return dict_vic, bone_list_vic


def kinect_dict_data(directory:str):
    
    # Kinect data pre processing
    df_kin = pd.read_csv(directory, sep = " ", header=None)
    df_kin = df_kin.drop(7*25, axis = 1)
    np_kin = df_kin.to_numpy() 

    keys_kin = ['SpineBase'
    ,'SpineMid'
    ,'Neck'
    ,'Head'
    ,'ShoulderLeft'
    ,'ElbowLeft'
    ,'WristLeft'
    ,'HandLeft'
    ,'ShoulderRight'
    ,'ElbowRight'
    ,'WristRight'
    ,'HandRight'
    ,'HipLeft'
    ,'KneeLeft'
    ,'AnkleLeft'
    ,'FootLeft'
    ,'HipRight'
    ,'KneeRight'
    ,'AnkleRight'
    ,'FootRight'
    ,'SpineShoulder'
    ,'HandTipLeft'
    ,'ThumbLeft'
    ,'HandTipRight'
    ,'ThumbRight']
    
    N = 7
    
    dict_kin = return_dict_data(keys_kin, N, np_kin)

    bone_list_kin = [(0, 1),(1, 20), (20, 2), (2, 3),(20, 4),(4, 5),(5, 6),
                    (6, 7),(7, 21),(6, 22),(20, 8),(8, 9),(9, 10),(10, 11),
                    (11, 23), (10, 24),(0, 12),(12, 13), (13, 14), (14, 15),
                    (0, 16),(16, 17),(17, 18), (18, 19)]
    
    return dict_kin, bone_list_kin


import scipy.signal as signal

def apply_butterworth_lowpass_filter(data, cutoff_freq, sampling_freq, order):
    # Calculate the normalized cutoff frequency
    normalized_cutoff = cutoff_freq / (0.5 * sampling_freq)
    
    # Design the Butterworth filter
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False, output='ba')
    
    # Apply zero-phase or forward-backward filtering
    filtered_data = signal.filtfilt(b, a, data, method='pad', padtype='even')
    
    return filtered_data

def resample_signal(ini_signal, original_sr=None, target_sr=None):
    # Determine the resampling ratio
    resampling_ratio = target_sr / original_sr
    
    # Calculate the new length of the resampled signal
    resampled_length = int(len(ini_signal) * resampling_ratio)
    
    # Resample the signal
    resampled_signal = signal.resample(ini_signal, resampled_length)
    
    return resampled_signal

def make_length_similar(signal1, signal2):
    """
    Align two signals to have the same length by either truncating the longer signal or padding the shorter signal.
    """
    # Truncate the longer signal
    if len(signal1) > len(signal2):
        signal1 = signal1[:len(signal2)]
    else:
        signal2 = signal2[:len(signal1)]
        
    return signal1, signal2


def rigid_transform_3D(A, B):
    # returns R, t such that B = RA + t
    
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t
    