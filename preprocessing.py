from scipy.io import wavfile
import scipy.io
from scipy.signal import spectrogram
import numpy as np
import os


# Process the Waveform Data
def processData(fileName):
    rate, data = wavfile.read(fileName)
    pad_len = int(8000*24 - data.shape[0])
    data = np.pad(data, (0, pad_len), 'constant', constant_values=(0,0))
    _, _, Sxx = spectrogram(data, fs=8000, window='hanning', detrend='linear', nperseg=4000, noverlap=2000, return_onesided=True, scaling='spectrum', mode='magnitude')
    S = np.amax(Sxx, axis=1)
    return S, Sxx.T

infested_field_directories = ['field/field/train/infested/', 'field/field/test/']   #Need to loop through folders
infested_lab_directories = ['lab/lab/infested/']

clean_field_directories = ['field/field/train/clean/'] #Need to loop through folders
clean_lab_directories = ['lab/lab/clean/']

#Process Field infested directories
infested_fft, infested_sxx = [], []
for directory in infested_field_directories:
    subdirectories = [f.path for f in os.scandir(directory) if f.is_dir()]
    for subdir in subdirectories:
        subfiles = [f.name for f in os.scandir(subdir) if f.is_file()]
        for fileName in subfiles:
            print(subdir+fileName)
            #Perform fft, concatenate, and save
            S, Sxx = processData(subdir+'/'+fileName)
            infested_fft.append(S)
            infested_sxx.append(Sxx)
            #inf

#Process Lab infested directories
for directory in infested_lab_directories:
    files = [f.name for f in os.scandir(directory) if f.is_file()]
    for fileName in files:
        print(directory + fileName)
        S, Sxx = processData(directory + '/'+fileName)
        infested_fft.append(S)
        infested_sxx.append(Sxx)

infested_sxx = np.array(infested_sxx)
infested_fft = np.array(infested_fft)
print(infested_fft.shape, infested_sxx.shape)
np.save('infested_fft.npy', infested_fft)
np.save('infested_sxx.npy', infested_sxx)

#Process the clean field data
clean_fft, clean_sxx = [], []
for directory in clean_field_directories:
    subdirectories = [f.path for f in os.scandir(directory) if f.is_dir()]
    for subdir in subdirectories:
        subfiles = [f.name for f in os.scandir(subdir) if f.is_file()]
        for fileName in subfiles:
            print(subdir+fileName)
            S, Sxx = processData(subdir+'/'+fileName)
            clean_fft.append(S)
            clean_sxx.append(Sxx)

#Process Lab clean directories
for directory in clean_lab_directories:
    files = [f.name for f in os.scandir(directory) if f.is_file()]
    for fileName in files:
        print(directory + fileName)
        S, Sxx = processData(directory + '/'+fileName)
        clean_fft.append(S)
        clean_sxx.append(Sxx)
clean_sxx = np.array(clean_sxx)
clean_fft = np.array(clean_fft)
print(clean_fft.shape, clean_sxx.shape)
np.save('clean_fft.npy', clean_fft)
np.save('clean_sxx.npy', clean_sxx)

'''
directory = 'field/field/train/infested/folder_1/F_20200218111504_52_T14.8.wav'
#directory = 'lab/lab/infested/infested_1.wav'
rate, data = wavfile.read(directory)
print(rate, data.shape)
pad_len = int(8000*24 - data.shape[0])
data = np.pad(data, (0, pad_len), 'constant', constant_values=(0,0))
length = data.shape[0]/rate
print('length', length)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
time = np.linspace(0., length, data.shape[0])
#plt.plot(time, data, label="Left channel")
#plt.legend()
#plt.xlabel("Time [s]")
#plt.ylabel("Amplitude")
#plt.show()
#plt.close()

f, t, Sxx = spectrogram(data, fs=8000, window='hanning', detrend='linear', nperseg=4000, noverlap=2000, return_onesided=True, scaling='density', mode='magnitude')
print(len(t), len(f))
scaler = StandardScaler().fit(Sxx)
Sxx_ = scaler.transform(Sxx)
plt.pcolormesh(t, f, Sxx_, shading='gouraud', cmap='cividis')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale('log')
plt.ylim([0.25, 4000])
plt.show()

print(Sxx.shape)
avg_S = np.amax(Sxx, axis=1)
print(avg_S.shape)

plt.plot(f, avg_S)
plt.grid(which='both')
plt.xscale('log')
plt.show()
'''