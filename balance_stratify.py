import numpy as np
import time 

#Randomly sample from data (we dont need 42GB of infested data...)
indx_range = np.random.choice(2700, 2700, replace=False)
print(indx_range)

#Grab only the randomly selected data
clean_fft = np.load('processed_data/clean_fft.npy')
clean_fft_mini = clean_fft[indx_range]
clean_sxx = np.load('processed_data/clean_sxx.npy')
clean_sxx_mini = clean_sxx[indx_range]
infested_fft = np.load('processed_data/infested_fft.npy')
infested_fft_mini = infested_fft[indx_range]
infested_sxx = np.load('processed_data/infested_sxx.npy')
infested_sxx_mini = infested_sxx[indx_range]

#Split into train and test indices
train_range = np.random.choice(2700, 2160, replace=False)

clean_fft_train = clean_fft_mini[train_range]
infested_fft_train = infested_fft_mini[train_range]
clean_sxx_train = clean_sxx_mini[train_range]
infested_sxx_train = infested_sxx_mini[train_range]

clean_fft_test = np.delete(clean_fft_mini, train_range, axis=0)
infested_fft_test = np.delete(infested_fft_mini, train_range, axis=0)
clean_sxx_test = np.delete(clean_sxx_mini, train_range, axis=0)
infested_sxx_test = np.delete(infested_sxx_mini, train_range, axis=0)

#Create the labels
clean_labels_train = np.zeros((2160, 1))
clean_labels_test = np.zeros((540, 1))
inf_labels_train = np.ones((2160, 1))
inf_labels_test = np.ones((540, 1))

#Combine clean and infested
fft_train = np.concatenate((clean_fft_train, infested_fft_train), axis=0)
sxx_train = np.concatenate((clean_sxx_train, infested_sxx_train), axis=0)
fft_test = np.concatenate((clean_fft_test, infested_fft_test), axis=0)
sxx_test = np.concatenate((clean_sxx_test, infested_sxx_test), axis=0)

labels_train = np.concatenate((clean_labels_train, inf_labels_train))
labels_test = np.concatenate((clean_labels_test, inf_labels_test))

#Finally we need to shuffle the data
shuff_range = np.random.choice(4320, 4320)
fft_train = fft_train[shuff_range]
sxx_train = sxx_train[shuff_range]
labels_train = labels_train[shuff_range]

shuff_range = np.random.choice(1080, 1080)
fft_test = fft_test[shuff_range]
sxx_test = sxx_test[shuff_range]
labels_test = labels_test[shuff_range]

#Save the train/test splits
np.save('training_data/fft_train.npy', fft_train)
np.save('training_data/fft_test.npy', fft_test)
np.save('training_data/sxx_train.npy', sxx_train)
np.save('training_data/sxx_test.npy', sxx_test)
np.save('training_data/train_labels.npy', labels_train)
np.save('training_data/test_labels.npy', labels_test)
