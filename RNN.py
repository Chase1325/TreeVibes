import os
import numpy as np
import tensorflow as tf
import kerastuner as kt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

class Tuner(kt.tuners.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=[32, 64, 128])
        super(Tuner, self).run_trial(trial, *args, **kwargs)

def tune_network(type='NN'):

    scaler = preprocessing.StandardScaler()

    fft = np.load('training_data/sxx_train.npy')
    labels = np.load('training_data/train_labels.npy')

    #Normalize
    fft_max = np.amax(fft)
    fft = fft / fft_max

    split = int(0.8*fft.shape[0])
    xTrain, yTrain = fft[:split, :, :], labels[:split]
    xTest, yTest = fft[split:, :, :], labels[split:]

    print(xTrain.shape, yTrain.shape, xTest.shape, yTest.shape)

    def model_struct(hp):

        #hp_embed = hp.Choice('embed', values=[32, 64, 128, 256, 512])
        hp_units = hp.Choice('units', values=[64, 128, 256, 512])
        hp_drop = hp.Choice('drop', values=[0.0, 0.125, 0.25, 0.5])

        #BUILD THE MODEL
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dropout(hp_drop))
        #model.add(tf.keras.layers.Embedding(fft.shape[-1], hp_embed, input_length=95))
        model.add(tf.keras.layers.LSTM(hp_units))
        model.add(tf.keras.layers.Dropout(hp_drop))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
        #COMPILE THE MODEL
        hp_lr = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-2])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr), 
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    tuner = Tuner(model_struct, 
                    objective='val_accuracy',
                    num_initial_points=10, 
                    max_trials=50, seed=0,
                    project_name='sxx/tuning')

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    tuner.search(xTrain, yTrain, epochs=1000, validation_data=(xTest, yTest), callbacks=[early_stop], use_multiprocessing=True, workers=4, verbose=1)

    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model=tuner.hypermodel.build(best_hps)
    
    #Get the best epoch
    history=best_model.fit(xTrain, yTrain, epochs=1000, callbacks=[early_stop], validation_data=(xTest, yTest), use_multiprocessing=True, workers=4, verbose=1)
    val_acc_per_epoch = history.history['val_loss']
    best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1

    #Re-Train the best epoch model
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(xTrain, yTrain, epochs=best_epoch, use_multiprocessing=True, workers=4, verbose=1)

    #Save the best model
    hypermodel.save('sxx/model')
    print(tuner.results_summary())

def test_network():
    scaler = preprocessing.StandardScaler()

    fft = np.load('training_data/sxx_train.npy')
    fft_test = np.load('training_data/sxx_test.npy')
    labels = np.load('training_data/train_labels.npy')
    labels_test = np.load('training_data/test_labels.npy')

    #Scale based on the training data as done during training
    #Normalize
    fft_max = np.amax(fft)
    fft = fft / fft_max
    fft_test = fft_test/fft_max

    #Load the trained model
    model = tf.keras.models.load_model('sxx/model')
    print(model.summary())

    #Make train and test prediction
    train_pred = np.where(model.predict(fft)>=0.5, 1, 0)
    test_pred = np.where(model.predict(fft_test)>=0.5, 1, 0)
    print(test_pred[:5])

    p, r, f, _ = precision_recall_fscore_support(labels_test, test_pred)
    print(accuracy_score(labels, train_pred), accuracy_score(labels_test, test_pred), np.mean(p), np.mean(r), np.mean(f))
    cm = confusion_matrix(labels_test, test_pred, normalize='true')
    df_cm = pd.DataFrame(cm)
    sb.heatmap(df_cm, annot=True)
    plt.show()

if __name__=='__main__':
    #tune_network()
    test_network()