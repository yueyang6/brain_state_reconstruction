import tensorflow as tf
from sklearn.metrics import r2_score
import glob
import pandas as pd
import random
import scipy.io as sio
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional
from util.custom_loss import custom_loss

# configure GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def get_sz(file):
    mat_f = sio.loadmat(file)
    dt = pd.DataFrame(mat_f['data'])
    return dt


def batched(i, arr, batch_size):
    return (arr[i * batch_size:(i + 1) * batch_size])


def test_on_batch_stateful(model, inputs, outputs, batch_size, nb_cuts):
    nb_batches = int(len(inputs) / batch_size)
    sum_pred = 0
    for i in range(nb_batches):
        if i % nb_cuts == 0:
            model.reset_states()
        x = batched(i, inputs, batch_size)
        y = batched(i, outputs, batch_size)
        sum_pred += model.test_on_batch(x, y)
    mean_pred = sum_pred / nb_batches
    return (mean_pred)


def define_stateful_val_loss_class(inputs, outputs, batch_size, nb_cuts):
    class ValidationCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.val_loss = []

        def on_epoch_end(self, epoch, logs={}):
            mean_pred = test_on_batch_stateful(self.model, inputs, outputs,
                                               batch_size, nb_cuts)
            # print('val_loss: {:0.3e}'.format(mean_pred), end='')
            self.val_loss += [mean_pred]

        def get_val_loss(self):
            return (self.val_loss)

    return (ValidationCallback)


class ResetStatesCallback(tf.keras.callbacks.Callback):
    def __init__(self, nb_cuts):
        self.counter = 0
        self.nb_cuts = nb_cuts

    def on_batch_begin(self, batch, logs={}):
        # reset states when nb_cuts batches are completed
        if self.counter % self.nb_cuts == 0:
            self.model.reset_states()
        self.counter += 1

    def on_epoch_end(self, epoch, logs={}):
        # reset states after each epoch
        self.model.reset_states()
        return(ResetStatesCallback)

if __name__ == '__main__':
    time_steps = 1200
    files_path = glob.glob('./simu_data/*.csv')
    X_list1 = []
    y_list1 = []
    X_list = []
    y_list = []
    for file in files_path:
        try:
            df = pd.read_csv(file, index_col=0)
            x = df.iloc[13:14, :].transpose()
            y = df.loc[:14, :].transpose()
        except (IndexError, KeyError, pd.errors.ParserError):
            print(file)
            continue
        else:
            X_list1.append(x)
            y_list1.append(y)

    length = len(x)
    n_split = 1

    random.Random(123).shuffle(X_list1)
    random.Random(123).shuffle(y_list1)
    features = 1
    X_list1 = np.stack(X_list1)

    X_list1 = np.transpose(X_list1,[X_list1.shape[0],X_list1.shape[1],features])
    y_list1 = np.stack(y_list1)
    y_list2 = np.copy(y_list1)
    mean_list = []
    std_list = []
    i = 0
    while i < X_list1.shape[2]:
        mean = X_list1[:,:,i].mean()
        std = X_list1[:,:,i].std()
        X_list1[:,:,i] = (X_list1[:,:,i] - mean) / std
        i += 1
    i = 0
    while i < y_list1.shape[2]:
        mean = y_list1[:,:,i].mean()
        std = y_list1[:,:,i].std()
        y_list2[:,:,i] = (y_list1[:,:,i] - mean) / std
        mean_list.append(mean)
        std_list.append(std)
        i += 1

    # train, test, val
    X_train_list = X_list1[:int(len(X_list1)*.8)]
    X_test_list = X_list1[int(len(X_list1)*.9):]
    y_train_list = y_list2[:int(len(y_list2)*.8)]
    y_test_list = y_list2[int(len(y_list2)*.9):]
    X_val = X_list1[int(len(X_list1)*.8):int(len(X_list1)*.9),:,:]
    y_val = y_list2[int(len(y_list2)*.8):int(len(y_list2)*.9),:,:]

    cut1 = [np.split(x, n_split, axis=0) for x in X_train_list]
    cut2 = [np.stack(x) for x in cut1]
    X_train_list = np.concatenate(cut2)

    cut1 = [np.split(x, n_split, axis=0) for x in y_train_list]
    cut2 = [np.stack(x) for x in cut1]
    y_train_list = np.concatenate(cut2)

    cut1 = [np.split(x, n_split, axis=0) for x in X_val]
    cut2 = [np.stack(x) for x in cut1]
    X_val = np.concatenate(cut2)

    cut1 = [np.split(x, n_split, axis=0) for x in y_val]
    cut2 = [np.stack(x) for x in cut1]
    y_val = np.concatenate(cut2)


    T_after_cut = 400
    batch_size = 1
    targets = 14
    nb_cuts = 1
    ValidationCallback = define_stateful_val_loss_class(X_val,
                                                        y_val,
                                                        batch_size,
                                                        nb_cuts)
    validation = ValidationCallback()

    # LSTM model
    model = keras.Sequential()
    model.add(Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(128,
                               return_sequences=True,
                               stateful=False), batch_input_shape=(batch_size, T_after_cut,features)))
    model.add(Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(32,
                               return_sequences=True,
                               stateful=False), batch_input_shape=(batch_size,T_after_cut ,128)))
    model.add(layers.TimeDistributed(layers.Dense(targets, activation='linear')))
    optimizer = keras.optimizers.RMSprop(lr=0.001)
    model.compile(loss=custom_loss, optimizer=optimizer, run_eagerly=True)

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=10, verbose=0,
                                                     mode='auto', restore_best_weights=True)
    ResetStates_Callback = ResetStatesCallback(nb_cuts)


    n_epochs = 10000
    n_batch = 1
    i = 0
    model.fit(X_train_list, y_train_list,
              epochs=n_epochs,
              batch_size=n_batch,
              verbose=0,
              shuffle=False,
              validation_data=(X_val, y_val),
              callbacks=[ResetStatesCallback(nb_cuts),
                         validation,
                         earlyStopping])

    model.save_weights('saved_weights/bi_128_32_stateless')

    y_val_list = []
    for i in range(X_val.shape[0]):
        y_val_pred = model.predict(X_val[i:i+1,:])
        y_val_list.append(y_val_pred)
    y_val_preds = np.concatenate(y_val_list,axis=0)

    y_pred_list = []
    for i in range(X_test_list.shape[0]):
        y_pred = model.predict(X_test_list[i:i+1,:])
        y_pred_list.append(y_pred)
    y_preds = np.concatenate(y_pred_list,axis=0)

    print('validation')
    print('I-P Potential R squared',r2_score(y_val[:,:,0],y_val_preds[:,:,0]))
    print('I-P Potential Derivative R squared',r2_score(y_val[:,:,1],y_val_preds[:,:,1]))
    print('P-I Potential R squared',r2_score(y_val[:,:,2],y_val_preds[:,:,2]))
    print('P-I Potential Derivative R squared',r2_score(y_val[:,:,3],y_val_preds[:,:,3]))
    print('P-E Potential R squared',r2_score(y_val[:,:,4],y_val_preds[:,:,4]))
    print('P-E Potential Derivative R squared',r2_score(y_val[:,:,5],y_val_preds[:,:,5]))
    print('E-P Potential R squared',r2_score(y_val[:,:,6],y_val_preds[:,:,6]))
    print('E-P Potential Derivative R squared',r2_score(y_val[:,:,7],y_val_preds[:,:,7]))
    print('Input R squared',r2_score(y_val[:,:,8],y_val_preds[:,:,8]))
    print('I-P Connectivity Strength R squared',r2_score(y_val[:,:,9],y_val_preds[:,:,9]))
    print('P-I Connectivity Strength R squared',r2_score(y_val[:,:,10],y_val_preds[:,:,10]))
    print('P-E Connectivity Strength R squared',r2_score(y_val[:,:,11],y_val_preds[:,:,11]))
    print('E-P Connectivity Strength R squared',r2_score(y_val[:,:,12],y_val_preds[:,:,12]))

    print('test')
    print('I-P Potential R squared',r2_score(y_test_list[:,:,0],y_preds[:,:,0]))
    print('I-P Potential Derivative R squared',r2_score(y_test_list[:,:,1],y_preds[:,:,1]))
    print('P-I Potential R squared',r2_score(y_test_list[:,:,2],y_preds[:,:,2]))
    print('P-I Potential Derivative R squared',r2_score(y_test_list[:,:,3],y_preds[:,:,3]))
    print('P-E Potential R squared',r2_score(y_test_list[:,:,4],y_preds[:,:,4]))
    print('P-E Potential Derivative R squared',r2_score(y_test_list[:,:,5],y_preds[:,:,5]))
    print('E-P Potential R squared',r2_score(y_test_list[:,:,6],y_preds[:,:,6]))
    print('E-P Potential Derivative R squared',r2_score(y_test_list[:,:,7],y_preds[:,:,7]))
    print('Input R squared',r2_score(y_test_list[:,:,8],y_preds[:,:,8]))
    print('I-P Connectivity Strength R squared',r2_score(y_test_list[:,:,9],y_preds[:,:,9]))
    print('P-I Connectivity Strength R squared',r2_score(y_test_list[:,:,10],y_preds[:,:,10]))
    print('P-E Connectivity Strength R squared',r2_score(y_test_list[:,:,11],y_preds[:,:,11]))
    print('E-P Connectivity Strength R squared',r2_score(y_test_list[:,:,12],y_preds[:,:,12]))
