from util.nmm import set_params_t
from util.kalman_filter import kalman_filter
import tensorflow as tf
from util.custom_loss import custom_loss
import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

timestep = 400
length = 10
arrays = []
alpha_i = -22e-3 * 1e3
alpha_e = 3.25e-3 * 1e3
random.seed(200)
x1s = []
x2s = []
ex_inputs = []
ex_input_list = []
# generate 6 combinations of taus
for i in range(6):
    x1 = random.uniform(0.01, 0.06)
    x2 = random.uniform(0.01, 0.06)
    ex_input = random.uniform(0,1000)
    ex_inputs.append(ex_input)
    x1s.append(x1)
    x2s.append(x2)
    ex_input_list.append(np.repeat(ex_input,timestep*length))
    arrays.append(np.stack([np.repeat(alpha_i*2*2.5*0.25*270/timestep/x2,timestep*length),
                            np.repeat(alpha_e*2*2.5*0.25*270/timestep/x1,timestep*length),
                            np.repeat(alpha_e*2*2.5*270/timestep/x1,timestep*length),
                            np.repeat(alpha_e*2*2.5*0.8*270/timestep/x1,timestep*length)]))
# connecting combinations of taus
for j in range(5):
    ex_input_list.insert(2*j+1,np.linspace(ex_inputs[j],
                                          ex_inputs[j+1],num=timestep*length,endpoint=False))
    arrays.insert(2*j+1, np.stack([np.linspace(alpha_i*2*2.5*0.25*270/timestep/x2s[j],
                                              alpha_i*2*2.5*0.25*270/timestep/x2s[j+1],num=timestep*length,endpoint=False),
                                  np.linspace(alpha_e*2*2.5*0.25*270/timestep/x1s[j],
                                              alpha_e*2*2.5*0.25*270/timestep/x1s[j+1],num=timestep*length,endpoint=False),
                                  np.linspace(alpha_e*2*2.5*270/timestep/x1s[j],
                                              alpha_e*2*2.5*270/timestep/x1s[j+1],num=timestep*length,endpoint=False),
                                  np.linspace(alpha_e*2*2.5*0.8*270/timestep/x1s[j],
                                              alpha_e*2*2.5*0.8*270/timestep/x1s[j+1],num=timestep*length,endpoint=False)]))
externals = np.concatenate(ex_input_list)
strengths = np.concatenate(arrays,axis=1)
# put random parameters into nmm to generate state vector and observation
states, obs = set_params_t(externals,np.empty(0),0.4e3,strengths)
df = np.concatenate([states, obs])
X_list1 = np.array(df).transpose()[np.newaxis,:]
X_list1[:,:,-1] = (X_list1[:,:,-1] + 5.191765126838112)/43.43043731786296

# model
batch_size = 1
T_after_cut = 400
targets = 14
model = keras.Sequential()
if len(physical_devices) == 0:
    model.add(Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True,
                                                 stateful=False),
                            batch_input_shape=(batch_size, T_after_cut, 1)))
    model.add(Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True,
                                                 stateful=False),
                            batch_input_shape=(batch_size, T_after_cut, 128)))
else:
    model.add(Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(128,
                               return_sequences=True,
                               stateful=False), batch_input_shape=(batch_size,T_after_cut ,1)))
    model.add(Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(32,
                               return_sequences=True,
                               stateful=False), batch_input_shape=(batch_size,T_after_cut ,128)))

model.add(layers.TimeDistributed(layers.Dense(targets, activation='linear')))

optimizer = keras.optimizers.RMSprop(lr=0.0001)
model.compile(loss=custom_loss, optimizer=optimizer, run_eagerly=True)
if len(physical_devices) == 0:
    model.load_weights('saved_weights/bi_in_0_1_noise_1_feature_cpu')
else:
    model.load_weights('saved_weights/bi_in_0_1_noise_1_feature_128_32_stateless')

model.reset_states()
y_pred = model.predict(X_list1[0:1,:,-1])
# mean std from training dataset
mean = np.array([-9430.31644298873, 0.061499979854792206, 668.8256234221615, 0.04783514578400952,
                                 2677.787166964063, 0.21137596872545505, 5151.0016393321275, -0.021202987884969188,
                                 4019.7113977999547, -594.7295733324208, 111.29960975111818, 444.8899198620829,
                                 355.9919073051981, -5.191765126838112])
std = np.array([4641.242193515244, 1158.7974784061403, 537.6514768407214, 229.25531862521478,
                                2138.339348499325, 914.4751895151975, 2833.2930138413954, 314.32492018793795,
                                3657.632353833552, 298.9879398394778, 62.72411580041822, 249.9163437388161,
                                199.90397250377836, 43.43043731786296])
y_resc = np.multiply(y_pred[0],std)+mean
y_truth = np.concatenate([states,obs])
o_str = np.divide(y_truth-mean[:,np.newaxis],std[:,np.newaxis])

# set the initialisation the same as the beginning of x1s/x2s/ex_inputs
y_pred_k = kalman_filter(X_list1[0:1,:,-1]*43.43043731786296-5.191765126838112, x1s[0],x2s[0], ex_inputs[0])
y_pred_k = np.nan_to_num(y_pred_k)
seconds = np.linspace(0,109,43600)

# remove the first
blue = y_truth[12,:-400]
orange = y_pred_k[12,:-400]
green = y_resc[:-400,12]

fig, axs = plt.subplots(3, 3)
fig.set_size_inches(10.5, 10.5)

axs[0, 0].plot(seconds,y_truth[9,:-400])
axs[0, 0].plot(seconds,y_pred_k[9,:-400])
axs[0, 0].plot(seconds,y_resc[:-400,9])
axs[0, 0].set_title('Alpha I-P',fontsize=10)
axs[0, 0].set_xlabel('Time(second)')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 1].plot(seconds,y_truth[10,:-400])
axs[0, 1].plot(seconds,y_pred_k[10,:-400])
axs[0, 1].plot(seconds,y_resc[:-400,10])
axs[0, 1].set_title('Alpha P-I',fontsize=10)
axs[0, 1].set_xlabel('Time(second)')
axs[0, 1].set_ylabel('Amplitude')
axs[0, 2].plot(seconds,y_truth[11,:-400])
axs[0, 2].plot(seconds,y_pred_k[11,:-400])
axs[0, 2].plot(seconds,y_resc[:-400,11])
axs[0, 2].set_title('Alpha P-E',fontsize=10)
axs[0, 2].set_xlabel('Time(second)')
axs[0, 2].set_ylabel('Amplitude')
axs[1, 0].plot(seconds,blue)
axs[1, 0].plot(seconds,orange)
axs[1, 0].plot(seconds,green)
axs[1, 0].set_title('Alpha E-P',fontsize=10)
axs[1, 0].set_xlabel('Time(second)')
axs[1, 0].set_ylabel('Amplitude')
axs[1, 1].plot(seconds,y_truth[8,:-400]/50)
axs[1, 1].plot(seconds,y_pred_k[8,:-400]/50)
axs[1, 1].plot(seconds,y_resc[:-400,8]/50)
axs[1, 1].set_title('Ext Input',fontsize=10)
axs[1, 1].set_xlabel('Time(second)')
axs[1, 1].set_ylabel('Firing Rate')
axs[1, 2].axis('off')
axs[2, 0].plot(seconds,y_truth[13,:-400])
axs[2, 0].set_title('Observation',fontsize=10)
axs[2, 0].set_xlabel('Time(second)')
axs[2, 0].set_ylabel('Membrane Potential(mV)')
axs[2, 2].plot(seconds,y_resc[:-400,0]/50+y_resc[:-400,6]/50+y_resc[:-400,8]/50, c='#2ca02c')
axs[2, 2].set_title('Observation',fontsize=10)
axs[2, 2].set_xlabel('Time(second)')
axs[2, 2].set_ylabel('Membrane Potential(mV)')
axs[2, 1].plot(seconds,y_pred_k[0,:-400]/50+y_pred_k[6,:-400]/50+y_pred_k[8,:-400]/50, c='#ff7f0e')
axs[2, 1].set_title('Observation',fontsize=10)
axs[2, 1].set_xlabel('Time(second)')
axs[2, 1].set_ylabel('Membrane Potential(mV)')

fig.legend([blue, orange, green],     # The line objects
           labels=['Truth','Kalman Filter','LSTM'],   # The labels for each line
           loc=[0.8,0.5],   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           )

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.4)
plt.show()
