import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


def custom_loss(y_true, y_pred):
    dt = 1 / 400
    scale = 50
    N_syn = 4
    N_inputs = 1
    N_states = 3 * N_syn + N_inputs
    f_max = 2.5

    ConnectivityConst = 270
    C1 = ConnectivityConst
    C2 = 0.8 * ConnectivityConst
    C3 = 0.25 * ConnectivityConst
    C4 = 0.25 * ConnectivityConst
    alpha_i = -22
    alpha_e = 3.25

    # mean and std for standardisation
    mean = tf.constant(np.array([-9430.31644298873, 0.061499979854792206, 668.8256234221615, 0.04783514578400952,
                                 2677.787166964063, 0.21137596872545505, 5151.0016393321275, -0.021202987884969188,
                                 4019.7113977999547, -594.7295733324208, 111.29960975111818, 444.8899198620829,
                                 355.9919073051981, -5.191765126838112]), dtype=tf.float32)
    std = tf.constant(np.array([4641.242193515244, 1158.7974784061403, 537.6514768407214, 229.25531862521478,
                                2138.339348499325, 914.4751895151975, 2833.2930138413954, 314.32492018793795,
                                3657.632353833552, 298.9879398394778, 62.72411580041822, 249.9163437388161,
                                199.90397250377836, 43.43043731786296]), dtype=tf.float32)
    y_true = tf.math.multiply(y_true, std) + mean
    y_pred = tf.math.multiply(y_pred, std) + mean
    y_true = tf.transpose(y_true, perm=[0, 2, 1])
    y_pred = tf.transpose(y_pred, perm=[0, 2, 1])
    y1 = y_true[:, -1:, :]
    y_hat = y_pred[:, -1:, :]
    x = y_true[:, :-1, :]
    x_hat = y_pred[:, :-1, :]

    alpha = y_pred[:, 9:13, :]
    alpha2tau = tf.constant(np.array([[alpha_i * 2 * f_max * C4 * dt], [alpha_e * 2 * f_max * C3 * dt],
                                      [alpha_e * 2 * f_max * C1 * dt], [alpha_e * 2 * f_max * C2 * dt]]),
                            dtype=tf.float32)
    tau = tf.divide(alpha2tau, alpha)

    # Define A

    Psi = np.zeros([2 * N_syn, 2 * N_syn, tau.shape[-1] - 1])
    for n in range(N_syn):
        index = 2 * n
        Psi[index: index + 2, index: index + 2, :] = np.transpose(
            np.stack([np.tile(np.array([[0., scale]], dtype=np.float32), [tau.shape[-1] - 1, 1]),
                      np.transpose(
                          np.concatenate([-1 / (scale * tau.numpy()[:, n, :-1] ** 2), -2 / tau.numpy()[:, n, :-1]],
                                         axis=0),
                          [1, 0])], axis=1), [1, 2, 0])
    Psi = tf.constant(Psi, dtype=tf.float32)
    a11 = tf.transpose(tf.tile([tf.eye(2 * N_syn, dtype=tf.float32)], [tau.shape[-1] - 1, 1, 1]),
                       [1, 2, 0]) + dt * Psi  # [1]+dt*Psi
    a12 = tf.zeros([2 * N_syn, N_syn + N_inputs, tau.shape[-1] - 1], dtype=tf.float32)  # [0]
    a21 = tf.zeros([N_syn + N_inputs, 2 * N_syn, tau.shape[-1] - 1], dtype=tf.float32)
    a22 = tf.transpose(tf.tile([tf.eye(N_syn + N_inputs, dtype=tf.float32)], [tau.shape[-1] - 1, 1, 1]),
                       [1, 2, 0])  # [1]
    # Concatenate horizontally
    a1 = tf.concat((a11, a12), axis=1)
    a2 = tf.concat((a21, a22), axis=1)
    # Concatenate vertically
    A = tf.transpose(tf.concat((a1, a2), axis=0), perm=[2, 0, 1])

    # Define B
    B = tf.constant(np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), dtype=tf.float32)

    # Define C
    C = tf.constant(np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0.],
                              [0., 0., 0.02, 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0.],
                              [0.02, 0., 0., 0., 0., 0., 0.02, 0., 0.02, 0., 0.,
                               0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0.],
                              [0.02, 0., 0., 0., 0., 0., 0.02, 0., 0.02, 0., 0.,
                               0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0.],
                              [0., 0., 0., 0., 0.02, 0., 0., 0., 0., 0., 0.,
                               0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0.]]), dtype=tf.float32)

    # Observation matrix H
    H = tf.constant(np.array([[0.02, 0., 0., 0., 0., 0., 0.02, 0., 0.02, 0., 0.,
                               0., 0.]]), dtype=tf.float32)

    v0 = 0.006 * 1e3
    varsigma = 1.699 / (600 / 1e3)

    # x_t contains all the states without the last time step
    x_t = x_hat[:, :, :-1]
    # x_t1 contains all the states without the first time step
    x_t1 = x_hat[:, :, 1:]
    v = tf.matmul(C, x_t)
    phi = 0.5 * tf.math.erf((v - v0) / tf.cast(tf.sqrt(2.) * varsigma, dtype=tf.float32)) + 0.5

    x_t1_hat = tf.transpose(tf.matmul(A, tf.transpose(x_t, perm=[2, 1, 0])), perm=[2, 1, 0]) + tf.math.multiply(
        tf.matmul(B, x_t),
        phi)

    # y measurement error
    mea_error = tf.divide(y1 - tf.matmul(H, x_hat), std[-1])
    # model error
    mod_error = tf.divide(tf.transpose(x_t1 - x_t1_hat, perm=[0, 2, 1]), std[:-1])
    # x measurement error
    x_error = tf.divide(tf.transpose(y_true - y_pred, perm=[0, 2, 1]), std)

    error = tf.concat([tf.math.reduce_mean(K.square(mod_error), axis=1),
                       tf.math.reduce_mean(K.square(mea_error), axis=-1)], 1) \
            + tf.math.reduce_mean(K.square(x_error), axis=1) \
            + tf.concat([tf.constant([[0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
                         tf.math.reduce_std(alpha, axis=-1),
                         tf.constant([[0.]])], axis=-1) \
            * tf.math.reduce_mean(K.square(mod_error)) * 0.1
    return error
