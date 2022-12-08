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
    mean = tf.constant(np.array([-5591.680806178452, 0.030124868655625796, 640.1735753166167, 0.01640663255599838,
                                        2566.320382108992, 0.08487297573162939, 4696.598675568893, -0.037053241151794654,
                                        1170.1196517289525, -1040.7360772645036, 112.60877309386701, 451.58847608068044,
                                        360.86408797278557, 5.5016211077300055]), dtype=tf.float32)
    std = tf.constant(np.array([1903.0240843546615, 1005.8731699486577, 359.8140025295742, 245.1809692216882,
                                        1429.7818110154528, 982.4860953833982, 1669.1478865525835, 130.51733962426687,
                                        1175.3961800381574, 282.70551289652354, 37.159899712024455, 147.77717628953505,
                                        118.30531054483221, 21.953960852108022]), dtype=tf.float32)
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
