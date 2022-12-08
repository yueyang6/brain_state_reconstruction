import numpy as np
import pandas as pd
import math


def g(v, v0, varsigma):
    """ g(.) is the sigmoidal activation function """

    v_shape = np.array(v.shape)
    if v_shape.size > 0:
        g_ = np.zeros(v_shape)
        for i in range(0, v_shape[0]):
            g_[i] = 0.5 * math.erf((v[i] - v0) / (np.sqrt(2) * varsigma)) + 0.5
    else:
        raise ValueError("The size of input parameter 'v' must be greater than zero.")

    g_ = np.reshape(g_, (g_.size, 1))

    return g_


def set_params_test(ext_input, input_offset, TimeOfSim, Fs, taus, sigma_R=1e-3):
    """
    modified based on Artemio's codes
    set_params, migrated from the MATLAB version by Pip Karoly
    Set parameters for the neural mass model

    Inputs:
        ext_input - the input to the model
        input_offset - value of the offset (to compensate for a DC offset if required - i.e. there is a DC shift in the model but data may not be recorded with DC component)
        TimeOfSim - length of time to simulate data for
        Fs - sampling frequency (Hz)
        taus - [excitatory_tau, inhibitory_tau]
        sigma_R (optional) - Default value: 1e-3

    Outputs:
        A,B,C,H: model and observation matrices (defined in Karoly et al 2018)
        N_states,N_syn,N_inputs,N_samples: model dimensions
        xi, y: simulated data (xi = state vector, y = measurement)
        v0,varsigma: model constants
        Q,R: model and measurement noise

    Neural mass model parameters have been modified from Jansen & Rit (1995)

    For further references see:

        [1] Freestone, D. R., Karoly, P. J., Ne拧i?, D., Aram, P., Cook, M. J., & Grayden, D. B. (2014).
        Estimation of effective connectivity via data-driven neural modeling. Frontiers in neuroscience, 8, 383

        [2] Ahmadizadeh, S., Karoly, P. J., Ne拧i?, D., Grayden, D. B., Cook, M. J., Soudry, D., & Freestone, D. R. (2018).
        Bifurcation analysis of two coupled Jansen-Rit neural mass models. PloS one, 13(3), e0192842.

        [3] Kuhlmann, L., Freestone, D. R., Manton, J. H., Heyse, B., Vereecke, H. E., Lipping, T., ... & Liley, D. T. (2016).
        Neural mass model-based tracking of anesthetic brain states. NeuroImage, 133, 438-456.
    """

    scale = 50  # This is to get states and derivatives on the same order of magnitude

    mV = True  # Set units of membrane potentials (True for mV, False for V)
    V2mVfactor = 1e3

    # Time parameters
    dt = 1 / Fs  # Time step (s)
    N_samples = np.int(np.round(TimeOfSim / dt))  # No. of time poitns to simulate

    input_offset = np.array(input_offset)  # Making sure it is a NumPy array
    if input_offset.size > 0:
        N_inputs = 2
    else:
        N_inputs = 1

    N_syn = 4  # No. of synapses
    N_states = 3 * N_syn + N_inputs  # No. of states / dimension of the state-space. Plus one for input

    # Define the distrubance covariance matrix
    sigma_all = 5e-8
    sigma_input = 5e-4
    sigma_params = 5e-5
    sigma_offset = 5e-6

    Q = np.identity(N_states) * (scale * np.sqrt(
        dt) * sigma_all) ** 2  # add a tiny bit of noise to all states (added for numerical stability)
    Q[2 * N_syn:, 2 * N_syn:] = np.identity(N_syn + N_inputs) * (scale * np.sqrt(dt) * sigma_params) ** 2
    Q[2 * N_syn, 2 * N_syn] = (scale * np.sqrt(dt) * sigma_input) ** 2
    if N_inputs > 1:
        Q[2 * N_syn + 1, 2 * N_syn + 1] = (scale * np.sqrt(dt) * sigma_offset) ** 2

    # Measurement disturbance covariance
    R = sigma_R ** 2

    # General parameters from Jansen and Rit
    # Sigmoid bits
    f_max = 2.5  # maximum firing rate (spikes/s)
    r = 600
    varsigma = 1.699 / r  # spikes/(Vs)
    varsigma_sq = varsigma ** 2  # V

    # Synaptic kernel time constants
    ex_tau = taus[0]  # excitatory synaptic time constant (s)
    in_tau = taus[1]  # inhibitory synaptic time constant (s)

    # Synaptic gains
    alpha_e = (3.25e-3 / ex_tau) * ex_tau  # gain of excitatory synapses (V)
    alpha_i = (-22e-3 / in_tau) * in_tau # gain of inhibitory synapses (V)
    v0 = 0.006

    # input to py population
    #
    # SCALE 1 - this is to avoid large differences between states upsetting the filter
    # (magnitude of the membrane potentials and their derivatives)
    ext_input = ext_input * scale
    # SCALE 2 - this converts a constant input to its effect on the pyramidal
    # membrane potential by taking the steady state limit of the synaptic kernel
    # (assumption that the input varies much slower than the state variables).
    ext_input = ext_input * alpha_e / ex_tau * ex_tau ** 2
    #       ~~~~~   ~~~~~~~~~~~~~~   ~~~~~~~~~
    #       input   synaptic gain    integral of kernel

    # measurement DC offset
    input_offset = input_offset * scale
    input_offset = input_offset * alpha_e / ex_tau * ex_tau ** 2

    if mV:
        Q = (V2mVfactor ** 2) * Q
        R = (V2mVfactor ** 2) * R

        r = r / V2mVfactor
        varsigma = 1.699 / r  # (spikes/(Vs))
        varsigma_sq = varsigma ** 2
        v0 = v0 * V2mVfactor
        alpha_e = alpha_e * V2mVfactor  # gain of excitatory synapses (V)
        alpha_i = alpha_i * V2mVfactor  # gain of inhibitory synapses (V)

        ext_input = ext_input * V2mVfactor
        input_offset = input_offset * V2mVfactor

    # Connectivity constants to relate to Jansen and Rit 1995 model
    ConnectivityConst = 270  # Jansen and Rit connectivity parameters. Either 135, 270 or 675
    C1 = ConnectivityConst
    C2 = 0.8 * ConnectivityConst
    C3 = 0.25 * ConnectivityConst
    C4 = 0.25 * ConnectivityConst

    #   Model structure
    #  ~~~~~~~~~~~~~~~~~
    #
    #           X
    #       __  |  __
    #      /  \ | /  \
    #     /  04 | 01  \
    #     |     P     |
    #  ^  |     | |   |  ^
    #  |  E     | v   I  |  direction of info
    #     03   /|\   02
    #     |   / | \   |
    #      \_/  |  \_/
    #           v
    # population types: E, P, I, X
    # synapses: 01 (IP), 02 (PI), 03 (PE), 04 (EP)

    # Initialize some variables
    tau = np.zeros([4, ])
    alpha = np.zeros([4, ])

    # This is the Observation function
    H = np.zeros([1, N_states])  # Initialize to zeros and later add 1s to states that contribute to EEG

    # Initialize adjancy matrix
    Gamma = np.zeros([2 * N_syn + N_inputs, 2 * N_syn + N_inputs])  # - plus 1 for input

    # Specify synapses
    syn_index = 0  # Python indexing starts at 0, hence syn_index 0 is equivalent to the syn_index 1 in Matlab

    # Syn1, connection from I to P
    tau[syn_index,] = in_tau
    alpha[syn_index,] = alpha_i * 2 * f_max * C4 * dt / tau[
        syn_index,]  # note the time constant and time step are in the gains
    presyn_inputs = np.array([2])  # the presynaptic population is getting inputs from synapses 2
    # Check if presyn_inputs not empty
    if presyn_inputs.size > 0:
        """ Note! the folllowing line was brought from Matlab code. -1 was
        turned into -2 to comply with Python indexing"""
        Gamma[2 * (
                    syn_index + 1) - 1, 2 * presyn_inputs - 2] = 1  # set the entries of Gamma corresponding to indices of presynaptic inputs to 1
    H[0, 2 * (syn_index + 1) - 2] = 1

    # Syn2, connection from P to I
    syn_index = syn_index + 1
    tau[syn_index,] = ex_tau
    alpha[syn_index,] = alpha_e * 2 * f_max * C3 * dt / tau[
        syn_index,]  # note the time constsnt and time step are in the gains
    presyn_inputs = np.array([1, 4, 5])  # the presynaptic population is getting inputs from synapses 1, 4, 5
    if presyn_inputs.size > 0:
        Gamma[2 * (syn_index + 1) - 1, 2 * presyn_inputs - 2] = 1
    H[0, 2 * (syn_index + 1) - 2] = 0  # set to one if it contributes to the EEG (i.e. if the synapse is to Py cells)

    # Syn3, connection from P to E
    syn_index = syn_index + 1
    tau[syn_index,] = ex_tau
    alpha[syn_index,] = alpha_e * 2 * f_max * C1 * dt / tau[
        syn_index,]  # note the time constsnt and time step are in the gains
    presyn_inputs = np.array([1, 4, 5])  # the presynaptic population is getting inputs from synapses 1, 4, 5
    if presyn_inputs.size > 0:
        Gamma[2 * (syn_index + 1) - 1, 2 * presyn_inputs - 2] = 1
    H[0, 2 * (syn_index + 1) - 2] = 0

    # Syn4, connection from E to P
    syn_index = syn_index + 1
    tau[syn_index,] = ex_tau
    alpha[syn_index,] = alpha_e * 2 * f_max * C2 * dt / tau[
        syn_index,]  # note the time constsnt and time step are in the gains
    presyn_inputs = np.array([3])  # the presynaptic population is getting inputs from synapses 3
    if presyn_inputs.size > 0:
        Gamma[2 * (syn_index + 1) - 1, 2 * presyn_inputs - 2] = 1
    H[0, 2 * (syn_index + 1) - 2] = 1

    # For input
    syn_index = syn_index + 1
    H[0, 2 * (syn_index + 1) - 2] = 1  # The input contributes to the observation function

    if N_inputs > 1:
        # Offset term
        H[0, 2 * (
                    syn_index + 1) - 1] = 1  # Offset contributes to the observation function. Notice the -1 instead of -2 in the indexing.

    # Rescale
    H = H / scale  # Scale! This helps deal with our numerical issues.

    # Define A
    #
    # A is made up of the submatrices Psi in a block diagonal structure.
    # There is a Psi for each connection in the model. This is where all the
    # synaptic time constants enter the system. Further, the scale paramter
    # enters here (and with C (multiplicative factor) and with the H (divisor).
    Psi = np.zeros([2 * N_syn, 2 * N_syn])  # initialise Psi, the component of A for fast states
    for n in range(0, N_syn):
        index = 2 * n
        Psi[index: index + 2, index: index + 2] = np.array([[0, scale],
                                                            [-1 / (scale * tau[n] ** 2), -2 / tau[n]]])

    # A = [1+dt*Psi, 0
    #          0   , 1]
    # Where 1 is the identity matrix of the appropriate size.
    a11 = np.identity(2 * N_syn) + dt * Psi  # [1]+dt*Psi
    a12 = np.zeros([2 * N_syn, N_syn + N_inputs])  # [0]
    a21 = np.zeros([N_syn + N_inputs, 2 * N_syn])  # [0]
    a22 = np.identity(N_syn + N_inputs)  # [1]
    # Concatenate horizontally
    a1 = np.concatenate((a11, a12), axis=1)
    a2 = np.concatenate((a21, a22), axis=1)
    # Concantenate vertically
    A = np.concatenate((a1, a2))

    # Define B
    #
    # Theta = [0 0 ... 0
    #          1 0 ... 0
    #          0 0 ... 0
    #          0 1 ... 0
    #              ...
    #          0 0 ... 0
    #          0 0 ... 1]
    # B = [0 Theta ; 0 0]
    Theta = np.zeros([2 * N_syn, N_syn])  # Theta is used twice!
    for n in range(0, N_syn):
        index = 2 * n
        Theta[index: index + 2, n: n + 1] = np.array([[0], [1]])
    b1 = np.concatenate((np.zeros([2 * N_syn, 2 * N_syn + N_inputs]), Theta), axis=1)
    b2 = np.zeros([N_syn + N_inputs, 3 * N_syn + N_inputs])
    B = np.concatenate((b1, b2))

    # Define C (adjacency matrix)
    c1 = np.concatenate((Gamma / scale, np.zeros([2 * N_syn + N_inputs, N_syn])), axis=1)
    c2 = np.zeros([N_syn, 3 * N_syn + N_inputs])
    C = np.concatenate((c1, c2))

    # Model structure has been defined. WE ARE NOW BORN TO RUN!
    # Set up forward simulation

    # Define initial conditions
    ve = 0;
    ze = 0;
    vp1 = 0;
    zp1 = 0;
    vp2 = 0;
    zp2 = 0;
    vi = 0;
    zi = 0;  # vp3 = 0; zp3 = 0; # <- unused variables. Legacy?
    x = np.array([ve, ze, vp1, zp1, vp2, zp2, vi, zi])
    xi = np.zeros([N_states, N_samples])
    i_off = np.array([input_offset])
    xi[:, 0] = np.concatenate((x, np.array([ext_input]), np.reshape(i_off, (i_off.size,)), alpha))
    # ext_input and alpha are set in set_params

    # np.random.seed(1)
    w = np.random.multivariate_normal(mean=np.zeros(N_states),
                                      cov=np.array(Q),
                                      size=N_samples)
    w = np.array(w)
    w = np.transpose(w)

    phi_p = np.zeros([1, N_samples])

    # Run the model forward
    for n in range(0, N_samples - 1):
        v = np.matmul(C, xi[:, n])
        phi = g(v, v0, varsigma)
        phi_p[0, n:n + 1] = phi[3:4, 0]
        xi[:, n + 1:n + 2] = np.matmul(A, xi[:, n:n + 1]) + np.multiply(np.matmul(B, xi[:, n:n + 1]), phi)


    # np.random.seed(1)
    v = np.sqrt(R) * np.random.randn(1, N_samples)
    y = np.matmul(H, xi) + v

    # Define output arguments
    output = {
        'A': A,
        'B': B,
        'C': C,
        'H': H,
        'N_states': N_states,
        'N_syn': N_syn,
        'N_inputs': N_inputs,
        'N_samples': N_samples,
        'xi': xi,
        'y': y,
        'v0': v0,
        'varsigma': varsigma,
        'Q': Q,
        'R': R,
    }
    # output state vector and observation
    df = pd.DataFrame(output['xi'])
    df.loc[len(df)] = output['y'][0]
    # remove first second to avoid model warm up issue
    df = df.loc[:, 400:]
    return df


def generate_recordings(input_offset, TimeOfSim, Fs, params, sigma_R=1e-3):
    """

    Inputs:

        input_offset - value of the offset (to compensate for a DC offset if required - i.e. there is a DC shift in the model but data may not be recorded with DC component)
        TimeOfSim - length of time to simulate data for
        Fs - sampling frequency (Hz)
        params - [external_input, excitatory_tau, inhibitory_tau]
        sigma_R (optional) - Default value: 1e-3

    Outputs:
        A,B,C,H: model and observation matrices (defined in Karoly et al 2018)
        N_states,N_syn,N_inputs,N_samples: model dimensions
        xi, y: simulated data (xi = state vector, y = measurement)
        v0,varsigma: model constants
        Q,R: model and measurement noise

    Neural mass model parameters have been modified from Jansen & Rit (1995)

    For further references see:

        [1] Freestone, D. R., Karoly, P. J., Ne拧i?, D., Aram, P., Cook, M. J., & Grayden, D. B. (2014).
        Estimation of effective connectivity via data-driven neural modeling. Frontiers in neuroscience, 8, 383

        [2] Ahmadizadeh, S., Karoly, P. J., Ne拧i?, D., Grayden, D. B., Cook, M. J., Soudry, D., & Freestone, D. R. (2018).
        Bifurcation analysis of two coupled Jansen-Rit neural mass models. PloS one, 13(3), e0192842.

        [3] Kuhlmann, L., Freestone, D. R., Manton, J. H., Heyse, B., Vereecke, H. E., Lipping, T., ... & Liley, D. T. (2016).
        Neural mass model-based tracking of anesthetic brain states. NeuroImage, 133, 438-456.
    """
    scale = 50  # This is to get states and derivatives on the same order of magnitude

    mV = True  # Set units of membrane potentials (True for mV, False for V)
    V2mVfactor = 1e3
    ext_input = params[0]
    ext_tag = params[0]

    # Time parameters
    dt = 1 / Fs  # Time step (s)
    N_samples = np.int(np.round(TimeOfSim / dt))  # No. of time poitns to simulate

    input_offset = np.array(input_offset)  # Making sure it is a NumPy array
    if input_offset.size > 0:
        N_inputs = 2
    else:
        N_inputs = 1

    N_syn = 4  # No. of synapses
    N_states = 3 * N_syn + N_inputs  # No. of states / dimension of the state-space. Plus one for input

    # Define the distrubance covariance matrix
    sigma_all = 5e-8
    sigma_input = 5e-4
    sigma_params = 5e-5
    sigma_offset = 5e-6

    Q = np.identity(N_states) * (scale * np.sqrt(
        dt) * sigma_all) ** 2  # add a tiny bit of noise to all states (added for numerical stability)
    Q[2 * N_syn:, 2 * N_syn:] = np.identity(N_syn + N_inputs) * (scale * np.sqrt(dt) * sigma_params) ** 2
    Q[2 * N_syn, 2 * N_syn] = (scale * np.sqrt(dt) * sigma_input) ** 2
    if N_inputs > 1:
        Q[2 * N_syn + 1, 2 * N_syn + 1] = (scale * np.sqrt(dt) * sigma_offset) ** 2

    # Measurement disturbance covariance
    R = sigma_R ** 2

    # General parameters from Jansen and Rit
    # Sigmoid bits
    f_max = 2.5  # maximum firing rate (spikes/s)
    r = 600
    varsigma = 1.699 / r  # spikes/(Vs)
    varsigma_sq = varsigma ** 2  # V

    # Synaptic kernel time constants
    ex_tau = params[1]
    in_tau = params[2]

    # Synaptic gains
    alpha_e = (3.25e-3 / ex_tau) * ex_tau  # gain of excitatory synapses (V)
    alpha_i = (-22e-3 / in_tau) * in_tau # gain of inhibitory synapses (V)
    v0 = 0.006

    # input to py population
    #
    # SCALE 1 - this is to avoid large differences between states upsetting the filter
    # (magnitude of the membrane potentials and their derivatives)
    ext_input = ext_input * scale
    # SCALE 2 - this converts a constant input to its effect on the pyramidal
    # membrane potential by taking the steady state limit of the synaptic kernel
    # (assumption that the input varies much slower than the state variables).
    ext_input = ext_input * alpha_e / ex_tau * ex_tau ** 2
    #       ~~~~~   ~~~~~~~~~~~~~~   ~~~~~~~~~
    #       input   synaptic gain    integral of kernel

    # measurement DC offset
    input_offset = input_offset * scale
    input_offset = input_offset * alpha_e / ex_tau * ex_tau ** 2

    if mV:
        Q = (V2mVfactor ** 2) * Q
        R = (V2mVfactor ** 2) * R

        r = r / V2mVfactor
        varsigma = 1.699 / r  # (spikes/(Vs))
        varsigma_sq = varsigma ** 2
        v0 = v0 * V2mVfactor
        alpha_e = alpha_e * V2mVfactor  # gain of excitatory synapses (V)
        alpha_i = alpha_i * V2mVfactor  # gain of inhibitory synapses (V)

        ext_input = ext_input * V2mVfactor
        input_offset = input_offset * V2mVfactor

    # Connectivity constants to relate to Jansen and Rit 1995 model
    ConnectivityConst = 270  # Jansen and Rit connectivity parameters. Either 135, 270 or 675
    C1 = ConnectivityConst
    C2 = 0.8 * ConnectivityConst
    C3 = 0.25 * ConnectivityConst
    C4 = 0.25 * ConnectivityConst

    #   Model structure
    #  ~~~~~~~~~~~~~~~~~
    #
    #           X
    #       __  |  __
    #      /  \ | /  \
    #     /  04 | 01  \
    #     |     P     |
    #  ^  |     | |   |  ^
    #  |  E     | v   I  |  direction of info
    #     03   /|\   02
    #     |   / | \   |
    #      \_/  |  \_/
    #           v
    # population types: E, P, I, X
    # synapses: 01 (IP), 02 (PI), 03 (PE), 04 (EP)

    # Initialize some variables
    tau = np.zeros([4, ])
    alpha = np.zeros([4, ])

    # This is the Observation function
    H = np.zeros([1, N_states])  # Initialize to zeros and later add 1s to states that contribute to EEG

    # Initialize adjancy matrix
    Gamma = np.zeros([2 * N_syn + N_inputs, 2 * N_syn + N_inputs])  # - plus 1 for input

    # Specify synapses
    syn_index = 0  # Python indexing starts at 0, hence syn_index 0 is equivalent to the syn_index 1 in Matlab

    # Syn1, connection from I to P
    tau[syn_index,] = in_tau
    alpha[syn_index,] = alpha_i * 2 * f_max * C4 * dt / tau[
        syn_index,]  # note the time constant and time step are in the gains
    presyn_inputs = np.array([2])  # the presynaptic population is getting inputs from synapses 2
    # Check if presyn_inputs not empty
    if presyn_inputs.size > 0:
        """ Note! the folllowing line was brought from Matlab code. -1 was
        turned into -2 to comply with Python indexing"""
        Gamma[2 * (
                    syn_index + 1) - 1, 2 * presyn_inputs - 2] = 1  # set the entries of Gamma corresponding to indices of presynaptic inputs to 1
    H[0, 2 * (syn_index + 1) - 2] = 1

    # Syn2, connection from P to I
    syn_index = syn_index + 1
    tau[syn_index,] = ex_tau
    alpha[syn_index,] = alpha_e * 2 * f_max * C3 * dt / tau[
        syn_index,]  # note the time constsnt and time step are in the gains
    presyn_inputs = np.array([1, 4, 5])  # the presynaptic population is getting inputs from synapses 1, 4, 5
    if presyn_inputs.size > 0:
        Gamma[2 * (syn_index + 1) - 1, 2 * presyn_inputs - 2] = 1
    H[0, 2 * (syn_index + 1) - 2] = 0  # set to one if it contributes to the EEG (i.e. if the synapse is to Py cells)

    # Syn3, connection from P to E
    syn_index = syn_index + 1
    tau[syn_index,] = ex_tau
    alpha[syn_index,] = alpha_e * 2 * f_max * C1 * dt / tau[
        syn_index,]  # note the time constsnt and time step are in the gains
    presyn_inputs = np.array([1, 4, 5])  # the presynaptic population is getting inputs from synapses 1, 4, 5
    if presyn_inputs.size > 0:
        Gamma[2 * (syn_index + 1) - 1, 2 * presyn_inputs - 2] = 1
    H[0, 2 * (syn_index + 1) - 2] = 0

    # Syn4, connection from E to P
    syn_index = syn_index + 1
    tau[syn_index,] = ex_tau
    alpha[syn_index,] = alpha_e * 2 * f_max * C2 * dt / tau[
        syn_index,]  # note the time constsnt and time step are in the gains
    presyn_inputs = np.array([3])  # the presynaptic population is getting inputs from synapses 3
    if presyn_inputs.size > 0:
        Gamma[2 * (syn_index + 1) - 1, 2 * presyn_inputs - 2] = 1
    H[0, 2 * (syn_index + 1) - 2] = 1

    # For input
    syn_index = syn_index + 1
    H[0, 2 * (syn_index + 1) - 2] = 1  # The input contributes to the observation function

    if N_inputs > 1:
        # Offset term
        H[0, 2 * (
                    syn_index + 1) - 1] = 1  # Offset contributes to the observation function. Notice the -1 instead of -2 in the indexing.

    # Rescale
    H = H / scale  # Scale! This helps deal with our numerical issues.

    # Define A
    #
    # A is made up of the submatrices Psi in a block diagonal structure.
    # There is a Psi for each connection in the model. This is where all the
    # synaptic time constants enter the system. Further, the scale paramter
    # enters here (and with C (multiplicative factor) and with the H (divisor).
    Psi = np.zeros([2 * N_syn, 2 * N_syn])  # initialise Psi, the component of A for fast states
    for n in range(0, N_syn):
        index = 2 * n
        Psi[index: index + 2, index: index + 2] = np.array([[0, scale],
                                                            [-1 / (scale * tau[n] ** 2), -2 / tau[n]]])

    # A = [1+dt*Psi, 0
    #          0   , 1]
    # Where 1 is the identity matrix of the appropriate size.
    a11 = np.identity(2 * N_syn) + dt * Psi  # [1]+dt*Psi
    a12 = np.zeros([2 * N_syn, N_syn + N_inputs])  # [0]
    a21 = np.zeros([N_syn + N_inputs, 2 * N_syn])  # [0]
    a22 = np.identity(N_syn + N_inputs)  # [1]
    # Concatenate horizontally
    a1 = np.concatenate((a11, a12), axis=1)
    a2 = np.concatenate((a21, a22), axis=1)
    # Concantenate vertically
    A = np.concatenate((a1, a2))

    # Define B
    #
    # Theta = [0 0 ... 0
    #          1 0 ... 0
    #          0 0 ... 0
    #          0 1 ... 0
    #              ...
    #          0 0 ... 0
    #          0 0 ... 1]
    # B = [0 Theta ; 0 0]
    Theta = np.zeros([2 * N_syn, N_syn])  # Theta is used twice!
    for n in range(0, N_syn):
        index = 2 * n
        Theta[index: index + 2, n: n + 1] = np.array([[0], [1]])
    b1 = np.concatenate((np.zeros([2 * N_syn, 2 * N_syn + N_inputs]), Theta), axis=1)
    b2 = np.zeros([N_syn + N_inputs, 3 * N_syn + N_inputs])
    B = np.concatenate((b1, b2))

    # Define C (adjacency matrix)
    c1 = np.concatenate((Gamma / scale, np.zeros([2 * N_syn + N_inputs, N_syn])), axis=1)
    c2 = np.zeros([N_syn, 3 * N_syn + N_inputs])
    C = np.concatenate((c1, c2))

    # Model structure has been defined. WE ARE NOW BORN TO RUN!
    # Set up forward simulation

    # Define initial conditions
    ve = 0;
    ze = 0;
    vp1 = 0;
    zp1 = 0;
    vp2 = 0;
    zp2 = 0;
    vi = 0;
    zi = 0;  # vp3 = 0; zp3 = 0; # <- unused variables. Legacy?
    x = np.array([ve, ze, vp1, zp1, vp2, zp2, vi, zi])
    xi = np.zeros([N_states, N_samples])
    i_off = np.array([input_offset])
    xi[:, 0] = np.concatenate((x, np.array([ext_input]), np.reshape(i_off, (i_off.size,)), alpha))
    # ext_input and alpha are set in set_params

    # np.random.seed(1)
    w = np.random.multivariate_normal(mean=np.zeros(N_states),
                                      cov=np.array(Q),
                                      size=N_samples)
    w = np.array(w)
    w = np.transpose(w)

    phi_p = np.zeros([1, N_samples])

    # Run the model forward
    for n in range(0, N_samples - 1):
        v = np.matmul(C, xi[:, n])
        phi = g(v, v0, varsigma)
        phi_p[0, n:n + 1] = phi[3:4, 0]
        xi[:, n + 1:n + 2] = np.matmul(A, xi[:, n:n + 1]) + np.multiply(np.matmul(B, xi[:, n:n + 1]), phi) + w[:,n:n + 1]


    # np.random.seed(1)
    v = np.sqrt(R) * np.random.randn(1, N_samples)
    y = np.matmul(H, xi) + v

    # Define output arguments
    output = {
        'A': A,
        'B': B,
        'C': C,
        'H': H,
        'N_states': N_states,
        'N_syn': N_syn,
        'N_inputs': N_inputs,
        'N_samples': N_samples,
        'xi': xi,
        'y': y,
        'v0': v0,
        'varsigma': varsigma,
        'Q': Q,
        'R': R,
    }
    # output state vector and observation
    df = pd.DataFrame(output['xi'])
    df.loc[len(df)] = output['y'][0]
    # remove the fist second to avoid model warm up issue
    df = df.loc[:, 400:]
    df.to_csv('ex%.4f_ih%.4f_in%.0f.csv' % (ex_tau, in_tau, ext_tag))

