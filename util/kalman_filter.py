import numpy as np
from tqdm import tqdm
from util.nmm import set_params, propagate_metrics


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_spd(A):
    k = 1
    B = (A + A.T)/2
    _, sigma, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(sigma),V))
    Ahat = (B+H)/2
    Ahat = (Ahat + Ahat.T)/2
    if isPD(Ahat):
        return Ahat, k
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(Ahat):
        mineig = np.min(np.real(np.linalg.eigvals(Ahat)))
        Ahat += I * (-mineig * k ** 2 + spacing)
        k += 1
        if k >= 1e5:
            k = -1
            return [Ahat, k]
    return [Ahat, k]


def kalman_filter(data, ex_tau, in_tau, ext_input, Fs=0.4e3):
    # ext_input = 10#300 # External input
    input_offset = np.empty(0)
    time = 5
    Seizure = data
    # Parameter initialization
    params = set_params(ex_tau, in_tau, ext_input, input_offset, time,Fs)

    if input_offset.size > 0:
        # Reset the offset
        my = np.mean(params['y'])
        input_offset = np.array([-my/0.0325]) # Scale (mV) to convert constant input to a steady-state effect on pyramidal membrane. NB DIVIDE BY 10e3 for VOLTS
        params = set_params(ex_tau, in_tau, ext_input, input_offset)

    # Retrive parameters into single variables
    A = params['A']
    B = params['B']
    C = params['C']
    H = params['H']
    N_inputs = params['N_inputs']
    N_samples = params['N_samples']
    N_states = params['N_states']
    N_syn = params['N_syn']
    Q = params['Q']
    R = params['R']
    v0 = params['v0']
    varsigma = params['varsigma']
    xi = params['xi']
    y = params['y']

    xi_hat_init = np.mean( params['xi'][:, np.int(np.round(N_samples/2))-1:] , axis = 1)
    P_hat_init = 10 * np.cov(params['xi'][:, np.int(np.round(N_samples/2))-1:])
    P_hat_init[2*N_syn:, 2*N_syn:] = np.eye(N_syn + N_inputs) * 10e-2

    # Set initial conditions for the Kalman Filter
    xi_hat = np.zeros([N_states, N_samples])
    P_hat = np.zeros([N_states, N_states, N_samples])
    P_diag = np.zeros([N_states, N_samples])

    xi_hat[:,0] = xi_hat_init
    P_hat[:,:,0] = P_hat_init

    anneal_on = 0 # Nice!
    kappa_0 = 10000
    T_end_anneal = N_samples/20

    # Get one channel at a time
    # NB - portal data is inverted. We need to scale it to some 'reasonable'
    # range for the model, but still capture amplitude differences between
    # seizures
    # y = -0.5 * Seizure[:, iCh-1:iCh]
    y = Seizure[0, :]
    N_samples = y.size

    # Redefine xi_hat and P because N_samples changed:
    #   Set initial conditions for the Kalman Filter
    xi_hat = np.zeros([N_states, N_samples])
    P_hat = np.zeros([N_states, N_states, N_samples])
    P_diag = np.zeros([N_states, N_samples])
    xi_hat[:,0] = xi_hat_init
    P_hat[:,:,0] = P_hat_init
    for t in tqdm(range(1,N_samples)):

        xi_0p = xi_hat[:, t-1].squeeze()
        P_0p = P_hat[:, :, t-1].squeeze()

        # Predict
        metrics = propagate_metrics(N_syn, N_states, N_inputs, A, B, C, P_0p, xi_0p, varsigma, v0, Q)
        xi_1m = metrics['xi_1m']
        P_1m = metrics['P_1m']

        if (t <= T_end_anneal) & (anneal_on):
            kappa = pow(kappa_0, (T_end_anneal-t-1)/(T_end_anneal-1))
        else:
            kappa = 1

        # K = P_1m*H'/(H*P_1m*H' + kappa*R);
        K = np.divide(np.matmul(P_1m, np.transpose(H)), np.matmul(H, np.matmul(P_1m, np.transpose(H))) + kappa*R)

        # Correct
        xi_1m = np.reshape(xi_1m, [xi_1m.size, 1])
        xi_hat[:, t:t+1] = xi_1m + K*(y[t] - np.matmul(H, xi_1m))

        P_hat[:,:,t] = np.matmul((np.identity(N_states) - np.matmul(K,H)), P_1m)
        try:
            P_hat[:,:,t] = (P_hat[:,:,t] + np.transpose(P_hat[:,:,t]))/2
            chol_matrix = np.linalg.cholesky(P_hat[:,:,t])
        except(np.linalg.LinAlgError):
            P_hat[:,:,t] , k = nearest_spd(P_hat[:,:,t])
            if k == -1:
                print('cannot find PSD')
        P_diag[:,t] = np.diag(P_hat[:,:,t])
    return xi_hat
