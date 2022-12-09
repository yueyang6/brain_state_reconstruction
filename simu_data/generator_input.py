import numpy as np
from multiprocessing import Pool
from itertools import repeat
from multiprocessing import freeze_support
from itertools import product
from statsmodels.stats.diagnostic import acorr_ljungbox
from util.nmm_train import generate_recordings, set_params_test


if __name__ == '__main__':
    # starting point of tau_e, tau_i
    tau_e = 0.01
    tau_i = 0.01
    # step of increase
    step = 0.01/3
    # testing params
    input_offset_test = np.empty(0)
    time_test = 5
    Fs_test = 0.4e3
    # generator params
    input_offset = np.empty(0)
    time = 10
    Fs = 0.4e3
    while tau_e <= 0.015:
        tau_i = 0.01
        while tau_i <= 0.015:
            ex_input = 0
            input_flag = 0
            osci_flag = 0
            min_input = 0
            max_input = 0
            # input has not been increased for 15 times
            while input_flag <= 15:
                result = set_params_test(ex_input, input_offset_test, time_test, Fs_test, (tau_e, tau_i))
                y = result.iloc[13, :]
                # statistical test
                ljun_test = acorr_ljungbox(y).loc[:,'lb_pvalue'].min()
                # and_stat = anderson_result[0]
                # and_crit = anderson_result[1][4]

                if ljun_test < 1e-4:
                    osci_flag += 1
                    # lowest external input
                    if osci_flag == 1:
                        min_input = ex_input
                        input_flag = 15
                    # highest external input
                    if ex_input > max_input:
                        max_input = ex_input
                else:
                    input_flag += 1
                    # generate recordings if oscillation exists
                    if osci_flag > 0:
                        tau_es = [tau_e]
                        tau_is = [tau_i]
                        # generate 30 recordings for the same taus
                        ex_inputs = np.linspace(min_input, max_input, num=30).tolist()
                        freeze_support()
                        pool = Pool()
                        pool.starmap(generate_recordings,zip(repeat(input_offset), repeat(time), repeat(Fs),
                                                    product(ex_inputs, tau_es, tau_is)))
                        pool.close()
                        break
                if ex_input < 100:
                    ex_input += 10
                elif ex_input < 4000:
                    ex_input += 100
                else:
                    ex_input += 1000
            # adjustable
            tau_i += step
        tau_e += step
