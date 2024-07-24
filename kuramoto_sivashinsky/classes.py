"""
From: https://github.com/awikner/res-noise-stabilization
"""

import numpy as np
from ks_etdrk4 import kursiv_predict

class NumericalModel:
    """Class for generating training or testing data using one of the test numerical models."""
    def __init__(self, tau=0.1, int_step=1, T=300, ttsplit=5000, u0=0, system='KS',
                 params=np.array([[], []], dtype=np.complex128), dnoise_gen = None, dnoise_scaling = 0):
        """Creates the NumericalModel object.
        Args:
            self: NumericalModel object.
            tau: Time step between measurements of the system dynamics.
            int_step: the number of numerical integration steps between each measurement.
            T: Total number of measurements.
            ttsplit: Number of measurements to be used in the training data set.
            u0: Initial condition for integration.
            system: Name of system to generate data from. Options: 'lorenz', 'KS','KS_d2175'
            params: Internal parameters for model integration. Currently only used by KS options.
        Returns:
            Complete NumericalModel object with precomputed internal parameters."""

        if system == 'KS':
            if isinstance(dnoise_gen, type(None)) or dnoise_scaling == 0:
                u_arr, self.params = kursiv_predict(u0, tau=tau, T=T, params=params, int_steps = int_step)
            else:
                dnoise = dnoise_gen.standard_normal((64, T*int_step+int_step))*np.sqrt(dnoise_scaling)
                u_arr, self.params = kursiv_predict(u0, tau=tau, T=T, params=params, int_steps=int_step,
                                                    noise= dnoise)
            self.input_size = u_arr.shape[0]
            u_arr = np.ascontiguousarray(u_arr) / (1.1876770355823614)
        elif system == 'KS_d2175':
            if isinstance(dnoise_gen, type(None)) or dnoise_scaling == 0:
                u_arr, self.params = kursiv_predict(u0, tau=tau, T=T, d=21.75, params=params, int_steps = int_step)
            else:
                dnoise = dnoise_gen.standard_normal((64, T * int_step + int_step)) * np.sqrt(dnoise_scaling)
                u_arr, self.params = kursiv_predict(u0, tau=tau, T=T, d=21.75, params=params,
                                                    int_steps=int_step, noise = dnoise)
            self.input_size = u_arr.shape[0]
            u_arr = np.ascontiguousarray(u_arr) / (1.2146066380280796)
        else:
            raise ValueError

        self.train_length = ttsplit
        self.u_arr_train = u_arr[:, :ttsplit + 1]

        # u[ttsplit], the (ttsplit + 1)st element, is the last in u_arr_train and the first in u_arr_test
        self.u_arr_test = u_arr[:, ttsplit:]