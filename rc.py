import numpy as np
from numpy.random import default_rng
from scipy import sparse
from scipy.sparse import linalg
from sklearn.linear_model import Ridge
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    # RC
    num_inputs : int = 3
    N : int = 50
    degree : float = 10.
    radius : float = 0.9
    leakage : float = 1.0
    bias : float = 0.5
    sigma : float = 1.
    discard_transient_length : int = 1000
    beta : float = 1e-8
    activation_func : '...' = np.tanh
    # data
    dt : float = 0.06
    int_dt : float = 0.001  # integration time step
    train_length : int = 10000
    prediction_steps : int = 10000


def _rescale_matrix(A, spectral_radius):
    if not isinstance(A, sparse.csr_matrix):
        A = sparse.csr_matrix(A)
    eigenvalues, _ = linalg.eigs(A)
    max_eigenvalue = max(abs(eigenvalues))
    A_scaled = A / max_eigenvalue * spectral_radius
    return(A_scaled)


def make_reservoir(h: Hyperparameters):
    rng = default_rng()
    nonzero_mat = rng.random((h.N, h.N)) < h.degree / h.N
    random_weights_mat = -1 + 2 * rng.random((h.N, h.N)) # uniform distribution on [-1, 1] at each matrix element
    A = nonzero_mat * random_weights_mat
    A[A == -0.0] = 0.0
    sA = sparse.csr_matrix(A)
    try:
        sA_rescaled = _rescale_matrix(sA, h.radius)
        return sA_rescaled
    except linalg.eigen.arpack.ArpackNoConvergence:
        return make_reservoir(h)
        

def make_input_matrix(h: Hyperparameters):
    rng = default_rng()
    W_in = h.sigma * (-1 + 2 * rng.random((h.N, h.num_inputs)))
    return W_in


def run_open_loop(A, W_in, input_data, h: Hyperparameters):
    """TODO"""
    res_states = np.zeros((h.train_length, h.N))
    for t in range(h.train_length - 1):
        res_states[t+1] = (1 - h.leakage) * res_states[t] \
                          + h.leakage * h.activation_func(A @ res_states[t] \
                                                          + W_in @ input_data[t+1] \
                                                          + h.bias)
    return res_states


def fit_output_weights(res_states, targets, h: Hyperparameters):
    """TODO"""
    clf = Ridge(alpha=h.beta, fit_intercept=False, solver='cholesky')
    clf.fit(res_states, targets)
    W_out = clf.coef_
    return W_out


def train_RC(input_data, h: Hyperparameters, train_targets=None):
    """TODO"""
    if train_targets is None:
        train_targets = input_data
    A = make_reservoir(h)
    W_in = make_input_matrix(h)
    res_states = run_open_loop(A, W_in, input_data, h)
    res_states_for_fit = res_states[h.discard_transient_length:-1]
    targets_for_fit = train_targets[h.discard_transient_length+1:h.train_length]
    W_out = fit_output_weights(res_states_for_fit, targets_for_fit, h)
    return W_out, A, W_in, res_states


def predict(W_out, A, W_in, training_res_states, h: Hyperparameters):
    """Closed loop prediction"""
    predictions = np.zeros((h.prediction_steps, h.num_inputs))
    res_state = training_res_states[-1]
    for t in range(h.prediction_steps):
        predictions[t] = W_out @ res_state
        res_state = (1 - h.leakage) * res_state \
             + h.leakage * h.activation_func(A @ res_state \
                                             + W_in @ predictions[t] \
                                             + h.bias)
    return predictions