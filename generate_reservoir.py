import numpy as np
from numpy.random import default_rng
from scipy import sparse
from scipy.sparse import linalg

def rescale(A, spectral_radius):
    if not isinstance(A, sparse.csr.csr_matrix):
        A = sparse.csr_matrix(A)
    
    # max_eigenvalue, _ = linalg.eigs(A, k=1, which='LM')
    # A_scaled = A / np.abs(max_eigenvalue) * spectral_radius
    eigenvalues, _ = linalg.eigs(A)
    max_eigenvalue = max(abs(eigenvalues))
    A_scaled = A / max_eigenvalue * spectral_radius
    return(A_scaled)


def erdos_renyi_reservoir(size, degree, radius):
    """TODO"""
    rng = default_rng()
    nonzero_mat = rng.random((size, size)) < degree / size # matrix of 1s
    random_weights_mat = -1 + 2 * rng.random((size, size)) # uniform distribution on [-1, 1] at each matrix element
    A = nonzero_mat * random_weights_mat
    A[A == -0.0] = 0.0
    sA = sparse.csr_matrix(A)

    # return rescale(sA, radius)

    try:
        sA_rescaled = rescale(sA, radius)
        return sA_rescaled
    except linalg.eigen.arpack.ArpackNoConvergence:
        # If cannot find eigenvalues, just start over
        return erdos_renyi_reservoir(size, degree, radius)
        # May cause infinite recursion but hopefully I avoid this! Sorry if it does


def modular_reservoir(size, num_comm, global_avg_degree, mix_ratio, radius):
    """

    Uses equations
    p_intra = k_avg / (N (1/n + (n-1)\chi / n))
    p_inter = \chi p_intra

    :param size: number of neurons in reservoir
    :param num_comm: nuber of distinct, same-size communities in reservoir
    :param global_avg_degree: avg. degree of nodes regardless of community structure
    :param mix_ratio: (\chi) ratio of (probability of a neuron connecting to one outside its community) / (probability of a neuron connecting to another in its community)
    :param radius: desired spectral radius
    :returns: 
    """

    assert size % num_comm == 0, 'Reservoir size is not an integer multiple of number of communities'
    comm_size = size // num_comm

    p_intra = global_avg_degree / (size * (1 / num_comm + (num_comm - 1) * mix_ratio / num_comm))
    p_inter = mix_ratio * p_intra
    assert p_intra <= 1, 'k_avg too high for this community structure'

    threshold_mat = np.array([[p_intra if (i // comm_size == j // comm_size) else p_inter for j in range(size)] for i in range(size)])

    rng = default_rng()
    nonzero_mat = rng.random((size, size)) < threshold_mat # matrix of 1s

    random_weights_mat = -1 + 2 * rng.random((size, size)) # uniform distribution on [-1, 1] at each matrix element
    A = nonzero_mat * random_weights_mat
    A[A == -0.0] = 0.0

    sA = sparse.csr_matrix(A)

    # return rescale(sA, radius)

    try:
        sA_rescaled = rescale(sA, radius)
        return sA_rescaled
    except linalg.eigen.arpack.ArpackNoConvergence:
        # If cannot find eigenvalues, just start over
        return modular_reservoir(size, num_comm, global_avg_degree, mix_ratio, radius)
        # May cause infinite recursion but hopefully I avoid this! Sorry if it does

def modular_reservoir_updated(size, num_comm, global_avg_degree, mix_ratio, radius):
    """

    Uses equations
    p_intra = n / N * (1 - \mu) * k_{avg}
    p_inter = \mu * k_{avg} / (N - N/n)

    :param size: number of neurons in reservoir
    :param num_comm: nuber of distinct, same-size communities in reservoir
    :param global_avg_degree: avg. degree of nodes regardless of community structure
    :param mix_ratio: (\mu) ratio of (avg. # of intercommunity links) / (avg. degree)
    :param radius: desired spectral radius
    :returns: 
    """

    assert size % num_comm == 0, 'Reservoir size is not an integer multiple of number of communities'
    comm_size = size // num_comm

    p_intra = num_comm / size * (1 - mix_ratio) * global_avg_degree
    p_inter = mix_ratio * global_avg_degree / (size - size/num_comm)
    assert p_intra <= 1, 'k_avg too high for this community structure'

    threshold_mat = np.array([[p_intra if (i // comm_size == j // comm_size) else p_inter for j in range(size)] for i in range(size)])

    rng = default_rng()
    nonzero_mat = rng.random((size, size)) < threshold_mat # matrix of 1s

    random_weights_mat = -1 + 2 * rng.random((size, size)) # uniform distribution on [-1, 1] at each matrix element
    A = nonzero_mat * random_weights_mat
    A[A == -0.0] = 0.0

    sA = sparse.csr_matrix(A)

    # return rescale(sA, radius)

    try:
        sA_rescaled = rescale(sA, radius)
        return sA_rescaled
    except linalg.eigen.arpack.ArpackNoConvergence:
        # If cannot find eigenvalues, just start over
        return modular_reservoir(size, num_comm, global_avg_degree, mix_ratio, radius)
        # May cause infinite recursion but hopefully I avoid this! Sorry if it does


def generate_W_in(num_inputs, res_size, sigma):
    """TODO"""
    rng = default_rng()
    W_in = sigma * (-1 + 2 * rng.random((res_size, num_inputs)))
    return W_in