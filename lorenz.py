import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import zscore

lyapunov_exponent = 0.9056
lyapunov_time = 1 / lyapunov_exponent

def lorenz_eqns(t, X, sigma, beta, rho):
    """The Lorenz equations"""
    x, y, z = X
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return dx_dt, dy_dt, dz_dt


def solve(n_step, tau, sigma=10, beta=8/3, rho=28, init_cond=(0, 1, 1.05), method='RK45', z_score=True, discard_transients=True):
    t_max = n_step * tau
    soln = solve_ivp(lorenz_eqns, (0, t_max), init_cond, args=(sigma, beta, rho),
                 dense_output=True, method=method)
    t = np.linspace(0, t_max, n_step+1)
    X = soln.sol(t)  # Polynomial interpolation of system state
    if z_score: X = zscore(X, axis=1)  # z score each of x, y, z separately
    if discard_transients: X = X[:, int(np.floor(n_step * 0.1)):]
    return X, t