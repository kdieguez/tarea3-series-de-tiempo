import numpy as np


def acf_manual(y, kmax):
    """
    Autocorrelación muestral r_k para k = 1, ..., kmax:
    r_k = sum_{t=k+1}^n (y_t - ybar)(y_{t-k} - ybar) / sum_{t=1}^n (y_t - ybar)^2
    """
    y = np.asarray(y, dtype=float)
    d = y - y.mean()             
    denom = np.sum(d ** 2)
    r = np.empty(kmax)
    for k in range(1, kmax + 1):
        r[k - 1] = np.sum(d[k:] * d[:-k]) / denom
    return r


def ols_ar1(y):
    """
    OLS de y_t contra y_{t-1} (con intercepto). Devuelve (c_hat, phi_hat).
    """
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y) - 1), y[:-1]])  # [1, y_{t-1}]
    beta, *_ = np.linalg.lstsq(X, y[1:], rcond=None)
    return beta[0], beta[1]