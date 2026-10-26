import numpy as np
from scipy import stats

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
    X = np.column_stack([np.ones(len(y) - 1), y[:-1]])  
    beta, *_ = np.linalg.lstsq(X, y[1:], rcond=None)
    return beta[0], beta[1]

def ljung_box(y, m, K=0):
    """
    Prueba de Ljung-Box con m rezagos.
    K = número de parámetros estimados (si se aplica a residuos de un modelo).
    Devuelve (Q, valor_p).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    r = acf_manual(y, m)
    k = np.arange(1, m + 1)
    Q = n * (n + 2) * np.sum(r ** 2 / (n - k))
    p = stats.chi2.sf(Q, df=m - K)    
    return Q, p