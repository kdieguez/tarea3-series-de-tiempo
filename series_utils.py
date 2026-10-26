import numpy as np
import pandas as pd
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

FIN_ENTRENAMIENTO = pd.Timestamp("2011-09-30")
ORIGEN_PRUEBA = pd.Timestamp("2011-10-02")
INICIO_PRUEBA = pd.Timestamp("2011-10-03")
FIN_PRUEBA = pd.Timestamp("2011-12-08")
DIA_INCOMPLETO = pd.Timestamp("2011-12-09")


def serie_diaria(df_limpio):
    """
    Venta neta diaria sobre el calendario completo, con los días sin
    transacciones como NaN y sin el último día (incompleto).
    """
    fecha = df_limpio["InvoiceDate"].dt.normalize()
    venta = df_limpio.groupby(fecha)["Importe"].sum()
    calendario = pd.date_range(venta.index.min(), venta.index.max(), freq="D")
    serie = venta.reindex(calendario)
    serie = serie[serie.index < DIA_INCOMPLETO]
    serie.name = "venta"
    return serie

def graficar_acf(r, n, titulo, archivo):
    import matplotlib.pyplot as plt
    kmax = len(r)
    banda = 1.96 / np.sqrt(n)
    k = np.arange(1, kmax + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.vlines(k, 0, r, color="steelblue", linewidth=2)
    ax.scatter(k, r, color="steelblue", zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(banda, color="red", linestyle="--", linewidth=1)
    ax.axhline(-banda, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(k)
    ax.set_xlabel("Rezago k (días abiertos)")
    ax.set_ylabel("$r_k$")
    ax.set_title(titulo)
    plt.tight_layout()
    plt.savefig(archivo, dpi=150, bbox_inches="tight")
    plt.close()