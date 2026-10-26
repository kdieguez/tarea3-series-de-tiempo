import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
matplotlib.use("Agg")
from series_utils import acf_manual, ols_ar1
from statsmodels.tsa.stattools import acf


def simular_ar1(phi, n=300, burn=200, c=0.0, sigma=1.0, rng=None):
    """
    Simula n pasos de un AR(1): Y_t = c + phi*Y_{t-1} + eps_t.
    Genera n + burn valores y descarta los primeros 'burn'.
    """
    if rng is None:
        rng = np.random.default_rng()
    total = n + burn
    eps = rng.normal(0, sigma, size=total)  
    y = np.zeros(total)                     
    for t in range(1, total):
        y[t] = c + phi * y[t - 1] + eps[t]
    return y[burn:]                     


phis = [0, 0.5, 0.9, 1]
rng = np.random.default_rng(2026)

fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
for ax, phi in zip(axes, phis):
    y = simular_ar1(phi, rng=rng)
    ax.plot(y, linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(f"AR(1) con $\\phi$ = {phi}")
    ax.set_ylabel("$y_t$")
axes[-1].set_xlabel("t")
plt.tight_layout()
plt.savefig("documento/figuras/p0_realizaciones.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figura guardada: documento/figuras/p0_realizaciones.png")

y_prueba = simular_ar1(0.5, rng=np.random.default_rng(1))
print("r_k a mano:     ", np.round(acf_manual(y_prueba, 5), 4))
print("r_k statsmodels:", np.round(acf(y_prueba, nlags=5)[1:], 4))
print("phi OLS:", round(ols_ar1(y_prueba)[1], 4), "(verdadero: 0.5)")

R = 5000
n = 300
banda = 1.96 / np.sqrt(n)                  
semillas = {0: 20260, 0.5: 20261, 0.9: 20262, 1: 20263}


def error_estandar(x):
    """Error estándar del promedio de x."""
    return np.std(x, ddof=1) / np.sqrt(len(x))


filas = []
for phi in phis:
    rng = np.random.default_rng(semillas[phi])
    r1 = np.empty(R)
    r5 = np.empty(R)
    phi_hat = np.empty(R)

    for i in range(R):
        y = simular_ar1(phi, n=n, rng=rng)
        r = acf_manual(y, 5)
        r1[i] = r[0]
        r5[i] = r[4]
        phi_hat[i] = ols_ar1(y)[1]

    rechazo = (np.abs(r1) > banda).astype(float)  

    filas.append({
        "phi": phi,
        "media_r1": r1.mean(),        "ee_r1": error_estandar(r1),
        "media_r5": r5.mean(),        "ee_r5": error_estandar(r5),
        "teorico_phi5": phi ** 5,
        "tasa_rechazo": rechazo.mean(), "ee_rechazo": error_estandar(rechazo),
        "media_phi_ols": phi_hat.mean(), "ee_phi_ols": error_estandar(phi_hat),
    })
    print(f"phi = {phi} listo")

tabla_ar1 = pd.DataFrame(filas)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(tabla_ar1.round(4))

os.makedirs("resultados", exist_ok=True)
tabla_ar1.to_csv("resultados/p0_tabla_ar1.csv", index=False)
print("Tabla guardada en resultados/p0_tabla_ar1.csv")