import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
matplotlib.use("Agg")
from series_utils import acf_manual, ols_ar1
from statsmodels.tsa.stattools import acf
from series_utils import ljung_box
from statsmodels.stats.diagnostic import acorr_ljungbox


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

t = np.arange(1, n + 1)
m_tendencia = 2 * t / n
m_semanal = np.where(t % 7 == 6, -1.5, 0.25)    

medias = {
    "Ruido blanco": np.zeros(n),
    "Tendencia": m_tendencia,
    "Semanal": m_semanal,
}
semillas_04 = {"Ruido blanco": 20264, "Tendencia": 20265, "Semanal": 20266}

rng_fig = np.random.default_rng(2027)
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, (nombre, m) in zip(axes, medias.items()):
    y = m + rng_fig.normal(0, 1, size=n)
    ax.plot(t, y, linewidth=0.8, label="$y_t$")
    ax.plot(t, m, color="red", linewidth=1.2, label="$m_t$")
    ax.set_title(nombre)
    ax.legend(loc="upper left")
axes[-1].set_xlabel("t")
plt.tight_layout()
plt.savefig("documento/figuras/p0_media_cambiante.png", dpi=150, bbox_inches="tight")
plt.close()

filas_04 = []
for nombre, m in medias.items():
    rng = np.random.default_rng(semillas_04[nombre])
    r1 = np.empty(R)
    r7 = np.empty(R)
    phi_hat = np.empty(R)

    for i in range(R):
        y = m + rng.normal(0, 1, size=n)
        r = acf_manual(y, 7)
        r1[i] = r[0]
        r7[i] = r[6]
        phi_hat[i] = ols_ar1(y)[1]

    rechazo = (np.abs(r1) > banda).astype(float)

    filas_04.append({
        "proceso": nombre,
        "media_r1": r1.mean(),          "ee_r1": error_estandar(r1),
        "media_r7": r7.mean(),          "ee_r7": error_estandar(r7),
        "tasa_rechazo": rechazo.mean(), "ee_rechazo": error_estandar(rechazo),
        "media_phi_ols": phi_hat.mean(), "ee_phi_ols": error_estandar(phi_hat),
    })

tabla_04 = pd.DataFrame(filas_04)
print(tabla_04.round(4))
tabla_04.to_csv("resultados/p0_tabla_media_cambiante.csv", index=False)
print("Tabla guardada en resultados/p0_tabla_media_cambiante.csv")

# Verificación contra statsmodels
y_prueba2 = m_semanal + np.random.default_rng(3).normal(0, 1, size=n)
Q_mano, p_mano = ljung_box(y_prueba2, 10)
lb_sm = acorr_ljungbox(y_prueba2, lags=[10])
print(f"Ljung-Box a mano:      Q = {Q_mano:.4f}, p = {p_mano:.4f}")
print(f"Ljung-Box statsmodels: Q = {lb_sm['lb_stat'].iloc[0]:.4f}, "
      f"p = {lb_sm['lb_pvalue'].iloc[0]:.4f}")

filas_05 = []
for nombre, m in medias.items():
    rng = np.random.default_rng(semillas_04[nombre])
    rech_r1 = np.empty(R)
    rech_lb6 = np.empty(R)
    rech_lb10 = np.empty(R)

    for i in range(R):
        y = m + rng.normal(0, 1, size=n)
        rech_r1[i] = abs(acf_manual(y, 1)[0]) > banda
        rech_lb6[i] = ljung_box(y, 6)[1] < 0.05
        rech_lb10[i] = ljung_box(y, 10)[1] < 0.05

    filas_05.append({
        "proceso": nombre,
        "rechazo_r1": rech_r1.mean(),    "ee_r1": error_estandar(rech_r1),
        "rechazo_LB6": rech_lb6.mean(),  "ee_LB6": error_estandar(rech_lb6),
        "rechazo_LB10": rech_lb10.mean(), "ee_LB10": error_estandar(rech_lb10),
    })

tabla_05 = pd.DataFrame(filas_05)
print(tabla_05.round(4))
tabla_05.to_csv("resultados/p0_tabla_ljungbox.csv", index=False)
print("Tabla guardada en resultados/p0_tabla_ljungbox.csv")