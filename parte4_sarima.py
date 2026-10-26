import numpy as np
import pandas as pd
import matplotlib
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
matplotlib.use("Agg")
from limpiar import cargar_datos, limpiar
from series_utils import ols_ar1
from series_utils import (serie_diaria, acf_manual, ljung_box, graficar_acf,
                          FIN_ENTRENAMIENTO)

pd.set_option("display.width", 200)

s = serie_diaria(limpiar(cargar_datos()))
abiertos = s[s.index <= FIN_ENTRENAMIENTO].dropna()
y = abiertos.values
n = len(y)
banda = 1.96 / np.sqrt(n)

nombres = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
dias = abiertos.index.day_of_week
D = pd.get_dummies(dias).astype(float)
D.columns = [nombres[d] for d in D.columns]

beta, *_ = np.linalg.lstsq(D.values, y, rcond=None)
e = y - D.values @ beta

print("=== 4.1 OLS con indicadoras de día de la semana ===")
print("Coeficientes (venta promedio por día):")
print(pd.Series(beta, index=D.columns).round(2).to_string())
print()

r_e = acf_manual(e, 21)
fuera = np.where(np.abs(r_e) > banda)[0] + 1
print("ACF de los residuos e_t:")
print(pd.DataFrame({"k": np.arange(1, 22), "r_k": r_e.round(4)}).to_string(index=False))
print(f"Rezagos fuera de las bandas: {len(fuera)} de 21: {fuera.tolist()}")
for m in (6, 12):
    Q, p = ljung_box(e, m)
    print(f"Ljung-Box Q({m}) = {Q:.2f}, p = {p:.2e}")

graficar_acf(r_e, n, "ACF de los residuos de la regresión con indicadoras de día",
             "documento/figuras/p4_acf_residuos_dias.png")

c_hat, phi_hat = ols_ar1(e)
u = e[1:] - c_hat - phi_hat * e[:-1]
n_u = len(u)
banda_u = 1.96 / np.sqrt(n_u)

print()
print("=== 4.2 AR(1) sobre los residuos ===")
print(f"c_hat = {c_hat:.2f}, phi_hat = {phi_hat:.4f}")

r_u = acf_manual(u, 21)
fuera_u = np.where(np.abs(r_u) > banda_u)[0] + 1
print("ACF de los residuos u_t del AR(1):")
print(pd.DataFrame({"k": np.arange(1, 22), "r_k": r_u.round(4)}).to_string(index=False))
print(f"Rezagos fuera de las bandas: {len(fuera_u)} de 21: {fuera_u.tolist()}")
for m in (6, 12):
    Q, p = ljung_box(u, m, K=1)
    print(f"Ljung-Box Q({m}) = {Q:.2f}, gl = {m - 1}, p = {p:.2e}")

graficar_acf(r_u, n_u, "ACF de los residuos del AR(1)",
             "documento/figuras/p4_acf_residuos_ar1.png")

warnings.simplefilter("ignore")

print()
print("=== 4.3 SARIMA sobre log(venta) ===")
print(f"Mínimo de la venta en días abiertos: £{y.min():,.2f}")
log_y = np.log(y)

configs = {
    "S1": dict(order=(1, 0, 0), seasonal_order=(1, 0, 0, 6), trend="c"),
    "S2": dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, 6), trend="n"),
}

ajustes = {}
for nombre, cfg in configs.items():
    modelo = SARIMAX(log_y, **cfg)
    res_def = modelo.fit(disp=False)
    res_500 = modelo.fit(disp=False, maxiter=500)

    print(f"\n--- {nombre} ---")
    print(f"maxiter por defecto: convergió = {res_def.mle_retvals['converged']}, "
          f"iteraciones = {res_def.mle_retvals.get('iterations')}")
    print(f"maxiter = 500:       convergió = {res_500.mle_retvals['converged']}, "
          f"iteraciones = {res_500.mle_retvals.get('iterations')}")
    comparacion = pd.DataFrame({
        "por_defecto": res_def.params,
        "maxiter_500": res_500.params,
    }, index=res_def.param_names)
    print(comparacion.round(4).to_string())

    ajustes[nombre] = res_def if res_def.mle_retvals["converged"] else res_500

    print()
print("=== 4.4 Diagnóstico de residuos ===")
for nombre, res in ajustes.items():
    resid = res.resid
    if nombre == "S2":
        resid = resid[7:]
    n_r = len(resid)
    r_res = acf_manual(resid, 21)
    fuera_r = np.where(np.abs(r_res) > 1.96 / np.sqrt(n_r))[0] + 1
    print(f"\n--- {nombre} (n = {n_r}) ---")
    print(f"Rezagos fuera de las bandas: {len(fuera_r)} de 21: {fuera_r.tolist()}")
    for m in (6, 12):
        Q, p = ljung_box(resid, m, K=2)
        print(f"Ljung-Box Q({m}) = {Q:.2f}, gl = {m - 2}, p = {p:.4f}")
    graficar_acf(r_res, n_r, f"ACF de los residuos de {nombre}",
                 f"documento/figuras/p4_acf_residuos_{nombre}.png")