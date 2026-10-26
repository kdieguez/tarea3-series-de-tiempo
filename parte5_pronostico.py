import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from limpiar import cargar_datos, limpiar
from series_utils import serie_diaria, FIN_ENTRENAMIENTO, ORIGEN_PRUEBA, FIN_PRUEBA
from sklearn.model_selection import KFold

warnings.simplefilter("ignore")
pd.set_option("display.width", 200)

s = serie_diaria(limpiar(cargar_datos()))
abiertos = s.dropna()
FECHA0 = s.index[0]
DIAS = [0, 1, 2, 3, 4, 6]


def matriz_dias(fechas):
    dow = fechas.day_of_week
    return np.column_stack([(dow == d).astype(float) for d in DIAS])


def rezago_semanal(fecha, k):
    d = fecha - pd.Timedelta(days=k)
    while d >= FECHA0:
        if d in abiertos.index:
            return abiertos[d]
        d -= pd.Timedelta(days=7)
    return np.nan


rezagos = pd.DataFrame({
    "lag7": [rezago_semanal(f, 7) for f in abiertos.index],
    "lag14": [rezago_semanal(f, 14) for f in abiertos.index],
}, index=abiertos.index)


def b1(historia, objetivos):
    return np.full(len(objetivos), historia.mean())


def b2(historia, objetivos):
    pred = []
    for f in objetivos:
        mismo_dia = historia[historia.index.day_of_week == f.day_of_week]
        pred.append(mismo_dia.iloc[-1])
    return np.array(pred)


def m1(historia, objetivos):
    X = np.column_stack([(historia.index - FECHA0).days, matriz_dias(historia.index)])
    beta, *_ = np.linalg.lstsq(X, historia.values, rcond=None)
    Xo = np.column_stack([(objetivos - FECHA0).days, matriz_dias(objetivos)])
    return Xo @ beta


def m2(historia, objetivos):
    R = rezagos.loc[historia.index]
    ok = R.notna().all(axis=1).values
    X = np.column_stack([matriz_dias(historia.index), R.values])[ok]
    beta, *_ = np.linalg.lstsq(X, historia.values[ok], rcond=None)
    Xo = np.column_stack([matriz_dias(objetivos), rezagos.loc[objetivos].values])
    return Xo @ beta


CONFIG_SARIMA = {
    "S1": dict(orden=dict(order=(1, 0, 0), seasonal_order=(1, 0, 0, 6), trend="c"),
               maxiter=500),
    "S2": dict(orden=dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, 6), trend="n"),
               maxiter=50),
}
convergencia = {"S1": [], "S2": []}


def sarima(nombre):
    cfg = CONFIG_SARIMA[nombre]

    def modelo(historia, objetivos):
        res = SARIMAX(np.log(historia.values), **cfg["orden"]).fit(
            disp=False, maxiter=cfg["maxiter"])
        convergencia[nombre].append(res.mle_retvals["converged"])
        return np.exp(res.forecast(steps=len(objetivos)))

    return modelo


def origen_movil(modelo, origenes, fin_eval):
    partes = []
    for o in origenes:
        historia = abiertos[abiertos.index <= o]
        objetivos = abiertos.index[(abiertos.index > o)
                                   & (abiertos.index <= o + pd.Timedelta(days=7))
                                   & (abiertos.index <= fin_eval)]
        if len(objetivos) == 0:
            continue
        partes.append(pd.DataFrame({
            "origen": o,
            "real": abiertos.loc[objetivos].values,
            "pronostico": modelo(historia, objetivos),
        }, index=objetivos))
    pron = pd.concat(partes)
    err = pron["real"] - pron["pronostico"]
    return np.mean(np.abs(err)), np.sqrt(np.mean(err ** 2)), len(err), pron


origenes_b = pd.date_range("2011-05-01", "2011-09-30", freq="W-SUN")
origenes_c = pd.date_range(ORIGEN_PRUEBA, FIN_PRUEBA, freq="W-SUN")

print("=== Orígenes ===")
print(f"(b): {len(origenes_b)} domingos, del {origenes_b[0].date()} al {origenes_b[-1].date()}")
print(f"(c): {len(origenes_c)} domingos, del {origenes_c[0].date()} al {origenes_c[-1].date()}")
print()

modelos = {"B1": b1, "B2": b2, "M1": m1, "M2": m2,
           "S1": sarima("S1"), "S2": sarima("S2")}

filas = []
pronosticos_prueba = {}
for nombre, modelo in modelos.items():
    mae_b, rmse_b, n_b, _ = origen_movil(modelo, origenes_b, FIN_ENTRENAMIENTO)
    mae_c, rmse_c, n_c, pron_c = origen_movil(modelo, origenes_c, FIN_PRUEBA)
    pronosticos_prueba[nombre] = pron_c["pronostico"]
    filas.append({"modelo": nombre,
                  "MAE_b": mae_b, "RMSE_b": rmse_b, "n_b": n_b,
                  "MAE_c": mae_c, "RMSE_c": rmse_c, "n_c": n_c})
    print(f"{nombre} listo")

tabla = pd.DataFrame(filas)
print()
print("=== Origen móvil: todos los modelos ===")
print(tabla.round(0).to_string(index=False))

print()
print("=== Convergencia de los SARIMA en cada origen ===")
for nombre, lista in convergencia.items():
    print(f"{nombre}: {sum(lista)} de {len(lista)} ajustes convergieron")

tabla.to_csv("resultados/p5_origen_movil.csv", index=False)
pron_df = pd.DataFrame(pronosticos_prueba)
pron_df.insert(0, "real", abiertos.loc[pron_df.index])
pron_df.to_csv("resultados/p5_pronosticos_prueba.csv")

fechas_cv = rezagos[rezagos.index <= FIN_ENTRENAMIENTO].dropna().index
semillas_cv = list(range(10))


def cv_aleatoria(modelo, fechas, semillas):
    maes, rmses = [], []
    for semilla in semillas:
        pred = np.empty(len(fechas))
        for idx_ent, idx_val in KFold(n_splits=5, shuffle=True,
                                       random_state=semilla).split(fechas):
            historia = abiertos.loc[fechas[idx_ent]]
            pred[idx_val] = modelo(historia, fechas[idx_val])
        err = abiertos.loc[fechas].values - pred
        maes.append(np.mean(np.abs(err)))
        rmses.append(np.sqrt(np.mean(err ** 2)))
    return np.mean(maes), np.std(maes, ddof=1), np.mean(rmses), np.std(rmses, ddof=1)


print()
print(f"=== (a) Validación cruzada aleatoria: {len(fechas_cv)} días, "
      f"5 folds, semillas {semillas_cv[0]} a {semillas_cv[-1]} ===")
filas_a = []
for nombre in ["B1", "M1", "M2"]:
    mae_m, mae_s, rmse_m, rmse_s = cv_aleatoria(modelos[nombre], fechas_cv, semillas_cv)
    filas_a.append({"modelo": nombre, "MAE_a": mae_m, "MAE_a_de": mae_s,
                    "RMSE_a": rmse_m, "RMSE_a_de": rmse_s})

tabla_completa = pd.DataFrame(filas_a).merge(tabla, on="modelo", how="right")
columnas = ["modelo", "MAE_a", "MAE_a_de", "RMSE_a", "RMSE_a_de",
            "MAE_b", "RMSE_b", "MAE_c", "RMSE_c"]
print(tabla_completa[columnas].round(0).to_string(index=False))
tabla_completa.to_csv("resultados/p5_tabla_completa.csv", index=False)

ma7_total = s.fillna(0).rolling(7, center=True).mean()
rezagos_trampa = rezagos.copy()
rezagos_trampa["ma7"] = ma7_total.loc[rezagos.index]


def m2_trampa(historia, objetivos):
    R = rezagos_trampa.loc[historia.index]
    ok = R.notna().all(axis=1).values
    X = np.column_stack([matriz_dias(historia.index), R.values])[ok]
    beta, *_ = np.linalg.lstsq(X, historia.values[ok], rcond=None)
    Xo = np.column_stack([matriz_dias(objetivos), rezagos_trampa.loc[objetivos].values])
    return Xo @ beta


print()
print("=== 5.4 La trampa: M2 + media móvil centrada ===")
filas_t = []
for nombre, modelo in [("M2", m2), ("M2 + MA7", m2_trampa)]:
    mae_a, mae_a_de, rmse_a, rmse_a_de = cv_aleatoria(modelo, fechas_cv, semillas_cv)
    mae_b, rmse_b, _, _ = origen_movil(modelo, origenes_b, FIN_ENTRENAMIENTO)
    filas_t.append({"modelo": nombre, "MAE_a": mae_a, "MAE_a_de": mae_a_de,
                    "RMSE_a": rmse_a, "MAE_b": mae_b, "RMSE_b": rmse_b})
tabla_trampa = pd.DataFrame(filas_t)
print(tabla_trampa.round(0).to_string(index=False))
tabla_trampa.to_csv("resultados/p5_trampa.csv", index=False)