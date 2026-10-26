import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from limpiar import cargar_datos, limpiar
from series_utils import serie_diaria
from series_utils import acf_manual, ljung_box

pd.set_option("display.width", 200)

s = serie_diaria(limpiar(cargar_datos()))
s0 = s.fillna(0)
ma7 = s0.rolling(7, center=True).mean()

print("=== 3.1 Media móvil centrada de 7 días ===")
print(f"Días sin media móvil (extremos): {ma7.isna().sum()}")
print()
print("Venta total y media móvil promedio por mes:")
resumen_mes = pd.DataFrame({
    "venta_total": s0.resample("MS").sum(),
    "ma7_promedio": ma7.resample("MS").mean(),
})
print(resumen_mes.round(0).to_string())

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(s.index, s.values, linewidth=0.7, color="lightsteelblue", label="Venta neta diaria")
ax.plot(ma7.index, ma7.values, linewidth=1.8, color="darkblue", label="Media móvil centrada de 7 días")
ax.set_title("Venta neta diaria y su tendencia")
ax.set_ylabel("Venta neta (£)")
ax.set_xlabel("Fecha")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("documento/figuras/p3_tendencia.png", dpi=150, bbox_inches="tight")
plt.close()

from series_utils import FIN_ENTRENAMIENTO

nombres = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

s0_entren = s0[s0.index <= FIN_ENTRENAMIENTO]
s_entren = s[s.index <= FIN_ENTRENAMIENTO]
ma7_entren = s0_entren.rolling(7, center=True).mean()

cociente = (s_entren / ma7_entren).dropna()
indices = cociente.groupby(cociente.index.day_of_week).agg(["mean", "count"])
indices.index = [nombres[d] for d in indices.index]
indices.columns = ["indice", "n_dias"]

print()
print("=== 3.2 Índices estacionales (entrenamiento) ===")
print(indices.round(4).to_string())
print(f"Promedio de los índices: {indices['indice'].mean():.4f}")
indices.to_csv("resultados/p3_indices_estacionales.csv")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(indices.index, indices["indice"], color="steelblue")
ax.axhline(1, color="gray", linestyle="--", linewidth=1)
ax.set_title("Índice estacional por día de la semana (entrenamiento)")
ax.set_ylabel("Índice ($y_t / MA_t$ promedio)")
plt.tight_layout()
plt.savefig("documento/figuras/p3_indices.png", dpi=150, bbox_inches="tight")
plt.close()

abiertos = s_entren.dropna()
y = abiertos.values
n = len(y)
kmax = 21
r = acf_manual(y, kmax)
banda = 1.96 / np.sqrt(n)
fuera = np.where(np.abs(r) > banda)[0] + 1
pico = int(np.argmax(r)) + 1

dist_pico = pd.Series((abiertos.index[pico:] - abiertos.index[:-pico]).days)

print()
print("=== 3.3 Autocorrelación (días abiertos del entrenamiento) ===")
print(f"n = {n}, banda = ±{banda:.4f}")
print(pd.DataFrame({"k": np.arange(1, kmax + 1), "r_k": r.round(4)}).to_string(index=False))
print(f"Rezagos fuera de las bandas: {len(fuera)} de {kmax}: {fuera.tolist()}")
print(f"r_1 = {r[0]:.4f} -> ¿pasa la prueba de retardo 1? {'Sí' if abs(r[0]) <= banda else 'No'}")
print(f"Pico en el rezago {pico} (r = {r[pico - 1]:.4f})")
print("Días calendario que separan las filas a ese rezago:")
print(dist_pico.value_counts().sort_index().to_string())

for m in (6, 12):
    Q, p = ljung_box(y, m)
    print(f"Ljung-Box Q({m}) = {Q:.2f}, p = {p:.2e}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.vlines(np.arange(1, kmax + 1), 0, r, color="steelblue", linewidth=2)
ax.scatter(np.arange(1, kmax + 1), r, color="steelblue", zorder=3)
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(banda, color="red", linestyle="--", linewidth=1)
ax.axhline(-banda, color="red", linestyle="--", linewidth=1)
ax.set_xticks(np.arange(1, kmax + 1))
ax.set_xlabel("Rezago k (días abiertos)")
ax.set_ylabel("$r_k$")
ax.set_title("ACF de la venta neta diaria (días abiertos del entrenamiento)")
plt.tight_layout()
plt.savefig("documento/figuras/p3_acf.png", dpi=150, bbox_inches="tight")
plt.close()