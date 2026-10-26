import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from limpiar import cargar_datos, limpiar

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

df = limpiar(cargar_datos())
df["Fecha"] = df["InvoiceDate"].dt.normalize()

venta = df.groupby("Fecha")["Importe"].sum()

calendario = pd.date_range(venta.index.min(), venta.index.max(), freq="D")
serie = venta.reindex(calendario)

print("=== 2.1 Serie diaria ===")
print(f"Primer día: {calendario[0].date()}, último día: {calendario[-1].date()}")
print(f"Días calendario (incluyendo ambos): {len(calendario)}")
print(f"Días con al menos una transacción: {serie.notna().sum()}")
print(f"Días sin transacciones: {serie.isna().sum()}")
print(f"Días con venta neta negativa: {(serie < 0).sum()}")
print()

nombres = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
faltantes = pd.DataFrame({"fecha": serie[serie.isna()].index})
faltantes["dia"] = faltantes["fecha"].dt.day_of_week.map(lambda d: nombres[d])

print("=== 2.2 Días sin transacciones por día de la semana ===")
print(faltantes["dia"].value_counts())
print()

dia_cerrado = faltantes["dia"].mode().iat[0]
print(f"Días sin transacciones que NO son {dia_cerrado}:")
print(faltantes[faltantes["dia"] != dia_cerrado].to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(serie.index, serie.values, linewidth=0.8)
ax.set_title("Venta neta diaria (los huecos son días sin transacciones)")
ax.set_ylabel("Venta neta (£)")
ax.set_xlabel("Fecha")
plt.tight_layout()
plt.savefig("documento/figuras/p2_serie_diaria.png", dpi=150, bbox_inches="tight")
plt.close()

print()
print("=== 2.4 El último día ===")
ultimo = calendario[-1]
dia_ult = df[df["Fecha"] == ultimo]
print(f"Fecha: {ultimo.date()} ({nombres[ultimo.day_of_week]})")
print(f"Primera factura: {dia_ult['InvoiceDate'].min().time()}")
print(f"Última factura:  {dia_ult['InvoiceDate'].max().time()}")
print(f"Facturas: {dia_ult['InvoiceNo'].nunique()}")
print(f"Venta neta: £{dia_ult['Importe'].sum():,.2f}")
print()

viernes = df[df["Fecha"].dt.day_of_week == 4]
resumen_viernes = viernes.groupby("Fecha").agg(
    primera=("InvoiceDate", "min"),
    ultima=("InvoiceDate", "max"),
    facturas=("InvoiceNo", "nunique"),
    venta=("Importe", "sum"),
)
resumen_viernes["primera"] = resumen_viernes["primera"].dt.time
resumen_viernes["ultima"] = resumen_viernes["ultima"].dt.time
print("Últimos viernes:")
print(resumen_viernes.tail(8).to_string())
print()
print("Mediana de los viernes anteriores al último:")
previos = resumen_viernes.iloc[:-1]
print(f"  Facturas: {previos['facturas'].median():.0f}")
print(f"  Venta neta: £{previos['venta'].median():,.2f}")

from series_utils import serie_diaria, FIN_ENTRENAMIENTO, INICIO_PRUEBA, FIN_PRUEBA

s = serie_diaria(df)
entren = s[s.index <= FIN_ENTRENAMIENTO]
prueba = s[(s.index >= INICIO_PRUEBA) & (s.index <= FIN_PRUEBA)]

print()
print("=== Serie final ===")
print(f"Del {s.index[0].date()} al {s.index[-1].date()}: {len(s)} días, "
      f"{s.notna().sum()} abiertos")
print(f"Entrenamiento: {len(entren)} días, {entren.notna().sum()} abiertos")
print(f"Prueba: {len(prueba)} días, {prueba.notna().sum()} abiertos")