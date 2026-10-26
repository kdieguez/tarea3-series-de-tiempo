import numpy as np
import pandas as pd
from limpiar import cargar_datos, limpiar
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 200)

CORTE = pd.Timestamp("2011-10-01")
FIN = pd.Timestamp("2011-12-08")

df = limpiar(cargar_datos())
cli = df[df["CustomerID"].notna()]
antes = cli[cli["InvoiceDate"] < CORTE]
ventas_antes = antes[~antes["InvoiceNo"].str.startswith("C")]

ultima = ventas_antes.groupby("CustomerID")["InvoiceDate"].max().dt.normalize()
rfm = pd.DataFrame({
    "R": (CORTE - ultima).dt.days,
    "F": ventas_antes.groupby("CustomerID")["InvoiceNo"].nunique(),
})
M_todos = antes.groupby("CustomerID")["Importe"].sum()
solo_canc = M_todos.index.difference(rfm.index)
rfm["M"] = M_todos.loc[rfm.index]

print("=== 6.1 RFM al 1 de octubre de 2011 ===")
print(f"Clientes con alguna transacción antes del corte: {len(M_todos):,}")
print(f"Clientes solo con cancelaciones (sin compras): {len(solo_canc)}")
print(f"Clientes con al menos una compra: {len(rfm):,}")
print()
print(rfm.describe().round(1).to_string())
print()
m_no_pos = rfm["M"] <= 0
print(f"Clientes con M <= 0: {m_no_pos.sum()}")
print(rfm[m_no_pos].sort_values("M").to_string())

rfm.to_csv("resultados/p6_rfm.csv")

print()
print("=== 6.2 Puntajes por quintiles ===")
print(f"Clientes con F = 1: {(rfm['F'] == 1).sum():,} ({(rfm['F'] == 1).mean():.1%})")
print(f"Clientes con F = 2: {(rfm['F'] == 2).sum():,} ({(rfm['F'] == 2).mean():.1%})")
try:
    pd.qcut(rfm["F"], 5, labels=[1, 2, 3, 4, 5])
except ValueError as error:
    print(f"pd.qcut directo sobre F falla: {error}")
print()

rfm = rfm.sort_index()
perm = np.random.default_rng(2026).permutation(len(rfm))
barajado = rfm.iloc[perm]
puntaje_F = pd.qcut(barajado["F"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["pR"] = pd.qcut(rfm["R"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["pF"] = puntaje_F.reindex(rfm.index).astype(int)
rfm["pM"] = pd.qcut(rfm["M"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

for col, var in [("pR", "R"), ("pF", "F"), ("pM", "M")]:
    resumen = rfm.groupby(col)[var].agg(["count", "min", "max"])
    print(f"Puntaje {col} (rango de {var} en cada puntaje):")
    print(resumen.to_string())
    print()

empatados = rfm[rfm["F"] == 2]["pF"].value_counts().sort_index()
print("Clientes con F = 2, según el puntaje que recibieron:")
print(empatados.to_string())

rfm.to_csv("resultados/p6_rfm.csv")

despues = cli[(cli["InvoiceDate"] >= CORTE) & (cli["InvoiceDate"] < FIN + pd.Timedelta(days=1))]
compradores = despues.loc[~despues["InvoiceNo"].str.startswith("C"), "CustomerID"].unique()
rfm["compra"] = rfm.index.isin(compradores).astype(int)
rfm["suma"] = rfm["pR"] + rfm["pF"] + rfm["pM"]

print()
print("=== 6.3 ¿Quién compró del 1 de octubre al 8 de diciembre? ===")
print(f"Clientes que compraron: {rfm['compra'].sum():,} de {len(rfm):,} "
      f"({rfm['compra'].mean():.1%})")
print()

tasas = pd.DataFrame({
    "R": rfm.groupby("pR")["compra"].mean(),
    "F": rfm.groupby("pF")["compra"].mean(),
    "M": rfm.groupby("pM")["compra"].mean(),
})
tasas.index.name = "puntaje"
print("Fracción que compró, por puntaje:")
print(tasas.round(3).to_string())
print()

auc = {
    "R (signo invertido)": roc_auc_score(rfm["compra"], -rfm["R"]),
    "F": roc_auc_score(rfm["compra"], rfm["F"]),
    "M": roc_auc_score(rfm["compra"], rfm["M"]),
    "Suma de puntajes": roc_auc_score(rfm["compra"], rfm["suma"]),
}
print("AUC como clasificadores de compra:")
for nombre, valor in auc.items():
    print(f"  {nombre}: {valor:.4f}")

tasas.to_csv("resultados/p6_tasas_por_puntaje.csv")
rfm.to_csv("resultados/p6_rfm.csv")

venta_post = despues.groupby("CustomerID")["Importe"].sum()
total_post = venta_post.sum()

campeones = rfm[(rfm["pR"] >= 4) & (rfm["pF"] >= 4) & (rfm["pM"] >= 4)].index
venta_campeones = venta_post.reindex(campeones).fillna(0).sum()

nuevos = venta_post.index.difference(rfm.index)
venta_nuevos = venta_post.loc[nuevos].sum()
venta_existentes = total_post - venta_nuevos

print()
print("=== 6.4 y 6.5: venta del 1 de octubre al 8 de diciembre ===")
print(f"Venta neta total (clientes con CustomerID): £{total_post:,.2f}")
print(f"Clientes que compraron o cancelaron en el periodo: {len(venta_post):,}")
print()
print(f"Campeones: {len(campeones):,} de {len(rfm):,} clientes de 6.1 "
      f"({len(campeones) / len(rfm):.1%})")
print(f"  Venta de los campeones: £{venta_campeones:,.2f} "
      f"({venta_campeones / total_post:.1%} del total)")
print(f"  Campeones que compraron en el periodo: "
      f"{rfm.loc[campeones, 'compra'].mean():.1%}")
print()
print(f"Clientes nuevos en el periodo: {len(nuevos):,} "
      f"({len(nuevos) / len(venta_post):.1%} de los clientes del periodo)")
print(f"  Venta de los nuevos: £{venta_nuevos:,.2f} "
      f"({venta_nuevos / total_post:.1%} del total)")
print(f"  Venta de clientes existentes: £{venta_existentes:,.2f} "
      f"({venta_existentes / total_post:.1%} del total)")