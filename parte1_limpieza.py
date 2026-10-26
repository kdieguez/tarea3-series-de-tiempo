import pandas as pd
from limpiar import cargar_datos, limpiar

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

df = cargar_datos()

print(df.head())
print()
print(df.dtypes)
print()

n_filas = len(df)
n_facturas = df["InvoiceNo"].nunique()
n_productos = df["StockCode"].nunique()
n_clientes = df["CustomerID"].nunique()       
frac_sin_cliente = df["CustomerID"].isna().mean()

print(f"Filas:      {n_filas:,}")
print(f"Facturas:   {n_facturas:,}")
print(f"Productos:  {n_productos:,}")
print(f"Clientes:   {n_clientes:,}")
print(f"Fracción de filas sin CustomerID: {frac_sin_cliente:.4f}")
print(f"Rango de fechas: {df['InvoiceDate'].min()}  a  {df['InvoiceDate'].max()}")
print()

print("Ejemplos de InvoiceNo que no empiezan con dígito:")
print(df.loc[~df["InvoiceNo"].str[0].str.isdigit(), "InvoiceNo"].head().tolist())
print("Ejemplos de StockCode con letras:")
print(df.loc[df["StockCode"].str.contains(r"[A-Za-z]"), "StockCode"].unique()[:10].tolist())
print("Ejemplos de CustomerID:")
print(df["CustomerID"].dropna().head().tolist())

df["Importe"] = df["Quantity"] * df["UnitPrice"]

dup = df.duplicated(keep="first")       # True para la 2a, 3a... copia
print("=== (a) Duplicados exactos ===")
print(f"Filas que son copia de otra: {dup.sum():,}")
print(f"Importe de esas copias: £{df.loc[dup, 'Importe'].sum():,.2f}")
print("Ejemplo (todas las copias de un caso):")
ejemplo = df[df.duplicated(keep=False)].head(4)
print(ejemplo)
print()

df["Prefijo"] = df["InvoiceNo"].str.extract(r"^(\D*)", expand=False)
df["Prefijo"] = df["Prefijo"].replace("", "(dígito)")
print("=== (b) Prefijos de InvoiceNo ===")
resumen_prefijo = df.groupby("Prefijo").agg(
    filas=("InvoiceNo", "size"),
    facturas=("InvoiceNo", "nunique"),
    importe=("Importe", "sum"),
    qty_negativa=("Quantity", lambda q: (q < 0).sum()),
    qty_positiva=("Quantity", lambda q: (q > 0).sum()),
)
print(resumen_prefijo)
print()

otros = df[~df["Prefijo"].isin(["(dígito)", "C"])]
print("Filas con otros prefijos:")
print(otros[["InvoiceNo", "StockCode", "Description", "Quantity",
             "UnitPrice", "CustomerID", "InvoiceDate"]])

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 40)

raros = df[~df["StockCode"].str.match(r"^\d{5}")]
print("=== (c) StockCodes que no empiezan con 5 dígitos ===")
resumen_codigos = raros.groupby("StockCode").agg(
    descripcion=("Description", lambda d: d.dropna().astype(str).mode().iat[0]
                 if d.notna().any() else ""),
    filas=("StockCode", "size"),
    importe=("Importe", "sum"),
    con_cliente=("CustomerID", lambda c: c.notna().sum()),
).sort_values("filas", ascending=False)
print(resumen_codigos)
print()

print("Códigos distintos:", df["StockCode"].nunique())
print("Códigos distintos al pasar a mayúsculas:", df["StockCode"].str.upper().nunique())
print()

print("=== (d) Precios <= 0 ===")
p0 = df["UnitPrice"] == 0
pneg = df["UnitPrice"] < 0
print(f"Precio = 0: {p0.sum():,} filas "
      f"({df.loc[p0, 'CustomerID'].notna().sum():,} con cliente)")
print(f"Precio < 0: {pneg.sum():,} filas")
print("Signo de la cantidad en filas con precio 0:")
print(pd.Series(df.loc[p0, "Quantity"]).apply(lambda q: "negativa" if q < 0 else "positiva").value_counts())
print("Descripciones más comunes con precio 0 (NaN = vacía):")
print(df.loc[p0, "Description"].value_counts(dropna=False).head(15))
print()
print("Filas con precio 0 Y cliente:")
print(df.loc[p0 & df["CustomerID"].notna(),
             ["InvoiceNo", "StockCode", "Description", "Quantity", "CustomerID"]].head(10))

neg_no_c = (df["Quantity"] < 0) & (df["Prefijo"] == "(dígito)")
print("=== (e) Cantidad negativa sin ser cancelación ===")
print(f"Filas: {neg_no_c.sum():,}")
print(f"De esas, con precio 0: {(neg_no_c & (df['UnitPrice'] == 0)).sum():,}")
print(f"De esas, con cliente: {df.loc[neg_no_c, 'CustomerID'].notna().sum():,}")
print(f"Importe total: £{df.loc[neg_no_c, 'Importe'].sum():,.2f}")
print("Descripciones más comunes:")
print(df.loc[neg_no_c, "Description"].value_counts(dropna=False).head(10))
print()

sin_cli = df["CustomerID"].isna()
venta_total = df["Importe"].sum()
print("=== (f) Filas sin CustomerID ===")
print(f"Filas: {sin_cli.sum():,} ({sin_cli.mean():.1%} de las filas)")
print(f"Importe: £{df.loc[sin_cli, 'Importe'].sum():,.2f} "
      f"({df.loc[sin_cli, 'Importe'].sum() / venta_total:.1%} de la venta neta total)")
print(f"Facturas sin cliente: {df.loc[sin_cli, 'InvoiceNo'].nunique():,}")
print("¿Hay facturas que mezclan filas con y sin cliente?")
mezcla = df.groupby("InvoiceNo")["CustomerID"].agg(lambda c: c.isna().any() and c.notna().any())
print(f"  {mezcla.sum()} facturas")
print("Países más comunes sin cliente:")
print(df.loc[sin_cli, "Country"].value_counts().head(5))
print("Códigos más comunes sin cliente:")
print(df.loc[sin_cli, "StockCode"].value_counts().head(5))

crudo = cargar_datos()      
df_limpio, bitacora = limpiar(crudo, devolver_bitacora=True)

importe_crudo = (crudo["Quantity"] * crudo["UnitPrice"]).sum()
print("=== 1.3 Bitácora de limpieza ===")
print(f"Antes de limpiar: {len(crudo):,} filas, £{importe_crudo:,.2f}")
print(bitacora.to_string(index=False))
print(f"Después de limpiar: {len(df_limpio):,} filas, £{df_limpio['Importe'].sum():,.2f}")

bitacora.to_csv("resultados/p1_bitacora.csv", index=False)

crudo["Importe"] = crudo["Quantity"] * crudo["UnitPrice"]
cols = ["InvoiceNo", "StockCode", "Description", "Quantity",
        "UnitPrice", "Importe", "CustomerID", "InvoiceDate"]

orden = crudo["Importe"].abs().sort_values(ascending=False).index
top6 = crudo.loc[orden[:6]]
print("\n=== 1.4 Las seis líneas con mayor |importe| ===")
print(top6[cols].to_string())

df_limpio["Fecha"] = df_limpio["InvoiceDate"].dt.date

for i, (idx, fila) in enumerate(top6.iterrows(), start=1):
    print(f"\n--- Línea {i}: factura {fila['InvoiceNo']}, "
          f"{fila['Description']}, £{fila['Importe']:,.2f} ---")

    ventana = ((crudo["InvoiceDate"] >= fila["InvoiceDate"] - pd.Timedelta(days=1))
               & (crudo["InvoiceDate"] <= fila["InvoiceDate"] + pd.Timedelta(days=3)))
    if pd.notna(fila["CustomerID"]):
        vecinas = crudo[ventana & (crudo["CustomerID"] == fila["CustomerID"])]
    else:
        vecinas = crudo[ventana & (crudo["StockCode"] == fila["StockCode"])]
    vecinas = vecinas[vecinas["Importe"].abs() >= 100]  
    print(vecinas[cols].to_string())

    sobrevive = ((df_limpio["InvoiceNo"] == fila["InvoiceNo"])
                 & (df_limpio["StockCode"] == str(fila["StockCode"]).upper())
                 & (df_limpio["Quantity"] == fila["Quantity"])).any()
    print(f"¿Sobrevive a limpiar()? {'Sí' if sobrevive else 'No'}")

    if pd.notna(fila["CustomerID"]):
        mismo_dia = ((df_limpio["CustomerID"] == fila["CustomerID"])
                     & (df_limpio["Fecha"] == fila["InvoiceDate"].date()))
        print(f"Neto de ese cliente ese día (datos limpios): "
              f"£{df_limpio.loc[mismo_dia, 'Importe'].sum():,.2f}")