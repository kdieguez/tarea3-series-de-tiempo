import os
import pandas as pd

RUTA_XLSX = "datos/Online Retail.xlsx"
RUTA_CACHE = "datos/online_retail.pkl"


def cargar_datos():
    """
    Carga Online Retail. La primera vez lee el .xlsx y guarda una
    copia en .pkl; las siguientes veces lee la copia.
    InvoiceNo, StockCode y CustomerID se leen como texto.
    """
    if os.path.exists(RUTA_CACHE):
        return pd.read_pickle(RUTA_CACHE)

    print("Leyendo el Excel...")
    df = pd.read_excel(
        RUTA_XLSX,
        dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str},
    )
    df["CustomerID"] = df["CustomerID"].str.replace(r"\.0$", "", regex=True)

    df.to_pickle(RUTA_CACHE)
    return df


CODIGOS_NO_PRODUCTO = {
    "POST", "DOT", "C2",
    "AMAZONFEE", "BANK CHARGES", "CRUK",
    "M", "B", "S", "PADS",
}


def limpiar(df, devolver_bitacora=False):
    """
    Limpia Online Retail para construir la venta neta.
    """
    df = df.copy()

    df["StockCode"] = df["StockCode"].str.upper().str.strip()
    df["Importe"] = df["Quantity"] * df["UnitPrice"]

    bitacora = []

    def aplicar(df, eliminar, regla):
        """Elimina las filas marcadas y anota la regla en la bitácora."""
        bitacora.append({
            "orden": len(bitacora) + 1,
            "regla": regla,
            "filas_eliminadas": int(eliminar.sum()),
            "importe_eliminado": df.loc[eliminar, "Importe"].sum(),
        })
        return df.loc[~eliminar]

    df = aplicar(df, df.duplicated(keep="first"), "(a) Duplicados exactos")

    empieza_digito = df["InvoiceNo"].str[0].str.isdigit()
    es_cancelacion = df["InvoiceNo"].str.startswith("C")
    df = aplicar(df, ~empieza_digito & ~es_cancelacion,
                 "(b) Facturas que no son venta ni cancelación")

    fecha = df["InvoiceDate"].dt.date
    es_m_canc = ((df["StockCode"] == "M") & df["InvoiceNo"].str.startswith("C")
                 & df["CustomerID"].notna())
    es_venta = ((df["StockCode"] != "M") & (df["Importe"] > 0)
                & df["CustomerID"].notna())

    canc = pd.DataFrame({
        "idx_c": df.index[es_m_canc],
        "CustomerID": df.loc[es_m_canc, "CustomerID"].values,
        "Fecha": fecha[es_m_canc].values,
        "monto": (-df.loc[es_m_canc, "Importe"]).round(2).values,
    })
    ventas = pd.DataFrame({
        "idx_v": df.index[es_venta],
        "CustomerID": df.loc[es_venta, "CustomerID"].values,
        "Fecha": fecha[es_venta].values,
        "monto": df.loc[es_venta, "Importe"].round(2).values,
    })
    pares = canc.merge(ventas, on=["CustomerID", "Fecha", "monto"])
    pares = pares.drop_duplicates("idx_c").drop_duplicates("idx_v")

    en_par = df.index.isin(pares["idx_c"]) | df.index.isin(pares["idx_v"])
    df = aplicar(df, pd.Series(en_par, index=df.index),
                 "(c1) Pares venta / cancelación manual del mismo monto")

    no_producto = (df["StockCode"].isin(CODIGOS_NO_PRODUCTO)
                   | df["StockCode"].str.startswith("GIFT_"))
    df = aplicar(df, no_producto, "(c2) Resto de códigos que no son productos")

    df = aplicar(df, df["UnitPrice"] <= 0, "(d) Precio cero o negativo")

    neg_no_c = (df["Quantity"] < 0) & ~df["InvoiceNo"].str.startswith("C")
    df = aplicar(df, neg_no_c, "(e) Cantidad negativa fuera de cancelaciones")

    df = df.reset_index(drop=True)
    if devolver_bitacora:
        return df, pd.DataFrame(bitacora)
    return df