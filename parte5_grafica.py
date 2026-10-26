import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pron = pd.read_csv("resultados/p5_pronosticos_prueba.csv", index_col=0, parse_dates=True)

grupos = [["S2", "B2", "B1"], ["M1", "M2", "S1"]]
colores = {"S2": "tab:green", "B2": "tab:orange", "B1": "tab:gray",
           "M1": "tab:purple", "M2": "tab:brown", "S1": "tab:red"}

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
for ax, grupo in zip(axes, grupos):
    ax.plot(pron.index, pron["real"], color="black", linewidth=1.8,
            marker="o", markersize=3, label="Real")
    for m in grupo:
        ax.plot(pron.index, pron[m], color=colores[m], linewidth=1.2,
                marker=".", markersize=4, label=m)
    ax.set_ylabel("Venta neta (£)")
    ax.legend(loc="upper left", ncol=4)
axes[0].set_title("Pronósticos de origen móvil semanal en el periodo de prueba")
axes[-1].set_xlabel("Fecha")
plt.tight_layout()
plt.savefig("documento/figuras/p5_pronosticos_prueba.png", dpi=150, bbox_inches="tight")
plt.close()

print("Error medio (real - pronóstico) por modelo:")
print((pron["real"].values[:, None] - pron.drop(columns="real")).mean().round(0).to_string())