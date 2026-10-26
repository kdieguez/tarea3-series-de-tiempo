# Tarea: Introducción a Series de Tiempo

Análisis y Ciencia de Datos, Facultad de Ingeniería, Universidad del Istmo.

## Requisitos

Python 3.11 o superior. Desde la raíz del repositorio:

```bash
python -m venv .venv
.venv\Scripts\activate          # En Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

Descargar `Online Retail.xlsx` de
https://archive.ics.uci.edu/dataset/352/online+retail
y guardarlo en `datos/`. La primera ejecución lee el Excel y crea una copia
`datos/online_retail.pkl` para acelerar las siguientes.

## Ejecución

Correr los scripts desde la raíz del repositorio, en este orden:

```bash
python parte0_simulacion.py
python parte1_limpieza.py
python parte2_calendario.py
python parte3_tendencia.py
python parte4_sarima.py
python parte5_pronostico.py
python parte5_grafica.py
python parte6_rfm.py
```

Las figuras se guardan en `documento/figuras/` y las tablas en `resultados/`.

## Estructura

- `limpiar.py`: carga de datos (`cargar_datos`) y limpieza (`limpiar`).
- `series_utils.py`: ACF, Ljung-Box, AR(1) por OLS, serie diaria y fechas de entrenamiento y prueba.
- `parteN_*.py`: código de cada parte de la tarea.
- `documento/`: documento en LaTeX (`main.tex`, `main.pdf`) y figuras.

## Semillas

- Parte 0: 2026 (figura AR(1)), 20260 a 20263 (réplicas AR(1)), 2027 (figura de medias), 20264 a 20266 (réplicas de 0.4 y 0.5), 1 y 3 (verificaciones).
- Parte 5: semillas 0 a 9 para la validación cruzada aleatoria.
- Parte 6: `numpy.random.default_rng(2026)` para el desempate de frecuencia.