# EDA Automático

App de Streamlit que hace un análisis exploratorio de datos (EDA) automático a partir de un archivo CSV, Excel o SQL/SQLite.

## Qué muestra

- Resumen general (filas, columnas, duplicados, memoria)
- Tipos de datos y valores nulos (tabla + gráfico)
- Estadísticas descriptivas (numéricas y categóricas)
- Distribuciones (histograma, boxplot, frecuencias categóricas)
- Matriz de correlación
- Detección de outliers (método IQR)
- Descarga del dataset en CSV

## Formatos soportados

- `.csv`
- `.xlsx` / `.xls` (con selección de hoja si tiene varias)
- `.db` / `.sqlite` / `.sqlite3` (con selección de tabla)
- `.sql` (dump ejecutado en una base SQLite en memoria, con selección de tabla)

## Cómo correrla en tu máquina

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre en `http://localhost:8501`.

## Cómo desplegarla gratis (Streamlit Community Cloud)

1. Subí estos tres archivos (`app.py`, `requirements.txt`, este `README.md`) a un repo de GitHub.
2. Entrá a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub.
3. "New app" → elegí el repo, la rama y `app.py` como archivo principal.
4. Deploy. En unos minutos tenés una URL pública para mostrar en tu portfolio.

## Ideas para extender (buenas para el portfolio)

- Generar un reporte PDF descargable con los hallazgos del EDA.
- Agregar detección automática de tipo de columna (fecha, id, texto libre) para adaptar el análisis.
- Sumar un modo de comparación entre dos datasets.
- Conectar directamente a una base de datos (Postgres/MySQL) además del archivo `.sql`/`.db`.
