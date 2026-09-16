# EDA Automático

App de Streamlit que hace un análisis exploratorio de datos (EDA) automático a partir de un archivo CSV, Excel o SQL/SQLite.

## Qué muestra

La app se organiza en pestañas:

**📋 EDA**
- Resumen general (filas, columnas, duplicados, memoria)
- Tipos de datos y valores nulos (tabla + gráfico)
- Estadísticas descriptivas (numéricas y categóricas)
- Distribuciones (histograma, boxplot, frecuencias categóricas)
- Matriz de correlación
- Outliers univariados (método IQR, columna por columna)
- Descarga del dataset en CSV

**🕵️ Outliers avanzados**
- Detección multivariada con Isolation Forest (considera todas las variables numéricas juntas, no columna por columna)
- Proyección PCA en 2D para visualizar qué filas se marcan como raras
- Control deslizante para ajustar la proporción esperada de outliers

**🤖 Clasificación (ML)**
- Elegís la columna que querés predecir (target)
- Entrena automáticamente 4 modelos: Regresión Logística, Árbol de Decisión, Random Forest y K-Vecinos (KNN)
- Compara accuracy entre modelos y muestra el reporte de clasificación y matriz de confusión del mejor
- Si el mejor modelo es Random Forest, muestra qué variables influyen más en la predicción

**📄 Reporte PDF**
- Genera un PDF descargable con todo lo anterior: resumen, nulos, estadísticas, correlación, outliers (univariados y multivariados) y resultados de clasificación (si se entrenó un modelo)

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

## Ideas para seguir extendiendo (buenas para el portfolio)

- Agregar detección automática de tipo de columna (fecha, id, texto libre) para adaptar el análisis.
- Sumar modelos de regresión (no solo clasificación) cuando el target es numérico continuo.
- Permitir ajustar hiperparámetros de los modelos desde la interfaz (ej. profundidad del árbol, cantidad de árboles).
- Sumar un modo de comparación entre dos datasets.
- Conectar directamente a una base de datos (Postgres/MySQL) además del archivo `.sql`/`.db`.
- Guardar el mejor modelo entrenado (`joblib`) para reutilizarlo sin volver a entrenar.
