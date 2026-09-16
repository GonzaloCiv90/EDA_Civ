"""
EDA Automático - App de Streamlit
Sube un CSV, Excel o base de datos SQLite y obtené un análisis exploratorio
de datos (EDA) automático: shape, tipos, nulos, duplicados, estadísticas
descriptivas, distribuciones, correlaciones y outliers.
"""

import io
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EDA Automático", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes, sheet_name) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


def get_excel_sheets(file_bytes: bytes):
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def get_sqlite_tables(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    conn = sqlite3.connect(tmp_path)
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    )["name"].tolist()
    conn.close()
    return tables, tmp_path


@st.cache_data(show_spinner=False)
def load_sqlite_table(tmp_path: str, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(tmp_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
    conn.close()
    return df


def load_sql_dump(file_bytes: bytes):
    """Ejecuta un dump .sql en una base SQLite en memoria y devuelve las tablas."""
    conn = sqlite3.connect(":memory:")
    script = file_bytes.decode("utf-8", errors="ignore")
    conn.executescript(script)
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    )["name"].tolist()
    return conn, tables


# ---------------------------------------------------------------------------
# Bloques del EDA
# ---------------------------------------------------------------------------

def resumen_general(df: pd.DataFrame):
    st.subheader("📋 Resumen general")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{df.shape[0]:,}")
    c2.metric("Columnas", f"{df.shape[1]:,}")
    c3.metric("Duplicados", f"{df.duplicated().sum():,}")
    c4.metric("Uso de memoria", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    st.markdown("**Primeras filas:**")
    st.dataframe(df.head(10), use_container_width=True)


def tipos_y_nulos(df: pd.DataFrame):
    st.subheader("🧬 Tipos de datos y valores nulos")
    info = pd.DataFrame({
        "Columna": df.columns,
        "Tipo": df.dtypes.astype(str).values,
        "Nulos": df.isnull().sum().values,
        "% Nulos": (df.isnull().mean() * 100).round(2).values,
        "Valores únicos": df.nunique().values,
    })
    st.dataframe(info, use_container_width=True)

    if info["Nulos"].sum() > 0:
        fig = px.bar(
            info[info["Nulos"] > 0],
            x="Columna", y="% Nulos",
            title="Porcentaje de valores nulos por columna",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No hay valores nulos en el dataset.")


def estadisticas_descriptivas(df: pd.DataFrame):
    st.subheader("📈 Estadísticas descriptivas")
    num_df = df.select_dtypes(include=np.number)
    cat_df = df.select_dtypes(exclude=np.number)

    if not num_df.empty:
        st.markdown("**Variables numéricas:**")
        st.dataframe(num_df.describe().T, use_container_width=True)
    if not cat_df.empty:
        st.markdown("**Variables categóricas:**")
        st.dataframe(cat_df.describe().T, use_container_width=True)


def distribuciones(df: pd.DataFrame):
    st.subheader("📊 Distribuciones")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.info("No hay columnas numéricas para graficar.")
        return

    col = st.selectbox("Elegí una columna numérica", num_cols, key="dist_col")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x=col, marginal="box", title=f"Histograma de {col}")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.box(df, y=col, title=f"Boxplot de {col}")
        st.plotly_chart(fig2, use_container_width=True)

    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    if cat_cols:
        cat_col = st.selectbox("Elegí una columna categórica", cat_cols, key="cat_col")
        counts = df[cat_col].value_counts().head(20).reset_index()
        counts.columns = [cat_col, "Frecuencia"]
        fig3 = px.bar(counts, x=cat_col, y="Frecuencia", title=f"Frecuencias de {cat_col}")
        st.plotly_chart(fig3, use_container_width=True)


def correlaciones(df: pd.DataFrame):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return
    st.subheader("🔗 Correlaciones")
    corr = num_df.corr(numeric_only=True)
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Matriz de correlación",
    )
    st.plotly_chart(fig, use_container_width=True)


def outliers(df: pd.DataFrame):
    num_df = df.select_dtypes(include=np.number)
    if num_df.empty:
        return
    st.subheader("⚠️ Detección de outliers (método IQR)")
    rows = []
    for col in num_df.columns:
        q1, q3 = num_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((num_df[col] < lower) | (num_df[col] > upper)).sum()
        rows.append({"Columna": col, "Outliers": n_out, "% del total": round(n_out / len(df) * 100, 2)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def run_eda(df: pd.DataFrame):
    resumen_general(df)
    st.divider()
    tipos_y_nulos(df)
    st.divider()
    estadisticas_descriptivas(df)
    st.divider()
    distribuciones(df)
    st.divider()
    correlaciones(df)
    st.divider()
    outliers(df)

    st.divider()
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar dataset como CSV", csv, "dataset.csv", "text/csv")


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------

st.title("📊 EDA Automático")
st.caption("Subí un archivo CSV, Excel o SQL/SQLite y obtené un análisis exploratorio de datos al instante.")

uploaded_file = st.file_uploader(
    "Subí tu archivo",
    type=["csv", "xlsx", "xls", "db", "sqlite", "sqlite3", "sql"],
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    ext = uploaded_file.name.split(".")[-1].lower()

    df = None

    try:
        if ext == "csv":
            df = load_csv(file_bytes)

        elif ext in ("xlsx", "xls"):
            sheets = get_excel_sheets(file_bytes)
            sheet = st.selectbox("Elegí la hoja", sheets) if len(sheets) > 1 else sheets[0]
            df = load_excel(file_bytes, sheet)

        elif ext in ("db", "sqlite", "sqlite3"):
            tables, tmp_path = get_sqlite_tables(file_bytes)
            if not tables:
                st.warning("No se encontraron tablas en la base de datos.")
            else:
                table = st.selectbox("Elegí la tabla", tables)
                df = load_sqlite_table(tmp_path, table)

        elif ext == "sql":
            conn, tables = load_sql_dump(file_bytes)
            if not tables:
                st.warning("No se encontraron tablas en el dump SQL.")
            else:
                table = st.selectbox("Elegí la tabla", tables)
                df = pd.read_sql_query(f"SELECT * FROM {table};", conn)

    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")

    if df is not None and not df.empty:
        run_eda(df)
    elif df is not None:
        st.warning("El archivo se cargó pero no tiene datos.")

else:
    st.info("Esperando que subas un archivo para comenzar el análisis.")
