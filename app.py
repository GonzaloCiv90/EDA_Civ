"""
EDA Automático - App de Streamlit
Sube un CSV, Excel o base de datos SQLite y obtené:
- Un análisis exploratorio de datos (EDA) automático
- Detección de outliers (univariado IQR y multivariado con Isolation Forest)
- Modelos de Machine Learning para clasificar los datos
- Un reporte en PDF con todo lo anterior
"""

import io
import sqlite3
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

st.set_page_config(page_title="EDA Automático", page_icon="📊", layout="wide")

if "resultados" not in st.session_state:
    st.session_state.resultados = {}


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
    conn = sqlite3.connect(":memory:")
    script = file_bytes.decode("utf-8", errors="ignore")
    conn.executescript(script)
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    )["name"].tolist()
    return conn, tables


# ---------------------------------------------------------------------------
# EDA básico
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


def tipos_y_nulos(df: pd.DataFrame) -> pd.DataFrame:
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
        fig = px.bar(info[info["Nulos"] > 0], x="Columna", y="% Nulos",
                     title="Porcentaje de valores nulos por columna")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No hay valores nulos en el dataset.")
    return info


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
    return num_df.describe().T if not num_df.empty else pd.DataFrame()


def distribuciones(df: pd.DataFrame):
    st.subheader("📊 Distribuciones")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.info("No hay columnas numéricas para graficar.")
        return
    col = st.selectbox("Elegí una columna numérica", num_cols, key="dist_col")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(df, x=col, marginal="box", title=f"Histograma de {col}"),
                         use_container_width=True)
    with c2:
        st.plotly_chart(px.box(df, y=col, title=f"Boxplot de {col}"), use_container_width=True)
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    if cat_cols:
        cat_col = st.selectbox("Elegí una columna categórica", cat_cols, key="cat_col")
        counts = df[cat_col].value_counts().head(20).reset_index()
        counts.columns = [cat_col, "Frecuencia"]
        st.plotly_chart(px.bar(counts, x=cat_col, y="Frecuencia", title=f"Frecuencias de {cat_col}"),
                         use_container_width=True)


def correlaciones(df: pd.DataFrame):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return None
    st.subheader("🔗 Correlaciones")
    corr = num_df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, title="Matriz de correlación")
    st.plotly_chart(fig, use_container_width=True)
    return corr


def outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    num_df = df.select_dtypes(include=np.number)
    if num_df.empty:
        return pd.DataFrame()
    st.subheader("⚠️ Outliers univariados (método IQR)")
    rows = []
    for col in num_df.columns:
        q1, q3 = num_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((num_df[col] < lower) | (num_df[col] > upper)).sum()
        rows.append({"Columna": col, "Outliers": n_out, "% del total": round(n_out / len(df) * 100, 2)})
    tabla = pd.DataFrame(rows)
    st.dataframe(tabla, use_container_width=True)
    return tabla


# ---------------------------------------------------------------------------
# Outliers multivariados (Isolation Forest)
# ---------------------------------------------------------------------------

def outliers_multivariados(df: pd.DataFrame):
    num_df = df.select_dtypes(include=np.number).dropna()
    if num_df.shape[1] < 2 or num_df.shape[0] < 10:
        st.info("Se necesitan al menos 2 columnas numéricas y 10 filas sin nulos para este análisis.")
        return None

    st.subheader("🕵️ Outliers multivariados (Isolation Forest)")
    st.caption("Detecta filas 'raras' considerando todas las variables numéricas a la vez, no columna por columna.")

    contamination = st.slider("Proporción esperada de outliers", 0.01, 0.25, 0.05, 0.01)

    scaler = StandardScaler()
    X = scaler.fit_transform(num_df)

    iso = IsolationForest(contamination=contamination, random_state=42)
    pred = iso.fit_predict(X)  # -1 = outlier, 1 = normal
    is_outlier = pred == -1

    n_out = is_outlier.sum()
    st.metric("Filas detectadas como outlier", f"{n_out} ({n_out / len(num_df) * 100:.1f}%)")

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    plot_df["Estado"] = np.where(is_outlier, "Outlier", "Normal")

    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Estado",
                      color_discrete_map={"Normal": "#4C78A8", "Outlier": "#E45756"},
                      title="Proyección PCA (2D) — outliers en rojo")
    st.plotly_chart(fig, use_container_width=True)

    filas_outlier = num_df[is_outlier]
    with st.expander("Ver filas marcadas como outlier"):
        st.dataframe(filas_outlier, use_container_width=True)

    return {"n_outliers": int(n_out), "pct": round(n_out / len(num_df) * 100, 2), "plot_df": plot_df}


# ---------------------------------------------------------------------------
# Clasificación (ML)
# ---------------------------------------------------------------------------

def preparar_datos_clasificacion(df: pd.DataFrame, target: str):
    data = df.dropna(subset=[target]).copy()
    y_raw = data[target]
    X = data.drop(columns=[target])

    # Codificar target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw.astype(str))

    # Separar tipos de columnas
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    if num_cols:
        X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
    if cat_cols:
        X[cat_cols] = X[cat_cols].fillna("desconocido")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y, le_target


def clasificacion(df: pd.DataFrame):
    st.subheader("🤖 Clasificación con Machine Learning")
    st.caption("Elegí la columna que querés predecir (la variable objetivo). "
               "Funciona mejor si tiene pocas categorías distintas.")

    candidatas = [c for c in df.columns if df[c].nunique() <= max(20, int(len(df) * 0.5)) and df[c].nunique() >= 2]
    if not candidatas:
        st.info("No se encontró ninguna columna con una cantidad de categorías razonable para clasificar.")
        return None

    target = st.selectbox("Variable objetivo (target)", candidatas)
    n_clases = df[target].nunique()
    st.caption(f"'{target}' tiene {n_clases} categorías distintas.")

    if st.button("Entrenar modelos de clasificación"):
        with st.spinner("Entrenando modelos..."):
            X, y, le_target = preparar_datos_clasificacion(df, target)

            if len(X) < 20:
                st.warning("Muy pocas filas utilizables para entrenar (mínimo recomendado: 20).")
                return None

            stratify = y if min(np.bincount(y)) >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=stratify
            )

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            modelos = {
                "Regresión Logística": LogisticRegression(max_iter=1000),
                "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
                "K-Vecinos (KNN)": KNeighborsClassifier(),
            }

            resultados_modelos = []
            mejores = {}
            for nombre, modelo in modelos.items():
                modelo.fit(X_train_s, y_train)
                pred = modelo.predict(X_test_s)
                acc = accuracy_score(y_test, pred)
                resultados_modelos.append({"Modelo": nombre, "Accuracy": round(acc, 4)})
                mejores[nombre] = (modelo, pred, acc)

            tabla_resultados = pd.DataFrame(resultados_modelos).sort_values("Accuracy", ascending=False)
            st.markdown("**Comparación de modelos:**")
            st.dataframe(tabla_resultados, use_container_width=True)
            st.plotly_chart(px.bar(tabla_resultados, x="Modelo", y="Accuracy",
                                    title="Accuracy por modelo", range_y=[0, 1]),
                             use_container_width=True)

            mejor_nombre = tabla_resultados.iloc[0]["Modelo"]
            mejor_modelo, mejor_pred, mejor_acc = mejores[mejor_nombre]
            st.success(f"Mejor modelo: **{mejor_nombre}** (accuracy: {mejor_acc:.2%})")

            clases_nombres = le_target.classes_.astype(str)
            reporte = classification_report(y_test, mejor_pred, target_names=clases_nombres,
                                             output_dict=True, zero_division=0)
            st.markdown("**Reporte de clasificación (mejor modelo):**")
            st.dataframe(pd.DataFrame(reporte).T.round(3), use_container_width=True)

            cm = confusion_matrix(y_test, mejor_pred)
            fig_cm = px.imshow(cm, text_auto=True, x=clases_nombres, y=clases_nombres,
                                labels=dict(x="Predicho", y="Real", color="Cantidad"),
                                title=f"Matriz de confusión — {mejor_nombre}")
            st.plotly_chart(fig_cm, use_container_width=True)

            importancias = None
            if mejor_nombre == "Random Forest":
                importancias = pd.Series(mejor_modelo.feature_importances_, index=X.columns) \
                    .sort_values(ascending=False).head(15)
                st.markdown("**Variables más importantes:**")
                st.plotly_chart(px.bar(importancias, orientation="h", title="Importancia de variables"),
                                 use_container_width=True)

            resultado = {
                "target": target,
                "tabla_resultados": tabla_resultados,
                "mejor_nombre": mejor_nombre,
                "mejor_acc": mejor_acc,
                "reporte": pd.DataFrame(reporte).T.round(3),
                "matriz_confusion": cm,
                "clases": clases_nombres,
                "importancias": importancias,
            }
            st.session_state.resultados["clasificacion"] = resultado
            return resultado
    return st.session_state.resultados.get("clasificacion")


# ---------------------------------------------------------------------------
# Reporte PDF
# ---------------------------------------------------------------------------

def _df_a_tabla_pdf(df: pd.DataFrame, max_filas: int = 15) -> Table:
    data = [list(df.reset_index().columns)] + df.reset_index().head(max_filas).round(3).astype(str).values.tolist()
    tabla = Table(data, repeatRows=1)
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C78A8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
    ]))
    return tabla


def _fig_a_imagen(fig, ancho_cm=16):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=ancho_cm * cm, height=ancho_cm * cm * 0.6)


def generar_pdf(df, info_nulos, describe_num, corr, iqr_tabla, multi_out, clasif) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Reporte de Análisis Exploratorio de Datos (EDA)", styles["Title"]))
    story.append(Paragraph(datetime.now().strftime("Generado el %d/%m/%Y a las %H:%M"), styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Resumen general", styles["Heading2"]))
    resumen_txt = (f"Filas: {df.shape[0]:,} &nbsp;&nbsp; Columnas: {df.shape[1]:,} &nbsp;&nbsp; "
                    f"Duplicados: {df.duplicated().sum():,}")
    story.append(Paragraph(resumen_txt, styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Tipos de datos y valores nulos", styles["Heading2"]))
    story.append(_df_a_tabla_pdf(info_nulos.set_index("Columna")))
    story.append(Spacer(1, 10))

    if not describe_num.empty:
        story.append(Paragraph("3. Estadísticas descriptivas (numéricas)", styles["Heading2"]))
        story.append(_df_a_tabla_pdf(describe_num))
        story.append(Spacer(1, 10))

    if corr is not None:
        story.append(Paragraph("4. Matriz de correlación", styles["Heading2"]))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns, fontsize=6)
        fig.colorbar(im)
        story.append(_fig_a_imagen(fig, ancho_cm=12))
        story.append(Spacer(1, 10))

    story.append(PageBreak())
    story.append(Paragraph("5. Outliers univariados (IQR)", styles["Heading2"]))
    if not iqr_tabla.empty:
        story.append(_df_a_tabla_pdf(iqr_tabla.set_index("Columna")))
    else:
        story.append(Paragraph("No hay columnas numéricas para evaluar.", styles["Normal"]))
    story.append(Spacer(1, 10))

    if multi_out is not None:
        story.append(Paragraph("6. Outliers multivariados (Isolation Forest)", styles["Heading2"]))
        story.append(Paragraph(
            f"Filas detectadas como outlier: {multi_out['n_outliers']} ({multi_out['pct']}%)",
            styles["Normal"]))
        pdf_df = multi_out["plot_df"]
        fig, ax = plt.subplots(figsize=(6, 5))
        for estado, color in [("Normal", "#4C78A8"), ("Outlier", "#E45756")]:
            sub = pdf_df[pdf_df["Estado"] == estado]
            ax.scatter(sub["PC1"], sub["PC2"], s=10, c=color, label=estado)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.set_title("Proyección PCA — outliers")
        story.append(_fig_a_imagen(fig, ancho_cm=12))
        story.append(Spacer(1, 10))

    if clasif is not None:
        story.append(PageBreak())
        story.append(Paragraph("7. Clasificación con Machine Learning", styles["Heading2"]))
        story.append(Paragraph(f"Variable objetivo: <b>{clasif['target']}</b>", styles["Normal"]))
        story.append(Spacer(1, 6))
        story.append(_df_a_tabla_pdf(clasif["tabla_resultados"].set_index("Modelo")))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"Mejor modelo: <b>{clasif['mejor_nombre']}</b> — Accuracy: {clasif['mejor_acc']:.2%}",
            styles["Normal"]))
        story.append(Spacer(1, 8))

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(clasif["matriz_confusion"], cmap="Blues")
        ax.set_xticks(range(len(clasif["clases"])))
        ax.set_xticklabels(clasif["clases"], rotation=45, fontsize=7)
        ax.set_yticks(range(len(clasif["clases"])))
        ax.set_yticklabels(clasif["clases"], fontsize=7)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        for i in range(clasif["matriz_confusion"].shape[0]):
            for j in range(clasif["matriz_confusion"].shape[1]):
                ax.text(j, i, clasif["matriz_confusion"][i, j], ha="center", va="center", fontsize=7)
        ax.set_title("Matriz de confusión")
        story.append(_fig_a_imagen(fig, ancho_cm=10))

        if clasif.get("importancias") is not None:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Variables más importantes (Random Forest):", styles["Normal"]))
            fig, ax = plt.subplots(figsize=(6, 4))
            clasif["importancias"].sort_values().plot(kind="barh", ax=ax)
            ax.set_title("Importancia de variables")
            story.append(_fig_a_imagen(fig, ancho_cm=12))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------

st.title("📊 EDA Automático")
st.caption("Subí un archivo CSV, Excel o SQL/SQLite y obtené EDA, outliers, "
           "clasificación con Machine Learning y un reporte en PDF.")

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
        tab_eda, tab_outliers, tab_ml, tab_pdf = st.tabs(
            ["📋 EDA", "🕵️ Outliers avanzados", "🤖 Clasificación (ML)", "📄 Reporte PDF"]
        )

        with tab_eda:
            resumen_general(df)
            st.divider()
            info_nulos = tipos_y_nulos(df)
            st.divider()
            describe_num = estadisticas_descriptivas(df)
            st.divider()
            distribuciones(df)
            st.divider()
            corr = correlaciones(df)
            st.divider()
            iqr_tabla = outliers_iqr(df)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar dataset como CSV", csv, "dataset.csv", "text/csv")

        with tab_outliers:
            multi_out = outliers_multivariados(df)

        with tab_ml:
            clasif = clasificacion(df)

        with tab_pdf:
            st.subheader("📄 Generar reporte PDF")
            st.caption("Incluye todo lo calculado en las otras pestañas. "
                       "Si querés que incluya la clasificación, entrenala primero en la pestaña correspondiente.")
            if st.button("Generar PDF"):
                with st.spinner("Generando reporte..."):
                    clasif_guardado = st.session_state.resultados.get("clasificacion")
                    pdf_bytes = generar_pdf(
                        df, info_nulos, describe_num, corr, iqr_tabla,
                        multi_out, clasif_guardado,
                    )
                st.success("Reporte generado.")
                st.download_button("⬇️ Descargar reporte PDF", pdf_bytes,
                                    "reporte_eda.pdf", "application/pdf")
    elif df is not None:
        st.warning("El archivo se cargó pero no tiene datos.")
else:
    st.info("Esperando que subas un archivo para comenzar el análisis.")
