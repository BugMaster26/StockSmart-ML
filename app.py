"""Aplicación web académica StockSmart ML."""
from __future__ import annotations

import hashlib
import io
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.classification import evaluate_models
from src.clustering import draw_dendrogram, segment_products
from src.features import build_weekly_features
from src.inventory_alerts import alerts_to_csv, create_inventory_alerts
from src.preprocessing import clean_transactions, make_demo_data


st.set_page_config(page_title="StockSmart ML", page_icon="📦", layout="wide")


@st.cache_data(show_spinner=False)
def load_frame(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Carga CSV/Excel en memoria; no escribe los archivos subidos a disco."""
    suffix = filename.lower().rsplit(".", maxsplit=1)[-1]
    if suffix in {"xlsx", "xlsm"}:
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", dtype={"StockCode": "string"})
    if suffix == "xls":
        return pd.read_excel(io.BytesIO(file_bytes), engine="xlrd", dtype={"StockCode": "string"})
    if suffix == "csv":
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig", dtype={"StockCode": "string"})
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1", dtype={"StockCode": "string"})
    raise ValueError("Formato no compatible. Usa CSV, XLSX, XLSM o XLS.")


@st.cache_data(show_spinner=False)
def cached_clean_transactions(frame: pd.DataFrame):
    return clean_transactions(frame)


@st.cache_data(show_spinner=False)
def cached_weekly_features(frame: pd.DataFrame, low_quantile: float, high_quantile: float):
    return build_weekly_features(frame, low_quantile, high_quantile)


@st.cache_data(show_spinner=False)
def cached_model_evaluation(features: pd.DataFrame, low_quantile: float, high_quantile: float):
    return evaluate_models(features, low_quantile=low_quantile, high_quantile=high_quantile)


@st.cache_data(show_spinner=False)
def cached_segmentation(frame: pd.DataFrame, n_clusters: int):
    return segment_products(frame, n_clusters)


def show_panel(clean: pd.DataFrame, report, origin_label: str) -> None:
    st.subheader("Resumen de ventas")
    st.caption(f"Origen de datos: {origin_label}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ingresos", f"{clean.Revenue.sum():,.2f}")
    col2.metric("Unidades vendidas", f"{clean.Quantity.sum():,.0f}")
    col3.metric("Productos", f"{clean.StockCode.nunique():,}")
    col4.metric("Registros válidos", f"{len(clean):,}")
    weekly = (clean.assign(Semana=clean.InvoiceDate.dt.to_period("W-SUN").dt.start_time)
              .groupby("Semana").Revenue.sum())
    st.subheader("Ingresos por semana")
    st.line_chart(weekly.rename("Ingresos"), y_label="Ingresos (moneda del archivo)", width="stretch")
    st.write(f"Preprocesamiento: {report.rows_input:,} filas de entrada, {report.rows_output:,} filas válidas, "
             f"{report.duplicate_rows:,} duplicados detectados, {report.cancellation_rows:,} cancelaciones y "
             f"{report.negative_quantity_rows:,} filas con cantidades negativas.")


def show_data_page(raw: pd.DataFrame, clean: pd.DataFrame, report) -> None:
    st.subheader("Carga y preprocesamiento")
    st.markdown("Columnas requeridas: `InvoiceDate`, `StockCode`, `Quantity` y `UnitPrice`. "
                "`InvoiceNo` permite reconocer facturas canceladas que empiezan por C.")
    st.subheader("Vista previa del archivo de origen")
    st.dataframe(raw.head(30), width="stretch")
    st.subheader("Informe de calidad")
    col1, col2 = st.columns(2)
    col1.dataframe(pd.DataFrame({"Control": ["Filas de entrada", "Filas válidas", "Duplicados", "Cancelaciones",
                                                    "Cantidades negativas", "Filas inválidas"],
                                 "Cantidad": [report.rows_input, report.rows_output, report.duplicate_rows,
                                              report.cancellation_rows, report.negative_quantity_rows,
                                              report.invalid_rows_removed]}), hide_index=True, width="stretch")
    col2.dataframe(pd.DataFrame({"Columna": list(report.missing_by_column),
                                 "Valores faltantes": list(report.missing_by_column.values())}),
                   hide_index=True, width="stretch")
    st.caption("Se eliminan filas con campos requeridos inválidos, cancelaciones (si se dispone de InvoiceNo), "
               "cantidades no positivas y duplicados exactos. Los datos procesados se descargan, no se guardan en el repositorio.")
    st.download_button("Descargar ventas procesadas", clean.to_csv(index=False).encode("utf-8-sig"),
                       "ventas_procesadas.csv", "text/csv")


def show_models_page(clean: pd.DataFrame, dataset_key: str) -> None:
    st.subheader("Predicción de demanda semanal")
    st.markdown("El objetivo es la demanda de la semana siguiente. Rezagos, promedio, variabilidad y tendencia "
                "se calculan con semanas observadas antes del objetivo.")
    low_pct, high_pct = st.slider("Percentiles para separar clases baja, media y alta", 0, 100, (33, 67), key="demand_thresholds")
    try:
        features, thresholds = cached_weekly_features(clean, low_pct / 100, high_pct / 100)
        st.caption(f"Límites descriptivos globales: baja ≤ {thresholds['bajo']:.1f}; "
                   f"media ≤ {thresholds['alto']:.1f}; alta > {thresholds['alto']:.1f} unidades.")
        if features.empty or features.Demand.nunique() < 2:
            st.warning("Datos insuficientes para evaluar clasificadores. Se necesitan historial semanal y variación de demanda.")
            return
        with st.spinner("Entrenando y evaluando KNN, Árbol de Decisión y MLP…"):
            result = cached_model_evaluation(features, low_pct / 100, high_pct / 100)
        if result["metrics"].empty:
            st.warning(result["warning"])
            return
        st.session_state["model_report"] = {"dataset_key": dataset_key, "result": result}
        st.dataframe(result["metrics"].style.format({c: "{:.3f}" for c in result["metrics"].columns if c != "Modelo"}),
                     hide_index=True, width="stretch")
        if result["warning"]:
            st.warning(result["warning"])
        st.success(f"Mayor F1 macro en esta partición: {result['selected']}")
        selected = st.selectbox("Matriz de confusión", list(result["confusion"]), key="confusion_model")
        st.dataframe(result["confusion"][selected], width="stretch")
        st.download_button("Descargar métricas CSV", result["metrics"].to_csv(index=False).encode("utf-8-sig"),
                           "metricas_modelos.csv", "text/csv")
        st.caption(f"Corte cronológico por semana: entrenamiento hasta {result['train_last_week'].date()}, "
                   f"prueba desde {result['test_first_week'].date()} ({result['train_rows']} filas de entrenamiento, "
                   f"{result['test_rows']} de prueba). Los límites usados en la evaluación se calculan solo con entrenamiento "
                   f"(bajo ≤ {result['thresholds']['bajo']:.1f}; medio ≤ {result['thresholds']['alto']:.1f}). "
                   "La selección depende de una única ventana temporal y no garantiza generalización.")
    except ValueError as exc:
        st.error(f"No se pudo evaluar la demanda: {exc}")


def show_clustering_page(clean: pd.DataFrame) -> None:
    st.subheader("Segmentación de productos")
    product_count = clean.StockCode.nunique()
    if product_count < 3:
        st.warning("Se requieren al menos tres productos para segmentar.")
        return
    k = st.slider("Número de grupos", 2, min(8, product_count - 1), min(3, product_count - 1), key="cluster_count")
    try:
        with st.spinner("Agrupando productos con variables estandarizadas…"):
            result = cached_segmentation(clean, k)
        st.caption("Ambos métodos usan las mismas variables escaladas: " + ", ".join(result["feature_names"]) +
                   ". Los gráficos muestran unidades e ingresos en sus unidades originales.")
        kmeans_tab, hierarchical_tab = st.tabs(["K-Means", "Clustering jerárquico"])
        with kmeans_tab:
            st.subheader("K-Means")
            fig, ax = plt.subplots()
            points = ax.scatter(result["products"].units, result["products"].revenue,
                                c=result["products"]["K-Means"], cmap="viridis")
            ax.set(xlabel="Unidades vendidas", ylabel="Ingresos", title="Segmentación K-Means")
            fig.colorbar(points, ax=ax, label="Grupo")
            st.pyplot(fig); plt.close(fig)
            st.write("Perfil medio de los grupos (unidades originales)")
            st.dataframe(result["profile_kmeans"], width="stretch")
            score = result["silhouette_kmeans"]
            st.metric("Silhouette", f"{score:.3f}" if score is not None else "No aplicable")
        with hierarchical_tab:
            st.subheader("Clustering jerárquico Ward")
            fig, ax = plt.subplots()
            points = ax.scatter(result["products"].units, result["products"].revenue,
                                c=result["products"]["Jerárquico"], cmap="plasma")
            ax.set(xlabel="Unidades vendidas", ylabel="Ingresos", title="Segmentación jerárquica")
            fig.colorbar(points, ax=ax, label="Grupo")
            st.pyplot(fig); plt.close(fig)
            st.write("Perfil medio de los grupos (unidades originales)")
            st.dataframe(result["profile_hierarchical"], width="stretch")
            score = result["silhouette_hierarchical"]
            st.metric("Silhouette", f"{score:.3f}" if score is not None else "No aplicable")
            st.subheader("Dendrograma Ward")
            fig, ax = plt.subplots(figsize=(10, 4))
            draw_dendrogram(result, max_leaves=30, ax=ax)
            ax.set_ylabel("Distancia")
            st.pyplot(fig); plt.close(fig)
            if product_count > 30:
                st.caption("Dendrograma resumido a 30 ramas para facilitar su lectura; la asignación usa todos los productos.")
        st.subheader("Asignación por producto")
        st.dataframe(result["products"], hide_index=True, width="stretch")
    except ValueError as exc:
        st.error(f"No se pudo segmentar: {exc}")


def show_inventory_page(clean: pd.DataFrame) -> None:
    st.subheader("Alertas orientativas de inventario")
    st.info("Ingresa o importa el stock actual real. Online Retail contiene ventas, pero no inventario histórico.")
    st.caption("La estimación es la media de las últimas cuatro semanas calendario hasta la última semana del archivo. Las semanas sin venta se cuentan como cero. Solo se alerta cuando la demanda estimada supera el stock actual.")
    products = sorted(clean.StockCode.unique())
    template = pd.DataFrame({"StockCode": products, "Stock": [pd.NA] * len(products)})
    st.download_button("Descargar plantilla de stock", template.to_csv(index=False).encode("utf-8-sig"),
                       "plantilla_stock_actual.csv", "text/csv")
    stock_upload = st.file_uploader("Importar stock CSV (StockCode, Stock)", type=["csv"], key="inventory_upload")
    try:
        if stock_upload:
            stock = pd.read_csv(stock_upload, dtype={"StockCode": "string"})
        else:
            stock = template
        if not {"StockCode", "Stock"}.issubset(stock.columns):
            st.error("El archivo debe contener las columnas StockCode y Stock.")
            return
        edited = st.data_editor(
            stock[["StockCode", "Stock"]], num_rows="dynamic", width="stretch",
            column_config={"Stock": st.column_config.NumberColumn("Stock actual (unidades)", min_value=0, step=1)},
            key="inventory_editor",
        )
        codes = edited.StockCode.astype("string").str.strip()
        quantities = pd.to_numeric(edited.Stock, errors="coerce")
        if codes.isna().any() or codes.eq("").any() or quantities.isna().any():
            st.warning("Completa el código y el stock actual de cada fila antes de calcular. Los campos vacíos no se asumen como cero.")
        elif (quantities < 0).any():
            st.error("El stock actual no puede ser negativo.")
        else:
            validated = edited.copy()
            validated["StockCode"], validated["Stock"] = codes, quantities
            alerts = create_inventory_alerts(clean, validated)
            st.dataframe(alerts, hide_index=True, width="stretch")
            st.download_button("Descargar alertas CSV", alerts_to_csv(alerts), "alertas_inventario.csv", "text/csv")
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.error(f"No se pudieron calcular las alertas: {exc}")
    st.caption("Límites: las ventas observadas pueden subestimar demanda si hubo quiebres de stock. La estimación no incluye estacionalidad, plazos de entrega, compras en tránsito ni inventario reservado; es una guía, no una orden automática de compra.")


def show_report_page(clean: pd.DataFrame, origin_label: str, dataset_key: str) -> None:
    st.subheader("Informe de resultados y limitaciones")
    st.write(f"Origen: **{origin_label}**. Filas procesadas: {len(clean):,}. Productos: {clean.StockCode.nunique():,}.")
    st.write("Modelos de demanda: KNN, Árbol de Decisión y MLP de scikit-learn. "
             "Segmentación: K-Means y clustering jerárquico Ward.")
    report = st.session_state.get("model_report")
    if report and report.get("dataset_key") == dataset_key and not report["result"]["metrics"].empty:
        result = report["result"]
        st.success(f"Mejor F1 macro medido en la evaluación cronológica: {result['selected']}")
        st.dataframe(result["metrics"], hide_index=True, width="stretch")
        st.download_button("Descargar tabla de métricas CSV", result["metrics"].to_csv(index=False).encode("utf-8-sig"),
                           "metricas_modelos.csv", "text/csv")
    else:
        st.info("Ve a Modelos de demanda para entrenar y evaluar los clasificadores; sus resultados aparecerán aquí.")
    st.markdown("**Limitaciones:** ventas observadas no equivalen a demanda completa si hubo falta de stock; no hay inventario histórico; la alerta usa una media de cuatro semanas y no modela plazos, estacionalidad ni compras en tránsito; la evaluación de clasificación usa una sola división cronológica.")


st.title("📦 StockSmart ML")
st.caption("Análisis de ventas, demanda, segmentos de productos y reposición")

with st.sidebar:
    st.header("Datos de entrada")
    source = st.radio("Origen", ["Demostración", "Cargar archivo"], key="data_source")
    uploaded = st.file_uploader("Subir CSV o Excel", type=["csv", "xlsx", "xlsm", "xls"], key="sales_upload") if source == "Cargar archivo" else None
    st.caption("Columnas requeridas: InvoiceDate, StockCode, Quantity y UnitPrice.")
    st.divider()
    section = st.radio("Navegación", ["Panel principal", "Carga de datos", "Modelos de demanda",
                                       "Clustering", "Alertas de inventario", "Informe"], key="main_navigation")

try:
    raw = make_demo_data() if source == "Demostración" else (
        load_frame(uploaded.getvalue(), uploaded.name) if uploaded is not None else None
    )
    if raw is None:
        st.info("Carga un CSV o Excel desde el menú lateral, o selecciona Demostración para explorar la aplicación.")
        st.stop()
    clean, cleaning_report = cached_clean_transactions(raw)
except (ValueError, ImportError, UnicodeDecodeError, pd.errors.ParserError,
        pd.errors.EmptyDataError, zipfile.BadZipFile) as exc:
    st.error(f"No se pudo leer o procesar el archivo: {exc}")
    st.stop()

if clean.empty:
    st.error("El archivo no contiene ventas válidas después del preprocesamiento. Revisa columnas, fechas, cantidades y precios.")
    st.stop()

origin = "DEMOSTRACIÓN · datos sintéticos" if source == "Demostración" else "Archivo cargado por el usuario"
if source == "Demostración":
    st.warning("DEMOSTRACIÓN: datos sintéticos generados localmente; no son resultados del dataset Online Retail.")
else:
    st.success("Datos cargados y procesados desde el archivo seleccionado.")

dataset_key = hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded is not None else "demo-seed-42-v1"

if section == "Panel principal":
    show_panel(clean, cleaning_report, origin)
elif section == "Carga de datos":
    show_data_page(raw, clean, cleaning_report)
elif section == "Modelos de demanda":
    show_models_page(clean, dataset_key)
elif section == "Clustering":
    show_clustering_page(clean)
elif section == "Alertas de inventario":
    show_inventory_page(clean)
elif section == "Informe":
    show_report_page(clean, origin, dataset_key)
