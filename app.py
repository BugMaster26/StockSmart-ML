"""Aplicación web académica StockSmart ML."""
from __future__ import annotations

import hashlib
import io
import logging
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.classification import evaluate_models
from src.clustering import draw_dendrogram, segment_products
from src.features import build_weekly_features
from src.inventory_alerts import alerts_to_csv, create_inventory_alerts, create_observed_inventory_alerts
from src.mendeley_data import MendeleyDataset, load_mendeley_dataset
from src.preprocessing import clean_transactions, make_demo_data


st.set_page_config(page_title="StockSmart ML", page_icon="📦", layout="wide")
logger = logging.getLogger(__name__)


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


@st.cache_data(show_spinner=False, max_entries=1)
def cached_weekly_features(frame: pd.DataFrame, low_quantile: float, high_quantile: float):
    return build_weekly_features(frame, low_quantile, high_quantile)


@st.cache_data(show_spinner=False, max_entries=1)
def cached_model_evaluation(features: pd.DataFrame, low_quantile: float, high_quantile: float):
    return evaluate_models(features, low_quantile=low_quantile, high_quantile=high_quantile)


@st.cache_data(show_spinner=False)
def cached_segmentation(frame: pd.DataFrame, n_clusters: int):
    return segment_products(frame, n_clusters)


@st.cache_data(show_spinner=False, max_entries=1)
def cached_mendeley_local(sales_path: str, sales_size: int, sales_mtime: int,
                          inventory_path: str, inventory_size: int, inventory_mtime: int):
    """Cachea por tamaño/mtime para evitar releer los CSV en cada interacción."""
    logger.info("Mendeley load started: mode=local sales_bytes=%d inventory_bytes=%d",
                sales_size, inventory_size)
    started = time.perf_counter()
    try:
        dataset = load_mendeley_dataset(sales_path, inventory_path)
    except Exception:
        logger.exception("Mendeley load failed: mode=local")
        raise
    logger.info("Mendeley load completed: mode=local seconds=%.2f sales_rows=%d inventory_rows=%d",
                time.perf_counter() - started, len(dataset.sales), len(dataset.inventory))
    return dataset


@st.cache_data(show_spinner=False, max_entries=1)
def cached_mendeley_upload(sales_name: str, sales_bytes: bytes,
                           inventory_name: str, inventory_bytes: bytes):
    """Cachea la importación conjunta de archivos Mendeley cargados en sesión."""
    logger.info("Mendeley load started: mode=upload sales_bytes=%d inventory_bytes=%d",
                len(sales_bytes), len(inventory_bytes))
    started = time.perf_counter()
    try:
        dataset = load_mendeley_dataset(sales_bytes, inventory_bytes)
    except Exception:
        logger.exception("Mendeley load failed: mode=upload")
        raise
    logger.info("Mendeley load completed: mode=upload seconds=%.2f sales_rows=%d inventory_rows=%d",
                time.perf_counter() - started, len(dataset.sales), len(dataset.inventory))
    return dataset


def show_panel(clean: pd.DataFrame, report, origin_label: str, dataset: MendeleyDataset | None = None) -> None:
    st.subheader("Resumen de ventas")
    st.caption(f"Origen de datos: {origin_label}")
    if dataset:
        sale_quality = dataset.quality["sales"]
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Monto de ventas brutas*", f"{sale_quality['gross_sales_amount']:,.2f}")
        col2.metric("Cantidad vendida*", f"{sale_quality['gross_units_positive_sales']:,.0f}")
        col3.metric("Productos", f"{sale_quality['products']:,}")
        col4.metric("Tiendas", f"{sale_quality['stores']:,}")
        col5.metric("Transacciones reportadas", f"{int(clean.TransactionCount.sum()):,}")
        col6.metric("Filas marcadas devolución", f"{sale_quality['return_rows_excluded_from_demand']:,}")
        st.caption(
            f"*Campos monetarios y cantidades tal como aparecen en el dataset; no se especifican moneda ni unidad física. "
            f"Devoluciones: {sale_quality['return_units_abs']:,.0f} cantidades y "
            f"{sale_quality['return_sales_amount']:,.2f} en Sales Amount (monto con signo). "
            f"Monto neto reportado por el archivo: {sale_quality['net_sales_amount_reported']:,.2f}."
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ingresos", f"{clean.Revenue.sum():,.2f}")
        col2.metric("Unidades vendidas", f"{clean.Quantity.sum():,.0f}")
        col3.metric("Productos", f"{clean.StockCode.nunique():,}")
        col4.metric("Registros válidos", f"{len(clean):,}")
    weekly = (clean.assign(Semana=clean.InvoiceDate.dt.to_period("W-SUN").dt.start_time)
              .groupby("Semana").Revenue.sum())
    st.subheader("Ingresos por semana")
    st.line_chart(weekly.rename("Ingresos"), y_label=("Monto reportado (moneda no indicada)" if dataset else "Ingresos"), width="stretch")
    if dataset:
        sale_quality = dataset.quality["sales"]
        st.write(
            f"Preprocesamiento: {sale_quality['rows']:,} filas fuente; "
            f"{sale_quality['positive_sales_rows_retained']:,} ventas positivas usadas; "
            f"{sale_quality['return_rows_excluded_from_demand']:,} devoluciones separadas; "
            f"{sale_quality['zero_quantity_rows_excluded']:,} filas con cantidad cero excluidas; "
            f"{sale_quality['invalid_or_unusable_rows']:,} filas inválidas o no interpretables."
        )
        products = clean.groupby("StockCode", as_index=False).agg(Quantity=("Quantity", "sum"), Revenue=("Revenue", "sum"))
        stores = clean.groupby("Store", as_index=False).agg(Quantity=("Quantity", "sum"), Revenue=("Revenue", "sum"))
        left, right = st.columns(2)
        left.subheader("10 productos con más cantidad registrada")
        left.bar_chart(products.nlargest(10, "Quantity").set_index("StockCode")["Quantity"])
        right.subheader("Ventas por tienda")
        right.bar_chart(stores.set_index("Store")["Revenue"])
    else:
        st.write(f"Preprocesamiento: {report.rows_input:,} filas de entrada, {report.rows_output:,} filas válidas, "
                 f"{report.duplicate_rows:,} duplicados detectados, {report.cancellation_rows:,} cancelaciones y "
                 f"{report.negative_quantity_rows:,} filas con cantidades negativas.")


def show_data_page(raw: pd.DataFrame, clean: pd.DataFrame, report, dataset: MendeleyDataset | None = None) -> None:
    if dataset:
        quality = dataset.quality
        sales_quality, inventory_quality = quality["sales"], quality["inventory"]
        relation = quality["relationship"]
        st.subheader("Dataset Mendeley: calidad, transformaciones y relación de tablas")
        st.caption("Los CSV originales se leen por bloques y permanecen sin modificaciones; las tablas siguientes son el resultado normalizado en memoria.")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Filas de ventas fuente", f"{sales_quality['rows']:,}")
        col2.metric("Ventas positivas usadas", f"{sales_quality['positive_sales_rows_retained']:,}")
        col3.metric("Filas de inventario", f"{inventory_quality['rows']:,}")
        col4.metric("Productos", f"{sales_quality['products']:,}")
        col5.metric("Tiendas", f"{sales_quality['stores']:,}")
        st.write(f"Periodo de ventas: **{sales_quality['date_min']} – {sales_quality['date_max']}**. "
                 f"Inicio de intervalos de inventario: **{inventory_quality['date_min']} – {inventory_quality['date_max']}**; "
                 f"{inventory_quality['open_ended_intervals']:,} intervalos terminan con el centinela 9999-12-31 (tratado como abierto).")

        st.markdown("**Tratamiento de ventas:** se conservan para la demanda las filas `Is Return = 0` con `Qty Sold > 0`; "
                    "las devoluciones se informan por separado y no se convierten en demanda positiva. "
                    "`Revenue` conserva `Sales Amount`; `UnitPrice` es una variable derivada `Sales Amount / Qty Sold` para compatibilidad con el pipeline existente.")
        missing_rows = []
        for section, details in [("Ventas", sales_quality), ("Inventario", inventory_quality)]:
            for column, count in details["missing_by_column"].items():
                missing_rows.append({"Tabla": section, "Columna": column, "Valores faltantes": count})
        st.subheader("Valores faltantes y exclusiones")
        if missing_rows:
            st.dataframe(pd.DataFrame(missing_rows), hide_index=True, width="stretch")
        else:
            st.success("No hay valores faltantes detectados en las columnas fuente.")
        st.dataframe(pd.DataFrame([
            {"Control": "Devoluciones separadas de la demanda", "Filas": sales_quality["return_rows_excluded_from_demand"]},
            {"Control": "Ventas positivas conservadas", "Filas": sales_quality["positive_sales_rows_retained"]},
            {"Control": "Cantidad cero excluida", "Filas": sales_quality["zero_quantity_rows_excluded"]},
            {"Control": "Filas no interpretables", "Filas": sales_quality["invalid_or_unusable_rows"]},
            {"Control": "Existencia negativa (se conserva y marca)", "Filas": inventory_quality["rows_with_negative_stock"]},
            {"Control": "Existencia cero (se conserva y marca)", "Filas": inventory_quality["rows_with_zero_stock"]},
        ]), hide_index=True, width="stretch")
        st.subheader("Validación de granularidad y unión")
        st.dataframe(pd.DataFrame([
            {"Control": "Product No presentes en ambas tablas", "Filas/valores": relation["product_codes_in_both"]},
            {"Control": "Store presentes en ambas tablas", "Filas/valores": relation["stores_in_both"]},
            {"Control": "Claves producto-tienda-canal compartidas", "Filas/valores": relation["exact_product_store_channel_pairs_in_both"]},
            {"Control": "Ventas positivas con un intervalo temporal compatible", "Filas/valores": relation["positive_sales_rows_with_one_date_compatible_inventory_interval"]},
            {"Control": "Ventas sin intervalo compatible", "Filas/valores": relation["positive_sales_rows_without_date_compatible_inventory_interval"]},
            {"Control": "Ventas con varios intervalos compatibles", "Filas/valores": relation["positive_sales_rows_with_multiple_date_compatible_inventory_intervals"]},
            {"Control": "Coincidencias temporales con estado distinto", "Filas/valores": relation["date_compatible_rows_with_status_mismatch"]},
            {"Control": "Intervalos de inventario solapados por producto-tienda-canal", "Filas/valores": inventory_quality["interval_overlap_rows_within_product_store_channel"]},
        ]), hide_index=True, width="stretch")
        st.caption("La unión temporal usa Product No, Store y Sales Channel, y exige Start Date ≤ fecha ≤ End Date (9999-12-31 se interpreta como abierto). Los estados de precio no son claves. Solo se utiliza stock para una clave/fecha con exactamente un intervalo; los demás casos no se comparan.")
        st.subheader("Columnas y vista normalizada")
        with st.expander("Columnas encontradas en los archivos originales"):
            left, right = st.columns(2)
            left.write("Ventas")
            left.code(", ".join(sales_quality["source_columns"]))
            right.write("Inventario")
            right.code(", ".join(inventory_quality["source_columns"]))
        sales_tab, inventory_tab = st.tabs(["Ventas normalizadas", "Intervalos de inventario normalizados"])
        with sales_tab:
            st.dataframe(clean.head(100), hide_index=True, width="stretch")
            st.download_button("Descargar ventas procesadas CSV", clean.to_csv(index=False).encode("utf-8-sig"),
                               "ventas_mendeley_procesadas.csv", "text/csv")
        with inventory_tab:
            st.dataframe(dataset.inventory.head(100), hide_index=True, width="stretch")
            st.download_button("Descargar inventario procesado CSV", dataset.inventory.to_csv(index=False).encode("utf-8-sig"),
                               "inventario_mendeley_procesado.csv", "text/csv")
        st.warning("No se especifican moneda ni unidades físicas. La descripción de producto incluye un PROD- distinto de Product No en las filas inspeccionadas; Product No es la clave y la descripción no se usa para unir.")
        st.caption("Licencia declarada en la ficha de Mendeley: CC BY 4.0. Los resultados corresponden a una muestra estratificada y no prueban ventas ni disponibilidad completas del negocio.")
        return

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
    stage = "weekly_feature_generation"
    try:
        logger.info("Demand page stage started: stage=%s input_rows=%d input_columns=%d",
                    stage, len(clean), len(clean.columns))
        started = time.perf_counter()
        features, thresholds = cached_weekly_features(clean, low_pct / 100, high_pct / 100)
        logger.info(
            "Demand page stage completed: stage=%s seconds=%.2f rows=%d columns=%d "
            "missing_cells=%d dataframe_mib=%.2f",
            stage, time.perf_counter() - started, len(features), len(features.columns),
            int(features.isna().sum().sum()), features.memory_usage(deep=True).sum() / 1024**2,
        )
        st.caption(f"Límites descriptivos globales: baja ≤ {thresholds['bajo']:.1f}; "
                   f"media ≤ {thresholds['alto']:.1f}; alta > {thresholds['alto']:.1f} unidades.")
        if features.empty or features.Demand.nunique() < 2:
            st.warning("Datos insuficientes para evaluar clasificadores. Se necesitan historial semanal y variación de demanda.")
            return
        stage = "model_evaluation"
        logger.info("Demand page stage started: stage=%s feature_rows=%d feature_columns=%d",
                    stage, len(features), len(features.columns))
        started = time.perf_counter()
        with st.spinner("Entrenando y evaluando KNN, Árbol de Decisión y MLP…"):
            result = cached_model_evaluation(features, low_pct / 100, high_pct / 100)
        logger.info("Demand page stage completed: stage=%s seconds=%.2f result_rows=%d",
                    stage, time.perf_counter() - started, len(result["metrics"]))
        if result["metrics"].empty:
            st.warning(result["warning"])
            return
        st.session_state["model_report"] = {"dataset_key": dataset_key, "result": result}
        stage = "metrics_table_render"
        logger.info("Demand page stage started: stage=%s metric_rows=%d metric_columns=%d",
                    stage, len(result["metrics"]), len(result["metrics"].columns))
        st.dataframe(result["metrics"].style.format({c: "{:.3f}" for c in result["metrics"].columns if c != "Modelo"}),
                     hide_index=True, width="stretch")
        logger.info("Demand page stage completed: stage=%s", stage)
        if result["warning"]:
            stage = "evaluation_warning_render"
            st.warning(result["warning"])
        stage = "selected_model_render"
        st.success(f"Mayor F1 macro en esta partición: {result['selected']}")
        stage = "confusion_matrix_selector"
        logger.info("Demand page stage started: stage=%s available_matrices=%d",
                    stage, len(result["confusion"]))
        selected = st.selectbox("Matriz de confusión", list(result["confusion"]), key="confusion_model")
        logger.info("Demand page stage completed: stage=%s selected=%s", stage, selected)
        stage = "confusion_matrix_render"
        st.dataframe(result["confusion"][selected], width="stretch")
        logger.info("Demand page stage completed: stage=%s", stage)
        stage = "metrics_csv_export"
        logger.info("Demand page stage started: stage=%s", stage)
        st.download_button("Descargar métricas CSV", result["metrics"].to_csv(index=False).encode("utf-8-sig"),
                           "metricas_modelos.csv", "text/csv")
        logger.info("Demand page stage completed: stage=%s", stage)
        stage = "temporal_summary_render"
        st.caption(f"Corte cronológico por semana: entrenamiento hasta {result['train_last_week'].date()}, "
                   f"prueba desde {result['test_first_week'].date()} ({result['train_rows']} filas de entrenamiento, "
                   f"{result['test_rows']} de prueba). Los límites usados en la evaluación se calculan solo con entrenamiento "
                   f"(bajo ≤ {result['thresholds']['bajo']:.1f}; medio ≤ {result['thresholds']['alto']:.1f}). "
                   "La selección depende de una única ventana temporal y no garantiza generalización.")
        logger.info("Demand page stage completed: stage=%s", stage)
    except ValueError as exc:
        logger.exception("Demand page failed: stage=%s", stage)
        st.error(f"No se pudo evaluar la demanda: {exc}")
    except Exception:
        logger.exception("Demand page crashed: stage=%s", stage)
        raise


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


def show_inventory_page(clean: pd.DataFrame, dataset: MendeleyDataset | None = None) -> None:
    if dataset is None:
        show_manual_inventory_page(clean)
        return
    st.subheader("Alertas de inventario")
    st.markdown("Las alertas basadas en el CSV usan solo inventario temporalmente compatible por producto, tienda y canal. La vista manual sigue disponible como alternativa.")
    observed_tab, manual_tab = st.tabs(["Existencia observada en intervalo", "Stock manual"])
    with observed_tab:
        try:
            alerts = create_observed_inventory_alerts(clean, dataset.inventory)
            if alerts.empty:
                st.warning("No hay intervalos de inventario y ventas suficientes para calcular alertas.")
            else:
                as_of = alerts["Fecha de consulta"].iloc[0]
                st.caption(f"Fecha de consulta: {as_of.date()}. La demanda es la media de las cuatro semanas calendario completas anteriores; filas sin venta en la ventana no se convierten en demanda cero.")
                counts = alerts["Alerta"].value_counts()
                col1, col2, col3 = st.columns(3)
                col1.metric("Claves evaluadas", f"{len(alerts):,}")
                col2.metric("Riesgo orientativo", f"{int(counts.get('Riesgo orientativo: demanda media supera existencia observada', 0)):,}")
                col3.metric("Sin intervalo compatible", f"{int(counts.get('Sin intervalo de inventario compatible', 0)):,}")
                st.dataframe(alerts, hide_index=True, width="stretch")
                st.download_button("Descargar alertas observadas CSV", alerts_to_csv(alerts),
                                   "alertas_inventario_observado.csv", "text/csv")
        except ValueError as exc:
            st.error(f"No se pudieron calcular alertas con inventario observado: {exc}")
        st.caption("Una existencia negativa o igual a cero se marca para revisión. La comparación con una media de ventas solo indica riesgo orientativo; no confirma una rotura de stock, no modela plazos y la ficha describe una muestra estratificada.")
    with manual_tab:
        st.caption("En esta alternativa, el stock se ingresa por producto y se contrasta con la demanda agregada entre tiendas; úsala si quieres proporcionar una existencia total manual.")
        show_manual_inventory_page(clean)


def show_manual_inventory_page(clean: pd.DataFrame) -> None:
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


def show_report_page(clean: pd.DataFrame, origin_label: str, dataset_key: str,
                     dataset: MendeleyDataset | None = None) -> None:
    st.subheader("Informe de resultados y limitaciones")
    st.write(f"Origen: **{origin_label}**. Filas procesadas: {len(clean):,}. Productos: {clean.StockCode.nunique():,}.")
    if dataset:
        sales_quality, inventory_quality = dataset.quality["sales"], dataset.quality["inventory"]
        st.write(f"Periodo de ventas real: {sales_quality['date_min']} a {sales_quality['date_max']}. "
                 f"Periodo de inicio de intervalos de inventario: {inventory_quality['date_min']} a {inventory_quality['date_max']}. "
                 "La moneda y las unidades físicas no están especificadas en los archivos.")
        quality_rows = []
        for section, details in [("Ventas", sales_quality), ("Inventario", inventory_quality)]:
            quality_rows.extend({"Tabla": section, "Columna": column, "Valores faltantes": count}
                                for column, count in details["missing_by_column"].items())
        quality_table = pd.DataFrame(quality_rows, columns=["Tabla", "Columna", "Valores faltantes"])
        if quality_table.empty:
            quality_table = pd.DataFrame([{"Tabla": "Ambas", "Columna": "Sin faltantes detectados", "Valores faltantes": 0}])
        st.download_button("Descargar resumen de calidad CSV", quality_table.to_csv(index=False).encode("utf-8-sig"),
                           "calidad_dataset_mendeley.csv", "text/csv")
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
    if dataset:
        st.markdown("**Limitaciones:** el dataset se describe como muestra estratificada; no especifica moneda ni unidad física. Las fechas Start/End determinan compatibilidad por intervalo, pero no documentan el instante de medición de Qty on hand. Hay ventas sin intervalo compatible y diferencias entre Sales Type y Stock Status. La alerta no confirma rotura de stock ni incorpora plazos o compras en tránsito. La clasificación agrega por producto entre tiendas y usa una sola división cronológica.")
    else:
        st.markdown("**Limitaciones:** ventas observadas no equivalen a demanda completa si hubo falta de stock; no hay inventario histórico; la alerta usa una media de cuatro semanas y no modela plazos, estacionalidad ni compras en tránsito; la evaluación de clasificación usa una sola división cronológica.")


st.title("📦 StockSmart ML")
st.caption("Análisis de ventas, demanda, segmentos de productos y reposición")

with st.sidebar:
    st.header("Datos de entrada")
    source = st.radio("Origen", ["Demostración", "Dataset Mendeley", "Archivo propio"], key="data_source")
    uploaded = st.file_uploader("Subir CSV o Excel", type=["csv", "xlsx", "xlsm", "xls"], key="sales_upload") if source == "Archivo propio" else None
    mendeley_sales_upload = None
    mendeley_inventory_upload = None
    if source == "Dataset Mendeley":
        local_data_dir = Path(__file__).resolve().parent / "data" / "raw"
        local_sales_path = local_data_dir / "retail_sales_ml_apl.csv"
        local_inventory_path = local_data_dir / "retail_inventory_ml_apl.csv"
        if local_sales_path.is_file() and local_inventory_path.is_file():
            st.info("Se usarán los dos CSV locales de data/raw/.")
        else:
            st.caption("En despliegues sin archivos locales, carga aquí ambos CSV originales de Mendeley.")
            mendeley_sales_upload = st.file_uploader("Ventas: retail_sales_ml_apl.csv", type=["csv"], key="mendeley_sales_upload")
            mendeley_inventory_upload = st.file_uploader("Inventario: retail_inventory_ml_apl.csv", type=["csv"], key="mendeley_inventory_upload")
    elif source == "Archivo propio":
        st.caption("Esquema requerido: InvoiceDate, StockCode, Quantity y UnitPrice. InvoiceNo es opcional.")
    else:
        st.caption("La demo usa datos sintéticos reproducibles.")
    st.divider()
    section = st.radio("Navegación", ["Panel principal", "Carga de datos", "Modelos de demanda",
                                       "Clustering", "Alertas de inventario", "Informe"], key="main_navigation")

try:
    dataset = None
    if source == "Demostración":
        raw = make_demo_data()
        clean, cleaning_report = cached_clean_transactions(raw)
        dataset_key = "demo-seed-42-v1"
        origin = "DEMOSTRACIÓN · datos sintéticos"
    elif source == "Archivo propio":
        if uploaded is None:
            st.info("Carga un CSV o Excel desde el menú lateral, o selecciona Demostración para explorar la aplicación.")
            st.stop()
        file_bytes = uploaded.getvalue()
        raw = load_frame(file_bytes, uploaded.name)
        clean, cleaning_report = cached_clean_transactions(raw)
        dataset_key = hashlib.sha256(file_bytes).hexdigest()
        origin = "Archivo cargado por el usuario"
    else:
        sales_path = Path(__file__).resolve().parent / "data" / "raw" / "retail_sales_ml_apl.csv"
        inventory_path = Path(__file__).resolve().parent / "data" / "raw" / "retail_inventory_ml_apl.csv"
        if sales_path.is_file() and inventory_path.is_file():
            sales_stat, inventory_stat = sales_path.stat(), inventory_path.stat()
            with st.spinner("Leyendo y validando los archivos Mendeley (se cachean por versión local)…"):
                dataset = cached_mendeley_local(
                    str(sales_path), sales_stat.st_size, sales_stat.st_mtime_ns,
                    str(inventory_path), inventory_stat.st_size, inventory_stat.st_mtime_ns,
                )
            dataset_key = hashlib.sha256(
                f"{sales_stat.st_size}:{sales_stat.st_mtime_ns}:{inventory_stat.st_size}:{inventory_stat.st_mtime_ns}".encode()
            ).hexdigest()
            origin = "MENDELEY REAL · CSV locales"
        elif mendeley_sales_upload is not None and mendeley_inventory_upload is not None:
            sales_bytes, inventory_bytes = mendeley_sales_upload.getvalue(), mendeley_inventory_upload.getvalue()
            with st.spinner("Leyendo y validando los dos CSV Mendeley…"):
                dataset = cached_mendeley_upload(
                    mendeley_sales_upload.name, sales_bytes,
                    mendeley_inventory_upload.name, inventory_bytes,
                )
            dataset_key = hashlib.sha256(sales_bytes + inventory_bytes).hexdigest()
            origin = "MENDELEY REAL · archivos cargados en esta sesión"
        else:
            st.info("Para Dataset Mendeley, coloca ambos CSV en data/raw/ o carga ventas e inventario desde el menú lateral.")
            st.stop()
        raw = dataset.sales
        clean, cleaning_report = dataset.sales, None
except (ValueError, ImportError, UnicodeDecodeError, pd.errors.ParserError,
        pd.errors.EmptyDataError, zipfile.BadZipFile, OSError) as exc:
    st.error(f"No se pudo leer o procesar el archivo: {exc}")
    st.stop()

if clean.empty:
    st.error("El archivo no contiene ventas válidas después del preprocesamiento. Revisa columnas, fechas, cantidades y precios.")
    st.stop()

if source == "Demostración":
    st.warning("DEMOSTRACIÓN: datos sintéticos generados localmente; no son resultados de ventas reales.")
elif dataset:
    st.success("Dataset Mendeley real seleccionado. Ventas positivas, devoluciones e inventario se tratan por separado.")
else:
    st.success("Datos cargados y procesados desde el archivo seleccionado.")

if section == "Panel principal":
    show_panel(clean, cleaning_report, origin, dataset)
elif section == "Carga de datos":
    show_data_page(raw, clean, cleaning_report, dataset)
elif section == "Modelos de demanda":
    show_models_page(clean, dataset_key)
elif section == "Clustering":
    show_clustering_page(clean)
elif section == "Alertas de inventario":
    show_inventory_page(clean, dataset)
elif section == "Informe":
    show_report_page(clean, origin, dataset_key, dataset)
