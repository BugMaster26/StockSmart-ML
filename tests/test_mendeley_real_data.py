from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.clustering import segment_products
from src.inventory_alerts import create_observed_inventory_alerts
from src.mendeley_data import load_mendeley_dataset


ROOT = Path(__file__).resolve().parents[1]
SALES_PATH = ROOT / "data" / "raw" / "retail_sales_ml_apl.csv"
INVENTORY_PATH = ROOT / "data" / "raw" / "retail_inventory_ml_apl.csv"

pytestmark = pytest.mark.skipif(
    not (SALES_PATH.is_file() and INVENTORY_PATH.is_file()),
    reason="Los CSV originales Mendeley son locales y están excluidos de Git.",
)


def test_real_mendeley_files_have_verified_schema_period_and_join_coverage():
    dataset = load_mendeley_dataset(SALES_PATH, INVENTORY_PATH)
    sales, inventory, quality = dataset.sales, dataset.inventory, dataset.quality
    assert quality["sales"]["rows"] == 125_751
    assert quality["inventory"]["rows"] == 284_755
    assert quality["sales"]["duplicate_rows"] == 0
    assert quality["inventory"]["duplicate_rows"] == 0
    assert quality["sales"]["date_min"] == "2025-06-01"
    assert quality["sales"]["date_max"] == "2026-04-24"
    assert quality["inventory"]["open_ended_intervals"] == 22_701
    assert quality["sales"]["products"] == quality["inventory"]["products"] == 2_326
    assert quality["sales"]["stores"] == quality["inventory"]["stores"] == 40
    assert quality["sales"]["return_rows_excluded_from_demand"] == 7_491
    assert quality["sales"]["positive_sales_rows_retained"] == len(sales) == 118_258
    assert quality["inventory"]["interval_overlap_rows_within_product_store_channel"] == 0
    assert quality["relationship"]["positive_sales_rows_with_one_date_compatible_inventory_interval"] == 85_461
    assert quality["relationship"]["positive_sales_rows_with_multiple_date_compatible_inventory_intervals"] == 0
    assert quality["relationship"]["positive_sales_rows_without_date_compatible_inventory_interval"] == 32_797
    assert sales["Revenue"].equals(sales["SalesAmount"])
    assert (inventory["Stock"] < 0).sum() == 1_380


def test_real_dataset_runs_product_clustering_and_observed_alerts_without_false_stockout_claims():
    dataset = load_mendeley_dataset(SALES_PATH, INVENTORY_PATH)
    clusters = segment_products(dataset.sales, n_clusters=3)
    assert len(clusters["products"]) == 2_326
    assert clusters["silhouette_kmeans"] is not None

    alerts = create_observed_inventory_alerts(dataset.sales, dataset.inventory)
    assert not alerts.empty
    assert alerts["StockCode"].notna().all()
    assert not alerts["Alerta"].str.contains("Rotura de stock confirmada", case=False).any()
    assert alerts["Fecha de consulta"].max() == dataset.sales["InvoiceDate"].max().normalize()


def test_real_mendeley_source_renders_quality_and_inventory_pages():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=60).run()
    app.radio[0].set_value("Dataset Mendeley").run(timeout=60)
    assert not app.exception
    assert any("Dataset Mendeley real seleccionado" in item.value for item in app.success)

    app.radio[1].set_value("Carga de datos").run(timeout=60)
    assert not app.exception
    assert any("calidad, transformaciones" in item.value for item in app.subheader)

    app.radio[1].set_value("Alertas de inventario").run(timeout=60)
    assert not app.exception
    assert "Descargar alertas observadas CSV" in [item.label for item in app.get("download_button")]
