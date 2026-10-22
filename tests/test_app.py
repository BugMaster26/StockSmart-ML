"""Pruebas de humo de la interfaz Streamlit y carga de formatos válidos."""
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


def make_valid_sales() -> pd.DataFrame:
    rows = []
    for product in range(4):
        for week in range(10):
            rows.append({
                "InvoiceNo": f"{product}-{week}",
                "StockCode": f"SKU-{product:02d}",
                "InvoiceDate": pd.Timestamp("2024-01-01") + pd.Timedelta(week * 7, unit="D"),
                "Quantity": 2 + product * 3 + (week % 3),
                "UnitPrice": 1.5 + product,
            })
    return pd.DataFrame(rows)


def run_app():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    return AppTest.from_file(app_path, default_timeout=30).run()


def test_demo_app_has_lateral_navigation_and_main_dashboard():
    at = run_app()
    assert not at.exception
    assert {radio.label for radio in at.radio} == {"Origen", "Navegación"}
    assert any("DEMOSTRACIÓN" in warning.value for warning in at.warning)
    dashboard_kpis = next(block.value for block in at.markdown if "<article class=\"dashboard-kpi" in block.value)
    assert all(label in dashboard_kpis for label in ("Ingresos", "Unidades vendidas", "Productos", "Registros válidos"))
    assert all(value in dashboard_kpis for value in ("207,736.60", "28,223", "12", "2,013"))


def test_demo_navigation_opens_models_clustering_and_inventory_pages():
    at = run_app()
    at.radio[1].set_value("Modelos de demanda").run(timeout=30)
    assert not at.exception
    assert any("Mayor F1 macro" in success.value for success in at.success)
    assert "Descargar métricas CSV" in [button.label for button in at.get("download_button")]

    at.radio[1].set_value("Informe").run(timeout=30)
    assert not at.exception
    assert any("Mejor F1 macro medido" in success.value for success in at.success)

    at.radio[1].set_value("Clustering").run(timeout=30)
    assert not at.exception
    assert any("K-Means" in subheader.value for subheader in at.subheader)

    at.radio[1].set_value("Alertas de inventario").run(timeout=30)
    assert not at.exception
    assert any("stock actual" in message.value.lower() for message in at.info)


def test_inventory_page_accepts_stock_csv_and_offers_alert_export():
    at = run_app()
    at.radio[1].set_value("Alertas de inventario").run(timeout=30)
    stock = pd.DataFrame({"StockCode": [f"DEMO-{i:03d}" for i in range(1, 13)], "Stock": [10] * 12})
    at.file_uploader[0].upload("stock.csv", stock.to_csv(index=False).encode("utf-8"), "text/csv").run(timeout=30)
    assert not at.exception
    labels = [button.label for button in at.get("download_button")]
    assert "Descargar alertas CSV" in labels


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_valid_uploaded_files_load_and_show_data_quality(extension):
    sales = make_valid_sales()
    sales.loc[0, "StockCode"] = "0007"
    if extension == "csv":
        content = sales.to_csv(index=False).encode("utf-8")
        mime = "text/csv"
    else:
        buffer = BytesIO()
        sales.to_excel(buffer, index=False)
        content = buffer.getvalue()
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    at = run_app()
    at.radio[0].set_value("Archivo propio").run(timeout=30)
    at.file_uploader[0].upload(f"ventas.{extension}", content, mime).run(timeout=30)
    assert not at.exception
    assert any("Datos cargados" in success.value for success in at.success)
    at.radio[1].set_value("Carga de datos").run(timeout=30)
    assert not at.exception
    assert any("Vista previa del archivo" in header.value for header in at.subheader)
    assert "0007" in at.dataframe[0].value.StockCode.tolist()
