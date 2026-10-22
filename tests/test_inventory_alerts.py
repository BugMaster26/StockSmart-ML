import pandas as pd
import pytest

from src.inventory_alerts import ALERT_COLUMNS, alerts_to_csv, create_inventory_alerts


def make_sales():
    rows = []
    quantities = {
        "A": {0: 10, 1: 20, 2: 30, 3: 40, 4: 50},
        "B": {1: 1, 3: 3, 4: 4},  # La semana 2 sin ventas debe completarse con cero.
        "D": {2: 2, 3: 3, 4: 4},  # Solo tres semanas de historial.
    }
    for product, sales_by_week in quantities.items():
        for week, quantity in sales_by_week.items():
            rows.append({"StockCode": product, "InvoiceDate": pd.Timestamp("2024-01-01") +
                         pd.Timedelta(week * 7, unit="D"), "Quantity": quantity})
    return pd.DataFrame(rows)


def test_alerts_use_latest_calendar_weeks_and_mark_shortages():
    stock = pd.DataFrame({"StockCode": ["A", "B", "C", "D"], "Stock": [34, 2, 0, 100]})
    result = create_inventory_alerts(make_sales(), stock).set_index("StockCode")
    # A usa [20, 30, 40, 50], no la ventana anterior que omitía la última semana.
    assert result.loc["A", "Demanda estimada semanal"] == 35
    assert result.loc["A", "Alerta"] == "Posible insuficiencia"
    # B usa [1, 0, 3, 4], incluyendo la semana del calendario sin ventas.
    assert result.loc["B", "Demanda estimada semanal"] == 2
    assert result.loc["B", "Alerta"] == "Stock suficiente"  # La regla exige superar el stock.
    assert result.loc["C", "Alerta"] == "Sin datos suficientes"
    assert result.loc["D", "Alerta"] == "Sin datos suficientes"
    assert "cuatro semanas" in result.loc["D", "Método de estimación"]


def test_empty_sales_keep_stock_rows_as_insufficient_history():
    stock = pd.DataFrame({"StockCode": ["A", "B"], "Stock": [4, 0]})
    result = create_inventory_alerts(pd.DataFrame(), stock)
    assert len(result) == 2
    assert result["Alerta"].eq("Sin datos suficientes").all()
    assert result["Demanda estimada semanal"].isna().all()


@pytest.mark.parametrize("stock, message", [
    (pd.DataFrame({"StockCode": ["A"]}), "columnas"),
    (pd.DataFrame({"StockCode": ["A", "A"], "Stock": [1, 2]}), "duplicados"),
    (pd.DataFrame({"StockCode": ["A"], "Stock": [None]}), "Completa"),
    (pd.DataFrame({"StockCode": ["A"], "Stock": [-1]}), "negativo"),
    (pd.DataFrame({"StockCode": [" "], "Stock": [1]}), "código"),
    (pd.DataFrame({"StockCode": [], "Stock": []}), "al menos un producto"),
])
def test_invalid_stock_input_is_rejected(stock, message):
    with pytest.raises(ValueError, match=message):
        create_inventory_alerts(pd.DataFrame(), stock)


def test_csv_export_round_trips_alerts_in_utf8():
    sales = pd.DataFrame({"StockCode": ["SKU-ñ"] * 4,
                          "InvoiceDate": pd.date_range("2024-01-01", periods=4, freq="W-MON"),
                          "Quantity": [2.5] * 4})
    alerts = create_inventory_alerts(sales, pd.DataFrame({"StockCode": ["SKU-ñ"], "Stock": [1]}))
    payload = alerts_to_csv(alerts)
    assert payload.startswith(b"\xef\xbb\xbf")
    restored = pd.read_csv(pd.io.common.BytesIO(payload))
    assert restored.columns.tolist() == ALERT_COLUMNS
    assert restored.loc[0, "Alerta"] == "Posible insuficiencia"
    assert restored.loc[0, "StockCode"] == "SKU-ñ"
