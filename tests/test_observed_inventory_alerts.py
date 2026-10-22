import pandas as pd

from src.inventory_alerts import OBSERVED_ALERT_COLUMNS, alerts_to_csv, create_observed_inventory_alerts


def make_sales():
    rows = []
    for week, quantity in enumerate([10, 20, 30, 40]):
        rows.append({
            "InvoiceDate": pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * week),
            "StockCode": "A", "Store": "S1", "SalesChannel": "C1", "Quantity": quantity,
        })
    # Sales at another store must not contribute to S1 demand.
    for week in range(4):
        rows.append({
            "InvoiceDate": pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * week),
            "StockCode": "A", "Store": "S2", "SalesChannel": "C1", "Quantity": 100,
        })
    return pd.DataFrame(rows)


def make_inventory():
    return pd.DataFrame([
        {"StartDate": "2024-01-01", "EndDate": None, "StockCode": "A", "Store": "S1", "SalesChannel": "C1", "Stock": 15},
        {"StartDate": "2024-01-01", "EndDate": "2024-01-28", "StockCode": "A", "Store": "S2", "SalesChannel": "C1", "Stock": 500},
        {"StartDate": "2024-01-01", "EndDate": None, "StockCode": "B", "Store": "S1", "SalesChannel": "C1", "Stock": 4},
        {"StartDate": "2024-01-01", "EndDate": None, "StockCode": "C", "Store": "S1", "SalesChannel": "C1", "Stock": 0},
        {"StartDate": "2024-01-01", "EndDate": None, "StockCode": "D", "Store": "S1", "SalesChannel": "C1", "Stock": -1},
    ])


def test_observed_alert_uses_product_store_channel_and_complete_weeks():
    result = create_observed_inventory_alerts(make_sales(), make_inventory(), as_of="2024-02-02")
    a_s1 = result.query("StockCode == 'A' and Store == 'S1'").iloc[0]
    a_s2 = result.query("StockCode == 'A' and Store == 'S2'").iloc[0]
    assert a_s1["Demanda media semanal (4 semanas completas)"] == 25
    assert "supera existencia" in a_s1["Alerta"]
    assert pd.isna(a_s2["Existencia observada"])
    assert a_s2["Alerta"] == "Sin intervalo de inventario compatible"
    assert a_s1["Fecha de consulta"] == pd.Timestamp("2024-02-02")


def test_zero_and_negative_observed_stock_are_flags_not_confirmed_stockouts():
    result = create_observed_inventory_alerts(make_sales(), make_inventory(), as_of="2024-02-02").set_index("StockCode")
    assert result.loc["C", "Alerta"] == "Existencia observada en cero; revisar"
    assert result.loc["D", "Alerta"] == "Existencia negativa en el registro; revisar"
    assert "no demuestra" in result.loc["D", "Método de estimación"]


def test_multiple_active_intervals_are_not_compared_and_csv_export_keeps_detail():
    inventory = pd.concat([
        make_inventory().iloc[[0]],
        make_inventory().iloc[[0]],
    ], ignore_index=True)
    inventory["StartDate"] = "2024-01-01"
    result = create_observed_inventory_alerts(make_sales(), inventory, as_of="2024-02-02")
    row = result.iloc[0]
    assert pd.isna(row["Existencia observada"])
    assert row["Alerta"] == "Intervalos de inventario ambiguos; no comparado"
    assert result.columns.tolist() == OBSERVED_ALERT_COLUMNS
    payload = alerts_to_csv(result)
    assert payload.startswith(b"\xef\xbb\xbf")
    restored = pd.read_csv(pd.io.common.BytesIO(payload))
    assert restored.columns.tolist() == OBSERVED_ALERT_COLUMNS


def test_no_sales_in_window_is_not_silently_treated_as_zero_demand():
    sales = make_sales().iloc[0:0]
    result = create_observed_inventory_alerts(sales, make_inventory(), as_of="2024-02-02")
    assert not result.empty
    assert result["Demanda media semanal (4 semanas completas)"].isna().all()
    positive_stock = result["Existencia observada"].gt(0)
    assert result.loc[positive_stock, "Alerta"].eq("Sin ventas observadas en la ventana; sin estimación").all()
    assert not result["Alerta"].str.contains("suficiente", case=False).any()
