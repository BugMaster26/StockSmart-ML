from io import BytesIO

import pandas as pd
import pytest

from src.mendeley_data import load_mendeley_dataset


SALES_COLUMNS = [
    "Transaction Date", "Sales Type", "Is Return", "Reason of Return", "Supplier",
    "Product No", "Product Description", "Product Division", "Product Category",
    "Product Subcategory", "Product Segment", "Store", "Sales Channel", "Qty Sold",
    "Sales Amount", "Cogs", "Number of Transactions",
]
INVENTORY_COLUMNS = [
    "Start Date", "End Date", "Stock Status", "Supplier", "Product No",
    "Product Description", "Product Division", "Product Category", "Product Subcategory",
    "Product Segment", "Store", "Store Type", "Sales Channel", "Qty on hand",
    "Stocks Selling Amount", "Cost of Stocks", "Stock Unit Selling Price", "Stock Unit Cost Price",
]


def make_sales():
    rows = [
        ["2025-01-01", "Full Price", 0, None, "Vendor 1", "PROD-1", "Item", "Division", "Category", "Subcategory", "SEG-1", "STR-1", "Channel A", 2, 20.0, 12.0, 1],
        ["2025-01-01", "Full Price", 1, None, "Vendor 1", "PROD-1", "Item", "Division", "Category", "Subcategory", "SEG-1", "STR-1", "Channel A", -1, -10.0, -6.0, 1],
        ["2025-01-01", "Full Price", 0, None, "Vendor 1", "PROD-1", "Item", "Division", "Category", "Subcategory", "SEG-1", "STR-1", "Channel A", 0, 0.0, 0.0, 1],
    ]
    return pd.DataFrame(rows, columns=SALES_COLUMNS)


def make_inventory(end="9999-12-31", store="STR-1", stock=5):
    rows = [["2025-01-01", end, "Full Price", "Vendor 1", "PROD-1", "Item", "Division", "Category", "Subcategory", "SEG-1", store, "Retail Storefront", "Channel A", stock, stock * 10.0, stock * 6.0, 10.0, 6.0]]
    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def as_csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8")


def test_mendeley_mapping_separates_returns_and_preserves_reported_revenue():
    loaded = load_mendeley_dataset(BytesIO(as_csv_bytes(make_sales())), BytesIO(as_csv_bytes(make_inventory())))
    assert len(loaded.sales) == 1
    sale = loaded.sales.iloc[0]
    assert sale.StockCode == "PROD-1"
    assert sale.Store == "STR-1"
    assert sale.Quantity == 2
    assert sale.UnitPrice == 10
    assert sale.Revenue == 20
    assert loaded.quality["sales"]["rows"] == 3
    assert loaded.quality["sales"]["positive_sales_rows_retained"] == 1
    assert loaded.quality["sales"]["return_rows_excluded_from_demand"] == 1
    assert loaded.quality["sales"]["return_units_abs"] == 1
    assert loaded.quality["sales"]["zero_quantity_rows_excluded"] == 1
    assert loaded.quality["sales"]["invalid_or_unusable_rows"] == 0
    assert loaded.quality["inventory"]["open_ended_intervals"] == 1
    assert pd.isna(loaded.inventory.iloc[0].EndDate)
    assert loaded.inventory.iloc[0].EndDateOpen
    assert loaded.quality["relationship"]["positive_sales_rows_with_one_date_compatible_inventory_interval"] == 1


def test_mendeley_missing_required_column_has_explicit_error():
    incomplete = pd.DataFrame({"Transaction Date": ["2025-01-01"]})
    with pytest.raises(ValueError, match="Faltan columnas"):
        load_mendeley_dataset(as_csv_bytes(incomplete), as_csv_bytes(make_inventory()))


def test_inventory_keeps_negative_stock_and_reports_it_instead_of_clamping():
    loaded = load_mendeley_dataset(as_csv_bytes(make_sales()), as_csv_bytes(make_inventory(stock=-2)))
    assert loaded.inventory.iloc[0].Stock == -2
    assert loaded.quality["inventory"]["rows_with_negative_stock"] == 1


def test_invalid_interval_does_not_count_as_compatible_inventory():
    inventory = make_inventory(end="2024-12-31")
    loaded = load_mendeley_dataset(as_csv_bytes(make_sales()), as_csv_bytes(inventory))
    assert loaded.quality["inventory"]["invalid_or_reversed_intervals"] == 1
    assert loaded.quality["relationship"]["positive_sales_rows_without_date_compatible_inventory_interval"] == 1


def test_inventory_overlap_is_reported_for_ambiguous_pair_intervals():
    inventory = pd.concat([
        make_inventory(end="2025-01-05"),
        make_inventory(end="2025-01-10"),
    ], ignore_index=True)
    loaded = load_mendeley_dataset(as_csv_bytes(make_sales().iloc[[0]]), as_csv_bytes(inventory))
    assert loaded.quality["inventory"]["interval_overlap_rows_within_product_store_channel"] == 1
    assert loaded.quality["relationship"]["positive_sales_rows_with_multiple_date_compatible_inventory_intervals"] == 1
