import pandas as pd
from src.preprocessing import clean_transactions


def test_clean_excludes_cancellations_invalids_and_duplicates():
    data = pd.DataFrame([
        {"InvoiceNo": "1", "InvoiceDate": "2024-01-01", "StockCode": "A", "Quantity": 2, "UnitPrice": 3},
        {"InvoiceNo": "1", "InvoiceDate": "2024-01-01", "StockCode": "A", "Quantity": 2, "UnitPrice": 3},
        {"InvoiceNo": "C2", "InvoiceDate": "2024-01-02", "StockCode": "A", "Quantity": 1, "UnitPrice": 3},
        {"InvoiceNo": "3", "InvoiceDate": "2024-01-03", "StockCode": "A", "Quantity": -1, "UnitPrice": 3},
    ])
    clean, report = clean_transactions(data)
    assert len(clean) == 1
    assert clean.iloc[0].Revenue == 6
    assert report.duplicate_rows == 1
    assert report.cancellation_rows == 1
    assert report.negative_quantity_rows == 1


def test_missing_required_column_has_clear_error():
    try:
        clean_transactions(pd.DataFrame({"InvoiceDate": []}))
    except ValueError as exc:
        assert "StockCode" in str(exc)
    else:
        raise AssertionError("Expected required-column validation")


def test_non_finite_quantities_and_prices_are_reported_as_invalid():
    data = pd.DataFrame([
        {"InvoiceDate": "2024-01-01", "StockCode": "A", "Quantity": 2, "UnitPrice": 3},
        {"InvoiceDate": "2024-01-02", "StockCode": "B", "Quantity": float("inf"), "UnitPrice": 2},
        {"InvoiceDate": "2024-01-03", "StockCode": "C", "Quantity": 1, "UnitPrice": float("nan")},
    ])
    clean, report = clean_transactions(data)
    assert clean.StockCode.tolist() == ["A"]
    assert report.invalid_rows_removed == 2
