"""Validación y limpieza de transacciones de venta."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("InvoiceDate", "StockCode", "Quantity", "UnitPrice")


@dataclass
class CleaningReport:
    rows_input: int
    rows_output: int
    missing_by_column: dict[str, int]
    duplicate_rows: int
    cancellation_rows: int
    negative_quantity_rows: int
    invalid_rows_removed: int


def validate_columns(data: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(missing))


def clean_transactions(data: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Normaliza tipos, elimina filas inválidas y excluye cancelaciones/devoluciones.

    Facturas cuyo identificador empieza por C y cantidades <= 0 no representan
    demanda vendida. Se informan antes de excluirlas para conservar trazabilidad.
    """
    validate_columns(data)
    frame = data.copy()
    missing = frame.isna().sum().astype(int).to_dict()
    duplicates = int(frame.duplicated().sum())
    invoice = frame["InvoiceNo"].astype("string") if "InvoiceNo" in frame else pd.Series("", index=frame.index, dtype="string")
    cancellation = invoice.str.upper().str.startswith("C", na=False)
    quantity = pd.to_numeric(frame["Quantity"], errors="coerce")
    price = pd.to_numeric(frame["UnitPrice"], errors="coerce")
    finite_values = pd.Series(np.isfinite(quantity), index=frame.index) & pd.Series(np.isfinite(price), index=frame.index)
    cancellation_count = int(cancellation.sum())
    negative_count = int((quantity < 0).sum())
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame["Quantity"] = quantity
    frame["UnitPrice"] = price
    invalid = (frame[list(REQUIRED_COLUMNS)].isna().any(axis=1) | ~finite_values
               | frame["StockCode"].astype("string").str.strip().eq(""))
    invalid_count = int(invalid.sum())
    frame = frame.loc[~invalid & ~cancellation & (frame["Quantity"] > 0) & (frame["UnitPrice"] >= 0)].copy()
    frame["StockCode"] = frame["StockCode"].astype(str).str.strip()
    frame["Revenue"] = frame["Quantity"] * frame["UnitPrice"]
    frame = frame.drop_duplicates().sort_values("InvoiceDate").reset_index(drop=True)
    return frame, CleaningReport(len(data), len(frame), missing, duplicates, cancellation_count, negative_count, invalid_count)


def make_demo_data(seed: int = 42) -> pd.DataFrame:
    """Genera transacciones ficticias reproducibles, identificadas como demo."""
    rng = np.random.default_rng(seed)
    # Fecha final fija para que la demo y sus métricas sean reproducibles.
    dates = pd.date_range(end=pd.Timestamp("2024-12-29"), periods=24 * 7, freq="D")
    rows = []
    for product_idx in range(1, 13):
        product = f"DEMO-{product_idx:03d}"
        base = 3 + product_idx * 1.7
        for date in dates:
            qty = int(rng.poisson(base * (1 + 0.18 * np.sin(date.dayofyear / 18))))
            if qty:
                rows.append({"InvoiceNo": f"D{len(rows):07d}", "StockCode": product,
                             "Description": f"Producto demostración {product_idx}", "Quantity": qty,
                             "InvoiceDate": date + pd.Timedelta(int(rng.integers(8, 20)), unit="h"),
                             "UnitPrice": round(float(1 + product_idx * .8), 2), "Country": "DEMO"})
    return pd.DataFrame(rows)
