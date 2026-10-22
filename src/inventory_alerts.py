"""Alertas orientativas que contrastan stock actual con demanda reciente."""
from __future__ import annotations

import numpy as np
import pandas as pd

ALERT_COLUMNS = ["StockCode", "Stock", "Demanda estimada semanal", "Alerta", "Método de estimación"]
STOCK_COLUMNS = {"StockCode", "Stock"}
SALES_COLUMNS = {"InvoiceDate", "StockCode", "Quantity"}


def validate_stock(stock: pd.DataFrame) -> pd.DataFrame:
    """Valida stock ingresado/importado; no completa valores ausentes con cero."""
    missing = STOCK_COLUMNS.difference(stock.columns)
    if missing:
        raise ValueError("El stock debe incluir las columnas StockCode y Stock.")
    result = stock[["StockCode", "Stock"]].copy()
    result["StockCode"] = result["StockCode"].astype("string").str.strip()
    if result.empty:
        raise ValueError("Ingresa al menos un producto con su stock actual.")
    if result.StockCode.isna().any() or result.StockCode.eq("").any():
        raise ValueError("Cada fila debe incluir un código de producto.")
    if result.StockCode.duplicated().any():
        duplicated = result.loc[result.StockCode.duplicated(keep=False), "StockCode"].unique().tolist()
        raise ValueError("Hay códigos de producto duplicados: " + ", ".join(duplicated))
    result["Stock"] = pd.to_numeric(result["Stock"], errors="coerce")
    if result.Stock.isna().any() or not np.isfinite(result.Stock.to_numpy(dtype=float)).all():
        raise ValueError("Completa todos los valores de stock con cantidades numéricas finitas.")
    if (result.Stock < 0).any():
        raise ValueError("El stock actual no puede ser negativo.")
    return result.reset_index(drop=True)


def create_inventory_alerts(sales: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    """Compara stock actual con la media de las cuatro últimas semanas observadas.

    Se estima directamente desde las transacciones limpias hasta la última semana
    presente en el archivo. Las semanas sin venta se completan con cero. Un SKU
    necesita abarcar al menos cuatro semanas para recibir una estimación.
    """
    stock_validated = validate_stock(stock)
    estimates = pd.DataFrame({"StockCode": pd.Series(dtype="string"),
                              "Demanda estimada semanal": pd.Series(dtype="float64")})
    if not sales.empty:
        missing = SALES_COLUMNS.difference(sales.columns)
        if missing:
            raise ValueError("Faltan columnas de ventas para estimar demanda: " + ", ".join(sorted(missing)))
        history = sales[["InvoiceDate", "StockCode", "Quantity"]].copy()
        history["InvoiceDate"] = pd.to_datetime(history.InvoiceDate, errors="coerce")
        history["StockCode"] = history.StockCode.astype("string").str.strip()
        history["Quantity"] = pd.to_numeric(history.Quantity, errors="coerce")
        history = history.dropna(subset=["InvoiceDate", "StockCode", "Quantity"])
        history = history.loc[np.isfinite(history.Quantity) & (history.Quantity > 0) & history.StockCode.ne("")].copy()
        if not history.empty:
            history["Week"] = history.InvoiceDate.dt.to_period("W-SUN").dt.start_time
            latest_week = history.Week.max()
            weekly = history.groupby(["StockCode", "Week"], as_index=False).Quantity.sum()
            rows = []
            for product, product_history in weekly.groupby("StockCode", sort=False):
                first_week = product_history.Week.min()
                weeks = pd.date_range(first_week, latest_week, freq="W-MON")
                demand = product_history.set_index("Week").Quantity.reindex(weeks, fill_value=0)
                if len(demand) >= 4:
                    rows.append({"StockCode": product, "Demanda estimada semanal": float(demand.tail(4).mean())})
            estimates = pd.DataFrame(rows, columns=["StockCode", "Demanda estimada semanal"])
    result = stock_validated.merge(estimates, on="StockCode", how="left", validate="one_to_one")
    estimate = result["Demanda estimada semanal"].to_numpy(dtype=float, na_value=np.nan)
    invalid_estimate = np.isfinite(estimate) & (estimate < 0)
    if invalid_estimate.any():
        raise ValueError("El historial contiene una estimación de demanda negativa.")
    result["Demanda estimada semanal"] = result["Demanda estimada semanal"].where(np.isfinite(estimate)).round(2)
    enough = result["Demanda estimada semanal"].notna()
    result["Alerta"] = np.where(
        ~enough, "Sin datos suficientes",
        np.where(result["Demanda estimada semanal"] > result.Stock, "Posible insuficiencia", "Stock suficiente"),
    )
    result["Método de estimación"] = np.where(
        enough, "Media de las últimas 4 semanas disponibles", "Sin cuatro semanas válidas de historial"
    )
    return result[ALERT_COLUMNS]


def alerts_to_csv(alerts: pd.DataFrame) -> bytes:
    """Serializa alertas para descarga CSV compatible con hojas de cálculo."""
    missing = set(ALERT_COLUMNS).difference(alerts.columns)
    if missing:
        raise ValueError("La tabla de alertas no tiene las columnas esperadas.")
    return alerts[ALERT_COLUMNS].to_csv(index=False).encode("utf-8-sig")
