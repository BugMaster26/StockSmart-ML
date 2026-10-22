"""Agregación semanal y variables predictoras sin información futura."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["lag_1", "lag_2", "lag_3", "mean_4", "std_4", "trend_4"]


def build_weekly_features(data: pd.DataFrame, low_quantile: float = .33, high_quantile: float = .67) -> tuple[pd.DataFrame, dict[str, float]]:
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("Los umbrales deben cumplir 0 ≤ bajo < alto ≤ 1.")
    required = {"InvoiceDate", "StockCode", "Quantity"}
    if not required.issubset(data.columns):
        raise ValueError(f"Se requieren columnas: {', '.join(sorted(required))}")
    frame = data.copy()
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame = frame.dropna(subset=["InvoiceDate", "StockCode", "Quantity"])
    frame = frame.loc[frame.Quantity > 0]
    if frame.empty:
        return pd.DataFrame(columns=["StockCode", "Week", *FEATURE_COLUMNS, "Demand", "Class"]), {"bajo": 0., "alto": 0.}
    weekly = (frame.assign(Week=frame.InvoiceDate.dt.to_period("W-SUN").dt.start_time)
              .groupby(["StockCode", "Week"], as_index=False).Quantity.sum()
              .rename(columns={"Quantity": "Demand"}))
    # Completar semanas sin ventas con cero para que los rezagos sean semanales.
    pieces = []
    for product, group in weekly.groupby("StockCode"):
        timeline = group.set_index("Week")["Demand"].sort_index()
        timeline = timeline.reindex(pd.date_range(timeline.index.min(), timeline.index.max(), freq="W-MON"), fill_value=0)
        part = pd.DataFrame({"StockCode": product, "Week": timeline.index, "Demand": timeline.values})
        # La fila de la semana t predice t+1. Al cierre de t, la venta de t
        # ya se conoce y es el rezago 1; nunca se usa información de t+1.
        observed = part.Demand
        part["lag_1"] = observed
        part["lag_2"] = observed.shift(1)
        part["lag_3"] = observed.shift(2)
        part["mean_4"] = observed.rolling(4).mean()
        part["std_4"] = observed.rolling(4).std().fillna(0)
        part["trend_4"] = part.lag_1 - part.lag_3
        part["Demand"] = part.Demand.shift(-1)
        pieces.append(part.dropna(subset=FEATURE_COLUMNS + ["Demand"]))
    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if result.empty:
        return pd.DataFrame(columns=["StockCode", "Week", *FEATURE_COLUMNS, "Demand", "Class"]), {"bajo": 0., "alto": 0.}
    low, high = result.Demand.quantile([low_quantile, high_quantile]).tolist()
    result["Class"] = np.where(result.Demand <= low, "Baja", np.where(result.Demand <= high, "Media", "Alta"))
    return result, {"bajo": float(low), "alto": float(high)}
