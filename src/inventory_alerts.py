"""Alertas orientativas que contrastan stock actual con demanda reciente."""
from __future__ import annotations

import numpy as np
import pandas as pd

ALERT_COLUMNS = ["StockCode", "Stock", "Demanda estimada semanal", "Alerta", "Método de estimación"]
OBSERVED_ALERT_COLUMNS = [
    "StockCode", "Store", "SalesChannel", "Fecha de consulta",
    "Inicio del intervalo", "Fin del intervalo", "Existencia observada",
    "Demanda media semanal (4 semanas completas)", "Alerta", "Método de estimación",
]
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


def create_observed_inventory_alerts(
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
    weeks: int = 4,
) -> pd.DataFrame:
    """Compara demanda reciente por producto/tienda con un intervalo de stock compatible.

    Solo usa intervalos que cubren ``as_of`` y que coinciden en producto, tienda
    y canal. Si hay cero o varias filas de inventario para una clave, el stock
    no se compara. La demanda usa las cuatro semanas calendario completas
    anteriores a la semana de ``as_of``; sin ventas observadas en esa ventana
    no se interpreta la ausencia de filas como demanda cero.
    """
    sales_required = {"InvoiceDate", "StockCode", "Store", "SalesChannel", "Quantity"}
    inventory_required = {"StartDate", "EndDate", "StockCode", "Store", "SalesChannel", "Stock"}
    if missing := sales_required.difference(sales.columns):
        raise ValueError("Faltan columnas de ventas para alertas observadas: " + ", ".join(sorted(missing)))
    if missing := inventory_required.difference(inventory.columns):
        raise ValueError("Faltan columnas de inventario para alertas observadas: " + ", ".join(sorted(missing)))
    if weeks < 1:
        raise ValueError("weeks debe ser al menos 1.")

    history = sales[list(sales_required)].copy()
    history["InvoiceDate"] = pd.to_datetime(history["InvoiceDate"], errors="coerce")
    for column in ("StockCode", "Store", "SalesChannel"):
        history[column] = history[column].astype("string").str.strip()
    history["Quantity"] = pd.to_numeric(history["Quantity"], errors="coerce")
    history = history.dropna(subset=["InvoiceDate", "StockCode", "Store", "SalesChannel", "Quantity"])
    history = history.loc[
        np.isfinite(history["Quantity"].to_numpy(dtype=float))
        & history["Quantity"].gt(0)
        & history["StockCode"].ne("")
        & history["Store"].ne("")
        & history["SalesChannel"].ne("")
    ].copy()

    stock_rows = inventory[list(inventory_required)].copy()
    stock_rows["StartDate"] = pd.to_datetime(stock_rows["StartDate"], errors="coerce")
    stock_rows["EndDate"] = pd.to_datetime(stock_rows["EndDate"], errors="coerce")
    for column in ("StockCode", "Store", "SalesChannel"):
        stock_rows[column] = stock_rows[column].astype("string").str.strip()
    stock_rows["Stock"] = pd.to_numeric(stock_rows["Stock"], errors="coerce")
    stock_rows = stock_rows.dropna(subset=["StartDate", "StockCode", "Store", "SalesChannel", "Stock"])
    stock_rows = stock_rows.loc[
        np.isfinite(stock_rows["Stock"].to_numpy(dtype=float))
        & stock_rows["StockCode"].ne("")
        & stock_rows["Store"].ne("")
        & stock_rows["SalesChannel"].ne("")
        & (stock_rows["EndDate"].isna() | stock_rows["EndDate"].ge(stock_rows["StartDate"]))
    ].copy()

    if as_of is None:
        if history.empty or stock_rows.empty:
            return pd.DataFrame(columns=OBSERVED_ALERT_COLUMNS)
        check_date = min(history["InvoiceDate"].max().normalize(), stock_rows["StartDate"].max().normalize())
    else:
        check_date = pd.Timestamp(as_of).normalize()
        if stock_rows.empty:
            return pd.DataFrame(columns=OBSERVED_ALERT_COLUMNS)

    stock_rows["EndEffective"] = stock_rows["EndDate"].fillna(pd.Timestamp.max.normalize())
    active = stock_rows.loc[
        stock_rows["StartDate"].le(check_date) & stock_rows["EndEffective"].ge(check_date)
    ].copy()
    pair_keys = ["StockCode", "Store", "SalesChannel"]
    active_counts = active.groupby(pair_keys, dropna=False).size()
    unique_keys = active_counts[active_counts.eq(1)].reset_index()[pair_keys]
    ambiguous_keys = active_counts[active_counts.gt(1)].reset_index()[pair_keys]
    active_unique = active.merge(unique_keys, on=pair_keys, how="inner", validate="many_to_one")

    current_week_start = check_date.to_period("W-SUN").start_time
    window_end = current_week_start - pd.Timedelta(days=1)
    window_start = window_end - pd.Timedelta(days=7 * weeks - 1)
    recent = history.loc[history["InvoiceDate"].between(window_start, window_end)].copy()
    recent["Week"] = recent["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    weekly = recent.groupby([*pair_keys, "Week"], as_index=False)["Quantity"].sum()
    demand_rows = []
    full_weeks = pd.date_range(window_start, periods=weeks, freq="7D")
    for key, product_history in weekly.groupby(pair_keys, sort=False, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        demand = product_history.set_index("Week")["Quantity"].reindex(full_weeks, fill_value=0)
        demand_rows.append({**dict(zip(pair_keys, key_tuple)), "_demand": float(demand.mean())})
    demand_frame = pd.DataFrame(demand_rows, columns=[*pair_keys, "_demand"])

    observed_records = active_unique[[*pair_keys, "StartDate", "EndDate", "Stock"]].copy()
    observed_records["_inventory_state"] = "compatible"
    candidates = pd.concat([observed_records,], ignore_index=True)
    recent_keys = recent[pair_keys].drop_duplicates()
    missing_stock = recent_keys.merge(
        active_counts.rename("_active_count").reset_index(), on=pair_keys, how="left"
    )
    missing_stock = missing_stock[missing_stock["_active_count"].isna()].drop(columns="_active_count")
    if not missing_stock.empty:
        missing_stock["StartDate"] = pd.NaT
        missing_stock["EndDate"] = pd.NaT
        missing_stock["Stock"] = np.nan
        missing_stock["_inventory_state"] = "no_compatible_interval"
        candidates = pd.concat([candidates, missing_stock], ignore_index=True)
    if not ambiguous_keys.empty:
        ambiguous = recent_keys.merge(ambiguous_keys.assign(_ambiguous=True), on=pair_keys, how="inner")
        ambiguous = ambiguous[ambiguous["_ambiguous"]].drop(columns="_ambiguous").drop_duplicates(pair_keys)
        if not ambiguous.empty:
            ambiguous["StartDate"] = pd.NaT
            ambiguous["EndDate"] = pd.NaT
            ambiguous["Stock"] = np.nan
            ambiguous["_inventory_state"] = "ambiguous_intervals"
            candidates = pd.concat([candidates, ambiguous], ignore_index=True)

    candidates = candidates.drop_duplicates(pair_keys, keep="first")
    result = candidates.merge(demand_frame, on=pair_keys, how="left", validate="one_to_one")
    labels, methods = [], []
    for row in result.to_dict("records"):
        state = row["_inventory_state"]
        stock_value = row["Stock"]
        demand_value = row["_demand"]
        if state == "ambiguous_intervals":
            labels.append("Intervalos de inventario ambiguos; no comparado")
            methods.append("Varias filas cubren la clave y fecha; se requiere revisar la granularidad.")
        elif state == "no_compatible_interval":
            labels.append("Sin intervalo de inventario compatible")
            methods.append("No se compara stock: falta intervalo por producto, tienda, canal y fecha.")
        elif stock_value < 0:
            labels.append("Existencia negativa en el registro; revisar")
            methods.append("El valor negativo se conserva; no demuestra por sí solo una rotura de stock.")
        elif stock_value == 0:
            labels.append("Existencia observada en cero; revisar")
            methods.append("Cero registrado en el intervalo; no confirma una rotura de stock.")
        elif pd.isna(demand_value):
            labels.append("Sin ventas observadas en la ventana; sin estimación")
            methods.append("La ausencia de filas no se interpreta como demanda cero.")
        elif demand_value > stock_value:
            labels.append("Riesgo orientativo: demanda media supera existencia observada")
            methods.append(f"Media de {weeks} semanas completas ({window_start.date()} a {window_end.date()}); no es rotura confirmada.")
        else:
            labels.append("Existencia observada no inferior a la demanda media")
            methods.append(f"Media de {weeks} semanas completas ({window_start.date()} a {window_end.date()}); no es garantía de disponibilidad.")

    result["Fecha de consulta"] = check_date
    result["Inicio del intervalo"] = result["StartDate"]
    result["Fin del intervalo"] = result["EndDate"]
    result["Existencia observada"] = result["Stock"]
    result["Demanda media semanal (4 semanas completas)"] = pd.to_numeric(result["_demand"], errors="coerce").round(2)
    result["Alerta"] = labels
    result["Método de estimación"] = methods
    return result[OBSERVED_ALERT_COLUMNS].sort_values(pair_keys, kind="mergesort").reset_index(drop=True)


def alerts_to_csv(alerts: pd.DataFrame) -> bytes:
    """Serializa alertas para descarga CSV compatible con hojas de cálculo."""
    if "StockCode" not in alerts.columns or "Alerta" not in alerts.columns:
        raise ValueError("La tabla de alertas no tiene las columnas esperadas.")
    columns = ALERT_COLUMNS if set(ALERT_COLUMNS).issubset(alerts.columns) and len(alerts.columns) == len(ALERT_COLUMNS) else alerts.columns.tolist()
    return alerts[columns].to_csv(index=False).encode("utf-8-sig")
