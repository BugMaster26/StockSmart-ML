"""Carga verificada de los CSV Retail Transactions and Stocks Data (Mendeley V1)."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd


SALES_COLUMNS = {
    "Transaction Date": "InvoiceDate",
    "Product No": "StockCode",
    "Store": "Store",
    "Sales Channel": "SalesChannel",
    "Sales Type": "SalesType",
    "Is Return": "IsReturn",
    "Qty Sold": "Quantity",
    "Sales Amount": "SalesAmount",
    "Cogs": "Cogs",
    "Number of Transactions": "TransactionCount",
    "Product Division": "ProductDivision",
    "Product Category": "ProductCategory",
    "Product Subcategory": "ProductSubcategory",
    "Product Segment": "ProductSegment",
    "Product Description": "ProductDescription",
}

INVENTORY_COLUMNS = {
    "Start Date": "StartDate",
    "End Date": "EndDate",
    "Stock Status": "StockStatus",
    "Product No": "StockCode",
    "Store": "Store",
    "Sales Channel": "SalesChannel",
    "Qty on hand": "Stock",
    "Stocks Selling Amount": "StockSellingAmount",
    "Cost of Stocks": "CostOfStocks",
    "Stock Unit Selling Price": "StockUnitSellingPrice",
    "Stock Unit Cost Price": "StockUnitCostPrice",
    "Product Division": "ProductDivision",
    "Product Category": "ProductCategory",
    "Product Subcategory": "ProductSubcategory",
    "Product Segment": "ProductSegment",
    "Product Description": "ProductDescription",
    "Supplier": "Supplier",
    "Store Type": "StoreType",
}

SALES_GRAIN = [
    "Transaction Date", "Product No", "Store", "Sales Channel", "Sales Type", "Is Return"
]
INVENTORY_GRAIN = [
    "Start Date", "End Date", "Product No", "Store", "Sales Channel", "Stock Status"
]
PAIR_KEYS = ["StockCode", "Store", "SalesChannel"]
SALES_NUMERIC = ["Qty Sold", "Sales Amount", "Cogs", "Number of Transactions", "Is Return"]
INVENTORY_NUMERIC = [
    "Qty on hand", "Stocks Selling Amount", "Cost of Stocks",
    "Stock Unit Selling Price", "Stock Unit Cost Price",
]


@dataclass
class MendeleyDataset:
    """Ventas positivas para análisis, intervalos originales normalizados e informe."""

    sales: pd.DataFrame
    inventory: pd.DataFrame
    quality: dict


def _source_and_name(source: str | Path | bytes | bytearray | BinaryIO) -> tuple[object, str, int | None]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró el archivo: {path}")
        return path, path.name, path.stat().st_size
    if isinstance(source, (bytes, bytearray)):
        payload = bytes(source)
        return BytesIO(payload), "archivo cargado", len(payload)
    if hasattr(source, "getvalue"):
        payload = source.getvalue()
        return BytesIO(payload), getattr(source, "name", "archivo cargado"), len(payload)
    return source, getattr(source, "name", "archivo cargado"), None


def _scan_csv(source, kind: str, chunksize: int) -> tuple[pd.DataFrame, dict]:
    columns_map = SALES_COLUMNS if kind == "sales" else INVENTORY_COLUMNS
    numeric_columns = SALES_NUMERIC if kind == "sales" else INVENTORY_NUMERIC
    key_columns = SALES_GRAIN if kind == "sales" else INVENTORY_GRAIN
    date_columns = ["Transaction Date"] if kind == "sales" else ["Start Date", "End Date"]
    reader_source, name, size = _source_and_name(source)
    missing = Counter()
    coercion_errors = Counter()
    value_counts: dict[str, Counter] = defaultdict(Counter)
    key_counts: Counter = Counter()
    row_hashes: set[int] = set()
    duplicate_rows = 0
    total_rows = 0
    source_columns: list[str] | None = None
    source_dtypes: dict[str, str] = {}
    selected_chunks: list[pd.DataFrame] = []
    numeric_summary: dict[str, dict] = {}
    date_summary: dict[str, dict] = {}
    description_id_mismatches = 0

    try:
        chunks = pd.read_csv(reader_source, chunksize=chunksize, low_memory=False)
        for chunk in chunks:
            if source_columns is None:
                source_columns = chunk.columns.tolist()
                required = set(columns_map).difference(source_columns)
                if required:
                    raise ValueError(
                        f"El CSV de {kind} no corresponde al archivo Mendeley esperado. "
                        f"Faltan columnas: {', '.join(sorted(required))}"
                    )
                source_dtypes = chunk.dtypes.astype(str).to_dict()
            total_rows += len(chunk)
            missing.update(chunk.isna().sum().to_dict())
            hashes = pd.util.hash_pandas_object(chunk, index=False).tolist()
            for row_hash in hashes:
                if row_hash in row_hashes:
                    duplicate_rows += 1
                else:
                    row_hashes.add(row_hash)
            key_counts.update(
                map(tuple, chunk[key_columns].astype("string").fillna("<NA>").itertuples(index=False, name=None))
            )

            for column in [
                "Sales Type", "Is Return", "Reason of Return", "Product Division",
                "Stock Status", "Store Type", "Sales Channel",
            ]:
                if column in chunk:
                    value_counts[column].update(
                        chunk[column].fillna("<NA>").astype(str).value_counts().to_dict()
                    )

            for column in numeric_columns:
                values = pd.to_numeric(chunk[column], errors="coerce")
                raw_nonmissing = chunk[column].notna()
                coercion_errors[column] += int((raw_nonmissing & values.isna()).sum())
                values = values.replace([np.inf, -np.inf], np.nan).dropna()
                current = numeric_summary.setdefault(
                    column, {"count": 0, "min": None, "max": None, "negative": 0, "zero": 0, "sum": 0.0}
                )
                current["count"] += len(values)
                if len(values):
                    current["min"] = float(values.min()) if current["min"] is None else min(current["min"], float(values.min()))
                    current["max"] = float(values.max()) if current["max"] is None else max(current["max"], float(values.max()))
                    current["negative"] += int((values < 0).sum())
                    current["zero"] += int((values == 0).sum())
                    current["sum"] += float(values.sum())

            for column in date_columns:
                raw = chunk[column].astype("string").str.strip()
                sentinel = raw.eq("9999-12-31") if column == "End Date" else pd.Series(False, index=raw.index)
                parsed = pd.to_datetime(raw.mask(sentinel), format="%Y-%m-%d", errors="coerce")
                current = date_summary.setdefault(
                    column, {"min": None, "max": None, "invalid": 0, "missing": 0, "open_ended": 0}
                )
                current["invalid"] += int((parsed.isna() & raw.notna() & ~sentinel).sum())
                current["missing"] += int(raw.isna().sum())
                current["open_ended"] += int(sentinel.sum())
                if parsed.notna().any():
                    low, high = parsed.min(), parsed.max()
                    current["min"] = low if current["min"] is None or low < current["min"] else current["min"]
                    current["max"] = high if current["max"] is None or high > current["max"] else current["max"]

            if "Product Description" in chunk:
                embedded = chunk["Product Description"].astype("string").str.extract(r"(PROD-\d+)", expand=False)
                description_id_mismatches += int((embedded.notna() & embedded.ne(chunk["Product No"].astype("string"))).sum())

            selected_chunks.append(chunk[[column for column in columns_map if column in chunk]].copy())
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"El archivo de {kind} está vacío o no tiene una cabecera CSV válida.") from exc

    if source_columns is None:
        raise ValueError(f"El archivo de {kind} está vacío.")
    selected = pd.concat(selected_chunks, ignore_index=True) if selected_chunks else pd.DataFrame()
    duplicates_by_key = sum(count - 1 for count in key_counts.values() if count > 1)
    quality = {
        "file": name,
        "size_bytes": size,
        "rows": total_rows,
        "source_columns": source_columns,
        "source_dtypes": source_dtypes,
        "columns_used": list(columns_map),
        "missing_by_column": {column: int(missing[column]) for column in source_columns if missing[column]},
        "duplicate_rows": duplicate_rows,
        "candidate_grain_columns": key_columns,
        "duplicate_candidate_grain_rows": duplicates_by_key,
        "distinct_candidate_grains": len(key_counts),
        "numeric_coercion_errors": {column: int(value) for column, value in coercion_errors.items() if value},
        "numeric_summary": {
            column: {
                **values,
                "mean": values["sum"] / values["count"] if values["count"] else None,
            }
            for column, values in numeric_summary.items()
        },
        "dates": {
            column: {
                **values,
                "min": values["min"].date().isoformat() if values["min"] is not None else None,
                "max": values["max"].date().isoformat() if values["max"] is not None else None,
            }
            for column, values in date_summary.items()
        },
        "categorical_counts": {column: dict(counts) for column, counts in value_counts.items()},
        "description_embedded_product_id_mismatches": description_id_mismatches,
    }
    return selected, quality


def _string_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("string").str.strip()


def _key_is_valid(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    valid = pd.Series(True, index=frame.index)
    for column in columns:
        valid &= frame[column].notna() & frame[column].ne("")
    return valid


def _interval_overlap_rows(inventory: pd.DataFrame) -> int:
    valid = inventory.dropna(subset=["StartDate", "StockCode", "Store", "SalesChannel"]).copy()
    if valid.empty:
        return 0
    valid["EndEffective"] = valid["EndDate"].fillna(pd.Timestamp.max.normalize())
    group_columns = ["StockCode", "Store", "SalesChannel"]
    valid = valid.sort_values(group_columns + ["StartDate", "EndEffective"], kind="mergesort")
    grouped = valid.groupby(group_columns, sort=False, dropna=False)
    cumulative_end = grouped["EndEffective"].cummax()
    prior_max_end = cumulative_end.groupby(
        [valid[column] for column in group_columns], sort=False, dropna=False
    ).shift()
    return int(valid["StartDate"].le(prior_max_end).sum())


def _relationship_quality(sales: pd.DataFrame, inventory: pd.DataFrame) -> dict:
    products_sales = set(sales["StockCode"].dropna().astype(str))
    products_inventory = set(inventory["StockCode"].dropna().astype(str))
    stores_sales = set(sales["Store"].dropna().astype(str))
    stores_inventory = set(inventory["Store"].dropna().astype(str))
    pairs_sales = set(map(tuple, sales[PAIR_KEYS].drop_duplicates().itertuples(index=False, name=None)))
    pairs_inventory = set(map(tuple, inventory[PAIR_KEYS].drop_duplicates().itertuples(index=False, name=None)))

    positive = sales[["InvoiceDate", *PAIR_KEYS, "SalesType"]].reset_index(names="_sale_row")
    valid_inventory = inventory.dropna(subset=["StartDate", "StockCode", "Store", "SalesChannel"]).copy()
    valid_inventory["EndEffective"] = valid_inventory["EndDate"].fillna(pd.Timestamp.max.normalize())
    candidates = positive.merge(
        valid_inventory[[*PAIR_KEYS, "StartDate", "EndEffective", "StockStatus"]],
        on=PAIR_KEYS,
        how="left",
        sort=False,
    )
    active = candidates[
        candidates["StartDate"].le(candidates["InvoiceDate"])
        & candidates["InvoiceDate"].le(candidates["EndEffective"])
    ]
    match_counts = active.groupby("_sale_row").size().reindex(positive["_sale_row"], fill_value=0)
    mismatched_status = active["SalesType"].ne(active["StockStatus"])
    return {
        "product_codes_in_both": len(products_sales & products_inventory),
        "sales_product_codes": len(products_sales),
        "inventory_product_codes": len(products_inventory),
        "stores_in_both": len(stores_sales & stores_inventory),
        "sales_stores": len(stores_sales),
        "inventory_stores": len(stores_inventory),
        "exact_product_store_channel_pairs_in_both": len(pairs_sales & pairs_inventory),
        "sales_product_store_channel_pairs": len(pairs_sales),
        "inventory_product_store_channel_pairs": len(pairs_inventory),
        "positive_sales_rows_with_one_date_compatible_inventory_interval": int(match_counts.eq(1).sum()),
        "positive_sales_rows_without_date_compatible_inventory_interval": int(match_counts.eq(0).sum()),
        "positive_sales_rows_with_multiple_date_compatible_inventory_intervals": int(match_counts.gt(1).sum()),
        "date_compatible_rows_with_status_mismatch": int(mismatched_status.sum()),
    }


def load_mendeley_dataset(
    sales_source: str | Path | bytes | bytearray | BinaryIO,
    inventory_source: str | Path | bytes | bytearray | BinaryIO,
    *,
    chunksize: int = 100_000,
) -> MendeleyDataset:
    """Carga los dos CSV V1 usando un mapeo explícito y devuelve datos auditables.

    Las filas de devolución se conservan en los totales de calidad, pero no se
    convierten en demanda positiva para los modelos. No se deduplican filas de
    distinta bandera de devolución: su grano declarado incluye esa bandera.
    """
    if chunksize < 1:
        raise ValueError("chunksize debe ser un entero positivo.")
    sales_source_frame, sales_quality = _scan_csv(sales_source, "sales", chunksize)
    inventory_source_frame, inventory_quality = _scan_csv(inventory_source, "inventory", chunksize)

    sales_raw = sales_source_frame.rename(columns=SALES_COLUMNS)
    sales = pd.DataFrame(index=sales_raw.index)
    sales["InvoiceDate"] = pd.to_datetime(sales_raw["InvoiceDate"], format="%Y-%m-%d", errors="coerce")
    for column in ["StockCode", "Store", "SalesChannel", "SalesType", "ProductDivision", "ProductCategory", "ProductSubcategory", "ProductSegment", "ProductDescription"]:
        sales[column] = _string_column(sales_raw, column)
    sales["Quantity"] = pd.to_numeric(sales_raw["Quantity"], errors="coerce")
    sales["SalesAmount"] = pd.to_numeric(sales_raw["SalesAmount"], errors="coerce")
    sales["Cogs"] = pd.to_numeric(sales_raw["Cogs"], errors="coerce")
    sales["TransactionCount"] = pd.to_numeric(sales_raw["TransactionCount"], errors="coerce")
    sales["IsReturn"] = pd.to_numeric(sales_raw["IsReturn"], errors="coerce")

    numeric_finite = (
        np.isfinite(sales["Quantity"].to_numpy(dtype=float, na_value=np.nan))
        & np.isfinite(sales["SalesAmount"].to_numpy(dtype=float, na_value=np.nan))
        & np.isfinite(sales["Cogs"].to_numpy(dtype=float, na_value=np.nan))
        & np.isfinite(sales["TransactionCount"].to_numpy(dtype=float, na_value=np.nan))
    )
    valid_keys = _key_is_valid(sales, ["StockCode", "Store", "SalesChannel"])
    valid_dates = sales["InvoiceDate"].notna()
    valid_flag = sales["IsReturn"].isin([0, 1])
    valid_rows = valid_keys & valid_dates & valid_flag & pd.Series(numeric_finite, index=sales.index)
    returns = valid_rows & sales["IsReturn"].eq(1) & sales["Quantity"].lt(0)
    positive_sales = valid_rows & sales["IsReturn"].eq(0) & sales["Quantity"].gt(0) & sales["SalesAmount"].ge(0)
    zero_quantity = valid_rows & sales["Quantity"].eq(0)
    mismatched_return_flag = valid_rows & (
        (sales["IsReturn"].eq(1) & sales["Quantity"].ge(0))
        | (sales["IsReturn"].eq(0) & sales["Quantity"].lt(0))
    )
    valid_rows = valid_rows & (returns | positive_sales | zero_quantity)
    usable_sales = sales.loc[positive_sales].copy()
    usable_sales["UnitPrice"] = usable_sales["SalesAmount"] / usable_sales["Quantity"]
    usable_sales["Revenue"] = usable_sales["SalesAmount"]
    usable_sales = usable_sales[[
        "InvoiceDate", "StockCode", "Store", "SalesChannel", "SalesType", "Quantity",
        "UnitPrice", "Revenue", "SalesAmount", "Cogs", "TransactionCount",
        "ProductDivision", "ProductCategory", "ProductSegment",
    ]].sort_values(["InvoiceDate", "StockCode", "Store"], kind="mergesort").reset_index(drop=True)

    inventory_raw = inventory_source_frame.rename(columns=INVENTORY_COLUMNS)
    inventory = pd.DataFrame(index=inventory_raw.index)
    inventory["StartDate"] = pd.to_datetime(inventory_raw["StartDate"], format="%Y-%m-%d", errors="coerce")
    end_raw = _string_column(inventory_raw, "EndDate")
    inventory["EndDateOpen"] = end_raw.eq("9999-12-31")
    inventory["EndDate"] = pd.to_datetime(end_raw.mask(inventory["EndDateOpen"]), format="%Y-%m-%d", errors="coerce")
    for column in ["StockCode", "Store", "SalesChannel", "StockStatus", "ProductDivision", "ProductCategory", "ProductSubcategory", "ProductSegment", "ProductDescription", "Supplier", "StoreType"]:
        inventory[column] = _string_column(inventory_raw, column)
    inventory["Stock"] = pd.to_numeric(inventory_raw["Stock"], errors="coerce")
    for column in ["StockSellingAmount", "CostOfStocks", "StockUnitSellingPrice", "StockUnitCostPrice"]:
        inventory[column] = pd.to_numeric(inventory_raw[column], errors="coerce")

    inventory_keys_valid = _key_is_valid(inventory, ["StockCode", "Store", "SalesChannel"])
    interval_order_valid = inventory["EndDateOpen"] | inventory["EndDate"].ge(inventory["StartDate"])
    interval_valid = inventory["StartDate"].notna() & interval_order_valid & inventory_keys_valid & inventory["Stock"].notna()
    inventory_bad_order = inventory["StartDate"].notna() & ~interval_order_valid
    inventory = inventory[[
        "StartDate", "EndDate", "EndDateOpen", "StockCode", "Store", "SalesChannel",
        "StockStatus", "Stock", "StockSellingAmount", "CostOfStocks",
        "StockUnitSellingPrice", "StockUnitCostPrice", "ProductDivision",
        "ProductCategory", "ProductSegment",
    ]]
    inventory_overlap_rows = _interval_overlap_rows(inventory.loc[interval_valid])
    relationship = _relationship_quality(usable_sales, inventory.loc[interval_valid])

    sales_quality.update({
        "date_min": sales["InvoiceDate"].min().date().isoformat() if sales["InvoiceDate"].notna().any() else None,
        "date_max": sales["InvoiceDate"].max().date().isoformat() if sales["InvoiceDate"].notna().any() else None,
        "products": int(sales.loc[valid_rows, "StockCode"].nunique()),
        "stores": int(sales.loc[valid_rows, "Store"].nunique()),
        "channels": int(sales.loc[valid_rows, "SalesChannel"].nunique()),
        "positive_sales_rows_retained": int(positive_sales.sum()),
        "return_rows_excluded_from_demand": int(returns.sum()),
        "return_units_abs": float(-sales.loc[returns, "Quantity"].sum()),
        "return_sales_amount": float(sales.loc[returns, "SalesAmount"].sum()),
        "gross_units_positive_sales": float(sales.loc[positive_sales, "Quantity"].sum()),
        "gross_sales_amount": float(sales.loc[positive_sales, "SalesAmount"].sum()),
        "net_units_reported": float(sales.loc[valid_rows, "Quantity"].sum()),
        "net_sales_amount_reported": float(sales.loc[valid_rows, "SalesAmount"].sum()),
        "zero_quantity_rows_excluded": int(zero_quantity.sum()),
        "return_flag_quantity_sign_mismatches": int(mismatched_return_flag.sum()),
        "invalid_or_unusable_rows": int((~valid_rows).sum()),
        "fractional_quantity_rows": int((sales.loc[valid_rows, "Quantity"] % 1 != 0).sum()),
        "description_embedded_product_id_mismatches": sales_quality["description_embedded_product_id_mismatches"],
    })
    inventory_quality.update({
        "date_min": inventory["StartDate"].min().date().isoformat() if inventory["StartDate"].notna().any() else None,
        "date_max": inventory["StartDate"].max().date().isoformat() if inventory["StartDate"].notna().any() else None,
        "products": int(inventory.loc[interval_valid, "StockCode"].nunique()),
        "stores": int(inventory.loc[interval_valid, "Store"].nunique()),
        "channels": int(inventory.loc[interval_valid, "SalesChannel"].nunique()),
        "open_ended_intervals": int(inventory["EndDateOpen"].sum()),
        "invalid_or_reversed_intervals": int(inventory_bad_order.sum() + inventory["StartDate"].isna().sum()),
        "rows_with_negative_stock": int((inventory["Stock"] < 0).sum()),
        "rows_with_zero_stock": int((inventory["Stock"] == 0).sum()),
        "rows_with_positive_stock": int((inventory["Stock"] > 0).sum()),
        "interval_overlap_rows_within_product_store_channel": inventory_overlap_rows,
        "valid_interval_rows": int(interval_valid.sum()),
    })
    quality = {
        "dataset": "Retail Transactions and Stocks Data, Mendeley Data V1",
        "doi": "10.17632/27x8mjm8k4.1",
        "license": "CC BY 4.0 (as stated by the dataset record)",
        "sales": sales_quality,
        "inventory": inventory_quality,
        "relationship": relationship,
        "limitations": [
            "La ficha califica el dataset como una muestra estratificada; cero filas observadas no prueban demanda real cero.",
            "La moneda de Sales Amount y las unidades físicas de Qty Sold / Qty on hand no están declaradas en los archivos.",
            "Start Date y End Date permiten comprobar pertenencia temporal a intervalos; la documentación no explica si Qty on hand es una medición al inicio, al final o constante durante todo el intervalo.",
            "Sales Type y Stock Status no coinciden en todas las filas con intervalo compatible; no se usan como claves de unión.",
            "Reason of Return está completamente vacío y Product Description contiene un ID PROD- distinto de Product No; no se usan para identificar productos.",
        ],
    }
    return MendeleyDataset(sales=usable_sales, inventory=inventory, quality=quality)
