"""Segmentación de productos con K-Means y agrupamiento jerárquico Ward."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = ["units", "revenue", "orders", "active_weeks", "weekly_mean", "weekly_std"]


def product_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Construye una fila por producto; media/desviación usan totales semanales."""
    required = {"InvoiceDate", "StockCode", "Quantity", "UnitPrice"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Faltan columnas para segmentar: " + ", ".join(sorted(missing)))
    frame = data.copy()
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame["UnitPrice"] = pd.to_numeric(frame["UnitPrice"], errors="coerce")
    frame = frame.dropna(subset=["InvoiceDate", "StockCode", "Quantity", "UnitPrice"])
    frame = frame.loc[(frame.Quantity > 0) & (frame.UnitPrice >= 0)].copy()
    if frame.empty:
        raise ValueError("No hay ventas válidas para resumir productos.")
    frame["Week"] = frame["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    frame["Revenue"] = frame["Quantity"] * frame["UnitPrice"]
    order_aggregation = ("InvoiceNo", "nunique") if "InvoiceNo" in frame else ("Quantity", "size")
    summary = frame.groupby("StockCode", as_index=False).agg(
        units=("Quantity", "sum"), revenue=("Revenue", "sum"),
        orders=order_aggregation, active_weeks=("Week", "nunique"))
    weekly = (frame.groupby(["StockCode", "Week"], as_index=False).Quantity.sum()
              .groupby("StockCode").Quantity.agg(weekly_mean="mean", weekly_std="std")
              .fillna(0).reset_index())
    summary = summary.merge(weekly, on="StockCode", how="left")
    summary[CLUSTER_FEATURES] = summary[CLUSTER_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    return summary.sort_values("StockCode").reset_index(drop=True)


def _silhouette_or_none(values: np.ndarray, labels: np.ndarray) -> float | None:
    groups = np.unique(labels)
    if 1 < len(groups) < len(labels):
        return float(silhouette_score(values, labels))
    return None


def _linkage_from_agglomerative(model: AgglomerativeClustering, n_samples: int) -> np.ndarray:
    """Converts the fitted sklearn Ward tree to SciPy linkage format."""
    children = model.children_
    counts = np.zeros(children.shape[0], dtype=float)
    for node_index, (left, right) in enumerate(children):
        left_count = 1 if left < n_samples else counts[left - n_samples]
        right_count = 1 if right < n_samples else counts[right - n_samples]
        counts[node_index] = left_count + right_count
    return np.column_stack([children, model.distances_, counts]).astype(float)


def segment_products(data: pd.DataFrame, n_clusters: int = 3, *, random_state: int = 42) -> dict:
    """Agrupa todos los productos con variables estandarizadas y devuelve perfiles.

    El número de grupos debe estar entre 2 y n_productos - 1 para que la
    segmentación y el índice silhouette sean interpretables.
    """
    summary = product_summary(data)
    n_products = len(summary)
    if not isinstance(n_clusters, int) or n_clusters < 2 or n_clusters >= n_products:
        raise ValueError(f"El número de grupos debe ser entero entre 2 y {n_products - 1}.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(summary[CLUSTER_FEATURES])
    distinct_profiles = np.unique(scaled, axis=0).shape[0]
    if distinct_profiles < n_clusters:
        raise ValueError(f"Solo hay {distinct_profiles} perfiles de producto distintos; reduce el número de grupos.")

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(scaled)
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", compute_distances=True)
    hierarchical_labels = hierarchical.fit_predict(scaled)

    summary["K-Means"] = kmeans_labels
    summary["Jerárquico"] = hierarchical_labels
    profile_kmeans = summary.groupby("K-Means")[CLUSTER_FEATURES].mean().round(2)
    profile_hierarchical = summary.groupby("Jerárquico")[CLUSTER_FEATURES].mean().round(2)
    linkage_matrix = _linkage_from_agglomerative(hierarchical, n_products)
    silhouette_kmeans = _silhouette_or_none(scaled, kmeans_labels)
    silhouette_hierarchical = _silhouette_or_none(scaled, hierarchical_labels)

    return {
        "products": summary,
        "profile_kmeans": profile_kmeans,
        "profile_hierarchical": profile_hierarchical,
        # Mantiene compatibilidad con la interfaz anterior.
        "profile": profile_kmeans,
        "silhouette_kmeans": silhouette_kmeans,
        "silhouette_hierarchical": silhouette_hierarchical,
        "silhouette": silhouette_kmeans,
        "scaled": scaled,
        "feature_names": CLUSTER_FEATURES.copy(),
        "linkage_matrix": linkage_matrix,
        "scaler": scaler,
        "n_clusters": n_clusters,
    }


def draw_dendrogram(result: dict, *, max_leaves: int = 30, ax=None) -> dict:
    """Dibuja el dendrograma; para catálogos amplios resume las ramas finales."""
    if max_leaves < 2:
        raise ValueError("max_leaves debe ser al menos 2.")
    labels = result["products"]["StockCode"].astype(str).tolist()
    truncated = len(labels) > max_leaves
    options = {"truncate_mode": "lastp", "p": max_leaves, "show_leaf_counts": True} if truncated else {}
    return dendrogram(result["linkage_matrix"], labels=labels, ax=ax,
                      leaf_rotation=70, **options)
