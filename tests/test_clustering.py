import numpy as np
import pandas as pd
import pytest

from src.clustering import CLUSTER_FEATURES, product_summary, segment_products, draw_dendrogram


def make_products(n_products=8, n_weeks=6):
    rows = []
    for product in range(n_products):
        for week in range(n_weeks):
            units = (product + 1) * (week + 1) if product < n_products // 2 else (product + 1) * (n_weeks - week)
            rows.append({
                "InvoiceNo": f"{product}-{week}",
                "StockCode": f"SKU-{product:02d}",
                "InvoiceDate": pd.Timestamp("2024-01-01") + pd.Timedelta(week * 7, unit="D"),
                "Quantity": units,
                "UnitPrice": product + 1.0,
            })
    return pd.DataFrame(rows)


def test_summary_has_real_weekly_statistics_and_product_rows():
    summary = product_summary(make_products())
    assert len(summary) == 8
    assert set(CLUSTER_FEATURES).issubset(summary.columns)
    assert (summary.active_weeks == 6).all()
    assert (summary.weekly_mean > 0).all()
    assert (summary.weekly_std > 0).all()


def test_both_algorithms_use_scaled_features_and_return_profiles():
    result = segment_products(make_products(), n_clusters=3)
    products = result["products"]
    assert set(products["K-Means"].unique()) == {0, 1, 2}
    assert set(products["Jerárquico"].unique()) == {0, 1, 2}
    assert result["scaled"].shape == (8, len(CLUSTER_FEATURES))
    assert np.allclose(result["scaled"].mean(axis=0), 0, atol=1e-10)
    assert set(result["profile_kmeans"].index) == {0, 1, 2}
    assert set(result["profile_hierarchical"].index) == {0, 1, 2}
    assert result["silhouette_kmeans"] is not None
    assert result["silhouette_hierarchical"] is not None
    assert result["linkage_matrix"].shape == (7, 4)


def test_dendrogram_can_be_drawn_and_truncated():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = segment_products(make_products(), n_clusters=2)
    fig, ax = plt.subplots()
    tree = draw_dendrogram(result, max_leaves=4, ax=ax)
    plt.close(fig)
    assert "icoord" in tree and "dcoord" in tree


@pytest.mark.parametrize("n_clusters", [1, 8, 9, 2.5])
def test_invalid_group_count_is_rejected(n_clusters):
    with pytest.raises(ValueError, match="número de grupos"):
        segment_products(make_products(), n_clusters=n_clusters)


def test_not_enough_valid_products_has_clear_error():
    data = make_products(n_products=1)
    with pytest.raises(ValueError, match="número de grupos"):
        segment_products(data, n_clusters=2)


def test_group_count_cannot_exceed_distinct_scaled_profiles():
    data = make_products(n_products=4)
    # Make all product-level variables identical while keeping separate codes.
    data["Quantity"] = 5
    data["UnitPrice"] = 2
    with pytest.raises(ValueError, match="perfiles de producto distintos"):
        segment_products(data, n_clusters=2)
