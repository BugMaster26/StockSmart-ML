import pandas as pd
from src.features import build_weekly_features


def test_features_are_lagged_and_labels_are_created():
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    data = pd.DataFrame({"InvoiceDate": dates, "StockCode": "A", "Quantity": range(1, 71)})
    features, thresholds = build_weekly_features(data)
    assert not features.empty
    assert {"lag_1", "mean_4", "Class"}.issubset(features.columns)
    assert features.iloc[0].lag_1 > 0
    assert thresholds["alto"] >= thresholds["bajo"]


def test_features_use_only_observed_weeks_before_target():
    # Un valor extremo en la semana objetivo no puede filtrarse a ninguna variable.
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    data = pd.DataFrame({"InvoiceDate": dates, "StockCode": "A", "Quantity": 1})
    baseline, _ = build_weekly_features(data)
    final_week = data.InvoiceDate.max().to_period("W-SUN")
    data.loc[data.InvoiceDate.dt.to_period("W-SUN") == final_week, "Quantity"] = 1000
    changed, _ = build_weekly_features(data)
    # Cambiar la demanda de la última semana solo altera el objetivo anterior
    # (que predice esa semana), nunca las variables predictoras de esa fila.
    row = baseline.index[-1]
    assert baseline.loc[row, "Demand"] != changed.loc[row, "Demand"]
    assert baseline.loc[row, ["lag_1", "lag_2", "lag_3", "mean_4", "std_4", "trend_4"]].equals(
        changed.loc[row, ["lag_1", "lag_2", "lag_3", "mean_4", "std_4", "trend_4"]]
    )


def test_training_row_lag_1_is_last_observed_week():
    dates = pd.date_range("2024-01-01", periods=84, freq="D")
    data = pd.DataFrame({"InvoiceDate": dates, "StockCode": "A", "Quantity": 2})
    features, _ = build_weekly_features(data)
    # Ventas semanales constantes; rezago inmediato debe representar la semana
    # anterior a la demanda objetivo.
    assert (features["lag_1"] == 14).all()
    assert (features["Demand"] == 14).all()
