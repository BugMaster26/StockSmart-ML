import pandas as pd
import pytest

pytest.importorskip("sklearn")

from contextlib import nullcontext
from sklearn import get_config

import src.classification as classification
from src.classification import evaluate_models
from src.features import build_weekly_features


def make_history():
    dates = pd.date_range("2024-01-01", periods=260, freq="D")
    quantities = [1 + ((i // 7) % 9) for i in range(len(dates))]
    return pd.DataFrame({"InvoiceDate": dates, "StockCode": "SKU-1", "Quantity": quantities})


def test_models_are_evaluated_on_later_complete_weeks():
    features, _ = build_weekly_features(make_history())
    result = evaluate_models(features, test_fraction=.25)
    assert set(result["metrics"]["Modelo"]) == {"KNN", "Árbol de Decisión", "MLP"}
    assert result["train_last_week"] < result["test_first_week"]
    assert result["train_rows"] + result["test_rows"] == len(features)
    train_only = features.loc[features.Week < result["test_first_week"], "Demand"]
    expected_thresholds = train_only.quantile([.33, .67]).tolist()
    assert result["thresholds"]["bajo"] == pytest.approx(expected_thresholds[0])
    assert result["thresholds"]["alto"] == pytest.approx(expected_thresholds[1])
    assert {"Accuracy", "Precision macro", "Recall macro", "F1 macro",
            "Precision baja", "Recall media", "F1 alta"}.issubset(result["metrics"].columns)
    assert set(result["confusion"]) == {"KNN", "Árbol de Decisión", "MLP"}
    assert result["thresholds"]["bajo"] < result["thresholds"]["alto"]


def test_insufficient_weeks_returns_explanation():
    short = pd.DataFrame({"Week": pd.to_datetime(["2024-01-01", "2024-01-08"]),
                          "StockCode": ["A", "A"], "Demand": [2, 3], "Class": ["Baja", "Alta"],
                          "lag_1": [1, 2], "lag_2": [0, 1], "lag_3": [0, 0],
                          "mean_4": [1., 1.5], "std_4": [0., .5], "trend_4": [1, 1]})
    result = evaluate_models(short)
    assert result["selected"] is None
    assert "semanas" in result["warning"]


def test_knn_prediction_memory_limit_preserves_model_outputs(monkeypatch):
    features, _ = build_weekly_features(make_history())
    original_context = classification.config_context
    default_working_memory = get_config()["working_memory"]

    # Establish outputs from the same code path with sklearn's ambient default.
    monkeypatch.setattr(classification, "config_context", lambda **_: nullcontext())
    baseline = evaluate_models(features, test_fraction=.25)

    requested_limits = []

    def recording_context(**kwargs):
        requested_limits.append(kwargs)
        return original_context(**kwargs)

    monkeypatch.setattr(classification, "config_context", recording_context)
    bounded = evaluate_models(features, test_fraction=.25)

    assert requested_limits == [{"working_memory": classification.KNN_PREDICT_WORKING_MEMORY_MIB}]
    assert get_config()["working_memory"] == default_working_memory
    pd.testing.assert_frame_equal(baseline["metrics"], bounded["metrics"])
    for model_name in baseline["predictions"]:
        assert (baseline["predictions"][model_name] == bounded["predictions"][model_name]).all()
        pd.testing.assert_frame_equal(baseline["confusion"][model_name], bounded["confusion"][model_name])


def test_knn_prediction_failure_is_logged_with_original_exception(monkeypatch, caplog):
    features, _ = build_weekly_features(make_history())
    original_knn = classification.KNeighborsClassifier

    class FailingKNN(original_knn):
        def predict(self, values):
            raise RuntimeError("diagnostic prediction failure")

    monkeypatch.setattr(classification, "KNeighborsClassifier", FailingKNN)
    with pytest.raises(RuntimeError, match="diagnostic prediction failure"):
        evaluate_models(features, test_fraction=.25)

    assert "model=KNN stage=predict" in caplog.text
    assert "diagnostic prediction failure" in caplog.text
    assert "Traceback" in caplog.text
