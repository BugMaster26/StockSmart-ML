"""Evaluación temporal de modelos de clasificación de demanda."""
from __future__ import annotations
import logging
import time

import pandas as pd
from sklearn import config_context
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)
KNN_PREDICT_WORKING_MEMORY_MIB = 128


def evaluate_models(features: pd.DataFrame, test_fraction: float = .2,
                    low_quantile: float = .33, high_quantile: float = .67) -> dict:
    if not 0.1 <= test_fraction < .5:
        raise ValueError("La proporción de prueba debe estar entre 0.1 y 0.5.")
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("Los percentiles deben cumplir 0 ≤ bajo < alto ≤ 1.")
    if features.empty or features.Demand.nunique() < 2:
        return {"metrics": pd.DataFrame(), "predictions": pd.DataFrame(), "confusion": {}, "selected": None,
                "warning": "Se necesitan datos de varias clases para evaluar modelos."}
    logger.info(
        "Demand evaluation started: rows=%d columns=%d missing_cells=%d dtypes=%s",
        len(features), len(features.columns), int(features.isna().sum().sum()),
        {column: str(dtype) for column, dtype in features.dtypes.items()},
    )
    # Separar por semanas únicas evita que una misma semana aparezca en ambos lados.
    ordered = features.sort_values(["Week", "StockCode"]).reset_index(drop=True).copy()
    weeks = ordered["Week"].drop_duplicates().sort_values().tolist()
    week_cut = int(len(weeks) * (1 - test_fraction))
    if week_cut < 2 or week_cut >= len(weeks):
        return {"metrics": pd.DataFrame(), "predictions": pd.DataFrame(), "confusion": {}, "selected": None,
                "warning": "No hay suficientes semanas únicas para una división cronológica."}
    test_start = weeks[week_cut]
    train, test = ordered.loc[ordered.Week < test_start].copy(), ordered.loc[ordered.Week >= test_start].copy()
    if train.empty or test.empty:
        return {"metrics": pd.DataFrame(), "predictions": pd.DataFrame(), "confusion": {}, "selected": None,
                "warning": "No hay suficientes filas para separar entrenamiento y prueba."}
    # Los umbrales de clase también se estiman solo con el tramo de entrenamiento.
    low, high = train.Demand.quantile([low_quantile, high_quantile]).tolist()
    if high <= low:
        return {"metrics": pd.DataFrame(), "predictions": pd.DataFrame(), "confusion": {}, "selected": None,
                "warning": "La demanda del entrenamiento no permite separar los umbrales de clase."}
    def assign_class(frame):
        return pd.Series(pd.cut(frame.Demand, [-float("inf"), low, high, float("inf")],
                                labels=["Baja", "Media", "Alta"], include_lowest=True).astype(str), index=frame.index)
    train["Class"] = assign_class(train)
    test["Class"] = assign_class(test)
    if train.Class.nunique() < 2:
        return {"metrics": pd.DataFrame(), "predictions": pd.DataFrame(), "confusion": {}, "selected": None,
                "warning": "El tramo de entrenamiento contiene una sola clase; no se pueden comparar clasificadores."}
    logger.info(
        "Demand evaluation split ready: train_rows=%d test_rows=%d train_last_week=%s "
        "test_first_week=%s train_class_counts=%s test_class_counts=%s",
        len(train), len(test), train.Week.max(), test.Week.min(),
        {str(label): int(count) for label, count in train.Class.value_counts().items()},
        {str(label): int(count) for label, count in test.Class.value_counts().items()},
    )
    categorical = ["StockCode"]
    numeric = [c for c in ("lag_1", "lag_2", "lag_3", "mean_4", "std_4", "trend_4") if c in features]
    prep = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])
    models = {"KNN": KNeighborsClassifier(n_neighbors=min(5, len(train))),
              "Árbol de Decisión": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
              "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1200, tol=1e-3,
                                   early_stopping=False, random_state=42)}
    rows, predictions, matrices = [], {}, {}
    labels = ["Baja", "Media", "Alta"]
    for name, estimator in models.items():
        pipe = make_pipeline(prep, estimator)
        started = time.perf_counter()
        logger.info("Demand model stage started: model=%s stage=fit train_rows=%d features=%d",
                    name, len(train), len(categorical + numeric))
        try:
            pipe.fit(train[categorical + numeric], train.Class)
        except Exception:
            logger.exception("Demand model stage failed: model=%s stage=fit", name)
            raise
        logger.info("Demand model stage completed: model=%s stage=fit seconds=%.2f",
                    name, time.perf_counter() - started)

        started = time.perf_counter()
        if name == "KNN":
            logger.info(
                "Demand model stage started: model=KNN stage=predict test_rows=%d "
                "working_memory_mib=%d",
                len(test), KNN_PREDICT_WORKING_MEMORY_MIB,
            )
        else:
            logger.info("Demand model stage started: model=%s stage=predict test_rows=%d",
                        name, len(test))
        try:
            if name == "KNN":
                # KNN's brute-force distance calculation otherwise inherits sklearn's
                # 1 GiB default temporary working-memory target. Chunking bounds the
                # distance workspace without changing the estimator or its predictions.
                with config_context(working_memory=KNN_PREDICT_WORKING_MEMORY_MIB):
                    predicted = pipe.predict(test[categorical + numeric])
            else:
                predicted = pipe.predict(test[categorical + numeric])
        except Exception:
            logger.exception("Demand model stage failed: model=%s stage=predict", name)
            raise
        logger.info("Demand model stage completed: model=%s stage=predict seconds=%.2f",
                    name, time.perf_counter() - started)

        try:
            p, r, f, _ = precision_recall_fscore_support(test.Class, predicted, labels=labels, zero_division=0)
        except Exception:
            logger.exception("Demand evaluation stage failed: model=%s stage=metrics", name)
            raise
        rows.append({"Modelo": name, "Accuracy": accuracy_score(test.Class, predicted),
                     "Precision macro": p.mean(), "Recall macro": r.mean(), "F1 macro": f.mean(),
                     **{metric + " " + label.lower(): score
                        for metric, scores in (("Precision", p), ("Recall", r), ("F1", f))
                        for label, score in zip(labels, scores)}})
        predictions[name] = predicted
        try:
            matrices[name] = pd.DataFrame(confusion_matrix(test.Class, predicted, labels=labels), index=labels, columns=labels)
        except Exception:
            logger.exception("Demand evaluation stage failed: model=%s stage=confusion_matrix", name)
            raise
    metrics = pd.DataFrame(rows).sort_values("F1 macro", ascending=False).reset_index(drop=True)
    class_counts = train.Class.value_counts().to_dict()
    warning = "Evaluación orientativa: hay clases con menos de 10 muestras en entrenamiento." if any(class_counts.get(c, 0) < 10 for c in labels) else None
    return {"metrics": metrics, "predictions": predictions, "confusion": matrices,
            "selected": str(metrics.iloc[0]["Modelo"]), "warning": warning,
            "thresholds": {"bajo": float(low), "alto": float(high)},
            "train_last_week": train.Week.max(), "test_first_week": test.Week.min(),
            "train_rows": len(train), "test_rows": len(test),
            "test_index": test.index, "test_data": test, "estimators": models}
