"""Download UCI data, compare models on validation, then evaluate one frozen model."""
import argparse
import hashlib
import io
import json
import platform
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, roc_auc_score)
from threadpoolctl import threadpool_limits
from modeling import REQUIRED, make_pipeline

ROOT = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def load_data(archive):
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with urlopen(URL, timeout=60) as response:
            archive.write_bytes(response.read())
    content = archive.read_bytes()
    with ZipFile(io.BytesIO(content)) as outer:
        with ZipFile(io.BytesIO(outer.read("bank-additional.zip"))) as inner:
            name = next(n for n in inner.namelist() if n.endswith("bank-additional-full.csv"))
            frame = pd.read_csv(io.BytesIO(inner.read(name)), sep=";")
    return frame, hashlib.sha256(content).hexdigest()


def metrics(y, p):
    y = np.asarray(y)
    k = max(1, int(np.ceil(len(y) * .1)))
    top = np.argsort(-p, kind="stable")[:k]
    prevalence = float(y.mean())
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "positive_rate": prevalence,
        "precision_at_top_10_percent": float(y[top].mean()),
        "lift_at_top_10_percent": float(y[top].mean() / prevalence),
        "recall_at_top_10_percent": float(y[top].sum() / y.sum()),
        "n": len(y),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/bank-marketing.zip")
    args = parser.parse_args()
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    raw, sha = load_data(args.archive)
    if len(raw) != 41188 or not set(raw.y.unique()) <= {"yes", "no"}:
        raise ValueError("Unexpected dataset version or labels")
    # Preserve source order; discard only exact repeated records, not equal feature vectors.
    clean = raw.drop_duplicates(keep="first").copy()
    n_train, n_val_end = int(len(clean) * .70), int(len(clean) * .85)
    parts = {"train": clean.iloc[:n_train], "validation": clean.iloc[n_train:n_val_end],
             "test": clean.iloc[n_val_end:]}
    X = {name: frame[REQUIRED] for name, frame in parts.items()}
    y = {name: frame.y.eq("yes").astype(int) for name, frame in parts.items()}
    assert X["train"].index.max() < X["validation"].index.min()
    assert X["validation"].index.max() < X["test"].index.min()
    quality = {
        "source_url": URL, "archive_sha256": sha,
        "original_rows": len(raw), "exact_duplicates_removed": len(raw)-len(clean),
        "retained_rows": len(clean), "excluded_feature": "duration",
        "reason": "Call duration is unavailable before the call finishes.",
        "unknown_counts": {c: int(raw[c].eq("unknown").sum()) for c in raw.select_dtypes("object")},
        "pdays_999_count": int(raw.pdays.eq(999).sum()),
        "split_policy": "Source-order 70/15/15; UCI documents chronological source ordering.",
        "splits": {name: {"n": len(frame), "positive_rate": float(y[name].mean()),
                           "source_row_min": int(frame.index.min()),
                           "source_row_max": int(frame.index.max())}
                   for name, frame in parts.items()},
    }
    (out / "data_quality.json").write_text(json.dumps(quality, indent=2))
    estimators = {
        "logistic_regression": LogisticRegression(C=1.0, max_iter=2000, random_state=42),
        "gradient_boosting_15_leaves": HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, learning_rate=.08, l2_regularization=1.0,
            early_stopping=False, random_state=42),
        "gradient_boosting_31_leaves": HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=31, learning_rate=.08, l2_regularization=1.0,
            early_stopping=False, random_state=42),
    }
    leaderboard, fitted = {}, {}
    with threadpool_limits(limits=2):
        for name, estimator in estimators.items():
            model = make_pipeline(estimator)
            start = perf_counter()
            model.fit(X["train"], y["train"])
            seconds = perf_counter()-start
            p = model.predict_proba(X["validation"])[:, 1]
            leaderboard[name] = {**metrics(y["validation"], p), "fit_seconds": seconds}
            fitted[name] = model
            print(name, json.dumps(leaderboard[name]), flush=True)
        best = max(leaderboard, key=lambda name: leaderboard[name]["average_precision"])
        model = fitted[best]  # no test-guided selection and no refit before evaluation
        pval = model.predict_proba(X["validation"])[:, 1]
        precision, recall, thresholds = precision_recall_curve(y["validation"], pval)
        f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-12)
        threshold = float(thresholds[np.argmax(f1)])
        ptest = model.predict_proba(X["test"])[:, 1]
        result = metrics(y["test"], ptest)
        result["f1_at_validation_selected_threshold"] = float(f1_score(y["test"], ptest >= threshold))
        # Timings are local batch inference, not network or single-request latency.
        batch = X["test"].iloc[:1000]
        model.predict_proba(batch)
        timings = []
        for _ in range(10):
            start = perf_counter()
            model.predict_proba(batch)
            timings.append((perf_counter()-start)*1000)
    predictions = pd.DataFrame({"source_row": X["test"].index, "label": y["test"].to_numpy(),
                                "probability": ptest})
    predictions.to_csv(out / "test_predictions.csv", index=False)
    segmented = X["test"][["job", "contact", "month"]].copy()
    segmented["label"], segmented["probability"] = y["test"].to_numpy(), ptest
    segments = []
    for column in ["job", "contact", "month"]:
        for value, group in segmented.groupby(column):
            if len(group) >= 100 and group.label.nunique() == 2:
                segments.append({"feature": column, "value": value,
                                 **metrics(group.label, group.probability.to_numpy())})
    pd.DataFrame(segments).to_csv(out / "segment_metrics.csv", index=False)
    report = {
        "selected_model": best, "selection_metric": "validation average precision",
        "validation_leaderboard": leaderboard, "test": result,
        "constant_score_baseline_test_average_precision": float(y["test"].mean()),
        "validation_selected_threshold": threshold,
        "local_batch_inference": {"rows": len(batch), "repeats": len(timings),
                                  "median_ms": float(np.median(timings)),
                                  "p95_ms": float(np.percentile(timings, 95))},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
        "limitations": ["One historical, source-order holdout; no deployment or causal uplift claimed.",
                        "No client identifiers: repeated-client independence cannot be verified.",
                        "No exact timestamp column: source order approximates chronology.",
                        "Class prevalence and economic indicators shift between chronological blocks.",
                        "Predicted probabilities are not externally calibrated for current customers."]
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    joblib.dump({"pipeline": model, "threshold": threshold, "required_columns": REQUIRED},
                out / "model.joblib", compress=3)
    X["test"].iloc[:5].to_csv(ROOT / "example_customers.csv", index=False)
    print("SELECTED", best, "TEST", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
