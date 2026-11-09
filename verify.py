"""Meaningful checks of leakage exclusions and frozen-model scoring behavior."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from modeling import NUMERIC, REQUIRED

ROOT = Path(__file__).resolve().parent
saved = joblib.load(ROOT / "artifacts/model.joblib")
model = saved["pipeline"]
frame = pd.read_csv(ROOT / "example_customers.csv")
with threadpool_limits(limits=2):
    before = model.predict_proba(frame)[:, 1]
    assert np.isfinite(before).all() and ((before >= 0) & (before <= 1)).all()
    altered = frame.assign(duration=999999, y="yes")
    np.testing.assert_allclose(before, model.predict_proba(altered)[:, 1])
    unknown = frame.copy()
    unknown["job"] = "never_seen_job"
    assert np.isfinite(model.predict_proba(unknown)).all()
    try:
        model.predict_proba(frame.drop(columns="age"))
    except ValueError as exc:
        assert "age" in str(exc)
    else:
        raise AssertionError("Missing required input was accepted")
assert "duration" not in REQUIRED
report = json.loads((ROOT / "artifacts/data_quality.json").read_text())
splits = report["splits"]
assert splits["train"]["source_row_max"] < splits["validation"]["source_row_min"]
assert splits["validation"]["source_row_max"] < splits["test"]["source_row_min"]
print("PASS: probabilities, duration exclusion, unknown category, schema, chronological blocks")
