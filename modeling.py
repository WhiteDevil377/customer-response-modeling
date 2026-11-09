"""Reusable preprocessing for a pre-call response-modeling prototype."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]
NUMERIC = ["age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
           "cons.conf.idx", "euribor3m", "nr.employed"]
REQUIRED = NUMERIC + CATEGORICAL


class PreCallFeatures(TransformerMixin, BaseEstimator):
    """Exclude call duration and encode the never-contacted sentinel explicitly."""
    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(REQUIRED, dtype=object)
        return self

    def transform(self, X):
        missing = sorted(set(REQUIRED) - set(X.columns))
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
        out = X[REQUIRED].copy()
        for col in NUMERIC:
            out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
            if np.isinf(out[col]).any():
                raise ValueError("Infinite numeric value in " + col)
        for col in CATEGORICAL:
            out[col] = out[col].fillna("unknown").astype(str).str.strip()
        out["previously_contacted"] = (out["pdays"].notna() & out["pdays"].ne(999)).astype(float)
        out["pdays"] = out["pdays"].mask(out["pdays"].eq(999))
        return out


def make_pipeline(estimator):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC + ["previously_contacted"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("features", PreCallFeatures()), ("preprocess", preprocess),
                     ("model", estimator)])
