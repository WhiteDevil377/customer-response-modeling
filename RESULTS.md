# Measured experiment results

Generated on 9 November 2026. These are historical offline results from the bundled implementation, not production outcomes.

## Data quality and splitting

Original rows: 41,188. Exact duplicates removed: 12. Retained rows: 41,176.

| Block | Rows | Positive rate | Source row range |
|---|---:|---:|---|
| train | 28,823 | 5.575% | 0 to 28831 |
| validation | 6,176 | 10.638% | 28832 to 35008 |
| test | 6,177 | 38.449% | 35009 to 41187 |

Preprocessing is fitted only on the first block. Post-call duration is excluded. Exact duplicate removal preserves source order.

## Validation comparison

| Model configuration | Average precision | ROC AUC | Brier score |
|---|---:|---:|---:|
| logistic_regression | 0.1991 | 0.6601 | 0.0917 |
| gradient_boosting_15_leaves | 0.1507 | 0.5927 | 0.0979 |
| gradient_boosting_31_leaves | 0.1442 | 0.5814 | 0.0987 |

Selected using validation average precision: **logistic_regression**. The same fitted model is then evaluated on the final block; no candidate is selected using test metrics.

## Final ordered holdout

| Metric | Value |
|---|---:|
| n | 6,177 |
| average_precision | 0.4464 |
| roc_auc | 0.5747 |
| brier_score | 0.2762 |
| positive_rate | 0.3845 |
| precision_at_top_10_percent | 0.4676 |
| lift_at_top_10_percent | 1.2163 |
| recall_at_top_10_percent | 0.1217 |
| f1_at_validation_selected_threshold | 0.5118 |

Constant-score average-precision reference: **0.3845**. Validation-selected decision threshold: **0.1481**.

The ranking result is modest. The later period has a much higher positive rate than the training block, and the saved probabilities have substantial error there. This does not support a production-readiness or causal business-impact claim. Compare class prevalence, inspect segment_metrics.csv, and study rolling evaluation or calibration as future work.

## Local inference measurement

Batch: 1,000 rows; repeats: 10; median: 6.03 ms; p95: 6.40 ms. This excludes model loading and network latency and is not a per-request benchmark.

## Versions

- python: 3.12.14
- numpy: 2.3.5
- pandas: 2.2.3
- scipy: 1.18.1
- scikit-learn: 1.9.1
- joblib: 1.6.0

## Source

Moro, S., Rita, P., and Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306. CC BY 4.0.

Archive SHA-256: `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4`.
