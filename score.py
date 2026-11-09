"""Batch-score a CSV using the frozen model and validation-selected threshold."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/model.joblib")
    args = parser.parse_args()
    saved = joblib.load(args.model)
    frame = pd.read_csv(args.input)
    with threadpool_limits(limits=2):
        probability = saved["pipeline"].predict_proba(frame)[:, 1]
    result = pd.DataFrame({"row": range(len(frame)), "response_probability": probability,
                           "above_validation_threshold": probability >= saved["threshold"]})
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
