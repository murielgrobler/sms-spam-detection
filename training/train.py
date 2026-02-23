import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("data/splits/val.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--C", type=float, default=4.0)
    parser.add_argument("--artifact_path", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--report_path", type=Path, default=Path("reports/metrics.json"))
    return parser.parse_args()


def load_xy(path: Path) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path)
    x = df["text"].astype(str).tolist()
    y = (df["label"].astype(str) == "spam").astype(int).tolist()
    return x, y


def evaluate(model: Pipeline, x: list[str], y: list[int]) -> dict:
    pred = model.predict(x)
    proba = model.predict_proba(x)[:, 1]

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def main() -> None:
    args = parse_args()

    x_train, y_train = load_xy(args.train)
    x_val, y_val = load_xy(args.val)
    x_test, y_test = load_xy(args.test)

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents=None,
                    ngram_range=(1, args.ngram_max),
                    max_features=args.max_features,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    C=args.C,
                    random_state=args.seed,
                    max_iter=2000,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    report = {
        "params": {
            "seed": args.seed,
            "max_features": args.max_features,
            "ngram_max": args.ngram_max,
            "C": args.C,
        },
        "val": evaluate(model, x_val, y_val),
        "test": evaluate(model, x_test, y_test),
    }

    args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.artifact_path)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Saved model: {args.artifact_path}")
    print(f"Saved report: {args.report_path}")
    print(json.dumps({"val": report["val"], "test": report["test"]}, indent=2))


if __name__ == "__main__":
    main()
