import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


def create_model(classifier_name: str, args) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents=None,
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
    )
    
    if classifier_name == "logistic":
        classifier = LogisticRegression(
            solver="liblinear",
            C=4.0,
            random_state=args.seed,
            max_iter=2000,
        )
    elif classifier_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=args.seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def main() -> None:
    args = parse_args()

    x_train, y_train = load_xy(args.train)
    x_val, y_val = load_xy(args.val)
    x_test, y_test = load_xy(args.test)

    models = {
        "logistic_regression": create_model("logistic", args),
        "random_forest": create_model("random_forest", args),
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n=== Training {model_name.replace('_', ' ').title()} ===")
        
        model.fit(x_train, y_train)
        
        val_metrics = evaluate(model, x_val, y_val)
        test_metrics = evaluate(model, x_test, y_test)
        
        results[model_name] = {
            "params": {
                "seed": args.seed,
                "max_features": args.max_features,
                "ngram_max": args.ngram_max,
            },
            "val": val_metrics,
            "test": test_metrics,
        }
        
        # Save each model artifact
        Path("artifacts").mkdir(exist_ok=True)
        model_path = Path(f"artifacts/{model_name}_model.joblib")
        joblib.dump(model, model_path)
        print(f"Saved model: {model_path}")
        
        print(f"Validation - Precision: {val_metrics['precision']:.3f}, Recall: {val_metrics['recall']:.3f}, F1: {val_metrics['f1']:.3f}, ROC-AUC: {val_metrics['roc_auc']:.3f}")
        print(f"Test - Precision: {test_metrics['precision']:.3f}, Recall: {test_metrics['recall']:.3f}, F1: {test_metrics['f1']:.3f}, ROC-AUC: {test_metrics['roc_auc']:.3f}")

    # Save comparison report
    Path("reports").mkdir(exist_ok=True)
    with open("reports/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== COMPARISON SUMMARY ===")
    for model_name, result in results.items():
        test = result["test"]
        print(f"{model_name.replace('_', ' ').title()}:")
        print(f"  Test F1: {test['f1']:.3f} | Precision: {test['precision']:.3f} | Recall: {test['recall']:.3f} | ROC-AUC: {test['roc_auc']:.3f}")

    print(f"\nDetailed results saved to: reports/model_comparison.json")


if __name__ == "__main__":
    main()
