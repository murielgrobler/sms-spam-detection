"""
Enhanced SMS spam detection training with TF-IDF + heuristic features.

This script combines traditional TF-IDF text features with engineered heuristic
features (link detection, money references, urgency indicators, etc.) to improve
spam detection performance.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Import shared functions from baseline training
import sys
sys.path.append('.')
from training.train import load_xy, evaluate
from features.tagging import extract_all_features, get_feature_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train enhanced SMS spam detection model")
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("data/splits/val.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument("--artifact_path", type=Path, default=Path("artifacts/enhanced_model.joblib"))
    parser.add_argument("--report_path", type=Path, default=Path("reports/enhanced_metrics.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--C", type=float, default=4.0)
    return parser.parse_args()


class EnhancedSpamPipeline:
    """Custom pipeline that combines TF-IDF and heuristic features for spam detection."""
    
    def __init__(self, tfidf_vectorizer, classifier):
        self.tfidf = tfidf_vectorizer
        self.classifier = classifier
        self.is_fitted = False
        
    def fit(self, texts, heuristic_features, y):
        """
        Fit the pipeline with texts and pre-computed heuristic features.
        
        Args:
            texts: List of text messages
            heuristic_features: Pre-computed heuristic features matrix
            y: Target labels
        """
        # Fit TF-IDF on texts
        tfidf_features = self.tfidf.fit_transform(texts)
        
        # Combine features
        combined_features = hstack([
            tfidf_features,
            csr_matrix(heuristic_features)
        ])
        
        # Fit classifier
        self.classifier.fit(combined_features, y)
        self.is_fitted = True
        return self
        
    def predict(self, texts, heuristic_features):
        """
        Predict with texts and pre-computed heuristic features.
        
        Args:
            texts: List of text messages
            heuristic_features: Pre-computed heuristic features matrix
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
            
        # Transform TF-IDF features
        tfidf_features = self.tfidf.transform(texts)
        
        # Combine features
        combined_features = hstack([
            tfidf_features,
            csr_matrix(heuristic_features)
        ])
        
        return self.classifier.predict(combined_features)
        
    def predict_proba(self, texts, heuristic_features):
        """
        Predict probabilities with texts and pre-computed heuristic features.
        
        Args:
            texts: List of text messages
            heuristic_features: Pre-computed heuristic features matrix
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
            
        # Transform TF-IDF features
        tfidf_features = self.tfidf.transform(texts)
        
        # Combine features
        combined_features = hstack([
            tfidf_features,
            csr_matrix(heuristic_features)
        ])
        
        return self.classifier.predict_proba(combined_features)


# No longer needed - feature extraction is now inside the model


def create_enhanced_pipeline(args) -> EnhancedSpamPipeline:
    """
    Create a pipeline that combines TF-IDF and heuristic features.
    
    Args:
        args: Command line arguments
        
    Returns:
        EnhancedSpamPipeline instance
    """
    # TF-IDF vectorizer for text features
    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents=None,
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
    )
    
    # Random Forest classifier
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=args.seed,
        n_jobs=-1,
    )
    
    return EnhancedSpamPipeline(tfidf, classifier)


def evaluate_enhanced_model(model, texts, heuristic_features, y_true):
    """Evaluate enhanced model with pre-computed heuristic features."""
    y_pred = model.predict(texts, heuristic_features)
    y_pred_proba = model.predict_proba(texts, heuristic_features)[:, 1]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def extract_heuristic_features_batch(texts):
    """Extract heuristic features for a batch of texts."""
    feature_names = get_feature_names()
    heuristic_features = []
    
    for text in texts:
        features = extract_all_features(text)
        feature_vector = [int(features[name]) for name in feature_names]
        heuristic_features.append(feature_vector)
    
    return np.array(heuristic_features, dtype=np.float32)


def main() -> None:
    args = parse_args()

    print("Loading data...")
    x_train, y_train = load_xy(args.train)
    x_val, y_val = load_xy(args.val)
    x_test, y_test = load_xy(args.test)

    print("Extracting heuristic features...")
    train_heuristic = extract_heuristic_features_batch(x_train)
    val_heuristic = extract_heuristic_features_batch(x_val)
    test_heuristic = extract_heuristic_features_batch(x_test)

    print(f"Training set: {len(x_train)} samples")
    print(f"Validation set: {len(x_val)} samples") 
    print(f"Test set: {len(x_test)} samples")
    print(f"Heuristic features: {len(get_feature_names())} features")
    print(f"Feature names: {get_feature_names()}")

    print("\nTraining enhanced model...")
    model = create_enhanced_pipeline(args)
    model.fit(x_train, train_heuristic, y_train)

    print("Evaluating model...")
    val_metrics = evaluate_enhanced_model(model, x_val, val_heuristic, y_val)
    test_metrics = evaluate_enhanced_model(model, x_test, test_heuristic, y_test)

    # Create report
    report = {
        "model_type": "enhanced_tfidf_random_forest",
        "features": {
            "tfidf": {
                "max_features": args.max_features,
                "ngram_range": [1, args.ngram_max]
            },
            "heuristic": {
                "count": len(get_feature_names()),
                "names": get_feature_names()
            }
        },
        "params": {
            "seed": args.seed,
            "max_features": args.max_features,
            "ngram_max": args.ngram_max,
            "C": args.C,
        },
        "val": val_metrics,
        "test": test_metrics,
    }

    # Save model
    args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.artifact_path)

    # Save report
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nSaved enhanced model: {args.artifact_path}")
    print(f"Saved report: {args.report_path}")
    
    print("\n=== ENHANCED MODEL RESULTS ===")
    print("Validation metrics:")
    print(f"  Precision: {val_metrics['precision']:.3f}")
    print(f"  Recall: {val_metrics['recall']:.3f}")
    print(f"  F1: {val_metrics['f1']:.3f}")
    print(f"  ROC-AUC: {val_metrics['roc_auc']:.3f}")
    
    print("Test metrics:")
    print(f"  Precision: {test_metrics['precision']:.3f}")
    print(f"  Recall: {test_metrics['recall']:.3f}")
    print(f"  F1: {test_metrics['f1']:.3f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
