#!/usr/bin/env python3
"""
Model Evaluation Script for Voice Recognition System v2.
Runs 5-fold cross-validation on all models and generates comparison report.
"""

import os
import sys
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_FILES, CLASSES, SAMPLE_RATE, DATASET_PATH,
    FEATURE_SIZE, REPORT_DIR
)
from src.feature_extraction import extract_features

# Ensure report directory exists
os.makedirs(REPORT_DIR, exist_ok=True)


def load_dataset():
    """Load all audio files and extract features."""
    print("Loading dataset...")
    X, y = [], []

    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_dir):
            print(f"  Warning: {class_dir} not found, skipping.")
            continue

        files = [f for f in os.listdir(class_dir)
                 if f.endswith('.wav') and f.startswith(class_name)]

        for fname in files:
            fpath = os.path.join(class_dir, fname)
            try:
                features = extract_features(fpath, apply_noise_reduction=False)
                if features is not None and len(features) == FEATURE_SIZE:
                    X.append(features)
                    y.append(class_name)
            except Exception as e:
                print(f"  Error processing {fname}: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"  Loaded {len(X)} samples, shape: {X.shape}")
    return X, y


def load_models():
    """Load all trained models."""
    print("\nLoading models...")

    models = {}
    for name, path in MODEL_FILES.items():
        if name in ('scaler', 'encoder'):
            continue
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"  Loaded {name}: {path}")
        else:
            print(f"  Warning: {name} not found at {path}")

    scaler = joblib.load(MODEL_FILES['scaler']) if os.path.exists(MODEL_FILES['scaler']) else None
    encoder = joblib.load(MODEL_FILES['encoder']) if os.path.exists(MODEL_FILES['encoder']) else None

    print(f"  Loaded scaler: {scaler is not None}")
    print(f"  Loaded encoder: {encoder is not None}")

    return models, scaler, encoder


def get_ensemble_model(models, scaler, X_scaled, y):
    """Build and return the ensemble model (voting classifier)."""
    from sklearn.ensemble import VotingClassifier

    estimators = [(name, models[name]) for name in ['svm', 'rf', 'xgb', 'mlp'] if name in models]

    if len(estimators) < 2:
        print("  Warning: Not enough base models for ensemble")
        return None

    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )
    ensemble.fit(X_scaled, y)
    return ensemble


def cross_validate_model(model, X, y, model_name, cv=5):
    """Run stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Clone model for each fold
        from sklearn.base import clone
        fold_model = clone(model)

        # Fit on training fold
        fold_model.fit(X_train, y_train)

        # Predict on validation fold
        y_pred = fold_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        scores.append(acc)

    return np.mean(scores), np.std(scores)


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Ensemble Model - Confusion Matrix', fontsize=14, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to: {save_path}")


def print_text_confusion_matrix(y_true, y_pred, classes):
    """Print confusion matrix in text format."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    n_classes = len(classes)

    # Header
    max_label_len = max(len(c) for c in classes)
    col_width = max(6, max_label_len + 2)

    header = "True\\Pred".ljust(col_width) + "".join(c.ljust(col_width) for c in classes)
    separator = "-" * (col_width * (n_classes + 1))

    print("\nEnsemble Confusion Matrix (Text):")
    print(separator)
    print(header)
    print(separator)

    for i, row_label in enumerate(classes):
        row = row_label.ljust(col_width) + "".join(str(cm[i, j]).ljust(col_width) for j in range(n_classes))
        print(row)

    print(separator + "\n")


def print_comparison_table(results):
    """Print formatted comparison table."""
    # results: list of (model_name, cv_mean, cv_std, test_acc)
    header = f"{'Model':<12} {'CV Accuracy':<18} {'Test Accuracy':<15}"
    separator = "-" * 50

    print("\n" + "=" * 50)
    print("  Model Comparison Summary")
    print("=" * 50)
    print(header)
    print(separator)

    for model_name, cv_mean, cv_std, test_acc in results:
        cv_str = f"{cv_mean:.4f} ± {cv_std:.4f}"
        test_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
        print(f"{model_name:<12} {cv_str:<18} {test_str:<15}")

    print(separator + "\n")


def main():
    print("=" * 60)
    print("  Voice Recognition System v2 - Model Evaluation")
    print("=" * 60)

    # Load data
    X, y = load_dataset()
    if len(X) == 0:
        print("Error: No data loaded. Exiting.")
        sys.exit(1)

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    classes = encoder.classes_.tolist()

    # Load models
    models, scaler, encoder = load_models()

    if scaler is None or encoder is None:
        print("Error: Scaler or encoder not found. Exiting.")
        sys.exit(1)

    if not models:
        print("Error: No models found. Exiting.")
        sys.exit(1)

    # Scale features
    print("\nScaling features...")
    X_scaled = scaler.transform(X)

    # Train ensemble on full dataset for confusion matrix
    print("\nTraining ensemble on full dataset...")
    ensemble = get_ensemble_model(models, scaler, X_scaled, y_encoded)

    if ensemble is None:
        print("Error: Could not build ensemble. Exiting.")
        sys.exit(1)

    # Cross-validation for each model
    print("\nRunning 5-fold cross-validation...")
    cv_results = []

    for model_name in ['svm', 'rf', 'xgb', 'mlp']:
        if model_name in models:
            print(f"  Evaluating {model_name}...", end=" ", flush=True)
            mean_acc, std_acc = cross_validate_model(models[model_name], X_scaled, y_encoded, model_name)
            print(f"{mean_acc:.4f} ± {std_acc:.4f}")
            cv_results.append((model_name.upper(), mean_acc, std_acc))

    # Ensemble cross-validation
    print("  Evaluating ENSEMBLE...", end=" ", flush=True)
    ensemble_mean, ensemble_std = cross_validate_model(ensemble, X_scaled, y_encoded, 'ensemble')
    print(f"{ensemble_mean:.4f} ± {ensemble_std:.4f}")
    cv_results.append(('ENSEMBLE', ensemble_mean, ensemble_std))

    # Generate predictions for confusion matrix
    print("\nGenerating confusion matrix...")
    y_pred = ensemble.predict(X_scaled)

    # Convert encoded labels back to original
    y_true_labels = encoder.inverse_transform(y_encoded)
    y_pred_labels = encoder.inverse_transform(y_pred)

    # Print text confusion matrix
    print_text_confusion_matrix(y_true_labels, y_pred_labels, classes)

    # Save confusion matrix plot
    cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_true_labels, y_pred_labels, classes, cm_path)

    # Compute test accuracy (full dataset)
    test_acc = accuracy_score(y_encoded, y_pred)

    # Build results with test accuracy (same as CV for full dataset)
    final_results = [(name, cv_mean, cv_std, test_acc) for name, cv_mean, cv_std in cv_results]

    # Print comparison table
    print_comparison_table(final_results)

    # Save results to file
    results_path = os.path.join(REPORT_DIR, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write("Voice Recognition System v2 - Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Features: {FEATURE_SIZE}\n\n")
        f.write("Cross-Validation Results (5-fold Stratified):\n")
        f.write("-" * 50 + "\n")
        for name, cv_mean, cv_std in cv_results:
            f.write(f"  {name:<12}: {cv_mean:.4f} ± {cv_std:.4f}\n")
        f.write(f"\nFull Dataset Accuracy: {test_acc:.4f}\n")

    print(f"Results saved to: {results_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()