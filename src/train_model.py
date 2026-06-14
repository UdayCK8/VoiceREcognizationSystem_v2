import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

from src.config import (
    DATASET_PATH, MODELS_DIR, CLASSES, SAMPLE_RATE,
    FEATURE_SIZE, TEST_SIZE, RANDOM_STATE, MODEL_FILES
)

from src.feature_extraction import extract_features, extract_features_from_audio
from src.augmentation import augment_sample


# ─────────────────────────────
# DATA LOADING
# ─────────────────────────────
def load_dataset(augment=True):
    X, y = [], []

    print("\nLoading dataset...")

    for label in CLASSES:
        folder = os.path.join(DATASET_PATH, label)

        if not os.path.exists(folder):
            print(f"[WARN] Missing: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        print(f"{label:>10}: {len(files)} files")

        for f in files:
            path = os.path.join(folder, f)

            # original
            X.append(extract_features(path))
            y.append(label)

            # augmentation
            if augment:
                try:
                    audio, sr = librosa.load(path, sr=SAMPLE_RATE)
                    for aug in augment_sample(audio, sr, n_augmentations=2):
                        X.append(extract_features_from_audio(aug, sr))
                        y.append(label)
                except:
                    pass

    return np.array(X), np.array(y)


# ─────────────────────────────
# TRAIN FUNCTION
# ─────────────────────────────
def train_model(name, grid, X_train, y_train, X_test, y_test):
    print(f"\n--- {name} ---")

    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    pred = model.predict(X_test)

    print("Best:", grid.best_params_)
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, zero_division=0))

    return model


# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    print("=" * 60)
    print(" VOICE AI TRAINING PIPELINE (UPGRADED)")
    print("=" * 60)

    X, y = load_dataset()

    print(f"\nSamples: {len(y)} | Features: {FEATURE_SIZE}")

    # encode
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # models
    svm = train_model("SVM",
        GridSearchCV(SVC(probability=True), {"C": [1, 10]}, cv=5),
        X_train, y_train, X_test, y_test
    )

    rf = train_model("RF",
        GridSearchCV(RandomForestClassifier(), {"n_estimators": [100, 200]}, cv=5),
        X_train, y_train, X_test, y_test
    )

    xgb = train_model("XGB",
        GridSearchCV(XGBClassifier(eval_metric="mlogloss"), {"max_depth": [3, 5]}, cv=5),
        X_train, y_train, X_test, y_test
    )

    mlp = train_model("MLP",
        GridSearchCV(MLPClassifier(max_iter=400), {"hidden_layer_sizes": [(128,), (256,)]}, cv=5),
        X_train, y_train, X_test, y_test
    )

    # ensemble
    ensemble = VotingClassifier(
        estimators=[("svm", svm), ("rf", rf), ("xgb", xgb), ("mlp", mlp)],
        voting="soft"
    )

    ensemble.fit(X_train, y_train)

    pred = ensemble.predict(X_test)

    print("\nENSEMBLE ACC:", accuracy_score(y_test, pred))
    print(confusion_matrix(y_test, pred))

    # save
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(svm, MODEL_FILES["svm"])
    joblib.dump(rf, MODEL_FILES["rf"])
    joblib.dump(xgb, MODEL_FILES["xgb"])
    joblib.dump(mlp, MODEL_FILES["mlp"])
    joblib.dump(ensemble, MODEL_FILES["ensemble"])
    joblib.dump(scaler, MODEL_FILES["scaler"])
    joblib.dump(encoder, MODEL_FILES["encoder"])

    print("\nSaved ✔")


if __name__ == "__main__":
    main()