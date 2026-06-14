#!/usr/bin/env python3
"""
Real-time Voice Command Predictor v2 (Fixed + Improved)
"""

import os
import sys
import numpy as np
import sounddevice as sd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_FILES, CLASSES, SAMPLE_RATE,
    FEATURE_SIZE, VAD_RMS_THRESHOLD, CONFIDENCE_THRESHOLD,
    RECORD_DURATION   # ✅ FIXED: missing in your code
)

from src.feature_extraction import extract_features_from_audio


# ── Global Model State ──
ensemble_model = None
scaler = None
label_encoder = None


def load_models():
    global ensemble_model, scaler, label_encoder

    print("\nLoading models...")

    ensemble_model = joblib.load(MODEL_FILES['ensemble'])
    scaler = joblib.load(MODEL_FILES['scaler'])
    label_encoder = joblib.load(MODEL_FILES['encoder'])

    print("  Ensemble model loaded")
    print("  Scaler loaded")
    print("  Label encoder loaded")
    print(f"  Classes: {list(label_encoder.classes_)}\n")


def record_audio(duration: float = RECORD_DURATION) -> np.ndarray:
    """Record audio safely"""

    device = None
    try:
        device = sd.default.device[0]
    except Exception:
        pass

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        device=device
    )
    sd.wait()

    return np.clip(audio.flatten(), -1.0, 1.0)


def check_vad(audio: np.ndarray, threshold: float = VAD_RMS_THRESHOLD) -> bool:
    """Simple RMS VAD"""

    rms = np.sqrt(np.mean(audio ** 2))
    return rms > threshold


def predict(raw_audio: np.ndarray):
    global ensemble_model, scaler, label_encoder

    try:
        features = extract_features_from_audio(
            raw_audio,
            SAMPLE_RATE,
            apply_noise_reduction=False
        )

        # ── FIX: empty / invalid features check ──
        if features is None or np.allclose(features, 0):
            return None, 0, {}

        # ── FIX: feature size validation ──
        if len(features) != FEATURE_SIZE:
            print(f"  [Feature mismatch: {len(features)} != {FEATURE_SIZE}]")
            return None, 0, {}

        # VAD check
        if not check_vad(raw_audio):
            print("  [No speech detected]")
            return None, 0, {}

        # Scale
        features = scaler.transform(features.reshape(1, -1))

        # Predict
        probas = ensemble_model.predict_proba(features)[0]

        idx = np.argmax(probas)
        confidence = float(probas[idx])
        label = label_encoder.inverse_transform([idx])[0]

        probs = {
            label_encoder.classes_[i]: float(probas[i])
            for i in range(len(probas))
        }

        # ── FIX: confidence rejection logic ──
        if confidence < CONFIDENCE_THRESHOLD:
            return None, confidence * 100, probs

        return label, confidence * 100, probs

    except Exception as e:
        print(f"  [Prediction error: {e}]")
        return None, 0, {}


def display_results(label, confidence, probs):
    print("\n" + "─" * 45)

    if label:
        print(f"  Predicted: {label}")
    else:
        print("  No confident prediction")

    print(f"  Confidence: {confidence:.1f}%")

    if confidence < CONFIDENCE_THRESHOLD * 100:
        print("  [Low confidence]")

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    print("\n  Distribution:")
    max_len = max((len(k) for k in probs), default=10)

    for cls, prob in sorted_probs:
        bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
        print(f"    {cls:<{max_len}} [{bar}] {prob*100:5.1f}%")

    print("─" * 45)


def main():
    print("\n" + "=" * 55)
    print(" Voice Recognition System v2 (Fixed)")
    print("=" * 55)
    print(f" Sample Rate: {SAMPLE_RATE}")
    print(f" Duration: {RECORD_DURATION}s")
    print(f" Classes: {', '.join(CLASSES)}")
    print("=" * 55)

    load_models()

    print("\nPress ENTER to record, Ctrl+C to exit\n")

    try:
        while True:
            input("[Press Enter] ")

            print("Recording...")
            audio = record_audio(RECORD_DURATION)

            label, confidence, probs = predict(audio)

            display_results(label, confidence, probs)

            print()

    except KeyboardInterrupt:
        print("\nGoodbye!\n")


if __name__ == "__main__":
    main()