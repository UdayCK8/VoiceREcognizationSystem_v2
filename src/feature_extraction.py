"""
Feature Extraction (UPGRADED + SAFE + CONSISTENT)
"""

import numpy as np
import librosa

from src.config import SAMPLE_RATE, N_MFCC, FEATURE_SIZE, MIN_AUDIO_LEN

# optional noise reduction
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except Exception:
    HAS_NOISEREDUCE = False


def extract_features(file_path: str, apply_noise_reduction: bool = False):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return extract_features_from_audio(y, sr, apply_noise_reduction)


def extract_features_from_audio(y, sr=SAMPLE_RATE, apply_noise_reduction=False):

    # normalize audio
    y = librosa.util.normalize(y)

    # optional noise reduction
    if apply_noise_reduction and HAS_NOISEREDUCE:
        try:
            y = nr.reduce_noise(y=y, sr=sr)
        except Exception:
            pass

    # trim silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # too short audio → safe fallback
    if len(y) < MIN_AUDIO_LEN:
        return np.zeros(FEATURE_SIZE, dtype=np.float32)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # spectral features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(chroma=chroma)

    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

    # helper
    def stats(x):
        return np.concatenate([np.mean(x, axis=-1), np.std(x, axis=-1)])

    features = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.max(mfcc, axis=1),
        np.mean(delta, axis=1), np.std(delta, axis=1), np.max(delta, axis=1),
        np.mean(delta2, axis=1), np.std(delta2, axis=1), np.max(delta2, axis=1),

        stats(chroma),
        stats(contrast),
        stats(tonnetz),

        np.array([
            np.mean(zcr), np.std(zcr),
            np.mean(centroid), np.std(centroid),
            np.mean(rolloff), np.std(rolloff),
            np.mean(rms), np.std(rms)
        ])
    ], dtype=np.float32)

    # safety check
    if len(features) != FEATURE_SIZE:
        return np.zeros(FEATURE_SIZE, dtype=np.float32)

    return features


def has_speech(audio, threshold=0.008):
    return np.sqrt(np.mean(audio**2)) > threshold