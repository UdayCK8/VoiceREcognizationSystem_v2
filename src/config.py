"""
Centralized configuration for Voice Recognition System v2
"""

import os

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
REPORT_DIR = os.path.join(BASE_DIR, "report")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# =====================================================
# AUDIO SETTINGS
# =====================================================

SAMPLE_RATE = 16000

DURATION = 2.0
RECORD_DURATION = DURATION 

N_MFCC = 40

MIN_AUDIO_LEN = 1000

VAD_RMS_THRESHOLD = 0.008

# =====================================================
# FEATURE EXTRACTION
# =====================================================

FEATURE_SIZE = 418

# =====================================================
# COMMAND CLASSES
# =====================================================

CLASSES = [
    "class",
    "down",
    "left",
    "mine",
    "no",
    "rcb",
    "right",
    "speak",
    "yes"

]

# =====================================================
# DATA AUGMENTATION
# =====================================================

AUGMENTATION_CONFIG = {
    "time_stretch_factors": [0.85, 0.90, 1.10, 1.15],

    "pitch_shift_steps": [
        -2,
        -1,
        1,
        2
    ],

    "noise_snrs_db": [
        15,
        20,
        25
    ],

    "volume_scales": [
        0.8,
        0.9,
        1.1,
        1.2
    ],

    "max_aug_per_sample": 3
}

# =====================================================
# MODEL FILES
# =====================================================

MODEL_FILES = {

    "svm":
    os.path.join(
        MODELS_DIR,
        "svm_model.pkl"
    ),

    "rf":
    os.path.join(
        MODELS_DIR,
        "rf_model.pkl"
    ),

    "xgb":
    os.path.join(
        MODELS_DIR,
        "xgb_model.pkl"
    ),

    "mlp":
    os.path.join(
        MODELS_DIR,
        "mlp_model.pkl"
    ),

    "ensemble":
    os.path.join(
        MODELS_DIR,
        "ensemble_model.pkl"
    ),

    "scaler":
    os.path.join(
        MODELS_DIR,
        "scaler.pkl"
    ),

    "encoder":
    os.path.join(
        MODELS_DIR,
        "label_encoder.pkl"
    )
}

# =====================================================
# TRAINING
# =====================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42

# =====================================================
# PREDICTION
# =====================================================

CONFIDENCE_THRESHOLD = 0.60

# =====================================================
# WEB APP
# =====================================================

HOST = "127.0.0.1"
PORT = 5000
DEBUG = True

# =====================================================
# RECORDING SETTINGS
# =====================================================

MAX_RECORD_SECONDS = 5

ALLOWED_AUDIO_EXTENSIONS = [
    "wav",
    "webm"
]