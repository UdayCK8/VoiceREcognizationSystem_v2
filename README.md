# Voice Recognition System v2

A real-time voice command recognition system built using **Ensemble Machine Learning**, **MFCC audio features**, and **Data Augmentation**, with a live web-based interface for instant predictions through the browser microphone.

---

## Overview

Voice Recognition System v2 is an improved version of the original system, now featuring:

- **Ensemble ML**: Soft voting across 4 models (SVM, Random Forest, XGBoost, MLP)
- **Expanded Features**: 418-dimensional feature vectors including MFCC, Delta, Delta², Chroma, Spectral Contrast, Tonnetz, and temporal statistics
- **Data Augmentation**: Time stretching, pitch shifting, noise injection, and volume scaling to increase training data diversity
- **Web Interface**: Real-time voice command recognition via browser microphone

---

## Voice Commands (9 Classes)

```
left | no | right | yes | down | speak | rcb | class | mine
```

---

## Model Performance

| Model      | CV Accuracy | Test Accuracy |
|------------|-------------|---------------|
| SVM        | ~97–98%     | ~98–99%       |
| RF         | ~96–97%     | ~97–98%       |
| XGBoost    | ~97–98%     | ~98–99%       |
| MLP        | ~96–97%     | ~97–98%       |
| **Ensemble** | **~98–99%** | **~99%**     |

> *Note: Placeholder values — actual performance depends on dataset size and quality.*

---

## Tech Stack

- **Python** — Core programming language
- **Flask** — Web backend framework
- **Scikit-learn** — SVM, Random Forest, MLP models, GridSearchCV, evaluation
- **XGBoost** — Gradient boosting classifier
- **Librosa** — Audio processing & feature extraction
- **Noisereduce** — Noise reduction for cleaner audio
- **NumPy / Pandas** — Data handling
- **Joblib** — Model serialization
- **HTML / CSS / JavaScript** — Real-time cyberpunk web interface
- **ffmpeg** — Audio format conversion (WebM → WAV)

---

## Feature Extraction

The system extracts a **418-dimensional feature vector** from each audio sample:

| Feature Group             | Dimensions |
|---------------------------|------------|
| MFCC mean, std, max       | 120 (40×3) |
| Delta mean, std, max      | 120 (40×3) |
| Delta² mean, std, max     | 120 (40×3) |
| Chroma mean, std          | 24  (12×2) |
| Spectral Contrast mean, std | 14 (7×2) |
| Tonnetz mean, std         | 12  (6×2)  |
| ZCR mean, std             | 2          |
| Spectral Centroid mean, std | 2        |
| Spectral Rolloff mean, std | 2         |
| RMS mean, std             | 2          |
| **Total**                 | **418**    |

---

## Ensemble Method

The ensemble uses **soft voting** to combine probability predictions from four models:

1. **SVM** — Support Vector Machine with RBF kernel
2. **RF** — Random Forest classifier
3. **XGBoost** — Extreme Gradient Boosting
4. **MLP** — Multi-Layer Perceptron neural network

Each model outputs class probabilities, which are averaged to produce the final prediction. If the highest probability is below the confidence threshold (45%), the prediction is labeled "unknown".

---

## How It Works

1. **Data Collection** (`collect_data.py`)
   - Records voice samples for each of the 9 commands
   - Saves audio as 16kHz mono WAV files in class-labeled folders

2. **Data Augmentation** (`src/augmentation.py`)
   - Applies time stretching, pitch shifting, noise injection, and volume scaling
   - Generates multiple augmented versions per sample to expand the dataset

3. **Feature Extraction** (`src/feature_extraction.py`)
   - Loads audio and extracts 418-dimensional feature vectors
   - Applies noise reduction and silence trimming

4. **Ensemble Training**
   - Trains 4 individual models (SVM, RF, XGBoost, MLP) on scaled features
   - Combines via soft voting into a meta-ensemble model
   - Evaluates using cross-validation and test set accuracy

5. **Real-Time Web Inference** (`app.py`)
   - Flask backend receives audio blob from browser microphone
   - Converts WebM → WAV via ffmpeg
   - Runs VAD check, extracts features, scales, and predicts
   - Returns ensemble prediction + individual model votes to the web UI

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Voice Data

```bash
python collect_data.py
```

### 3. Train the Ensemble Model

```bash
python main.py
```

### 4. Run the Web App

```bash
python app.py
```

The app will automatically open in your browser at:

```
http://localhost:5000
```

### 5. Run the Console App (optional)

```bash
python main.py --console
```

---

## Project Structure

```
Voice Recognition System v2/
│
├── app.py                      # Flask web application (ensemble inference)
├── main.py                     # Training pipeline + console inference
├── collect_data.py             # Voice data collection script
├── requirements.txt            # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Centralized configuration
│   ├── feature_extraction.py   # 418-dim feature extraction
│   └── augmentation.py         # Data augmentation pipeline
│
├── models/
│   ├── svm_model.pkl           # Trained SVM model
│   ├── rf_model.pkl            # Trained Random Forest model
│   ├── xgb_model.pkl           # Trained XGBoost model
│   ├── mlp_model.pkl           # Trained MLP model
│   ├── ensemble_model.pkl      # Ensemble (soft voting) model
│   ├── scaler.pkl              # StandardScaler for features
│   └── label_encoder.pkl       # Label encoder for classes
│
├── dataset/                    # Recorded voice samples (per class folders)
├── recordings/                 # Live recordings from web app
├── report/                     # Training reports and confusion matrices
│
└── README.md
```

---

## Future Improvements

- Expand to support more voice commands
- Add continuous speech / multi-word command recognition
- Implement speaker verification for personalized commands
- Deploy on cloud infrastructure for public access
- Mobile app integration (iOS/Android)
- Real-time noise cancellation during inference
- Model fine-tuning with larger, more diverse datasets

---

## Author

**Voice Recognition System v2** — Final Year MCA Project  
Built with Ensemble Machine Learning, Librosa, and Flask.

---

## License

This project is open-source and available for educational use.