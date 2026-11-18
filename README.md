# Safe-AI Voice

Safe-AI Voice is a speech emotion recognition prototype that analyzes a short WAV recording and reports one of eight emotions:

`neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`

The project includes:

- A FastAPI WavLM API in `src/api.py`
- A browser microphone interface in `frontend/index.html`
- A Random Forest baseline trained from acoustic features
- WavLM embedding generation and classifier training scripts
- Hyperparameter optimization for the WavLM classifier
- RAVDESS metadata and prebuilt model artifacts

## Important limitation

This is a research prototype. A prediction is a model output, not a psychological diagnosis or an emergency alert.

## Project structure

```text
safe-ai-voice/
├── data/
│   ├── metadata.csv
│   ├── wavlm_embeddings.npz
│   └── ravdess/Audio_Speech_Actors_01-24_16k/
├── frontend/index.html
├── models/
│   ├── random_forest.joblib
│   ├── wavlm_classifier.joblib
│   └── wavlm_optimized.joblib
├── results/
├── src/
│   ├── api.py
│   ├── prepare_data.py
│   ├── features.py
│   ├── train.py
│   ├── train_wavlm.py
│   ├── optimize_wavlm.py
│   └── predict.py
└── requirements.txt
```

## Requirements

- Python 3.10 or newer
- macOS, Linux, or Windows
- A microphone for the browser interface
- Internet access on the first WavLM API or training run to download `microsoft/wavlm-base`

## Setup

Run all commands below from the repository root.

```bash
cd "/Users/hobby/Desktop/3il Courses/I3FE/Project SAFE_AI/safe-ai-voice"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install fastapi uvicorn[standard] python-multipart \
  joblib librosa numpy pandas torch transformers \
  scikit-learn matplotlib
```

The repository's `requirements.txt` is currently empty. Install the packages above before running the project.

To deactivate the virtual environment later:

```bash
deactivate
```

## Run the complete web application

### 1. Start the API

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

On startup, the API loads `models/wavlm_optimized.joblib` and downloads or loads `microsoft/wavlm-base`.

Open the API documentation at:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health

The API accepts a WAV file at `POST /predict` and limits uploads to 20 MB and 15 seconds.

### 2. Start the frontend

In a second terminal, activate the same virtual environment and serve the frontend:

```bash
cd "/Users/hobby/Desktop/3il Courses/I3FE/Project SAFE_AI/safe-ai-voice"
source .venv/bin/activate
python -m http.server 5500 --directory frontend
```

Open:

- http://127.0.0.1:5500

Allow microphone access in the browser, press **Start Recording**, speak for up to 10 seconds, then press **Stop & Analyze**.

The browser interface sends its recording to `http://127.0.0.1:8000/predict`.

## Test the API with a WAV file

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data/ravdess/Audio_Speech_Actors_01-24_16k/Actor_01/03-01-03-01-01-01-01.wav"
```

Use `-F "file=@/absolute/path/to/audio.wav"` to test another file. Only `.wav` uploads are accepted.

## Run the Random Forest baseline

Make a prediction with the existing baseline model:

```bash
python src/predict.py data/ravdess/Audio_Speech_Actors_01-24_16k/Actor_01/03-01-03-01-01-01-01.wav
```

The script prints the predicted emotion, confidence, and probability for every emotion.

To retrain the baseline:

```bash
python src/prepare_data.py
python src/train.py
```

`src/train.py` performs an actor-independent split, trains `models/random_forest.joblib`, and writes:

- `results/classification_report.txt`
- `results/confusion_matrix.png`

## Train the WavLM pipeline

Generate the metadata and cached embeddings:

```bash
python src/prepare_data.py
python src/train_wavlm.py
```

`src/train_wavlm.py` creates `data/wavlm_embeddings.npz` and `models/wavlm_classifier.joblib`. The cached embeddings are reused on subsequent runs.

For hyperparameter optimization:

```bash
python src/optimize_wavlm.py
```

This compares logistic regression, linear SVM, RBF SVM, and Random Forest using group cross-validation, then saves the best model as `models/wavlm_optimized.joblib`, which is used by the API.

## Retraining notes

- `prepare_data.py` searches for RAVDESS `.wav` files and creates `data/metadata.csv`.
- Training splits are actor-independent: an actor may not appear in both training and test data.
- RAVDESS audio must be under `data/ravdess/`; the scripts recognize spoken, channel-01 files.
- Delete `data/wavlm_embeddings.npz` to force `train_wavlm.py` to regenerate all WavLM embeddings.
- WavLM training and embedding extraction can be slow on CPU. macOS Apple Silicon uses MPS during `train_wavlm.py`; the API currently runs WavLM on CPU.

## Reported baseline results

The checked-in reports show the following speaker-independent test performance:

- WavLM plus logistic regression: 52.22% accuracy
- MFCC plus Random Forest: 41.39% accuracy

These values are dataset splits and do not guarantee similar performance on live microphone recordings.

## Troubleshooting

**`Classifier not found: models/wavlm_optimized.joblib`**  
Run `python src/optimize_wavlm.py`, or restore the prebuilt model file.

**`No WAV files found in data/ravdess/`**  
Verify that RAVDESS files are present under `data/ravdess/` and run `python src/prepare_data.py` from the project root.

**`ModuleNotFoundError`**  
Activate the virtual environment and install the packages listed in Setup.

**Microphone permission or frontend error**  
Use `http://127.0.0.1:5500`, grant microphone access, and keep the API running on port 8000. `file://` or a different frontend origin is not supported by the API CORS configuration.

**Audio prediction returns HTTP 400**  
The file must be WAV format, non-empty, and no longer than 15 seconds.

**Audio prediction returns HTTP 413**  
The uploaded file exceeds 20 MB.
