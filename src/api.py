from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from pathlib import Path
import tempfile

import joblib
import librosa
import numpy as np
import torch

from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import WavLMModel


MODEL_NAME = "microsoft/wavlm-base"
CLASSIFIER_PATH = Path("models/wavlm_optimized.joblib")

SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 15
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

DEVICE = "cpu"

wavlm = None
classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global wavlm, classifier

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier not found: {CLASSIFIER_PATH}"
        )

    print("Loading WavLM...")
    wavlm = WavLMModel.from_pretrained(MODEL_NAME)
    wavlm.to(DEVICE)
    wavlm.eval()

    print("Loading trained classifier...")
    classifier = joblib.load(CLASSIFIER_PATH)

    print("Safe-AI Voice API ready!")

    yield


app = FastAPI(
    title="Safe-AI Voice API",
    description="Speech emotion recognition using WavLM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def extract_embedding(audio_path):
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Empty audio recording")

    duration = len(audio) / sr

    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"Audio exceeds {MAX_AUDIO_SECONDS} seconds"
        )

    inputs = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = wavlm(inputs).last_hidden_state
        embedding = output.mean(dim=1)

    return embedding.cpu().numpy(), duration


@app.get("/")
def root():
    return {
        "service": "Safe-AI Voice",
        "status": "running",
        "model": "WavLM + Logistic Regression"
    }


@app.get("/health")
def health():
    return {
        "status": "ready",
        "wavlm_loaded": wavlm is not None,
        "classifier_loaded": classifier is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only WAV files are supported"
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Audio file is too large"
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Empty uploaded file"
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp:
            temp.write(content)
            temp_path = Path(temp.name)

        embedding, duration = extract_embedding(temp_path)

        probabilities = classifier.predict_proba(
            embedding
        )[0]

        classes = classifier.classes_

        index = int(np.argmax(probabilities))

        emotion_probabilities = {
            str(emotion): round(float(prob), 4)
            for emotion, prob in zip(
                classes,
                probabilities
            )
        }

        return {
            "modality": "audio",
            "model": "wavlm-base",
            "predicted_emotion": str(classes[index]),
            "confidence": round(
                float(probabilities[index]), 4
            ),
            "duration_seconds": round(duration, 2),
            "probabilities": emotion_probabilities
        }

    except (ValueError, librosa.util.exceptions.ParameterError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print(f"Prediction error: {error}")
        raise HTTPException(
            status_code=500,
            detail="Audio prediction failed"
        )

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
