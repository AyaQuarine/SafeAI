
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import torch

from transformers import WavLMModel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

MODEL_NAME = "microsoft/wavlm-base"

CACHE_PATH = Path("data/wavlm_embeddings.npz")
MODEL_PATH = Path("models/wavlm_classifier.joblib")
RESULT_PATH = Path("results/wavlm_report.txt")

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def extract_embedding(path, model):
    audio, _ = librosa.load(
        path,
        sr=16000,
        mono=True
    )

    # WavLM expects a 16 kHz waveform
    inputs = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = model(inputs).last_hidden_state

        # Mean pooling over time
        embedding = output.mean(dim=1)

    return embedding.squeeze(0).cpu().numpy()


def build_embeddings():
    if CACHE_PATH.exists():
        print("Loading cached WavLM embeddings...")
        data = np.load(CACHE_PATH)

        return (
            data["X"],
            data["y"],
            data["groups"]
        )

    df = pd.read_csv("data/metadata.csv")

    print("Loading WavLM...")
    model = WavLMModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    X, y, groups = [], [], []

    for i, row in df.iterrows():
        embedding = extract_embedding(
            row["path"],
            model
        )

        X.append(embedding)
        y.append(row["emotion"])
        groups.append(row["actor"])

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(df)}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)

    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        CACHE_PATH,
        X=X,
        y=y,
        groups=groups
    )

    return X, y, groups


def main():
    print("Device:", DEVICE)

    X, y, groups = build_embeddings()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )

    print("Training classifier...")
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print(f"\nWavLM accuracy: {accuracy:.4f}")
    print(report)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(classifier, MODEL_PATH)

    RESULT_PATH.write_text(
        f"Accuracy: {accuracy:.4f}\n\n{report}",
        encoding="utf-8"
    )

    print("WavLM classifier saved!")


if __name__ == "__main__":
    main()
