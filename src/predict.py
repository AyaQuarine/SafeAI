
import argparse
from pathlib import Path

import joblib
import numpy as np

from features import extract_features

MODEL_PATH = Path("models/random_forest.joblib")


def predict_emotion(audio_path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run python src/train.py first."
        )

    model = joblib.load(MODEL_PATH)

    features = extract_features(audio_path)

    # Model expects shape (samples, features)
    X = features.reshape(1, -1)

    probabilities = model.predict_proba(X)[0]

    results = {
        emotion: round(float(prob), 4)
        for emotion, prob in zip(
            model.classes_,
            probabilities
        )
    }

    predicted_emotion = model.classes_[
        np.argmax(probabilities)
    ]

    return {
        "predicted_emotion": predicted_emotion,
        "confidence": round(float(np.max(probabilities)), 4),
        "probabilities": results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_path",
        help="Path to a WAV audio file"
    )

    args = parser.parse_args()

    result = predict_emotion(args.audio_path)

    print("\nSafe-AI Voice Prediction")
    print("------------------------")
    print("Emotion:", result["predicted_emotion"])
    print("Confidence:", result["confidence"])

    print("\nEmotion probabilities:")

    for emotion, probability in sorted(
        result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"{emotion:12s}: {probability:.2%}")
