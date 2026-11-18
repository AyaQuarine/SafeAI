
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/ravdess")
OUTPUT_FILE = Path("data/metadata.csv")

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

def prepare_dataset():
    rows = []

    # Search recursively, regardless of ZIP folder structure
    audio_files = sorted(DATA_DIR.rglob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(
            "No WAV files found in data/ravdess/"
        )

    for path in audio_files:
        parts = path.stem.split("-")

        if len(parts) != 7:
            continue

        modality, channel, emotion, intensity, \
            statement, repetition, actor = parts

        if modality != "03" or channel != "01":
            continue

        if emotion not in EMOTIONS:
            continue

        rows.append({
            "path": str(path.resolve()),
            "emotion": EMOTIONS[emotion],
            "emotion_id": int(emotion),
            "actor": int(actor),
            "intensity": int(intensity),
            "statement": int(statement),
            "repetition": int(repetition),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No valid RAVDESS speech files.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Dataset successfully prepared!")
    print(f"Total audio files: {len(df)}")
    print(f"Unique speakers: {df['actor'].nunique()}")
    print("\nEmotion distribution:")
    print(df["emotion"].value_counts())
    print(f"\nSaved metadata to {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_dataset()
