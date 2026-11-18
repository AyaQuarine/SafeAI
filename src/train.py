
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from features import extract_features


DATA_FILE = Path("data/metadata.csv")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def load_features():
    df = pd.read_csv(DATA_FILE)

    X = []
    y = []
    groups = []

    for i, row in df.iterrows():
        try:
            features = extract_features(row["path"])

            X.append(features)
            y.append(row["emotion"])
            groups.append(row["actor"])

        except Exception as error:
            print(f"Skipping {row['path']}: {error}")

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(df)} files")

    if not X:
        raise ValueError("No audio features extracted.")

    return (
        np.array(X),
        np.array(y),
        np.array(groups)
    )


def main():
    print("Extracting audio features...")

    X, y, groups = load_features()

    print("Feature matrix:", X.shape)

    # Speaker-independent split
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

    print(
        "Training speakers:",
        np.unique(groups[train_idx])
    )
    print(
        "Testing speakers:",
        np.unique(groups[test_idx])
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy:.4f}")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print("\nClassification report:")
    print(report)

    # Save report
    (RESULTS_DIR / "classification_report.txt").write_text(
        f"Accuracy: {accuracy:.4f}\n\n{report}",
        encoding="utf-8"
    )

    # Confusion matrix
    labels = sorted(np.unique(y))

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    ).plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues"
    )

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=150
    )
    plt.close(fig)

    # Save model
    joblib.dump(
        model,
        MODEL_DIR / "random_forest.joblib"
    )

    print("\nModel saved successfully!")


if __name__ == "__main__":
    main()
