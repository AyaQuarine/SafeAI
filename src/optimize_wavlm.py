
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    GridSearchCV
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

DATA_PATH = "data/wavlm_embeddings.npz"

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]
groups = data["groups"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

train_groups = groups[train_idx]

cv = GroupKFold(n_splits=3)

models = {
    "Logistic Regression": (
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced"
            )
        ),
        {
            "logisticregression__C": [0.01, 0.1, 1, 10]
        }
    ),

    "Linear SVM": (
        make_pipeline(
            StandardScaler(),
            SVC(
                kernel="linear",
                class_weight="balanced"
            )
        ),
        {
            "svc__C": [0.01, 0.1, 1, 10]
        }
    ),

    "RBF SVM": (
        make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                class_weight="balanced"
            )
        ),
        {
            "svc__C": [0.1, 1, 10],
            "svc__gamma": ["scale", 0.001, 0.01]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),
        {
            "n_estimators": [200, 400],
            "max_depth": [None, 20]
        }
    )
}

results = []

for name, (model, params) in models.items():
    print(f"\nOptimizing {name}...")

    search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    search.fit(
        X_train,
        y_train,
        groups=train_groups
    )

    results.append({
        "name": name,
        "cv_f1": search.best_score_,
        "model": search.best_estimator_,
        "params": search.best_params_
    })

    print("CV Macro-F1:", round(search.best_score_, 4))
    print("Best parameters:", search.best_params_)

results.sort(
    key=lambda result: result["cv_f1"],
    reverse=True
)

best = results[0]

print("\nBest model:", best["name"])

predictions = best["model"].predict(X_test)

print(
    "Test accuracy:",
    round(accuracy_score(y_test, predictions), 4)
)

print(
    "Test macro-F1:",
    round(f1_score(y_test, predictions, average="macro"), 4)
)

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

Path("models").mkdir(exist_ok=True)

joblib.dump(
    best["model"],
    "models/wavlm_optimized.joblib"
)

print("\nOptimized model saved!")
