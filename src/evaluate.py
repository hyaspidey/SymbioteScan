"""Evaluation utilities for SymbioteScan."""

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(
    data_path="data/combined_features.csv",
    model_path="symbiotescan_model.pkl",
    target_column="Type"
):
    """Evaluate the trained SymbioteScan model."""

    data = pd.read_csv(data_path)

    saved = joblib.load(model_path)

    model = saved["model"]
    feature_columns = saved["feature_columns"]

    if target_column not in data.columns:
        raise ValueError("Target column not found.")

    y = data[target_column].astype(str).str.strip()

    X = data.drop(columns=[target_column])

    X = X.reindex(columns=feature_columns)

    predictions = model.predict(X)

    print("\nSymbioteScan Model Evaluation")
    print("============================")

    print(
        "Accuracy:",
        accuracy_score(y, predictions)
    )

    print(
        "Precision:",
        precision_score(
            y, predictions,
            pos_label="Bot",
            zero_division=0
        )
    )

    print(
        "Recall:",
        recall_score(
            y, predictions,
            pos_label="Bot",
            zero_division=0
        )
    )

    print(
        "F1-score:",
        f1_score(
            y, predictions,
            pos_label="Bot",
            zero_division=0
        )
    )

    print("\nClassification Report:")
    print(classification_report(
        y,
        predictions,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y,
        predictions,
        labels=["Human", "Bot"]
    ))


if __name__ == "__main__":
    evaluate_model()
