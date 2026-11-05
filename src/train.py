"""Train the SymbioteScan bot detection model."""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def train_model(
    data_path="data/combined_features.csv",
    target_column="Type"
):
    """Train and evaluate the bot detection model."""

    data = pd.read_csv(data_path)

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    # Separate target and behavioral features.
    y = data[target_column].astype(str).str.strip()

    if not set(y.unique()).issubset({"Human", "Bot"}):
        raise ValueError("Unexpected target labels found.")

    X = data.drop(columns=[target_column])

    # Exclude identifiers and potentially identifying fields.
    X = X.drop(
        columns=[
            col for col in X.columns
            if col.lower() in ["actor", "a_acc"]
            or "ip" in col.lower()
        ],
        errors="ignore"
    )

    # Use numeric behavioral features only.
    X = X.select_dtypes(include=["number"])

    if X.empty:
        raise ValueError("No numeric features found.")

    if X.columns.duplicated().any():
        raise ValueError("Duplicate feature names found.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Training complete!")
    print("Accuracy:", accuracy_score(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns)
        },
        "symbiotescan_model.pkl"
    )

    print("\nModel saved successfully!")

    return model


if __name__ == "__main__":
    train_model()
