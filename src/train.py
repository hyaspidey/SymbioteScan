"""Train the SymbioteScan bot detection model."""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def train_model(data_path, target_column="bot_label"):
    """Train, evaluate, and save the bot detection model."""

    # Load dataset
    data = pd.read_csv(data_path)

    # Check target column
    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    # Separate features and labels
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Keep numeric features for this starter model
    X = X.select_dtypes(include=["number"])

    if X.empty:
        raise ValueError("No numeric feature columns found.")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Build model pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)

    print("Training complete!")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Save model
    joblib.dump(model, "symbiotescan_model.pkl")

    print("\nModel saved successfully!")

    return model
