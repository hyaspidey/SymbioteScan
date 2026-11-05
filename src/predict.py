"""Prediction utilities for SymbioteScan."""

import pandas as pd
import joblib


def predict_player(
    player_data,
    model_path="symbiotescan_model.pkl"
):
    """Predict whether a player is Human or Bot."""

    saved = joblib.load(model_path)

    model = saved["model"]
    feature_columns = saved["feature_columns"]

    # Convert input into a DataFrame.
    if isinstance(player_data, dict):
        player_data = pd.DataFrame([player_data])

    elif not isinstance(player_data, pd.DataFrame):
        raise TypeError(
            "Player data must be a dictionary or DataFrame."
        )

    # Exclude identifiers and IP-related fields.
    player_data = player_data.drop(
        columns=[
            col for col in player_data.columns
            if col.lower() in ["actor", "a_acc"]
            or "ip" in col.lower()
        ],
        errors="ignore"
    )

    # Keep the exact training features and their order.
    player_data = player_data.reindex(
        columns=feature_columns
    )

    prediction = model.predict(player_data)[0]

    probabilities = model.predict_proba(
        player_data
    )[0]

    classes = model.classes_

    confidence = max(probabilities)

    result = {
        "prediction": prediction,
        "confidence": round(float(confidence) * 100, 2),
        "probabilities": {
            str(label): round(float(probability) * 100, 2)
            for label, probability in zip(classes, probabilities)
        }
    }

    return result


if __name__ == "__main__":
    print("SymbioteScan prediction module is ready.")
