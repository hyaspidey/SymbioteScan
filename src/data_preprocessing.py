"""Data preparation utilities for SymbioteScan."""

import pandas as pd
from pathlib import Path


def clean_column_names(columns):
    """Normalize column names."""
    return [
        str(column).strip().lower().replace(" ", "_")
        for column in columns
    ]


def merge_feature_files(data_folder):
    """Validate and merge the five feature datasets."""

    files = {
        "player_information": "Player information features.csv",
        "player_actions": "Player actions features.csv",
        "group_activities": "Group activities features.csv",
        "social_diversity": "Social interaction diversity features.csv",
        "network_measures": "Network measures features.csv",
    }

    data_folder = Path(data_folder)
    datasets = {}

    for name, filename in files.items():
        path = data_folder / f"(after) {filename}"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {path}"
            )

        data = pd.read_csv(path)

        if "Actor" not in data.columns:
            raise ValueError(
                f"{filename} is missing Actor."
            )

        if "Type" not in data.columns:
            raise ValueError(
                f"{filename} is missing Type."
            )

        if data["Actor"].duplicated().any():
            raise ValueError(
                f"Duplicate Actor IDs in {filename}."
            )

        datasets[name] = data

    reference = datasets["player_information"]
    reference_ids = set(reference["Actor"])

    for name, data in datasets.items():

        if set(data["Actor"]) != reference_ids:
            raise ValueError(
                f"Actor IDs do not match in {name}."
            )

        labels = reference.set_index("Actor")["Type"]
        other_labels = data.set_index("Actor")["Type"]

        if not labels.sort_index().equals(
            other_labels.sort_index()
        ):
            raise ValueError(
                f"Type labels do not match in {name}."
            )

    merged = reference.copy()

    for name, data in list(datasets.items())[1:]:
        features = data.drop(columns=["Type"])

        merged = merged.merge(
            features,
            on="Actor",
            how="inner",
            validate="one_to_one"
        )

    # Remove identifiers from the final dataset.
    merged = merged.drop(
        columns=["Actor", "A_Acc"],
        errors="ignore"
    )

    # Exclude potentially identifying IP-related columns.
    merged = merged.drop(
        columns=[
            column for column in merged.columns
            if "ip" in column.lower()
        ],
        errors="ignore"
    )

    if merged.isnull().any().any():
        raise ValueError(
            "Missing values found in merged dataset."
        )

    output_path = data_folder / "combined_features.csv"
    merged.to_csv(output_path, index=False)

    print("Dataset successfully merged!")
    print("Rows:", len(merged))
    print("Columns:", len(merged.columns))
    print("Saved to:", output_path)

    return merged
