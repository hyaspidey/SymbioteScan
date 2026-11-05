"""Data preparation utilities for SymbioteScan."""

def clean_column_names(columns):
    """Normalize column names for consistent processing."""
    return [str(column).strip().lower().replace(" ", "_") for column in columns]

if __name__ == "__main__":
    print(clean_column_names(["session_duration", "actions_per_minute", "bot_label"]))
    def normalize_data(data):
    """Normalize numeric columns using min-max scaling."""
    numeric_data = data.select_dtypes(include=["number"])

    return (numeric_data - numeric_data.min()) / (
        numeric_data.max() - numeric_data.min()
    )
