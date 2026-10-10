"""Data preparation utilities for SymbioteScan."""

def clean_column_names(columns):
    """Normalize column names for consistent processing."""
    return [str(column).strip().lower().replace(" ", "_") for column in columns]

if __name__ == "__main__":
    print(clean_column_names(["Session Duration", "Actions Per Minute", "Bot Label"]))
