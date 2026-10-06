import pandas as pd
from .config import DATASET_PATH


def load_dataset(path=DATASET_PATH):
    """Load the UCI SMS Spam Collection dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download the UCI SMS Spam Collection and place it there."
        )

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    valid_labels = {"ham", "spam"}
    df = df[df["label"].isin(valid_labels)].copy()

    if df.empty:
        raise ValueError("Dataset contains no valid ham/spam rows.")

    df["target"] = (df["label"] == "spam").astype(int)

    return df
