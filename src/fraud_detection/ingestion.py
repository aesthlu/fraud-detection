import pandas as pd
import os

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at {path}")

    df = pd.read_csv(path)

    return df


def basic_data_checks(df: pd.DataFrame):
    print("=== SHAPE ===")
    print(df.shape)

    print("\n=== FRAUD RATIO ===")
    print(df["Class"].value_counts(normalize=True))

    print("\n=== GENERAL STATS ===")
    print(df.describe())

    print("\n=== TIME RANGE ===")
    print(df["Time"].min(), "→", df["Time"].max())

def temporal_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    """Découpe en train / validation / test selon le temps (pas de mélange aléatoire).

    Un split aléatoire laisserait le modèle « voir le futur » : en production,
    on prédit toujours des transactions postérieures aux données d'entraînement.
    """
    train_end = df["Time"].quantile(train_ratio)
    val_end = df["Time"].quantile(train_ratio + val_ratio)

    train = df[df["Time"] <= train_end]
    val = df[(df["Time"] > train_end) & (df["Time"] <= val_end)]
    test = df[df["Time"] > val_end]

    return train, val, test
