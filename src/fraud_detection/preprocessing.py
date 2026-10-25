import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Log-transform amount (très classique en banque)
    df["Amount_log"] = np.log1p(df["Amount"])

    # Standardisation du montant
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount_log"]])

    return df, scaler

def preprocess_inference(df, scaler):
    df = df.copy()

    df["Amount_log"] = np.log1p(df["Amount"])
    df["Amount_scaled"] = scaler.transform(df[["Amount_log"]])

    return df