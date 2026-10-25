import os

import joblib
from fraud_detection.ingestion import load_data
from fraud_detection.preprocessing import preprocess
from fraud_detection.features import build_features
from fraud_detection.train import temporal_split, train_xgb, RUS_SMOTE
from fraud_detection.evaluate import evaluate_model, business_threshold

os.makedirs("models", exist_ok=True)


def ingestion_task():
    df = load_data("data/creditcard.csv")
    return df

def preprocessing_task(df):
    df, scaler = preprocess(df)
    joblib.dump(scaler, "models/amount_scaler.pkl")
    return df

def feature_engineering_task(df):
    df = build_features(df)
    FEATURES = df.drop("Class", axis=1).columns.tolist()
    joblib.dump(FEATURES, "models/features.pkl")
    return df

def training_task(df):
    train, test = temporal_split(df)

    X_train = train.drop("Class", axis=1)
    y_train = train["Class"]
    X_test  = test.drop("Class", axis=1)
    y_test  = test["Class"]

    X_train_resampled, y_train_resampled = RUS_SMOTE(X_train, y_train)

    model = train_xgb(X_train_resampled, y_train_resampled)

    joblib.dump(model, "models/fraud_model.pkl")

    return model, X_test, y_test

def evaluation_task(model, X_test, y_test):
    roc, pr, probs = evaluate_model(model, X_test, y_test)
    threshold = business_threshold(y_test, probs)

    joblib.dump(threshold, "models/optimal_threshold.pkl")

    return roc, pr, threshold