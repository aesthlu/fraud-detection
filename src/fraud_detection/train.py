def temporal_split(df, split_ratio=0.7):
    split_time = df["Time"].quantile(split_ratio)

    train = df[df["Time"] <= split_time]
    test  = df[df["Time"] > split_time]

    return train, test

def RUS_SMOTE(X, y):
    from imblearn.combine import SMOTEENN
    smote_enn = SMOTEENN(random_state=42)
    X_res, y_res = smote_enn.fit_resample(X, y)
    return X_res, y_res

from sklearn.linear_model import LogisticRegression

def train_logistic(X, y):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )
    model.fit(X, y)
    return model

from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

def train_xgb(X, y):
    from fraud_detection.config import XGB_PARAMS

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    with mlflow.start_run(run_name="xgboost_training"):

        mlflow.log_params(XGB_PARAMS)

        model = XGBClassifier(
            **XGB_PARAMS,
            scale_pos_weight=scale_pos_weight
        )

        model.fit(X, y)

        mlflow.log_param("n_estimators", XGB_PARAMS["n_estimators"])
        mlflow.log_param("max_depth", XGB_PARAMS["max_depth"])
        mlflow.log_param("learning_rate", XGB_PARAMS["learning_rate"])
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        mlflow.sklearn.log_model(sk_model=model, name="fraud_model")

    return model

# mlflow ui