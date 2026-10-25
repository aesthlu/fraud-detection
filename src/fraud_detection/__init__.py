import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="fraud-detection")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Entraîner, évaluer et enregistrer un modèle")
    train.add_argument("--config", default="configs/train.yaml")

    export = sub.add_parser("export", help="Exporter le modèle champion pour le serving")
    export.add_argument("--config", default="configs/train.yaml")
    export.add_argument("--output", default="model")

    args = parser.parse_args()

    if args.command == "train":
        from fraud_detection.config import load_config
        from fraud_detection.pipeline import run_training

        run_training(load_config(args.config))

    elif args.command == "export":
        import mlflow
        from mlflow import MlflowClient

        from fraud_detection.config import load_config
        from fraud_detection.registry import export_champion

        cfg = load_config(args.config)["mlflow"]
        mlflow.set_tracking_uri(cfg["tracking_uri"])
        metadata = export_champion(MlflowClient(), cfg["registered_model_name"], args.output)
        print(
            f"Champion v{metadata['model_version']} exporté dans {args.output}/ "
            f"(seuil {metadata['threshold']})"
        )
