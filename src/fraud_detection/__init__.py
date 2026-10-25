import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="fraud-detection")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Entraîner, évaluer et enregistrer un modèle")
    train.add_argument("--config", default="configs/train.yaml")

    args = parser.parse_args()

    if args.command == "train":
        from fraud_detection.config import load_config
        from fraud_detection.pipeline import run_training

        run_training(load_config(args.config))
