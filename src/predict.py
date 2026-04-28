"""Inference script: load saved models and generate a submission CSV.

This is a standalone alternative to ensemble.py — it loads the saved
model artifacts and re-runs prediction without needing the stored .npy
probability files. Useful for re-generating a submission after changing
the ensemble weights manually.

Usage:
    python src/predict.py --config configs/config.yaml
    python src/predict.py --config configs/config.yaml --weights 0.4 0.35 0.25
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
import yaml

from src.features import build_features
from utils.seed import set_global_seed


def load_data_test(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load train (needed for target encoding) and test data.

    Args:
        cfg: Config dict.

    Returns:
        Tuple of (df_train, df_test, test_ids).
    """
    train_values = pd.read_csv(cfg["data"]["train_values"], encoding="utf-8")
    train_labels = pd.read_csv(cfg["data"]["train_labels"], encoding="utf-8")
    df_train = train_values.merge(train_labels, on=cfg["data"]["id_col"])

    df_test = pd.read_csv(cfg["data"]["test_values"], encoding="utf-8")
    test_ids = df_test[cfg["data"]["id_col"]].values
    return df_train, df_test, test_ids


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate submission for Richter-Predictor"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to use (default: from config ensemble.models_order)",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Blend weights (one per model, will be normalized to sum to 1)",
    )
    return parser.parse_args()


def main() -> None:
    """Main prediction entrypoint."""
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_global_seed(cfg["seed"])

    models_dir = Path(cfg["output"]["models_dir"])
    submissions_dir = Path(cfg["output"]["submissions_dir"])
    submissions_dir.mkdir(parents=True, exist_ok=True)

    ensemble_cfg = cfg["ensemble"]
    models_order: list[str] = args.models or ensemble_cfg["models_order"]

    # Weights: CLI > config defaults
    if args.weights is not None:
        weights = np.array(args.weights, dtype=float)
    else:
        weights = np.array(
            ensemble_cfg["initial_weights"][: len(models_order)], dtype=float
        )
    weights = weights / weights.sum()

    # -- Load data --
    print("Loading data...")
    df_train, df_test, test_ids = load_data_test(cfg)

    # Use all of df_train as the "train fold" for target encoding
    # (no split needed here — we just need the encoder fitted on train)
    # Create a dummy val (first 10 rows) so build_features API is satisfied
    df_dummy_val = df_train.iloc[:10].reset_index(drop=True)

    # -- Predict with each model --
    test_probas: list[np.ndarray] = []

    for model_name in models_order:
        model_path = models_dir / f"{model_name}_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing {model_path}. Run: python src/train.py --model {model_name}"
            )

        print(f"Predicting with {model_name}...")
        model = joblib.load(model_path)

        mode = "catboost" if model_name == "cat" else "lgb_xgb"
        _, _, X_test, _ = build_features(df_train, df_dummy_val, df_test, cfg, mode)

        proba = model.predict_proba(X_test)
        test_probas.append(proba)
        print(f"  {model_name} proba shape: {proba.shape}")

    # -- Blend --
    blended = sum(w * p for w, p in zip(weights, test_probas))
    damage_grade = np.argmax(blended, axis=1) + 1  # 0-indexed → 1,2,3

    # -- Write submission --
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_tag = "_".join(models_order)
    filename = f"{timestamp}_{models_tag}_predict.csv"
    submission_path = submissions_dir / filename

    submission = pd.DataFrame({"building_id": test_ids, "damage_grade": damage_grade})
    submission.to_csv(submission_path, index=False, encoding="utf-8")

    print(f"\nSubmission saved: {submission_path}")
    print(f"Shape: {submission.shape}")
    print(f"Weights used: {dict(zip(models_order, weights.tolist()))}")
    print(
        f"damage_grade distribution:\n{submission['damage_grade'].value_counts().sort_index()}"
    )


if __name__ == "__main__":
    main()
