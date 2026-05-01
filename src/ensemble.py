"""Ensemble: optimize blend weights and generate final submission.

Loads val/test probabilities saved by train.py for each model,
optimizes weights on val F1-micro via Nelder-Mead, then blends
test probabilities and writes a timestamped submission CSV.

Usage:
    python src/ensemble.py --config configs/config.yaml
    python src/ensemble.py --config configs/config.yaml --models lgb cat
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# Weight optimization
# ---------------------------------------------------------------------------


def _blend_proba(probas: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Compute weighted average of probability arrays.

    Args:
        probas: List of (n_samples, n_classes) arrays.
        weights: Weight vector, will be normalized to sum to 1.

    Returns:
        Weighted average probability array (n_samples, n_classes).
    """
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    return sum(wi * p for wi, p in zip(w, probas))


def optimize_weights(
    val_probas: list[np.ndarray],
    y_val: np.ndarray,
    initial_weights: list[float],
    seed: int,
) -> np.ndarray:
    """Find optimal blend weights by maximizing F1-micro on the validation set.

    Uses Nelder-Mead (simplex) to minimize negative F1-micro.
    Weights are constrained to [0, 1] and normalized to sum to 1 internally.

    Args:
        val_probas: List of (n_val, 3) probability arrays, one per model.
        y_val: True validation labels (0-indexed).
        initial_weights: Starting weight guess (one per model).
        seed: Random seed for reproducibility.

    Returns:
        Optimal weight array (sums to 1).
    """
    np.random.seed(seed)

    def neg_f1(weights: np.ndarray) -> float:
        blended = _blend_proba(val_probas, np.abs(weights))
        preds = np.argmax(blended, axis=1)
        return -f1_score(y_val, preds, average="micro")

    result = minimize(
        neg_f1,
        x0=np.array(initial_weights, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    optimal = np.abs(result.x)
    optimal = optimal / optimal.sum()
    return optimal


# ---------------------------------------------------------------------------
# Stacking meta-learner
# ---------------------------------------------------------------------------


def stacking_ensemble(
    val_probas: list[np.ndarray],
    y_val: np.ndarray,
    test_probas: list[np.ndarray],
    seed: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train a stacking meta-learner on OOF probabilities.

    Stacks all model probabilities horizontally (n_models * n_classes features),
    trains LogisticRegression with internal CV to avoid overfitting on the
    same OOF predictions used for weight optimization.

    Args:
        val_probas: List of (n_val, n_classes) OOF probability arrays.
        y_val: True validation labels (0-indexed).
        test_probas: List of (n_test, n_classes) test probability arrays.
        seed: Random seed.
        n_splits: Number of CV folds for meta-learner training.

    Returns:
        Tuple of (val_meta_proba, test_meta_proba, oof_f1_micro).
    """
    # Stack: (n_samples, n_models * n_classes)
    X_meta_val = np.hstack(val_probas)
    X_meta_test = np.hstack(test_probas)

    n_samples = len(y_val)
    n_classes = val_probas[0].shape[1]

    # Internal CV to get unbiased OOF predictions from meta-learner
    oof_meta = np.zeros((n_samples, n_classes), dtype=np.float64)
    test_meta_accum = np.zeros((X_meta_test.shape[0], n_classes), dtype=np.float64)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_meta_val, y_val)):
        X_tr, X_va = X_meta_val[tr_idx], X_meta_val[va_idx]
        y_tr = y_val[tr_idx]

        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
        lr.fit(X_tr, y_tr)

        oof_meta[va_idx] = lr.predict_proba(X_va)
        test_meta_accum += lr.predict_proba(X_meta_test)

        fold_preds = np.argmax(oof_meta[va_idx], axis=1)
        fold_f1 = f1_score(y_val[va_idx], fold_preds, average="micro")
        print(f"  Stacking fold {fold_idx + 1}/{n_splits}: F1={fold_f1:.4f}")

    test_meta = test_meta_accum / n_splits
    oof_preds = np.argmax(oof_meta, axis=1)
    oof_f1 = f1_score(y_val, oof_preds, average="micro")

    return oof_meta, test_meta, oof_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ensemble GBM models for Richter-Predictor"
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
        help="Models to ensemble (default: from config ensemble.models_order)",
    )
    parser.add_argument(
        "--method",
        choices=["weighted_avg", "stacking"],
        default="weighted_avg",
        help="Ensemble method: weighted_avg (Nelder-Mead) or stacking (LogisticRegression)",
    )
    return parser.parse_args()


def main() -> None:
    """Main ensemble entrypoint."""
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_global_seed(cfg["seed"])

    models_dir = Path(cfg["output"]["models_dir"])
    submissions_dir = Path(cfg["output"]["submissions_dir"])
    submissions_dir.mkdir(parents=True, exist_ok=True)

    ensemble_cfg = cfg["ensemble"]
    models_order: list[str] = args.models or ensemble_cfg["models_order"]
    # Look up weights by model name so a custom --models order doesn't silently
    # pick up weights intended for a different model.
    config_weight_map: dict[str, float] = dict(
        zip(ensemble_cfg["models_order"], ensemble_cfg["initial_weights"])
    )
    initial_weights: list[float] = [
        config_weight_map.get(m, 1.0 / len(models_order)) for m in models_order
    ]

    # -- Load artifacts --
    print("Loading val probabilities...")
    val_probas: list[np.ndarray] = []
    test_probas: list[np.ndarray] = []

    for model_name in models_order:
        val_path = models_dir / f"{model_name}_val_proba.npy"
        test_path = models_dir / f"{model_name}_test_proba.npy"
        if not val_path.exists():
            raise FileNotFoundError(
                f"Missing {val_path}. Run: python src/train.py --model {model_name}"
            )
        if not test_path.exists():
            raise FileNotFoundError(
                f"Missing {test_path}. Run: python src/train.py --model {model_name}"
            )
        val_probas.append(np.load(val_path))
        test_probas.append(np.load(test_path))
        print(
            f"  Loaded {model_name}: val={val_probas[-1].shape}, test={test_probas[-1].shape}"
        )

    y_val = np.load(models_dir / "val_true.npy")
    test_ids = np.load(models_dir / "test_ids.npy")

    # -- Individual model scores --
    print("\nIndividual val F1-micro scores:")
    for name, proba in zip(models_order, val_probas):
        preds = np.argmax(proba, axis=1)
        f1 = f1_score(y_val, preds, average="micro")
        print(f"  {name}: {f1:.4f}")

    # -- Equal-weight baseline --
    equal_weights = np.ones(len(models_order)) / len(models_order)
    blended_equal = _blend_proba(val_probas, equal_weights)
    f1_equal = f1_score(y_val, np.argmax(blended_equal, axis=1), average="micro")
    print(f"\nEqual-weight ensemble F1-micro: {f1_equal:.4f}")

    if args.method == "stacking":
        # -- Stacking meta-learner --
        print("\nTraining stacking meta-learner (LogisticRegression, 5-fold CV)...")
        _, test_blend, f1_opt = stacking_ensemble(
            val_probas, y_val, test_probas, cfg["seed"]
        )
        print(f"Stacking OOF F1-micro: {f1_opt:.4f}")
        method_tag = "stacking"
    else:
        # -- Optimize weights (Nelder-Mead) --
        print("\nOptimizing ensemble weights (Nelder-Mead)...")
        optimal_weights = optimize_weights(
            val_probas, y_val, initial_weights, cfg["seed"]
        )

        blended_opt = _blend_proba(val_probas, optimal_weights)
        f1_opt = f1_score(y_val, np.argmax(blended_opt, axis=1), average="micro")

        print("Optimal weights:")
        for name, w in zip(models_order, optimal_weights):
            print(f"  {name}: {w:.4f}")
        print(f"Optimized ensemble F1-micro: {f1_opt:.4f}")
        test_blend = _blend_proba(test_probas, optimal_weights)
        method_tag = "wavg"

    # -- Generate submission --
    print("\nGenerating submission...")
    damage_grade = np.argmax(test_blend, axis=1) + 1  # convert 0-indexed back to 1,2,3

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_tag = "_".join(models_order)
    filename = f"{timestamp}_{models_tag}_{method_tag}_f1micro_{f1_opt:.4f}.csv"
    submission_path = submissions_dir / filename

    submission = pd.DataFrame({"building_id": test_ids, "damage_grade": damage_grade})
    submission.to_csv(submission_path, index=False, encoding="utf-8")

    print(f"Submission saved: {submission_path}")
    print(f"Submission shape: {submission.shape}")
    print(
        f"damage_grade distribution:\n{submission['damage_grade'].value_counts().sort_index()}"
    )


if __name__ == "__main__":
    main()
