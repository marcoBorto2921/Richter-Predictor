"""Frank & Hall ordinal decomposition training for Richter's Predictor.

Decomposes 3-class ordinal classification into 2 binary classifiers:
  - P(y > 0): separates class 0 from classes {1, 2}
  - P(y > 1): separates classes {0, 1} from class 2

Reconstructs class probabilities:
  P(class 0) = 1 - P(y > 0)
  P(class 1) = P(y > 0) - P(y > 1)
  P(class 2) = P(y > 1)

Saves outputs as {model}_fh_val_proba.npy and {model}_fh_test_proba.npy
so ensemble.py can blend them with --models lgb_fh cat xgb_fh.

Usage:
    python src/train_fh.py --config configs/config.yaml --model lgb
    python src/train_fh.py --config configs/config.yaml --model xgb
    python src/train_fh.py --config configs/config.yaml --model cat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features import build_features
from utils.seed import set_global_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)

THRESHOLDS = [0, 1]  # P(y > 0) and P(y > 1)


# ---------------------------------------------------------------------------
# Binary model factories
# ---------------------------------------------------------------------------


def _make_lgb_binary(params: dict) -> LGBMClassifier:
    """Create a binary LGBMClassifier with fixed objective and metric.

    Args:
        params: Additional keyword arguments forwarded to LGBMClassifier.

    Returns:
        Configured LGBMClassifier instance.
    """
    return LGBMClassifier(objective="binary", metric="binary_logloss", **params)


def _make_xgb_binary(params: dict) -> XGBClassifier:
    """Create a binary XGBClassifier with fixed objective and eval metric.

    Args:
        params: Additional keyword arguments forwarded to XGBClassifier.

    Returns:
        Configured XGBClassifier instance.
    """
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)


def _make_cat_binary(params: dict, cat_cols: list[str]) -> CatBoostClassifier:
    """Create a binary CatBoostClassifier with fixed loss and eval metric.

    Args:
        params: Additional keyword arguments forwarded to CatBoostClassifier.
        cat_cols: List of categorical feature names passed via cat_features.

    Returns:
        Configured CatBoostClassifier instance.
    """
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        cat_features=cat_cols if cat_cols else None,
        **params,
    )


# ---------------------------------------------------------------------------
# Binary training with early stopping
# ---------------------------------------------------------------------------


def _train_binary_lgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
) -> tuple[LGBMClassifier, int]:
    """Train binary LGB. Returns (model, best_iteration)."""
    n_est = params.pop("n_estimators")
    early_rounds = params.pop("early_stopping_rounds")
    model = _make_lgb_binary({**params, "n_estimators": n_est})
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(early_rounds, verbose=False), log_evaluation(-1)],
    )
    return model, int(model.best_iteration_)


def _train_binary_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
) -> tuple[XGBClassifier, int]:
    """Train binary XGB. Returns (model, best_iteration)."""
    n_est = params.pop("n_estimators")
    early_rounds = params.pop("early_stopping_rounds")
    model = _make_xgb_binary(
        {
            **params,
            "n_estimators": n_est,
            "early_stopping_rounds": early_rounds,
        }
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, int(model.best_iteration) + 1


def _train_binary_cat(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
    cat_cols: list[str],
) -> tuple[CatBoostClassifier, int]:
    """Train binary CatBoost. Returns (model, best_iteration)."""
    iterations = params.pop("iterations")
    early_rounds = params.pop("early_stopping_rounds")
    model = _make_cat_binary({**params, "iterations": iterations}, cat_cols)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=early_rounds,
        verbose=False,
    )
    return model, int(model.get_best_iteration()) + 1


def train_binary(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
    cat_cols: list[str],
) -> tuple[object, int]:
    """Dispatch binary training to the correct backend.

    Args:
        model_name: One of "lgb", "xgb", "cat".
        X_train: Training feature matrix.
        y_train: Binary training labels.
        X_val: Validation feature matrix.
        y_val: Binary validation labels.
        params: Model hyperparameters (mutated internally — caller must copy).
        cat_cols: Categorical column names (used by CatBoost only).

    Returns:
        Tuple of (fitted model, best iteration count).
    """
    p = dict(params)
    if model_name == "lgb":
        return _train_binary_lgb(X_train, y_train, X_val, y_val, p)
    elif model_name == "xgb":
        return _train_binary_xgb(X_train, y_train, X_val, y_val, p)
    elif model_name == "cat":
        return _train_binary_cat(X_train, y_train, X_val, y_val, p, cat_cols)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")


def predict_binary_proba(model_name: str, model: object, X: pd.DataFrame) -> np.ndarray:
    """Get P(positive class) from binary model. Returns 1D array."""
    proba = model.predict_proba(X)
    if proba.ndim == 2:
        return proba[:, 1]
    return proba.flatten()


# ---------------------------------------------------------------------------
# Optuna objectives (per binary threshold)
# ---------------------------------------------------------------------------


def _lgb_binary_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_estimators: int,
    early_stopping_rounds: int,
    seed: int,
) -> float:
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 31, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": n_estimators,
        "early_stopping_rounds": early_stopping_rounds,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": seed,
    }
    model, best_iter = _train_binary_lgb(X_train, y_train, X_val, y_val, params)
    trial.set_user_attr("best_iter", best_iter)
    proba = predict_binary_proba("lgb", model, X_val)
    return -log_loss(y_val, proba)  # maximize negative logloss


def _xgb_binary_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_estimators: int,
    early_stopping_rounds: int,
    seed: int,
) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": n_estimators,
        "early_stopping_rounds": early_stopping_rounds,
        "tree_method": "hist",
        "verbosity": 0,
        "random_state": seed,
    }
    model, best_iter = _train_binary_xgb(X_train, y_train, X_val, y_val, params)
    trial.set_user_attr("best_iter", best_iter)
    proba = predict_binary_proba("xgb", model, X_val)
    return -log_loss(y_val, proba)


def _cat_binary_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
    iterations: int,
    early_stopping_rounds: int,
    seed: int,
) -> float:
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "iterations": iterations,
        "early_stopping_rounds": early_stopping_rounds,
        "verbose": 0,
        "random_state": seed,
    }
    model, best_iter = _train_binary_cat(
        X_train, y_train, X_val, y_val, params, cat_cols
    )
    trial.set_user_attr("best_iter", best_iter)
    proba = predict_binary_proba("cat", model, X_val)
    return -log_loss(y_val, proba)


def run_optuna_binary(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
    cfg: dict,
    threshold_k: int,
    n_trials: int,
) -> tuple[dict, int]:
    """Run Optuna for one binary classifier. Returns (best_params, best_iter).

    Args:
        model_name: One of "lgb", "xgb", "cat".
        X_train: Training feature matrix.
        y_train: Binary training labels.
        X_val: Validation feature matrix.
        y_val: Binary validation labels.
        cat_cols: Categorical column names (CatBoost only).
        cfg: Full project config dict.
        threshold_k: Threshold index (0 or 1) — used to offset Optuna seed.
        n_trials: Number of Optuna trials.

    Returns:
        Tuple of (best hyperparameters dict, best iteration count).
    """
    seed: int = cfg["seed"]
    model_cfg: dict = cfg["models"][model_name]
    n_estimators: int = model_cfg.get("n_estimators", model_cfg.get("iterations", 3000))
    early_rounds: int = model_cfg["early_stopping_rounds"]

    sampler = optuna.samplers.TPESampler(seed=seed + threshold_k)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    if model_name == "lgb":

        def obj(trial: optuna.Trial) -> float:
            return _lgb_binary_objective(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
                n_estimators,
                early_rounds,
                seed,
            )
    elif model_name == "xgb":

        def obj(trial: optuna.Trial) -> float:
            return _xgb_binary_objective(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
                n_estimators,
                early_rounds,
                seed,
            )
    elif model_name == "cat":

        def obj(trial: optuna.Trial) -> float:
            return _cat_binary_objective(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
                cat_cols,
                n_estimators,
                early_rounds,
                seed,
            )
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    study.optimize(
        obj,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"    Best logloss: {-best.value:.4f}, iter: {best.user_attrs['best_iter']}")
    return best.params, best.user_attrs["best_iter"]


# ---------------------------------------------------------------------------
# Refit on combined train+val
# ---------------------------------------------------------------------------


def refit_binary(
    model_name: str,
    X_combined: pd.DataFrame,
    y_combined: np.ndarray,
    params: dict,
    best_iter: int,
    cat_cols: list[str],
) -> object:
    """Refit binary model on train+val combined, no early stopping.

    Args:
        model_name: One of "lgb", "xgb", "cat".
        X_combined: Combined train+val feature matrix.
        y_combined: Combined binary labels.
        params: Best hyperparameters from Optuna (must not contain early stopping keys).
        best_iter: Number of trees/iterations to train.
        cat_cols: Categorical column names (CatBoost only).

    Returns:
        Fitted model.
    """
    p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}

    if model_name == "lgb":
        p.pop("n_estimators", None)
        model = _make_lgb_binary({**p, "n_estimators": best_iter})
        model.fit(X_combined, y_combined)
    elif model_name == "xgb":
        p.pop("n_estimators", None)
        model = _make_xgb_binary({**p, "n_estimators": best_iter})
        model.fit(X_combined, y_combined, verbose=False)
    elif model_name == "cat":
        p.pop("iterations", None)
        model = _make_cat_binary({**p, "iterations": best_iter}, cat_cols)
        model.fit(X_combined, y_combined, verbose=False)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    return model


# ---------------------------------------------------------------------------
# Frank & Hall reconstruction
# ---------------------------------------------------------------------------


def reconstruct_proba(p_gt0: np.ndarray, p_gt1: np.ndarray) -> np.ndarray:
    """Reconstruct 3-class probabilities from binary cumulative predictions.

    Args:
        p_gt0: P(y > 0) — probability of class 1 or 2.
        p_gt1: P(y > 1) — probability of class 2.

    Returns:
        Array of shape (n_samples, 3) with P(class 0), P(class 1), P(class 2).
    """
    p0 = 1.0 - p_gt0
    p1 = p_gt0 - p_gt1
    p2 = p_gt1

    # Clip negative probabilities from numerical imprecision
    proba = np.column_stack([p0, p1, p2])
    proba = np.clip(proba, 0.0, 1.0)

    # Re-normalize rows to sum to 1
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = proba / row_sums
    return proba


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge train values + labels; load test values.

    Args:
        cfg: Full project config dict.

    Returns:
        Tuple of (df_train, df_test) DataFrames.
    """
    train_values = pd.read_csv(cfg["data"]["train_values"], encoding="utf-8")
    train_labels = pd.read_csv(cfg["data"]["train_labels"], encoding="utf-8")
    df_train = train_values.merge(train_labels, on=cfg["data"]["id_col"])
    df_test = pd.read_csv(cfg["data"]["test_values"], encoding="utf-8")
    return df_train, df_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Frank & Hall ordinal decomposition training"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lgb", "xgb", "cat"],
        required=True,
        help="Model to train: lgb, xgb, or cat",
    )
    parser.add_argument("--no-optuna", action="store_true", help="Skip Optuna HPO")
    parser.add_argument(
        "--fh-trials",
        type=int,
        default=None,
        help="Optuna trials per binary classifier (default: optuna.fh_trials from config)",
    )
    return parser.parse_args()


def main() -> None:
    """Main Frank & Hall training entrypoint."""
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_global_seed(cfg["seed"])

    seed: int = cfg["seed"]
    models_dir = Path(cfg["output"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # fh_trials: CLI flag > config key > fallback to optuna.n_trials
    if args.fh_trials is not None:
        n_trials: int = args.fh_trials
    else:
        n_trials = cfg["optuna"].get("fh_trials", cfg["optuna"]["n_trials"])

    tag = f"{args.model.upper()}-FH"

    # -- Load data --
    print(f"[{tag}] Loading data...")
    df_train_full, df_test = load_data(cfg)
    target_col = cfg["data"]["target_col"]
    id_col = cfg["data"]["id_col"]

    y_all = df_train_full[target_col].values - 1  # 0-indexed

    # -- Stratified split (same seed as train.py) --
    idx = np.arange(len(df_train_full))
    idx_train, idx_val = train_test_split(
        idx,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=y_all,
    )
    df_tr = df_train_full.iloc[idx_train].reset_index(drop=True)
    df_va = df_train_full.iloc[idx_val].reset_index(drop=True)
    y_train = y_all[idx_train]
    y_val = y_all[idx_val]

    # -- Feature engineering --
    mode = "catboost" if args.model == "cat" else "lgb_xgb"
    print(f"[{tag}] Building features (mode={mode})...")
    X_train, X_val, X_test, cat_cols = build_features(df_tr, df_va, df_test, cfg, mode)

    # Save test IDs and val labels (same as train.py)
    np.save(models_dir / "test_ids.npy", df_test[id_col].values)
    val_true_path = models_dir / "val_true.npy"
    if val_true_path.exists():
        existing = np.load(val_true_path)
        if not np.array_equal(existing, y_val):
            raise RuntimeError(
                "val_true.npy labels mismatch — delete models/ and retrain."
            )
    np.save(val_true_path, y_val)

    # -- Train binary classifiers for each threshold --
    val_binary_probas: dict[int, np.ndarray] = {}
    test_binary_probas: dict[int, np.ndarray] = {}
    binary_models: dict[int, object] = {}

    model_cfg: dict = cfg["models"][args.model]
    n_estimators: int = model_cfg.get("n_estimators", model_cfg.get("iterations", 3000))
    early_rounds: int = model_cfg["early_stopping_rounds"]

    for k in THRESHOLDS:
        y_bin_train = (y_train > k).astype(int)
        y_bin_val = (y_val > k).astype(int)
        pos_rate = y_bin_train.mean()
        print(f"\n[{tag}] Training P(y > {k}) — positive rate: {pos_rate:.3f}")

        default_params = dict(model_cfg)

        # Baseline
        model_base, _ = train_binary(
            args.model, X_train, y_bin_train, X_val, y_bin_val, default_params, cat_cols
        )
        base_proba = predict_binary_proba(args.model, model_base, X_val)
        base_acc = ((base_proba > 0.5).astype(int) == y_bin_val).mean()
        print(f"  Baseline accuracy: {base_acc:.4f}")

        if args.no_optuna:
            val_binary_probas[k] = base_proba
            if X_test is not None:
                test_binary_probas[k] = predict_binary_proba(
                    args.model, model_base, X_test
                )
            binary_models[k] = model_base
        else:
            # Optuna HPO
            print(f"  Running Optuna ({n_trials} trials)...")
            best_params, best_iter = run_optuna_binary(
                args.model,
                X_train,
                y_bin_train,
                X_val,
                y_bin_val,
                cat_cols,
                cfg,
                k,
                n_trials=n_trials,
            )

            # Eval with best params (uses config values for n_estimators/early_stopping)
            eval_params = dict(best_params)
            if args.model in ("lgb", "xgb"):
                eval_params["n_estimators"] = n_estimators
                eval_params["early_stopping_rounds"] = early_rounds
            else:
                eval_params["iterations"] = n_estimators
                eval_params["early_stopping_rounds"] = early_rounds
            eval_params["random_state"] = seed

            model_best, _ = train_binary(
                args.model,
                X_train,
                y_bin_train,
                X_val,
                y_bin_val,
                eval_params,
                cat_cols,
            )
            val_binary_probas[k] = predict_binary_proba(args.model, model_best, X_val)

            # Refit on combined
            print(f"  Refitting on train+val (n_iter={best_iter})...")
            X_combined = pd.concat([X_train, X_val], ignore_index=True)
            y_bin_combined = np.concatenate([y_bin_train, y_bin_val])

            refit_params = dict(best_params)
            refit_params["random_state"] = seed
            if args.model == "lgb":
                refit_params.setdefault("verbose", -1)
                refit_params.setdefault("n_jobs", -1)

            model_refit = refit_binary(
                args.model,
                X_combined,
                y_bin_combined,
                refit_params,
                best_iter,
                cat_cols,
            )
            if X_test is not None:
                test_binary_probas[k] = predict_binary_proba(
                    args.model, model_refit, X_test
                )
            binary_models[k] = model_refit

    # -- Reconstruct 3-class probabilities --
    print(f"\n[{tag}] Reconstructing 3-class probabilities...")
    val_proba = reconstruct_proba(val_binary_probas[0], val_binary_probas[1])
    test_proba = reconstruct_proba(test_binary_probas[0], test_binary_probas[1])

    # -- Evaluate --
    preds = np.argmax(val_proba, axis=1)
    f1 = f1_score(y_val, preds, average="micro")
    print(f"[{tag}] F1-micro (val): {f1:.4f}")

    # -- Save artifacts --
    fh_tag = f"{args.model}_fh"
    joblib.dump(binary_models, models_dir / f"{fh_tag}_models.pkl")
    np.save(models_dir / f"{fh_tag}_val_proba.npy", val_proba)
    np.save(models_dir / f"{fh_tag}_test_proba.npy", test_proba)

    print(f"[{tag}] Done. F1-micro: {f1:.4f}")
    print(f"  Saved: {models_dir}/{fh_tag}_models.pkl")
    print(f"  Saved: {models_dir}/{fh_tag}_val_proba.npy")
    print(f"  Saved: {models_dir}/{fh_tag}_test_proba.npy")
    print(f"\n  Use in ensemble: python src/ensemble.py --models {fh_tag} cat ...")


if __name__ == "__main__":
    main()
