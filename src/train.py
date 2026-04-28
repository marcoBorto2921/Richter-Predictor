"""Training script for Richter's Predictor.

Trains one GBM model (lgb / xgb / cat) with:
  1. Stratified 80/20 holdout split
  2. Feature engineering via src/features.py
  3. Optuna HPO (100 trials, TPE sampler, F1-micro on val)
  4. Post-HPO refit on train+val combined (best n_estimators from best trial)
  5. Saves model + val probabilities for ensemble

Usage:
    python src/train.py --config configs/config.yaml --model lgb
    python src/train.py --config configs/config.yaml --model xgb
    python src/train.py --config configs/config.yaml --model cat
    python src/train.py --config configs/config.yaml --model lgb --no-optuna
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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features import build_features
from utils.seed import set_global_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge train values + labels; load test values.

    Args:
        cfg: Config dict.

    Returns:
        Tuple of (df_train, df_test). df_train includes target column.
    """
    train_values = pd.read_csv(cfg["data"]["train_values"], encoding="utf-8")
    train_labels = pd.read_csv(cfg["data"]["train_labels"], encoding="utf-8")
    df_train = train_values.merge(train_labels, on=cfg["data"]["id_col"])

    df_test = pd.read_csv(cfg["data"]["test_values"], encoding="utf-8")
    return df_train, df_test


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def _make_lgb(params: dict) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        **params,
    )


def _make_xgb(params: dict) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        **params,
    )


def _make_cat(params: dict, cat_cols: list[str]) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Micro",
        cat_features=cat_cols if cat_cols else None,
        **params,
    )


# ---------------------------------------------------------------------------
# Single-model training (with early stopping)
# ---------------------------------------------------------------------------


def _train_lgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
) -> tuple[LGBMClassifier, int]:
    """Train LightGBM with early stopping. Returns (model, best_iteration)."""
    n_estimators = params.pop("n_estimators", 3000)
    early_rounds = params.pop("early_stopping_rounds", 100)
    model = _make_lgb({**params, "n_estimators": n_estimators})
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(early_rounds, verbose=False), log_evaluation(-1)],
    )
    best_iter = int(model.best_iteration_)
    return model, best_iter


def _train_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
) -> tuple[XGBClassifier, int]:
    """Train XGBoost with early stopping. Returns (model, best_iteration)."""
    n_estimators = params.pop("n_estimators", 3000)
    early_rounds = params.pop("early_stopping_rounds", 100)
    model = _make_xgb({**params, "n_estimators": n_estimators})
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=early_rounds,
    )
    # best_iteration is 0-based; add 1 so refit uses the correct n_estimators
    best_iter = int(model.best_iteration) + 1
    return model, best_iter


def _train_cat(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
    cat_cols: list[str],
) -> tuple[CatBoostClassifier, int]:
    """Train CatBoost with early stopping. Returns (model, best_iteration)."""
    iterations = params.pop("iterations", 3000)
    early_rounds = params.pop("early_stopping_rounds", 100)
    model = _make_cat({**params, "iterations": iterations}, cat_cols)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=early_rounds,
        verbose=False,
    )
    # get_best_iteration() is 0-based; add 1 so refit uses the correct iterations count
    best_iter = int(model.get_best_iteration()) + 1
    return model, best_iter


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
    cat_cols: list[str],
) -> tuple[object, int]:
    """Dispatch training to the correct model backend.

    Args:
        model_name: "lgb", "xgb", or "cat".
        X_train: Training features.
        y_train: Training labels (0-indexed: 0, 1, 2).
        X_val: Validation features.
        y_val: Validation labels (0-indexed).
        params: Hyperparameter dict (mutable — pops early_stopping_rounds).
        cat_cols: Categorical column names (used by CatBoost only).

    Returns:
        Tuple of (trained_model, best_iteration).
    """
    p = dict(params)  # don't mutate caller's dict
    if model_name == "lgb":
        return _train_lgb(X_train, y_train, X_val, y_val, p)
    elif model_name == "xgb":
        return _train_xgb(X_train, y_train, X_val, y_val, p)
    elif model_name == "cat":
        return _train_cat(X_train, y_train, X_val, y_val, p, cat_cols)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")


def refit_model(
    model_name: str,
    X_combined: pd.DataFrame,
    y_combined: np.ndarray,
    params: dict,
    best_iter: int,
    cat_cols: list[str],
) -> object:
    """Refit model on train+val combined using best_iter from Optuna.

    No early stopping — n_estimators/iterations set to best_iter directly.

    Args:
        model_name: "lgb", "xgb", or "cat".
        X_combined: Combined train + val features.
        y_combined: Combined train + val labels (0-indexed).
        params: Best hyperparameter dict from Optuna (without early_stopping_rounds).
        best_iter: Best iteration count from the winning Optuna trial.
        cat_cols: Categorical column names (CatBoost only).

    Returns:
        Fitted model.
    """
    p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}

    if model_name == "lgb":
        p.pop("n_estimators", None)
        model = _make_lgb({**p, "n_estimators": best_iter})
        model.fit(X_combined, y_combined)

    elif model_name == "xgb":
        p.pop("n_estimators", None)
        model = _make_xgb({**p, "n_estimators": best_iter})
        model.fit(X_combined, y_combined, verbose=False)

    elif model_name == "cat":
        p.pop("iterations", None)
        model = _make_cat({**p, "iterations": best_iter}, cat_cols)
        model.fit(X_combined, y_combined, verbose=False)

    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    return model


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------


def _lgb_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
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
        "n_estimators": 3000,
        "early_stopping_rounds": 100,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }
    model, best_iter = _train_lgb(X_train, y_train, X_val, y_val, params)
    trial.set_user_attr("best_iter", best_iter)
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average="micro")


def _xgb_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": 3000,
        "early_stopping_rounds": 100,
        "tree_method": "hist",
        "verbosity": 0,
        "random_state": 42,
    }
    model, best_iter = _train_xgb(X_train, y_train, X_val, y_val, params)
    trial.set_user_attr("best_iter", best_iter)
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average="micro")


def _cat_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
) -> float:
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "iterations": 3000,
        "early_stopping_rounds": 100,
        "verbose": 0,
        "random_state": 42,
    }
    model, best_iter = _train_cat(X_train, y_train, X_val, y_val, params, cat_cols)
    trial.set_user_attr("best_iter", best_iter)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds.flatten().astype(int), average="micro")
    return f1


def run_optuna(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
    cfg: dict,
) -> tuple[dict, int]:
    """Run Optuna HPO study. Returns (best_params, best_iter).

    Args:
        model_name: "lgb", "xgb", or "cat".
        X_train: Training features.
        y_train: Training labels (0-indexed).
        X_val: Validation features.
        y_val: Validation labels (0-indexed).
        cat_cols: Categorical columns (CatBoost only).
        cfg: Config dict.

    Returns:
        Tuple of (best_hyperparams_dict, best_iteration_count).
    """
    optuna_cfg = cfg["optuna"]
    sampler = optuna.samplers.TPESampler(seed=cfg["seed"])
    study = optuna.create_study(
        direction=optuna_cfg["direction"],
        sampler=sampler,
    )

    if model_name == "lgb":

        def obj(trial: optuna.Trial) -> float:
            return _lgb_objective(trial, X_train, y_train, X_val, y_val)

    elif model_name == "xgb":

        def obj(trial: optuna.Trial) -> float:
            return _xgb_objective(trial, X_train, y_train, X_val, y_val)

    elif model_name == "cat":

        def obj(trial: optuna.Trial) -> float:
            return _cat_objective(trial, X_train, y_train, X_val, y_val, cat_cols)

    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    study.optimize(
        obj,
        n_trials=optuna_cfg["n_trials"],
        timeout=optuna_cfg.get("timeout"),
        show_progress_bar=optuna_cfg.get("show_progress_bar", True),
    )

    best_trial = study.best_trial
    best_params = best_trial.params
    best_iter = best_trial.user_attrs.get("best_iter", 500)
    print(f"  Best F1-micro (Optuna): {best_trial.value:.4f}")
    print(f"  Best iteration: {best_iter}")
    print(f"  Best params: {best_params}")
    return best_params, best_iter


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def predict_proba(model_name: str, model: object, X: pd.DataFrame) -> np.ndarray:
    """Get class probabilities. Returns array of shape (n_samples, 3).

    Args:
        model_name: "lgb", "xgb", or "cat".
        model: Fitted model.
        X: Feature matrix.

    Returns:
        Probability array (n_samples, 3).
    """
    if model_name == "cat":
        return model.predict_proba(X)
    return model.predict_proba(X)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a GBM model for Richter-Predictor"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lgb", "xgb", "cat"],
        required=True,
        help="Model to train: lgb, xgb, or cat",
    )
    parser.add_argument(
        "--no-optuna",
        action="store_true",
        help="Skip Optuna HPO — train with config defaults only",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_global_seed(cfg["seed"])

    models_dir = Path(cfg["output"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # -- Load data --
    print(f"[{args.model.upper()}] Loading data...")
    df_train_full, df_test = load_data(cfg)
    target_col = cfg["data"]["target_col"]
    id_col = cfg["data"]["id_col"]

    # Convert labels to 0-indexed (1,2,3 → 0,1,2)
    y_all = df_train_full[target_col].values - 1

    # -- Stratified split --
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
    print(f"[{args.model.upper()}] Building features (mode={mode})...")
    X_train, X_val, X_test, cat_cols = build_features(df_tr, df_va, df_test, cfg, mode)

    # Save test building IDs for predict.py
    test_ids = df_test[id_col].values
    np.save(models_dir / "test_ids.npy", test_ids)

    # Save val true labels (for ensemble.py).
    # If the file already exists (another model was trained first), assert that the
    # labels match — a mismatch means different split seeds were used across runs.
    val_true_path = models_dir / "val_true.npy"
    if val_true_path.exists():
        existing = np.load(val_true_path)
        if not np.array_equal(existing, y_val):
            raise RuntimeError(
                "val_true.npy already exists but labels differ from the current split. "
                "Delete models/ and retrain all models with the same config."
            )
    np.save(val_true_path, y_val)

    # -- Baseline train (default params from config) --
    default_params = dict(cfg["models"][args.model])
    print(f"[{args.model.upper()}] Training with default params...")
    model_default, _ = train_model(
        args.model, X_train, y_train, X_val, y_val, default_params, cat_cols
    )
    preds_default = model_default.predict(X_val)
    if args.model == "cat":
        preds_default = preds_default.flatten().astype(int)
    f1_default = f1_score(y_val, preds_default, average="micro")
    print(f"[{args.model.upper()}] Baseline F1-micro: {f1_default:.4f}")

    # -- Optuna HPO --
    if args.no_optuna:
        print(f"[{args.model.upper()}] Skipping Optuna (--no-optuna flag set).")
        final_model = model_default
        val_proba = predict_proba(args.model, final_model, X_val)
        test_proba = predict_proba(args.model, final_model, X_test)
        best_f1 = f1_default
    else:
        print(
            f"[{args.model.upper()}] Running Optuna HPO ({cfg['optuna']['n_trials']} trials)..."
        )
        best_params, best_iter = run_optuna(
            args.model, X_train, y_train, X_val, y_val, cat_cols, cfg
        )

        # Evaluate on val using the best Optuna trial's model (before refit)
        # Build model with best_params to get val probabilities
        eval_params = dict(best_params)
        if args.model in ("lgb", "xgb"):
            eval_params["n_estimators"] = 3000
            eval_params["early_stopping_rounds"] = 100
        else:
            eval_params["iterations"] = 3000
            eval_params["early_stopping_rounds"] = 100
        eval_params["random_state"] = 42

        model_eval, _ = train_model(
            args.model, X_train, y_train, X_val, y_val, eval_params, cat_cols
        )
        val_proba = predict_proba(args.model, model_eval, X_val)
        preds_best = np.argmax(val_proba, axis=1)
        best_f1 = f1_score(y_val, preds_best, average="micro")
        print(f"[{args.model.upper()}] Best-params F1-micro (val): {best_f1:.4f}")

        # -- Post-HPO refit on train+val combined --
        print(f"[{args.model.upper()}] Refitting on train+val (n_iter={best_iter})...")
        X_combined = pd.concat([X_train, X_val], ignore_index=True)
        y_combined = np.concatenate([y_train, y_val])

        # Add model-specific iteration key to best_params for refit
        refit_params = dict(best_params)
        refit_params["random_state"] = 42
        if args.model in ("lgb", "xgb"):
            if "verbose" not in refit_params:
                refit_params["verbose"] = -1
            if "n_jobs" not in refit_params and args.model == "lgb":
                refit_params["n_jobs"] = -1

        final_model = refit_model(
            args.model, X_combined, y_combined, refit_params, best_iter, cat_cols
        )
        test_proba = predict_proba(args.model, final_model, X_test)

    # -- Save artifacts --
    print(f"[{args.model.upper()}] Saving artifacts...")
    joblib.dump(final_model, models_dir / f"{args.model}_model.pkl")
    np.save(models_dir / f"{args.model}_val_proba.npy", val_proba)
    np.save(models_dir / f"{args.model}_test_proba.npy", test_proba)

    print(f"[{args.model.upper()}] Done. Best F1-micro: {best_f1:.4f}")
    print(f"  Saved: {models_dir}/{args.model}_model.pkl")
    print(f"  Saved: {models_dir}/{args.model}_val_proba.npy")
    print(f"  Saved: {models_dir}/{args.model}_test_proba.npy")


if __name__ == "__main__":
    main()
