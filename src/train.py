"""Training script for Richter's Predictor.

Trains one GBM model (lgb / xgb / cat) with:
  1. 5-fold Stratified K-Fold CV (default) or 80/20 holdout (--no-kfold)
  2. Feature engineering via src/features.py
  3. Optuna HPO on fold 0 (k-fold) or holdout val (holdout mode)
  4. Train all folds with best HP → OOF predictions + averaged test predictions
  5. Post-HPO refit on full train with best n_estimators
  6. Saves model + val probabilities + test probabilities for ensemble

Usage:
    python src/train.py --config configs/config.yaml --model lgb
    python src/train.py --config configs/config.yaml --model xgb
    python src/train.py --config configs/config.yaml --model cat
    python src/train.py --config configs/config.yaml --model lgb --no-optuna
    python src/train.py --config configs/config.yaml --model lgb --no-kfold
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.features import build_features
from utils.seed import set_global_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


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


def _make_lgb(params: dict, n_classes: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        metric="multi_logloss",
        **params,
    )


def _make_xgb(params: dict, n_classes: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
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


def _make_et(params: dict) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(**params)


def _make_rf(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**params)


# ---------------------------------------------------------------------------
# Single-model training (with early stopping)
# ---------------------------------------------------------------------------


def _train_lgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
    n_classes: int,
) -> tuple[LGBMClassifier, int]:
    """Train LightGBM with early stopping. Returns (model, best_iteration)."""
    n_estimators = params.pop("n_estimators")
    early_rounds = params.pop("early_stopping_rounds")
    model = _make_lgb({**params, "n_estimators": n_estimators}, n_classes)
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
    n_classes: int,
) -> tuple[XGBClassifier, int]:
    """Train XGBoost with early stopping. Returns (model, best_iteration)."""
    n_estimators = params.pop("n_estimators")
    early_rounds = params.pop("early_stopping_rounds")
    model = _make_xgb(
        {
            **params,
            "n_estimators": n_estimators,
            "early_stopping_rounds": early_rounds,
        },
        n_classes,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
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
    iterations = params.pop("iterations")
    early_rounds = params.pop("early_stopping_rounds")
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


def _train_et(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict,
) -> tuple[ExtraTreesClassifier, int]:
    """Train ExtraTrees. No early stopping — single-shot bagging.

    Returns (model, n_estimators) for API consistency.
    """
    model = _make_et(params)
    model.fit(X_train, y_train)
    return model, params.get("n_estimators", 500)


def _train_rf(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict,
) -> tuple[RandomForestClassifier, int]:
    """Train RandomForest. No early stopping — single-shot bagging.

    Returns (model, n_estimators) for API consistency.
    """
    model = _make_rf(params)
    model.fit(X_train, y_train)
    return model, params.get("n_estimators", 500)


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
    cat_cols: list[str],
    n_classes: int,
) -> tuple[object, int]:
    """Dispatch training to the correct model backend.

    Args:
        model_name: "lgb", "xgb", "cat", "et", or "rf".
        X_train: Training features.
        y_train: Training labels (0-indexed: 0, 1, 2).
        X_val: Validation features.
        y_val: Validation labels (0-indexed).
        params: Hyperparameter dict (mutable — pops early_stopping_rounds).
        cat_cols: Categorical column names (used by CatBoost only).
        n_classes: Number of target classes.

    Returns:
        Tuple of (trained_model, best_iteration).
    """
    p = dict(params)  # don't mutate caller's dict
    if model_name == "lgb":
        return _train_lgb(X_train, y_train, X_val, y_val, p, n_classes)
    elif model_name == "xgb":
        return _train_xgb(X_train, y_train, X_val, y_val, p, n_classes)
    elif model_name == "cat":
        return _train_cat(X_train, y_train, X_val, y_val, p, cat_cols)
    elif model_name == "et":
        return _train_et(X_train, y_train, p)
    elif model_name == "rf":
        return _train_rf(X_train, y_train, p)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")


def refit_model(
    model_name: str,
    X_combined: pd.DataFrame,
    y_combined: np.ndarray,
    params: dict,
    best_iter: int,
    cat_cols: list[str],
    n_classes: int,
) -> object:
    """Refit model on full training data using best_iter from Optuna.

    No early stopping — n_estimators/iterations set to best_iter directly.

    Args:
        model_name: "lgb", "xgb", "cat", "et", or "rf".
        X_combined: Full training features.
        y_combined: Full training labels (0-indexed).
        params: Best hyperparameter dict from Optuna (without early_stopping_rounds).
        best_iter: Best iteration count from the winning Optuna trial.
        cat_cols: Categorical column names (CatBoost only).
        n_classes: Number of target classes.

    Returns:
        Fitted model.
    """
    p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}

    if model_name == "lgb":
        p.pop("n_estimators", None)
        model = _make_lgb({**p, "n_estimators": best_iter}, n_classes)
        model.fit(X_combined, y_combined)

    elif model_name == "xgb":
        p.pop("n_estimators", None)
        model = _make_xgb({**p, "n_estimators": best_iter}, n_classes)
        model.fit(X_combined, y_combined, verbose=False)

    elif model_name == "cat":
        p.pop("iterations", None)
        model = _make_cat({**p, "iterations": best_iter}, cat_cols)
        model.fit(X_combined, y_combined, verbose=False)

    elif model_name == "et":
        model = _make_et(p)
        model.fit(X_combined, y_combined)

    elif model_name == "rf":
        model = _make_rf(p)
        model.fit(X_combined, y_combined)

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
    seed: int,
    n_est_ceiling: int,
    es_rounds: int,
    n_classes: int,
) -> float:
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 31, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": n_est_ceiling,
        "early_stopping_rounds": es_rounds,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": seed,
    }
    model, best_iter = _train_lgb(X_train, y_train, X_val, y_val, params, n_classes)
    trial.set_user_attr("best_iter", best_iter)
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average="micro")


def _xgb_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
    n_est_ceiling: int,
    es_rounds: int,
    n_classes: int,
) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": n_est_ceiling,
        "early_stopping_rounds": es_rounds,
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 0,
        "random_state": seed,
    }
    model, best_iter = _train_xgb(X_train, y_train, X_val, y_val, params, n_classes)
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
    seed: int,
    n_est_ceiling: int,
    es_rounds: int,
) -> float:
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "iterations": n_est_ceiling,
        "early_stopping_rounds": es_rounds,
        "verbose": 0,
        "random_state": seed,
    }
    model, best_iter = _train_cat(X_train, y_train, X_val, y_val, params, cat_cols)
    trial.set_user_attr("best_iter", best_iter)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds.flatten().astype(int), average="micro")
    return f1


def _et_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
    n_estimators: int = 1000,
) -> float:
    # n_estimators fixed — not a key lever for ET; searching it only inflates trial time.
    # Focus search on structural hyperparameters that most affect bias/variance.
    params = {
        "n_estimators": n_estimators,
        "max_depth": trial.suggest_int("max_depth", 10, 45),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
        ),
        "n_jobs": -1,
        "random_state": seed,
    }
    model = _make_et(params)
    model.fit(X_train, y_train)
    trial.set_user_attr("best_iter", n_estimators)
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average="micro")


def run_optuna(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
    cfg: dict,
    n_classes: int,
) -> tuple[dict, int]:
    """Run Optuna HPO study. Returns (best_params, best_iter).

    Args:
        model_name: "lgb", "xgb", "cat", "et", or "rf".
        X_train: Training features.
        y_train: Training labels (0-indexed).
        X_val: Validation features.
        y_val: Validation labels (0-indexed).
        cat_cols: Categorical columns (CatBoost only).
        cfg: Config dict.
        n_classes: Number of target classes.

    Returns:
        Tuple of (best_hyperparams_dict, best_iteration_count).
    """
    optuna_cfg = cfg["optuna"]
    sampler = optuna.samplers.TPESampler(seed=cfg["seed"])
    study = optuna.create_study(
        direction=optuna_cfg["direction"],
        sampler=sampler,
    )

    seed = cfg["seed"]
    n_est_ceiling = cfg["optuna"]["n_estimators_ceiling"]
    es_rounds = cfg["optuna"]["early_stopping_rounds"]

    if model_name == "lgb":

        def obj(trial: optuna.Trial) -> float:
            return _lgb_objective(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
                seed,
                n_est_ceiling,
                es_rounds,
                n_classes,
            )

    elif model_name == "xgb":

        def obj(trial: optuna.Trial) -> float:
            return _xgb_objective(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
                seed,
                n_est_ceiling,
                es_rounds,
                n_classes,
            )

    elif model_name == "cat":

        def obj(trial: optuna.Trial) -> float:
            return _cat_objective(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
                cat_cols,
                seed,
                n_est_ceiling,
                es_rounds,
            )

    elif model_name == "et":
        et_n_estimators = cfg["models"]["et"]["n_estimators"]

        def obj(trial: optuna.Trial) -> float:
            return _et_objective(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
                seed,
                n_estimators=et_n_estimators,
            )

    elif model_name == "rf":
        raise ValueError(
            "RandomForest has no Optuna objective. Use --no-optuna with --model rf."
        )

    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    n_trials = (
        optuna_cfg.get("n_trials_et", optuna_cfg["n_trials"])
        if model_name == "et"
        else optuna_cfg["n_trials"]
    )
    study.optimize(
        obj,
        n_trials=n_trials,
        timeout=optuna_cfg.get("timeout"),
        show_progress_bar=optuna_cfg.get("show_progress_bar", True),
    )

    best_trial = study.best_trial
    best_params = best_trial.params
    best_iter = best_trial.user_attrs.get(
        "best_iter", cfg["optuna"]["best_iter_fallback"]
    )
    logger.info("Best F1-micro (Optuna): %.4f", best_trial.value)
    logger.info("Best iteration: %d", best_iter)
    logger.info("Best params: %s", best_params)
    return best_params, best_iter


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def predict_proba(model_name: str, model: object, X: pd.DataFrame) -> np.ndarray:
    """Get class probabilities. Returns array of shape (n_samples, n_classes).

    Args:
        model_name: "lgb", "xgb", "cat", "et", or "rf".
        model: Fitted model.
        X: Feature matrix.

    Returns:
        Probability array (n_samples, n_classes).
    """
    return model.predict_proba(X)


# ---------------------------------------------------------------------------
# K-Fold training
# ---------------------------------------------------------------------------


def train_kfold(
    model_name: str,
    df_train_full: pd.DataFrame,
    df_test: pd.DataFrame,
    y_all: np.ndarray,
    cfg: dict,
    use_optuna: bool,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    """Train model with Stratified K-Fold CV.

    Optuna runs on fold 0 to find best HP. Then all folds train with those HP.
    OOF predictions are assembled in original sample order.
    Test predictions are averaged across folds.

    Args:
        model_name: "lgb", "xgb", "cat", "et", or "rf".
        df_train_full: Full training DataFrame with target column.
        df_test: Test DataFrame (no target).
        y_all: Full target array (0-indexed).
        cfg: Config dict.
        use_optuna: Whether to run Optuna HPO on fold 0.
        n_classes: Number of target classes.

    Returns:
        Tuple of (oof_proba, test_proba_avg, oof_f1, fold_f1_list).
        oof_proba: (n_train, n_classes) OOF probabilities in original order.
        test_proba_avg: (n_test, n_classes) averaged test probabilities.
        oof_f1: F1-micro on full OOF predictions.
        fold_f1_list: Per-fold F1-micro scores.
    """
    cv_cfg = cfg["cv"]
    n_splits = cv_cfg["n_splits"]
    mode = "catboost" if model_name == "cat" else "lgb_xgb"

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])

    n_train = len(df_train_full)
    oof_proba = np.zeros((n_train, n_classes), dtype=np.float64)
    test_proba_per_fold: list[np.ndarray] = []
    fold_f1_list: list[float] = []
    best_iters: list[int] = []

    best_params: dict | None = None
    t_start_all = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train_full, y_all)):
        t_fold_start = time.time()
        elapsed = t_fold_start - t_start_all
        print(
            f"[{model_name.upper()}] === Fold {fold_idx + 1}/{n_splits} === "
            f"(elapsed: {elapsed:.0f}s)",
            flush=True,
        )

        df_tr = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_va = df_train_full.iloc[val_idx].reset_index(drop=True)
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]

        # Feature engineering per fold (target encoding fitted on train fold only)
        # ET/RF: use_embeddings=False, use_cat_te=False — subsampling dilutes TE signal
        logger.info("[%s] Building features (mode=%s)...", model_name.upper(), mode)
        use_emb = False if model_name in ("et", "rf") else None
        use_cte = False if model_name in ("et", "rf") else None

        # Load AutoFE selected features if available (lgb only — XGB hurt by these features)
        autofe_feats: list[str] | None = None
        if model_name == "lgb":
            autofe_path = Path(cfg["output"]["models_dir"]) / "autofe_best_features.json"
            if autofe_path.exists():
                with open(autofe_path, encoding="utf-8") as _f:
                    autofe_feats = json.load(_f).get("selected", [])

        X_train, X_val, X_test, cat_cols = build_features(
            df_tr, df_va, df_test, cfg, mode,
            use_embeddings=use_emb, use_cat_te=use_cte,
            autofe_features=autofe_feats,
        )

        # Optuna on fold 0 only
        if fold_idx == 0 and use_optuna:
            optuna_cfg = cfg["optuna"]
            n_trials_display = (
                optuna_cfg.get("n_trials_et", optuna_cfg["n_trials"])
                if model_name == "et"
                else optuna_cfg["n_trials"]
            )
            logger.info(
                "[%s] Running Optuna on fold 0 (%d trials)...",
                model_name.upper(),
                n_trials_display,
            )
            best_params, _ = run_optuna(
                model_name, X_train, y_train, X_val, y_val, cat_cols, cfg, n_classes
            )

        # Build training params
        model_cfg = cfg["models"][model_name]
        if best_params is not None:
            train_params = dict(best_params)
            if model_name in ("lgb", "xgb"):
                train_params["n_estimators"] = model_cfg["n_estimators"]
                train_params["early_stopping_rounds"] = model_cfg[
                    "early_stopping_rounds"
                ]
            elif model_name in ("et", "rf"):
                # ET/RF use n_estimators (fixed during Optuna); no early stopping
                train_params["n_estimators"] = model_cfg["n_estimators"]
                train_params.setdefault("n_jobs", -1)
            else:
                train_params["iterations"] = model_cfg["iterations"]
                train_params["early_stopping_rounds"] = model_cfg[
                    "early_stopping_rounds"
                ]
            train_params["random_state"] = cfg["seed"]
        else:
            train_params = dict(model_cfg)

        # Add LGB-specific defaults
        if model_name == "lgb":
            train_params.setdefault("verbose", -1)
            train_params.setdefault("n_jobs", -1)

        # Train fold
        logger.info("[%s] Training fold %d...", model_name.upper(), fold_idx + 1)
        model, fold_best_iter = train_model(
            model_name,
            X_train,
            y_train,
            X_val,
            y_val,
            train_params,
            cat_cols,
            n_classes,
        )
        best_iters.append(fold_best_iter)

        # OOF predictions for this fold
        fold_proba = predict_proba(model_name, model, X_val)
        oof_proba[val_idx] = fold_proba

        # Test predictions (store per fold for weighted averaging)
        fold_test_proba = predict_proba(model_name, model, X_test)
        test_proba_per_fold.append(fold_test_proba)

        # Fold score
        fold_preds = np.argmax(fold_proba, axis=1)
        fold_f1 = f1_score(y_val, fold_preds, average="micro")
        fold_f1_list.append(fold_f1)
        t_fold_end = time.time()
        print(
            f"[{model_name.upper()}] Fold {fold_idx + 1} F1-micro: {fold_f1:.4f} "
            f"(best_iter={fold_best_iter}, took {t_fold_end - t_fold_start:.0f}s)",
            flush=True,
        )

    # Fold-weighted bagging: weight test predictions by fold F1-micro
    fold_weights = np.array(fold_f1_list, dtype=np.float64)
    fold_weights = fold_weights / fold_weights.sum()
    test_proba_avg = sum(w * p for w, p in zip(fold_weights, test_proba_per_fold))

    # Save per-fold test proba (allows re-weighting without retraining)
    models_dir = Path(cfg["output"]["models_dir"])
    for i, fp in enumerate(test_proba_per_fold):
        np.save(models_dir / f"{model_name}_test_proba_fold{i}.npy", fp)

    # Full OOF score
    oof_preds = np.argmax(oof_proba, axis=1)
    oof_f1 = f1_score(y_all, oof_preds, average="micro")

    t_total = time.time() - t_start_all
    print(
        f"\n[{model_name.upper()}] === K-Fold Summary ({t_total:.0f}s total) ===",
        flush=True,
    )
    for i, f1 in enumerate(fold_f1_list):
        print(f"  Fold {i + 1}: {f1:.4f}", flush=True)
    print(
        f"  Mean fold F1: {np.mean(fold_f1_list):.4f} (+/- {np.std(fold_f1_list):.4f})",
        flush=True,
    )
    print(f"  OOF F1-micro: {oof_f1:.4f}", flush=True)
    print(f"  Fold weights: {[f'{w:.4f}' for w in fold_weights]}", flush=True)
    print(f"  Best iters: {best_iters}", flush=True)

    # Persist best HP + mean best_iter so refit can run without re-Optuna
    if best_params is not None:
        hp_record = {
            "best_params": best_params,
            "mean_best_iter": int(round(float(np.mean(best_iters)))),
            "best_iters_per_fold": best_iters,
            "oof_f1": float(oof_f1),
        }
        hp_path = Path(cfg["output"]["models_dir"]) / f"{model_name}_best_params.json"
        with open(hp_path, "w", encoding="utf-8") as fh:
            json.dump(hp_record, fh, indent=2)
        logger.info("[%s] Best HP saved → %s", model_name.upper(), hp_path)

    return oof_proba, test_proba_avg, oof_f1, fold_f1_list


# ---------------------------------------------------------------------------
# Holdout training (legacy --no-kfold path)
# ---------------------------------------------------------------------------


def train_holdout(
    model_name: str,
    df_train_full: pd.DataFrame,
    df_test: pd.DataFrame,
    y_all: np.ndarray,
    cfg: dict,
    use_optuna: bool,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train model with single 80/20 stratified holdout (legacy mode).

    Args:
        model_name: "lgb", "xgb", or "cat".
        df_train_full: Full training DataFrame with target column.
        df_test: Test DataFrame (no target).
        y_all: Full target array (0-indexed).
        cfg: Config dict.
        use_optuna: Whether to run Optuna HPO.
        n_classes: Number of target classes.

    Returns:
        Tuple of (val_proba, test_proba, best_f1).
    """
    mode = "catboost" if model_name == "cat" else "lgb_xgb"

    # Stratified split
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

    # Feature engineering
    # ET/RF: use_embeddings=False, use_cat_te=False — subsampling dilutes TE signal
    logger.info("[%s] Building features (mode=%s)...", model_name.upper(), mode)
    use_emb = False if model_name in ("et", "rf") else None
    use_cte = False if model_name in ("et", "rf") else None
    X_train, X_val, X_test, cat_cols = build_features(
        df_tr, df_va, df_test, cfg, mode, use_embeddings=use_emb, use_cat_te=use_cte
    )

    # Baseline train
    model_cfg = cfg["models"][model_name]
    default_params = dict(model_cfg)
    logger.info("[%s] Training with default params...", model_name.upper())
    model_default, _ = train_model(
        model_name, X_train, y_train, X_val, y_val, default_params, cat_cols, n_classes
    )
    preds_default = model_default.predict(X_val)
    if model_name == "cat":
        preds_default = preds_default.flatten().astype(int)
    f1_default = f1_score(y_val, preds_default, average="micro")
    logger.info("[%s] Baseline F1-micro: %.4f", model_name.upper(), f1_default)

    if not use_optuna:
        logger.info("[%s] Skipping Optuna (--no-optuna flag set).", model_name.upper())
        val_proba = predict_proba(model_name, model_default, X_val)
        test_proba = predict_proba(model_name, model_default, X_test)
        return val_proba, test_proba, f1_default

    # Optuna HPO
    logger.info(
        "[%s] Running Optuna HPO (%d trials)...",
        model_name.upper(),
        cfg["optuna"]["n_trials"],
    )
    best_params, best_iter = run_optuna(
        model_name, X_train, y_train, X_val, y_val, cat_cols, cfg, n_classes
    )

    # Evaluate best params on val
    eval_params = dict(best_params)
    if model_name in ("lgb", "xgb"):
        eval_params["n_estimators"] = model_cfg["n_estimators"]
        eval_params["early_stopping_rounds"] = model_cfg["early_stopping_rounds"]
    elif model_name in ("et", "rf"):
        eval_params["n_estimators"] = model_cfg["n_estimators"]
        eval_params.setdefault("n_jobs", -1)
    else:
        eval_params["iterations"] = model_cfg["iterations"]
        eval_params["early_stopping_rounds"] = model_cfg["early_stopping_rounds"]
    eval_params["random_state"] = cfg["seed"]

    model_eval, _ = train_model(
        model_name, X_train, y_train, X_val, y_val, eval_params, cat_cols, n_classes
    )
    val_proba = predict_proba(model_name, model_eval, X_val)
    preds_best = np.argmax(val_proba, axis=1)
    best_f1 = f1_score(y_val, preds_best, average="micro")
    logger.info("[%s] Best-params F1-micro (val): %.4f", model_name.upper(), best_f1)

    # Post-HPO refit on train+val combined
    logger.info(
        "[%s] Refitting on train+val (n_iter=%d)...", model_name.upper(), best_iter
    )
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = np.concatenate([y_train, y_val])

    refit_params = dict(best_params)
    refit_params["random_state"] = cfg["seed"]
    if model_name in ("lgb", "xgb"):
        if "verbose" not in refit_params:
            refit_params["verbose"] = -1
        if "n_jobs" not in refit_params and model_name == "lgb":
            refit_params["n_jobs"] = -1

    final_model = refit_model(
        model_name, X_combined, y_combined, refit_params, best_iter, cat_cols, n_classes
    )
    test_proba = predict_proba(model_name, final_model, X_test)

    # Save refit model
    models_dir = Path(cfg["output"]["models_dir"])
    joblib.dump(final_model, models_dir / f"{model_name}_model.pkl")

    return val_proba, test_proba, best_f1


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
        choices=["lgb", "xgb", "cat", "et", "rf"],
        required=True,
        help="Model to train: lgb, xgb, cat, et, or rf",
    )
    parser.add_argument(
        "--no-optuna",
        action="store_true",
        help="Skip Optuna HPO — train with config defaults only",
    )
    parser.add_argument(
        "--no-kfold",
        action="store_true",
        help="Use single 80/20 holdout instead of k-fold CV",
    )
    parser.add_argument(
        "--refit-full",
        action="store_true",
        help=(
            "Refit on full train (all folds combined) using best HP from JSON. "
            "Overwrites test_proba.npy only — val_proba/OOF unchanged. "
            "Requires a prior k-fold run to have saved {model}_best_params.json."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Main training entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_global_seed(cfg["seed"])

    models_dir = Path(cfg["output"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("[%s] Loading data...", args.model.upper())
    df_train_full, df_test = load_data(cfg)
    target_col = cfg["data"]["target_col"]
    id_col = cfg["data"]["id_col"]

    # Convert labels to 0-indexed (1,2,3 → 0,1,2)
    y_all = df_train_full[target_col].values - 1

    # Save test building IDs for predict.py
    test_ids = df_test[id_col].values
    np.save(models_dir / "test_ids.npy", test_ids)

    use_optuna = not args.no_optuna
    n_classes = len(cfg["features"]["classes"])

    if args.refit_full:
        # Refit on full train using saved best HP — only overwrites test_proba.npy
        hp_path = models_dir / f"{args.model}_best_params.json"
        if not hp_path.exists():
            raise FileNotFoundError(
                f"{hp_path} not found. Run k-fold training first to generate best HP."
            )
        with open(hp_path, encoding="utf-8") as fh:
            hp_record = json.load(fh)

        best_params = hp_record["best_params"]
        mean_best_iter = hp_record["mean_best_iter"]
        logger.info(
            "[%s] Refit-full: best_params=%s mean_best_iter=%d",
            args.model.upper(), best_params, mean_best_iter,
        )

        # Build features on full train (no val split — use a dummy val = first 1000 rows)
        mode = "catboost" if args.model == "cat" else "lgb_xgb"
        use_emb = False if args.model in ("et", "rf") else None
        use_cte = False if args.model in ("et", "rf") else None
        # Load AutoFE features (lgb only — consistent with kfold path)
        autofe_feats_refit: list[str] | None = None
        if args.model == "lgb":
            autofe_path_r = models_dir / "autofe_best_features.json"
            if autofe_path_r.exists():
                with open(autofe_path_r, encoding="utf-8") as _f:
                    autofe_feats_refit = json.load(_f).get("selected", [])
        dummy_val = df_train_full.iloc[:1000].reset_index(drop=True)
        df_tr = df_train_full.reset_index(drop=True)
        X_full, _, X_test, cat_cols = build_features(
            df_tr, dummy_val, df_test, cfg, mode,
            use_embeddings=use_emb, use_cat_te=use_cte,
            autofe_features=autofe_feats_refit,
        )
        y_full = y_all

        refit_params = dict(best_params)
        refit_params["random_state"] = cfg["seed"]
        if args.model in ("lgb", "xgb") and "verbose" not in refit_params:
            refit_params["verbose"] = -1
        if args.model == "lgb" and "n_jobs" not in refit_params:
            refit_params["n_jobs"] = -1

        logger.info("[%s] Refitting on %d samples...", args.model.upper(), len(X_full))
        final_model = refit_model(
            args.model, X_full, y_full, refit_params, mean_best_iter, cat_cols, n_classes,
        )
        test_proba = predict_proba(args.model, final_model, X_test)
        np.save(models_dir / f"{args.model}_test_proba.npy", test_proba)
        joblib.dump(final_model, models_dir / f"{args.model}_model.pkl")
        print(
            f"[{args.model.upper()}] Refit-full done. "
            f"test_proba.npy overwritten ({test_proba.shape}).",
            flush=True,
        )
        return

    if args.no_kfold:
        # Legacy holdout mode
        logger.info("[%s] Mode: single holdout (--no-kfold)", args.model.upper())
        val_proba, test_proba, best_f1 = train_holdout(
            args.model, df_train_full, df_test, y_all, cfg, use_optuna, n_classes
        )

        # Save val true labels for ensemble (holdout only)
        idx = np.arange(len(df_train_full))
        _, idx_val = train_test_split(
            idx,
            test_size=cfg["split"]["test_size"],
            random_state=cfg["split"]["random_state"],
            stratify=y_all,
        )
        y_val = y_all[idx_val]
        val_true_path = models_dir / "val_true.npy"
        if val_true_path.exists():
            existing = np.load(val_true_path)
            if not np.array_equal(existing, y_val):
                raise RuntimeError(
                    "val_true.npy already exists but labels differ. "
                    "Delete models/ and retrain all models with the same config."
                )
        np.save(val_true_path, y_val)

    else:
        # K-Fold CV mode (default)
        logger.info(
            "[%s] Mode: %d-fold stratified k-fold",
            args.model.upper(),
            cfg["cv"]["n_splits"],
        )
        oof_proba, test_proba, oof_f1, fold_f1_list = train_kfold(
            args.model, df_train_full, df_test, y_all, cfg, use_optuna, n_classes
        )
        val_proba = oof_proba
        best_f1 = oof_f1

        # Save val true labels (full OOF — all training samples)
        val_true_path = models_dir / "val_true.npy"
        if val_true_path.exists():
            existing = np.load(val_true_path)
            if len(existing) != len(y_all) or not np.array_equal(existing, y_all):
                raise RuntimeError(
                    "val_true.npy already exists but labels differ. "
                    "Delete models/ and retrain all models with the same config."
                )
        np.save(val_true_path, y_all)

    # Save artifacts
    print(f"[{args.model.upper()}] Saving artifacts...", flush=True)
    np.save(models_dir / f"{args.model}_val_proba.npy", val_proba)
    np.save(models_dir / f"{args.model}_test_proba.npy", test_proba)

    print(f"[{args.model.upper()}] Done. F1-micro: {best_f1:.4f}", flush=True)
    print(f"  Saved: {models_dir}/{args.model}_val_proba.npy", flush=True)
    print(f"  Saved: {models_dir}/{args.model}_test_proba.npy", flush=True)


if __name__ == "__main__":
    main()
