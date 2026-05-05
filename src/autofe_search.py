"""AutoFE search via Optuna for Richter's Predictor.

Searches the optimal subset of ~24 candidate features on top of the base
feature pipeline.  Uses LightGBM on fold 0 only for speed (no embeddings).

Usage:
    python src/autofe_search.py --config configs/config.yaml [--n-trials 40]

Output:
    models/autofe_best_features.json  — dict {feature_name: bool, score: float}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.features import build_features
from utils.seed import set_global_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate feature definitions
# ---------------------------------------------------------------------------

CANDIDATE_NAMES: list[str] = [
    # Foundation interactions (EDA: foundation_type='r' overrepresented in hard samples)
    "foundation_r_flag",
    "foundation_r_x_geo2",
    # Structural non-linearities
    "floors_sq",
    "floors_x_height",
    "area_x_floors",
    "height_to_floors",
    # Age non-linearities
    "age_sq",
    "age_x_floors",
    # Material-age degradation
    "cement_brick_x_age",
    "mud_stone_x_floors",
    # Material composite flags
    "strong_structure",
    "weak_structure",
    # Geo-material interactions
    "geo1_x_mud_stone",
    "geo2_x_floors",
    "rc_eng_x_geo1",
    "mud_stone_x_geo2",
    # Age-geo interaction (different from existing age_x_geo2)
    "age_x_geo1",
    # Composite risk flags
    "old_mud_stone",
    "foundation_x_mud",
    # Position (EDA: position='t' overrepresented in hard samples)
    "position_t_flag",
    "position_t_x_geo2",
    # Secondary use density
    "secondary_count",
    "secondary_x_area",
    # Family density
    "families_x_area",
]

_SECONDARY_COLS = [
    "has_secondary_use_agriculture",
    "has_secondary_use_hotel",
    "has_secondary_use_rental",
    "has_secondary_use_institution",
    "has_secondary_use_school",
    "has_secondary_use_industry",
    "has_secondary_use_health_post",
    "has_secondary_use_gov_office",
    "has_secondary_use_use_police",
    "has_secondary_use_other",
]


def compute_candidates(
    df_raw: pd.DataFrame,
    X_base: pd.DataFrame,
) -> pd.DataFrame:
    """Compute all candidate features from raw columns + base encoded features.

    Args:
        df_raw: Raw DataFrame for this fold (before feature engineering).
        X_base: Base feature matrix returned by build_features() for the same fold.

    Returns:
        DataFrame with one column per candidate feature, aligned with df_raw index.
    """
    cands: dict[str, np.ndarray] = {}

    # Raw columns
    found_r = (df_raw["foundation_type"] == "r").astype(np.int8).values
    position_t = (df_raw["position"] == "t").astype(np.int8).values
    mud_stone = df_raw["has_superstructure_mud_mortar_stone"].values.astype(np.int8)
    rc_eng = df_raw["has_superstructure_rc_engineered"].values.astype(np.int8)
    cement_brick = df_raw["has_superstructure_cement_mortar_brick"].values.astype(np.int8)
    adobe = df_raw["has_superstructure_adobe_mud"].values.astype(np.int8)
    bamboo = df_raw["has_superstructure_bamboo"].values.astype(np.int8)
    age = df_raw["age"].values.astype(np.float32)
    floors = df_raw["count_floors_pre_eq"].values.astype(np.float32)
    height = df_raw["height_percentage"].values.astype(np.float32)
    area = df_raw["area_percentage"].values.astype(np.float32)
    families = df_raw["count_families"].values.astype(np.float32)
    secondary = df_raw["has_secondary_use"].values.astype(np.float32)

    # Geo TE columns from base features (already OOF-fitted)
    geo2_k2 = X_base["geo_level_2_id_te_k2"].values.astype(np.float32)
    geo1_k2 = X_base["geo_level_1_id_te_k2"].values.astype(np.float32)

    # Foundation
    cands["foundation_r_flag"] = found_r
    cands["foundation_r_x_geo2"] = (found_r.astype(np.float32) * geo2_k2)

    # Structural
    cands["floors_sq"] = floors ** 2
    cands["floors_x_height"] = floors * height
    cands["area_x_floors"] = area * floors
    cands["height_to_floors"] = height / (floors + 0.5)

    # Age
    cands["age_sq"] = age ** 2
    cands["age_x_floors"] = age * floors

    # Material-age
    cands["cement_brick_x_age"] = cement_brick.astype(np.float32) * age
    cands["mud_stone_x_floors"] = mud_stone.astype(np.float32) * floors

    # Material composites
    cands["strong_structure"] = (rc_eng | cement_brick).astype(np.int8)
    cands["weak_structure"] = (mud_stone | adobe | bamboo).astype(np.int8)

    # Geo-material
    cands["geo1_x_mud_stone"] = geo1_k2 * mud_stone.astype(np.float32)
    cands["geo2_x_floors"] = geo2_k2 * floors
    cands["rc_eng_x_geo1"] = rc_eng.astype(np.float32) * geo1_k2
    cands["mud_stone_x_geo2"] = mud_stone.astype(np.float32) * geo2_k2

    # Age-geo
    cands["age_x_geo1"] = age * geo1_k2

    # Composite risk
    cands["old_mud_stone"] = ((age > 25) & (mud_stone == 1)).astype(np.int8)
    cands["foundation_x_mud"] = (found_r.astype(bool) & (mud_stone == 1)).astype(np.int8)

    # Position (EDA: position='t' in hard samples)
    cands["position_t_flag"] = position_t
    cands["position_t_x_geo2"] = position_t.astype(np.float32) * geo2_k2

    # Secondary
    sec_sum = sum(
        df_raw[c].values.astype(np.int8)
        for c in _SECONDARY_COLS
        if c in df_raw.columns
    )
    cands["secondary_count"] = sec_sum.astype(np.int8)
    cands["secondary_x_area"] = secondary * area

    # Family density
    cands["families_x_area"] = families * area

    return pd.DataFrame(cands, index=df_raw.index)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _autofe_objective(
    trial: optuna.Trial,
    X_tr_base: pd.DataFrame,
    cands_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va_base: pd.DataFrame,
    cands_va: pd.DataFrame,
    y_va: np.ndarray,
    lgb_params: dict,
    n_classes: int,
    seed: int,
) -> float:
    """Optuna objective: select candidate subset → train LGB fold 0 → F1-micro."""
    selected = [
        name
        for name in CANDIDATE_NAMES
        if trial.suggest_categorical(name, [True, False])
    ]

    if selected:
        X_tr = pd.concat([X_tr_base, cands_tr[selected]], axis=1)
        X_va = pd.concat([X_va_base, cands_va[selected]], axis=1)
    else:
        X_tr = X_tr_base
        X_va = X_va_base

    params = dict(lgb_params)
    n_est = params.pop("n_estimators")
    es = params.pop("early_stopping_rounds")

    model = LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        metric="multi_logloss",
        n_estimators=n_est,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[early_stopping(es, verbose=False), log_evaluation(-1)],
    )
    preds = model.predict(X_va)
    return float(f1_score(y_va, preds, average="micro"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run AutoFE search and save best feature selection to JSON."""
    parser = argparse.ArgumentParser(description="AutoFE search via Optuna (LGB fold 0)")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--n-trials", type=int, default=40, help="Optuna trials")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f)

    seed: int = cfg["seed"]
    set_global_seed(seed)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    train_values = pd.read_csv(cfg["data"]["train_values"], encoding="utf-8")
    train_labels = pd.read_csv(cfg["data"]["train_labels"], encoding="utf-8")
    df_train = train_values.merge(train_labels, on=cfg["data"]["id_col"])
    target_col: str = cfg["data"]["target_col"]
    classes: list[int] = sorted(df_train[target_col].unique())
    n_classes = len(classes)

    # 0-indexed target
    y_all = (df_train[target_col] - 1).values.astype(int)

    # -----------------------------------------------------------------------
    # Fold 0 split (same StratifiedKFold as train.py)
    # -----------------------------------------------------------------------
    skf = StratifiedKFold(
        n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed
    )
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(df_train, y_all)):
        if fold_idx == 0:
            break

    df_tr_raw = df_train.iloc[tr_idx].reset_index(drop=True)
    df_va_raw = df_train.iloc[va_idx].reset_index(drop=True)
    y_tr = y_all[tr_idx]
    y_va = y_all[va_idx]

    # -----------------------------------------------------------------------
    # Base features (no embeddings for speed)
    # -----------------------------------------------------------------------
    print("[AutoFE] Building base features (fold 0, no embeddings)...", flush=True)
    X_tr_base, X_va_base, _, _ = build_features(
        df_tr_raw, df_va_raw, None, cfg, mode="lgb_xgb",
        use_embeddings=False, use_cat_te=True,
    )
    print(f"[AutoFE] Base feature shape: {X_tr_base.shape}", flush=True)

    # Baseline score (no extra features)
    lgb_cfg = cfg["models"]["lgb"]
    hp_path = Path(cfg["output"]["models_dir"]) / "lgb_best_params.json"
    if hp_path.exists():
        with open(hp_path, encoding="utf-8") as f:
            saved_json = json.load(f)
        # JSON structure: {"best_params": {...}, "mean_best_iter": ..., ...}
        saved_hp = saved_json.get("best_params", saved_json)
        lgb_params: dict = {
            "num_leaves": saved_hp.get("num_leaves", lgb_cfg["num_leaves"]),
            "min_child_samples": saved_hp.get("min_child_samples", lgb_cfg["min_child_samples"]),
            "colsample_bytree": saved_hp.get("colsample_bytree", lgb_cfg["colsample_bytree"]),
            "subsample": saved_hp.get("subsample", lgb_cfg.get("subsample", 0.8)),
            "subsample_freq": 1,
            "learning_rate": saved_hp.get("learning_rate", lgb_cfg["learning_rate"]),
            "reg_alpha": saved_hp.get("reg_alpha", lgb_cfg["reg_alpha"]),
            "reg_lambda": saved_hp.get("reg_lambda", lgb_cfg["reg_lambda"]),
            "n_estimators": min(lgb_cfg["n_estimators"], 600),
            "early_stopping_rounds": lgb_cfg["early_stopping_rounds"],
        }
        print("[AutoFE] Using saved LGB best HP.", flush=True)
    else:
        lgb_params = {
            "num_leaves": lgb_cfg["num_leaves"],
            "min_child_samples": lgb_cfg["min_child_samples"],
            "colsample_bytree": lgb_cfg["colsample_bytree"],
            "subsample": lgb_cfg.get("subsample", 0.8),
            "subsample_freq": 1,
            "learning_rate": lgb_cfg["learning_rate"],
            "reg_alpha": lgb_cfg["reg_alpha"],
            "reg_lambda": lgb_cfg["reg_lambda"],
            "n_estimators": min(lgb_cfg["n_estimators"], 600),
            "early_stopping_rounds": lgb_cfg["early_stopping_rounds"],
        }
        print("[AutoFE] No saved HP found, using config defaults.", flush=True)

    # -----------------------------------------------------------------------
    # Compute all candidate features
    # -----------------------------------------------------------------------
    print("[AutoFE] Computing candidate features...", flush=True)
    cands_tr = compute_candidates(df_tr_raw, X_tr_base)
    cands_va = compute_candidates(df_va_raw, X_va_base)
    print(f"[AutoFE] {len(CANDIDATE_NAMES)} candidates computed.", flush=True)

    # Baseline score (all base features, no candidates)
    print("[AutoFE] Computing baseline score (no candidates)...", flush=True)
    base_params = dict(lgb_params)
    n_est_base = base_params.pop("n_estimators")
    es_base = base_params.pop("early_stopping_rounds")
    model_base = LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        metric="multi_logloss",
        n_estimators=n_est_base,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        **base_params,
    )
    model_base.fit(
        X_tr_base,
        y_tr,
        eval_set=[(X_va_base, y_va)],
        callbacks=[early_stopping(es_base, verbose=False), log_evaluation(-1)],
    )
    baseline_score = float(f1_score(y_va, model_base.predict(X_va_base), average="micro"))
    print(f"[AutoFE] Baseline fold-0 F1: {baseline_score:.4f}", flush=True)

    # -----------------------------------------------------------------------
    # Optuna search
    # -----------------------------------------------------------------------
    print(
        f"[AutoFE] Starting Optuna ({args.n_trials} trials, LGB fold 0, no embeddings)...",
        flush=True,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def objective(trial: optuna.Trial) -> float:
        return _autofe_objective(
            trial,
            X_tr_base, cands_tr, y_tr,
            X_va_base, cands_va, y_va,
            dict(lgb_params), n_classes, seed,
        )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    best_trial = study.best_trial
    best_score = best_trial.value
    best_selected = [
        name for name in CANDIDATE_NAMES if best_trial.params.get(name, False)
    ]
    not_selected = [n for n in CANDIDATE_NAMES if n not in best_selected]

    print("\n" + "=" * 60, flush=True)
    print(f"[AutoFE] Best fold-0 F1: {best_score:.4f}  (baseline: {baseline_score:.4f}  delta: {best_score - baseline_score:+.4f})")
    print(f"[AutoFE] Selected ({len(best_selected)}/{len(CANDIDATE_NAMES)}):", flush=True)
    for name in best_selected:
        print(f"  + {name}", flush=True)
    print(f"[AutoFE] Not selected ({len(not_selected)}):", flush=True)
    for name in not_selected:
        print(f"  - {name}", flush=True)

    # Save result
    out_path = Path(cfg["output"]["models_dir"]) / "autofe_best_features.json"
    result = {
        "selected": best_selected,
        "not_selected": not_selected,
        "fold0_score": best_score,
        "fold0_baseline": baseline_score,
        "fold0_delta": best_score - baseline_score,
        "n_trials": args.n_trials,
        "all_params": {name: bool(best_trial.params.get(name, False)) for name in CANDIDATE_NAMES},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[AutoFE] Saved to {out_path}", flush=True)
    print("[AutoFE] Done.", flush=True)


if __name__ == "__main__":
    main()
