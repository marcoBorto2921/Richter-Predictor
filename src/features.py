"""Feature engineering pipeline for Richter's Predictor.

Two modes:
- "lgb_xgb": target-encodes geo_level_* (9 features), label-encodes low-card cats.
- "catboost": keeps all categoricals as-is, relies on CatBoost native handling.

Target encoding is always fitted on the train fold only (no leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fit_target_encoder(
    df_train: pd.DataFrame,
    geo_col: str,
    target_col: str,
    classes: list[int],
    smoothing: float,
) -> dict[str, tuple[pd.Series, float]]:
    """Fit a smoothed class-conditional target encoder on the training fold.

    For each class k, computes:
        P_smoothed(y=k | geo) = (n_geo * P_group + smoothing * P_global) / (n_geo + smoothing)

    Args:
        df_train: Training DataFrame containing geo_col and target_col.
        geo_col: Name of the geographic column to encode.
        target_col: Name of the target column.
        classes: List of class labels (e.g. [1, 2, 3]).
        smoothing: Additive smoothing weight toward the global class rate.

    Returns:
        Dict mapping encoded column name to (mapping_series, global_fallback).
    """
    result: dict[str, tuple[pd.Series, float]] = {}
    for k in classes:
        col_name = f"{geo_col}_te_k{k}"
        y_binary = (df_train[target_col] == k).astype(float)
        global_rate = float(y_binary.mean())
        stats = y_binary.groupby(df_train[geo_col]).agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_rate) / (
            stats["count"] + smoothing
        )
        result[col_name] = (smoothed, global_rate)
    return result


def _apply_target_encoder(
    df: pd.DataFrame,
    geo_col: str,
    encoder: dict[str, tuple[pd.Series, float]],
) -> pd.DataFrame:
    """Apply a pre-fitted target encoder. Unseen values get the global fallback.

    Args:
        df: DataFrame to transform (must contain geo_col).
        geo_col: Name of the geographic column.
        encoder: Output of _fit_target_encoder for this geo_col.

    Returns:
        DataFrame with new target-encoded columns appended.
    """
    out = df.copy()
    for col_name, (mapping, fallback) in encoder.items():
        out.loc[:, col_name] = out[geo_col].map(mapping).fillna(fallback)
    return out


def _add_shared_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add engineered features shared across all model modes.

    Features added:
        - superstructure_count: sum of 11 binary superstructure flags
        - has_secondary_use_any: binary OR of all secondary-use flags
        - height_to_area: height_percentage / (area_percentage + 1) — slenderness proxy
        - floors_ge3: indicator for 3+ pre-earthquake floors
        - age_x_area: age * area_percentage — building mass over time

    Args:
        df: Input DataFrame.
        cfg: Config dict with features section.

    Returns:
        DataFrame with new features appended.
    """
    out = df.copy()
    super_cols = cfg["features"]["superstructure_cols"]
    sec_cols = cfg["features"]["secondary_use_cols"]

    out.loc[:, "superstructure_count"] = out[super_cols].sum(axis=1)
    out.loc[:, "has_secondary_use_any"] = (out[sec_cols].sum(axis=1) > 0).astype(np.int8)
    out.loc[:, "height_to_area"] = out["height_percentage"] / (out["area_percentage"] + 1.0)
    out.loc[:, "floors_ge3"] = (out["count_floors_pre_eq"] >= 3).astype(np.int8)
    out.loc[:, "age_x_area"] = out["age"] * out["area_percentage"]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame | None,
    cfg: dict,
    mode: str = "lgb_xgb",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Build feature matrices for training, validation, and test sets.

    For "lgb_xgb" mode:
        - geo_level_* are target-encoded (9 new float cols, originals dropped).
        - Low-cardinality categoricals are label-encoded to integers.
        - Interaction feature age_x_geo2_damage (age * geo_level_2 damage encoding) added.

    For "catboost" mode:
        - All categoricals (geo + low-card) are kept as-is.
        - CatBoost will handle them via ordered target statistics internally.
        - cat_cols list is returned for use with cat_features parameter.

    Target encoding is fitted on df_train only — df_val and df_test receive
    the transform with unseen values filled by the global class rate.

    Args:
        df_train: Training set with all 39 features + target column.
        df_val: Validation set with all 39 features + target column.
        df_test: Test set with 39 features, no target column. May be None.
        cfg: Config dict loaded from config.yaml.
        mode: "lgb_xgb" or "catboost".

    Returns:
        Tuple of (X_train, X_val, X_test_or_None, cat_cols).
        cat_cols is non-empty only in "catboost" mode.
    """
    if mode not in ("lgb_xgb", "catboost"):
        raise ValueError(f"mode must be 'lgb_xgb' or 'catboost', got {mode!r}")

    target_col = cfg["data"]["target_col"]
    geo_cols: list[str] = cfg["features"]["geo_cols"]
    cat_cols_low: list[str] = cfg["features"]["cat_cols_low"]
    classes: list[int] = cfg["features"]["classes"]
    smoothing: float = cfg["features"]["target_encoding"]["smoothing"]

    # -- Shared feature engineering (no leakage, no target info) --
    tr = _add_shared_features(df_train, cfg)
    va = _add_shared_features(df_val, cfg)
    te = _add_shared_features(df_test, cfg) if df_test is not None else None

    if mode == "lgb_xgb":
        # -- Target-encode geo columns --
        encoders: dict[str, dict] = {}
        for geo_col in geo_cols:
            enc = _fit_target_encoder(tr, geo_col, target_col, classes, smoothing)
            encoders[geo_col] = enc
            tr = _apply_target_encoder(tr, geo_col, enc)
            va = _apply_target_encoder(va, geo_col, enc)
            if te is not None:
                te = _apply_target_encoder(te, geo_col, enc)

        # Interaction: age × geo_level_2 class-2 encoding (high-damage zones × age)
        geo2_k2_col = "geo_level_2_id_te_k2"
        tr.loc[:, "age_x_geo2_damage"] = tr["age"] * tr[geo2_k2_col]
        va.loc[:, "age_x_geo2_damage"] = va["age"] * va[geo2_k2_col]
        if te is not None:
            te.loc[:, "age_x_geo2_damage"] = te["age"] * te[geo2_k2_col]

        # -- Label-encode low-cardinality categoricals --
        le_map: dict[str, LabelEncoder] = {}
        for col in cat_cols_low:
            le = LabelEncoder()
            le.fit(tr[col].astype(str))
            le_map[col] = le

        for col, le in le_map.items():
            for frame in [tr, va] + ([te] if te is not None else []):
                known = set(le.classes_)
                cleaned = (
                    frame[col]
                    .astype(str)
                    .apply(lambda x: x if x in known else le.classes_[0])
                )
                frame[col] = le.transform(cleaned).astype(int)

        # -- Drop original geo columns (replaced by target-encoded versions) --
        drop_cols = geo_cols + [target_col]
        X_train = tr.drop(columns=[c for c in drop_cols if c in tr.columns])
        X_val = va.drop(columns=[c for c in drop_cols if c in va.columns])
        X_test = (
            te.drop(columns=[c for c in geo_cols if c in te.columns])
            if te is not None
            else None
        )
        return X_train, X_val, X_test, []

    else:  # catboost mode
        # Keep all categoricals as strings — CatBoost handles encoding internally.
        # Convert geo cols to string so CatBoost treats them as categorical.
        for col in geo_cols + cat_cols_low:
            tr[col] = tr[col].astype(str)
            va[col] = va[col].astype(str)
            if te is not None:
                te[col] = te[col].astype(str)

        drop_cols = [target_col]
        X_train = tr.drop(columns=[c for c in drop_cols if c in tr.columns])
        X_val = va.drop(columns=[c for c in drop_cols if c in va.columns])
        X_test = te if te is not None else None

        cat_cols = geo_cols + cat_cols_low
        return X_train, X_val, X_test, cat_cols
