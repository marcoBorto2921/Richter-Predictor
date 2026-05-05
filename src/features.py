"""Feature engineering pipeline for Richter's Predictor.

Two modes:
- "lgb_xgb": target-encodes geo_level_* (9 features), label-encodes low-card cats.
- "catboost": keeps all categoricals as-is, relies on CatBoost native handling.

Target encoding is always fitted on the train fold only (no leakage).
"""

from __future__ import annotations

from pathlib import Path

import joblib
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


def _fit_empirical_bayes_encoder(
    df_train: pd.DataFrame,
    geo_col: str,
    target_col: str,
    classes: list[int],
) -> dict[str, tuple[pd.Series, float]]:
    """Fit an empirical Bayes encoder with adaptive shrinkage on training fold.

    GLMM-style encoding: instead of fixed smoothing weight, the shrinkage per
    category adapts based on estimated between-group variance (τ²).

    Formula per category c:
        encoded(c) = (n_c * θ_c + global_mean / τ²) / (n_c + 1/τ²)

    Categories with few samples shrink heavily toward global mean.
    Categories with many samples keep their group mean.
    τ² is estimated from the data via method-of-moments.

    Args:
        df_train: Training DataFrame containing geo_col and target_col.
        geo_col: Name of the geographic column to encode.
        target_col: Name of the target column.
        classes: List of class labels (e.g. [1, 2, 3]).

    Returns:
        Dict mapping encoded column name to (mapping_series, global_fallback).
    """
    result: dict[str, tuple[pd.Series, float]] = {}
    for k in classes:
        col_name = f"{geo_col}_te_k{k}"
        y_binary = (df_train[target_col] == k).astype(float)
        global_mean = float(y_binary.mean())

        stats = y_binary.groupby(df_train[geo_col]).agg(["count", "mean", "var"])
        n_groups = len(stats)
        group_counts = stats["count"]
        group_means = stats["mean"]

        # Method-of-moments estimate of between-group variance τ²
        # τ² = Var(group_means) - E(within_var / n)
        grand_var = float(group_means.var(ddof=1)) if n_groups > 1 else 0.0
        avg_within_var = float(
            (stats["var"].fillna(0) / group_counts.clip(lower=1)).mean()
        )
        tau_sq = max(grand_var - avg_within_var, 1e-8)

        # Adaptive shrinkage: precision_prior = 1/τ²
        precision_prior = 1.0 / tau_sq
        shrunk = (group_counts * group_means + precision_prior * global_mean) / (
            group_counts + precision_prior
        )
        result[col_name] = (shrunk, global_mean)
    return result


def _replace_rare_categories(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame | None,
    geo_cols: list[str],
    threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Replace rare geo categories (freq < threshold in train) with 'rare' sentinel.

    Fitted on df_train frequencies only — df_val and df_test values not in the
    frequent set are also mapped to 'rare' (handles unseen categories too).

    Args:
        df_train: Training DataFrame.
        df_val: Validation DataFrame.
        df_test: Test DataFrame (may be None).
        geo_cols: List of geo column names.
        threshold: Minimum frequency to keep a category.

    Returns:
        Tuple of (df_train, df_val, df_test) with rare categories replaced.
    """
    tr = df_train.copy()
    va = df_val.copy()
    te = df_test.copy() if df_test is not None else None

    for col in geo_cols:
        freq = tr[col].value_counts()
        frequent = set(freq[freq >= threshold].index)
        tr.loc[:, col] = tr[col].where(tr[col].isin(frequent), other=-1)
        va.loc[:, col] = va[col].where(va[col].isin(frequent), other=-1)
        if te is not None:
            te.loc[:, col] = te[col].where(te[col].isin(frequent), other=-1)

    return tr, va, te


def _add_shared_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add engineered features shared across all model modes.

    Features added:
        - superstructure_count: sum of 11 binary superstructure flags
        - has_secondary_use_any: binary OR of all secondary-use flags
        - height_to_area: height_percentage / (area_percentage + 1) — slenderness proxy
        - floors_ge3: indicator for 3+ pre-earthquake floors
        - age_x_area: age * area_percentage — building mass over time
        - material_strength: ordinal composite — RC engineered (+3) to adobe/bamboo (-2)
        - mud_stone_x_age: mud_mortar_stone × age — old degraded weak-material buildings
        - rc_eng_x_floors: rc_engineered × count_floors_pre_eq — modern multistory resilience

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
    out.loc[:, "has_secondary_use_any"] = (out[sec_cols].sum(axis=1) > 0).astype(
        np.int8
    )
    out.loc[:, "height_to_area"] = out["height_percentage"] / (
        out["area_percentage"] + 1.0
    )
    out.loc[:, "floors_ge3"] = (out["count_floors_pre_eq"] >= 3).astype(np.int8)
    out.loc[:, "age_x_area"] = out["age"] * out["area_percentage"]

    return out


def _fit_geo_damage_variance(
    df_train: pd.DataFrame,
    geo_col: str,
    target_col: str,
    smoothing: float = 10.0,
) -> tuple[pd.Series, float]:
    """Fit smoothed std(damage_grade) per geo group on training fold.

    High variance zones have mixed building quality — signal beyond mean damage.
    Smoothed toward global std to handle small groups.

    Args:
        df_train: Training DataFrame with geo_col and target_col.
        geo_col: Geographic grouping column.
        target_col: Target column name.
        smoothing: Additive smoothing weight toward global std.

    Returns:
        Tuple of (mapping_series, global_fallback_std).
    """
    y = df_train[target_col].astype(float)
    global_std = float(y.std())
    stats = y.groupby(df_train[geo_col]).agg(["count", "std"]).fillna(0)
    smoothed = (stats["count"] * stats["std"] + smoothing * global_std) / (
        stats["count"] + smoothing
    )
    return smoothed, global_std


def _add_lgb_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features that benefit LGB/XGB but hurt tree ensembles (ET/RF).

    Features added:
        - material_strength: ordinal composite — RC engineered (+3) to adobe/bamboo (-2)
        - mud_stone_x_age: mud_mortar_stone × age — old degraded weak-material buildings
        - rc_eng_x_floors: rc_engineered × count_floors_pre_eq — modern multistory resilience

    Args:
        df: Input DataFrame (already has shared features applied).

    Returns:
        DataFrame with new features appended.
    """
    out = df.copy()

    out.loc[:, "material_strength"] = (
        out["has_superstructure_rc_engineered"] * 3
        + out["has_superstructure_rc_non_engineered"] * 2
        + out["has_superstructure_cement_mortar_brick"] * 1
        - out["has_superstructure_mud_mortar_stone"] * 1
        - out["has_superstructure_adobe_mud"] * 2
        - out["has_superstructure_bamboo"] * 2
    ).astype(np.int8)

    out.loc[:, "mud_stone_x_age"] = (
        out["has_superstructure_mud_mortar_stone"] * out["age"]
    ).astype(np.float32)

    out.loc[:, "rc_eng_x_floors"] = (
        out["has_superstructure_rc_engineered"] * out["count_floors_pre_eq"]
    ).astype(np.float32)

    return out


def _apply_geo_embeddings(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    te: pd.DataFrame | None,
    cfg: dict,
    models_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Append pre-trained geo entity embedding features to dataframes.

    Loads embedding matrices (geo_emb_*.npy) and LabelEncoders (geo_enc_*.pkl)
    saved by src/geo_embeddings.py. Each geo_level_* column is looked up in the
    embedding matrix and expanded to `dim` float columns named
    `{col}_emb_0`, `{col}_emb_1`, ..., `{col}_emb_{dim-1}`.

    Unseen / rare IDs (including ThresholdReplacer -1 sentinel) map to the
    UNKNOWN embedding at index 0 (zero-initialized, never trained).

    Args:
        tr: Training DataFrame (must still contain raw geo_level_* columns).
        va: Validation DataFrame.
        te: Test DataFrame (may be None).
        cfg: Config dict.
        models_dir: Directory containing geo_emb_*.npy and geo_enc_*.pkl artifacts.

    Returns:
        Tuple of (tr, va, te) with embedding columns appended in-place.
    """
    geo_cols: list[str] = cfg["features"]["geo_cols"]

    for col in geo_cols:
        tag = col.replace("geo_level_", "").replace("_id", "")
        emb_matrix: np.ndarray = np.load(
            models_dir / f"geo_emb_{tag}.npy"
        )  # (vocab+1, dim)
        le: LabelEncoder = joblib.load(models_dir / f"geo_enc_{tag}.pkl")
        known: set[str] = set(le.classes_)
        dim: int = emb_matrix.shape[1]

        frames = [tr, va] + ([te] if te is not None else [])
        for frame in frames:
            raw = frame[col].astype(str).values
            indices = np.zeros(len(raw), dtype=np.int64)  # default: UNKNOWN_IDX=0
            mask = np.array([v in known for v in raw])
            if mask.any():
                indices[mask] = le.transform(raw[mask]) + 1  # shift known → 1..N
            emb_vecs: np.ndarray = emb_matrix[indices]  # (n, dim)
            for d in range(dim):
                frame.loc[:, f"{col}_emb_{d}"] = emb_vecs[:, d].astype(np.float32)

    return tr, va, te


_SECONDARY_COLS_AUTOFE = [
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


def _add_autofe_features(
    df_raw: pd.DataFrame,
    df_encoded: pd.DataFrame,
    selected: list[str],
) -> pd.DataFrame:
    """Append AutoFE-selected candidate features to an encoded DataFrame.

    Uses raw categorical columns (df_raw: foundation_type, position) and
    already-encoded geo TE columns (df_encoded: geo_level_*_te_k*).

    Args:
        df_raw: Raw input DataFrame (pre-encoding), same reset index as df_encoded.
        df_encoded: Encoded feature DataFrame to append columns to.
        selected: List of feature names to compute and append.

    Returns:
        df_encoded with selected features appended (copy).
    """
    if not selected:
        return df_encoded

    out = df_encoded.copy()

    # Raw inputs
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

    # Encoded geo TE columns
    geo2_k2 = df_encoded["geo_level_2_id_te_k2"].values.astype(np.float32)
    geo1_k2 = df_encoded["geo_level_1_id_te_k2"].values.astype(np.float32)

    _all: dict[str, np.ndarray] = {
        "foundation_r_flag": found_r,
        "foundation_r_x_geo2": found_r.astype(np.float32) * geo2_k2,
        "floors_sq": floors ** 2,
        "floors_x_height": floors * height,
        "area_x_floors": area * floors,
        "height_to_floors": height / (floors + 0.5),
        "age_sq": age ** 2,
        "age_x_floors": age * floors,
        "cement_brick_x_age": cement_brick.astype(np.float32) * age,
        "mud_stone_x_floors": mud_stone.astype(np.float32) * floors,
        "strong_structure": (rc_eng | cement_brick).astype(np.int8),
        "weak_structure": (mud_stone | adobe | bamboo).astype(np.int8),
        "geo1_x_mud_stone": geo1_k2 * mud_stone.astype(np.float32),
        "geo2_x_floors": geo2_k2 * floors,
        "rc_eng_x_geo1": rc_eng.astype(np.float32) * geo1_k2,
        "mud_stone_x_geo2": mud_stone.astype(np.float32) * geo2_k2,
        "age_x_geo1": age * geo1_k2,
        "old_mud_stone": ((age > 25) & (mud_stone == 1)).astype(np.int8),
        "foundation_x_mud": (found_r.astype(bool) & (mud_stone == 1)).astype(np.int8),
        "position_t_flag": position_t,
        "position_t_x_geo2": position_t.astype(np.float32) * geo2_k2,
        "secondary_count": sum(
            df_raw[c].values.astype(np.int8)
            for c in _SECONDARY_COLS_AUTOFE
            if c in df_raw.columns
        ).astype(np.int8),
        "secondary_x_area": secondary * area,
        "families_x_area": families * area,
    }

    for name in selected:
        if name in _all:
            out.loc[:, name] = _all[name].astype(np.float32)

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
    use_embeddings: bool | None = None,
    use_cat_te: bool | None = None,
    autofe_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Build feature matrices for training, validation, and test sets.

    For "lgb_xgb" mode:
        - geo_level_* are target-encoded via GLMM (9 new float cols, originals dropped).
        - Low-cardinality categoricals are GLMM target-encoded (24 float cols) when
          use_cat_te=True (default), otherwise label-encoded to integers.
        - Interaction feature age_x_geo2_damage (age * geo_level_2 damage encoding) added.
        - Optional: geo entity embedding columns appended (config-gated or caller-gated).

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
        use_embeddings: Override for geo embedding application.
            None (default) → read from cfg['features']['geo_embedding']['enabled'].
            True/False → override config, regardless of config value.
            Use False for ET/RF (random subsampling dilutes embedding signal).
        use_cat_te: Whether to GLMM target-encode low-cardinality categoricals.
            None (default) → True for LGB/XGB (better than label encoding).
            False → fall back to label encoding (used for ET/RF).

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
    encoding_method: str = cfg["features"].get("encoding_method", "manual_te")

    # -- Replace rare geo categories (fitted on train freq only) --
    rare_threshold: int = cfg["features"].get("rare_threshold", 0)
    if rare_threshold > 0:
        df_train, df_val, df_test = _replace_rare_categories(
            df_train, df_val, df_test, geo_cols, rare_threshold
        )

    # -- Shared feature engineering (no leakage, no target info) --
    tr = _add_shared_features(df_train, cfg)
    va = _add_shared_features(df_val, cfg)
    te = _add_shared_features(df_test, cfg) if df_test is not None else None

    if mode == "lgb_xgb":
        # Material interaction features — beneficial for LGB/XGB, hurt ET/RF
        tr = _add_lgb_xgb_features(tr)
        va = _add_lgb_xgb_features(va)
        if te is not None:
            te = _add_lgb_xgb_features(te)
        if encoding_method == "glmm":
            # -- Empirical Bayes encoding (GLMM-style adaptive shrinkage) --
            for geo_col in geo_cols:
                enc = _fit_empirical_bayes_encoder(tr, geo_col, target_col, classes)
                tr = _apply_target_encoder(tr, geo_col, enc)
                va = _apply_target_encoder(va, geo_col, enc)
                if te is not None:
                    te = _apply_target_encoder(te, geo_col, enc)
        else:
            # -- Manual smoothed target encoding (original approach) --
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

        # Geo damage variance: config-gated (tested 2026-05-05, hurts LGB −0.0018)
        if cfg["features"].get("geo3_damage_std", False):
            geo3_col = "geo_level_3_id"
            var_map, var_fallback = _fit_geo_damage_variance(tr, geo3_col, target_col)
            for frame in [tr, va] + ([te] if te is not None else []):
                frame.loc[:, "geo3_damage_std"] = (
                    frame[geo3_col].map(var_map).fillna(var_fallback).astype(np.float32)
                )

        # -- Optional: geo entity embedding features --
        emb_cfg = cfg["features"].get("geo_embedding", {})
        emb_enabled = (
            use_embeddings
            if use_embeddings is not None
            else emb_cfg.get("enabled", False)
        )
        if emb_enabled:
            models_dir = Path(cfg["output"]["models_dir"])
            tr, va, te = _apply_geo_embeddings(tr, va, te, cfg, models_dir)

        # -- Encode low-cardinality categoricals --
        # use_cat_te=True (default): GLMM TE — 3 float cols per cat (P(grade=k|cat))
        # use_cat_te=False: label-encode to integers (used for ET/RF)
        apply_cat_te = use_cat_te if use_cat_te is not None else True
        if apply_cat_te:
            for col in cat_cols_low:
                enc = _fit_empirical_bayes_encoder(tr, col, target_col, classes)
                tr = _apply_target_encoder(tr, col, enc)
                va = _apply_target_encoder(va, col, enc)
                if te is not None:
                    te = _apply_target_encoder(te, col, enc)
            # Drop original low-card cat columns (replaced by TE versions)
            for frame in [tr, va] + ([te] if te is not None else []):
                frame.drop(columns=[c for c in cat_cols_low if c in frame.columns], inplace=True)
        else:
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

        # -- AutoFE: append selected candidate features --
        if autofe_features:
            X_train = _add_autofe_features(df_train.reset_index(drop=True), X_train, autofe_features)
            X_val = _add_autofe_features(df_val.reset_index(drop=True), X_val, autofe_features)
            if X_test is not None and df_test is not None:
                X_test = _add_autofe_features(df_test.reset_index(drop=True), X_test, autofe_features)

        return X_train, X_val, X_test, []

    else:  # catboost mode
        # Keep all categoricals as strings — CatBoost handles encoding internally.
        # Convert geo cols to string so CatBoost treats them as categorical.
        # Use .astype() on full dict to avoid pandas CoW FutureWarning.
        str_cols = {col: str for col in geo_cols + cat_cols_low if col in tr.columns}
        tr = tr.astype(str_cols)
        va = va.astype(str_cols)
        if te is not None:
            te = te.astype(str_cols)

        drop_cols = [target_col]
        X_train = tr.drop(columns=[c for c in drop_cols if c in tr.columns])
        X_val = va.drop(columns=[c for c in drop_cols if c in va.columns])
        X_test = te if te is not None else None

        cat_cols = geo_cols + cat_cols_low
        return X_train, X_val, X_test, cat_cols
