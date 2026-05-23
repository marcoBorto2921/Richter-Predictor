"""
Extra EDA analyses from Opus review — Richter's Predictor.
Run: .venv/Scripts/python notebooks/03_eda_extra.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRAIN_VALUES = "data/raw/train_values.csv"
TRAIN_LABELS = "data/raw/train_labels.csv"
SEED = 42


def load() -> pd.DataFrame:
    vals = pd.read_csv(TRAIN_VALUES, encoding="utf-8")
    labels = pd.read_csv(TRAIN_LABELS, encoding="utf-8")
    return vals.merge(labels, on="building_id")


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


# ── 1. Conflicting-label duplicates ──────────────────────────────────────────


def check_conflicting_labels(df: pd.DataFrame) -> None:
    section("1. CONFLICTING-LABEL DUPLICATES")
    feature_cols = [c for c in df.columns if c not in ("building_id", "damage_grade")]
    grouped = df.groupby(feature_cols, sort=False)["damage_grade"]
    n_unique = grouped.nunique()
    conflicting_groups = n_unique[n_unique > 1]
    total_conflicting_rows = grouped.filter(lambda x: x.nunique() > 1).shape[0]
    print(f"Groups with >1 damage_grade (conflicting): {len(conflicting_groups):,}")
    print(
        f"Rows in conflicting groups: {total_conflicting_rows:,} ({total_conflicting_rows / len(df) * 100:.2f}%)"
    )

    # exact-feature-identical rows (label may differ)
    exact_dups = df.duplicated(subset=feature_cols, keep=False)
    print(f"\nFeature-identical rows (any label): {exact_dups.sum():,}")
    dup_df = df[exact_dups].copy()
    dup_counts = dup_df.groupby(feature_cols, sort=False)["damage_grade"].nunique()
    pure_dups = (dup_counts == 1).sum()
    conflict_dups = (dup_counts > 1).sum()
    print(f"  Same label (pure duplicates): {pure_dups:,} groups")
    print(f"  Different labels (conflict):  {conflict_dups:,} groups")

    if len(conflicting_groups) > 0:
        # Sample entropy distribution
        entropies = grouped.apply(
            lambda x: stats.entropy(x.value_counts(normalize=True), base=2)
        )
        print("\nEntropy of conflicting groups (base-2):")
        print(
            f"  mean={entropies[n_unique > 1].mean():.3f}, "
            f"max={entropies[n_unique > 1].max():.3f}"
        )


# ── 2. Group leakage: geo3 in val vs train ───────────────────────────────────


def check_group_leakage(df: pd.DataFrame) -> None:
    section("2. GROUP LEAKAGE — geo_level_3_id in holdout")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    import lightgbm as lgb

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["damage_grade"], random_state=SEED
    )

    train_geo3 = set(train_df["geo_level_3_id"].unique())
    val_geo3 = set(val_df["geo_level_3_id"].unique())
    seen = val_geo3 & train_geo3
    unseen = val_geo3 - train_geo3

    print(f"Val geo3 IDs: {len(val_geo3):,}")
    print(f"  Seen in train: {len(seen):,} ({len(seen) / len(val_geo3) * 100:.1f}%)")
    print(
        f"  Unseen in train: {len(unseen):,} ({len(unseen) / len(val_geo3) * 100:.1f}%)"
    )

    # Quick LGB for F1-micro seen vs unseen
    cat_cols = [
        "foundation_type",
        "roof_type",
        "ground_floor_type",
        "other_floor_type",
        "position",
        "plan_configuration",
        "land_surface_condition",
        "legal_ownership_status",
    ]
    feature_cols = [c for c in df.columns if c not in ("building_id", "damage_grade")]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["damage_grade"] - 1
    X_val = val_df[feature_cols].copy()
    y_val = val_df["damage_grade"].values

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=63,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val) + 1

    val_seen_mask = val_df["geo_level_3_id"].isin(train_geo3).values
    val_unseen_mask = ~val_seen_mask

    f1_overall = f1_score(y_val, preds, average="micro")
    f1_seen = (
        f1_score(y_val[val_seen_mask], preds[val_seen_mask], average="micro")
        if val_seen_mask.sum() > 0
        else float("nan")
    )
    f1_unseen = (
        f1_score(y_val[val_unseen_mask], preds[val_unseen_mask], average="micro")
        if val_unseen_mask.sum() > 0
        else float("nan")
    )

    print("\nQuick LGB (300 rounds) F1-micro:")
    print(f"  Overall:           {f1_overall:.4f}")
    print(f"  Seen geo3 rows:    {f1_seen:.4f}  (n={val_seen_mask.sum():,})")
    print(f"  Unseen geo3 rows:  {f1_unseen:.4f}  (n={val_unseen_mask.sum():,})")
    print(f"  Leakage gap (seen - unseen): {f1_seen - f1_unseen:+.4f}")


# ── 3. Zero-value audit ───────────────────────────────────────────────────────


def check_zero_values(df: pd.DataFrame) -> None:
    section("3. ZERO-VALUE AUDIT")
    for col in ["age", "area_percentage"]:
        zero_mask = df[col] == 0
        n_zero = zero_mask.sum()
        print(f"\n{col}=0: {n_zero:,} rows ({n_zero / len(df) * 100:.2f}%)")
        if n_zero > 0:
            print("  Grade distribution (zero):")
            zero_dist = (
                df[zero_mask]["damage_grade"].value_counts(normalize=True).sort_index()
            )
            for g, p in zero_dist.items():
                print(f"    Grade {g}: {p * 100:.1f}%")
            print("  Grade distribution (non-zero):")
            nonzero_dist = (
                df[~zero_mask]["damage_grade"].value_counts(normalize=True).sort_index()
            )
            for g, p in nonzero_dist.items():
                print(f"    Grade {g}: {p * 100:.1f}%")
            # Mean damage
            mean_zero = df[zero_mask]["damage_grade"].mean()
            mean_nonzero = df[~zero_mask]["damage_grade"].mean()
            print(
                f"  Mean damage: zero={mean_zero:.3f}  nonzero={mean_nonzero:.3f}  diff={mean_zero - mean_nonzero:+.3f}"
            )


# ── 4. Per-geo1 class distribution ───────────────────────────────────────────


def check_geo1_class_distribution(df: pd.DataFrame) -> None:
    section("4. PER-GEO_LEVEL_1 CLASS DISTRIBUTION")
    geo1_stats = (
        df.groupby("geo_level_1_id")["damage_grade"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .rename(columns={1: "Grade1%", 2: "Grade2%", 3: "Grade3%"})
    )
    geo1_stats = geo1_stats * 100
    geo1_stats["mean_damage"] = df.groupby("geo_level_1_id")["damage_grade"].mean()
    geo1_stats["n"] = df.groupby("geo_level_1_id")["damage_grade"].count()
    geo1_stats = geo1_stats.sort_values("Grade3%", ascending=False)
    print(geo1_stats.to_string(float_format=lambda x: f"{x:.1f}"))


# ── 5. Mud_mortar_stone × geo_level_1_id interaction ────────────────────────


def check_mud_geo1_interaction(df: pd.DataFrame) -> None:
    section("5. MUD_MORTAR_STONE × GEO_LEVEL_1 INTERACTION")
    result = (
        df.groupby(["geo_level_1_id", "has_superstructure_mud_mortar_stone"])[
            "damage_grade"
        ]
        .agg(["mean", "count"])
        .unstack("has_superstructure_mud_mortar_stone")
    )
    result.columns = ["mean_nomud", "mean_mud", "n_nomud", "n_mud"]
    result = result.assign(
        delta=result["mean_mud"] - result["mean_nomud"],
        n=result["n_nomud"] + result["n_mud"],
    ).sort_values("delta", ascending=False)
    print("Mean damage: mud=1 vs mud=0 per geo_level_1 (sorted by mud penalty):")
    print(
        result[["mean_nomud", "mean_mud", "delta", "n_mud", "n_nomud"]].to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )
    print(f"\nDelta range: {result['delta'].min():.3f} to {result['delta'].max():.3f}")
    print(f"Std of delta across regions: {result['delta'].std():.3f}")


# ── 6. Slenderness quantitative ───────────────────────────────────────────────


def check_slenderness(df: pd.DataFrame) -> None:
    section("6. SLENDERNESS (height/area) ANALYSIS")
    from sklearn.feature_selection import mutual_info_classif

    df = df.copy()
    df["slenderness"] = df["height_percentage"] / df["area_percentage"].clip(lower=1)

    # Quintile analysis
    df["slenderness_q"] = pd.qcut(
        df["slenderness"], q=5, labels=False, duplicates="drop"
    )
    q_stats = df.groupby("slenderness_q")["damage_grade"].agg(
        mean="mean",
        grade3_pct=lambda x: (x == 3).mean() * 100,
        grade1_pct=lambda x: (x == 1).mean() * 100,
        n="count",
    )
    q_ranges = df.groupby("slenderness_q")["slenderness"].agg(["min", "max"])
    q_stats["range"] = q_ranges.apply(
        lambda r: f"{r['min']:.2f}-{r['max']:.2f}", axis=1
    )
    print("Slenderness quintiles vs damage:")
    print(
        q_stats[["range", "mean", "grade3_pct", "grade1_pct", "n"]].to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # MI
    X = df[["slenderness"]].fillna(0)
    y = df["damage_grade"]
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=SEED)[0]
    print(f"\nSlenderness MI vs damage_grade: {mi:.4f}")
    print("(For reference: count_floors MI~0.030, height_percentage MI~0.020)")


# ── 7. Rare categorical levels — Wilson CI ───────────────────────────────────


def check_rare_categorical_cis(df: pd.DataFrame) -> None:
    section("7. RARE CATEGORICAL LEVELS — WILSON CI ON GRADE 3 RATE")

    cat_cols = [
        "foundation_type",
        "roof_type",
        "ground_floor_type",
        "other_floor_type",
        "position",
        "plan_configuration",
        "land_surface_condition",
        "legal_ownership_status",
    ]

    RARE_THRESHOLD = 500  # levels with fewer than this many samples

    def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
        if n == 0:
            return (0.0, 0.0)
        p_hat = successes / n
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        half_width = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
        return (max(0, center - half_width), min(1, center + half_width))

    rows = []
    for col in cat_cols:
        counts = df[col].value_counts()
        rare_levels = counts[counts < RARE_THRESHOLD].index
        for level in rare_levels:
            mask = df[col] == level
            n = mask.sum()
            g3 = (df[mask]["damage_grade"] == 3).sum()
            g3_rate = g3 / n
            lo, hi = wilson_ci(g3, n)
            rows.append(
                {
                    "column": col,
                    "level": level,
                    "n": n,
                    "grade3_rate": g3_rate,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "ci_width": hi - lo,
                }
            )

    if rows:
        rare_df = pd.DataFrame(rows).sort_values("n")
        print(f"Rare levels (n < {RARE_THRESHOLD}) with Grade 3 rate + 95% Wilson CI:")
        print(rare_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print(f"No levels with fewer than {RARE_THRESHOLD} samples found.")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading data...")
    df = load()
    print(f"Loaded: {df.shape}")

    check_conflicting_labels(df)
    check_zero_values(df)
    check_geo1_class_distribution(df)
    check_mud_geo1_interaction(df)
    check_slenderness(df)
    check_rare_categorical_cis(df)

    # Group leakage last (requires LGB training)
    check_group_leakage(df)

    print("\n\nEXTRA EDA DONE")


if __name__ == "__main__":
    main()
