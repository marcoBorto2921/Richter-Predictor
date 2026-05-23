"""
EDA — Richter's Predictor
Run: .venv/Scripts/python notebooks/02_eda_full.py 2>&1 | tee outputs/eda/eda_raw.log
Outputs: stdout flags + plots in outputs/eda/
"""

import sys
import warnings
from itertools import combinations
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

TRAIN_VALUES_PATH = "data/raw/train_values.csv"
TRAIN_LABELS_PATH = "data/raw/train_labels.csv"
TEST_PATH = "data/raw/test_values.csv"
TARGET_COL = "damage_grade"
ID_COL = "building_id"
TASK = "classification"
MODALITY = "tabular"

# ── helpers ───────────────────────────────────────────────────────────────────


def _flag(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", flush=True)


def _savefig(name: str) -> None:
    path = Path("outputs/eda") / name
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    _flag("INFO", f"Plot saved: {path}")


# ── load + join ───────────────────────────────────────────────────────────────


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge train_values + train_labels; load test_values."""
    values = pd.read_csv(TRAIN_VALUES_PATH, encoding="utf-8")
    labels = pd.read_csv(TRAIN_LABELS_PATH, encoding="utf-8")
    train = values.merge(labels, on=ID_COL)
    test = pd.read_csv(TEST_PATH, encoding="utf-8")
    _flag("INFO", f"Train: {train.shape} | Test: {test.shape}")
    return train, test


# ── Step 1b — Governing Equation Search ──────────────────────────────────────


def check_governing_equations(train: pd.DataFrame) -> None:
    """Search for near-perfect linear relationships between numeric features."""
    print("\n" + "=" * 60)
    print("GOVERNING EQUATION SEARCH")
    print("=" * 60)
    num_cols = [
        c
        for c in train.select_dtypes(include="number").columns
        if c not in (TARGET_COL, ID_COL)
    ]
    results = []
    for a, b in combinations(num_cols, 2):
        sub = train[[a, b]].dropna()
        if len(sub) < 100:
            continue
        corr = float(np.corrcoef(sub[a].values, sub[b].values)[0, 1])
        if abs(corr) > 0.99:
            results.append((a, b, corr))
    if results:
        _flag(
            "CRITICAL",
            "Near-perfect linear relationships found — may indicate leakage or redundancy:",
        )
        for a, b, r in sorted(results, key=lambda x: -abs(x[2])):
            _flag("CRITICAL", f"  r={r:+.4f}  {a}  <->  {b}")
    else:
        _flag("INFO", "No governing equations found (no |r| > 0.99 pairs).")
    print()


# ── Step 1c — Feature Recovery Check ─────────────────────────────────────────


def check_feature_recovery(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Check for train-only columns missing from test."""
    print("\n" + "=" * 60)
    print("FEATURE RECOVERY CHECK")
    print("=" * 60)
    train_only = set(train.columns) - set(test.columns) - {TARGET_COL, ID_COL}
    if not train_only:
        _flag("INFO", "All train features are present in test. No recovery needed.")
    else:
        for col in sorted(train_only):
            _flag(
                "WARN", f"Train-only column: {col} — not in test. Check if recoverable."
            )
    print()


# ── Step 1d — Domain Hypotheses ───────────────────────────────────────────────


def print_domain_hypotheses() -> None:
    """Print the domain hypotheses to verify during analysis."""
    print("\n" + "=" * 60)
    print("DOMAIN HYPOTHESES (pre-registered)")
    print("=" * 60)
    hypotheses = [
        {
            "id": "H1",
            "mechanism": "Older buildings have weaker materials due to decay -> higher damage grade",
            "feature": "age",
            "verification": "KDE of age by damage_grade — Grade 3 should right-skew toward higher age",
        },
        {
            "id": "H2",
            "mechanism": "Mud/adobe superstructure is brittle -> collapses at Grade 3 disproportionately",
            "feature": "has_superstructure_adobe_mud, has_superstructure_mud_mortar_stone",
            "verification": "Stacked bar: % Grade 3 per superstructure type — adobe/mud should be highest",
        },
        {
            "id": "H3",
            "mechanism": "Geographic location captures soil amplification + fault proximity -> dominant predictor",
            "feature": "geo_level_1/2/3_id",
            "verification": "eta-squared of geo_level_3 vs damage_grade — expect R² > 0.15",
        },
        {
            "id": "H4",
            "mechanism": "RC-engineered buildings resist shaking through ductility -> mostly Grade 1",
            "feature": "has_superstructure_rc_engineered",
            "verification": "Grade distribution for rc_engineered=1 vs 0 — should show strong Grade 1 enrichment",
        },
        {
            "id": "H5",
            "mechanism": "Taller + weaker material combination is the worst case (overturning x brittleness)",
            "feature": "count_floors_pre_eq x superstructure material flags",
            "verification": "Mean damage grade: tall buildings (floors >= 3) with mud/adobe vs RC — expect 2+ grade gap",
        },
    ]
    for h in hypotheses:
        print(f"\n  {h['id']}: {h['mechanism']}")
        print(f"  Feature: {h['feature']}")
        print(f"  Verify:  {h['verification']}")
    print()


# ── Domain EDA Plan checks ────────────────────────────────────────────────────


def check_near_constant_flags(train: pd.DataFrame) -> None:
    """Audit near-constant secondary use flags on full dataset."""
    print("\n" + "=" * 60)
    print("NEAR-CONSTANT COLUMN AUDIT")
    print("=" * 60)
    suspect_cols = [
        "has_secondary_use_institution",
        "has_secondary_use_school",
        "has_secondary_use_health_post",
        "has_secondary_use_gov_office",
        "has_secondary_use_use_police",
        "has_superstructure_bamboo",
        "has_superstructure_other",
    ]
    for col in suspect_cols:
        if col not in train.columns:
            continue
        vc = train[col].value_counts(normalize=True)
        dominant_val = vc.index[0]
        dominant_pct = vc.iloc[0] * 100
        positive_pct = train[col].mean() * 100
        flag = (
            "CRITICAL"
            if dominant_pct > 99.9
            else ("WARN" if dominant_pct > 99 else "INFO")
        )
        _flag(
            flag,
            f"{col}: {dominant_pct:.2f}% = {dominant_val} | positive rate: {positive_pct:.2f}%",
        )
    print()


def check_geo_level_analysis(train: pd.DataFrame) -> None:
    """Mean damage grade per geo level, variance explained, group sizes."""
    print("\n" + "=" * 60)
    print("GEO-LEVEL ANALYSIS")
    print("=" * 60)

    # Hierarchical consistency
    check = train.groupby("geo_level_3_id")["geo_level_2_id"].nunique()
    broken = (check > 1).sum()
    if broken > 0:
        _flag(
            "WARN",
            f"geo_level_3 not fully nested in geo_level_2: {broken} geo3 IDs have >1 geo2 parent",
        )
    else:
        _flag(
            "INFO",
            "geo_level_3 is fully nested within geo_level_2 (hierarchical consistency OK)",
        )

    check2 = train.groupby("geo_level_2_id")["geo_level_1_id"].nunique()
    broken2 = (check2 > 1).sum()
    if broken2 > 0:
        _flag(
            "WARN",
            f"geo_level_2 not fully nested in geo_level_1: {broken2} geo2 IDs have >1 geo1 parent",
        )
    else:
        _flag(
            "INFO",
            "geo_level_2 is fully nested within geo_level_1 (hierarchical consistency OK)",
        )

    # Variance explained by geo_level_3 (eta-squared)
    grand_mean = train[TARGET_COL].mean()
    ss_total = ((train[TARGET_COL] - grand_mean) ** 2).sum()
    group_means = train.groupby("geo_level_3_id")[TARGET_COL].transform("mean")
    ss_between = ((group_means - grand_mean) ** 2).sum()
    eta2 = ss_between / ss_total
    _flag(
        "INFO",
        f"geo_level_3 eta-squared (variance explained): {eta2:.4f} ({eta2 * 100:.1f}% of target variance)",
    )

    # Group sizes for geo_level_3
    geo3_sizes = train.groupby("geo_level_3_id").size()
    print(
        f"\n  geo_level_3 group sizes: min={geo3_sizes.min()}, median={geo3_sizes.median():.0f}, "
        f"max={geo3_sizes.max()}, mean={geo3_sizes.mean():.1f}"
    )
    small_groups = (geo3_sizes < 5).sum()
    _flag(
        "WARN" if small_groups > 0 else "INFO",
        f"geo_level_3 groups with < 5 samples: {small_groups} ({small_groups / len(geo3_sizes) * 100:.1f}% of all groups)",
    )

    # Mean damage per geo_level_1
    geo1_damage = (
        train.groupby("geo_level_1_id")[TARGET_COL]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    geo1_damage.columns = ["geo1", "mean_damage", "std_damage", "count"]
    geo1_damage = geo1_damage.sort_values("mean_damage")
    print("\n  Mean damage grade per geo_level_1 (sorted):")
    for _, row in geo1_damage.iterrows():
        print(
            f"    geo1={int(row['geo1'])}: mean={row['mean_damage']:.3f} ± {row['std_damage']:.3f} (n={int(row['count'])})"
        )

    # Plot: geo1 mean damage
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].bar(range(len(geo1_damage)), geo1_damage["mean_damage"], color="steelblue")
    axes[0].errorbar(
        range(len(geo1_damage)),
        geo1_damage["mean_damage"],
        yerr=geo1_damage["std_damage"],
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[0].set_xticks(range(len(geo1_damage)))
    axes[0].set_xticklabels(geo1_damage["geo1"].astype(int), rotation=90, fontsize=7)
    axes[0].set_ylabel("Mean damage grade")
    axes[0].set_title("Mean Damage Grade per geo_level_1")
    axes[0].axhline(
        grand_mean,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Overall mean={grand_mean:.2f}",
    )
    axes[0].legend()

    # Geo3 group size distribution
    axes[1].hist(geo3_sizes.values, bins=50, color="coral", edgecolor="white")
    axes[1].set_xlabel("Buildings per geo_level_3 group")
    axes[1].set_ylabel("Count of geo3 groups")
    axes[1].set_title("geo_level_3 Group Size Distribution")
    axes[1].axvline(5, color="red", linestyle="--", label="n=5 threshold")
    axes[1].legend()
    plt.tight_layout()
    _savefig("geo_level_analysis.png")
    print()


def check_superstructure_analysis(train: pd.DataFrame) -> None:
    """Co-occurrence, material quality score, stacked bar vs damage."""
    print("\n" + "=" * 60)
    print("SUPERSTRUCTURE MATERIAL ANALYSIS")
    print("=" * 60)

    super_cols = [c for c in train.columns if c.startswith("has_superstructure_")]

    # Prevalence per flag
    print("\n  Superstructure flag prevalence (% of buildings):")
    prevalence = train[super_cols].mean().sort_values(ascending=False)
    for col, pct in prevalence.items():
        label = col.replace("has_superstructure_", "")
        print(f"    {label:<30} {pct * 100:5.1f}%")

    # % Grade 3 per flag
    print("\n  % Grade 3 (destruction) per superstructure type (when flag=1):")
    grade3_rate = {}
    for col in super_cols:
        subset = train[train[col] == 1]
        if len(subset) < 50:
            continue
        g3_pct = (subset[TARGET_COL] == 3).mean() * 100
        g1_pct = (subset[TARGET_COL] == 1).mean() * 100
        grade3_rate[col] = g3_pct
        label = col.replace("has_superstructure_", "")
        print(
            f"    {label:<30} Grade3={g3_pct:.1f}%  Grade1={g1_pct:.1f}%  (n={len(subset):,})"
        )

    # Count of active flags per building
    train["_n_super"] = train[super_cols].sum(axis=1)
    flag_vs_damage = train.groupby("_n_super")[TARGET_COL].agg(["mean", "count"])
    print("\n  Mean damage grade by number of active superstructure flags:")
    print(flag_vs_damage.to_string())
    train.drop(columns=["_n_super"], inplace=True)

    # Material quality score (max ordinal material present)
    material_rank = {
        "has_superstructure_rc_engineered": 6,
        "has_superstructure_rc_non_engineered": 5,
        "has_superstructure_cement_mortar_brick": 4,
        "has_superstructure_cement_mortar_stone": 3,
        "has_superstructure_mud_mortar_brick": 2,
        "has_superstructure_mud_mortar_stone": 2,
        "has_superstructure_stone_flag": 1,
        "has_superstructure_adobe_mud": 0,
        "has_superstructure_bamboo": 1,
        "has_superstructure_timber": 1,
        "has_superstructure_other": 1,
    }
    score = pd.Series(0, index=train.index)
    for col, rank in material_rank.items():
        if col in train.columns:
            score = score.where(
                train[col] == 0, other=score.where(score >= rank, other=rank)
            )
    train["_mat_score"] = score
    mat_vs_damage = train.groupby("_mat_score")[TARGET_COL].agg(["mean", "count"])
    print("\n  Mean damage grade by material quality score (0=worst, 6=best):")
    print(mat_vs_damage.to_string())
    train.drop(columns=["_mat_score"], inplace=True)

    # Plot: % Grade 3 per superstructure (sorted)
    if grade3_rate:
        labels = [k.replace("has_superstructure_", "") for k in grade3_rate]
        values = list(grade3_rate.values())
        sorted_pairs = sorted(zip(values, labels))
        vals_s, labs_s = zip(*sorted_pairs)
        overall_g3 = (train[TARGET_COL] == 3).mean() * 100

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        colors = ["coral" if v > overall_g3 else "steelblue" for v in vals_s]
        axes[0].barh(range(len(labs_s)), vals_s, color=colors)
        axes[0].axvline(
            overall_g3,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Overall {overall_g3:.1f}%",
        )
        axes[0].set_yticks(range(len(labs_s)))
        axes[0].set_yticklabels(labs_s, fontsize=8)
        axes[0].set_xlabel("% Grade 3 (near-total destruction)")
        axes[0].set_title(
            "Grade 3 Rate per Superstructure Type\n(coral = above average)"
        )
        axes[0].legend()

        # Co-occurrence heatmap
        co = train[super_cols].T.dot(train[super_cols])
        np.fill_diagonal(co.values, 0)
        short_labels = [c.replace("has_superstructure_", "")[:12] for c in super_cols]
        im = axes[1].imshow(co.values, cmap="Blues", aspect="auto")
        axes[1].set_xticks(range(len(super_cols)))
        axes[1].set_yticks(range(len(super_cols)))
        axes[1].set_xticklabels(short_labels, rotation=90, fontsize=7)
        axes[1].set_yticklabels(short_labels, fontsize=7)
        axes[1].set_title("Superstructure Co-occurrence Matrix")
        plt.colorbar(im, ax=axes[1])
        plt.tight_layout()
        _savefig("superstructure_analysis.png")
    print()


def check_structural_numerics(train: pd.DataFrame) -> None:
    """Outlier in age, slenderness scatter, floor-height ratio."""
    print("\n" + "=" * 60)
    print("STRUCTURAL NUMERIC ANALYSIS")
    print("=" * 60)

    # Age outliers
    age_p99 = train["age"].quantile(0.99)
    age_extreme = (train["age"] > 200).sum()
    _flag(
        "WARN" if age_extreme > 0 else "INFO",
        f"age: max={train['age'].max()}, p99={age_p99:.0f}, "
        f"buildings with age > 200: {age_extreme:,} ({age_extreme / len(train) * 100:.2f}%)",
    )

    # Correlation matrix for structural numerics
    struct_cols = [
        "age",
        "count_floors_pre_eq",
        "area_percentage",
        "height_percentage",
        "count_families",
    ]
    corr = train[struct_cols].corr(method="pearson")
    print("\n  Pearson correlation matrix (structural numerics):")
    print(corr.round(3).to_string())

    high_pairs = []
    for i in range(len(struct_cols)):
        for j in range(i + 1, len(struct_cols)):
            c = abs(corr.iloc[i, j])
            if c > 0.5:
                high_pairs.append(f"{struct_cols[i]} — {struct_cols[j]}: {c:.3f}")
    if high_pairs:
        _flag("INFO", f"Notable correlations (|r| > 0.5): {high_pairs}")

    # Slenderness: height/area scatter colored by damage grade
    sample = train.sample(min(50_000, len(train)), random_state=42)
    slenderness = sample["height_percentage"] / (sample["area_percentage"] + 1)
    print(
        f"\n  Slenderness (height/area): min={slenderness.min():.2f}, "
        f"max={slenderness.max():.2f}, median={slenderness.median():.2f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # age distribution per damage grade
    for grade, color in zip([1, 2, 3], ["green", "orange", "red"]):
        subset = sample[sample[TARGET_COL] == grade]["age"].clip(upper=200)
        axes[0].hist(
            subset,
            bins=40,
            alpha=0.5,
            color=color,
            label=f"Grade {grade}",
            density=True,
        )
    axes[0].set_xlabel("Age (capped at 200)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Age Distribution by Damage Grade")
    axes[0].legend()

    # slenderness vs damage
    grade_colors = {1: "green", 2: "orange", 3: "red"}
    for grade in [1, 2, 3]:
        mask = sample[TARGET_COL] == grade
        axes[1].scatter(
            sample.loc[mask, "area_percentage"],
            sample.loc[mask, "height_percentage"],
            alpha=0.15,
            s=5,
            color=grade_colors[grade],
            label=f"Grade {grade}",
        )
    axes[1].set_xlabel("area_percentage")
    axes[1].set_ylabel("height_percentage")
    axes[1].set_title("Area vs Height by Damage Grade")
    axes[1].legend(markerscale=3)

    # floors distribution per grade
    floor_damage = (
        train.groupby(["count_floors_pre_eq", TARGET_COL]).size().unstack(fill_value=0)
    )
    floor_damage_pct = floor_damage.div(floor_damage.sum(axis=1), axis=0) * 100
    floor_damage_pct.plot(
        kind="bar",
        stacked=True,
        ax=axes[2],
        color=["green", "orange", "red"],
        alpha=0.8,
    )
    axes[2].set_xlabel("Number of floors")
    axes[2].set_ylabel("% of buildings")
    axes[2].set_title("Damage Grade Distribution by Floor Count")
    axes[2].legend(title="Grade", labels=["1", "2", "3"])
    axes[2].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    _savefig("structural_numerics.png")
    print()


def check_feature_interactions(train: pd.DataFrame) -> None:
    """Domain-motivated interaction checks: agexfoundation, floorsxmaterial, planxdamage."""
    print("\n" + "=" * 60)
    print("FEATURE INTERACTIONS (DOMAIN-MOTIVATED)")
    print("=" * 60)

    # age x foundation_type heatmap
    train["_age_bin"] = pd.cut(
        train["age"],
        bins=[0, 10, 25, 50, 100, 1000],
        labels=["0-10", "11-25", "26-50", "51-100", "100+"],
    )
    age_found = (
        train.groupby(["_age_bin", "foundation_type"])[TARGET_COL].mean().unstack()
    )
    print("\n  Mean damage grade by (age_bin x foundation_type):")
    print(age_found.round(3).to_string())
    train.drop(columns=["_age_bin"], inplace=True)

    # plan_configuration damage rates
    plan_damage = (
        train.groupby("plan_configuration")[TARGET_COL]
        .agg(
            grade3_pct=lambda x: (x == 3).mean() * 100,
            grade1_pct=lambda x: (x == 1).mean() * 100,
            mean_damage="mean",
            count="count",
        )
        .sort_values("grade3_pct", ascending=False)
    )
    print("\n  Damage by plan_configuration (sorted by % Grade 3):")
    print(plan_damage.round(2).to_string())
    # flag: is 'd' (rectangular) actually safest?
    if "d" in plan_damage.index:
        d_rank = list(plan_damage.index).index("d")
        _flag(
            "INFO",
            f"plan='d' (rectangular) ranks #{len(plan_damage) - d_rank} for % Grade 3 "
            f"(lower is safer, rank 1 = least damage)",
        )

    # position x floors
    pos_floors = (
        train.groupby(["position", "count_floors_pre_eq"])[TARGET_COL].mean().unstack()
    )
    print("\n  Mean damage grade by (position x floors):")
    print(pos_floors.round(3).to_string())

    # land_surface_condition x foundation_type
    slope_found = (
        train.groupby(["land_surface_condition", "foundation_type"])[TARGET_COL]
        .mean()
        .unstack()
    )
    print("\n  Mean damage grade by (land_surface x foundation_type):")
    print(slope_found.round(3).to_string())

    # Plot: agexfoundation heatmap and plan_config bar
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    if not age_found.empty:
        im = axes[0].imshow(
            age_found.values, cmap="RdYlGn_r", aspect="auto", vmin=1.5, vmax=2.8
        )
        axes[0].set_xticks(range(len(age_found.columns)))
        axes[0].set_yticks(range(len(age_found.index)))
        axes[0].set_xticklabels(age_found.columns, fontsize=9)
        axes[0].set_yticklabels(age_found.index, fontsize=9)
        plt.colorbar(im, ax=axes[0])
        axes[0].set_xlabel("Foundation type")
        axes[0].set_ylabel("Age bin")
        axes[0].set_title("Mean Damage: Age x Foundation Type")
        for i in range(len(age_found.index)):
            for j in range(len(age_found.columns)):
                val = age_found.iloc[i, j]
                if not np.isnan(val):
                    axes[0].text(
                        j, i, f"{val:.2f}", ha="center", va="center", fontsize=7
                    )

    plan_damage["grade3_pct"].sort_values().plot(kind="barh", ax=axes[1], color="coral")
    axes[1].axvline(
        (train[TARGET_COL] == 3).mean() * 100,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Overall Grade 3 rate",
    )
    axes[1].set_xlabel("% Grade 3")
    axes[1].set_title("Grade 3 Rate by Plan Configuration")
    axes[1].legend()
    plt.tight_layout()
    _savefig("feature_interactions.png")
    print()


def check_categorical_damage_rates(train: pd.DataFrame) -> None:
    """Bar charts: mean damage per category for all low-card cats."""
    print("\n" + "=" * 60)
    print("CATEGORICAL FEATURE DAMAGE RATES")
    print("=" * 60)

    cat_cols = [
        c for c in train.select_dtypes(include="object").columns if c != TARGET_COL
    ]
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        if i >= len(axes):
            break
        damage_by_cat = (
            train.groupby(col)[TARGET_COL]
            .agg(
                grade3_pct=lambda x: (x == 3).mean() * 100,
                grade1_pct=lambda x: (x == 1).mean() * 100,
                mean_damage="mean",
                count="count",
            )
            .sort_values("grade3_pct", ascending=False)
        )
        print(f"\n  {col}:")
        print(damage_by_cat.round(2).to_string())

        # stacked bar
        damage_dist = (
            train.groupby(col)[TARGET_COL]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
            * 100
        )
        damage_dist = damage_dist.reindex(columns=[1, 2, 3])
        damage_dist.plot(
            kind="bar",
            stacked=True,
            ax=axes[i],
            color=["green", "orange", "red"],
            alpha=0.8,
            legend=(i == 0),
        )
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=45, labelsize=8)
        axes[i].set_ylabel("% buildings" if i % 4 == 0 else "")

    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Damage Grade Distribution by Categorical Feature", fontsize=12)
    plt.tight_layout()
    _savefig("categorical_damage_rates.png")
    print()


def check_quick_lgb_importance(train: pd.DataFrame) -> None:
    """Quick LightGBM (200 rounds) for feature importance + confusion matrix."""
    print("\n" + "=" * 60)
    print("QUICK LGB IMPORTANCE + CONFUSION MATRIX")
    print("=" * 60)

    try:
        import lightgbm as lgb
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        _flag("WARN", "lightgbm or sklearn not installed — skipping quick LGB check")
        return

    feature_cols = [c for c in train.columns if c not in (TARGET_COL, ID_COL)]
    X = train[feature_cols].copy()
    y = train[TARGET_COL].copy()

    # Encode categoricals
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=63,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )

    preds = clf.predict(X_val)
    f1 = f1_score(y_val, preds, average="micro")
    _flag("INFO", f"Quick LGB F1-micro (20% holdout): {f1:.4f}")

    print("\n  Classification report:")
    print(classification_report(y_val, preds, digits=3))

    # Per-class recall analysis
    cm = confusion_matrix(y_val, preds, labels=[1, 2, 3])
    print("\n  Confusion matrix (rows=true, cols=pred):")
    cm_df = pd.DataFrame(
        cm, index=["True 1", "True 2", "True 3"], columns=["Pred 1", "Pred 2", "Pred 3"]
    )
    print(cm_df.to_string())

    grade1_recall = cm[0, 0] / cm[0].sum()
    grade3_recall = cm[2, 2] / cm[2].sum()
    _flag(
        "WARN" if grade1_recall < 0.5 else "INFO",
        f"Grade 1 recall: {grade1_recall:.3f} — main F1-micro bottleneck if low",
    )
    _flag("INFO", f"Grade 3 recall: {grade3_recall:.3f}")
    grade23_confusion = cm[1, 2] + cm[2, 1]
    _flag(
        "INFO",
        f"Grade 2↔3 confusion: {grade23_confusion:,} misclassifications (expected from label noise)",
    )

    # Feature importance (split + gain)
    imp_split = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    print("\n  Top 20 features by split importance:")
    print(imp_split.head(20).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    imp_split.head(20).sort_values().plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("LGB Feature Importance (split, top 20)")
    axes[0].set_xlabel("Importance")

    # Confusion matrix heatmap
    im = axes[1].imshow(cm, cmap="Blues", aspect="auto")
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_xticklabels(["Pred 1", "Pred 2", "Pred 3"])
    axes[1].set_yticklabels(["True 1", "True 2", "True 3"])
    axes[1].set_title(f"Confusion Matrix (F1-micro={f1:.4f})")
    for i in range(3):
        for j in range(3):
            axes[1].text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=10,
            )
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    _savefig("lgb_importance_confusion.png")

    # Misclassified Grade 1 profile
    val_df = X_val.copy()
    val_df["true"] = y_val.values
    val_df["pred"] = preds
    g1_as_2 = val_df[(val_df["true"] == 1) & (val_df["pred"] == 2)]
    g1_correct = val_df[(val_df["true"] == 1) & (val_df["pred"] == 1)]
    if len(g1_as_2) > 10:
        print(f"\n  Grade 1 misclassified as Grade 2: {len(g1_as_2):,} samples")
        print("  Feature means: misclassified vs correctly classified Grade 1:")
        compare_cols = [
            "age",
            "count_floors_pre_eq",
            "area_percentage",
            "height_percentage",
        ]
        for col in compare_cols:
            m_wrong = g1_as_2[col].mean()
            m_right = g1_correct[col].mean() if len(g1_correct) > 0 else float("nan")
            print(f"    {col}: wrong={m_wrong:.1f}, correct={m_right:.1f}")
    print()


def check_train_test_geo_coverage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Check how many geo IDs in test are unseen in train."""
    print("\n" + "=" * 60)
    print("GEO COVERAGE: TRAIN vs TEST")
    print("=" * 60)

    for col in ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]:
        train_ids = set(train[col].unique())
        test_ids = set(test[col].unique())
        unseen = test_ids - train_ids

        pct_unseen = len(unseen) / len(test_ids) * 100 if test_ids else 0
        flag = "CRITICAL" if pct_unseen > 10 else ("WARN" if pct_unseen > 1 else "INFO")
        _flag(
            flag,
            f"{col}: {len(train_ids)} train IDs, {len(test_ids)} test IDs, "
            f"{len(unseen)} unseen in test ({pct_unseen:.1f}%) — target encoding fallback needed for these",
        )
    print()


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("EDA START — Richter's Predictor")
    print("=" * 60)

    train, test = load_data()

    # Standard core + tabular checks
    from src.eda.core import run_core_checks
    from src.eda.tabular import run_tabular_checks

    try:
        run_core_checks(
            train.drop(columns=[ID_COL]),
            test.drop(columns=[ID_COL]),
            TARGET_COL,
            task=TASK,
        )
    except Exception as exc:
        print(f"[ERROR] run_core_checks failed: {exc}", flush=True)

    try:
        run_tabular_checks(
            train.drop(columns=[ID_COL]),
            test.drop(columns=[ID_COL]),
            TARGET_COL,
            task=TASK,
        )
    except Exception as exc:
        print(f"[ERROR] run_tabular_checks failed: {exc}", flush=True)

    # Step 1b — governing equations
    check_governing_equations(train)

    # Step 1c — feature recovery
    check_feature_recovery(train, test)

    # Step 1d — domain hypotheses
    print_domain_hypotheses()

    # EDA plan domain-specific checks
    check_near_constant_flags(train)
    check_train_test_geo_coverage(train, test)
    check_geo_level_analysis(train)
    check_superstructure_analysis(train)
    check_structural_numerics(train)
    check_feature_interactions(train)
    check_categorical_damage_rates(train)
    check_quick_lgb_importance(train)

    print("=" * 60)
    print("EDA DONE — check outputs/eda/ for plots")
    print("=" * 60)


if __name__ == "__main__":
    main()
