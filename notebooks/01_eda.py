"""Exploratory Data Analysis for Richter's Predictor.

Outputs plots to reports/ (or the path in config output.reports_dir) and prints
summary stats to stdout.

Usage:
    python notebooks/01_eda.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

plt.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train (values + labels merged) and test values."""
    tv = pd.read_csv(cfg["data"]["train_values"], encoding="utf-8")
    tl = pd.read_csv(cfg["data"]["train_labels"], encoding="utf-8")
    df = tv.merge(tl, on=cfg["data"]["id_col"])
    df_test = pd.read_csv(cfg["data"]["test_values"], encoding="utf-8")
    return df, df_test


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def print_basic_stats(df: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """Print shape, dtypes, and missing-value summary."""
    print("=" * 60)
    print("BASIC STATS")
    print("=" * 60)
    print(f"Train shape : {df.shape}")
    print(f"Test shape  : {df_test.shape}")
    print(f"\nDtypes:\n{df.dtypes.value_counts()}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"\nMissing values:\n{missing}")
    else:
        print("\nNo missing values in train.")


def plot_target_distribution(
    df: pd.DataFrame, target_col: str, reports_dir: Path
) -> None:
    """Bar chart of damage_grade distribution."""
    counts = df[target_col].value_counts().sort_index()
    pct = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        counts.index.astype(str), counts.values, color=["#4CAF50", "#FF9800", "#F44336"]
    )
    for grade, cnt, p in zip(counts.index, counts.values, pct.values):
        ax.text(str(grade), cnt + 500, f"{p:.1f}%", ha="center", fontsize=10)
    ax.set_xlabel("damage_grade")
    ax.set_ylabel("count")
    ax.set_title("Target distribution — damage_grade")
    plt.tight_layout()
    out = reports_dir / "target_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"\nTarget distribution:\n{counts}")
    print(f"Saved: {out}")


def plot_geo_cardinality(df: pd.DataFrame, geo_cols: list[str]) -> None:
    """Show unique-value counts for geo level columns."""
    print("\nGeo column cardinalities:")
    for col in geo_cols:
        n = df[col].nunique()
        print(f"  {col}: {n} unique values")


def plot_numeric_distributions(
    df: pd.DataFrame, target_col: str, reports_dir: Path
) -> None:
    """KDE plots of numeric features by damage grade."""
    numeric_cols = [
        "age",
        "area_percentage",
        "height_percentage",
        "count_floors_pre_eq",
        "count_families",
    ]
    colors = {1: "#4CAF50", 2: "#FF9800", 3: "#F44336"}

    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(18, 4))
    for ax, col in zip(axes, numeric_cols):
        for grade, color in colors.items():
            subset = df.loc[df[target_col] == grade, col]
            subset.plot.kde(ax=ax, label=f"grade {grade}", color=color, alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.legend(fontsize=7)
    plt.suptitle("Numeric feature distributions by damage grade", y=1.02)
    plt.tight_layout()
    out = reports_dir / "numeric_distributions.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_categorical_damage_rates(
    df: pd.DataFrame, target_col: str, cat_cols: list[str], reports_dir: Path
) -> None:
    """Heatmap of mean damage grade per category level."""
    n = len(cat_cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cat_cols):
        pivot = (
            df.groupby(col)[target_col].value_counts(normalize=True).unstack().fillna(0)
        )
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            annot=True,
            fmt=".2f",
            cbar=False,
            linewidths=0.5,
        )
        ax.set_title(col)
        ax.set_xlabel("damage_grade")
        ax.set_ylabel("")

    plt.suptitle("Damage grade rate by categorical feature", y=1.02)
    plt.tight_layout()
    out = reports_dir / "cat_damage_rates.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_superstructure_damage(
    df: pd.DataFrame, target_col: str, super_cols: list[str], reports_dir: Path
) -> None:
    """Bar chart: mean damage grade for buildings with/without each superstructure type."""
    records = []
    for col in super_cols:
        label = col.replace("has_superstructure_", "")
        mean_with = df.loc[df[col] == 1, target_col].mean()
        mean_without = df.loc[df[col] == 0, target_col].mean()
        records.append(
            {"superstructure": label, "with": mean_with, "without": mean_without}
        )

    sdf = pd.DataFrame(records).sort_values("with")
    x = np.arange(len(sdf))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(
        x - width / 2,
        sdf["with"],
        width,
        label="has flag=1",
        color="#F44336",
        alpha=0.8,
    )
    ax.barh(
        x + width / 2,
        sdf["without"],
        width,
        label="has flag=0",
        color="#4CAF50",
        alpha=0.8,
    )
    ax.set_yticks(x)
    ax.set_yticklabels(sdf["superstructure"], fontsize=9)
    ax.set_xlabel("mean damage_grade")
    ax.set_title("Mean damage grade by superstructure type")
    ax.legend()
    plt.tight_layout()
    out = reports_dir / "superstructure_damage.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def plot_geo_damage_heatmap(
    df: pd.DataFrame, target_col: str, reports_dir: Path
) -> None:
    """Top-30 geo_level_1 by mean damage grade (bar chart)."""
    geo_damage = (
        df.groupby("geo_level_1_id")[target_col].mean().sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    geo_damage.head(30).plot.bar(ax=ax, color="#FF9800", edgecolor="black", alpha=0.8)
    ax.set_xlabel("geo_level_1_id")
    ax.set_ylabel("mean damage_grade")
    ax.set_title("Mean damage grade by geo_level_1 (top 30)")
    ax.axhline(df[target_col].mean(), color="red", linestyle="--", label="overall mean")
    ax.legend()
    plt.tight_layout()
    out = reports_dir / "geo_damage_heatmap.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def compute_feature_importance_proxy(df: pd.DataFrame, target_col: str) -> None:
    """Print Pearson correlation of numeric features with damage_grade."""
    numeric_cols = [
        "age",
        "area_percentage",
        "height_percentage",
        "count_floors_pre_eq",
        "count_families",
    ]
    print("\nNumeric feature correlations with damage_grade:")
    corr = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
    print(corr.sort_values(key=abs, ascending=False).to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="EDA for Richter-Predictor")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Run full EDA pipeline."""
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    reports_dir = Path(cfg.get("output", {}).get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    df, df_test = load_data(cfg)
    target_col = cfg["data"]["target_col"]
    geo_cols = cfg["features"]["geo_cols"]
    cat_cols_low = cfg["features"]["cat_cols_low"]
    super_cols = cfg["features"]["superstructure_cols"]

    print_basic_stats(df, df_test)
    plot_target_distribution(df, target_col, reports_dir)
    plot_geo_cardinality(df, geo_cols)
    plot_numeric_distributions(df, target_col, reports_dir)
    plot_categorical_damage_rates(df, target_col, cat_cols_low[:4], reports_dir)
    plot_superstructure_damage(df, target_col, super_cols, reports_dir)
    plot_geo_damage_heatmap(df, target_col, reports_dir)
    compute_feature_importance_proxy(df, target_col)

    print(f"\nEDA complete. Plots saved to {reports_dir}/")


if __name__ == "__main__":
    main()
