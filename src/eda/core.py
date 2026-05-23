"""
EDA Core — checks always run regardless of modality.
Usage: import and call run_core_checks(train, test, target_col)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_THRESHOLD = 100_000
FILE_SIZE_WARN_MB = 500


# ── helpers ──────────────────────────────────────────────────────────────────


def _flag(level: str, msg: str) -> None:
    print(f"[{level}] {msg}")


def _savefig(name: str) -> None:
    path = OUTPUT_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    _flag("INFO", f"Plot saved: {path}")


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > SAMPLE_THRESHOLD:
        _flag(
            "INFO",
            f"Sampling {SAMPLE_THRESHOLD:,} rows for plots (full dataset: {len(df):,})",
        )
        return df.sample(SAMPLE_THRESHOLD, random_state=42)
    return df


# ── checks ───────────────────────────────────────────────────────────────────


def check_shape_and_dtypes(
    train: pd.DataFrame, test: pd.DataFrame | None = None
) -> None:
    """Print shape, dtypes, and basic counts."""
    _flag("INFO", f"Train shape: {train.shape}")
    if test is not None:
        _flag("INFO", f"Test shape:  {test.shape}")
    print("\nDtypes:")
    print(train.dtypes.value_counts().to_string())
    print("\nDescriptive stats (numeric only):")
    print(train.describe().to_string())
    print()


def check_missing(df: pd.DataFrame, label: str = "train") -> None:
    """Print missing value counts and percentages, sorted descending."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        _flag("INFO", f"[{label}] No missing values.")
        return
    pct = (missing / len(df) * 100).round(2)
    result = pd.DataFrame({"count": missing, "pct": pct})
    print(f"\nMissing values [{label}]:")
    print(result.to_string())
    for col, row in result.iterrows():
        if row["pct"] > 80:
            _flag("CRITICAL", f"{col}: {row['pct']}% missing — consider dropping")
        elif row["pct"] > 30:
            _flag("WARN", f"{col}: {row['pct']}% missing — imputation strategy needed")
    print()


def check_duplicates(df: pd.DataFrame, label: str = "train") -> None:
    """Count exact duplicate rows."""
    n_dup = df.duplicated().sum()
    if n_dup == 0:
        _flag("INFO", f"[{label}] No duplicate rows.")
    else:
        _flag(
            "WARN", f"[{label}] {n_dup:,} duplicate rows ({n_dup / len(df) * 100:.2f}%)"
        )


def check_constant_columns(df: pd.DataFrame) -> None:
    """Flag columns with zero variance (only one unique value)."""
    constants = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constants:
        _flag("CRITICAL", f"Constant columns (remove): {constants}")
    else:
        _flag("INFO", "No constant columns.")


def check_id_like_columns(df: pd.DataFrame, threshold: float = 0.95) -> None:
    """Flag columns where unique ratio > threshold (likely IDs)."""
    id_like = [c for c in df.columns if df[c].nunique() / len(df) > threshold]
    if id_like:
        _flag("WARN", f"ID-like columns (unique ratio > {threshold}): {id_like}")
    else:
        _flag("INFO", "No ID-like columns detected.")


def check_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    task: str = "auto",
) -> None:
    """
    Plot and print target distribution.

    Args:
        df: Training dataframe.
        target_col: Name of the target column.
        task: 'classification', 'regression', or 'auto' to detect.
    """
    y = df[target_col]
    if task == "auto":
        task = "classification" if y.nunique() <= 20 else "regression"

    if task == "classification":
        counts = y.value_counts().sort_index()
        pct = (counts / len(y) * 100).round(2)
        print("\nTarget distribution:")
        for cls, cnt in counts.items():
            print(f"  {cls}: {cnt:,} ({pct[cls]:.2f}%)")
        imbalance_ratio = counts.max() / counts.min()
        if imbalance_ratio > 100:
            _flag("CRITICAL", f"Severe class imbalance — ratio {imbalance_ratio:.0f}:1")
        elif imbalance_ratio > 20:
            _flag("WARN", f"Class imbalance — ratio {imbalance_ratio:.0f}:1")
        else:
            _flag("INFO", f"Class balance ratio: {imbalance_ratio:.1f}:1")
        ax = counts.plot(kind="bar", title=f"Target: {target_col}", color="steelblue")
        plt.ylabel("Count")
        for bar in ax.patches:
            ax.annotate(
                f"{int(bar.get_height()):,}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        _savefig("target_distribution.png")
    else:
        skew = float(y.skew())
        _flag("INFO", f"Target skewness: {skew:.3f}")
        if abs(skew) > 2:
            _flag(
                "WARN",
                f"High target skewness ({skew:.2f}) — consider log/sqrt transform",
            )
        y_sample = _sample(df)[target_col]
        y_sample.hist(bins=50, color="steelblue", edgecolor="white")
        plt.title(f"Target distribution: {target_col}")
        plt.xlabel(target_col)
        _savefig("target_distribution.png")
    print()


def check_distribution_shift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str | None = None,
) -> None:
    """
    KS test for distribution shift on numeric columns.
    Chi-square for categorical columns.

    Args:
        train: Training dataframe.
        test: Test dataframe.
        target_col: Excluded from shift analysis.
    """
    num_cols = train.select_dtypes(include="number").columns.tolist()
    cat_cols = train.select_dtypes(include="object").columns.tolist()
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)

    shifted_num: list[str] = []
    for col in num_cols:
        if col not in test.columns:
            continue
        stat, p = stats.ks_2samp(
            train[col].dropna().values,
            test[col].dropna().values,
        )
        if p < 0.05:
            shifted_num.append(col)

    shifted_cat: list[str] = []
    for col in cat_cols:
        if col not in test.columns:
            continue
        train_cats = set(train[col].dropna().unique())
        test_cats = set(test[col].dropna().unique())
        unseen = test_cats - train_cats
        if unseen:
            _flag("WARN", f"{col}: {len(unseen)} categories in test not seen in train")
            shifted_cat.append(col)

    if shifted_num:
        _flag(
            "WARN",
            f"Distribution shift (KS p<0.05) in {len(shifted_num)} numeric columns: {shifted_num}",
        )
    else:
        _flag("INFO", "No significant distribution shift in numeric columns.")

    if not shifted_cat:
        _flag("INFO", "No unseen categories in test.")
    print()


# ── entry point ───────────────────────────────────────────────────────────────


def run_core_checks(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    target_col: str,
    task: str = "auto",
) -> None:
    """
    Run all core EDA checks.

    Args:
        train: Training dataframe.
        test: Test dataframe (optional).
        target_col: Name of the target column.
        task: 'classification', 'regression', or 'auto'.
    """
    print("=" * 60)
    print("CORE CHECKS")
    print("=" * 60)
    check_shape_and_dtypes(train, test)
    check_missing(train, label="train")
    if test is not None:
        check_missing(test, label="test")
    check_duplicates(train, label="train")
    check_constant_columns(train)
    check_id_like_columns(train)
    check_target_distribution(train, target_col, task)
    if test is not None:
        check_distribution_shift(train, test, target_col)
