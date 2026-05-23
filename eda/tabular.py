"""
EDA Tabular — checks for tabular data (numerical + categorical features).
Usage: import and call run_tabular_checks(train, test, target_col, task)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_THRESHOLD = 100_000
HIGH_CARDINALITY_THRESHOLD = 50
RARE_VALUE_THRESHOLD = 0.01  # 1%
LEAKAGE_CORRELATION_THRESHOLD = 0.95
HIGH_SKEW_THRESHOLD = 2.0
HIGH_CORR_THRESHOLD = 0.9


def _flag(level: str, msg: str) -> None:
    print(f"[{level}] {msg}")


def _savefig(name: str) -> None:
    path = OUTPUT_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    _flag("INFO", f"Plot saved: {path}")


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > SAMPLE_THRESHOLD:
        return df.sample(SAMPLE_THRESHOLD, random_state=42)
    return df


# ── numerical ─────────────────────────────────────────────────────────────────


def check_numerical_distributions(df: pd.DataFrame, target_col: str) -> None:
    """Histograms + skewness flags for all numeric columns."""
    num_cols = [
        c for c in df.select_dtypes(include="number").columns if c != target_col
    ]
    if not num_cols:
        return

    sample = _sample(df)
    ncols = 4
    nrows = int(np.ceil(len(num_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        axes[i].hist(
            sample[col].dropna(), bins=40, color="steelblue", edgecolor="white"
        )
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=7)

    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numerical Feature Distributions", fontsize=11)
    plt.tight_layout()
    _savefig("numerical_distributions.png")

    skewed = []
    for col in num_cols:
        skew = float(df[col].skew())
        if abs(skew) > HIGH_SKEW_THRESHOLD:
            skewed.append(f"{col} (skew={skew:.2f})")
    if skewed:
        _flag("WARN", f"High skewness (>{HIGH_SKEW_THRESHOLD}): {skewed}")
    else:
        _flag("INFO", "No high skewness detected in numerical features.")


def check_outliers(df: pd.DataFrame, target_col: str) -> None:
    """IQR-based outlier count per numerical column."""
    num_cols = [
        c for c in df.select_dtypes(include="number").columns if c != target_col
    ]
    outlier_report: list[str] = []
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        pct = n_out / len(df) * 100
        if pct > 5:
            outlier_report.append(f"{col}: {n_out:,} ({pct:.1f}%)")

    if outlier_report:
        _flag("WARN", f"Columns with >5% outliers (IQR): {outlier_report}")
    else:
        _flag("INFO", "No columns with >5% outliers detected.")


def check_correlations(df: pd.DataFrame, target_col: str) -> None:
    """Pearson correlation heatmap + multicollinearity flags."""
    num_cols = [
        c for c in df.select_dtypes(include="number").columns if c != target_col
    ]
    if len(num_cols) < 2:
        return

    corr = df[num_cols].corr(method="pearson")

    fig_size = max(8, len(num_cols) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(num_cols, fontsize=7)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    _savefig("correlation_heatmap.png")

    # multicollinearity
    high_corr_pairs: list[str] = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c = abs(corr.iloc[i, j])
            if c > HIGH_CORR_THRESHOLD:
                high_corr_pairs.append(f"{num_cols[i]} — {num_cols[j]}: {c:.3f}")
    if high_corr_pairs:
        _flag("WARN", f"High collinearity (>{HIGH_CORR_THRESHOLD}): {high_corr_pairs}")
    else:
        _flag("INFO", "No high collinearity detected.")


def check_mutual_information(
    df: pd.DataFrame,
    target_col: str,
    task: str = "auto",
    top_n: int = 20,
) -> None:
    """
    Mutual information between each feature and the target.

    Args:
        df: Training dataframe.
        target_col: Name of the target column.
        task: 'classification', 'regression', or 'auto'.
        top_n: Number of top features to display.
    """
    y = df[target_col]
    if task == "auto":
        task = "classification" if y.nunique() <= 20 else "regression"

    num_cols = [
        c for c in df.select_dtypes(include="number").columns if c != target_col
    ]
    if not num_cols:
        return

    X = df[num_cols].fillna(df[num_cols].median())

    mi_fn = mutual_info_classif if task == "classification" else mutual_info_regression
    mi_scores = mi_fn(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=num_cols).sort_values(ascending=False)

    print(f"\nTop {top_n} features by mutual information with target:")
    print(mi_series.head(top_n).round(4).to_string())

    # leakage check
    # (correlation-based as cross-check)
    corr_with_target = df[num_cols].corrwith(y).abs()
    leakage_suspects = corr_with_target[
        corr_with_target > LEAKAGE_CORRELATION_THRESHOLD
    ].index.tolist()
    if leakage_suspects:
        _flag(
            "CRITICAL",
            f"Leakage suspects (|corr with target| > {LEAKAGE_CORRELATION_THRESHOLD}): "
            f"{leakage_suspects} — verify these are available at inference time",
        )

    # bar chart top N
    mi_series.head(top_n).sort_values().plot(
        kind="barh", figsize=(8, max(4, top_n * 0.3)), color="steelblue"
    )
    plt.title(f"Mutual Information with Target (top {top_n})")
    plt.xlabel("MI score")
    plt.tight_layout()
    _savefig("mutual_information.png")
    print()


# ── categorical ───────────────────────────────────────────────────────────────


def check_categorical_features(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    target_col: str,
) -> None:
    """Cardinality, rare values, and chi-square vs target."""
    cat_cols = [
        c for c in train.select_dtypes(include="object").columns if c != target_col
    ]
    if not cat_cols:
        _flag("INFO", "No categorical columns found.")
        return

    print("\nCategorical features:")
    for col in cat_cols:
        n_unique = train[col].nunique()
        top_val, top_freq = (
            train[col].value_counts().iloc[0],
            train[col].value_counts().iloc[0] / len(train),
        )
        print(f"  {col}: {n_unique} unique | top='{top_val}' ({top_freq:.1%})")

        if n_unique > HIGH_CARDINALITY_THRESHOLD:
            _flag("WARN", f"{col}: high cardinality ({n_unique} unique values)")

        # rare values
        freq = train[col].value_counts(normalize=True)
        rare = freq[freq < RARE_VALUE_THRESHOLD]
        if not rare.empty:
            _flag(
                "WARN",
                f"{col}: {len(rare)} rare values (<{RARE_VALUE_THRESHOLD:.0%} freq) — "
                "risky for encoding across folds",
            )

        # unseen in test
        if test is not None and col in test.columns:
            unseen = set(test[col].dropna().unique()) - set(
                train[col].dropna().unique()
            )
            if unseen:
                _flag("WARN", f"{col}: {len(unseen)} categories in test not in train")

    # chi-square vs target (classification only)
    y = train[target_col]
    if y.nunique() <= 20:
        print("\nChi-square vs target (p-value, lower = more associated):")
        chi_results: list[tuple[str, float]] = []
        for col in cat_cols:
            try:
                contingency = pd.crosstab(train[col], y)
                chi2, p, *_ = stats.chi2_contingency(contingency)
                chi_results.append((col, p))
            except Exception:
                pass
        chi_results.sort(key=lambda x: x[1])
        for col, p in chi_results[:10]:
            print(f"  {col}: p={p:.4f}")
    print()


# ── shift analysis ────────────────────────────────────────────────────────────


def check_train_test_shift(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    target_col: str,
) -> None:
    """Adversarial validation + KS test + PSI to detect train/test distribution shift.

    Args:
        train: Training dataframe.
        test: Test dataframe (optional).
        target_col: Name of the target column.
    """
    if test is None:
        _flag("INFO", "No test set provided — skipping shift analysis.")
        return

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    common_cols = [
        c
        for c in train.columns
        if c != target_col
        and c in test.columns
        and train[c].dtype in ("float64", "float32", "int64", "int32")
    ]
    if not common_cols:
        _flag("INFO", "No common numeric columns for shift analysis.")
        return

    # ── KS test per feature ───────────────────────────────────────────────────
    # Run before AV: identifies WHICH features are shifted and by how much.
    # Informs whether to drop features from the model or monitor the CV/LB gap.
    print("\nKS test per feature (train vs test distribution):")
    ks_results: list[tuple[str, float, float, float, float]] = []
    for col in common_cols:
        tr_vals = train[col].dropna().values
        te_vals = test[col].dropna().values
        if len(tr_vals) == 0 or len(te_vals) == 0:
            continue
        ks_stat, ks_p = stats.ks_2samp(tr_vals, te_vals)
        tr_mean = float(np.mean(tr_vals))
        te_mean = float(np.mean(te_vals))
        ks_results.append((col, ks_stat, ks_p, tr_mean, te_mean))

    ks_results.sort(key=lambda x: -x[1])  # sort by KS statistic descending
    sig_shifted = [r for r in ks_results if r[2] < 0.01]

    print(
        f"  {'Feature':<35} {'KS stat':>8} {'p-value':>10} {'Train mean':>10} {'Test mean':>10}"
    )
    print("  " + "-" * 75)
    for col, ks, p, tr_m, te_m in ks_results[:15]:
        sig = " ***" if p < 0.01 else (" *" if p < 0.05 else "")
        print(f"  {col:<35} {ks:>8.4f} {p:>10.4f} {tr_m:>10.3f} {te_m:>10.3f}{sig}")

    if len(sig_shifted) > 0:
        sig_names = [r[0] for r in sig_shifted[:5]]
        _flag(
            "WARN",
            f"{len(sig_shifted)} features significantly shifted (p<0.01): {sig_names}",
        )
    else:
        _flag(
            "INFO",
            "No features significantly shifted between train and test (KS p<0.01).",
        )

    # ── Adversarial validation ────────────────────────────────────────────────
    av_train = train[common_cols].fillna(-999)
    av_test = test[common_cols].fillna(-999)
    X_av = pd.concat([av_train, av_test], axis=0)
    y_av = np.array([0] * len(av_train) + [1] * len(av_test))

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    av_scores = cross_val_score(clf, X_av, y_av, cv=3, scoring="roc_auc")
    av_auc = float(np.mean(av_scores))

    # Feature importance from AV
    clf.fit(X_av, y_av)
    importances = pd.Series(
        clf.feature_importances_,
        index=common_cols,
    ).sort_values(ascending=False)
    top_shifted = importances.head(5)

    print(f"\nAdversarial Validation AUC: {av_auc:.4f}")
    print(f"Top shifted features (AV importance): {list(top_shifted.index)}")

    if av_auc > 0.75:
        _flag(
            "CRITICAL",
            f"Severe train/test shift (AV AUC={av_auc:.3f}). "
            f"Top shifted: {list(top_shifted.head(3).index)}. "
            "CV may not predict LB. Consider dropping shifted non-predictive features.",
        )
    elif av_auc > 0.65:
        _flag(
            "WARN",
            f"Moderate train/test shift (AV AUC={av_auc:.3f}). Monitor CV/LB gap.",
        )
    elif av_auc > 0.55:
        _flag("INFO", f"Mild train/test shift (AV AUC={av_auc:.3f}). Likely fine.")
    else:
        _flag("INFO", f"No significant shift detected (AV AUC={av_auc:.3f}).")

    # ── PSI per feature ───────────────────────────────────────────────────────
    print("\nPSI per feature (>0.2 = significant shift):")
    psi_severe: list[str] = []
    for col in common_cols:
        try:
            train_vals = train[col].dropna()
            test_vals = test[col].dropna()
            if len(train_vals) == 0 or len(test_vals) == 0:
                continue
            bins = np.histogram_bin_edges(train_vals, bins=10)
            train_hist = np.histogram(train_vals, bins=bins)[0] / len(train_vals) + 1e-6
            test_hist = np.histogram(test_vals, bins=bins)[0] / len(test_vals) + 1e-6
            psi = float(
                np.sum((test_hist - train_hist) * np.log(test_hist / train_hist))
            )
            if psi > 0.2:
                psi_severe.append(f"{col} (PSI={psi:.3f})")
        except Exception:
            pass
    if psi_severe:
        _flag("WARN", f"PSI > 0.2 (significant shift): {psi_severe}")
    else:
        _flag("INFO", "No features with PSI > 0.2.")
    print()


# ── OOD analysis ──────────────────────────────────────────────────────────────


def check_ood_analysis(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    target_col: str,
) -> None:
    """Out-of-distribution analysis: distance of test points to nearest training points.

    Measures how many test points fall outside the training distribution in feature
    space. Critical for understanding where the model is extrapolating and which
    predictions are most uncertain.

    Outputs:
    - Fraction of test points that are OOD (dist > train p95)
    - Distance quartile breakdown with mean predicted confidence
    - Plot: train vs test distance distributions

    Why this matters:
    - OOD test points → model extrapolates → predictions default to majority class
    - Isolated uncertain points drive threshold sensitivity (the decision zone)
    - If most OOD points cluster in a specific feature direction → indicates systematic
      gap in training data (e.g. underrepresented geographic region or climate regime)

    Args:
        train: Training dataframe (features only, no target).
        test: Test dataframe (optional).
        target_col: Name of the target column (to exclude from distance computation).
    """
    if test is None:
        _flag("INFO", "No test set provided — skipping OOD analysis.")
        return

    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    common_cols = [
        c
        for c in train.columns
        if c != target_col
        and c in test.columns
        and train[c].dtype in ("float64", "float32", "int64", "int32")
    ]
    if len(common_cols) < 2:
        _flag("INFO", "Not enough common numeric columns for OOD analysis.")
        return

    print("\n" + "=" * 60)
    print("OOD ANALYSIS — Test Distance to Training Distribution")
    print("=" * 60)

    X_train = train[common_cols].fillna(train[common_cols].median()).values
    X_test = test[common_cols].fillna(train[common_cols].median()).values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # k=5 nearest neighbors
    K = 5
    nn = NearestNeighbors(n_neighbors=K + 1, algorithm="ball_tree", n_jobs=-1)
    nn.fit(X_train_s)

    # Train: LOO distances (exclude self = index 0)
    train_dists_raw, _ = nn.kneighbors(X_train_s)
    train_mean_dist = train_dists_raw[:, 1:].mean(axis=1)  # skip self

    # Test: distances to all training points
    test_dists_raw, _ = nn.kneighbors(X_test_s, n_neighbors=K)
    test_mean_dist = test_dists_raw.mean(axis=1)

    # OOD threshold: train 95th percentile
    train_p95 = float(np.percentile(train_mean_dist, 95))
    ood_mask = test_mean_dist > train_p95
    ood_frac = float(ood_mask.mean())
    n_ood = int(ood_mask.sum())

    print(
        f"Train mean dist to k-NN (internal): {train_mean_dist.mean():.3f} ± {train_mean_dist.std():.3f}"
    )
    print(
        f"Test  mean dist to k-NN (to train): {test_mean_dist.mean():.3f} ± {test_mean_dist.std():.3f}"
    )
    print(f"Train p95 dist (OOD threshold):      {train_p95:.3f}")
    print(
        f"Test OOD fraction (dist > train p95): {ood_frac * 100:.1f}% ({n_ood}/{len(test_mean_dist)} points)"
    )

    if ood_frac > 0.15:
        _flag(
            "CRITICAL",
            f"{ood_frac * 100:.1f}% of test points are OOD — model extrapolates heavily. Predictions for these points are unreliable.",
        )
    elif ood_frac > 0.07:
        _flag(
            "WARN",
            f"{ood_frac * 100:.1f}% of test points are OOD. Monitor threshold sensitivity for these points.",
        )
    else:
        _flag(
            "INFO",
            f"{ood_frac * 100:.1f}% of test points are OOD — test is well-covered by training distribution.",
        )

    # Isolation quartile breakdown
    print("\nTest isolation quartile breakdown:")
    print(f"  {'Quartile':<20} {'N':>6} {'Mean dist':>12} {'Pct of test':>12}")
    print("  " + "-" * 54)
    for q_lo, q_hi in [(0, 25), (25, 50), (50, 75), (75, 100)]:
        lo = float(np.percentile(test_mean_dist, q_lo))
        hi = float(np.percentile(test_mean_dist, q_hi))
        mask = (test_mean_dist >= lo) & (
            test_mean_dist < hi if q_hi < 100 else test_mean_dist <= hi
        )
        n = int(mask.sum())
        mean_d = float(test_mean_dist[mask].mean()) if n > 0 else 0.0
        label = "most in-dist" if q_lo == 0 else ("most OOD" if q_hi == 100 else "")
        print(
            f"  Q{q_lo:02d}–Q{q_hi:02d} {label:<12} {n:>6} {mean_d:>12.3f} {n / len(test_mean_dist) * 100:>11.1f}%"
        )

    _flag(
        "INFO",
        f"Decision zone (OOD Q75-Q100): {n_ood} points where predictions may differ "
        "by threshold choice. Examine if these cluster in a specific feature direction.",
    )

    # Plot: distance distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        train_mean_dist,
        bins=50,
        alpha=0.6,
        color="steelblue",
        label="Train (LOO)",
        density=True,
    )
    axes[0].hist(
        test_mean_dist, bins=50, alpha=0.6, color="coral", label="Test", density=True
    )
    axes[0].axvline(
        train_p95, color="red", linestyle="--", label=f"Train p95={train_p95:.2f}"
    )
    axes[0].set_xlabel("Mean distance to 5-NN (standardized)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distance to Training Distribution")
    axes[0].legend()

    # Which features contribute most to OOD distance?
    # Compare mean feature values: OOD test points vs in-distribution test points
    ood_test = test[common_cols].fillna(train[common_cols].median()).values[ood_mask]

    train_arr = train[common_cols].fillna(train[common_cols].median()).values

    if len(ood_test) > 5:
        # Standardized mean difference: OOD vs train
        tr_std = np.std(train_arr, axis=0) + 1e-8
        tr_mean = np.mean(train_arr, axis=0)
        ood_diff = (np.mean(ood_test, axis=0) - tr_mean) / tr_std
        ood_series = (
            pd.Series(ood_diff, index=common_cols).abs().sort_values(ascending=False)
        )
        top_ood_features = ood_series.head(10)

        top_ood_features.sort_values().plot(kind="barh", ax=axes[1], color="coral")
        axes[1].set_title(
            "Top Features Driving OOD Distance\n(standardized mean diff: OOD test vs train)"
        )
        axes[1].set_xlabel("|Standardized mean difference|")
    else:
        axes[1].text(
            0.5,
            0.5,
            "Too few OOD points\nfor feature analysis",
            ha="center",
            va="center",
        )
        axes[1].set_title("OOD Feature Analysis")

    plt.tight_layout()
    _savefig("ood_analysis.png")

    # Summary flag
    if len(ood_test) > 5:
        top3 = list(ood_series.head(3).index)
        _flag(
            "INFO",
            f"OOD points differ most in: {top3}. Check if these features are also top model predictors.",
        )
    print()


# ── categorical vs target plots ──────────────────────────────────────────────


def check_categorical_vs_target_plots(
    train: pd.DataFrame,
    target_col: str,
    task: str = "auto",
) -> None:
    """Box plots: target distribution per category for top categoricals.

    Args:
        train: Training dataframe.
        target_col: Name of the target column.
        task: 'classification', 'regression', or 'auto'.
    """
    y = train[target_col]
    if task == "auto":
        task = "classification" if y.nunique() <= 20 else "regression"

    cat_cols = [
        c for c in train.select_dtypes(include="object").columns if c != target_col
    ]
    if not cat_cols or task != "regression":
        return  # box plots most useful for regression target vs categorical

    # Pick top-6 categoricals by cardinality
    cat_cols = sorted(cat_cols, key=lambda c: train[c].nunique())[:6]

    ncols = 3
    nrows = int(np.ceil(len(cat_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cat_cols):
        top_cats = train[col].value_counts().head(10).index
        subset = train[train[col].isin(top_cats)]
        subset.boxplot(column=target_col, by=col, ax=axes[i])
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=45, labelsize=7)

    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Target Distribution by Category", fontsize=11)
    plt.tight_layout()
    _savefig("categorical_vs_target.png")


# ── entry point ───────────────────────────────────────────────────────────────


def run_tabular_checks(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    target_col: str,
    task: str = "auto",
) -> None:
    """
    Run all tabular EDA checks.

    Args:
        train: Training dataframe.
        test: Test dataframe (optional).
        target_col: Name of the target column.
        task: 'classification', 'regression', or 'auto'.
    """
    print("=" * 60)
    print("TABULAR CHECKS")
    print("=" * 60)
    check_numerical_distributions(train, target_col)
    check_outliers(train, target_col)
    check_correlations(train, target_col)
    check_mutual_information(train, target_col, task)
    check_categorical_features(train, test, target_col)
    check_categorical_vs_target_plots(train, target_col, task)
    check_train_test_shift(train, test, target_col)
    check_ood_analysis(train, test, target_col)
