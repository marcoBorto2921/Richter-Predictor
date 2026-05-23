"""
Error analysis — Richter's Predictor.
Trains quick LGB, identifies systematic misclassification patterns.
Run: .venv/Scripts/python notebooks/04_error_analysis.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import lightgbm as lgb

SEED = 42
TRAIN_VALUES = "data/raw/train_values.csv"
TRAIN_LABELS = "data/raw/train_labels.csv"


def load() -> pd.DataFrame:
    vals = pd.read_csv(TRAIN_VALUES, encoding="utf-8")
    labels = pd.read_csv(TRAIN_LABELS, encoding="utf-8")
    return vals.merge(labels, on="building_id")


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def train_quick_lgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[lgb.LGBMClassifier, pd.DataFrame]:
    """Train 500-round LGB, return model + val dataframe with predictions."""
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
    feature_cols = [
        c for c in train_df.columns if c not in ("building_id", "damage_grade")
    ]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["damage_grade"] - 1
    X_val = val_df[feature_cols].copy()

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    val_out = val_df.copy()
    val_out["pred"] = model.predict(X_val) + 1
    proba = model.predict_proba(X_val)
    val_out["prob_1"] = proba[:, 0]
    val_out["prob_2"] = proba[:, 1]
    val_out["prob_3"] = proba[:, 2]
    val_out["correct"] = val_out["pred"] == val_out["damage_grade"]
    val_out["error_type"] = val_out.apply(
        lambda r: f"{int(r.damage_grade)}->{int(r.pred)}" if not r.correct else "ok",
        axis=1,
    )

    return model, val_out


def confusion_summary(val: pd.DataFrame) -> None:
    section("CONFUSION MATRIX")
    y_true = val["damage_grade"].values
    y_pred = val["pred"].values
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    f1 = f1_score(y_true, y_pred, average="micro")
    print(f"Overall F1-micro: {f1:.4f}")
    print(f"\n{'':>12} Pred-1  Pred-2  Pred-3")
    for i, grade in enumerate([1, 2, 3]):
        row = cm[i]
        total = row.sum()
        pcts = [f"{v / total * 100:5.1f}%" for v in row]
        print(f"  True-{grade} ({total:6,}) " + "  ".join(pcts))

    # Per-class precision, recall
    print("\nPer-class F1:")
    per_class = f1_score(y_true, y_pred, average=None)
    for g, s in zip([1, 2, 3], per_class):
        print(f"  Grade {g}: F1={s:.4f}")


def error_by_geo(val: pd.DataFrame) -> None:
    section("ERRORS BY GEO_LEVEL_1")
    geo_stats = (
        val.groupby("geo_level_1_id")
        .agg(
            n=("damage_grade", "count"),
            f1=("correct", "mean"),
            grade3_true=("damage_grade", lambda x: (x == 3).mean()),
        )
        .sort_values("f1")
    )
    print("Bottom 10 regions (worst accuracy):")
    print(geo_stats.head(10).to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nTop 5 regions (best accuracy):")
    print(geo_stats.tail(5).to_string(float_format=lambda x: f"{x:.3f}"))

    # Unseen geo3
    section("ERRORS: UNSEEN geo_level_3_id IN VAL")

    print(
        "(see group leakage section in EDA_FINDINGS.md — unseen geo3 F1-micro: 0.6420)"
    )


def error_by_material(val: pd.DataFrame) -> None:
    section("ERRORS BY SUPERSTRUCTURE MATERIAL")
    mat_cols = [c for c in val.columns if c.startswith("has_superstructure_")]
    for col in mat_cols:
        sub = val[val[col] == 1]
        if len(sub) < 100:
            continue
        acc = sub["correct"].mean()
        grade3_pct = (sub["damage_grade"] == 3).mean()
        print(f"  {col:<45} n={len(sub):6,}  acc={acc:.3f}  true_G3%={grade3_pct:.2f}")


def error_by_error_type(val: pd.DataFrame) -> None:
    section("ERROR TYPE BREAKDOWN")
    et = val[~val["correct"]]["error_type"].value_counts()
    total_errors = len(val[~val["correct"]])
    print(f"Total errors: {total_errors:,} ({total_errors / len(val) * 100:.1f}%)")
    for etype, cnt in et.items():
        print(f"  {etype}: {cnt:,} ({cnt / total_errors * 100:.1f}%)")


def grade1_error_profile(val: pd.DataFrame) -> None:
    """Analyze Grade 1 buildings predicted as Grade 2/3 — where does the model fail?"""
    section("GRADE 1 MISCLASSIFICATION PROFILE")
    g1_true = val[val["damage_grade"] == 1]
    g1_wrong = g1_true[~g1_true["correct"]]
    g1_right = g1_true[g1_true["correct"]]

    print(
        f"True Grade 1: {len(g1_true):,}  Correct: {len(g1_right):,} ({len(g1_right) / len(g1_true) * 100:.1f}%)  Missed: {len(g1_wrong):,}"
    )
    print(f"  -> predicted as Grade 2: {(g1_wrong['pred'] == 2).sum():,}")
    print(f"  -> predicted as Grade 3: {(g1_wrong['pred'] == 3).sum():,}")

    # Geo1 breakdown
    print("\nGrade 1 recall by geo_level_1 (worst regions):")
    g1_geo = (
        g1_true.groupby("geo_level_1_id")["correct"]
        .agg(["mean", "count"])
        .sort_values("mean")
    )
    g1_geo.columns = ["recall", "n_grade1"]
    print(
        g1_geo[g1_geo["n_grade1"] >= 20]
        .head(10)
        .to_string(float_format=lambda x: f"{x:.3f}")
    )

    # Material profile of misclassified Grade 1
    print("\nMaterial rates: missed Grade 1 vs correct Grade 1:")
    mat_cols = [c for c in val.columns if c.startswith("has_superstructure_")]
    for col in mat_cols:
        r_wrong = g1_wrong[col].mean()
        r_right = g1_right[col].mean()
        if max(r_wrong, r_right) > 0.05:
            print(
                f"  {col:<45} wrong={r_wrong:.2f}  right={r_right:.2f}  diff={r_wrong - r_right:+.2f}"
            )


def grade23_confusion_profile(val: pd.DataFrame) -> None:
    """Analyze Grade 2/3 confusion — the dominant error type."""
    section("GRADE 2/3 CONFUSION PROFILE")
    errors_23 = val[val["error_type"] == "2->3"]
    errors_32 = val[val["error_type"] == "3->2"]

    print(f"Grade 2 predicted as 3: {len(errors_23):,}")
    print(f"Grade 3 predicted as 2: {len(errors_32):,}")

    # Foundation type distribution in errors
    print("\nFoundation type in 2->3 errors (vs overall Grade 2):")
    g2_all = val[val["damage_grade"] == 2]
    for cat in ["foundation_type", "roof_type", "land_surface_condition"]:
        print(f"\n  {cat}:")
        err_dist = errors_23[cat].value_counts(normalize=True)
        base_dist = g2_all[cat].value_counts(normalize=True)
        for level in err_dist.index[:5]:
            e = err_dist.get(level, 0)
            b = base_dist.get(level, 0)
            if abs(e - b) > 0.03:
                print(f"    {level}: err={e:.2f}  base={b:.2f}  diff={e - b:+.2f}")

    # Age distribution
    print(
        f"\n  age (2->3 errors): mean={errors_23['age'].mean():.1f}, median={errors_23['age'].median():.1f}"
    )
    print(
        f"  age (all Grade 2):  mean={g2_all['age'].mean():.1f}, median={g2_all['age'].median():.1f}"
    )
    print(
        f"  age (3->2 errors): mean={errors_32['age'].mean():.1f}, median={errors_32['age'].median():.1f}"
    )

    # Geo1 — which regions drive 2/3 confusion
    print("\nGeo1 regions with highest Grade 2->3 error rate:")
    g2_geo = g2_all.groupby("geo_level_1_id").agg(
        n=("damage_grade", "count"),
        wrong_as_3=("error_type", lambda x: (x == "2->3").sum()),
    )
    g2_geo["rate"] = g2_geo["wrong_as_3"] / g2_geo["n"]
    print(
        g2_geo[g2_geo["n"] >= 100]
        .sort_values("rate", ascending=False)
        .head(8)
        .to_string(float_format=lambda x: f"{x:.3f}")
    )


def high_confidence_errors(val: pd.DataFrame) -> None:
    """Errors where model is confident but wrong — structural failures."""
    section("HIGH-CONFIDENCE ERRORS (max_prob > 0.8, wrong)")
    val = val.copy()
    val["max_prob"] = val[["prob_1", "prob_2", "prob_3"]].max(axis=1)
    hce = val[(val["max_prob"] > 0.8) & (~val["correct"])]
    print(f"High-confidence errors: {len(hce):,} ({len(hce) / len(val) * 100:.2f}%)")
    if len(hce) > 0:
        print("\nError types in high-confidence errors:")
        print(hce["error_type"].value_counts().to_string())
        print("\nGeo1 distribution of high-confidence errors:")
        print(hce["geo_level_1_id"].value_counts().head(10).to_string())
        print(
            f"\nMean age: {hce['age'].mean():.1f} vs val mean {val['age'].mean():.1f}"
        )
        print("Foundation type dist:")
        print(hce["foundation_type"].value_counts(normalize=True).head(5).to_string())


def main() -> None:
    print("Loading data...")
    df = load()

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["damage_grade"], random_state=SEED
    )
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}")

    print("Training LGB (500r)...")
    model, val = train_quick_lgb(train_df, val_df)

    confusion_summary(val)
    error_by_error_type(val)
    error_by_geo(val)
    error_by_material(val)
    grade1_error_profile(val)
    grade23_confusion_profile(val)
    high_confidence_errors(val)

    print("\n\nERROR ANALYSIS DONE")


if __name__ == "__main__":
    main()
