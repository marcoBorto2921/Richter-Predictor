"""FE Validation — Phase 3.5.

Quick 80/20 LGB (500 rounds) via build_features() to validate new engineered features.
Compares F1-micro against known baseline (raw features, same split: 0.7430).

Run: .venv/Scripts/python fe_validation.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from src.features import build_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SEED = 42
KNOWN_BASELINE_F1 = (
    0.7430  # quick LGB 500r 80/20 with raw features (error analysis run)
)

CONFIG_PATH = "configs/config.yaml"
TRAIN_VALUES = "data/raw/train_values.csv"
TRAIN_LABELS = "data/raw/train_labels.csv"
TEST_VALUES = "data/raw/test_values.csv"


def main() -> None:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load data
    train_values = pd.read_csv(TRAIN_VALUES, encoding="utf-8")
    train_labels = pd.read_csv(TRAIN_LABELS, encoding="utf-8")
    df_full = train_values.merge(train_labels, on="building_id")
    df_test = pd.read_csv(TEST_VALUES, encoding="utf-8")

    target_col = cfg["data"]["target_col"]

    # 80/20 stratified split — same as error analysis baseline
    df_tr, df_va = train_test_split(
        df_full, test_size=0.2, stratify=df_full[target_col], random_state=SEED
    )
    df_tr = df_tr.reset_index(drop=True)
    df_va = df_va.reset_index(drop=True)

    y_val = df_va[target_col].values

    log.info("Train: %d  Val: %d", len(df_tr), len(df_va))

    # Build features (lgb_xgb mode, no embeddings for speed)
    cfg_no_emb = cfg.copy()
    cfg_no_emb["features"] = dict(cfg["features"])
    cfg_no_emb["features"]["geo_embedding"] = {"enabled": False}

    X_train, X_val, X_test, _ = build_features(
        df_tr,
        df_va,
        df_test,
        cfg_no_emb,
        mode="lgb_xgb",
        use_embeddings=False,
        use_cat_te=True,
        autofe_features=None,
    )

    log.info("Feature matrix: train %s  val %s", X_train.shape, X_val.shape)

    # Train quick LGB — 500 rounds, no early stopping for comparability
    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        verbose=-1,
        random_state=SEED,
        n_jobs=-1,
    )
    y_train_shifted = df_tr[target_col].values - 1
    model.fit(X_train, y_train_shifted)

    preds = model.predict(X_val) + 1
    f1 = f1_score(y_val, preds, average="micro")
    train_f1 = f1_score(
        y_train_shifted + 1, model.predict(X_train) + 1, average="micro"
    )

    print("\n" + "=" * 60)
    print("CHECK 1 -- PREDICTIVE SIGNAL")
    print("=" * 60)
    print(f"Val F1-micro:         {f1:.4f}")
    print(f"Known baseline F1:    {KNOWN_BASELINE_F1:.4f}  (raw features, same split)")
    delta = f1 - KNOWN_BASELINE_F1
    print(f"Delta vs baseline:    {delta:+.4f}")
    result1 = "PASS" if f1 > KNOWN_BASELINE_F1 else "WARN (no improvement vs baseline)"
    print(f"Result:               {result1}")

    print("\n" + "=" * 60)
    print("CHECK 2 -- LEAKAGE DETECTION")
    print("=" * 60)
    print(f"Train F1 (in-sample): {train_f1:.4f}")
    print(f"Val F1 (OOF):         {f1:.4f}")
    ratio = train_f1 / f1 if f1 > 0 else 999.0
    print(f"Ratio train/val:      {ratio:.3f}")
    if ratio > 1.20:
        result2 = "CRITICAL -- likely leakage"
    elif ratio > 1.10:
        result2 = "WARN -- monitor"
    else:
        result2 = "PASS"
    print(f"Result:               {result2}")

    print("\n" + "=" * 60)
    print("CHECK 3 -- ADVERSARIAL VALIDATION (train vs test)")
    print("=" * 60)
    n_tr = min(10000, len(X_train))
    n_te = min(10000, len(X_test))
    adv_train = pd.concat(
        [
            X_train.sample(n_tr, random_state=SEED).assign(_is_test=0),
            X_test.sample(n_te, random_state=SEED).assign(_is_test=1),
        ]
    ).reset_index(drop=True)
    adv_y = adv_train.pop("_is_test")
    adv_model = LGBMClassifier(n_estimators=100, verbose=-1, random_state=SEED)
    from sklearn.model_selection import cross_val_score

    adv_scores = cross_val_score(adv_model, adv_train, adv_y, cv=3, scoring="roc_auc")
    adv_auc = float(np.mean(adv_scores))
    print(f"Adversarial AUC:      {adv_auc:.4f}")
    if adv_auc > 0.85:
        result3 = "CRITICAL -- train/test shift"
    elif adv_auc > 0.70:
        result3 = "WARN -- some shift"
    else:
        result3 = "PASS"
    print(f"Result:               {result3}")

    print("\n" + "=" * 60)
    print("CHECK 4 -- FEATURE IMPORTANCES")
    print("=" * 60)
    importances = model.feature_importances_
    feat_names = X_train.columns.tolist()
    dead = [f for f, imp in zip(feat_names, importances) if imp == 0]
    top5 = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:5]
    print("Top 5 features:")
    for name, imp in top5:
        print(f"  {name:<50} {imp:>6}")
    print(f"\nTotal features: {len(feat_names)}")
    print(f"Dead features (importance=0): {len(dead)}")
    if dead:
        print("  " + ", ".join(dead[:20]))

    # New feature importances specifically
    print("\nNew feature importances:")
    new_feats = [
        "age_is_zero",
        "brittle_material",
        "ductile_building",
        "slope_x_foundation_r",
        "plan_irregular",
        "geo1_high_seismic",
        "brittle_x_floors",
        "high_seismic_x_brittle",
        "material_vs_geo3",
    ]
    for feat in new_feats:
        if feat in feat_names:
            idx = feat_names.index(feat)
            print(f"  {feat:<45} {importances[idx]:>6}")
        else:
            print(f"  {feat:<45} MISSING")

    pct_dead = len(dead) / len(feat_names)
    if pct_dead > 0.50:
        result4 = "CRITICAL -- >50% dead features"
    elif pct_dead > 0.30:
        result4 = "WARN -- >30% dead features"
    else:
        result4 = "INFO"
    print(f"\nResult:               {result4}")

    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)
    criticals = [r for r in [result1, result2, result3, result4] if "CRITICAL" in r]
    if criticals:
        print("STOP -- CRITICAL issues found")
    else:
        print("GO -- proceed to Phase 4")


if __name__ == "__main__":
    main()
