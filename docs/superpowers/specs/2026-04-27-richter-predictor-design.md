# Richter's Predictor — Design Spec

**Date:** 2026-04-27
**Competition:** DrivenData #57 — Modeling Earthquake Damage
**Goal:** Top leaderboard position (stretch: F1-micro ≥ 0.757)
**Compute:** CPU only (LGB/XGB/CAT train in < 5 min each)

---

## 1. Project Structure

```
Richter-Predictor/
├── .claude/
│   ├── CLAUDE.md
│   ├── PROJECT_PLAN.md
│   └── PROJECT_LOG.md
├── configs/
│   └── config.yaml          # all hyperparameters, paths, seeds
├── data/
│   ├── raw/                 # train_values.csv, test_values.csv, train_labels.csv
│   └── processed/           # encoded features, splits
├── src/
│   ├── features.py          # feature engineering pipeline
│   ├── train.py             # training script (lgb / xgb / cat)
│   ├── predict.py           # inference + submission generation
│   └── ensemble.py          # blend predictions from multiple models
├── submissions/             # timestamped CSVs ready to upload
├── models/                  # saved model artifacts
├── notebooks/               # EDA only, no training logic
└── requirements.txt
```

---

## 2. Data Split

- Stratified 80/20 holdout on `damage_grade`
- No k-fold unless score plateaus and stacking is needed
- All target-encoding computed on train fold only (no leakage)

---

## 3. Feature Engineering

### Categorical features
- Low-cardinality categoricals (foundation type, roof type, floor type, land surface, position, plan configuration, legal ownership): ordinal or one-hot for LGB/XGB; passed raw to CatBoost
- `geo_level_1/2/3_id`: kept as categoricals + target-encoded damage mean per geo group

### Interaction features
- `geo2_damage_mean × age` — old buildings in high-risk zones
- `age × area_percentage` — building mass over time
- `height_percentage / area_percentage` — building slenderness proxy
- `count_floors_pre_eq` binned (1, 2, 3+)

### Aggregate features
- `superstructure_count` — sum of 11 binary superstructure flags
- `has_secondary_use` — binary aggregate of secondary use flags

### Prohibited
- No external data (competition rules)

---

## 4. Models

Three models trained independently:

| Model | Key hyperparameters |
|-------|-------------------|
| LightGBM | `num_leaves=127`, `min_child_samples=20`, `colsample_bytree=0.8` |
| XGBoost | `max_depth=7`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method=hist` |
| CatBoost | `depth=8`, `l2_leaf_reg=3`, native categorical handling |

- All use `multiclass` objective (3 classes)
- Hyperparameters tuned with **Optuna** (100 trials each), metric: F1-micro on holdout
- Post-processing: threshold sweep on predicted probabilities to maximize F1-micro

---

## 5. Ensemble

- Weighted average of predicted probabilities from all three models
- Weights optimized on holdout via `scipy.optimize.minimize` (Nelder-Mead), constrained to sum to 1
- Expected weight distribution: ~40% LGB / ~35% CAT / ~25% XGB (adjust per actual scores)

---

## 6. Submission Pipeline

```bash
python src/train.py --config configs/config.yaml --model lgb
python src/train.py --config configs/config.yaml --model xgb
python src/train.py --config configs/config.yaml --model cat
python src/ensemble.py --config configs/config.yaml
```

Output: `submissions/YYYYMMDD_HHMMSS_f1micro_0.XXXX.csv`

---

## 7. Experiment Workflow

- Every experiment follows `ml-experiment-discipline` skill: hypothesis → run → verdict
- All hyperparameters in `configs/config.yaml`, scripts accept `--config path`
- Each run logs F1-micro + config snapshot with timestamp
- If F1-micro > current best → generate submission CSV

---

## 8. Targets

| Target | F1-micro |
|--------|----------|
| Baseline (competitive midfield) | ≥ 0.74 |
| Stretch (top tier GBM ensemble) | ≥ 0.757 |

---

## 9. Optional Escalation

If score plateaus below stretch target after ensemble tuning:
- Add one MLP/TabNet model for ensemble diversity (requires GPU)
- Enable k-fold for stacked meta-learner

These are NOT in scope for the initial implementation.
