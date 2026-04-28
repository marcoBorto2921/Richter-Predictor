# Richter's Predictor — Modeling Earthquake Damage

DrivenData competition #57. Predict building damage grade (1/2/3) from structural
and geographic features collected after the 2015 Gorkha earthquake in Nepal.

**Metric**: F1-micro (higher is better)
**Deadline**: 2027-10-05

---

## Problem

Multi-class ordinal classification over ~260k buildings. Each building is described
by 39 features: three nested geographic IDs, structural attributes (floors, age,
area/height percentages), material flags (foundation, roof, floor type), 11 binary
superstructure flags (adobe, brick, concrete, timber, etc.), and secondary-use flags.

Target: `damage_grade` ∈ {1 = low damage, 2 = medium, 3 = near-total destruction}.

---

## Approach

**Model family**: LightGBM + CatBoost + XGBoost ensemble.

GBMs consistently dominate tabular classification benchmarks (TabArena 2025). Neural
networks (MLP, TabNet) score 0.56–0.57 F1-micro on this specific dataset vs 0.745–0.755
for a tuned GBM ensemble. Random Forest plateaus at ~0.72.

**Feature engineering**:
- `geo_level_1/2/3_id` are the #1 and #2 most important features. They receive
  class-conditional smoothed target encoding for LGB/XGB (9 new features); CatBoost
  handles them natively via ordered target statistics.
- Aggregate features: `superstructure_count`, `has_secondary_use_any`,
  `height_to_area` ratio, `age_x_area`, `age_x_geo2_damage`.
- One-hot encoding of geo_level_* is explicitly prohibited: creates 10k+ sparse columns.

**Ensemble**:
- Weighted average of predicted probabilities from LGB + CatBoost + XGBoost.
- Weights optimized on holdout via `scipy.optimize.minimize` (Nelder-Mead).

**Tuning**:
- Optuna TPE sampler, 100 trials per model, maximize F1-micro on holdout.
- Post-HPO: refit on train+val combined with best hyperparameters (often improves score).

**Ordinal structure** (Phase 3):
- Frank & Hall decomposition (3-class → 2 binary classifiers) deferred until
  baseline ≥ 0.75 is stable.

---

## Leaderboard Targets

| Tier | F1-micro |
|------|---------|
| Competition leader | 0.7558 |
| Strong GBM ensemble | 0.748–0.750 |
| Decent baseline (XGBoost, minimal FE) | 0.735–0.744 |
| Random Forest | ~0.72 |
| Neural networks (MLP) | 0.56–0.57 |

---

## Setup

```bash
git clone https://github.com/marcoBorto2921/Richter-Predictor.git
cd Richter-Predictor
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install --upgrade pip && pip install -r requirements.txt
```

Place competition data files in `data/raw/`:
- `train_values.csv`
- `train_labels.csv`
- `test_values.csv`

---

## Usage

```bash
# Train individual models (--no-optuna for quick smoke test)
python src/train.py --config configs/config.yaml --model lgb
python src/train.py --config configs/config.yaml --model cat
python src/train.py --config configs/config.yaml --model xgb

# Ensemble + submission
python src/ensemble.py --config configs/config.yaml
```

Or via Make:

```bash
make smoke       # LGB only, no Optuna — first score in ~2 min
make train-lgb
make train-all
make ensemble
make all         # full pipeline
```

---

## Repository Structure

```
Richter-Predictor/
├── configs/config.yaml       — all hyperparameters
├── src/
│   ├── features.py           — FE pipeline (target encoding, interactions)
│   ├── train.py              — training + Optuna HPO per model
│   ├── ensemble.py           — weight optimization + submission generation
│   └── predict.py            — standalone inference with saved models
├── utils/seed.py             — global seed utility
├── data/raw/                 — competition CSVs (gitignored)
├── models/                   — saved model artifacts (gitignored)
├── submissions/              — timestamped submission CSVs (gitignored)
├── TECHNICAL_CHOICES.md      — architectural decision record
└── requirements.txt
```

---

## Technical Decisions

See `TECHNICAL_CHOICES.md` for the full decision record: model family selection,
geo feature encoding, CatBoost categorical handling, ensemble method, CV strategy,
HPO approach, and ordinal structure plan.
