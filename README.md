# Richter's Predictor — Modeling Earthquake Damage

DrivenData competition #57. Predict building damage grade (1/2/3) from structural
and geographic features collected after the 2015 Gorkha earthquake in Nepal.

**Metric**: F1-micro &nbsp;|&nbsp; **Best score**: 0.7518 (rank ~225, top 9% of 2,675 teams)

---

## Problem

Multi-class ordinal classification over ~260k buildings. Each building is described
by 39 features: three nested geographic IDs (`geo_level_1/2/3_id`), structural
attributes (floors, age, area/height percentages), material flags (foundation, roof,
floor type), 11 binary superstructure flags (adobe, brick, concrete, timber, etc.),
and secondary-use flags.

Target: `damage_grade` ∈ {1 = low damage, 2 = medium damage, 3 = near-total destruction}.

---

## Results

| Submission | OOF F1-micro | LB F1-micro | Notes |
|------------|-------------|-------------|-------|
| First submit — LGB+CAT+XGB weighted avg | 0.7511 | 0.7476 | rank 595 |
| + 5-fold CV | 0.7496 | 0.7484 | rank 513 |
| + ExtraTrees (4th model) + stacking | 0.7508 | 0.7503 | rank 343 |
| + Random Forest (5th model) + stacking | 0.7516 | 0.7506 | rank ~310 |
| + Neural geo embeddings | 0.7520 | 0.7513 | rank 255 |
| + AutoFE (15 features, LGB only) | **0.7524** | **0.7518** | rank ~225 |

Competition leader: 0.7558. Documented GBM ceiling: ~0.757.

---

## Architecture

### Models

Five-model ensemble trained with 5-fold Stratified K-Fold CV:

| Model | OOF F1-micro | Notes |
|-------|-------------|-------|
| LightGBM | 0.7494 | + geo embeddings + AutoFE (15 features) |
| CatBoost | 0.7489 | no Optuna — default params are optimal (3 separate HPO runs confirmed) |
| XGBoost | 0.7499 | + geo embeddings, GPU (`device: cuda`) |
| ExtraTrees | 0.7486 | no embeddings — random subsampling dilutes embedding signal |
| RandomForest | 0.7468 | no embeddings — same reason |

Final prediction: weighted average of predicted class probabilities, weights
optimized via Nelder-Mead on OOF predictions.

Neural networks (MLP, TabNet) were tested and reach 0.56–0.57 F1-micro on this
dataset, a ~0.18 gap versus a tuned GBM ensemble.

### Feature Engineering

| Feature group | Method | Gain |
|---------------|--------|------|
| `geo_level_1/2/3_id` | GLMM empirical Bayes encoding → 9 floats | #1 lever |
| Geo entity embeddings | MLP autoencoder → 56D per building | +0.003 OOF |
| Low-card categoricals | GLMM target encoding → 24 floats (LGB/XGB) | marginal |
| AutoFE candidates (32 pool) | Optuna selects 15 best → LGB only | +0.0004 OOF |
| `superstructure_count`, `material_strength`, `mud_stone_x_age`, `rc_eng_x_floors` | manual composites | stable baseline |
| Rare geo category handling | freq < 3 in train fold → −1 sentinel | +0.0018 XGB |

CatBoost receives all categoricals (including `geo_level_*`) raw via `cat_features`
and handles encoding internally via Ordered Target Statistics — its primary advantage
over LGB/XGB on this dataset.

### CV and Tuning

- 5-fold Stratified K-Fold; Optuna (TPE, 30 trials) on fold 0, then all 5 folds
  trained with best hyperparameters
- OOF predictions cover all 260k training rows — used for ensemble weight optimization
- Target encoding and geo embeddings fitted on the training fold only (no leakage)

---

## Key Lessons

A few non-obvious findings from 10 tracked experiments:

- **Features that improve individual models can hurt the ensemble.** Frequency
  encoding and geo×structure interactions each improved LGB/XGB OOF by +0.001–0.002
  but dropped ensemble OOF by −0.007. The models became more similar to CatBoost,
  reducing diversity and flattening the weight optimizer.

- **CatBoost GPU is a trap on this dataset.** `task_type: "GPU"` drops from 0.7489
  to 0.7404 (−0.0085). GPU mode switches from Ordered target statistics to
  Borders/BinarizedTarget encoding, destroying performance on high-cardinality
  categoricals.

- **CAT Optuna is not a lever.** Three independent HPO runs, all converging below
  the default parameter F1. CatBoost's defaults are well-calibrated for this class
  of problem.

- **Stacking requires decorrelated models.** With 3 correlated GBMs, LogisticRegression
  stacking scored 0.7494 vs weighted avg 0.7497. Adding ExtraTrees and RandomForest
  made stacking viable — 5-model stacking reached 0.7516.

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full experiment record and
[`TECHNICAL_CHOICES.md`](TECHNICAL_CHOICES.md) for architectural decision rationale.

---

## Repository Structure

```
Richter-Predictor/
├── configs/
│   └── config.yaml           all hyperparameters (no hardcoded values in src/)
├── src/
│   ├── features.py           FE pipeline (GLMM, embeddings, AutoFE, cat TE)
│   ├── train.py              5-fold CV + Optuna HPO, all 5 models
│   ├── ensemble.py           weighted avg + stacking + submission generation
│   ├── predict.py            standalone inference with saved model artifacts
│   ├── geo_embeddings.py     MLP-based geo entity embedding training
│   ├── autofe_search.py      Optuna-based automatic feature selection
│   └── train_fh.py           Frank & Hall ordinal decomposition (experiment)
├── utils/
│   └── seed.py               global seed utility
├── models/
│   ├── lgb_best_params.json  best LGB hyperparameters from Optuna
│   ├── xgb_best_params.json  best XGB hyperparameters from Optuna
│   └── autofe_best_features.json  AutoFE-selected feature subset
├── notebooks/
│   └── 01_eda.py             exploratory data analysis
├── data/
│   ├── raw/                  competition CSVs (not included — see Setup)
│   └── processed/            encoded features, not included
├── submissions/              timestamped submission CSVs (not included)
├── requirements.txt
├── requirements-dev.txt
├── TECHNICAL_CHOICES.md      architectural decision record
└── EXPERIMENTS.md            full experiment log with results
```

---

## Setup

```bash
git clone https://github.com/marcoBortolotti2921/Richter-Predictor.git
cd Richter-Predictor
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install --upgrade pip && pip install -r requirements.txt
```

Download the competition data from [DrivenData #57](https://www.drivendata.org/competitions/57/nepal-earthquake/data/)
and place the three CSV files in `data/raw/`:

```
data/raw/train_values.csv
data/raw/train_labels.csv
data/raw/test_values.csv
```

---

## Usage

```bash
# Train geo entity embeddings (required before LGB/XGB training)
python src/geo_embeddings.py --config configs/config.yaml

# Train individual models (add --no-optuna for a quick smoke test)
python src/train.py --config configs/config.yaml --model lgb
python src/train.py --config configs/config.yaml --model cat
python src/train.py --config configs/config.yaml --model xgb
python src/train.py --config configs/config.yaml --model et
python src/train.py --config configs/config.yaml --model rf

# Ensemble + submission CSV
python src/ensemble.py --config configs/config.yaml
```

Or via Make:

```bash
make smoke        # LGB only, no Optuna — first score in ~2 min
make train-all    # all 5 models with Optuna
make ensemble     # weighted avg + submission CSV
make all          # full pipeline
```

Output: `submissions/YYYYMMDD_HHMMSS_f1micro_0.XXXX.csv`

---

## Hardware

Developed on Windows 11, NVIDIA RTX 2050 (4 GB VRAM). XGBoost uses
`device: cuda`. CatBoost and tree ensembles run on CPU (CatBoost GPU mode
degrades performance — see above).
