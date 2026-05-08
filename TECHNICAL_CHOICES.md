# Technical Choices — Richter's Predictor

Architectural decision record. Covers model family, feature engineering, ensemble
method, CV strategy, and HPO approach. Experiment outcomes (what was tried and
what happened) are in [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

### Decision: Model Family

**Choice**: Gradient Boosted Decision Trees — LightGBM + CatBoost + XGBoost as
the primary trio, extended with ExtraTrees and RandomForest for ensemble diversity.

**Rationale**: GBMs consistently dominate tabular classification benchmarks. On this
specific dataset, LightGBM reaches F1-micro 0.749 with proper tuning; neural networks
(MLP, TabNet) score 0.56–0.57 — a ~0.18 gap. Multiple public solutions and the
2024–2025 tabular benchmarking literature confirm GBM dominance on structured data
with high-cardinality categoricals [Comprehensive Survey of Tabular Classification
2024–2025].

ExtraTrees and RandomForest were added specifically for ensemble diversity: their
bagging-based, non-boosted predictions are decorrelated from the GBMs and make
stacking viable (see Ensemble Method decision below).

**Alternatives considered**:
- MLP / TabNet: rejected — documented 0.57 vs 0.75 underperformance on this dataset
- TabPFN-2.5: requires GPU/API for > 50k rows; distillation to tree ensemble adds
  complexity with unproven gain on this task
- Single best model: lower ceiling than a well-calibrated 5-model ensemble

---

### Decision: Feature Engineering for Geo Features

**Choice**: GLMM empirical Bayes encoding for `geo_level_1/2/3_id` — compute
class-conditional P(damage_grade=k | geo_level) with adaptive shrinkage based on
estimated between-group variance (τ²). 9 new float features (3 classes × 3 levels).

**Rationale**: `geo_level_3_id` and `geo_level_2_id` are the #1 and #2 most
important features (LightGBM importances ~26.7 and ~20.1). High cardinality
(hundreds to thousands of unique values) makes one-hot encoding impractical.
GLMM-style encoding provides adaptive regularization: rare geo categories shrink
heavily toward the global rate, frequent ones keep their group mean. Superior to
fixed-weight smoothing (manual TE) for high-cardinality IDs [Regularized target
encoding outperforms traditional methods, Springer 2022].

Target encoding is always fitted on the training fold only and applied to
validation/test — no leakage.

**Alternatives considered**:
- One-hot encoding: rejected — creates 10k+ sparse columns on geo_level_3_id
- Manual smoothed TE (fixed smoothing=10): replaced by GLMM; theoretically
  inferior for categories with very different group sizes
- Label encoding: loses the damage-conditional signal entirely

---

### Decision: CatBoost Categorical Handling

**Choice**: Pass ALL categoricals (including `geo_level_*`) directly via
`cat_features` without any pre-encoding. Do NOT enable `task_type: "GPU"`.

**Rationale**: CatBoost's Ordered Target Statistics handle high-cardinality
categoricals without leakage, outperforming manual target encoding. This is
CatBoost's primary advantage over LGB/XGB on this dataset and its main
contribution to ensemble diversity.

GPU mode (`task_type: "GPU"`) is explicitly disabled: it switches from Ordered
target statistics to Borders/BinarizedTarget encoding, causing a catastrophic
−0.0085 F1 drop (0.7489 CPU → 0.7404 GPU). This was verified empirically.

**Alternatives considered**:
- Pre-encode then pass as numeric: rejected — discards the Ordered TS advantage
- task_type GPU: tested, catastrophic for high-cardinality categoricals

---

### Decision: Geo Entity Embeddings

**Choice**: Train a supervised MLP on `geo_level_*` IDs using the full training
set, extract the learned embedding vectors, and append them as numeric features
for LGB and XGB only.

Architecture: 3 embedding tables (dims 8/16/32 → 56D total), 2 hidden layers
[128, 64], dropout 0.3, 30 epochs, Adam + CosineAnnealingLR.

**Rationale**: GLMM encoding captures P(damage | geo) per class — effectively the
first moment. A supervised embedding learns richer latent structure (neighborhood
effects, spatial clusters) that target encoding cannot represent. Empirical result:
LGB +0.0012, XGB +0.0024 OOF.

Embeddings are applied to LGB and XGB only. ExtraTrees and RandomForest use random
feature subsampling per split — this dilutes the dense embedding vectors and was
confirmed to hurt both models (ET −0.0006, RF −0.0011 when embeddings were applied).

**Alternatives considered**:
- Unsupervised autoencoder: rejected in favour of supervised training (task signal
  guides the embedding geometry directly)
- Apply to all models: tested, hurts ET and RF

---

### Decision: Automatic Feature Engineering

**Choice**: Define a pool of 32 candidate features (structural interactions,
geo×material products, age-based composites), then use Optuna (40 trials) to
select the optimal binary subset using OOF F1-micro on fold 0 of LightGBM as
the objective. Apply the 15 selected features to LGB only.

Selected features: `foundation_r_flag`, `foundation_r_x_geo2`, `floors_x_height`,
`height_to_floors`, `mud_stone_x_floors`, `weak_structure`, `geo2_x_floors`,
`rc_eng_x_geo1`, `mud_stone_x_geo2`, `age_x_geo1`, `old_mud_stone`,
`foundation_x_mud`, `position_t_flag`, `secondary_count`, `secondary_x_area`.

**Rationale**: Manual feature selection is expensive and biased. Framing FE as a
combinatorial search problem and delegating it to Optuna is principled and
reproducible. Baseline AutoFE result: fold-0 delta +0.0019, OOF +0.0004, ensemble
+0.0001. Features applied to LGB only — testing on CAT/XGB/ET/RF is deferred.

**Alternatives considered**:
- Manual selection: rejected in favour of systematic search
- Exhaustive search: 2^32 combinations — Optuna with 40 trials is a practical
  approximation

---

### Decision: Ensemble Method

**Choice**: Weighted average of predicted class probabilities from all 5 models.
Weights optimized on OOF predictions via `scipy.optimize.minimize` (Nelder-Mead),
constrained to sum to 1.

Optimal weights: CAT 0.26, LGB 0.25, XGB 0.20, ET 0.15, RF 0.15.

**Rationale**: The 5 models use fundamentally different tree-building strategies
(leaf-wise, symmetric oblivious, depth-wise, bagging × 2) and different categorical
handling, providing genuine diversity. Weighted probability averaging is consistent
with probability calibration theory. Stacking (LogisticRegression meta-learner on
OOF probabilities) was tested exhaustively — see Experiments.

**Stacking notes**: With 3 correlated GBMs, stacking scored 0.7494 vs weighted avg
0.7497. With 5 models (including ET and RF), stacking reached 0.7519 OOF — higher
than the 5-model weighted avg of 0.7520 on some runs. Final submissions use weighted
avg (marginally more stable). Both methods perform comparably at 5 models.

**Alternatives considered**:
- Hard voting: discards probability calibration information
- Rank averaging: less principled for a calibrated multiclass problem
- Stacking only: see above — viable at 5 models, marginal vs weighted avg

---

### Decision: CV Strategy

**Choice**: 5-fold Stratified K-Fold. Optuna on fold 0 only; best hyperparameters
used for all 5 folds. OOF predictions cover all 260k training rows.

**Rationale**: OOF predictions over 260k rows provide a reliable ensemble weight
optimizer. Stratification on `damage_grade` preserves class distribution across
folds. The per-fold training sets are large enough (~208k rows) that Optuna on fold 0
finds hyperparameters that generalize to all folds without significant variance.

Target encoding and geo embeddings are fitted on each fold's training split —
no leakage from validation or test rows.

**Alternatives considered**:
- 80/20 holdout: used in early stages, replaced by k-fold for more robust OOF
- GroupKFold: no meaningful group structure in the data (buildings are independent)
- Full Optuna on all 5 folds: 5× computational cost with marginal gain; fold 0
  findings transfer well empirically

---

### Decision: Hyperparameter Tuning

**Choice**: Optuna TPE sampler on LGB and XGB (30 trials each, fold 0). CatBoost
uses default parameters. ExtraTrees and RandomForest use default parameters.

Post-HPO: LGB and XGB are refit on the full training set (all 260k rows) with
the best hyperparameters found — this step recovered +0.0004 LB on one submission.

**Rationale**: TPE dramatically outperforms grid/random search on GBM
hyperparameter spaces. CatBoost Optuna was run three independent times (20, 30,
and 40 trials); all three converged below CatBoost's default F1. The defaults
are well-calibrated for ordered categorical data. ET and RF defaults (max_depth=25,
n_estimators=1000, max_features=sqrt) were confirmed optimal after 21 Optuna
trials reached no improvement over default.

CatBoost Optuna is NOT a viable lever on this dataset — confirmed definitively.

**Alternatives considered**:
- Grid search: impractical for the GBM hyperparameter space
- Random search: 30–50% less efficient than TPE at the same trial count
- Tune ET/RF with Optuna: tested, defaults win

---

### Decision: Ordinal Structure

**Choice**: Standard multiclass objective (`objective: multiclass`) across all
5 models.

**Rationale**: Frank & Hall decomposition (3-class → 2 binary classifiers
P(y>1), P(y>2)) was implemented and tested. Individual LGB-FH model improved
+0.0003 OOF. However, ensemble LGB-FH+CAT+XGB scored 0.7473 on the leaderboard
vs 0.7478 with standard multiclass (−0.0005). The ordinal structure signal on
this dataset is weak — the GBMs capture it implicitly through the geo and material
features. F&H introduces model complexity without LB gain.

**Alternatives considered**:
- Frank & Hall decomposition: tested, worse on LB
- Ordinal regression with ordinal loss: not tested — F&H failure suggested ordinal
  structure is not a useful inductive bias here
