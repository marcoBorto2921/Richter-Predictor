# Technical Choices — Richter's Predictor

---

### Decision: Model Family

**Choice**: Gradient Boosted Decision Trees (LightGBM + CatBoost + XGBoost)

**Rationale**: GBMs consistently dominate tabular classification benchmarks and this specific
competition. LightGBM reaches F1-micro 0.75+ with proper tuning; neural networks (MLP, TabNet)
score 0.56–0.57 on this dataset — a 0.18 gap. Multiple public solutions and the 2024–2025 tabular
benchmarking literature confirm GBM dominance on structured data with high-cardinality
categoricals [Comprehensive Survey of Tabular Classification 2024–2025].

**Alternatives considered**:
- MLP / TabNet: rejected — documented 0.57 vs 0.75 underperformance on this specific dataset
- TabPFN-2.5: interesting but requires GPU/API for full datasets > 50k rows; distillation to tree ensemble is complex to set up and the gain over a tuned GBM ensemble is not proven on this task
- Random Forest: solid but plateaus at ~0.72; not worth including in ensemble

---

### Decision: Feature Engineering for Geo Features

**Choice**: Target encoding — compute conditional probability P(damage_grade=k | geo_level) for
each of the 3 classes and 3 geo levels → 9 new features. Computed on train fold only.

**Rationale**: `geo_level_3_id` and `geo_level_2_id` are the #1 and #2 most important features
(importance ~26.7 and ~20.1 respectively). High cardinality (hundreds of unique values) makes
one-hot encoding impractical (creates 10k+ sparse columns). Target encoding preserves the
signal with fixed dimensionality. This is the single biggest FE lever reported by public
solutions on this competition.

**Alternatives considered**:
- One-hot encoding: rejected — creates 10k+ sparse columns, catastrophic on geo_level_3_id
- Ordinal encoding (label encoding): loses the damage-conditional signal
- GLMM encoder (feature-engine): more principled regularization, under consideration for Phase 3
  — regularizes automatically based on level frequency, more stable on very high cardinality
  [Regularized target encoding outperforms traditional methods, Springer 2022]
- PCA on all features: rejected — consistently hurts (multiple participant reports)

---

### Decision: CatBoost Categorical Handling

**Choice**: Pass ALL categoricals (including geo_level_*) directly via `cat_features` parameter
without any pre-encoding.

**Rationale**: CatBoost's Ordered Target Statistics handle high-cardinality categorical features
without target leakage, outperforming manual target encoding on this class of feature. This is
CatBoost's primary advantage over LGB/XGB for this dataset [CatBoost vs XGBoost 2025].
Pre-encoding would discard this advantage.

**Alternatives considered**:
- Pre-encode then pass as numeric: rejected — loses CatBoost's ordered TS advantage
- Target-encode then pass as numeric (same as XGB treatment): would work but misses the point
  of including CatBoost in the ensemble (less diversity)

---

### Decision: Ensemble Method

**Choice**: Weighted average of predicted probabilities from LGB + CatBoost + XGBoost. Weights
optimized on holdout via `scipy.optimize.minimize` (Nelder-Mead), constrained to sum to 1.
Starting point: LGB 40%, CAT 35%, XGB 25%.

**Rationale**: The three GBM families use different tree-building strategies (leaf-wise, symmetric
oblivious, depth-wise) and different categorical handling, providing genuine diversity. Weighted
probability averaging consistently outperforms hard voting for this task. Stacking was tested
by multiple participants and found to add complexity without measurable gain at this dataset scale.
[DrivenData community + benchmark papers]

**Alternatives considered**:
- Hard voting: discards probability calibration information
- Stacking meta-learner: overkill for this dataset; not observed in any top-10 public solution
- Rank averaging: less principled than probability averaging for a calibrated multiclass problem
- Adding Random Forest: negligible gain for the added complexity

---

### Decision: CV Strategy

**Choice**: Stratified 80/20 holdout on `damage_grade`.

**Rationale**: Simple and fast. Sufficient for ~260k rows (holdout set has ~52k samples —
large enough for reliable F1-micro estimation). GroupKFold not needed as buildings are
independent observations with no explicit time or group structure. StratifiedKFold (5-fold)
would give more robust estimates but is slower; reserved for Phase 3 if score plateaus.

**Alternatives considered**:
- StratifiedKFold (5-fold): more robust, but 5x training time; not needed until tuning phase
- GroupKFold: no meaningful group structure in this dataset
- Time-based split: no temporal component in the data

---

### Decision: Hyperparameter Tuning

**Choice**: Optuna with TPE sampler, 100 trials per model, optimizing F1-micro on holdout.
Post-HPO: refit on train+val combined with best hyperparameters.

**Rationale**: Bayesian optimization (TPE) dramatically outperforms grid/random search on
GBM hyperparameter spaces. 100 trials is the community standard for this competition.
The post-HPO refit on combined train+val is a critical but often missed step — it consistently
improves results and sometimes reorders model rankings [Tabular Data: Is Deep Learning all
you need? 2025].

**Alternatives considered**:
- Grid search: impractical for the GBM hyperparameter space (num_leaves × depth × lr × ...)
- Random search: 30–50% less efficient than TPE at same trial count
- HyperOpt: equivalent to Optuna TPE but less maintained

---

### Decision: Ordinal Structure

**Choice** (Phase 1): Treat as standard multiclass — `objective: multiclass`.

**Rationale**: Fastest to implement and validate. Most public solutions use standard multiclass
and reach 0.745–0.755 range.

**Planned experiment** (Phase 3, after baseline ≥ 0.75):
- Frank & Hall decomposition: decompose into N-1 binary classifiers ("grade > 1?", "grade > 2?").
  Preserves ordinal rank structure during boosting. Reported HIGH impact on QWK [State of the Art
  in Ordinal Tabular Classification for Earthquake Damage Assessment 2025].
- Dual Loss: cross-entropy + ordinal residual term to enforce rank-consistent probabilities.

**Alternatives considered**:
- Ordinal regression (Frank & Hall): deferred to Phase 3 — adds complexity before baseline is solid
- Direct ordinal loss (GBNet): interesting but requires PyTorch integration overhead
