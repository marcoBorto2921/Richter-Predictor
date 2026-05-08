# Experiments — Richter's Predictor

Full record of experiments run during development. Organized thematically.
Each entry includes the hypothesis, what was done, the result, and the verdict.

---

## Score Progression

| Date | Configuration | OOF F1 | LB F1 | Notes |
|------|--------------|--------|-------|-------|
| 2026-04-28 | LGB baseline | 0.7460 | — | no Optuna |
| 2026-04-28 | XGB baseline | 0.7471 | — | no Optuna |
| 2026-04-28 | CAT baseline | 0.7514 | — | no Optuna |
| 2026-04-28 | LGB+XGB Optuna + CAT ensemble | 0.7511 | **0.7476** | rank 595/2675 |
| 2026-04-29 | + GLMM encoding | 0.7506 | **0.7478** | rank 578 |
| 2026-04-29 | + 5-fold CV | 0.7496 | **0.7484** | rank 513 |
| 2026-04-30 | + ThresholdReplacer | 0.7499 | — | rare geo → sentinel |
| 2026-04-30 | + ExtraTrees (4th) + stacking | **0.7508** | **0.7503** | rank 343 |
| 2026-05-01 | + RandomForest (5th) + stacking | **0.7516** | **0.7506** | |
| 2026-05-01 | + geo entity embeddings | **0.7519** | **0.7510** | +0.0004 LB |
| 2026-05-02 | 5-model weighted avg + embeddings | **0.7520** | **0.7513** | rank 255 |
| 2026-05-04 | + XGB re-Optuna (107D feature space) | **0.7523** | — | new best OOF |
| 2026-05-05 | + AutoFE 15 features (LGB) | **0.7524** | **0.7518** | rank ~225, top 9% |

---

## 1. Feature Engineering

### GLMM Empirical Bayes Encoding

**Hypothesis**: Adaptive shrinkage based on estimated between-group variance (τ²)
is more principled than fixed-weight target encoding — rare geo categories should
shrink more aggressively toward the global rate.

**What was done**: Replaced manual smoothed TE (smoothing=10 fixed) with an
analytical GLMM-style encoder. Method-of-moments estimate of τ² per column.
`category_encoders.GLMMEncoder` was considered but killed after 10+ minutes
processing geo_level_3 (11k categories, 260k rows) — the custom O(n)
implementation computes the same shrinkage formula instantly.

**Result**: LGB +0.0020, XGB +0.0010 individually. LB +0.0002 (0.7476 → 0.7478).

**Verdict**: Marginal. GLMM reduces individual model overfitting on rare geo
categories but makes LGB and XGB more similar to CatBoost, reducing diversity.
Individual gains don't fully propagate to the ensemble. Kept as default encoding.

---

### Low-Cardinality Categorical Target Encoding

**Hypothesis**: `foundation_type`, `ground_floor_type`, `other_floor_type`,
`roof_type` are label-encoded integers for LGB/XGB — this loses the conditional
damage signal. GLMM TE (3 floats per column × 8 columns = 24 features) should
recover it.

**What was done**: Applied GLMM encoding to the 8 low-cardinality categorical
columns for LGB/XGB (chi2 correlations with target: 30k–48k). CatBoost receives
them raw via `cat_features` as before.

**Result**: LGB 0.7498 (+0.0009 vs 0.7489), XGB 0.7511 (+0.0012 vs 0.7499).
Ensemble OOF **0.7522** (+0.0002 vs 0.7520). Marginal.

**Verdict**: Kept enabled. Individual models improve, ensemble gain is small
because CatBoost already captures this signal natively — the new features reduce
cross-model diversity slightly.

---

### Geo Damage Variance Feature

**Hypothesis**: std(damage_grade) per geo_level_3 (OOF-fitted, smoothed) captures
structural heterogeneity within geographic zones — signal beyond the mean encoding.

**What was done**: `_fit_geo_damage_variance()` in features.py, config-gated.
Smoothed std per geo_level_3 group.

**Result**: LGB OOF **0.7480** (−0.0018 vs baseline 0.7498).

**Verdict**: Rejected. The geo_level_3 damage variance is already captured
implicitly by the GLMM TE (mean per zone × 3 classes). Adding the std introduces
noise without incremental signal. Config key `geo3_damage_std: false`.

---

### AutoFE via Optuna

**Hypothesis**: Manually selecting from 32 candidate features (structural
interactions, geo×material products, age composites) is biased. Framing feature
selection as a binary combinatorial search and delegating to Optuna is principled
and reproducible.

**What was done**: `src/autofe_search.py` — Optuna TPE, 40 trials, LGB fold-0
OOF F1 as objective. Each trial is a binary vector over the 32-feature pool.
Best trial found in trial #3 (fast convergence). 15 of 32 candidates selected.

Selected features: foundation_r_flag, foundation_r_x_geo2, floors_x_height,
height_to_floors, mud_stone_x_floors, weak_structure, geo2_x_floors, rc_eng_x_geo1,
mud_stone_x_geo2, age_x_geo1, old_mud_stone, foundation_x_mud, position_t_flag,
secondary_count, secondary_x_area.

**Result**: LGB OOF 0.7494 (+0.0004 vs restore baseline 0.7490). Ensemble wavg
OOF 0.7524 (+0.0001). LB 0.7518 (+0.0001 vs 0.7517).

**Verdict**: Confirmed. Modest gain. Applied to LGB only — not yet tested on
CAT/XGB/ET/RF.

---

### Frequency Encoding + Geo×Structure Interactions

**Hypothesis**: Frequency encoding of geo columns + interaction terms between
geo_level_1/2 and structure type (6 new features total) should capture joint
geographic-structural damage patterns.

**What was done**: Added freq encodings for geo_level_1/2/3 and
geo_level_1/2 × structure_type interaction features.

**Result**: Individual models improved (LGB +0.0010, XGB +0.0022) but ensemble
OOF dropped to 0.7492 (−0.0007 vs 0.7499). Ensemble weights flattened:
0.39/0.35/0.26 vs 0.65/0.34/0.00 before. Reverted.

**Verdict**: Rejected. This is a recurring pattern on this dataset: features that
reduce the structural difference between LGB/XGB and CatBoost hurt the ensemble
even when they help individual models. CatBoost already captures geo×structure
interactions internally — adding them explicitly to LGB/XGB makes the models
converge to similar representations.

---

## 2. Ensemble Architecture

### 3-Model Stacking (GBMs Only)

**Hypothesis**: LogisticRegression on OOF probabilities (9 input features:
3 models × 3 classes) finds non-linear combinations that the weighted avg misses.

**What was done**: LogisticRegression stacking with 5-fold CV on OOF proba.
Tested C={0.01, 0.1, 1, 10}. Also tested adding argmax rank features.

**Result**: Stacking OOF 0.7494 vs weighted avg 0.7497. Adding rank features
hurt further (0.7465). All C values converge to ~0.7494.

**Verdict**: Failed. The 3 GBMs are too correlated for a meta-learner to
extract useful information. The LR coefficients end up approximately equal
— the stacking degrades to an unoptimized average.

---

### ExtraTrees as 4th Model

**Hypothesis**: ExtraTrees (bootstrap=False, random threshold splits) produces
decorrelated predictions vs GBMs — adds genuine diversity that enables stacking.

**What was done**: ExtraTreesClassifier, 1000 trees, max_depth=25, no Optuna.

**Result**: ET standalone OOF 0.7486. 4-model weighted avg 0.7500 (+0.0001).
4-model stacking **0.7508** (+0.0009 vs 3-model). Stacking folds:
0.7508/0.7525/0.7520/0.7526/0.7500.

**Verdict**: Confirmed. ET's fundamentally different inductive bias (no bootstrap,
random threshold) is enough to break the 3-GBM correlation deadlock. Stacking
became viable.

---

### ExtraTrees Optuna HPO

**Hypothesis**: ET with tuned max_depth, min_samples_leaf, max_features outperforms
default params (OOF 0.7486).

**What was done**: Optuna, 40 trials on fold 0. Search: max_depth [3,40],
min_samples_leaf [1,50], max_features [sqrt, log2, 0.3, 0.5, 0.7, 1.0].

**Result**: Best fold-0 after 21 trials = 0.7471 < default 0.7486. Aborted.

**Verdict**: Failed. Default params are already optimal. ET's strength comes from
its structural differences from GBMs, not from tuned regularization.

---

### Random Forest as 5th Model

**Hypothesis**: RF (bootstrap=True, best split) adds diversity decorrelated from
both ET (no bootstrap) and the 3 GBMs.

**What was done**: RandomForestClassifier, 1000 trees, max_depth=25, no Optuna.

**Result**: RF standalone OOF 0.7468 (weaker than ET). 5-model weighted avg
0.7512 (+0.0012 vs 4-model 0.7500). 5-model stacking **0.7516** (+0.0008).
LB 0.7506 (rank ~310).

**Verdict**: Confirmed. RF's bootstrap mechanism adds useful variance diversity
even though its standalone performance is weaker than ET. The 5-model ensemble
is materially better than the 4-model version.

Optimal weighted avg weights: CAT 0.49, ET 0.15, RF 0.16, LGB 0.14, XGB 0.06
(at this stage — shifted after embedding integration).

---

### Geo Embeddings on ET and RF

**Hypothesis**: Neural geo embedding features (+56D) should also help ET and RF.

**What was done**: Retrained ET and RF with embedding columns appended.

**Result**: ET −0.0006, RF −0.0011 vs their baselines without embeddings.

**Verdict**: Failed. Random feature subsampling per split (sqrt features) dilutes
the dense embedding vectors — the models cannot consistently use the embedding
signal when they only see a random subset of features at each node. Hard-coded
`use_embeddings=False` for ET and RF in train.py.

---

## 3. Neural Components

### Geo Entity Embeddings

**Hypothesis**: A supervised MLP learns richer latent structure for geo_level_*
than target encoding alone — neighborhood effects, spatial clusters, hierarchical
geography.

**What was done**: `src/geo_embeddings.py` — MLP with embedding tables (dims
8/16/32), 2 hidden layers [128, 64], dropout 0.3, Adam, CosineAnnealingLR,
30 epochs, trained on full training set. Embedding vectors appended as numeric
features for LGB and XGB only.

**Result**: LGB +0.0012, XGB +0.0024 OOF. 5-model stacking 0.7519 (+0.0003).
5-model weighted avg 0.7520 (+0.0008 vs 0.7512). LB 0.7513 (rank 255,
+0.0007 vs 0.7506).

Side effect: LGB best_iter dropped from ~600 to ~300 after embedding integration —
the embeddings carry the geo signal more directly, allowing earlier stopping.

**Verdict**: Confirmed. Best single lever in Phase 5. Effect larger on XGB
than LGB — XGB benefits more from dense numeric representations.

---

### Frank & Hall Ordinal Decomposition

**Hypothesis**: Decomposing the 3-class problem into 2 binary classifiers
("grade > 1?" and "grade > 2?") forces the model to respect ordinal rank
structure during boosting, recovering signal from the ordinal target.

**What was done**: `src/train_fh.py` — two LGB and two XGB binary classifiers
per level. Probabilities reconstructed as: P(1) = 1 − P(y>1),
P(2) = P(y>1) − P(y>2), P(3) = P(y>2).

**Result**: LGB-FH OOF 0.7483 (+0.0003 vs multiclass). XGB-FH OOF 0.7469
(−0.0012 vs multiclass). Ensemble LGB-FH+CAT+XGB: 0.7513 OOF, **0.7473 LB**
(−0.0005 vs 0.7478 with standard multiclass).

**Verdict**: Failed on LB. LGB-FH slightly improved OOF but XGB-FH was worse
and the ensemble overfit the holdout structure. The ordinal rank signal is weak
on this dataset — the GBMs capture it implicitly through geo features without
explicit decomposition.

---

### Threshold Optimization

**Hypothesis**: Argmax is suboptimal for class-imbalanced targets. Sweeping
decision thresholds for class 1 and class 3 can improve F1-micro.

**What was done**: `optimize_thresholds` in ensemble.py — Nelder-Mead over
[t0, t2], starting from [1/3, 1/3] on OOF probabilities.

**Result**: Optimal thresholds t0=0.4693, t2=0.4935 → delta +0.0000.
Thresholds converge to near-argmax.

**Verdict**: No gain. The stacking LogisticRegression and Nelder-Mead weight
optimizer together already calibrate the output probabilities well. No systematic
bias remains for a threshold sweep to correct.

---

## 4. Hyperparameter Tuning

### LGB Re-Optuna (107D Feature Space)

**Hypothesis**: The original LGB hyperparameters were tuned on 51 features
(no embeddings). With 107D (including 56 embedding features), the optimal
hyperparameters may differ.

**Result**: Best params: num_leaves=54, lr=0.056, colsample=0.52, subsample=0.66.
OOF **0.7489** (+0.0001 vs 0.7488). Ensemble invariant.

**Verdict**: Marginal. The original params were already well-adapted to the 107D
space. Re-Optuna cost ~43 min for +0.0001 OOF that did not propagate to the ensemble.

---

### XGB Re-Optuna (107D Feature Space)

**Hypothesis**: Same as LGB re-Optuna but XGB showed a larger gain from the
embedding features (+0.0024), suggesting its optimal params may shift more.

**Result**: Best params: max_depth=7, lr=0.040. OOF **0.7499** (+0.0004 vs 0.7495).
5-model weighted avg 0.7523 (new best OOF at the time).

**Verdict**: Marginal gain, worth the cost. max_depth=7 (was the default already)
confirmed. Lower lr compensates for richer 107D feature space.

---

### CatBoost Optuna (3 independent runs)

**Hypothesis**: CatBoost's default hyperparameters can be improved with Optuna.

**Run 1**: 20 trials, fold 0 — aborted at trial 8 (best 0.7481 < default 0.7489,
52 min total).
**Run 2**: 30 trials, fold 0 — completed, best 0.7473 < default 0.7489.
**Run 3**: 12/30 trials — best 0.7473 < default 0.7489. Stopped early.

**Verdict**: Definitively failed. CatBoost's defaults are well-calibrated for
ordered categorical data. The search space (depth, l2_leaf_reg, lr,
bagging_temperature, random_strength) shows no clear gradient toward improvement.
Do not attempt CatBoost HPO on this dataset.

---

### XGB Expanded Search Space

**Hypothesis**: Adding gamma [0, 2] and extending max_depth to 12 (with 100 trials
on GPU) finds configurations that the original 50-trial search missed.

**Result**: OOF 0.7469 vs 0.7471 with old 50-trial search. Worse.

**Verdict**: Failed. A wider search space with the same trial budget disperses
Optuna's exploration — the TPE sampler cannot build an accurate surrogate model
over the larger space in 100 trials. Narrower space + fewer trials outperformed.

---

### CatBoost GPU

**Hypothesis**: `task_type: "GPU"` accelerates CatBoost training without degrading
accuracy.

**Result**: OOF **0.7404** vs 0.7489 CPU (−0.0085).

**Verdict**: Catastrophic. GPU mode switches from Ordered target statistics
(the default, designed for categorical features) to Borders/BinarizedTarget
encoding. This is not a configuration issue — it is a fundamental algorithmic
switch that destroys the main CatBoost advantage on high-cardinality categoricals.
Never use `task_type: "GPU"` with `cat_features` on this class of dataset.

---

### Fold-Weighted Bagging

**Hypothesis**: Weight test predictions by each fold's validation F1 before
averaging — better folds contribute more.

**Result**: Fold F1 values across all 5 folds vary by ±0.001 — the weights
converge to ~0.200 each. Test probabilities are virtually identical to uniform
average.

**Verdict**: No impact. Fold quality is too uniform for differential weighting
to make a difference. Would be relevant if fold quality varied by ≥0.005.
