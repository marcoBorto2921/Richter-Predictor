"""
EDA + Error Analysis — Richter's Predictor
Mode 2, Level 0 (Plateau Buster) + EDA Steps 0,1,3b,4b,5,8,9
Run from project root: python notebooks/01_eda.py
"""

import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUTS = Path("reports/eda")
OUTPUTS.mkdir(parents=True, exist_ok=True)

TARGET_COL = "damage_grade"

# ──────────────────────────────────────────────
# STEP 0 — Load
# ──────────────────────────────────────────────
log.info("=== STEP 0 — LOAD ===")
train_values = pd.read_csv("data/raw/train_values.csv", encoding="utf-8")
train_labels = pd.read_csv("data/raw/train_labels.csv", encoding="utf-8")
test = pd.read_csv("data/raw/test_values.csv", encoding="utf-8")

train = train_values.merge(train_labels, on="building_id")

log.info("Train: %s | Test: %s", train.shape, test.shape)
log.info(
    "Cols in train not in test: %s",
    set(train.columns) - set(test.columns) - {TARGET_COL},
)
log.info("Dtypes:\n%s", train.dtypes.value_counts().to_string())

# ──────────────────────────────────────────────
# STEP 1 — Target distribution
# ──────────────────────────────────────────────
log.info("\n=== STEP 1 — TARGET ===")
vc = train[TARGET_COL].value_counts(normalize=True).sort_index()
log.info("Class distribution (normalized):\n%s", vc.to_string())
log.info("Counts:\n%s", train[TARGET_COL].value_counts().sort_index().to_string())

fig, ax = plt.subplots(figsize=(6, 4))
train[TARGET_COL].value_counts().sort_index().plot(
    kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"]
)
ax.set_title("Target distribution — damage_grade")
ax.set_xlabel("damage_grade")
ax.set_ylabel("count")
plt.tight_layout()
fig.savefig(OUTPUTS / "01_target_distribution.png", dpi=100)
plt.close()

# ──────────────────────────────────────────────
# STEP 3b — Categoricals: chi2 vs target
# ──────────────────────────────────────────────
log.info("\n=== STEP 3b — CHI2 CATEGORICALS vs TARGET ===")
cat_cols = train.select_dtypes(include="object").columns.tolist()
num_cols = [
    c
    for c in train.select_dtypes(include="number").columns
    if c not in ("building_id", TARGET_COL)
]

log.info("Numeric features: %d | Categorical: %d", len(num_cols), len(cat_cols))

chi2_results = []
for col in cat_cols:
    ct = pd.crosstab(train[col], train[TARGET_COL])
    chi2, p, dof, _ = chi2_contingency(ct)
    chi2_results.append(
        {
            "feature": col,
            "chi2": round(chi2, 2),
            "p_value": round(p, 6),
            "n_unique": train[col].nunique(),
        }
    )

# Also run chi2 on binary/integer columns (superstructure flags, etc.)
binary_cols = [c for c in num_cols if train[c].nunique() <= 5]
for col in binary_cols:
    ct = pd.crosstab(train[col].astype(str), train[TARGET_COL])
    chi2, p, dof, _ = chi2_contingency(ct)
    chi2_results.append(
        {
            "feature": col,
            "chi2": round(chi2, 2),
            "p_value": round(p, 6),
            "n_unique": train[col].nunique(),
        }
    )

chi2_df = pd.DataFrame(chi2_results).sort_values("chi2", ascending=False)
log.info("Chi2 vs damage_grade (all features):\n%s", chi2_df.to_string(index=False))

# ──────────────────────────────────────────────
# STEP 4b — Distribution shift: train vs test
# ──────────────────────────────────────────────
log.info("\n=== STEP 4b — TRAIN vs TEST SHIFT ===")


def population_stability_index(
    train_col: pd.Series, test_col: pd.Series, n_bins: int = 10
) -> float:
    """Compute PSI between train and test distributions."""
    combined = pd.concat([train_col, test_col]).dropna()
    bins = np.quantile(combined, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 3:
        return 0.0
    train_counts = np.histogram(train_col.dropna(), bins=bins)[0]
    test_counts = np.histogram(test_col.dropna(), bins=bins)[0]
    train_pct = (train_counts / len(train_col)).clip(1e-6)
    test_pct = (test_counts / len(test_col)).clip(1e-6)
    return float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))


shift_report = []
for col in num_cols:
    if col not in test.columns:
        continue
    psi = population_stability_index(train[col], test[col])
    ks_stat, ks_pval = ks_2samp(train[col].dropna(), test[col].dropna())
    shift_report.append(
        {
            "feature": col,
            "psi": round(psi, 4),
            "ks_stat": round(ks_stat, 4),
            "ks_pval": round(ks_pval, 4),
            "severity": "SEVERE" if psi > 0.2 else ("MODERATE" if psi > 0.1 else "ok"),
        }
    )

shift_df = pd.DataFrame(shift_report).sort_values("psi", ascending=False)
log.info("Distribution shift (PSI):\n%s", shift_df.to_string(index=False))

severe = shift_df[shift_df["severity"] == "SEVERE"]["feature"].tolist()
moderate = shift_df[shift_df["severity"] == "MODERATE"]["feature"].tolist()
log.info("SEVERE shift (%d): %s", len(severe), severe)
log.info("MODERATE shift (%d): %s", len(moderate), moderate)

# Adversarial validation (lightweight)
log.info("\nAdversarial validation...")
av_features = [c for c in num_cols if c in test.columns]
train_av = train[av_features].fillna(-999).copy()
test_av = test[av_features].fillna(-999).copy()
combined_av = pd.concat(
    [train_av.assign(_is_test=0), test_av.assign(_is_test=1)], ignore_index=True
)
X_av = combined_av[av_features].values
y_av = combined_av["_is_test"].values

clf_av = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
cv_av = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
av_aucs = [
    roc_auc_score(
        y_av[vi], clf_av.fit(X_av[ti], y_av[ti]).predict_proba(X_av[vi])[:, 1]
    )
    for ti, vi in cv_av.split(X_av, y_av)
]
av_auc_mean = np.mean(av_aucs)
clf_av.fit(X_av, y_av)
av_importance = pd.Series(clf_av.feature_importances_, index=av_features).sort_values(
    ascending=False
)
log.info("AV AUC: %.4f +/- %.4f", av_auc_mean, np.std(av_aucs))
log.info(
    "Top 10 features separating train vs test:\n%s", av_importance.head(10).to_string()
)

# ──────────────────────────────────────────────
# STEP 5 — Correlation with target
# ──────────────────────────────────────────────
log.info("\n=== STEP 5 — CORRELATION WITH TARGET ===")
corr_pearson = train[num_cols + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
corr_spearman = (
    train[num_cols + [TARGET_COL]].corr(method="spearman")[TARGET_COL].drop(TARGET_COL)
)

log.info(
    "Top 15 Pearson:\n%s",
    corr_pearson.abs().sort_values(ascending=False).head(15).to_string(),
)
log.info(
    "\nTop 15 Spearman:\n%s",
    corr_spearman.abs().sort_values(ascending=False).head(15).to_string(),
)

corr_matrix = train[num_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [
    (col, row, upper.loc[row, col])
    for col in upper.columns
    for row in upper.index
    if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > 0.90
]
log.info("\nHigh correlation pairs (>0.90): %d", len(high_corr_pairs))
for c1, c2, v in sorted(high_corr_pairs, key=lambda x: -x[2])[:10]:
    log.info("  %s <-> %s: %.3f", c1, c2, v)

# Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    train[num_cols].corr(),
    ax=ax,
    cmap="coolwarm",
    center=0,
    annot=len(num_cols) <= 20,
    fmt=".1f",
    linewidths=0.3,
)
ax.set_title("Feature correlation matrix (train)")
plt.tight_layout()
fig.savefig(OUTPUTS / "05_correlation_matrix.png", dpi=80)
plt.close()

# ──────────────────────────────────────────────
# STEP 8 — RF quick importance
# ──────────────────────────────────────────────
log.info("\n=== STEP 8 — RF QUICK IMPORTANCE ===")
X_rf = train[num_cols].fillna(-999)
y_rf = train[TARGET_COL]
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_rf, y_rf)
importance = pd.Series(rf.feature_importances_, index=num_cols).sort_values(
    ascending=False
)
log.info("Top 20 RF importance:\n%s", importance.head(20).to_string())
log.info("\nBottom 10 (lowest signal):\n%s", importance.tail(10).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
importance.head(20).sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 20 RF feature importance (quick, numeric only)")
plt.tight_layout()
fig.savefig(OUTPUTS / "08_rf_importance.png", dpi=100)
plt.close()

# ──────────────────────────────────────────────
# MODE 2, LEVEL 0 — Error analysis on ensemble OOF
# ──────────────────────────────────────────────
log.info("\n=== MODE 2 LEVEL 0 — ERROR ANALYSIS (ensemble OOF) ===")

# Load OOF probabilities from all models
model_names = ["lgb", "cat", "xgb", "et", "rf"]
val_probas = {}
for m in model_names:
    path = f"models/{m}_val_proba.npy"
    if Path(path).exists():
        val_probas[m] = np.load(path)
        log.info("Loaded %s val proba: %s", m, val_probas[m].shape)

# Weighted ensemble (same weights as last ensemble run)
weights = {"lgb": 0.2424, "cat": 0.2704, "xgb": 0.1714, "et": 0.1638, "rf": 0.1521}
ensemble_proba = sum(weights[m] * val_probas[m] for m in model_names if m in val_probas)
ensemble_pred = ensemble_proba.argmax(axis=1) + 1  # damage_grade is 1-indexed

# Align with train labels (same ordering as kfold)
y_true_raw = train.sort_values("building_id")[
    TARGET_COL
].values  # NOTE: may not match fold order
# Use the index from training labels directly (they're already aligned)
y_true = train[TARGET_COL].values  # 260601 samples

log.info("\nPer-class classification report:")
report = classification_report(
    y_true, ensemble_pred, target_names=["grade_1", "grade_2", "grade_3"]
)
log.info("\n%s", report)

# Confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_true,
    ensemble_pred,
    display_labels=["grade_1", "grade_2", "grade_3"],
    ax=ax,
    normalize="true",
    values_format=".2f",
    cmap="Blues",
)
ax.set_title("Normalized Confusion Matrix — Ensemble OOF")
plt.tight_layout()
fig.savefig(OUTPUTS / "m2_confusion_matrix.png", dpi=100)
plt.close()
log.info("Confusion matrix saved.")

# Hardest samples: lowest confidence in true class
true_class_prob = ensemble_proba[np.arange(len(y_true)), y_true - 1]  # 0-indexed
hardest_idx = np.argsort(true_class_prob)[:200]
hardest_df = train.iloc[hardest_idx].copy()
hardest_df["true_label"] = y_true[hardest_idx]
hardest_df["pred_label"] = ensemble_pred[hardest_idx]
hardest_df["true_class_prob"] = true_class_prob[hardest_idx]
hardest_df = hardest_df.sort_values("true_class_prob")

log.info("\nTop 10 hardest samples:")
log.info(
    hardest_df[
        ["building_id", "true_label", "pred_label", "true_class_prob"] + num_cols[:5]
    ]
    .head(10)
    .to_string(index=False)
)

# Error rate by feature value (find patterns in hardest samples)
log.info("\n--- Pattern analysis in 200 hardest samples ---")
hard_mask = np.zeros(len(train), dtype=bool)
hard_mask[hardest_idx] = True

for col in [
    "foundation_type",
    "roof_type",
    "land_surface_condition",
    "position",
    "plan_configuration",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_rc_non_engineered",
    "has_superstructure_rc_engineered",
]:
    if col not in train.columns:
        continue
    hard_rate = train.loc[hard_mask, col].value_counts(normalize=True)
    all_rate = train[col].value_counts(normalize=True)
    diff = (hard_rate - all_rate).sort_values(ascending=False)
    top_overrep = diff.head(3)
    if top_overrep.abs().max() > 0.05:
        log.info(
            "  %s — overrepresented in hard samples: %s", col, top_overrep.to_dict()
        )

# Confusion breakdown: which class pairs are most confused
log.info("\n--- Confusion breakdown ---")
conf_matrix = pd.crosstab(y_true, ensemble_pred, rownames=["true"], colnames=["pred"])
log.info("\n%s", conf_matrix.to_string())

# Error rate by geo_level_1 (are some geographic zones harder?)
train_copy = train.copy()
train_copy["error"] = (y_true != ensemble_pred).astype(int)
geo1_err = (
    train_copy.groupby("geo_level_1_id")["error"].mean().sort_values(ascending=False)
)
log.info("\nTop 10 geo_level_1 by error rate:\n%s", geo1_err.head(10).to_string())

# Error rate by age quantile
train_copy["age_bin"] = pd.qcut(train_copy["age"], q=10, duplicates="drop")
age_err = train_copy.groupby("age_bin", observed=True)["error"].mean()
log.info("\nError rate by age quantile:\n%s", age_err.to_string())

# Error rate by height_percentage
train_copy["height_bin"] = pd.qcut(
    train_copy["height_percentage"], q=5, duplicates="drop"
)
height_err = train_copy.groupby("height_bin", observed=True)["error"].mean()
log.info("\nError rate by height quantile:\n%s", height_err.to_string())

# ──────────────────────────────────────────────
# STEP 9 — Domain hypotheses
# (printed for manual review — no code)
# ──────────────────────────────────────────────
log.info("\n=== STEP 9 — DOMAIN HYPOTHESES ===")
hypotheses = """
IPOTESI 1: Edifici con fondamenta deboli (r = dirt/bamboo) in zone ad alta scossa (alto geo_damage) collassano totalmente (grade 3), ma questo effetto è non-lineare — serve interazione foundation_type × geo_damage.
FEATURE: foundation_type_onehot × geo_level_2_class2_encoding
VERIFICA: boxplot geo_damage per foundation_type; confusion matrix per foundation_type

IPOTESI 2: Il numero di piani amplifica il danno solo oltre una soglia (≥3 piani) — sotto la soglia non c'è effetto. Catturato solo parzialmente da floors_ge3 (binario).
FEATURE: count_floors_pre_eq^2 (quadratic), o floors_x_height = floors × height_percentage
VERIFICA: error rate per floors bin

IPOTESI 3: Edifici con uso secondario (mercato, scuola, ufficio) sono costruiti più solidamente → danno atteso minore. Ma uso misto (residenziale + commerciale) aumenta esposizione al carico.
FEATURE: secondary_use_type (one-hot delle categorie specifiche, non solo any), has_secondary_use_any × area_percentage
VERIFICA: chi2 ogni flag secondario vs target

IPOTESI 4: La combinazione di materiale (superstructure) × età cattura degrado: mattoni vecchi degradano, cemento nuovo regge. Interazione: cement_mortar_brick × age_bin.
FEATURE: has_superstructure_cement_mortar_brick × age, has_superstructure_mud_mortar_stone × age
VERIFICA: F1-micro per interazione vs singola feature

IPOTESI 5: Il danno di un edificio è influenzato dal danno degli edifici vicini (effetto contagio strutturale). geo_level_3 cattura già questo parzialmente, ma la *varianza* del danno nel quartiere potrebbe aggiungere segnale.
FEATURE: std(damage_grade per geo_level_3) → std_damage_geo3 (target-encoded OOF)
VERIFICA: Pearson di std_damage_geo3 vs target
"""
log.info(hypotheses)

log.info("\nAll plots saved to: %s", OUTPUTS.resolve())
log.info("Files: %s", [f.name for f in sorted(OUTPUTS.glob("*.png"))])
