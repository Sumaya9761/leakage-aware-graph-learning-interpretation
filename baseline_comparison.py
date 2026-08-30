#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                              f1_score, recall_score, confusion_matrix)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    print('XGBoost available')
except ImportError:
    HAS_XGB = False
    print('XGBoost not found — skipping')

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
    print('LightGBM available')
except ImportError:
    HAS_LGBM = False
    print('LightGBM not found — skipping')

import matplotlib.pyplot as plt
import seaborn as sns

print('All imports OK')


# In[2]:


try:
    from google.colab import files
    uploaded = files.upload()   # select adni_diagnosis_dxsum_no_cdr.csv
except ImportError:
    pass


# In[3]:


DATA_PATH  = 'adni_diagnosis_dxsum_no_cdr.csv'
OUTPUT_DIR = Path('.')

SEED          = 42
N_OUTER_FOLDS = 5
NUM_SEEDS     = 5
SEEDS         = [SEED + i for i in range(NUM_SEEDS)]

# 18 source variables (clinical_/mri_/pet_ predictors, no CDR)
# + visit_month + visit_age = 20 predictors
SOURCE_VARS = [
    'clinical_entry_age', 'clinical_GENDER', 'clinical_EDUCAT',
    'clinical_APOE4_count', 'clinical_MMSCORE', 'clinical_FAQTOTAL',
    'clinical_LDELTOTAL', 'clinical_TRABSCOR',
    'mri_hippocampus_vol_mean', 'mri_entorhinal_vol_mean',
    'mri_amygdala_vol_mean', 'mri_inferior_temporal_vol_mean',
    'mri_middle_temporal_vol_mean', 'mri_lateral_ventricle_vol_mean',
    'mri_inf_lat_vent_vol_mean',
    'clinical_ABETA42', 'pet_PTAU', 'pet_TAU',
]
DERIVED_VARS = ['visit_month', 'visit_age']

ID_COLS    = ['subject_id', 'clinical_session_id', 'merge_key', 'time_key']
TARGET_COL = 'DIAGNOSIS'
CLASS_MAP  = {1: 0, 2: 1, 3: 2}   # ADNI: 1=NC, 2=MCI, 3=AD  →  0, 1, 2
CLASS_NAMES = ['NC', 'MCI', 'AD']

print(f'Seeds:       {SEEDS}')
print(f'Outer folds: {N_OUTER_FOLDS}')
print(f'Data:        {DATA_PATH}')
print(f'Predictors:  {len(SOURCE_VARS)} source + {len(DERIVED_VARS)} derived = '
      f'{len(SOURCE_VARS) + len(DERIVED_VARS)}')


# ## 1  Data Loading & Preprocessing

# In[4]:


df_raw = pd.read_csv(DATA_PATH)
print(f'Loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns')


cdr_cols = [c for c in df_raw.columns if 'CDR' in c.upper() or 'CDGLOBAL' in c.upper()]
if cdr_cols:
    raise ValueError(f'CDR columns found (forbidden): {cdr_cols}')
print('CDR guard passed')

df = df_raw.copy()


def parse_visit_month(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    if s in ('sc', 'scr', 'screening', 'bl', 'baseline'):
        return 0.0
    m = re.match(r'^m(\d+)$', s)
    return float(m.group(1)) if m else np.nan

if 'clinical_session_id' in df.columns:
    df['visit_month'] = df['clinical_session_id'].apply(parse_visit_month)
    n_missing = df['visit_month'].isna().sum()
    counts = df['visit_month'].value_counts().sort_index()
    print(f'visit_month: {counts.to_dict()}')
    if n_missing > 0:
        print(f'  {n_missing} rows have unparseable session IDs -> visit_month left as NaN; '
              'imputed per outer-training-fold below (no global fill).')

    # visit_age = entry_age + visit_month / 12, matching paper8_updated.py lines 4354-4356.
    # NaN visit_month propagates to NaN visit_age; both are imputed per outer-training-fold only.
    if 'clinical_entry_age' in df.columns:
        df['visit_age'] = df['clinical_entry_age'] + df['visit_month'] / 12.0
        print('Derived visit_age = clinical_entry_age + visit_month / 12')


df[TARGET_COL] = df[TARGET_COL].map(CLASS_MAP)
assert df[TARGET_COL].notna().all(), 'Unknown DIAGNOSIS values after mapping'
df[TARGET_COL] = df[TARGET_COL].astype(int)
print(f'\nClass distribution (0=NC 1=MCI 2=AD):')
print(df[TARGET_COL].value_counts().sort_index().rename({0:'NC',1:'MCI',2:'AD'}))


# 20 predictors: 18 source variables + visit_month + visit_age. Fixed set, no missingness-based dropping.
feature_cols = [c for c in SOURCE_VARS + DERIVED_VARS if c in df.columns]
missing_requested = [c for c in SOURCE_VARS + DERIVED_VARS if c not in df.columns]
if missing_requested:
    raise KeyError(f'Requested predictors not found in data: {missing_requested}')

print(f'\nFeature columns ({len(feature_cols)}):')
for c in feature_cols:
    pct_nan = df[c].isna().mean() * 100
    print(f'  {c:<40} dtype={df[c].dtype}  NaN={pct_nan:.1f}%')


# In[5]:


# Same split as hybrid_gnn.py's run_nested_cv(): StratifiedGroupKFold(5, shuffle=True, seed=42)
# on subject_id groups. Reusing df[TARGET_COL] (already 0/1/2) since it's the same encoding.
subject_ids = df['subject_id'].values
y_for_split = df[TARGET_COL].values
dummy_X = np.zeros(len(y_for_split))

outer_cv = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)

df['outer_fold'] = -1
for outer_i, (_, test_idx) in enumerate(outer_cv.split(dummy_X, y=y_for_split, groups=subject_ids)):
    df.iloc[test_idx, df.columns.get_loc('outer_fold')] = outer_i

assert (df['outer_fold'] >= 0).all(), 'Some rows were not assigned to an outer fold!'
df['outer_fold'] = df['outer_fold'].astype(int)
print(f'Generated {N_OUTER_FOLDS} outer folds via StratifiedGroupKFold(n_splits={N_OUTER_FOLDS}, '
      f'shuffle=True, random_state={SEED}), same as hybrid_gnn.py run_nested_cv().')

print('\nRows per outer fold (test-set size when that fold is held out):')
print(df['outer_fold'].value_counts().sort_index())

# Pairwise subject disjointness, nonempty partitions, all 3 classes present
fold_subjects = {i: set(df.loc[df['outer_fold'] == i, 'subject_id']) for i in range(N_OUTER_FOLDS)}
for i in range(N_OUTER_FOLDS):
    for j in range(i + 1, N_OUTER_FOLDS):
        assert fold_subjects[i].isdisjoint(fold_subjects[j]), f'Subject leakage between folds {i} and {j}!'
print('Subject-disjointness verified across all outer fold pairs.')

fold_counts_rows = []
for i in range(N_OUTER_FOLDS):
    test_mask_i  = (df['outer_fold'] == i)
    train_mask_i = ~test_mask_i

    assert test_mask_i.sum() > 0,  f'Fold {i}: empty test partition!'
    assert train_mask_i.sum() > 0, f'Fold {i}: empty train partition!'

    train_classes = set(df.loc[train_mask_i, TARGET_COL].unique())
    test_classes  = set(df.loc[test_mask_i,  TARGET_COL].unique())
    assert train_classes == {0, 1, 2}, f'Fold {i}: train partition missing class(es) {({0,1,2} - train_classes)}!'
    assert test_classes  == {0, 1, 2}, f'Fold {i}: test partition missing class(es) {({0,1,2} - test_classes)}!'

    fold_counts_rows.append({
        'fold': i,
        'train_sessions': int(train_mask_i.sum()),
        'test_sessions':  int(test_mask_i.sum()),
        'train_subjects': df.loc[train_mask_i, 'subject_id'].nunique(),
        'test_subjects':  df.loc[test_mask_i,  'subject_id'].nunique(),
    })
print('Nonempty partitions and NC/MCI/AD presence verified for every train/test partition.')

fold_counts_df = pd.DataFrame(fold_counts_rows)
fold_counts_path = OUTPUT_DIR / 'baseline_fold_counts.csv'
fold_counts_df.to_csv(fold_counts_path, index=False)
print(f'\nFold-specific session/subject counts (saved to {fold_counts_path}):')
print(fold_counts_df.to_string(index=False))

# Subject IDs per fold, for verification
fold_subject_ids = {
    str(i): sorted(df.loc[df['outer_fold'] == i, 'subject_id'].astype(str).unique().tolist())
    for i in range(N_OUTER_FOLDS)
}
fold_subjects_path = OUTPUT_DIR / 'baseline_outer_fold_subjects.json'
with open(fold_subjects_path, 'w') as f:
    json.dump({
        'seed': SEED,
        'outer_folds': N_OUTER_FOLDS,
        'shuffle': True,
        'fold_to_subject_ids': fold_subject_ids,
    }, f, indent=2)
print(f'Saved per-fold subject IDs to {fold_subjects_path} for verification.')


# In[6]:


def build_preprocessor(df, train_mask, cols, scale=True):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scaler', StandardScaler()))
    pipe = Pipeline(steps)
    pipe.fit(df.loc[train_mask, cols].values)
    return pipe


# In[7]:


def compute_metrics(y_true, y_pred, y_proba):
    m = {}
    m['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    m['macro_f1']          = f1_score(y_true, y_pred, average='macro', zero_division=0)

    recalls = recall_score(y_true, y_pred, labels=[0, 1, 2],
                           average=None, zero_division=0)
    m['nc_recall']  = recalls[0]
    m['mci_recall'] = recalls[1]
    m['ad_recall']  = recalls[2]

    try:
        m['macro_auc'] = roc_auc_score(
            y_true, y_proba, multi_class='ovr', average='macro'
        )
    except ValueError:
        m['macro_auc'] = float('nan')

    return m


# In[8]:


TREE_MODELS = {'Random Forest', 'Extra Trees', 'XGBoost', 'LightGBM'}

def make_sklearn_models(seed):
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=5000,
            solver='lbfgs', random_state=seed,
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=600, max_features='sqrt', min_samples_leaf=2,
            class_weight='balanced_subsample', n_jobs=-1, random_state=seed,
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=600, max_features='sqrt', min_samples_leaf=2,
            class_weight='balanced', n_jobs=-1, random_state=seed,
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', C=2.0, gamma='scale', class_weight='balanced',
            probability=True, random_state=seed,
        ),

        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            alpha=1e-3, learning_rate_init=5e-4,
            max_iter=600, early_stopping=False,
            random_state=seed,
        ),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=seed, n_jobs=-1,
            verbosity=0,
        )
    else:
        raise ImportError('XGBoost is required (one of the 7 requested models) but is not installed.')
    if HAS_LGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=seed, n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ImportError('LightGBM is required (one of the 7 requested models) but is not installed.')
    return models


print('Sklearn model roster:', list(make_sklearn_models(SEED).keys()))


# ## 4  Main Training Loop (5 reconstructed outer folds × 5-seed ensemble)

# In[9]:


all_results   = []
conf_matrices = {}

print('Starting training...\n' + '='*70)
print(f'Each model is trained with {len(SEEDS)} seeds {SEEDS} per fold; predict_proba is '
      f'averaged across seeds before computing metrics (mirrors the GNN\'s multi-seed ensemble).')

for fold in range(N_OUTER_FOLDS):
    print(f'\n-- Outer fold {fold} {"":-<60}')


    test_mask  = (df['outer_fold'] == fold).values
    train_mask = ~test_mask

    y_train = df.loc[train_mask, TARGET_COL].values
    y_test  = df.loc[test_mask,  TARGET_COL].values
    print(f'   train={train_mask.sum():,}  test={test_mask.sum():,}')


    pre_scaled = build_preprocessor(df, train_mask, feature_cols, scale=True)
    pre_raw    = build_preprocessor(df, train_mask, feature_cols, scale=False)

    X_tr_s = pre_scaled.transform(df.loc[train_mask, feature_cols].values)
    X_te_s = pre_scaled.transform(df.loc[test_mask,  feature_cols].values)

    X_tr_r = pre_raw.transform(df.loc[train_mask, feature_cols].values)
    X_te_r = pre_raw.transform(df.loc[test_mask,  feature_cols].values)

    proba_accum = {}
    for seed in SEEDS:
        models = make_sklearn_models(seed)
        for model_name, model in models.items():
            use_raw = model_name in TREE_MODELS
            X_tr = X_tr_r if use_raw else X_tr_s
            X_te = X_te_r if use_raw else X_te_s

            model.fit(X_tr, y_train)
            proba = model.predict_proba(X_te)
            proba_accum.setdefault(model_name, []).append(proba)

    for model_name, proba_list in proba_accum.items():
        avg_proba = np.mean(proba_list, axis=0)
        y_pred    = avg_proba.argmax(axis=1)

        metrics = compute_metrics(y_test, y_pred, avg_proba)
        metrics.update({'fold': fold, 'model': model_name, 'n_seeds': len(proba_list)})
        all_results.append(metrics)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        conf_matrices.setdefault(model_name, []).append(cm)

        print(f'   {model_name:<26}  '
              f'bal_acc={metrics["balanced_accuracy"]:.3f}  '
              f'auc={metrics["macro_auc"]:.3f}  '
              f'f1={metrics["macro_f1"]:.3f}  '
              f'(avg of {len(proba_list)} seeds)')


print('\nTraining complete.')
results_df = pd.DataFrame(all_results)


# ## 5  Results Aggregation (across the 5 outer folds)

# In[10]:


METRIC_COLS = ['balanced_accuracy', 'macro_auc', 'macro_f1',
               'nc_recall', 'mci_recall', 'ad_recall']

def compute_ci(values, confidence=0.95):
    n   = len(values)
    se  = stats.sem(values)
    t   = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    mar = t * se
    return np.mean(values) - mar, np.mean(values) + mar


summary_rows = []
for model_name, grp in results_df.groupby('model'):
    row = {'model': model_name}
    for col in METRIC_COLS:
        vals         = grp[col].values
        lo, hi       = compute_ci(vals)
        row[f'{col}_mean'] = np.mean(vals)
        row[f'{col}_std']  = np.std(vals, ddof=1)
        row[f'{col}_ci95_lo'] = lo
        row[f'{col}_ci95_hi'] = hi
    summary_rows.append(row)

summary_df = (
    pd.DataFrame(summary_rows)
      .sort_values('balanced_accuracy_mean', ascending=False)
      .reset_index(drop=True)
)


print('Comparison table (sorted by balanced accuracy, mean ± std across 5 outer folds)\n')
display_cols = ['model'] + [f'{m}_mean' for m in METRIC_COLS] + [f'{m}_std' for m in METRIC_COLS]
print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))


# In[11]:


# Compact table: mean ± std for each metric
compact = summary_df[['model']].copy()
for m in METRIC_COLS:
    compact[m] = (
        summary_df[f'{m}_mean'].map('{:.3f}'.format) + ' ± ' +
        summary_df[f'{m}_std'].map('{:.3f}'.format)
    )

print('\nCompact table (mean ± std)\n')
print(compact.to_string(index=False))


# In[12]:


fig, ax = plt.subplots(figsize=(12, 5))

models_ordered = summary_df['model'].tolist()
means  = summary_df['balanced_accuracy_mean'].values
stds   = summary_df['balanced_accuracy_std'].values

x = np.arange(len(models_ordered))
bars = ax.bar(x, means, yerr=stds, capsize=5,
              color=sns.color_palette('muted', len(models_ordered)),
              edgecolor='black', linewidth=0.6)

ax.set_xticks(x)
ax.set_xticklabels(models_ordered, rotation=25, ha='right', fontsize=10)
ax.set_ylabel('Balanced Accuracy (mean ± std, 5 outer folds)')
ax.set_title('Baseline Model Comparison — Balanced Accuracy')
ax.set_ylim(0, 1.05)
ax.axhline(1/3, color='grey', linestyle='--', linewidth=0.8, label='Random chance (3-class)')
ax.legend()

for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'baseline_balanced_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: baseline_balanced_accuracy.png')


# In[13]:


recall_cols = {'NC': 'nc_recall_mean', 'MCI': 'mci_recall_mean', 'AD': 'ad_recall_mean'}

fig, ax = plt.subplots(figsize=(14, 5))
n_models = len(models_ordered)
n_groups = 3
bar_w    = 0.25
palette  = sns.color_palette('Set2', n_groups)

for gi, (cls_name, col) in enumerate(recall_cols.items()):
    offsets = x + (gi - 1) * bar_w
    vals    = [summary_df.loc[summary_df['model'] == m, col].values[0]
               for m in models_ordered]
    ax.bar(offsets, vals, width=bar_w, label=cls_name,
           color=palette[gi], edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(models_ordered, rotation=25, ha='right', fontsize=10)
ax.set_ylabel('Recall (mean across 5 outer folds)')
ax.set_title('Per-class Recall by Model')
ax.set_ylim(0, 1.05)
ax.legend(title='Class')

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'baseline_per_class_recall.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: baseline_per_class_recall.png')


# In[14]:


model_list = list(conf_matrices.keys())
n = len(model_list)
ncols = 3
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = np.array(axes).flatten()

for ax, model_name in zip(axes, model_list):
    cm_avg = np.mean(conf_matrices[model_name], axis=0).astype(float)
    # Row-normalise to show recall per class
    cm_norm = cm_avg / cm_avg.sum(axis=1, keepdims=True)

    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        vmin=0, vmax=1, ax=ax,
        annot_kws={'size': 9},
    )
    ax.set_title(model_name, fontsize=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')


for ax in axes[n:]:
    ax.set_visible(False)

fig.suptitle('Confusion Matrices (row-normalised, averaged over 5 outer folds)', y=1.01)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'baseline_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: baseline_confusion_matrices.png')


# In[15]:


# Per-fold results
per_fold_path = OUTPUT_DIR / 'baseline_results_per_fold.csv'
results_df.to_csv(per_fold_path, index=False)
print(f'Saved: {per_fold_path}')

# Summary (mean ± std ± 95% CI across the 5 outer folds)
summary_path = OUTPUT_DIR / 'baseline_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f'Saved: {summary_path}')

print('\nFinal summary (balanced accuracy):')
print(summary_df[['model', 'balanced_accuracy_mean', 'balanced_accuracy_std',
                   'balanced_accuracy_ci95_lo', 'balanced_accuracy_ci95_hi']]
      .to_string(index=False, float_format=lambda x: f'{x:.4f}'))

