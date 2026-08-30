#!/usr/bin/env python3
"""
Leakage-aware hybrid GNN for NC/MCI/AD diagnosis-label classification.

The model combines:
  - a population-similarity GCN,
  - a within-subject temporal GCN, and
  - a visit-level MLP branch.

Evaluation uses subject-level splitting and supports:
  - primary held-out evaluation,
  - nested cross-validation,
  - inductive sensitivity analysis,
  - architectural ablations, and
  - post-hoc explanation analyses.

Default population-graph similarity is based on cognitive/functional
features using an RBF kernel and subject-aware top-K sparsification.

Usage:
  python hybrid_gnn.py \
    --single single study_data_*.csv \
    --out_dir results/main_model \
    --verbose
"""
import re, json, argparse, random, time, math, copy, itertools
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist

import torch
from torch import nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.pipeline import Pipeline as SkPipeline


# Utilities

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_subject_key(cols):
    subject_regex = re.compile(r"(subject|oasis|oas3|participant).*id|^subject$", re.I)
    for c in cols:
        if subject_regex.search(c):
            return c
    for c in ["subject_id", "subjectid", "oasisid", "oas3id", "participant_id", "subject"]:
        if c in cols:
            return c
    return None


def parse_visit_month(session_id):
    """Parse clinical_session_id to numeric months from screening. Returns NaN for unrecognized strings."""
    if pd.isna(session_id):
        return np.nan
    s = str(session_id).strip().lower()
    if s in ('sc', 'scr', 'screening', 'bl', 'baseline'):
        return 0.0
    m = re.match(r'^m(\d+)$', s)
    if m:
        return float(m.group(1))
    return np.nan


# Edge Weight Features
# Clinical + MRI features for edge weights.
# MRI volumes help separate MCI from NC/AD where clinical scores alone cannot.
EDGE_FEATURE_NAMES = [
    'clinical_APOE4_count',
    'clinical_MMSCORE',
    'clinical_FAQTOTAL',
    'clinical_LDELTOTAL',
    'clinical_TRABSCOR',
    'mri_hippocampus_vol_mean',
    'mri_entorhinal_vol_mean',
    'mri_amygdala_vol_mean',
    'mri_inferior_temporal_vol_mean',
    'mri_middle_temporal_vol_mean',

]


EDGE_ABLATION_GROUPS = {
    "all": [
        'clinical_APOE4_count',
        'clinical_MMSCORE',
        'clinical_FAQTOTAL',
        'clinical_LDELTOTAL',
        'clinical_TRABSCOR',
        'mri_hippocampus_vol_mean',
        'mri_entorhinal_vol_mean',
        'mri_amygdala_vol_mean',
        'mri_inferior_temporal_vol_mean',
        'mri_middle_temporal_vol_mean',
    ],
    "cognition": [
        'clinical_MMSCORE',
        'clinical_FAQTOTAL',
        'clinical_LDELTOTAL',
        'clinical_TRABSCOR',
    ],
    "cognition_cdr": [
    'clinical_MMSCORE',
    'clinical_FAQTOTAL',
    'clinical_LDELTOTAL',
    'clinical_TRABSCOR',
    'clinical_CDGLOBAL',
],

    "mri": [
        'mri_hippocampus_vol_mean',
        'mri_entorhinal_vol_mean',
        'mri_amygdala_vol_mean',
        'mri_inferior_temporal_vol_mean',
        'mri_middle_temporal_vol_mean',
    ],
    "apoe_demo": [
        'clinical_APOE4_count',
        'clinical_entry_age',
        'clinical_EDUCAT',
    ],
}


# Features that must NEVER be perturbed in counterfactual explanations.
# Includes genetic markers, immutable demographics, and identifiers.
COUNTERFACTUAL_FIXED_FEATURES = frozenset({
    "clinical_APOE4_count",      # genetic — immutable
    "PHS",                       # polygenic hazard score — genetic, immutable
    "clinical_entry_age",        # age — immutable
    "clinical_EDUCAT",           # education — fixed at baseline
    "subject_id", "merge_key", "time_key", "clinical_session_id",
})


def _get_modifiable_mask(feature_names):
    """Return bool array (F,) — True for features eligible for counterfactual perturbation.

    Modifiable: MRI volumes (mri_*), CSF biomarkers (csf_*), PET (pet_*),
                and clinical cognitive scores in EDGE_ABLATION_GROUPS['cognition'].
    Fixed:      APOE4, age, education, identifiers (COUNTERFACTUAL_FIXED_FEATURES).
    """
    cognition_set = set(EDGE_ABLATION_GROUPS["cognition"])
    mask = []
    for name in feature_names:
        if name in COUNTERFACTUAL_FIXED_FEATURES:
            mask.append(False)
        elif any(name.startswith(p) for p in ("mri_", "csf_", "pet_")):
            mask.append(True)
        elif name in cognition_set:
            mask.append(True)
        else:
            mask.append(False)
    return np.array(mask, dtype=bool)


def get_edge_weight_features(columns):
    
    return [col for col in EDGE_FEATURE_NAMES if col in columns]


# Adjacency Matrix

def compute_similarity_matrix(X_edge, kernel='rbf', rbf_gamma=None):
    """Build full similarity matrix from cognitive features.

    RBF gives sharper contrast than linear (near nodes score high, far ones
    drop off fast), which helps separate MCI from NC/AD. rbf_gamma defaults
    to the median-distance heuristic if not given.
    """
    dists = squareform(pdist(X_edge, metric='euclidean'))

    if kernel == 'rbf':
        # Median heuristic for bandwidth if not provided
        if rbf_gamma is None:
            upper_tri = dists[np.triu_indices_from(dists, k=1)]
            median_dist = np.median(upper_tri)
            rbf_gamma = 1.0 / (2.0 * median_dist ** 2 + 1e-8)
        similarity = np.exp(-rbf_gamma * dists ** 2)
    else:
        max_dist = dists.max()
        if max_dist > 0:
            norm_dists = dists / max_dist
        else:
            norm_dists = np.zeros_like(dists)
        similarity = 1.0 - norm_dists

    np.fill_diagonal(similarity, 0.0)
    return similarity


def apply_temporal_decay(similarity, visit_months, subject_ids, temporal_lambda):
    """Modulate edge weights by temporal proximity for same-subject edges.

    Same-subject pairs: weight *= exp(-lambda * |t_i - t_j|).
    Cross-subject edges are unaffected 
    """
    vm = np.asarray(visit_months, dtype=np.float64)
    delta_t = np.abs(vm[:, None] - vm[None, :])
    temporal_weight = np.exp(-temporal_lambda * delta_t)

    # Same-subject mask via broadcasting
    sids = np.asarray(subject_ids)
    same_subject = (sids[:, None] == sids[None, :])

    # Apply decay only to same-subject edges; cross-subject edges keep weight 1.0
    decay = np.where(same_subject, temporal_weight, 1.0)
    modulated = similarity * decay
    np.fill_diagonal(modulated, 0.0)

    # Diagnostics
    ss_pairs = same_subject & ~np.eye(len(vm), dtype=bool)
    n_ss = int(ss_pairs.sum())
    if n_ss > 0:
        ss_weights = temporal_weight[ss_pairs]
        print(f"[temporal] lambda={temporal_lambda}, same-subject pairs={n_ss}, "
              f"decay: mean={ss_weights.mean():.3f}, min={ss_weights.min():.3f}, max={ss_weights.max():.3f}")

    return modulated


def apply_temporal_decay_causal(similarity, visit_months, subject_ids, temporal_lambda):
    """Causal counterpart to apply_temporal_decay, for the inductive population graph.

    For same-subject pairs, an edge is zeroed if node j's visit comes after
    node i's — a node must not see its own future visit — otherwise it decays
    by exp(-temporal_lambda * (t_i - t_j)). Cross-subject edges stay at
    weight 1.0, same as apply_temporal_decay, which is left unchanged since
    the transductive path still uses it directly.
    """
    vm = np.asarray(visit_months, dtype=np.float64)
    delta_t = vm[:, None] - vm[None, :]  # t_i - t_j
    causal = delta_t >= 0.0  # j is at or before i
    temporal_weight = np.exp(-temporal_lambda * delta_t)

    sids = np.asarray(subject_ids)
    same_subject = (sids[:, None] == sids[None, :])

    # Same-subject + causal: decay by exp(-lambda*(t_i-t_j)).
    # Same-subject + anticipating (j after i): weight 0 (dropped).
    # Cross-subject: weight 1.0, unaffected — identical to apply_temporal_decay.
    decay = np.where(same_subject, np.where(causal, temporal_weight, 0.0), 1.0)
    modulated = similarity * decay
    np.fill_diagonal(modulated, 0.0)

    # Diagnostics
    eye = np.eye(len(vm), dtype=bool)
    ss_causal_pairs = same_subject & causal & ~eye
    ss_dropped_pairs = same_subject & ~causal
    n_ss = int(ss_causal_pairs.sum())
    n_dropped = int(ss_dropped_pairs.sum())
    if n_ss > 0:
        ss_weights = temporal_weight[ss_causal_pairs]
        print(f"[temporal-causal] lambda={temporal_lambda}, causal same-subject pairs kept={n_ss}, "
              f"anticipating same-subject pairs dropped={n_dropped}, "
              f"decay: mean={ss_weights.mean():.3f}, min={ss_weights.min():.3f}, max={ss_weights.max():.3f}")

    return modulated, {
        "same_subject_pairs_kept_causal": n_ss,
        "same_subject_pairs_dropped_anticipating": n_dropped,
    }


def topk_sparsify(similarity, k):
    """Sparsify similarity matrix: each node keeps only its top-K most similar neighbors.
    The result is made symmetric (if i is in j's top-K OR j is in i's top-K, keep the edge).
    """
    N = similarity.shape[0]
    actual_k = min(k, N - 1)
    sparse = np.zeros_like(similarity)

    for i in range(N):
        row = similarity[i].copy()
        row[i] = -1  # exclude self
        topk_idx = np.argpartition(row, -actual_k)[-actual_k:]
        sparse[i, topk_idx] = similarity[i, topk_idx]

    # Make symmetric: keep edge if either direction has it
    sparse = np.maximum(sparse, sparse.T)
    np.fill_diagonal(sparse, 0.0)
    return sparse


def topk_sparsify_subject_aware(similarity, k, subject_ids, max_same_subject=1):
    """Sparsify similarity matrix with per-subject neighbor constraints.

    For each node, selects top-K neighbors but limits neighbors from any
    single subject to at most `max_same_subject` (0 blocks same-subject
    edges entirely). Freed slots are backfilled with the next most similar
    cross-subject neighbors.
    """
    N = similarity.shape[0]
    actual_k = min(k, N - 1)
    sparse = np.zeros_like(similarity)
    same_kept = 0
    same_blocked = 0

    for i in range(N):
        row = similarity[i].copy()
        row[i] = -1  # exclude self
        sorted_idx = np.argsort(row)[::-1]  # descending similarity

        subject_count = {}
        accepted = 0
        for j in sorted_idx:
            if accepted >= actual_k:
                break
            if row[j] <= 0:
                break
            is_same_subject = (subject_ids[j] == subject_ids[i])
            if is_same_subject:
                cnt = subject_count.get(subject_ids[j], 0)
                if cnt >= max_same_subject:
                    same_blocked += 1
                    continue
                same_kept += 1
            subject_count[subject_ids[j]] = subject_count.get(subject_ids[j], 0) + 1
            sparse[i, j] = similarity[i, j]
            accepted += 1

    # Make symmetric: keep edge if either direction has it
    sparse = np.maximum(sparse, sparse.T)
    np.fill_diagonal(sparse, 0.0)
    return sparse, {"same_subject_kept": same_kept, "same_subject_blocked": same_blocked}


def print_subject_neighbor_stats(A, subject_ids, prefix="[subject]"):
   
    N = A.shape[0]
    n_edges = int(np.count_nonzero(A))

    same_subject_edges = 0
    per_node_same = np.zeros(N, dtype=int)
    per_node_total = np.zeros(N, dtype=int)

    for i in range(N):
        neighbors = np.where(A[i] > 0)[0]
        per_node_total[i] = len(neighbors)
        same = np.sum(subject_ids[neighbors] == subject_ids[i])
        per_node_same[i] = same
        same_subject_edges += same

    dominated = int(np.sum((per_node_same > per_node_total / 2) & (per_node_total > 0)))

    print(f"  {prefix} Edges (directed): {n_edges}")
    print(f"  {prefix} Same-subject edges: {same_subject_edges} "
          f"({100 * same_subject_edges / max(n_edges, 1):.1f}%)")
    print(f"  {prefix} Per-node same-subj neighbors: "
          f"mean={per_node_same.mean():.1f}, max={per_node_same.max()}")
    print(f"  {prefix} Dominated nodes (>50% same-subj): {dominated}/{N}")


def build_temporal_adjacency(visit_months, subject_ids, temporal_lambda=0.05, topk=None):
    """Build adjacency with only same-subject edges, weighted by temporal proximity."""
    N = len(visit_months)
    vm = np.asarray(visit_months, dtype=np.float64)
    sids = np.asarray(subject_ids)

    same_subject = (sids[:, None] == sids[None, :])
    delta_t = np.abs(vm[:, None] - vm[None, :])
    temporal_weight = np.exp(-temporal_lambda * delta_t)

    A_temporal = np.where(same_subject, temporal_weight, 0.0)
    np.fill_diagonal(A_temporal, 0.0)

    if topk is not None:
        for i in range(N):
            row = A_temporal[i].copy()
            nonzero = np.nonzero(row)[0]
            if len(nonzero) > topk:
                vals = row[nonzero]
                keep_idx = nonzero[np.argpartition(vals, -topk)[-topk:]]
                mask_arr = np.zeros(N, dtype=bool)
                mask_arr[keep_idx] = True
                A_temporal[i, ~mask_arr] = 0.0
        A_temporal = np.maximum(A_temporal, A_temporal.T)
        np.fill_diagonal(A_temporal, 0.0)

    A_hat = A_temporal + np.eye(N)
    A_temporal_norm = symmetric_normalize(A_hat)
    return A_temporal, A_temporal_norm


def drop_edge(A_norm_tensor, drop_rate=0.1):
    """DropEdge: randomly zero out edges during training to reduce over-smoothing."""
    if drop_rate <= 0:
        return A_norm_tensor
    mask = torch.rand_like(A_norm_tensor) > drop_rate
    # Keep diagonal (self-loops) intact
    diag = torch.diag(torch.diag(A_norm_tensor))
    off_diag = A_norm_tensor - diag
    return diag + off_diag * mask.float()


def symmetric_normalize(A_hat):
    """Compute D^{-1/2} A_hat D^{-1/2} (Kipf & Welling normalization)."""
    D = np.array(A_hat.sum(axis=1)).flatten()
    D_inv_sqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
    D_inv_sqrt_mat = np.diag(D_inv_sqrt)
    A_norm = D_inv_sqrt_mat @ A_hat @ D_inv_sqrt_mat
    return A_norm


# Inductive sensitivity analysis: adjacency helpers
# Everything below builds strictly-directed graphs where held-out (val/test)
# nodes may receive edges from training nodes but never emit them, so
# training-node neighborhoods are unaffected by which held-out nodes are
# present. Used only by run_single_fold_inductive(); the transductive path
# above (compute_similarity_matrix, topk_sparsify*, symmetric_normalize,
# build_temporal_adjacency) is untouched.

def row_normalize(A_hat):
    """Compute D^{-1} A_hat (row-stochastic normalization). Unused — kept as
    an alternative to symmetric_normalize for the inductive path."""
    D = np.array(A_hat.sum(axis=1)).flatten()
    D_inv = np.where(D > 0, 1.0 / D, 0.0)
    A_norm = A_hat * D_inv[:, None]
    return A_norm


def compute_similarity_cross(X_new, X_train, kernel='rbf', gamma=None, max_dist=None):
    """Similarity of new (held-out) nodes against training nodes only.

    Rectangular (n_new, n_tr) counterpart to compute_similarity_matrix. Kernel
    parameters (gamma for rbf, max_dist for linear) are frozen from the
    training block, so held-out nodes can't influence the similarity scale.
    Linear-kernel distances are clipped to [0, 1] rather than renormalized,
    since renormalizing would reintroduce that dependence.
    """
    dists = cdist(X_new, X_train, metric='euclidean')

    if kernel == 'rbf':
        similarity = np.exp(-gamma * dists ** 2)
    else:
        if max_dist > 0:
            norm_dists = dists / max_dist
        else:
            norm_dists = np.zeros_like(dists)
        similarity = np.clip(1.0 - norm_dists, 0.0, 1.0)

    return similarity


def apply_temporal_decay_cross(similarity_cross, visit_months_new, subject_ids_new,
                                visit_months_train, subject_ids_train, temporal_lambda):
    """Cross-block counterpart to apply_temporal_decay.

    Modulates held-out-to-training similarity by temporal proximity for
    same-subject pairs (weight *= exp(-lambda * |t_i - t_j|)); cross-subject
    entries are left at weight 1.0, same as apply_temporal_decay. 
    """
    vm_new = np.asarray(visit_months_new, dtype=np.float64)
    vm_train = np.asarray(visit_months_train, dtype=np.float64)
    delta_t = np.abs(vm_new[:, None] - vm_train[None, :])
    temporal_weight = np.exp(-temporal_lambda * delta_t)

    sids_new = np.asarray(subject_ids_new)
    sids_train = np.asarray(subject_ids_train)
    same_subject = (sids_new[:, None] == sids_train[None, :])

    decay = np.where(same_subject, temporal_weight, 1.0)
    return similarity_cross * decay


def topk_sparsify_directed(similarity, k):
    """Directed counterpart to topk_sparsify: each node keeps its top-K most
    similar neighbors, but the result is NOT symmetrized. Used for A_train
    in the inductive path so training-node neighborhoods stay fixed once
    held-out nodes are introduced.
    """
    N = similarity.shape[0]
    actual_k = min(k, N - 1)
    sparse = np.zeros_like(similarity)

    for i in range(N):
        row = similarity[i].copy()
        row[i] = -1  # exclude self
        topk_idx = np.argpartition(row, -actual_k)[-actual_k:]
        sparse[i, topk_idx] = similarity[i, topk_idx]

    np.fill_diagonal(sparse, 0.0)
    return sparse


def topk_sparsify_subject_aware_directed(similarity, k, subject_ids, max_same_subject=1):
    """Directed counterpart to topk_sparsify_subject_aware: same per-node
    subject-capped top-K selection, but the result is NOT symmetrized.
    Used for A_train in the inductive path.
    """
    N = similarity.shape[0]
    actual_k = min(k, N - 1)
    sparse = np.zeros_like(similarity)
    same_kept = 0
    same_blocked = 0

    for i in range(N):
        row = similarity[i].copy()
        row[i] = -1  # exclude self
        sorted_idx = np.argsort(row)[::-1]  # descending similarity

        subject_count = {}
        accepted = 0
        for j in sorted_idx:
            if accepted >= actual_k:
                break
            if row[j] <= 0:
                break
            is_same_subject = (subject_ids[j] == subject_ids[i])
            if is_same_subject:
                cnt = subject_count.get(subject_ids[j], 0)
                if cnt >= max_same_subject:
                    same_blocked += 1
                    continue
                same_kept += 1
            subject_count[subject_ids[j]] = subject_count.get(subject_ids[j], 0) + 1
            sparse[i, j] = similarity[i, j]
            accepted += 1

    np.fill_diagonal(sparse, 0.0)
    return sparse, {"same_subject_kept": same_kept, "same_subject_blocked": same_blocked}


def topk_sparsify_cross(similarity_cross, k):
    """Directed top-K selection of held-out nodes' edges into training nodes
    only. Rectangular (n_new, n_tr) counterpart to topk_sparsify; naturally
    directed since the shape is asymmetric (no reverse edges are possible).
    """
    n_new, n_tr = similarity_cross.shape
    actual_k = min(k, n_tr)
    sparse = np.zeros_like(similarity_cross)

    for i in range(n_new):
        row = similarity_cross[i]
        topk_idx = np.argpartition(row, -actual_k)[-actual_k:] if actual_k > 0 else np.array([], dtype=int)
        sparse[i, topk_idx] = row[topk_idx]

    return sparse


def topk_sparsify_cross_subject_aware(similarity_cross, k, subject_ids_new, subject_ids_train,
                                       max_same_subject=1):
    """Directed, subject-capped top-K selection of held-out nodes' edges into
    training nodes only. Rectangular counterpart to topk_sparsify_subject_aware,
    reusing the same per-row subject-cap selection loop, restricted to
    training-node columns.
    """
    n_new, n_tr = similarity_cross.shape
    actual_k = min(k, n_tr)
    sparse = np.zeros_like(similarity_cross)
    same_kept = 0
    same_blocked = 0

    for i in range(n_new):
        row = similarity_cross[i]
        sorted_idx = np.argsort(row)[::-1]  # descending similarity

        subject_count = {}
        accepted = 0
        for j in sorted_idx:
            if accepted >= actual_k:
                break
            if row[j] <= 0:
                break
            is_same_subject = (subject_ids_train[j] == subject_ids_new[i])
            if is_same_subject:
                cnt = subject_count.get(subject_ids_train[j], 0)
                if cnt >= max_same_subject:
                    same_blocked += 1
                    continue
                same_kept += 1
            subject_count[subject_ids_train[j]] = subject_count.get(subject_ids_train[j], 0) + 1
            sparse[i, j] = row[j]
            accepted += 1

    return sparse, {"same_subject_kept": same_kept, "same_subject_blocked": same_blocked}


def build_temporal_adjacency_causal(visit_months, subject_ids, temporal_lambda=0.05, topk=None):
    """Causal same-subject temporal adjacency: node i aggregates only from
    same-subject visits at or before it (|t_i - t_j| -> one-directional,
    never symmetrized). Returns the raw matrix; self-loops and normalization
    are applied later by the inductive orchestrator.
    """
    N = len(visit_months)
    vm = np.asarray(visit_months, dtype=np.float64)
    sids = np.asarray(subject_ids)

    same_subject = (sids[:, None] == sids[None, :])
    delta_t = vm[:, None] - vm[None, :]  # t_i - t_j
    causal = delta_t >= 0.0  # j is at or before i
    temporal_weight = np.exp(-temporal_lambda * delta_t)

    A_temporal_causal = np.where(same_subject & causal, temporal_weight, 0.0)
    np.fill_diagonal(A_temporal_causal, 0.0)

    if topk is not None:
        for i in range(N):
            row = A_temporal_causal[i].copy()
            nonzero = np.nonzero(row)[0]
            if len(nonzero) > topk:
                vals = row[nonzero]
                keep_idx = nonzero[np.argpartition(vals, -topk)[-topk:]]
                mask_arr = np.zeros(N, dtype=bool)
                mask_arr[keep_idx] = True
                A_temporal_causal[i, ~mask_arr] = 0.0

    return A_temporal_causal


# Edge Analysis

def explain_edge(i, j, X_edge_raw, X_edge_std, edge_feat_cols, kernel='rbf', rbf_gamma=None):
    """Explain why edge (i, j) exists by decomposing per-feature contributions."""
    raw_i, raw_j = X_edge_raw[i], X_edge_raw[j]
    std_i, std_j = X_edge_std[i], X_edge_std[j]

    result = {}
    for k, col in enumerate(edge_feat_cols):
        diff_std = abs(std_i[k] - std_j[k])

        if kernel == 'rbf' and rbf_gamma is not None:
            contrib = float(np.exp(-rbf_gamma * diff_std ** 2))
        else:
            contrib = None  # filled in by caller if needed for linear

        direction = "i > j" if raw_i[k] > raw_j[k] else ("i < j" if raw_i[k] < raw_j[k] else "equal")

        result[col] = {
            "patient_i_val": float(raw_i[k]),
            "patient_j_val": float(raw_j[k]),
            "abs_diff_std": float(diff_std),
            "direction": direction,
            "contribution_to_similarity": contrib,
        }

    # For linear kernel, compute contributions post-hoc using max diff
    if kernel != 'rbf' or rbf_gamma is None:
        max_diff = max((v["abs_diff_std"] for v in result.values()), default=1.0)
        if max_diff == 0:
            max_diff = 1.0
        for v in result.values():
            v["contribution_to_similarity"] = float(1.0 - v["abs_diff_std"] / max_diff)

    return result


def analyze_graph_structure(A, X_edge_raw, X_edge_std, y_all, edge_feat_cols,
                            classes, c2i, kernel='rbf', rbf_gamma=None):
    """Analyze graph edges by class pair to understand class separation.

    For each of the 6 class pairs (NC-NC, NC-MCI, ..., AD-AD):
      - Count edges and mean similarity
      - Mean per-feature absolute diff (standardized) across edges
      - Top features driving similarity vs differentiating
    """
    i2c = {v: k for k, v in c2i.items()}
    num_feats = len(edge_feat_cols)
    analysis = {}

    print(f"\n{'=' * 70}")
    print(f"GRAPH STRUCTURE ANALYSIS — Edge Feature Contributions by Class Pair")
   

    for ca_idx in range(len(classes)):
        for cb_idx in range(ca_idx, len(classes)):
            ca_name, cb_name = i2c[ca_idx], i2c[cb_idx]
            pair_key = f"{ca_name}-{cb_name}"

            # Find all edges for this class pair
            nodes_a = np.where(y_all == ca_idx)[0]
            nodes_b = np.where(y_all == cb_idx)[0]

            edge_sims = []
            feat_diffs = []  # (num_edges, num_feats)

            for ni in nodes_a:
                targets = nodes_b if ca_idx != cb_idx else nodes_b[nodes_b > ni]
                for nj in targets:
                    w = A[ni, nj]
                    if w > 0:
                        edge_sims.append(w)
                        feat_diffs.append(np.abs(X_edge_std[ni] - X_edge_std[nj]))

            n_edges = len(edge_sims)
            if n_edges == 0:
                analysis[pair_key] = {"n_edges": 0, "mean_sim": 0.0, "features": {}}
                print(f"\n  {pair_key:>8s}: 0 edges")
                continue

            mean_sim = float(np.mean(edge_sims))
            median_sim = float(np.median(edge_sims))
            feat_diffs_arr = np.array(feat_diffs)  # (n_edges, F)
            mean_diffs = feat_diffs_arr.mean(axis=0)  # (F,)

            # Rank features: lowest mean diff = drove similarity, highest = differentiated
            feat_ranking = sorted(range(num_feats), key=lambda k: mean_diffs[k])
            driving_feats = [(edge_feat_cols[k], float(mean_diffs[k])) for k in feat_ranking[:3]]
            differing_feats = [(edge_feat_cols[k], float(mean_diffs[k])) for k in feat_ranking[-3:]]

            analysis[pair_key] = {
                "n_edges": n_edges,
                "mean_sim": mean_sim,
                "median_sim": median_sim,
                "mean_feat_diffs": {edge_feat_cols[k]: float(mean_diffs[k]) for k in range(num_feats)},
                "driving_features": driving_feats,
                "differentiating_features": differing_feats,
            }

            print(f"\n  {pair_key:>8s}: {n_edges:5d} edges | mean_sim={mean_sim:.4f} median_sim={median_sim:.4f}")
            print(f"    Most similar (drove connection):")
            for fname, d in driving_feats:
                print(f"      {fname:<40s}  mean_std_diff={d:.4f}")
            print(f"    Most different (weakened connection):")
            for fname, d in differing_feats:
                print(f"      {fname:<40s}  mean_std_diff={d:.4f}")

    return analysis


# Focal Loss

class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        targets_hard = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1), 1)

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        pt = (probs * targets_hard).sum(dim=1)
        focal_term = (1 - pt).pow(self.gamma)

        if self.label_smoothing > 0:
            targets_smooth = targets_hard * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        else:
            targets_smooth = targets_hard

        log_pt = (log_probs * targets_smooth).sum(dim=1)

        if self.alpha is not None:
            at = self.alpha[targets]
            loss = -at * focal_term * log_pt
        else:
            loss = -focal_term * log_pt

        return loss.mean() if self.reduction == 'mean' else loss.sum()


# Spectral GCN Model

class GCNLayer(nn.Module):
    """Single GCN layer: output = A_norm @ (X @ W + b)"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, H, A_norm):
        support = self.linear(H)
        return torch.mm(A_norm, support)


class SpectralGCN(nn.Module):
    """Hybrid GCN + MLP model for AD classification.

    Runs up to three branches in parallel:
      1. GCN branch — spectral convolutions with JK aggregation over the
         cross-patient phenotypic similarity graph.
      2. Temporal GCN branch — same
         architecture, but over a same-subject-only graph, so it picks up
         longitudinal disease progression instead of cross-patient similarity.
      3. MLP branch — raw node features, no graph smoothing.

    """
    def __init__(self, in_dim, hidden_dim, num_classes, num_gcn_layers=3,
                 dropout=0.15, use_bn=True, drop_edge_rate=0.1,
                 feat_mask_rate=0.05,
                 use_temporal_branch=False, temporal_hidden_dim=128,
                 temporal_num_layers=2, use_mlp_branch=True,
                 use_jk=True):
        super().__init__()
        self.gcn_convs = nn.ModuleList()
        self.gcn_bns = nn.ModuleList() if use_bn else None
        self.use_bn = use_bn
        self.dropout = nn.Dropout(dropout)
        self.num_gcn_layers = num_gcn_layers
        self.drop_edge_rate = drop_edge_rate
        self.feat_mask_rate = feat_mask_rate
        self.use_temporal_branch = use_temporal_branch
        self.use_mlp_branch = use_mlp_branch
        self.use_jk = use_jk

        # Residual projection: input dim -> hidden dim
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        # GCN encoder layers (all output hidden_dim)
        self.gcn_convs.append(GCNLayer(in_dim, hidden_dim))
        if use_bn:
            self.gcn_bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_gcn_layers - 1):
            self.gcn_convs.append(GCNLayer(hidden_dim, hidden_dim))
            if use_bn:
                self.gcn_bns.append(nn.BatchNorm1d(hidden_dim))

        # JK aggregation: concat all layer outputs -> hidden_dim * num_layers (or hidden_dim when disabled)
        jk_dim = hidden_dim * num_gcn_layers if use_jk else hidden_dim

        # Temporal GCN branch: same-subject longitudinal graph
        self.temporal_gcn_convs = None
        self.temporal_gcn_bns = None
        self.temporal_input_proj = None
        temporal_jk_dim = 0

        if use_temporal_branch:
            self.temporal_gcn_convs = nn.ModuleList()
            self.temporal_gcn_bns = nn.ModuleList() if use_bn else None
            self.temporal_input_proj = (nn.Linear(in_dim, temporal_hidden_dim)
                                        if in_dim != temporal_hidden_dim else nn.Identity())

            self.temporal_gcn_convs.append(GCNLayer(in_dim, temporal_hidden_dim))
            if use_bn:
                self.temporal_gcn_bns.append(nn.BatchNorm1d(temporal_hidden_dim))

            for _ in range(temporal_num_layers - 1):
                self.temporal_gcn_convs.append(GCNLayer(temporal_hidden_dim, temporal_hidden_dim))
                if use_bn:
                    self.temporal_gcn_bns.append(nn.BatchNorm1d(temporal_hidden_dim))

            temporal_jk_dim = temporal_hidden_dim * temporal_num_layers if use_jk else temporal_hidden_dim

        # MLP branch: processes raw node features WITHOUT graph smoothing
        if use_mlp_branch:
            mlp_branch_dim = hidden_dim // 2
            self.mlp_branch = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, mlp_branch_dim),
                nn.ReLU(),
                nn.BatchNorm1d(mlp_branch_dim) if use_bn else nn.Identity(),
                nn.Dropout(dropout * 0.5),
            )
        else:
            mlp_branch_dim = 0
            self.mlp_branch = None

        # Final classifier on concatenated [GCN_jk | temporal_jk | MLP_branch]
        combined_dim = jk_dim + temporal_jk_dim + mlp_branch_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, X, A_norm, X_edge=None, edge_index=None, A_temporal=None):
        # Node feature masking augmentation during training
        if self.training and self.feat_mask_rate > 0:
            mask = torch.rand_like(X) > self.feat_mask_rate
            X_masked = X * mask.float()
        else:
            X_masked = X

        # MLP branch: raw node features (no graph smoothing) 
        H_mlp = self.mlp_branch(X_masked) if self.use_mlp_branch else None

        #  GCN branch (similarity graph) 
        if self.training and self.drop_edge_rate > 0:
            A_used = drop_edge(A_norm, self.drop_edge_rate)
        else:
            A_used = A_norm

        H = X_masked
        layer_outputs = []

        for i, conv in enumerate(self.gcn_convs):
            H_res = self.input_proj(H) if i == 0 else H
            H = conv(H, A_used)
            if self.use_bn:
                H = self.gcn_bns[i](H)
            H = F.relu(H + H_res)
            H = self.dropout(H)
            layer_outputs.append(H)

        # JK aggregation: concatenate all layer outputs (or use only last layer when disabled)
        H_jk = torch.cat(layer_outputs, dim=1) if self.use_jk else layer_outputs[-1]

        # Temporal GCN branch (same-subject longitudinal graph) 
        if self.use_temporal_branch and A_temporal is not None:
            H_t = X_masked
            temporal_outputs = []

            for i, conv in enumerate(self.temporal_gcn_convs):
                H_t_res = self.temporal_input_proj(H_t) if i == 0 else H_t
                H_t = conv(H_t, A_temporal)
                if self.use_bn:
                    H_t = self.temporal_gcn_bns[i](H_t)
                H_t = F.relu(H_t + H_t_res)
                H_t = self.dropout(H_t)
                temporal_outputs.append(H_t)

            H_temporal_jk = torch.cat(temporal_outputs, dim=1) if self.use_jk else temporal_outputs[-1]
            parts = [H_jk, H_temporal_jk] + ([H_mlp] if H_mlp is not None else [])
            H_combined = torch.cat(parts, dim=1)
        else:
            parts = [H_jk] + ([H_mlp] if H_mlp is not None else [])
            H_combined = torch.cat(parts, dim=1)

        primary_out = self.classifier(H_combined)
        return primary_out, None


# Data Helpers

def build_preprocessor(Xdf, ignore_cols=("subject_id", "merge_key", "time_key", "clinical_session_id")):
   
    num_cols = [c for c in Xdf.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(Xdf[c])]
    cat_cols = [c for c in Xdf.columns if c not in ignore_cols and not pd.api.types.is_numeric_dtype(Xdf[c])]
    num_pipe = SkPipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = SkPipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")
    return pre, num_cols, cat_cols


def subject_stratified_split(df, target_col, group_col, val_frac=0.1, test_frac=0.1, seed=42):
    """Subject-level stratified split into train / val / test (80/10/10)."""
    subjects = df[group_col].unique()
    sub_labels = {s: df[df[group_col] == s][target_col].mode().iat[0] for s in subjects}

    subs = np.array(list(sub_labels.keys()))
    labels = np.array([sub_labels[s] for s in subs])

    subs_trainval, subs_test = train_test_split(
        subs, test_size=test_frac, stratify=labels, random_state=seed
    )
    labels_trainval = np.array([sub_labels[s] for s in subs_trainval])

    val_size_adj = val_frac / (1.0 - test_frac)
    subs_train, subs_val = train_test_split(
        subs_trainval, test_size=val_size_adj, stratify=labels_trainval, random_state=seed
    )

    train_mask = df[group_col].isin(subs_train).values
    val_mask = df[group_col].isin(subs_val).values
    test_mask = df[group_col].isin(subs_test).values
    return train_mask, val_mask, test_mask


# Training
class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing to min_lr."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / max(self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * alpha
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


def train_model(model, X, A_norm, y, train_mask, val_mask, optimizer, criterion,
                scheduler=None, max_epochs=400, patience=40, device=None, verbose=True,
                monitor="val_bal", X_edge=None, edge_index=None,
                A_temporal=None):
    """Train with early stopping on validation balanced accuracy."""
    X = X.to(device)
    A_norm = A_norm.to(device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    tr = torch.as_tensor(train_mask, dtype=torch.bool, device=device)
    va = torch.as_tensor(val_mask, dtype=torch.bool, device=device)

    # Move edge data to device if using learned edges
    if X_edge is not None:
        X_edge = X_edge.to(device)
    if edge_index is not None:
        edge_index = edge_index.to(device)
    if A_temporal is not None:
        A_temporal = A_temporal.to(device)

    if monitor == "val_bal":
        best = {"epoch": -1, "metric": -float("inf"), "state": None}
    else:
        best = {"epoch": -1, "metric": float("inf"), "state": None}
    wait = 0
    t0 = time.time()
    loss_history = {"train": [], "val": []}

    for ep in range(1, max_epochs + 1):
        if scheduler is not None:
            scheduler.step(ep)

        model.train()
        optimizer.zero_grad()
        out, _ = model(X, A_norm, X_edge=X_edge, edge_index=edge_index, A_temporal=A_temporal)
        loss = criterion(out[tr], y[tr])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_eval, _ = model(X, A_norm, X_edge=X_edge, edge_index=edge_index, A_temporal=A_temporal)
            train_loss = criterion(out_eval[tr], y[tr]).item()
            val_loss = criterion(out_eval[va], y[va]).item()
            train_pred = out_eval[tr].argmax(dim=1).cpu().numpy()
            val_pred = out_eval[va].argmax(dim=1).cpu().numpy()
            train_bal = balanced_accuracy_score(y[tr].cpu().numpy(), train_pred)
            val_bal = balanced_accuracy_score(y[va].cpu().numpy(), val_pred)

        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)

        metric_now = val_bal if monitor == "val_bal" else val_loss
        if monitor == "val_bal":
            better = metric_now > best["metric"]
        else:
            better = metric_now < best["metric"]

        if better:
            best["metric"] = metric_now
            best["epoch"] = ep
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (ep == 1 or ep % 10 == 0):
            dt = time.time() - t0
            t0 = time.time()
            mark = "*" if wait == 0 else " "
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  [epoch {ep:04d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"train_bal={train_bal:.4f}  val_bal={val_bal:.4f}  lr={cur_lr:.6f}  {mark}  ({dt:.1f}s)")

        if wait >= patience:
            if verbose:
                print(f"  [early-stop] epoch {ep}. Best {monitor} at epoch {best['epoch']}.")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    best["loss_history"] = loss_history
    return model, best


def train_model_inductive(model, X_tr, A_train_norm, y_tr, X_train_val, A_train_val_norm,
                          y_val_block, val_submask, optimizer, criterion,
                          scheduler=None, max_epochs=400, patience=40, device=None, verbose=True,
                          monitor="val_bal",
                          A_temporal_train=None, A_temporal_train_val=None):
    """Inductive counterpart to train_model: trains on G_train only, early-stops
    on a validation forward pass through the extended G_train_val graph.

    Unlike train_model, the train and validation forward passes use different
    (X, A_norm) pairs since the graphs have different shapes (G_train has
    n_tr nodes, G_train_val has n_tr + n_val) — held-out val nodes are never
    part of the training forward/backward pass. No optimizer step or
    BatchNorm running-stat update happens on the validation forward pass:
    it runs under model.eval() (BatchNorm uses running stats, dropout /
    DropEdge / feature-masking are disabled) and torch.no_grad().
    """
    X_tr = X_tr.to(device)
    A_train_norm = A_train_norm.to(device)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)

    X_train_val = X_train_val.to(device)
    A_train_val_norm = A_train_val_norm.to(device)
    y_val_block_t = torch.as_tensor(y_val_block, dtype=torch.long, device=device)
    val_sub = torch.as_tensor(val_submask, dtype=torch.bool, device=device)

    if A_temporal_train is not None:
        A_temporal_train = A_temporal_train.to(device)
    if A_temporal_train_val is not None:
        A_temporal_train_val = A_temporal_train_val.to(device)

    if monitor == "val_bal":
        best = {"epoch": -1, "metric": -float("inf"), "state": None}
    else:
        best = {"epoch": -1, "metric": float("inf"), "state": None}
    wait = 0
    t0 = time.time()
    loss_history = {"train": [], "val": []}

    for ep in range(1, max_epochs + 1):
        if scheduler is not None:
            scheduler.step(ep)

        model.train()
        optimizer.zero_grad()
        out, _ = model(X_tr, A_train_norm, A_temporal=A_temporal_train)
        loss = criterion(out, y_tr_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_train_eval, _ = model(X_tr, A_train_norm, A_temporal=A_temporal_train)
            train_loss = criterion(out_train_eval, y_tr_t).item()
            train_pred = out_train_eval.argmax(dim=1).cpu().numpy()
            train_bal = balanced_accuracy_score(y_tr_t.cpu().numpy(), train_pred)

            # Validation forward on the extended (train+val) graph, for early stopping.
            assert not model.training
            out_val_eval, _ = model(X_train_val, A_train_val_norm, A_temporal=A_temporal_train_val)
            val_loss = criterion(out_val_eval[val_sub], y_val_block_t[val_sub]).item()
            val_pred = out_val_eval[val_sub].argmax(dim=1).cpu().numpy()
            val_bal = balanced_accuracy_score(y_val_block_t[val_sub].cpu().numpy(), val_pred)

        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)

        metric_now = val_bal if monitor == "val_bal" else val_loss
        if monitor == "val_bal":
            better = metric_now > best["metric"]
        else:
            better = metric_now < best["metric"]

        if better:
            best["metric"] = metric_now
            best["epoch"] = ep
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (ep == 1 or ep % 10 == 0):
            dt = time.time() - t0
            t0 = time.time()
            mark = "*" if wait == 0 else " "
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  [epoch {ep:04d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"train_bal={train_bal:.4f}  val_bal={val_bal:.4f}  lr={cur_lr:.6f}  {mark}  ({dt:.1f}s)")

        if wait >= patience:
            if verbose:
                print(f"  [early-stop] epoch {ep}. Best {monitor} at epoch {best['epoch']}.")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    best["loss_history"] = loss_history
    return model, best


def plot_loss_curves(all_best_info, out_dir, title_prefix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_seeds = len(all_best_info)
    fig, axes = plt.subplots(1, n_seeds, figsize=(5 * n_seeds, 4), squeeze=False)
    for i, info in enumerate(all_best_info):
        ax = axes[0, i]
        hist = info.get("loss_history", {})
        train_l = hist.get("train", [])
        val_l = hist.get("val", [])
        if not train_l:
            continue
        epochs = range(1, len(train_l) + 1)
        ax.plot(epochs, train_l, label="Train Loss", linewidth=1.2)
        ax.plot(epochs, val_l, label="Val Loss", linewidth=1.2)
        best_ep = info.get("epoch", None)
        if best_ep is not None and best_ep <= len(val_l):
            ax.axvline(x=best_ep, color="gray", linestyle="--", alpha=0.6, label=f"Best epoch ({best_ep})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        seed_label = info.get("seed", i + 1)
        ax.set_title(f"{title_prefix}Seed {seed_label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = Path(out_dir) / f"{title_prefix.replace(' ', '_').rstrip('_')}loss_curves.png" if title_prefix else Path(out_dir) / "loss_curves.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [saved] Loss curves -> {save_path}")


CLASS_DISPLAY = {"0": "NC", "0.5": "MCI", "1+": "AD"}
CLASS_COLORS = {"0": "#2196F3", "0.5": "#FF9800", "1+": "#F44336"}


def plot_roc_curves(probs, y_true, classes, out_dir):
    """One-vs-rest ROC curves for each class, plus a macro-average curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    all_fpr = np.linspace(0, 1, 200)
    mean_tprs = []

    for c_idx, cls in enumerate(classes):
        y_bin = (y_true == c_idx).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c_idx])
        roc_auc = auc(fpr, tpr)
        label = CLASS_DISPLAY.get(cls, cls)
        color = CLASS_COLORS.get(cls, None)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})", linewidth=2, color=color)
        mean_tprs.append(np.interp(all_fpr, fpr, tpr))
        mean_tprs[-1][0] = 0.0

    # Macro-average ROC
    mean_tpr = np.mean(mean_tprs, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, label=f"Macro-avg (AUC = {macro_auc:.3f})",
            linewidth=2, linestyle="--", color="black")

    ax.plot([0, 1], [0, 1], "k:", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = Path(out_dir) / "roc_curves.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  [saved] ROC curves -> {save_path}")




def _extract_embeddings(model, X, A_norm, device, X_edge=None, edge_index=None, A_temporal=None):
    """Forward pass up to H_combined (pre-classifier), returns (N, D) numpy array."""
    model.eval()
    with torch.no_grad():
        X_d = X.to(device)
        A_d = A_norm.to(device)
        X_e = X_edge.to(device) if X_edge is not None else None
        ei = edge_index.to(device) if edge_index is not None else None
        A_t = A_temporal.to(device) if A_temporal is not None else None

        # MLP branch
        H_mlp = model.mlp_branch(X_d) if model.use_mlp_branch else None

        # GCN branch
        A_used = A_d

        H = X_d
        layer_outputs = []
        for i, conv in enumerate(model.gcn_convs):
            H_res = model.input_proj(H) if i == 0 else H
            H = conv(H, A_used)
            if model.use_bn:
                H = model.gcn_bns[i](H)
            H = F.relu(H + H_res)
            layer_outputs.append(H)
        H_jk = torch.cat(layer_outputs, dim=1) if model.use_jk else layer_outputs[-1]

        # Temporal branch
        if model.use_temporal_branch and A_t is not None:
            H_t = X_d
            temporal_outputs = []
            for i, conv in enumerate(model.temporal_gcn_convs):
                H_t_res = model.temporal_input_proj(H_t) if i == 0 else H_t
                H_t = conv(H_t, A_t)
                if model.use_bn:
                    H_t = model.temporal_gcn_bns[i](H_t)
                H_t = F.relu(H_t + H_t_res)
                temporal_outputs.append(H_t)
            H_temporal_jk = torch.cat(temporal_outputs, dim=1) if model.use_jk else temporal_outputs[-1]
            parts = [H_jk, H_temporal_jk] + ([H_mlp] if H_mlp is not None else [])
            H_combined = torch.cat(parts, dim=1)
        else:
            parts = [H_jk] + ([H_mlp] if H_mlp is not None else [])
            H_combined = torch.cat(parts, dim=1)

    return H_combined.cpu().numpy()


def plot_umap_embeddings(models, X, A_norm, y, mask, classes, c2i, device, out_dir,
                         X_edge=None, edge_index=None, A_temporal=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import umap
    except ImportError:
        print("  [skip] umap-learn not installed — skipping UMAP plot")
        return

    # Average embeddings across ensemble models
    all_embs = []
    for m in models:
        emb = _extract_embeddings(m, X, A_norm, device, X_edge=X_edge,
                                   edge_index=edge_index, A_temporal=A_temporal)
        all_embs.append(emb)
    avg_emb = np.mean(all_embs, axis=0)  # (N, D)

    mask_arr = np.asarray(mask, dtype=bool)
    emb_masked = avg_emb[mask_arr]
    y_masked = np.asarray(y)[mask_arr]

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3)
    coords = reducer.fit_transform(emb_masked)

    fig, ax = plt.subplots(figsize=(8, 7))
    for c_idx, cls in enumerate(classes):
        sel = y_masked == c_idx
        label = CLASS_DISPLAY.get(cls, cls)
        color = CLASS_COLORS.get(cls, None)
        ax.scatter(coords[sel, 0], coords[sel, 1], label=label, s=25, alpha=0.7,
                   color=color, edgecolors="white", linewidths=0.3)

    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.set_title("UMAP of Learned Node Embeddings (Test Set)", fontsize=14)
    ax.legend(fontsize=11, markerscale=1.5)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_path = Path(out_dir) / "umap_embeddings.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  [saved] UMAP embeddings -> {save_path}")


def plot_seed_confusion_matrix(per_seed_results, seeds, target_seed, classes, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    if target_seed not in seeds:
        print(f"  [skip] Seed {target_seed} not found among trained seeds {seeds}")
        return
    idx = seeds.index(target_seed)
    if idx >= len(per_seed_results) or not per_seed_results[idx].get("confusion_matrix"):
        print(f"  [skip] No confusion matrix available for seed {target_seed}")
        return

    cm = np.array(per_seed_results[idx]["confusion_matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm / row_sums * 100.0, 0.0)
    labels = [CLASS_DISPLAY.get(c, c) for c in classes]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
            ax.text(j, i, f"{cm[i, j]:.0f}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=12, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — Seed {target_seed}", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_path = out_dir / f"confusion_matrix_seed{target_seed}.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  [saved] Seed {target_seed} confusion matrix -> {save_path}")


def plot_mci_recall_ablation(out_dir):
    """Scan sibling result directories for metrics.json and plot MCI recall comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    parent = out_dir.parent
    entries = []
    current_ablation = None

    for d in sorted(parent.iterdir()):
        mf = d / "metrics.json"
        if not mf.is_file():
            continue
        try:
            with open(mf) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        ablation = data.get("args", {}).get("edge_ablation", d.name)
        ci = data.get("seed_confidence_intervals", {}).get("per_class_recall", {}).get("0.5", {})
        mean_val = ci.get("mean", None)
        std_val = ci.get("std", 0.0)
        if mean_val is None:
            # Fallback: get from classification report
            report = data.get("ensemble_test", {}).get("classification_report", {})
            mci_recall = report.get("1", {}).get("recall", None)
            if mci_recall is None:
                continue
            mean_val = mci_recall
            std_val = 0.0
        entries.append({"ablation": ablation, "mean": mean_val, "std": std_val, "dir": d.name})
        if d.resolve() == out_dir.resolve():
            current_ablation = ablation

    if len(entries) < 2:
        print(f"  [skip] MCI recall ablation — need >= 2 sibling result dirs with metrics.json "
              f"(found {len(entries)})")
        return

    entries.sort(key=lambda x: x["mean"], reverse=True)
    labels = [e["ablation"] for e in entries]
    means = [e["mean"] for e in entries]
    stds = [e["std"] for e in entries]

    fig, ax = plt.subplots(figsize=(8, max(3, len(entries) * 0.7)))
    bars = ax.barh(range(len(entries)), means, xerr=stds, capsize=4,
                   color=CLASS_COLORS["0.5"], alpha=0.85, edgecolor="gray")

    # Highlight current run
    for i, e in enumerate(entries):
        if e["ablation"] == current_ablation:
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(2.5)
        ax.text(means[i] + stds[i] + 0.005, i, f"{means[i]:.3f} \u00b1 {stds[i]:.3f}",
                va="center", fontsize=9)

    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("MCI Recall", fontsize=12)
    ax.set_title("MCI Recall by Edge Ablation Setting", fontsize=13)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    save_path = out_dir / "mci_recall_ablation.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  [saved] MCI recall ablation -> {save_path}")


def plot_feature_importance(class_importance, classes, out_dir, top_k=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not class_importance or all(len(v) == 0 for v in class_importance.values()):
        print("  [skip] No feature importance data available")
        return

    def _clean_name(name):
        for prefix in ("clinical_", "mri_", "pet_"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name.replace("_", " ")

    n_cls = len(classes)
    fig, axes = plt.subplots(1, n_cls, figsize=(6 * n_cls, 6))
    if n_cls == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        items = class_importance.get(cls, [])[:top_k]
        if not items:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{CLASS_DISPLAY.get(cls, cls)}", fontsize=13)
            continue
        items = items[::-1]  # reverse so #1 is at top
        names = [_clean_name(n) for n, _ in items]
        scores = [s for _, s in items]
        ax.barh(range(len(items)), scores, color=CLASS_COLORS.get(cls, "#666"), alpha=0.85)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"Top {top_k} Features — {CLASS_DISPLAY.get(cls, cls)}", fontsize=13)
        ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    save_path = Path(out_dir) / f"feature_importance_top{top_k}.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  [saved] Feature importance -> {save_path}")


def plot_neighbor_influence(node_results, y_all, classes, c2i, out_dir):
    """Bar chart of top-10 neighbor influences for one correctly-predicted MCI patient."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    if not node_results:
        print("  [skip] No neighbor influence data available")
        return

    i2c = {v: k for k, v in c2i.items()}
    mci_idx = c2i.get("0.5")
    if mci_idx is None:
        print("  [skip] MCI class not found in c2i")
        return

    # Find a correctly predicted MCI patient
    target_node = None
    for nidx, info in node_results.items():
        if int(y_all[nidx]) == mci_idx and info.get("predicted_label") == "0.5":
            if info.get("neighbors"):
                target_node = nidx
                break

    if target_node is None:
        # Fallback: any MCI patient with neighbors
        for nidx, info in node_results.items():
            if int(y_all[nidx]) == mci_idx and info.get("neighbors"):
                target_node = nidx
                break

    if target_node is None:
        print("  [skip] No MCI patient found in neighbor influence results")
        return

    info = node_results[target_node]
    neighbors = info["neighbors"][:10]
    class_dist = info.get("class_distribution", {})

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(1, 5, figure=fig)
    ax_main = fig.add_subplot(gs[0, :4])
    ax_dist = fig.add_subplot(gs[0, 4])

    # Main bar chart — reversed so rank 1 is at top
    # Colored/labeled by each neighbor's PREDICTED class (never y_all), matching
    # the prediction-based graph-agreement metric.
    neighbors_rev = neighbors[::-1]
    bar_colors = []
    bar_labels = []
    for nb in neighbors_rev:
        nb_cls = nb.get("neighbor_predicted_class", "?")
        bar_colors.append(CLASS_COLORS.get(nb_cls, "#999"))
        bar_labels.append(f"Node {nb['neighbor_idx']} ({CLASS_DISPLAY.get(nb_cls, nb_cls)})")

    pcts = [nb.get("influence_pct", 0) for nb in neighbors_rev]
    ax_main.barh(range(len(neighbors_rev)), pcts, color=bar_colors, alpha=0.85, edgecolor="gray")
    ax_main.set_yticks(range(len(neighbors_rev)))
    ax_main.set_yticklabels(bar_labels, fontsize=9)
    ax_main.set_xlabel("Influence (%)", fontsize=11)
    ax_main.set_title(f"Neighbor Influence (predicted class) — MCI Patient (node {target_node})", fontsize=13)
    ax_main.grid(True, axis="x", alpha=0.3)

    legend_handles = [Patch(facecolor=CLASS_COLORS.get(cls, "#999"), label=CLASS_DISPLAY.get(cls, cls))
                      for cls in classes]
    ax_main.legend(handles=legend_handles, loc="lower right", fontsize=9)

    # Stacked bar for class distribution
    bottom = 0.0
    for cls in classes:
        pct = class_dist.get(cls, 0.0)
        ax_dist.bar(0, pct, bottom=bottom, color=CLASS_COLORS.get(cls, "#999"),
                    width=0.5, edgecolor="white", linewidth=0.5)
        if pct > 5:
            ax_dist.text(0, bottom + pct / 2, f"{pct:.0f}%", ha="center", va="center", fontsize=9)
        bottom += pct
    ax_dist.set_xlim(-0.5, 0.5)
    ax_dist.set_ylim(0, 100)
    ax_dist.set_xticks([])
    ax_dist.set_ylabel("Class Distribution (%)", fontsize=10)
    ax_dist.set_title("Neighbors\n(predicted class)", fontsize=11)

    fig.tight_layout()
    save_path = Path(out_dir) / "neighbor_influence_plot.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  [saved] Neighbor influence -> {save_path}")


def plot_uncertainty_violin(uncertainty, y_all, test_mask, classes, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if uncertainty is None or "predictive_entropy" not in uncertainty:
        print("  [skip] No uncertainty data available")
        return

    test_idx = np.where(np.asarray(test_mask, dtype=bool))[0]
    ent = uncertainty["predictive_entropy"][test_idx]
    labels = np.asarray(y_all)[test_idx]

    data = []
    colors = []
    tick_labels = []
    for cls in classes:
        c_idx = {c: i for i, c in enumerate(classes)}[cls]
        vals = ent[labels == c_idx]
        data.append(vals)
        colors.append(CLASS_COLORS.get(cls, "#999"))
        tick_labels.append(CLASS_DISPLAY.get(cls, cls))

    fig, ax = plt.subplots(figsize=(7, 6))
    positions = list(range(len(classes)))

    parts = ax.violinplot(data, positions=positions, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.7)

    bp = ax.boxplot(data, positions=positions, widths=0.15, showfliers=False,
                    patch_artist=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"))

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=12)
    ax.set_ylabel("Predictive Entropy", fontsize=12)
    ax.set_title("Uncertainty Distribution by True Class", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_path = Path(out_dir) / "uncertainty_violin.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  [saved] Uncertainty violin -> {save_path}")


# Evaluation

def evaluate(model, X, A_norm, y, mask, device=None,
             X_edge=None, edge_index=None, A_temporal=None):
    model.eval()
    X_e = X_edge.to(device) if X_edge is not None else None
    ei = edge_index.to(device) if edge_index is not None else None
    A_t = A_temporal.to(device) if A_temporal is not None else None
    with torch.no_grad():
        logits, _ = model(X.to(device), A_norm.to(device), X_edge=X_e, edge_index=ei, A_temporal=A_t)
        pred = logits.argmax(dim=1).cpu().numpy()
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool)
    y_m, p_m = y[mask], pred[mask]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_m, p_m)),
        "confusion_matrix": confusion_matrix(y_m, p_m).tolist(),
        "classification_report": classification_report(y_m, p_m, output_dict=True),
    }


def compute_ci(values, confidence=0.95, clip=None):
    """Compute mean, std, and confidence interval for a list of values.

    Uses t-distribution for small samples (n < 30), z-distribution otherwise.
    `clip` is an optional (lo, hi) tuple to clamp ci_lower/ci_upper — e.g.
    (0, 1) for metrics like recall, F1, or AP that are bounded to [0, 1].
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    if n <= 1:
        return {"mean": mean, "std": std, "ci_lower": mean, "ci_upper": mean, "ci_level": confidence}

    from scipy import stats
    se = std / np.sqrt(n)
    if n < 30:
        t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_val * se
    else:
        z_val = stats.norm.ppf((1 + confidence) / 2)
        margin = z_val * se

    ci_lower = float(mean - margin)
    ci_upper = float(mean + margin)
    if clip is not None:
        ci_lower = max(clip[0], ci_lower)
        ci_upper = min(clip[1], ci_upper)

    return {
        "mean": mean, "std": std,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
        "ci_level": confidence,
    }


def compute_uncertainty_metrics(per_seed_probs: np.ndarray) -> dict:
    """Ensemble uncertainty from per-seed softmax probs (S, N, C): predictive
    entropy H[E[p]], its aleatoric (E[H[p]]) and epistemic (difference)
    components, plus max confidence and winning-class prob std across seeds.
    """
    mean_probs = per_seed_probs.mean(axis=0)                                          # (N, C)
    eps = 1e-8
    pred_ent = -(mean_probs * np.log(mean_probs + eps)).sum(axis=1)                   # (N,)
    seed_ents = -(per_seed_probs * np.log(per_seed_probs + eps)).sum(axis=2)          # (S, N)
    mean_ent = seed_ents.mean(axis=0)                                                 # (N,)
    epistemic = np.clip(pred_ent - mean_ent, 0.0, None)                              # (N,)
    max_prob = mean_probs.max(axis=1)                                                 # (N,)
    pred_class = mean_probs.argmax(axis=1)                                            # (N,)
    pred_std = per_seed_probs[:, np.arange(len(pred_class)), pred_class].std(axis=0) # (N,)
    return {
        "mean_probs": mean_probs,
        "predictive_entropy": pred_ent,
        "mean_entropy": mean_ent,
        "epistemic_uncertainty": epistemic,
        "max_prob": max_prob,
        "pred_std": pred_std,
    }


def _save_uncertainty_csv(unc: dict, y_all, test_mask, classes: list, out_dir, df=None) -> None:
    """Save per-node uncertainty estimates for test nodes to uncertainty_estimates.csv."""
    test_indices = np.where(np.asarray(test_mask, dtype=bool))[0]
    threshold = float(np.percentile(unc["predictive_entropy"][test_indices], 75))
    rows = []
    for node_i in test_indices:
        row = {
            "node_idx": int(node_i),
            "true_class": classes[int(y_all[node_i])],
            "pred_class": classes[int(unc["mean_probs"][node_i].argmax())],
            "max_prob": float(unc["max_prob"][node_i]),
            "predictive_entropy": float(unc["predictive_entropy"][node_i]),
            "epistemic_uncertainty": float(unc["epistemic_uncertainty"][node_i]),
            "aleatoric_uncertainty": float(unc["mean_entropy"][node_i]),
            "pred_std": float(unc["pred_std"][node_i]),
            "is_high_uncertainty": bool(unc["predictive_entropy"][node_i] > threshold),
        }
        if df is not None and "subject_id" in df.columns:
            row["subject_id"] = str(df.iloc[node_i]["subject_id"])
        if df is not None and "clinical_session_id" in df.columns:
            row["clinical_session_id"] = str(df.iloc[node_i]["clinical_session_id"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "uncertainty_estimates.csv", index=False)


def _save_ensemble_probabilities_csv(res_te, y_all, test_mask, classes, out_dir, df=None) -> None:
    """Save per-test-node ensemble-averaged class probabilities to ensemble_probabilities.csv."""
    test_indices = np.where(np.asarray(test_mask, dtype=bool))[0]
    probs = res_te["probs"]
    rows = []
    for node_i in test_indices:
        row = {
            "node_idx": int(node_i),
            "true_class": classes[int(y_all[node_i])],
            "pred_class": classes[int(probs[node_i].argmax())],
        }
        for c_idx, cls in enumerate(classes):
            row[f"prob_{CLASS_DISPLAY.get(cls, cls)}"] = float(probs[node_i, c_idx])
        if df is not None and "subject_id" in df.columns:
            row["subject_id"] = str(df.iloc[node_i]["subject_id"])
        if df is not None and "clinical_session_id" in df.columns:
            row["clinical_session_id"] = str(df.iloc[node_i]["clinical_session_id"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "ensemble_probabilities.csv", index=False)


def _build_uncertainty_summary(res_te: dict, y_all, test_mask, classes: list, c2i: dict) -> dict:
    """Build per-class uncertainty summary for metrics.json."""
    if "uncertainty" not in res_te:
        return {}
    unc = res_te["uncertainty"]
    test_idx = np.where(np.asarray(test_mask, dtype=bool))[0]
    if len(test_idx) == 0:
        return {}
    threshold = float(np.percentile(unc["predictive_entropy"][test_idx], 75))
    out = {}
    for cls in classes:
        ci = c2i[cls]
        cls_idx = test_idx[np.asarray(y_all)[test_idx] == ci]
        if len(cls_idx) == 0:
            continue
        ents = unc["predictive_entropy"][cls_idx]
        out[cls] = {
            "mean_predictive_entropy": float(ents.mean()),
            "mean_max_prob": float(unc["max_prob"][cls_idx].mean()),
            "mean_epistemic_uncertainty": float(unc["epistemic_uncertainty"][cls_idx].mean()),
            "mean_aleatoric_uncertainty": float(unc["mean_entropy"][cls_idx].mean()),
            "frac_high_uncertainty": float((ents > threshold).mean()),
            "n_nodes": int(len(cls_idx)),
        }
    return out


def print_uncertainty_summary(unc: dict, y_all, test_mask, classes: list, c2i: dict) -> None:
    test_idx = np.where(np.asarray(test_mask, dtype=bool))[0]
    if len(test_idx) == 0:
        return
    threshold = float(np.percentile(unc["predictive_entropy"][test_idx], 75))
    print("\n  === ENSEMBLE UNCERTAINTY SUMMARY ===")
    print(f"  High-uncertainty threshold (75th pct): {threshold:.4f} nats")
    for cls in classes:
        ci = c2i[cls]
        cls_idx = test_idx[np.asarray(y_all)[test_idx] == ci]
        if len(cls_idx) == 0:
            continue
        ents = unc["predictive_entropy"][cls_idx]
        mean_conf = float(unc["max_prob"][cls_idx].mean())
        frac_hi = float((ents > threshold).mean())
        print(f"\n  [{cls}]  n={len(cls_idx)}  mean_conf={mean_conf:.3f}  "
              f"mean_entropy={ents.mean():.4f}  high_unc_frac={frac_hi:.2f}")
        # Show up to 3 high-uncertainty examples
        hi_mask = ents > threshold
        hi_nodes = cls_idx[hi_mask][:3]
        for node_i in hi_nodes:
            pred_cls = classes[int(unc["mean_probs"][node_i].argmax())]
            print(f"    node {node_i:4d}  pred={pred_cls}  "
                  f"conf={unc['max_prob'][node_i]:.3f}  "
                  f"entropy={unc['predictive_entropy'][node_i]:.4f}  "
                  f"epistemic={unc['epistemic_uncertainty'][node_i]:.4f}")
    print()


def ensemble_evaluate(models, X, A_norm, y, mask, device,
                       X_edge=None, edge_index=None, A_temporal=None):
    """Average probabilities from all models, then take argmax."""
    X_e = X_edge.to(device) if X_edge is not None else None
    ei = edge_index.to(device) if edge_index is not None else None
    A_t = A_temporal.to(device) if A_temporal is not None else None
    all_probs_list = []
    for m in models:
        m.eval()
        with torch.no_grad():
            logits, _ = m(X.to(device), A_norm.to(device), X_edge=X_e, edge_index=ei, A_temporal=A_t)
            all_probs_list.append(F.softmax(logits.cpu(), dim=1).numpy())
    per_seed_probs = np.stack(all_probs_list)   # (S, N, C)
    probs = per_seed_probs.mean(axis=0)

    unc = compute_uncertainty_metrics(per_seed_probs)
    pred = probs.argmax(axis=1)
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool)
    y_m, p_m = y[mask], pred[mask]
    probs_m = probs[mask]

    # AUC One-vs-Rest (macro)
    try:
        auc_ovr = float(roc_auc_score(y_m, probs_m, multi_class='ovr', average='macro'))
    except (ValueError, IndexError):
        auc_ovr = float('nan')

    # Per-class Average Precision
    n_classes = probs_m.shape[1]
    per_class_ap = {}
    for c in range(n_classes):
        try:
            per_class_ap[str(c)] = float(average_precision_score(
                (y_m == c).astype(int), probs_m[:, c]))
        except (ValueError, IndexError):
            per_class_ap[str(c)] = float('nan')

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_m, p_m)),
        "confusion_matrix": confusion_matrix(y_m, p_m).tolist(),
        "classification_report": classification_report(y_m, p_m, output_dict=True),
        "probs": probs,
        "uncertainty": unc,
        "auc_ovr_macro": auc_ovr,
        "per_class_ap": per_class_ap,
    }


def get_feature_names(pre, num_cols, cat_cols):
    """Extract feature names from the fitted ColumnTransformer.
    Uses get_feature_names_out() which works on sklearn >= 1.0.
    Falls back to manual construction if that fails.
    """
    try:
        all_names = pre.get_feature_names_out()
        # sklearn ColumnTransformer prefixes names with transformer name (e.g. "num__feature").
        # Strip the prefix so downstream code can match against raw column names.
        # Safe no-op when no "__" is present (older sklearn or already stripped).
        stripped = [n.split("__", 1)[-1] if "__" in n else n for n in all_names]
        return stripped
    except Exception:
        pass
    # Fallback: numeric columns first, then one-hot encoded categoricals
    names = list(num_cols)
    try:
        if "cat" in pre.named_transformers_:
            cat_pipe = pre.named_transformers_["cat"]
            ohe = cat_pipe[-1]
            if hasattr(ohe, "get_feature_names_out"):
                cat_names = ohe.get_feature_names_out(cat_cols)
                names.extend(cat_names)
    except Exception:
        pass
    return names


# GNN Explainer

def gnn_explainer_feature_mask(model, X, A_norm, node_idx, target_class,
                                device, epochs=200, lr=0.01, reg_coeff=0.01,
                                X_edge=None, edge_index=None,
                                A_temporal=None):

    num_features = X.shape[1]
    # Learnable mask parameters (passed through sigmoid -> [0, 1])
    feat_mask_param = torch.nn.Parameter(torch.zeros(num_features, device=device))

    optimizer = torch.optim.Adam([feat_mask_param], lr=lr)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for _ in range(epochs):
        optimizer.zero_grad()
        mask = torch.sigmoid(feat_mask_param)
        X_masked = X * mask.unsqueeze(0)  # broadcast (N, F) * (1, F)

        logits, _ = model(X_masked, A_norm, X_edge=X_edge, edge_index=edge_index, A_temporal=A_temporal)
        log_probs = F.log_softmax(logits, dim=1)

        # Maximize log-prob of target class for this node + L1 sparsity
        loss = -log_probs[node_idx, target_class] + reg_coeff * mask.sum()
        loss.backward()
        optimizer.step()

    for p in model.parameters():
        p.requires_grad_(True)

    return torch.sigmoid(feat_mask_param).detach().cpu().numpy()


def compute_feature_importance(models, X, A_norm, y, mask, feature_names,
                                classes, c2i, device, top_k=25, epochs=200,
                                X_edge=None, edge_index=None,
                                A_temporal=None):

    X_t = X.clone().to(device)
    A_t = A_norm.clone().to(device)
    X_e = X_edge.clone().to(device) if X_edge is not None else None
    ei = edge_index.clone().to(device) if edge_index is not None else None
    A_t_temporal = A_temporal.clone().to(device) if A_temporal is not None else None
    y_arr = np.asarray(y)
    mask_arr = np.asarray(mask, dtype=bool)

    # Get ensemble predictions
    all_preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            logits, _ = m(X_t, A_t, X_edge=X_e, edge_index=ei, A_temporal=A_t_temporal)
            all_preds.append(logits.cpu())
    avg_out = torch.stack(all_preds).mean(dim=0)
    ensemble_pred = avg_out.argmax(dim=1).numpy()

    class_importance = {}

    for cls_label in classes:
        cls_idx = c2i[cls_label]

        # Correctly classified test samples of this class
        cls_mask = mask_arr & (y_arr == cls_idx) & (ensemble_pred == cls_idx)
        cls_indices = np.where(cls_mask)[0]

        if len(cls_indices) == 0:
            print(f"  [explainer] No correctly classified samples for class {cls_label}, skipping")
            class_importance[cls_label] = []
            continue

        print(f"  [explainer] Class {cls_label}: explaining {len(cls_indices)} samples...")

        # Collect feature masks across all samples and all ensemble models
        all_masks = []
        patient_masks = {}  # node_idx -> avg importance across models
        for patient_idx in cls_indices:
            per_model_masks = []
            for m in models:
                importance = gnn_explainer_feature_mask(
                    m, X_t, A_t, int(patient_idx), cls_idx,
                    device, epochs=epochs,
                    X_edge=X_e, edge_index=ei, A_temporal=A_t_temporal
                )
                all_masks.append(importance)
                per_model_masks.append(importance)
            patient_masks[int(patient_idx)] = np.mean(per_model_masks, axis=0)

        # Average across all samples and models
        avg_importance = np.mean(all_masks, axis=0)  # (F,)

        # Map to feature names and sort
        n_feats = min(len(feature_names), len(avg_importance))
        feat_imp = [(feature_names[i], float(avg_importance[i])) for i in range(n_feats)]
        feat_imp.sort(key=lambda x: x[1], reverse=True)

        class_importance[cls_label] = feat_imp[:top_k]

    return class_importance, patient_masks, ensemble_pred


def build_patient_evidence(patient_masks, ensemble_pred, y,
                            feature_names, c2i, feat_df,
                            top_k_patient=3):
    """Build per-patient GNNExplainer evidence from precomputed masks.

    patient_masks only contains correctly-classified test patients — that
    filtering happens upstream in compute_feature_importance.
    """
    y_arr = np.asarray(y)
    i2c = {v: k for k, v in c2i.items()}
    feat_df_cols = set(feat_df.columns)

    patient_evidence = []

    for node_idx, avg_imp in patient_masks.items():
        true_cls = i2c.get(int(y_arr[node_idx]), "?")
        pred_cls = i2c.get(int(ensemble_pred[node_idx]), "?")

        # Top-K features for this patient
        n_feats = min(len(feature_names), len(avg_imp))
        ranked = sorted(range(n_feats), key=lambda i: avg_imp[i], reverse=True)

        evidence = []
        for fi in ranked[:top_k_patient]:
            fname = feature_names[fi]
            raw_val = None
            if fname in feat_df_cols:
                raw_val = feat_df.iloc[node_idx][fname]
                try:
                    raw_val = float(raw_val)
                except (ValueError, TypeError):
                    raw_val = str(raw_val)
            evidence.append({
                "feature": fname,
                "importance": float(avg_imp[fi]),
                "raw_value": raw_val,
            })

        patient_evidence.append({
            "node_idx": node_idx,
            "true_class": true_cls,
            "pred_class": pred_cls,
            "evidence": evidence,
        })

    return patient_evidence


def _save_patient_evidence_csv(patient_evidence, out_dir):
    """Save patient-level GNNExplainer evidence to CSV."""
    rows = []
    for pe in patient_evidence:
        for rank, ev in enumerate(pe["evidence"], 1):
            rows.append({
                "node_idx": pe["node_idx"],
                "true_class": pe["true_class"],
                "pred_class": pe["pred_class"],
                "rank": rank,
                "feature": ev["feature"],
                "importance": ev["importance"],
                "raw_value": ev["raw_value"],
            })
    pd.DataFrame(rows).to_csv(out_dir / "patient_evidence.csv", index=False)


def print_patient_evidence_summary(patient_evidence, classes):
    rng = np.random.RandomState(42)

    print("\n  Patient-Level GNNExplainer Evidence ")

    for cls_label in classes:
        cls_patients = [pe for pe in patient_evidence if pe["true_class"] == cls_label]
        if not cls_patients:
            continue

        # Sample up to 3 representative patients
        n_show = min(3, len(cls_patients))
        chosen = rng.choice(len(cls_patients), size=n_show, replace=False)

        print(f"\n  --- Class: {cls_label} ({len(cls_patients)} patients) ---")

        for idx in chosen:
            pe = cls_patients[idx]
            print(f"\n    Patient (node {pe['node_idx']})  |  True: {pe['true_class']}  |  Pred: {pe['pred_class']}")
            print(f"    Top features driving this prediction:")
            for rank, ev in enumerate(pe["evidence"], 1):
                val_str = f"{ev['raw_value']:.4f}" if isinstance(ev['raw_value'], float) else str(ev['raw_value'])
                print(f"      {rank}. {ev['feature']:<40s}  value={val_str:<12s}  importance={ev['importance']:.4f}")


# Neighbor Influence Explanation

# Features considered clinically meaningful for connection explanations.
# Used by print_neighbor_influence_summary to describe why two nodes are linked.
_CLINICAL_INTERP_FEATURES = [
    ("mri_hippocampus_vol_mean",      "hippocampal volume"),
    ("mri_entorhinal_vol_mean",       "entorhinal cortex volume"),
    ("mri_amygdala_vol_mean",         "amygdala volume"),
    ("mri_inferior_temporal_vol_mean","inferior temporal volume"),
    ("mri_middle_temporal_vol_mean",  "middle temporal volume"),
    ("clinical_MMSCORE",              "MMSE cognitive score"),
    ("clinical_FAQTOTAL",             "FAQ functional score"),
    ("clinical_LDELTOTAL",            "logical memory score"),
    ("clinical_TRABSCOR",             "Trail Making Test score"),
    ("clinical_APOE4_count",          "APOE4 genotype"),
]


def compute_neighbor_influence(model, X, A_norm, node_idx, top_k=10,
                                device=None, X_edge=None, edge_index=None,
                                A_temporal=None):
    """Quantify how much each neighbor contributed to a node's prediction.

    Uses final node embeddings from the GCN after JK aggregation but before
    the classifier head. For each neighbor j of target node i (self excluded):

        influence(i, j) = edge_weight(i, j) * cosine_similarity(emb_i, emb_j)

    Edge weights come from A_norm, the normalized adjacency already built by
    the pipeline (RBF / linear / learned, temporal or not).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    X_d = X.to(device)
    A_d = A_norm.to(device)
    X_e = X_edge.to(device) if X_edge is not None else None
    ei  = edge_index.to(device) if edge_index is not None else None
    A_t = A_temporal.to(device) if A_temporal is not None else None

    with torch.no_grad():
        # ---- Replicate forward pass up to JK aggregation ----
        # Node feature masking is disabled (model.training == False)
        X_masked = X_d

        # Determine which adjacency to use (mirrors SpectralGCN.forward)
        A_used = A_d

        # GCN branch: collect all layer outputs for JK
        H = X_masked
        layer_outputs = []
        for i, conv in enumerate(model.gcn_convs):
            H_res = model.input_proj(H) if i == 0 else H
            H = conv(H, A_used)
            if model.use_bn:
                H = model.gcn_bns[i](H)
            H = F.relu(H + H_res)
            layer_outputs.append(H)
        H_jk = torch.cat(layer_outputs, dim=1) if model.use_jk else layer_outputs[-1]  # (N, D)

        # Temporal GCN branch (if active)
        if model.use_temporal_branch and A_t is not None:
            H_t = X_masked
            temporal_outputs = []
            for i, conv in enumerate(model.temporal_gcn_convs):
                H_t_res = model.temporal_input_proj(H_t) if i == 0 else H_t
                H_t = conv(H_t, A_t)
                if model.use_bn:
                    H_t = model.temporal_gcn_bns[i](H_t)
                H_t = F.relu(H_t + H_t_res)
                temporal_outputs.append(H_t)
            H_temporal_jk = torch.cat(temporal_outputs, dim=1) if model.use_jk else temporal_outputs[-1]
            H_mlp_ni = model.mlp_branch(X_masked) if model.use_mlp_branch else None
            parts = [H_jk, H_temporal_jk] + ([H_mlp_ni] if H_mlp_ni is not None else [])
            H_combined = torch.cat(parts, dim=1)
        else:
            H_mlp_ni = model.mlp_branch(X_masked) if model.use_mlp_branch else None
            parts = [H_jk] + ([H_mlp_ni] if H_mlp_ni is not None else [])
            H_combined = torch.cat(parts, dim=1)

        # H_combined shape: (N, combined_dim) — post-JK, pre-classifier embeddings
        embeddings = H_combined  # (N, D)

        # ---- Compute neighbor influences (self excluded) ----
        edge_weights_row = A_norm[node_idx].to(device)  # (N,)
        neighbor_indices = [
            j for j in torch.where(edge_weights_row > 0)[0].cpu().tolist()
            if j != node_idx  # exclude self-loop
        ]

        emb_i = embeddings[node_idx]  # (D,)
        norm_i = emb_i.norm().clamp(min=1e-8)

        results = []
        for j in neighbor_indices:
            w_ij = float(edge_weights_row[j].item())
            emb_j = embeddings[j]
            norm_j = emb_j.norm().clamp(min=1e-8)
            cos_sim = float((emb_i @ emb_j / (norm_i * norm_j)).item())
            # Clamp cosine to [0, 1] so negative similarity doesn't invert influence
            cos_sim_clamped = max(0.0, cos_sim)
            influence = w_ij * cos_sim_clamped
            results.append({
                "neighbor_idx": int(j),
                "influence_score": round(influence, 8),
                "edge_weight": round(w_ij, 8),
                "embedding_similarity": round(cos_sim, 8),
            })

    results.sort(key=lambda x: x["influence_score"], reverse=True)
    return results[:top_k] if top_k is not None else results


def explain_neighbor_influences(models, X, A_norm, y_all, mask, classes, c2i,
                                 device, top_k=10, out_dir=None, verbose=True,
                                 X_edge=None, edge_index=None, A_temporal=None,
                                 X_edge_raw=None, X_edge_std=None,
                                 edge_feat_cols=None, kernel='rbf', rbf_gamma=None,
                                 df=None):
    """Run neighbor influence explanation for all test nodes, averaged over the
    ensemble: per node, averages influence across models, drops the
    self-neighbor, normalizes to a % contribution within the top-K, buckets
    neighbors by class, and computes a graph agreement score (% influence
    from same-class neighbors). Also calls explain_edge on the top neighbor.

    class_distribution, dominant neighbor class, and graph_agreement are all
    based on ensemble *predicted* classes, never y_all ground truth — the
    true-label fields are audit-only and don't feed the agreement math.

    Returns a dict keyed by node_idx with the ranked neighbor list,
    class_distribution, graph_agreement, top-neighbor edge explanation, and
    predicted/true class labels.
    """
    if not models:
        raise ValueError("explain_neighbor_influences() requires a non-empty models list")

    i2c = {v: k for k, v in c2i.items()}
    node_indices = np.where(np.asarray(mask, dtype=bool))[0]


    all_node_results = {}
    csv_rows = []
    summary_rows = []

    # ---- Pre-compute ensemble predictions for all nodes (one pass, cached) ----
    # Average softmax probabilities across models, then argmax — uncalibrated,
    # matching ensemble_evaluate(). Computed once and reused for the index
    # visit, its neighbors, and predicted_label so all three agree.
    with torch.no_grad():
        _X_dev  = X.to(device)
        _A_dev  = A_norm.to(device)
        _Xe_dev = X_edge.to(device)    if X_edge    is not None else None
        _ei_dev = edge_index.to(device) if edge_index is not None else None
        _At_dev = A_temporal.to(device) if A_temporal is not None else None
        _all_probs = []
        for m in models:
            m.eval()
            out, _ = m(_X_dev, _A_dev, X_edge=_Xe_dev, edge_index=_ei_dev, A_temporal=_At_dev)
            _all_probs.append(F.softmax(out.cpu(), dim=1))
        _avg_probs = torch.stack(_all_probs).mean(dim=0)   # (N, C)
        _all_preds = _avg_probs.argmax(dim=1).numpy()      # (N,)

    for node_idx in node_indices:
        true_label = i2c.get(int(y_all[node_idx]), "?")  # AUDIT ONLY
        predicted_label = i2c.get(int(_all_preds[int(node_idx)]), "?")

        # ---- Collect per-model influence dicts keyed by neighbor_idx ----
        model_results = []
        for m in models:
            res = compute_neighbor_influence(
                m, X, A_norm, int(node_idx), top_k=None,
                device=device, X_edge=X_edge, edge_index=edge_index,
                A_temporal=A_temporal
            )
            model_results.append({r["neighbor_idx"]: r for r in res})

        # Union of neighbor indices across models (self already excluded inside compute)
        all_neighbor_idxs = set()
        for mr in model_results:
            all_neighbor_idxs.update(mr.keys())

        # Average scores across models
        averaged = []
        for j in all_neighbor_idxs:
            scores = [mr[j]["influence_score"]      if j in mr else 0.0 for mr in model_results]
            ew     = [mr[j]["edge_weight"]           if j in mr else 0.0 for mr in model_results]
            sims   = [mr[j]["embedding_similarity"]  if j in mr else 0.0 for mr in model_results]
            averaged.append({
                "neighbor_idx": j,
                "influence_score": round(float(np.mean(scores)), 8),
                "edge_weight": round(float(np.mean(ew)), 8),
                "embedding_similarity": round(float(np.mean(sims)), 8),
            })

        averaged.sort(key=lambda x: x["influence_score"], reverse=True)
        top_neighbors = averaged[:top_k]

        # ---- Normalize influence scores to % within top-K ----
        # (scores/ranking untouched by the predicted-vs-true class bucketing below)
        total_influence = sum(nb["influence_score"] for nb in top_neighbors)
        for nb in top_neighbors:
            pct = (nb["influence_score"] / total_influence * 100.0
                   if total_influence > 0 else 0.0)
            nb["influence_pct"] = round(pct, 2)
            nb["neighbor_predicted_class"] = i2c.get(
                int(_all_preds[int(nb["neighbor_idx"])]), "?"
            )
            nb["neighbor_true_class"] = i2c.get(
                int(y_all[nb["neighbor_idx"]]), "?"
            )  # AUDIT ONLY — not used in distribution/agreement arithmetic below

        # ---- Class influence distribution (% per class, over top-K) ----
        # Bucketed by PREDICTED class, matching ensemble_evaluate — never y_all.
        class_influence_raw = {cls: 0.0 for cls in classes}
        for nb in top_neighbors:
            nb_pred_cls = nb["neighbor_predicted_class"]
            if nb_pred_cls in class_influence_raw:
                class_influence_raw[nb_pred_cls] += nb["influence_score"]
        class_distribution = {
            cls: round(class_influence_raw[cls] / total_influence * 100.0
                       if total_influence > 0 else 0.0, 2)
            for cls in classes
        }

        # Graph agreement score (prediction-based) 
        same_class_influence = class_influence_raw.get(predicted_label, 0.0)
        graph_agreement = round(
            same_class_influence / total_influence * 100.0
            if total_influence > 0 else 0.0, 2
        )

        # AUDIT ONLY: old true-label-based agreement, for comparison 
        # Computed independently from y_all; never feeds class_distribution,
        # graph_agreement, or any downstream consumer.
        label_class_influence_raw = {cls: 0.0 for cls in classes}
        for nb in top_neighbors:
            nb_true_cls = nb["neighbor_true_class"]
            if nb_true_cls in label_class_influence_raw:
                label_class_influence_raw[nb_true_cls] += nb["influence_score"]
        label_based_agreement = round(
            label_class_influence_raw.get(true_label, 0.0) / total_influence * 100.0
            if total_influence > 0 else 0.0, 2
        )

        # ---- explain_edge for top neighbor (data-driven biomarker explanation) ----
        top_neighbor_edge_explanation = None
        if (top_neighbors
                and X_edge_raw is not None
                and X_edge_std is not None
                and edge_feat_cols is not None):
            top_j = top_neighbors[0]["neighbor_idx"]
            top_neighbor_edge_explanation = explain_edge(
                int(node_idx), int(top_j),
                X_edge_raw, X_edge_std, edge_feat_cols,
                kernel=kernel, rbf_gamma=rbf_gamma
            )

        all_node_results[int(node_idx)] = {
            "neighbors": top_neighbors,
            "class_distribution": class_distribution,
            "graph_agreement": graph_agreement,
            "label_based_agreement": label_based_agreement,  # AUDIT ONLY — unused downstream
            "top_neighbor_edge_explanation": top_neighbor_edge_explanation,
            "predicted_label": predicted_label,
            "node_predicted_class": predicted_label,
            "node_true_class": true_label,  # AUDIT ONLY
        }

        # CSV rows 
        _has_subj = df is not None and "subject_id" in df.columns
        for rank, nb in enumerate(top_neighbors, 1):
            row_dict = {
                "node_idx": int(node_idx),
                "node_predicted_class": predicted_label,
                "node_true_class": true_label,  # AUDIT ONLY
                "rank": rank,
                "neighbor_idx": nb["neighbor_idx"],
                "neighbor_predicted_class": nb["neighbor_predicted_class"],
                "neighbor_true_class": nb["neighbor_true_class"],  # AUDIT ONLY
                "influence_score": nb["influence_score"],
                "influence_pct": nb["influence_pct"],
                "edge_weight": nb["edge_weight"],
                "embedding_similarity": nb["embedding_similarity"],
                "graph_agreement_pct": graph_agreement,
                **{f"class_influence_pct_{cls}": class_distribution[cls] for cls in classes},
            }
            if _has_subj:
                row_dict["subject_id"] = str(df.iloc[int(node_idx)]["subject_id"])
            csv_rows.append(row_dict)

        # Graph-agreement summary row (one per index visit) 
        dominant_cls = max(class_distribution, key=lambda c: class_distribution[c])
        summary_row = {
            "node_idx": int(node_idx),
            "node_predicted_class": predicted_label,
            "node_true_class": true_label,  # AUDIT ONLY
            "graph_agreement_pct": graph_agreement,
            "label_based_agreement_pct_AUDIT_ONLY": label_based_agreement,
            "dominant_neighbor_predicted_class": dominant_cls,
            "dominant_neighbor_pct": class_distribution[dominant_cls],
            "top_k": len(top_neighbors),
            **{f"class_influence_pct_{cls}": class_distribution[cls] for cls in classes},
        }
        if _has_subj:
            summary_row["subject_id"] = str(df.iloc[int(node_idx)]["subject_id"])
        summary_rows.append(summary_row)

    if out_dir is not None and csv_rows:
        save_path = out_dir / "neighbor_influence.csv"
        pd.DataFrame(csv_rows).to_csv(save_path, index=False)
        if verbose:
            print(f"\n  [neighbor-influence] Results saved to {save_path}")

    if out_dir is not None and summary_rows:
        summary_path = out_dir / "graph_agreement_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        if verbose:
            print(f"  [neighbor-influence] Graph agreement summary saved to {summary_path}")

    return all_node_results


def test_agreement_is_label_free(models, X, A_norm, y_all, mask, classes, c2i, device, **kw):
    """Regression test: graph_agreement must depend only on model predictions, never on y_all.

    Runs explain_neighbor_influences() once with the original y_all and once with a
    permuted y_all (predictions are unaffected, since they only depend on models/X/A_norm),
    then asserts graph_agreement, class_distribution, predicted_label, and every neighbor's
    ranking/predicted-class/influence values are unchanged. Only the audit-only true-label
    fields (node_true_class, neighbor_true_class) are expected to move — and at least one
    must, or the permutation didn't exercise anything.

    label_based_agreement is NOT asserted to differ: with few classes/nodes, the old
    true-label-based percentage can coincidentally match after permutation, so a mismatch
    just prints a warning rather than failing.

    **kw is forwarded to explain_neighbor_influences; out_dir and verbose are stripped out
    first so they can't collide with the explicit out_dir=None, verbose=False passed below.
    """
    kw = dict(kw)
    kw.pop("out_dir", None)
    kw.pop("verbose", None)

    res_orig = explain_neighbor_influences(
        models, X, A_norm, y_all, mask, classes, c2i, device,
        out_dir=None, verbose=False, **kw
    )

    y_permuted = np.random.default_rng(0).permutation(np.asarray(y_all))
    res_perm = explain_neighbor_influences(
        models, X, A_norm, y_permuted, mask, classes, c2i, device,
        out_dir=None, verbose=False, **kw
    )

    node_indices = sorted(res_orig.keys())
    assert node_indices == sorted(res_perm.keys()), (
        "explain_neighbor_influences returned different node sets for the same mask "
        "before/after permuting y_all — this should be impossible since y_all does not "
        "affect which nodes are explained"
    )

    n_tested = 0
    any_true_label_field_changed = False
    any_label_based_agreement_differed = False

    for node_idx in node_indices:
        a = res_orig[node_idx]
        b = res_perm[node_idx]

        # Must be unchanged: everything derived from predictions 
        assert np.isclose(a["graph_agreement"], b["graph_agreement"]), (
            f"node {node_idx}: graph_agreement changed under y_all permutation "
            f"({a['graph_agreement']} vs {b['graph_agreement']})"
        )
        for cls in classes:
            va = a["class_distribution"].get(cls, 0.0)
            vb = b["class_distribution"].get(cls, 0.0)
            assert np.isclose(va, vb), (
                f"node {node_idx}: class_distribution[{cls}] changed under y_all permutation "
                f"({va} vs {vb})"
            )
        assert a["predicted_label"] == b["predicted_label"], (
            f"node {node_idx}: predicted_label changed under y_all permutation "
            f"({a['predicted_label']} vs {b['predicted_label']})"
        )

        nbs_a, nbs_b = a["neighbors"], b["neighbors"]
        assert [nb["neighbor_idx"] for nb in nbs_a] == [nb["neighbor_idx"] for nb in nbs_b], (
            f"node {node_idx}: neighbor ranking changed under y_all permutation"
        )
        for nb_a, nb_b in zip(nbs_a, nbs_b):
            assert nb_a.get("neighbor_predicted_class") == nb_b.get("neighbor_predicted_class"), (
                f"node {node_idx}: neighbor {nb_a['neighbor_idx']} predicted class changed "
                f"under y_all permutation"
            )
            assert np.isclose(nb_a["influence_score"], nb_b["influence_score"]), (
                f"node {node_idx}: neighbor {nb_a['neighbor_idx']} influence_score changed"
            )
            assert np.isclose(nb_a["influence_pct"], nb_b["influence_pct"]), (
                f"node {node_idx}: neighbor {nb_a['neighbor_idx']} influence_pct changed"
            )

        # Audit-only true-label fields ARE expected to change 
        if a.get("node_true_class") != b.get("node_true_class"):
            any_true_label_field_changed = True
        for nb_a, nb_b in zip(nbs_a, nbs_b):
            if nb_a.get("neighbor_true_class") != nb_b.get("neighbor_true_class"):
                any_true_label_field_changed = True

        if not np.isclose(a.get("label_based_agreement", 0.0), b.get("label_based_agreement", 0.0)):
            any_label_based_agreement_differed = True

        n_tested += 1

    assert any_true_label_field_changed, (
        "permuting y_all did not change any audit-only true-label field for any node — "
        "the permutation had no effect, so this test did not exercise anything meaningful"
    )

    if not any_label_based_agreement_differed:
        print("  [test_agreement_is_label_free] WARNING: label_based_agreement did not differ "
              "for any node after permuting y_all (can happen by coincidence, e.g. on small graphs).")

    print(f"  [test_agreement_is_label_free] PASS — graph_agreement is label-free "
          f"across {n_tested} tested nodes.")


def print_neighbor_influence_summary(node_results, y_all, c2i, classes):
    """Print a structured per-node neighbor influence report.

    """
    i2c = {v: k for k, v in c2i.items()}
    class_display = {"0": "Normal Cognition (NC)", "0.5": "MCI", "1+": "AD"}

  
    feat_display = {col: label for col, label in _CLINICAL_INTERP_FEATURES}

    # Select representative examples: up to 3 correctly predicted nodes per class 
    samples_per_class = 3
    rng = np.random.default_rng(seed=42)

    # Group node indices by true class, keeping only correctly predicted ones
    class_to_nodes: dict = {cls: [] for cls in classes}
    for nidx, ndata in node_results.items():
        if not ndata.get("neighbors"):
            continue
        true_cls = i2c.get(int(y_all[nidx]), "?")
        pred_cls = ndata.get("predicted_label", "?")
        if true_cls == pred_cls and true_cls in class_to_nodes:
            class_to_nodes[true_cls].append(nidx)

    # Randomly sample up to samples_per_class per class
    selected_nodes: list = []
    for cls in classes:
        pool = class_to_nodes[cls]
        if len(pool) == 0:
            continue
        n_pick = min(samples_per_class, len(pool))
        chosen = rng.choice(pool, size=n_pick, replace=False).tolist()
        selected_nodes.extend(chosen)

    # Sort for consistent output ordering, then convert to set for fast lookup
    selected_nodes.sort()
    selected_nodes_set = set(selected_nodes)

    cls_counts = {cls: len(class_to_nodes[cls]) for cls in classes}
   
    print(f"NEIGHBOR INFLUENCE : REPRESENTATIVE EXAMPLES")
    print(f"  (correctly predicted; up to {samples_per_class} per class; "
          f"pool: {cls_counts})")
    print(f"  Selected nodes: {selected_nodes}")


    for node_idx, node_data in node_results.items():
        if node_idx not in selected_nodes_set:
            continue
        neighbors = node_data.get("neighbors", [])
        class_distribution = node_data.get("class_distribution", {})
        graph_agreement = node_data.get("graph_agreement", 0.0)
        edge_explanation = node_data.get("top_neighbor_edge_explanation", None)

        if not neighbors:
            continue

        # AUDIT ONLY — true class is shown for reference but never drives
        # class_distribution, dominant-class, or graph_agreement below.
        true_cls = i2c.get(int(y_all[node_idx]), "?")
        true_display = class_display.get(true_cls, true_cls)
        pred_cls = node_data.get("node_predicted_class", node_data.get("predicted_label", "?"))
        pred_display = class_display.get(pred_cls, pred_cls)

        top_nb = neighbors[0]
        top_nb_cls = top_nb.get("neighbor_predicted_class", "?")
        top_display = class_display.get(top_nb_cls, top_nb_cls)

        # Dominant class by total % influence
        dominant_cls = max(class_distribution, key=lambda c: class_distribution[c])
        dominant_display = class_display.get(dominant_cls, dominant_cls)
        dominant_pct = class_distribution[dominant_cls]

        # Agreement phrasing
        if graph_agreement >= 70:
            agreement_phrase = "strongly consistent"
        elif graph_agreement >= 40:
            agreement_phrase = "moderately consistent"
        else:
            agreement_phrase = "mixed"

        #  Compute biomarker data 
        top_sim_feats = []
        if edge_explanation is not None:
            feat_by_sim = sorted(
                edge_explanation.items(),
                key=lambda kv: kv[1]["contribution_to_similarity"] or 0.0,
                reverse=True
            )
            top_sim_feats = feat_by_sim[:3]

        print(f"\nPatient {node_idx}  (Predicted: {pred_display} | True: {true_display} [audit])")

        print(f"  Prediction context")
        print(f"    Dominant neighbor class : {dominant_display} ({dominant_pct:.1f}%)  [by predicted class]")
        print(f"    Graph agreement         : {graph_agreement:.1f}%  ({agreement_phrase})  [prediction-based]")
        print(f"    Strongest neighbor      : Node {top_nb['neighbor_idx']} [{top_display}]"
              f"  ({top_nb['influence_pct']:.1f}% of influence)")

        print(f"  Key biomarker similarities  (node  vs  neighbor)")
        if top_sim_feats:
            for feat_col, info in top_sim_feats:
                fname  = feat_display.get(feat_col, feat_col)
                vi     = info["patient_i_val"]
                vj     = info["patient_j_val"]
                contrib = info["contribution_to_similarity"] or 0.0
                print(f"    {fname:<32}  {vi:7.2f}  vs  {vj:7.2f}  (sim={contrib:.3f})")
        else:
            print(f"    (unavailable — pass edge feature arrays to enable)")

        print(f"  Neighborhood composition (by predicted class)")
        short_cls = {"0": "CN", "0.5": "MCI", "1+": "AD"}
        for cls in classes:
            bar_pct = class_distribution.get(cls, 0.0)
            filled  = int(bar_pct / 5)
            bar     = "█" * filled + "░" * (20 - filled)
            print(f"    {short_cls.get(cls, cls):<4} {bar}  {bar_pct:5.1f}%")

        print(f"  Top influential neighbors")
        print(f"    {'Rank':>4}  {'Node':>6}  {'Pred':<6}  {'Inf%':>6}")
        for rank, nb in enumerate(neighbors[:5], 1):
            nb_cls     = nb.get("neighbor_predicted_class", "?")
            nb_short   = short_cls.get(nb_cls, nb_cls)
            print(f"    {rank:4d}  {nb['neighbor_idx']:6d}  {nb_short:<6}  {nb['influence_pct']:5.1f}%")

        print(f"  Interpretation")
        if top_sim_feats:
            top3_names   = [feat_display.get(f, f) for f, _ in top_sim_feats]
            biomarker_str = ", ".join(top3_names)
            print(f"    Patient {node_idx} (predicted {pred_display}) is {agreement_phrase}ly aligned with its "
                  f"{dominant_display} neighbors ({dominant_pct:.1f}% influence); the strongest "
                  f"connection (Node {top_nb['neighbor_idx']}, {top_nb['influence_pct']:.1f}%) is "
                  f"driven by similar {biomarker_str}.")
        else:
            print(f"    Patient {node_idx} (predicted {pred_display}) is {agreement_phrase}ly aligned with its "
                  f"{dominant_display} neighbors ({dominant_pct:.1f}% influence).")


def _cf_ensemble_predict(models, X_cf, A_norm, device, X_edge=None, edge_index=None, A_temporal=None):
    """Run ensemble forward pass on a modified X_cf, return (pred_class_idx, avg_probs)."""
    X_dev  = X_cf.to(device)
    A_dev  = A_norm.to(device)
    Xe_dev = X_edge.to(device)    if X_edge    is not None else None
    ei_dev = edge_index.to(device) if edge_index is not None else None
    At_dev = A_temporal.to(device) if A_temporal is not None else None
    logits_list = []
    with torch.no_grad():
        for m in models:
            m.eval()
            out, _ = m(X_dev, A_dev, X_edge=Xe_dev, edge_index=ei_dev, A_temporal=At_dev)
            logits_list.append(out.cpu())
    avg_logits = torch.stack(logits_list).mean(dim=0)   # (N, C)
    probs = torch.softmax(avg_logits, dim=-1)
    return avg_logits.argmax(dim=1).numpy(), probs.numpy()


def find_counterfactual_for_node(
    models, X_tensor, A_norm_tensor,
    node_idx, current_pred_idx,
    ranked_features, modifiable_mask, feature_names,
    device,
    perturb_pcts=(0.5, 1.0, 1.5, 2.0),
    max_features=2,
    X_edge=None, edge_index=None, A_temporal=None,
):
    """Search for the smallest feature perturbation that flips node_idx's prediction.

    Only touches modifiable features (modifiable_mask), restricted to the
    top-15 by importance. Tries n_changes = 1, 2, ... up to max_features,
    testing every combination of that many features and every +/- direction
    at each perturb_pct step, and stops at the first perturbation that
    flips the predicted class. Since X_tensor is already standardized
    (mean~0, std~1), perturb_pct is just an additive shift in standard
    deviations (0.5 = half a SD, 2.0 = two SDs). Returns None if nothing
    within budget flips the prediction.
    """
    feat_name_to_idx = {name: i for i, name in enumerate(feature_names)}

    # Build ordered candidate list: modifiable features from ranked_features
    candidates = []
    for feat_name, _imp in ranked_features:
        idx = feat_name_to_idx.get(feat_name)
        if idx is not None and modifiable_mask[idx]:
            candidates.append((feat_name, idx))
    if not candidates:
        return None

    pool = candidates[:5]  # limit search space — 5 is sufficient; more causes combinatorial explosion

    for n_changes in range(1, min(max_features, len(pool)) + 1):
        for combo in itertools.combinations(pool, n_changes):
            # Try each perturb step size
            for pct in perturb_pcts:
                # Try all ± direction combinations for the n_changes features
                for directions in itertools.product((-1, 1), repeat=n_changes):
                    X_cf = X_tensor.clone()
                    for (feat_name, fidx), sign in zip(combo, directions):
                        X_cf[node_idx, fidx] = X_cf[node_idx, fidx] + sign * pct

                    preds, _ = _cf_ensemble_predict(
                        models, X_cf, A_norm_tensor, device,
                        X_edge=X_edge, edge_index=edge_index, A_temporal=A_temporal
                    )
                    new_pred = int(preds[node_idx])
                    if new_pred != current_pred_idx:
                        changes = []
                        for (feat_name, fidx), sign in zip(combo, directions):
                            changes.append({
                                "feature": feat_name,
                                "col_idx": fidx,
                                "original_std": float(X_tensor[node_idx, fidx].item()),
                                "perturbed_std": float(X_cf[node_idx, fidx].item()),
                                "direction": "+" if sign > 0 else "-",
                                "perturb_pct": pct,
                            })
                        return {
                            "node_idx": node_idx,
                            "pred_class_idx": current_pred_idx,
                            "cf_class_idx": new_pred,
                            "n_changes": n_changes,
                            "changes": changes,
                        }
    return None


def compute_counterfactuals(
    models, X_tensor, A_norm_tensor, y_all, mask,
    feature_names, feature_importance, classes, c2i, modifiable_mask, device,
    perturb_pcts=(0.5, 1.0, 1.5, 2.0),
    max_features=2,
    out_dir=None, verbose=True,
    X_edge=None, edge_index=None, A_temporal=None,
    df=None,
):
    """Compute counterfactual explanations for all test nodes.

    """
    i2c = {v: k for k, v in c2i.items()}
    node_indices = np.where(np.asarray(mask, dtype=bool))[0]

    # Pre-compute ensemble predictions for all nodes
    preds_all, _ = _cf_ensemble_predict(
        models, X_tensor, A_norm_tensor, device,
        X_edge=X_edge, edge_index=edge_index, A_temporal=A_temporal
    )

    cf_results = []
    n_found = 0

    for node_idx in node_indices:
        true_cls = i2c.get(int(y_all[node_idx]), "?")
        pred_cls_idx = int(preds_all[node_idx])
        pred_cls = i2c.get(pred_cls_idx, "?")

        # Use feature importance ranked for predicted class (drives what to perturb)
        ranked = feature_importance.get(pred_cls, [])
        if not ranked:
            # Fall back to any available class
            for cls in classes:
                ranked = feature_importance.get(cls, [])
                if ranked:
                    break

        cf = find_counterfactual_for_node(
            models, X_tensor, A_norm_tensor,
            int(node_idx), pred_cls_idx,
            ranked, modifiable_mask, feature_names, device,
            perturb_pcts=perturb_pcts, max_features=max_features,
            X_edge=X_edge, edge_index=edge_index, A_temporal=A_temporal,
        )

        if cf is not None:
            cf["true_class"] = true_cls
            cf["pred_class"] = pred_cls
            cf["cf_class"] = i2c.get(cf["cf_class_idx"], "?")
            for ch in cf["changes"]:
                ch.pop("col_idx", None)   # not needed in output
            cf_results.append(cf)
            n_found += 1
        else:
            cf_results.append({
                "node_idx": int(node_idx),
                "true_class": true_cls,
                "pred_class": pred_cls,
                "cf_class": None,
                "n_changes": None,
                "changes": [],
            })

    if verbose:
        print(f"\n  [counterfactual] {n_found}/{len(node_indices)} test nodes have a counterfactual "
              f"within {max_features} feature change(s).")

    if out_dir is not None and cf_results:
        _has_subj = df is not None and "subject_id" in df.columns
        rows = []
        for cf in cf_results:
            base = {
                "node_idx":   cf["node_idx"],
                "true_class": cf["true_class"],
                "pred_class": cf["pred_class"],
                "cf_class":   cf["cf_class"],
                "n_changes":  cf["n_changes"],
            }
            if _has_subj:
                base["subject_id"] = str(df.iloc[cf["node_idx"]]["subject_id"])
            for rank, ch in enumerate(cf["changes"], 1):
                base[f"feat_{rank}"]      = ch["feature"]
                base[f"orig_std_{rank}"]  = ch["original_std"]
                base[f"pert_std_{rank}"]  = ch["perturbed_std"]
                base[f"dir_{rank}"]       = ch["direction"]
                base[f"pct_{rank}"]       = ch["perturb_pct"]
            rows.append(base)
        pd.DataFrame(rows).to_csv(out_dir / "counterfactual_explanations.csv", index=False)
        if verbose:
            print(f"  [counterfactual] Results saved to {out_dir}/counterfactual_explanations.csv")

    if verbose:
        print_counterfactual_summary(cf_results, i2c, classes)

    return cf_results


def print_counterfactual_summary(cf_results, i2c, classes):
    feat_display = {col: label for col, label in _CLINICAL_INTERP_FEATURES}
    class_display = {"0": "Normal Cognition (NC)", "0.5": "MCI", "1+": "AD"}

    rng = np.random.default_rng(seed=42)
    samples_per_class = 3

    # Group: correctly predicted nodes that have a counterfactual
    class_to_nodes: dict = {cls: [] for cls in classes}
    for cf in cf_results:
        if cf["cf_class"] is None:
            continue
        if cf["true_class"] == cf["pred_class"] and cf["true_class"] in class_to_nodes:
            class_to_nodes[cf["true_class"]].append(cf)

    selected = []
    for cls in classes:
        pool = class_to_nodes[cls]
        if not pool:
            continue
        chosen = rng.choice(len(pool), size=min(samples_per_class, len(pool)), replace=False)
        selected.extend([pool[i] for i in chosen])
    selected.sort(key=lambda x: x["node_idx"])

    cls_counts = {cls: len(class_to_nodes[cls]) for cls in classes}
    print(f"\nCOUNTERFACTUAL EXPLANATIONS : REPRESENTATIVE EXAMPLES")
    print(f"  (correctly predicted with counterfactual found; up to {samples_per_class} per class; "
          f"pool: {cls_counts})")

    for cf in selected:
        true_disp = class_display.get(cf["true_class"], cf["true_class"])
        cf_disp   = class_display.get(cf["cf_class"],   cf["cf_class"])
        print(f"\nPatient {cf['node_idx']}  ({true_disp})")
        print(f"  Original prediction : {true_disp}")
        print(f"  Counterfactual      : {cf_disp}   ({cf['n_changes']} feature change(s))")
        print(f"  Required changes")
        for ch in cf["changes"]:
            dname = feat_display.get(ch["feature"], ch["feature"])
            pct_label = f"{ch['direction']}{ch['perturb_pct']:.2f} SD"
            direction_word = "increase" if ch["direction"] == "+" else "decrease"
            print(f"    {dname:<36}  {ch['original_std']:+.3f} -> {ch['perturbed_std']:+.3f}"
                  f"  ({pct_label})   [{direction_word}]")
        if cf["changes"]:
            top_feat = feat_display.get(cf["changes"][0]["feature"], cf["changes"][0]["feature"])
            direction_word = "increase" if cf["changes"][0]["direction"] == "+" else "decrease"
            print(f"  Interpretation")
            print(f"    Patient {cf['node_idx']} ({true_disp}) would be reclassified as {cf_disp} "
                  f"if {top_feat} {'increased' if direction_word == 'increase' else 'decreased'} "
                  f"by ~{cf['changes'][0]['perturb_pct']:.2f} standard deviations. "
                  f"This suggests {top_feat} is a critical margin feature for this patient.")


def print_metrics(split_name, res, classes, c2i=None):
    print(f"\n  === {split_name.upper()} ===")
    print(f"  Balanced Accuracy: {res['balanced_accuracy']:.4f}")
    cm = res.get("confusion_matrix", [])
    if cm:
        print("  Confusion Matrix (rows=true, cols=pred):")
        header = "        " + "  ".join(f"{c:>5}" for c in classes)
        print(header)
        for i, row in enumerate(cm):
            print(f"    {classes[i]:>3}  " + "  ".join(f"{v:5d}" for v in row))
    rep = res.get("classification_report", {})
    for c in classes:
        c_str = str(c2i[c]) if c2i is not None else str(c)
        if c_str in rep:
            p = rep[c_str].get("precision", 0)
            r = rep[c_str].get("recall", 0)
            f1 = rep[c_str].get("f1-score", 0)
            s = int(rep[c_str].get("support", 0))
            print(f"    Class {str(c):>4}: P={p:.4f}  R={r:.4f}  F1={f1:.4f}  support={s}")
    for avg_key in ["macro avg", "weighted avg"]:
        if avg_key in rep:
            p = rep[avg_key].get("precision", 0)
            r = rep[avg_key].get("recall", 0)
            f1 = rep[avg_key].get("f1-score", 0)
            label = avg_key.replace(" avg", "").title()
            print(f"    {label:>8} Avg: P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    # AUC-OvR and per-class AP
    auc_ovr = res.get("auc_ovr_macro")
    if auc_ovr is not None:
        print(f"  AUC-OvR (macro): {auc_ovr:.4f}")
    per_class_ap = res.get("per_class_ap", {})
    if per_class_ap:
        ap_parts = []
        for c in classes:
            c_str = str(c2i[c]) if c2i is not None else str(c)
            ap_val = per_class_ap.get(c_str, float('nan'))
            ap_parts.append(f"{c}={ap_val:.4f}")
        print(f"  Per-class AP: {', '.join(ap_parts)}")


# Single-Fold Pipeline

def run_single_fold(df, y_all, edge_feat_cols, classes, c2i,
                    train_mask, val_mask, test_mask,
                    args, device, out_dir=None,
                    num_seeds_override=None,
                    verbose=True, save_models=False,
                    run_explainer=False):
    
    num_seeds = num_seeds_override if num_seeds_override is not None else args.num_seeds

    #  Preprocess node features (fit on train only) 
    feat_df = df.drop(columns=["DIAGNOSIS"])
    pre, num_cols, cat_cols = build_preprocessor(feat_df)
    pre.fit(feat_df[train_mask].reset_index(drop=True))
    X_all = pre.transform(feat_df)
    X_all = X_all.toarray() if hasattr(X_all, "toarray") else np.asarray(X_all)
    if verbose:
        print(f"[features] {X_all.shape[1]} node features after preprocessing")

    #  Edge weight features: impute, standardize, compute distances 
    edge_raw = df[edge_feat_cols].copy()
    for col in edge_feat_cols:
        edge_raw[col] = pd.to_numeric(edge_raw[col], errors='coerce')
    train_medians = edge_raw[train_mask].median()
    edge_raw = edge_raw.fillna(train_medians)
    edge_scaler = StandardScaler()
    edge_scaler.fit(edge_raw[train_mask].values)
    X_edge = edge_scaler.transform(edge_raw.values).astype(np.float64)
    if verbose:
        print(f"\n[edge] Computing pairwise similarity ({X_edge.shape[0]} nodes, kernel={args.kernel})...")

    # Full similarity matrix
    similarity_full = compute_similarity_matrix(X_edge, kernel=args.kernel, rbf_gamma=args.rbf_gamma)

    # Temporal decay on same-subject edges
    subject_ids = df['subject_id'].values
    if args.temporal_lambda > 0 and not args.no_temporal_edges and "visit_month" in df.columns:
        similarity_full = apply_temporal_decay(
            similarity_full, df["visit_month"].values, subject_ids, args.temporal_lambda
        )
    elif verbose:
        if args.no_temporal_edges:
            print(f"[temporal] Edge decay disabled (--no_temporal_edges)")
        elif args.temporal_lambda <= 0:
            print(f"[temporal] Edge decay disabled (temporal_lambda={args.temporal_lambda})")

    # Top-K sparsification (with optional subject-aware constraints)
    unique_subjects = len(set(subject_ids))
    has_multi_visit = unique_subjects < len(subject_ids)

    if not args.no_subject_constraint and has_multi_visit:
        if verbose:
            print(f"\n[subject-constraint] mode=cap, max_same_subject={args.max_same_subject}")
            print(f"[subject-constraint] {len(subject_ids)} nodes from {unique_subjects} subjects "
                  f"({len(subject_ids) / unique_subjects:.1f} visits/subject avg)")
            A_baseline = topk_sparsify(similarity_full, k=args.topk)
            print_subject_neighbor_stats(A_baseline, subject_ids, prefix="[subject-pre]")
        A, subj_stats = topk_sparsify_subject_aware(
            similarity_full, k=args.topk, subject_ids=subject_ids,
            max_same_subject=args.max_same_subject
        )
        if verbose:
            print_subject_neighbor_stats(A, subject_ids, prefix="[subject-post]")
            print(f"[subject-constraint] Same-subj edges kept: {subj_stats['same_subject_kept']}, "
                  f"blocked: {subj_stats['same_subject_blocked']}")
    else:
        if verbose:
            if args.no_subject_constraint:
                print(f"\n[subject-constraint] Disabled (--no_subject_constraint)")
            elif not has_multi_visit:
                print(f"\n[subject-constraint] All subjects unique — constraint not needed")
        A = topk_sparsify(similarity_full, k=args.topk)

    # Apply similarity threshold
    if args.sim_threshold > 0:
        before_edges = int(np.count_nonzero(A))
        A[A < args.sim_threshold] = 0.0
        after_edges = int(np.count_nonzero(A))
        if verbose:
            print(f"[edge] Threshold={args.sim_threshold}: pruned {before_edges - after_edges} weak edges")

    if verbose:
        n_total_possible = A.shape[0] * (A.shape[0] - 1)
        n_edges = int(np.count_nonzero(A))
        sparsity = 1.0 - (n_edges / n_total_possible) if n_total_possible > 0 else 0
        nonzero_sim = A[A > 0]
        degrees = (A > 0).sum(axis=1)
        print(f"[edge] Adjacency matrix: {A.shape}")
        print(f"[edge] Top-K: {args.topk}")
        print(f"[edge] Edges kept: {n_edges}/{n_total_possible} ({100*(1-sparsity):.1f}%, sparsity={100*sparsity:.1f}%)")
        print(f"[edge] Degree: min={degrees.min()}, max={degrees.max()}, mean={degrees.mean():.1f}")
        if len(nonzero_sim) > 0:
            print(f"[edge] Similarity (non-zero): [{nonzero_sim.min():.4f}, {nonzero_sim.max():.4f}], mean={nonzero_sim.mean():.4f}")
        n_isolated = int(np.sum(degrees == 0))
        if n_isolated > 0:
            print(f"[edge] WARNING: {n_isolated} isolated nodes!")

    # Graph structure analysis 
    if args.analyze_graph and out_dir is not None:
        analysis_gamma = args.rbf_gamma
        if args.kernel == 'rbf' and analysis_gamma is None:
            upper_dists = pdist(X_edge, metric='euclidean')
            median_dist = np.median(upper_dists)
            analysis_gamma = 1.0 / (2.0 * median_dist ** 2 + 1e-8)
        graph_analysis = analyze_graph_structure(
            A, edge_raw.values, X_edge, y_all, edge_feat_cols,
            classes, c2i, kernel=args.kernel, rbf_gamma=analysis_gamma
        )
        csv_rows = []
        for cls_label in classes:
            cls_idx = c2i[cls_label]
            cls_nodes = np.where(y_all == cls_idx)[0]
            pairs = []
            for ni in cls_nodes:
                for nj in cls_nodes:
                    if nj > ni and A[ni, nj] > 0:
                        pairs.append((ni, nj, A[ni, nj]))
            pairs.sort(key=lambda x: x[2], reverse=True)
            for ni, nj, sim in pairs[:20]:
                edge_info = explain_edge(ni, nj, edge_raw.values, X_edge, edge_feat_cols,
                                         kernel=args.kernel, rbf_gamma=analysis_gamma)
                for feat_name, info in edge_info.items():
                    csv_rows.append({
                        "class": cls_label, "patient_i_idx": int(ni), "patient_j_idx": int(nj),
                        "similarity": float(sim), "feature": feat_name,
                        "patient_i_val": info["patient_i_val"], "patient_j_val": info["patient_j_val"],
                        "abs_diff_std": info["abs_diff_std"], "contribution": info["contribution_to_similarity"],
                    })
        edge_exp_path = out_dir / "edge_explanations.csv"
        pd.DataFrame(csv_rows).to_csv(edge_exp_path, index=False)
        print(f"\n[analyze] Edge explanations saved to {edge_exp_path}")
        print(f"[analyze] Graph structure analysis saved to {out_dir / 'graph_analysis.json'}")
        with open(out_dir / "graph_analysis.json", "w") as f:
            json.dump(graph_analysis, f, indent=2)

    # Normalize adjacency: A_hat = A + I, then D^{-1/2} A_hat D^{-1/2}
    N = A.shape[0]
    A_hat = A + np.eye(N)
    A_norm = symmetric_normalize(A_hat)
    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    A_norm_tensor = torch.tensor(A_norm, dtype=torch.float32)

    # Temporal adjacency (optional)
    A_temporal_norm_tensor = None
    if args.use_temporal_branch:
        if "visit_month" not in df.columns:
            if verbose:
                print("[temporal-branch] WARNING: visit_month not available, disabling temporal branch")
            args = copy.deepcopy(args)
            args.use_temporal_branch = False
        else:
            temporal_lam = args.temporal_lambda if args.temporal_lambda > 0 else 0.05
            A_temporal_raw, A_temporal_norm_mat = build_temporal_adjacency(
                df["visit_month"].values, df["subject_id"].values,
                temporal_lambda=temporal_lam, topk=args.temporal_topk
            )
            A_temporal_norm_tensor = torch.tensor(A_temporal_norm_mat, dtype=torch.float32)
            if verbose:
                t_degrees = (A_temporal_raw > 0).sum(axis=1)
                t_edges = int(np.count_nonzero(A_temporal_raw))
                print(f"\n[temporal-branch] Temporal graph built (same-subject edges only)")
                print(f"[temporal-branch] Edges: {t_edges}, lambda={temporal_lam}, topk={args.temporal_topk}")
                print(f"[temporal-branch] Degree: min={t_degrees.min()}, max={t_degrees.max()}, "
                      f"mean={t_degrees.mean():.1f}")

    X_edge_tensor = None
    edge_index_tensor = None

    #  Class weights & loss 
    train_labels = y_all[train_mask]
    cnts = np.bincount(train_labels, minlength=len(classes))
    inv = 1.0 / np.clip(cnts, 1, None)
    wts = (inv ** args.class_weight_exp) / (inv ** args.class_weight_exp).sum() * len(classes)
    w = torch.tensor(wts, dtype=torch.float32, device=device)
    if verbose:
        print(f"\n[loss] FocalLoss(gamma={args.focal_gamma}, smoothing={args.label_smoothing})")
        print(f"[loss] Class weights (exp={args.class_weight_exp}): {dict(zip(classes, np.round(wts, 3)))}")

    #  Multi-seed ensemble training 
    seeds = [args.seed + i for i in range(num_seeds)]
    ensemble_models: List[SpectralGCN] = []
    all_best_info = []
    _load_checkpoints = getattr(args, "load_checkpoints", False)

    if _load_checkpoints:
        # Skip training: load saved model_spectral_gcn_seed*.pt checkpoints 
        # Preprocessing, split, and graph construction above are unchanged, so the
        # architecture built here must match training exactly for state_dict to load.
        if out_dir is None:
            raise ValueError("--load_checkpoints requires out_dir to locate checkpoint files")
        if verbose:
            print(f"\n[checkpoint] --load_checkpoints set: loading saved models instead of training")
        for run_i, seed in enumerate(seeds):
            ckpt_path = Path(out_dir) / f"model_spectral_gcn_seed{seed}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"--load_checkpoints set but checkpoint not found: {ckpt_path}"
                )
            model = SpectralGCN(
                in_dim=X_all.shape[1], hidden_dim=args.hidden, num_classes=len(classes),
                num_gcn_layers=args.num_gcn_layers, dropout=args.dropout, use_bn=args.use_bn,
                drop_edge_rate=args.drop_edge_rate, feat_mask_rate=args.feat_mask_rate,
                use_temporal_branch=args.use_temporal_branch,
                temporal_hidden_dim=args.temporal_branch_hidden,
                temporal_num_layers=args.temporal_branch_layers,
                use_mlp_branch=args.use_mlp_branch,
                use_jk=args.use_jk,
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            ensemble_models.append(model)
            all_best_info.append({"seed": seed, "epoch": None, "metric": None, "loss_history": {}})
            if verbose:
                print(f"  [checkpoint] Loaded {ckpt_path}")
    else:
        for run_i, seed in enumerate(seeds):
            set_seed(seed)
            if verbose:

                print(f"TRAINING RUN {run_i + 1}/{num_seeds} (seed={seed}, max {args.epochs} epochs, "
                      f"patience={args.patience}, monitor={args.monitor})")


            model = SpectralGCN(
                in_dim=X_all.shape[1], hidden_dim=args.hidden, num_classes=len(classes),
                num_gcn_layers=args.num_gcn_layers, dropout=args.dropout, use_bn=args.use_bn,
                drop_edge_rate=args.drop_edge_rate, feat_mask_rate=args.feat_mask_rate,
                use_temporal_branch=args.use_temporal_branch,
                temporal_hidden_dim=args.temporal_branch_hidden,
                temporal_num_layers=args.temporal_branch_layers,
                use_mlp_branch=args.use_mlp_branch,
                use_jk=args.use_jk,
            ).to(device)

            if run_i == 0 and verbose:
                total_params = sum(p.numel() for p in model.parameters())
                jk_dim = args.hidden * args.num_gcn_layers
                print(f"[model] SpectralGCN: {X_all.shape[1]} -> {args.hidden} (x{args.num_gcn_layers} GCN) "
                      f"-> JK({jk_dim}) -> MLP -> {len(classes)}")
                print(f"[model] BN={args.use_bn}, Dropout={args.dropout}, DropEdge={args.drop_edge_rate}, "
                      f"FeatMask={args.feat_mask_rate}, Residual=True")
                print(f"[model] Parameters: {total_params:,}")
                if args.use_temporal_branch:
                    t_jk = args.temporal_branch_hidden * args.temporal_branch_layers
                    print(f"[model] Temporal GCN branch: {X_all.shape[1]} -> {args.temporal_branch_hidden} "
                          f"(x{args.temporal_branch_layers}) -> JK({t_jk})")
                print(f"[model] Ensemble seeds: {seeds}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            criterion = FocalLoss(alpha=w, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
            scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.warmup_epochs,
                                              total_epochs=args.epochs, min_lr=args.min_lr)

            model, best = train_model(
                model, X_tensor, A_norm_tensor, y_all,
                train_mask, val_mask, optimizer, criterion,
                scheduler=scheduler, max_epochs=args.epochs, patience=args.patience,
                device=device, verbose=verbose,
                monitor=args.monitor,
                X_edge=X_edge_tensor, edge_index=edge_index_tensor,
                A_temporal=A_temporal_norm_tensor
            )

            metric_name = "val_bal" if args.monitor == "val_bal" else "val_loss"
            if verbose:
                print(f"  Run {run_i + 1}: Best epoch={best['epoch']}, Best {metric_name}={best['metric']:.4f}")

            ensemble_models.append(model)
            all_best_info.append({"seed": seed, "epoch": best["epoch"], "metric": best["metric"],
                                  "loss_history": best.get("loss_history", {})})

    # Plot loss curves 
    if out_dir is not None and not _load_checkpoints:
        plot_loss_curves(all_best_info, out_dir)

    # Ensemble evaluation 
    if verbose:
        print(f"\nENSEMBLE EVALUATION ({num_seeds} seeds)")

    res_tr = ensemble_evaluate(ensemble_models, X_tensor, A_norm_tensor, y_all, train_mask, device,
                               X_edge=X_edge_tensor, edge_index=edge_index_tensor,
                               A_temporal=A_temporal_norm_tensor)
    res_va = ensemble_evaluate(ensemble_models, X_tensor, A_norm_tensor, y_all, val_mask, device,
                               X_edge=X_edge_tensor, edge_index=edge_index_tensor,
                               A_temporal=A_temporal_norm_tensor)
    has_test = np.asarray(test_mask, dtype=bool).any()
    if has_test:
        res_te = ensemble_evaluate(ensemble_models, X_tensor, A_norm_tensor, y_all, test_mask, device,
                                   X_edge=X_edge_tensor, edge_index=edge_index_tensor,
                                   A_temporal=A_temporal_norm_tensor)
    else:
        res_te = {"balanced_accuracy": float('nan'), "confusion_matrix": [],
                  "classification_report": {}, "probs": np.empty((0, len(classes)))}

    if verbose:
        print_metrics("Train", res_tr, classes, c2i)
        print_metrics("Validation", res_va, classes, c2i)
        if has_test:
            print_metrics("Test", res_te, classes, c2i)

    # Save uncertainty estimates CSV 
    if has_test and "uncertainty" in res_te:
        _save_uncertainty_csv(res_te["uncertainty"], y_all, test_mask, classes, out_dir, df=df)
        if verbose:
            print_uncertainty_summary(res_te["uncertainty"], y_all, test_mask, classes, c2i)

    #  Per-seed metrics (for confidence intervals) 
    per_seed_results = []
    seed_bal_ci = {}
    seed_per_class_recalls = {}

    if has_test:
        for i, m in enumerate(ensemble_models):
            ind_res = evaluate(m, X_tensor, A_norm_tensor, y_all, test_mask, device=device,
                               X_edge=X_edge_tensor, edge_index=edge_index_tensor, A_temporal=A_temporal_norm_tensor)
            per_seed_results.append(ind_res)
            if verbose:
                if i == 0:
                    print(f"\n  --- Individual model test results ---")
                print(f"    Seed {seeds[i]}: Test bal_acc = {ind_res['balanced_accuracy']:.4f}")

        # Compute seed-level confidence intervals
        seed_bal_accs = [r['balanced_accuracy'] for r in per_seed_results]
        seed_bal_ci = compute_ci(seed_bal_accs)

        for cls in classes:
            cls_key = str(c2i[cls])
            recalls = [r['classification_report'].get(cls_key, {}).get('recall', 0.0)
                       for r in per_seed_results]
            seed_per_class_recalls[cls] = compute_ci(recalls, clip=(0, 1))

        if verbose and num_seeds > 1:
            print(f"\n  --- Confidence Intervals (over {num_seeds} seeds) ---")
            print(f"  Balanced Accuracy: {seed_bal_ci['mean']:.4f} +/- {seed_bal_ci['std']:.4f}  "
                  f"  95% CI: [{seed_bal_ci['ci_lower']:.4f}, {seed_bal_ci['ci_upper']:.4f}]")
            for cls in classes:
                ci = seed_per_class_recalls[cls]
                print(f"  {cls} Recall:          {ci['mean']:.4f} +/- {ci['std']:.4f}  "
                      f"  95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    result = {
        "train": res_tr, "val": res_va, "test": res_te,
        "models": ensemble_models, "best_info": all_best_info,
        "per_seed_results": per_seed_results,
        "seed_confidence_intervals": {
            "balanced_accuracy": seed_bal_ci,
            "per_class_recall": seed_per_class_recalls,
        },
        "pre": pre, "num_cols": num_cols, "cat_cols": cat_cols, "X_all": X_all,
        "X_edge_tensor": X_edge_tensor, "edge_index_tensor": edge_index_tensor,
        "A_temporal_norm_tensor": A_temporal_norm_tensor,
        "X_tensor": X_tensor, "A_norm_tensor": A_norm_tensor,
    }

    # Reset the RNG once, outside the training/checkpoint branches, so GNNExplainer and
    # counterfactual search are reproducible from here regardless of whether the ensemble
    # was just trained or loaded from checkpoint. This intentionally changes torch's RNG
    # state relative to wherever the original training run left it, so GNNExplainer /
    # counterfactual outputs from a fresh end-to-end run may differ from previously
    # generated ones. Graph agreement itself is unaffected: compute_neighbor_influence
    # runs under model.eval() with dropout and DropEdge disabled, so it has no RNG dependence.
    set_seed(args.seed)

    # Optional: GNNExplainer 
    _cf_feature_importance = None   # populated below if run_explainer; reused by counterfactual block
    _cf_feature_names = None        # likewise
    if run_explainer and out_dir is not None:
        feature_names = get_feature_names(pre, num_cols, cat_cols)
        if len(feature_names) != X_all.shape[1]:
            feature_names = [f"feature_{i}" for i in range(X_all.shape[1])]
        class_importance, patient_masks, _expl_ensemble_pred = compute_feature_importance(
            ensemble_models, X_tensor, A_norm_tensor, y_all, test_mask,
            feature_names, classes, c2i, device, top_k=25, epochs=200,
            X_edge=X_edge_tensor, edge_index=edge_index_tensor,
            A_temporal=A_temporal_norm_tensor
        )
        result["feature_importance"] = class_importance
        _cf_feature_importance = class_importance
        _cf_feature_names = feature_names
        if verbose:
            for cls_label in classes:
                feats = class_importance.get(cls_label, [])
                print(f"\n  --- Top {len(feats)} features for class {cls_label} ---")
                for rank, (fname, score) in enumerate(feats, 1):
                    print(f"    {rank:2d}. {fname:<40s}  importance={score:.6f}")
        rows = []
        for cls_label in classes:
            for rank, (fname, score) in enumerate(class_importance.get(cls_label, []), 1):
                rows.append({"class": cls_label, "rank": rank, "feature": fname, "importance": score})
        imp_df = pd.DataFrame(rows)
        imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
        if verbose:
            print(f"\n  [explainer] Feature importance saved to {out_dir}/feature_importance.csv")

        # Patient-level GNNExplainer evidence (reuses masks from above)
        if np.asarray(test_mask, dtype=bool).any() and patient_masks:
            patient_evidence = build_patient_evidence(
                patient_masks, _expl_ensemble_pred, y_all,
                feature_names, c2i, feat_df, top_k_patient=3
            )
            result["patient_evidence"] = patient_evidence
            _save_patient_evidence_csv(patient_evidence, out_dir)
            if verbose:
                print_patient_evidence_summary(patient_evidence, classes)
                print(f"\n  [explainer] Patient evidence saved to {out_dir}/patient_evidence.csv")

    # Optional: Neighbor Influence Explanation 
    if run_explainer and out_dir is not None and np.asarray(test_mask, dtype=bool).any():
        # Compute rbf_gamma used during graph construction (needed by explain_edge)
        _ni_rbf_gamma = args.rbf_gamma
        if args.kernel == 'rbf' and _ni_rbf_gamma is None:
            _upper_dists = pdist(X_edge, metric='euclidean')
            _median_dist = np.median(_upper_dists)
            _ni_rbf_gamma = 1.0 / (2.0 * _median_dist ** 2 + 1e-8)
        node_results = explain_neighbor_influences(
            ensemble_models, X_tensor, A_norm_tensor, y_all, test_mask,
            classes, c2i, device, top_k=10, out_dir=out_dir, verbose=verbose,
            X_edge=X_edge_tensor, edge_index=edge_index_tensor,
            A_temporal=A_temporal_norm_tensor,
            X_edge_raw=edge_raw.values,
            X_edge_std=X_edge,
            edge_feat_cols=edge_feat_cols,
            kernel=args.kernel,
            rbf_gamma=_ni_rbf_gamma,
            df=df,
        )
        print_neighbor_influence_summary(node_results, y_all, c2i, classes)
        result["neighbor_influence"] = node_results

        if getattr(args, "test_label_free", False):
            test_agreement_is_label_free(
                ensemble_models, X_tensor, A_norm_tensor, y_all, test_mask,
                classes, c2i, device,
                X_edge=X_edge_tensor, edge_index=edge_index_tensor,
                A_temporal=A_temporal_norm_tensor,
                X_edge_raw=edge_raw.values, X_edge_std=X_edge,
                edge_feat_cols=edge_feat_cols, kernel=args.kernel, rbf_gamma=_ni_rbf_gamma,
                df=df,
            )
            result["label_free_test_passed"] = True
            return result  # exit cleanly — skip counterfactuals/save-models/report stages
    elif getattr(args, "test_label_free", False):
        raise ValueError(
            "--test_label_free requires explanation mode enabled (drop --no_explain) "
            "and a nonempty test_mask to explain."
        )

    # Optional: Counterfactual Explanations 
    if getattr(args, "counterfactual", False) and not args.no_explain and np.asarray(test_mask, dtype=bool).any():
        # Ensure feature_names is available (may already be set from GNNExplainer block above)
        if _cf_feature_names is None:
            _cf_feature_names = get_feature_names(pre, num_cols, cat_cols)
            if len(_cf_feature_names) != X_all.shape[1]:
                _cf_feature_names = [f"feature_{i}" for i in range(X_all.shape[1])]
        # Compute feature importance if GNNExplainer was not run
        if _cf_feature_importance is None:
            _cf_feature_importance, _, _ = compute_feature_importance(
                ensemble_models, X_tensor, A_norm_tensor, y_all, test_mask,
                _cf_feature_names, classes, c2i, device, top_k=25, epochs=200,
                X_edge=X_edge_tensor, edge_index=edge_index_tensor,
                A_temporal=A_temporal_norm_tensor
            )
        modifiable_mask = _get_modifiable_mask(_cf_feature_names)
        cf_results = compute_counterfactuals(
            ensemble_models, X_tensor, A_norm_tensor, y_all, test_mask,
            _cf_feature_names, _cf_feature_importance, classes, c2i, modifiable_mask, device,
            perturb_pcts=tuple(args.cf_perturb_steps),
            max_features=args.cf_max_features,
            out_dir=out_dir, verbose=verbose,
            X_edge=X_edge_tensor, edge_index=edge_index_tensor,
            A_temporal=A_temporal_norm_tensor,
            df=df,
        )
        result["counterfactuals"] = cf_results

  
    
    if save_models and out_dir is not None and not _load_checkpoints:
        for i, m in enumerate(ensemble_models):
            torch.save(m.state_dict(), out_dir / f"model_spectral_gcn_seed{seeds[i]}.pt")

    return result


# Inductive Sensitivity Analysis Pipeline
#
# Everything below is additive: run_single_fold (above) and the transductive
# main() flow are never called by, or modified for, this path. Reached only
# via --eval_mode inductive (default is "transductive", which is untouched).

def verify_inductive_graphs(A_train, A_train_val, A_train_test, n_tr, n_val, n_te,
                            probe_model, device,
                            X_tr_tensor, A_train_norm_tensor,
                            X_train_val_tensor, A_train_val_norm_tensor,
                            X_train_test_tensor, A_train_test_norm_tensor,
                            subject_id_train, subject_id_val, subject_id_test,
                            AT_train=None, AT_train_val=None, AT_train_test=None,
                            vm_train=None, sid_train_arr=None,
                            vm_train_val=None, sid_train_val_arr=None,
                            vm_train_test=None, sid_train_test_arr=None,
                            A_temporal_train_norm_tensor=None,
                            A_temporal_train_val_norm_tensor=None,
                            A_temporal_train_test_norm_tensor=None):
    """Sanity-checks inductive graph construction before any real training
    happens. Uses a throwaway probe model (any weights work — check 4 is a
    structural property of directed message-passing + row-normalization
    under eval(), not of learned weights), so this can run at the very start
    of run_single_fold_inductive. Raises AssertionError on failure: if a
    check fails, the graph construction is wrong and nothing downstream can
    be trusted.

    Returns a dict of check outcomes / measured values for inductive_config.json.
    """
    # Top-left block is an exact copy of A_train, not a rebuild.
    assert np.array_equal(A_train_val[:n_tr, :n_tr], A_train), \
        "Inductive check 1 failed: A_train_val top-left block is not an exact copy of A_train!"
    assert np.array_equal(A_train_test[:n_tr, :n_tr], A_train), \
        "Inductive check 1 failed: A_train_test top-left block is not an exact copy of A_train!"

    # Training nodes never receive edges from held-out nodes.
    assert not np.any(A_train_val[:n_tr, n_tr:]), \
        "Inductive check 2 failed: A_train_val[:n_tr, n_tr:] is not all zero!"
    assert not np.any(A_train_test[:n_tr, n_tr:]), \
        "Inductive check 2 failed: A_train_test[:n_tr, n_tr:] is not all zero!"

    # Held-out nodes never see each other in the population graph
    # (post self-loop, the new-node block must be diagonal).
    A_hat_train_val = A_train_val + np.eye(n_tr + n_val)
    A_hat_train_test = A_train_test + np.eye(n_tr + n_te)
    val_block = A_hat_train_val[n_tr:, n_tr:]
    off_diag_val = ~np.eye(n_val, dtype=bool)
    assert not np.any(val_block[off_diag_val]), \
        "Inductive check 3 failed: A_train_val new-node block is not diagonal!"
    test_block = A_hat_train_test[n_tr:, n_tr:]
    off_diag_test = ~np.eye(n_te, dtype=bool)
    assert not np.any(test_block[off_diag_test]), \
        "Inductive check 3 failed: A_train_test new-node block is not diagonal!"

    #  The real test: under eval(), training-block logits must match
    # across A_train, A_train_val, and A_train_test — this is what actually
    # proves held-out nodes cannot influence training-node predictions.
    probe_model.eval()
    with torch.no_grad():
        logits_train, _ = probe_model(
            X_tr_tensor.to(device), A_train_norm_tensor.to(device),
            A_temporal=A_temporal_train_norm_tensor.to(device) if A_temporal_train_norm_tensor is not None else None)
        logits_tv, _ = probe_model(
            X_train_val_tensor.to(device), A_train_val_norm_tensor.to(device),
            A_temporal=A_temporal_train_val_norm_tensor.to(device) if A_temporal_train_val_norm_tensor is not None else None)
        logits_tt, _ = probe_model(
            X_train_test_tensor.to(device), A_train_test_norm_tensor.to(device),
            A_temporal=A_temporal_train_test_norm_tensor.to(device) if A_temporal_train_test_norm_tensor is not None else None)
    max_diff_val = float((logits_train - logits_tv[:n_tr]).abs().max().item())
    max_diff_test = float((logits_train - logits_tt[:n_tr]).abs().max().item())
    assert torch.allclose(logits_train, logits_tv[:n_tr], atol=1e-5), \
        f"Inductive check 4 failed: train-block logits differ between G_train and G_train_val (max abs diff={max_diff_val:.3e})!"
    assert torch.allclose(logits_train, logits_tt[:n_tr], atol=1e-5), \
        f"Inductive check 4 failed: train-block logits differ between G_train and G_train_test (max abs diff={max_diff_test:.3e})!"

    #  Every temporal edge is same-subject and non-anticipating (t_j <= t_i).
    # The temporal graphs (AT_*) are same-subject-only by construction, so
    # every nonzero entry must be same-subject; the population graphs (A_*)
    # are mostly cross-subject, so only the same-subject subset of their
    # nonzero entries is checked for causality.
    def _check_causal(AT, vm, sid, label):
        if AT is None:
            return
        rows, cols = np.nonzero(AT)
        if len(rows) == 0:
            return
        assert np.all(np.asarray(sid)[rows] == np.asarray(sid)[cols]), \
            f"Inductive check 5 failed: {label} has a temporal edge across subjects!"
        assert np.all(np.asarray(vm)[rows] >= np.asarray(vm)[cols]), \
            f"Inductive check 5 failed: {label} has a temporal edge violating causality (t_j <= t_i)!"

    _check_causal(AT_train, vm_train, sid_train_arr, "AT_train")
    _check_causal(AT_train_val, vm_train_val, sid_train_val_arr, "AT_train_val")
    _check_causal(AT_train_test, vm_train_test, sid_train_test_arr, "AT_train_test")

    def _check_population_causal(A, vm, sid, label):
        if vm is None:
            return
        rows, cols = np.nonzero(A)
        if len(rows) == 0:
            return
        sid_arr = np.asarray(sid)
        vm_arr = np.asarray(vm)
        same_subject_mask = sid_arr[rows] == sid_arr[cols]
        if not np.any(same_subject_mask):
            return
        r, c = rows[same_subject_mask], cols[same_subject_mask]
        assert np.all(vm_arr[r] >= vm_arr[c]), \
            f"Inductive check 5 failed: {label} has a same-subject population edge violating causality (t_j <= t_i)!"

    _check_population_causal(A_train, vm_train, sid_train_arr, "A_train")
    _check_population_causal(A_train_val, vm_train_val, sid_train_val_arr, "A_train_val")
    _check_population_causal(A_train_test, vm_train_test, sid_train_test_arr, "A_train_test")

    # Train/val/test subjects are disjoint.
    set_tr, set_val, set_te = set(subject_id_train), set(subject_id_val), set(subject_id_test)
    assert set_tr.isdisjoint(set_val), "Inductive check 6 failed: train/val subject leakage!"
    assert set_tr.isdisjoint(set_te), "Inductive check 6 failed: train/test subject leakage!"
    assert set_val.isdisjoint(set_te), "Inductive check 6 failed: val/test subject leakage!"

    return {
        "check1_block_copy": True,
        "check2_top_right_zero": True,
        "check3_new_node_block_diagonal": True,
        "check4_max_abs_logit_diff_train_val": max_diff_val,
        "check4_max_abs_logit_diff_train_test": max_diff_test,
        "check5_temporal_causal_valid": True,
        "check5_population_causal_valid": True,
        "check6_subject_disjoint": True,
    }


def run_single_fold_inductive(df, y_all, edge_feat_cols, classes, c2i,
                              train_mask, val_mask, test_mask,
                              args, device, out_dir=None,
                              verbose=True):
    """Run the inductive sensitivity analysis using training-fitted graph settings.

Validation and test nodes only receive messages from training nodes.
This path skips explanation and counterfactual steps and saves the
inductive results and configuration files.
    """
    num_seeds = args.num_seeds

    idx_tr = np.where(np.asarray(train_mask, dtype=bool))[0]
    idx_val = np.where(np.asarray(val_mask, dtype=bool))[0]
    idx_te = np.where(np.asarray(test_mask, dtype=bool))[0]
    n_tr, n_val, n_te = len(idx_tr), len(idx_val), len(idx_te)

    # --- Node features: fit preprocessor on train only, transform (row-independent) ---
    feat_df = df.drop(columns=["DIAGNOSIS"])
    pre, num_cols, cat_cols = build_preprocessor(feat_df)
    pre.fit(feat_df.iloc[idx_tr].reset_index(drop=True))
    X_all = pre.transform(feat_df)
    X_all = X_all.toarray() if hasattr(X_all, "toarray") else np.asarray(X_all)
    X_tr_np, X_val_np, X_te_np = X_all[idx_tr], X_all[idx_val], X_all[idx_te]
    if verbose:
        print(f"[inductive/features] {X_all.shape[1]} node features after preprocessing")

    #  Edge weight features: impute/standardize on train only, transform (row-independent) 
    edge_raw = df[edge_feat_cols].copy()
    for col in edge_feat_cols:
        edge_raw[col] = pd.to_numeric(edge_raw[col], errors='coerce')
    train_medians = edge_raw.iloc[idx_tr].median()
    edge_raw = edge_raw.fillna(train_medians)
    edge_scaler = StandardScaler()
    edge_scaler.fit(edge_raw.iloc[idx_tr].values)
    X_edge_all = edge_scaler.transform(edge_raw.values).astype(np.float64)
    X_edge_tr, X_edge_val, X_edge_te = X_edge_all[idx_tr], X_edge_all[idx_val], X_edge_all[idx_te]

    subject_ids_all = df['subject_id'].values
    sid_tr, sid_val, sid_te = subject_ids_all[idx_tr], subject_ids_all[idx_val], subject_ids_all[idx_te]

    # Freeze kernel parameters from the training block only 
    frozen_gamma = args.rbf_gamma
    frozen_max_dist = None
    train_dists = pdist(X_edge_tr, metric='euclidean')
    if args.kernel == 'rbf':
        if frozen_gamma is None:
            median_dist_tr = np.median(train_dists)
            frozen_gamma = 1.0 / (2.0 * median_dist_tr ** 2 + 1e-8)
    else:
        frozen_max_dist = float(train_dists.max()) if len(train_dists) > 0 else 0.0
    if verbose:
        print(f"[inductive/kernel] kernel={args.kernel}, frozen gamma={frozen_gamma}, "
              f"frozen max_dist={frozen_max_dist} (train block only, n_tr={n_tr})")

    # --- Population similarity graph ---
    sim_train = compute_similarity_matrix(X_edge_tr, kernel=args.kernel, rbf_gamma=frozen_gamma)
    sim_val_train = compute_similarity_cross(X_edge_val, X_edge_tr, kernel=args.kernel,
                                             gamma=frozen_gamma, max_dist=frozen_max_dist)
    sim_test_train = compute_similarity_cross(X_edge_te, X_edge_tr, kernel=args.kernel,
                                              gamma=frozen_gamma, max_dist=frozen_max_dist)

    has_visit_month = "visit_month" in df.columns
    vm_all = df["visit_month"].values if has_visit_month else None

    # Per-block visit_month/subject_id arrays, computed unconditionally
    # (whenever has_visit_month) since they're needed by three consumers:
    # the causal population decay just below, the temporal GCN branch
    # further down, and verify_inductive_graphs' causality checks (population
    # + temporal graphs) — the population check must run even when
    # --no_temporal_branch is set, since population same-subject edges exist
    # independent of that flag.
    vm_tr_arr = vm_all[idx_tr] if has_visit_month else None
    vm_val_arr = vm_all[idx_val] if has_visit_month else None
    vm_te_arr = vm_all[idx_te] if has_visit_month else None
    vm_train_val_arr = np.concatenate([vm_tr_arr, vm_val_arr]) if has_visit_month else None
    vm_train_test_arr = np.concatenate([vm_tr_arr, vm_te_arr]) if has_visit_month else None
    sid_train_val_arr = np.concatenate([sid_tr, sid_val])
    sid_train_test_arr = np.concatenate([sid_tr, sid_te])

    apply_temporal = (args.temporal_lambda > 0 and not args.no_temporal_edges and has_visit_month)
    temporal_causal_stats = None
    if apply_temporal:
        sim_train, temporal_causal_stats = apply_temporal_decay_causal(
            sim_train, vm_tr_arr, sid_tr, args.temporal_lambda)
        # Cross-block: subjects are disjoint across splits (train vs val, train
        # vs test), so same_subject is never True here — apply_temporal_decay_cross's
        # symmetric |t_i-t_j| formula is safe to keep as-is; there is no
        # same-subject "own later visit" case across a split boundary to guard against.
        sim_val_train = apply_temporal_decay_cross(sim_val_train, vm_val_arr, sid_val, vm_tr_arr, sid_tr, args.temporal_lambda)
        sim_test_train = apply_temporal_decay_cross(sim_test_train, vm_te_arr, sid_te, vm_tr_arr, sid_tr, args.temporal_lambda)
    elif verbose:
        print(f"[inductive/temporal] Edge decay disabled")

    unique_subjects_tr = len(set(sid_tr))
    has_multi_visit = unique_subjects_tr < n_tr
    use_subject_constraint = (not args.no_subject_constraint) and has_multi_visit

    if use_subject_constraint:
        A_train, subj_stats_train = topk_sparsify_subject_aware_directed(
            sim_train, k=args.topk, subject_ids=sid_tr, max_same_subject=args.max_same_subject)
        A_val_train, subj_stats_val = topk_sparsify_cross_subject_aware(
            sim_val_train, k=args.topk, subject_ids_new=sid_val, subject_ids_train=sid_tr,
            max_same_subject=args.max_same_subject)
        A_test_train, subj_stats_test = topk_sparsify_cross_subject_aware(
            sim_test_train, k=args.topk, subject_ids_new=sid_te, subject_ids_train=sid_tr,
            max_same_subject=args.max_same_subject)
    else:
        A_train = topk_sparsify_directed(sim_train, k=args.topk)
        A_val_train = topk_sparsify_cross(sim_val_train, k=args.topk)
        A_test_train = topk_sparsify_cross(sim_test_train, k=args.topk)

    if args.sim_threshold > 0:
        A_train[A_train < args.sim_threshold] = 0.0
        A_val_train[A_val_train < args.sim_threshold] = 0.0
        A_test_train[A_test_train < args.sim_threshold] = 0.0

    if verbose:
        print(f"[inductive/edge] A_train: {A_train.shape}, edges(directed)={int(np.count_nonzero(A_train))}")
        print(f"[inductive/edge] A_val_train: {A_val_train.shape}, edges={int(np.count_nonzero(A_val_train))}")
        print(f"[inductive/edge] A_test_train: {A_test_train.shape}, edges={int(np.count_nonzero(A_test_train))}")

    # --- Assemble block matrices (top-left is a literal copy, not a rebuild) ---
    A_train_val = np.zeros((n_tr + n_val, n_tr + n_val), dtype=A_train.dtype)
    A_train_val[:n_tr, :n_tr] = A_train.copy()
    A_train_val[n_tr:, :n_tr] = A_val_train

    A_train_test = np.zeros((n_tr + n_te, n_tr + n_te), dtype=A_train.dtype)
    A_train_test[:n_tr, :n_tr] = A_train.copy()
    A_train_test[n_tr:, :n_tr] = A_test_train

    # Self-loop + normalize, using symmetric_normalize — the same operator the
    # transductive path uses — rather than row_normalize (see row_normalize's
    # docstring), so the inductive and transductive arms differ only in
    # evaluation protocol, not in normalization. Train-block invariance still
    # holds: training rows have zero mass on held-out columns (the top-right
    # block is zero by construction), so a training row's own out-degree —
    # and therefore its D^-1/2 factor, used for both its row and column
    # position in symmetric_normalize's formula — is unchanged whether
    # computed from A_train alone or from the extended A_train_val/A_train_test.
    # verify_inductive_graphs check 4 confirms this holds numerically.
    G_train_norm = symmetric_normalize(A_train + np.eye(n_tr))
    G_train_val_norm = symmetric_normalize(A_train_val + np.eye(n_tr + n_val))
    G_train_test_norm = symmetric_normalize(A_train_test + np.eye(n_tr + n_te))

    X_tr_tensor = torch.tensor(X_tr_np, dtype=torch.float32)
    X_train_val_tensor = torch.tensor(np.concatenate([X_tr_np, X_val_np], axis=0), dtype=torch.float32)
    X_train_test_tensor = torch.tensor(np.concatenate([X_tr_np, X_te_np], axis=0), dtype=torch.float32)
    G_train_norm_tensor = torch.tensor(G_train_norm, dtype=torch.float32)
    G_train_val_norm_tensor = torch.tensor(G_train_val_norm, dtype=torch.float32)
    G_train_test_norm_tensor = torch.tensor(G_train_test_norm, dtype=torch.float32)

    # Temporal branch (optional): causal same-subject graph 
    AT_train = AT_val = AT_test = None
    AT_train_val = AT_train_test = None
    G_train_temporal_norm_tensor = None
    G_train_val_temporal_norm_tensor = None
    G_train_test_temporal_norm_tensor = None

    use_temporal_branch = args.use_temporal_branch and has_visit_month
    if args.use_temporal_branch and not has_visit_month and verbose:
        print("[inductive/temporal-branch] WARNING: visit_month not available, disabling temporal branch")

    if use_temporal_branch:
        temporal_lam = args.temporal_lambda if args.temporal_lambda > 0 else 0.05

        AT_train = build_temporal_adjacency_causal(vm_tr_arr, sid_tr, temporal_lambda=temporal_lam,
                                                    topk=args.temporal_topk)
        AT_val = build_temporal_adjacency_causal(vm_val_arr, sid_val, temporal_lambda=temporal_lam,
                                                  topk=args.temporal_topk)
        AT_test = build_temporal_adjacency_causal(vm_te_arr, sid_te, temporal_lambda=temporal_lam,
                                                   topk=args.temporal_topk)

        AT_train_val = np.zeros((n_tr + n_val, n_tr + n_val), dtype=AT_train.dtype)
        AT_train_val[:n_tr, :n_tr] = AT_train.copy()
        AT_train_val[n_tr:, n_tr:] = AT_val  # legitimately non-zero: a held-out subject's own visits

        AT_train_test = np.zeros((n_tr + n_te, n_tr + n_te), dtype=AT_train.dtype)
        AT_train_test[:n_tr, :n_tr] = AT_train.copy()
        AT_train_test[n_tr:, n_tr:] = AT_test

        # symmetric_normalize here too — see the comment above the population
        # graph's normalization for why this doesn't break train-block invariance.
        G_train_temporal_norm_tensor = torch.tensor(
            symmetric_normalize(AT_train + np.eye(n_tr)), dtype=torch.float32)
        G_train_val_temporal_norm_tensor = torch.tensor(
            symmetric_normalize(AT_train_val + np.eye(n_tr + n_val)), dtype=torch.float32)
        G_train_test_temporal_norm_tensor = torch.tensor(
            symmetric_normalize(AT_train_test + np.eye(n_tr + n_te)), dtype=torch.float32)

        if verbose:
            print(f"[inductive/temporal-branch] AT_train edges={int(np.count_nonzero(AT_train))}, "
                  f"AT_val (new-node diag block) edges={int(np.count_nonzero(AT_val))}, "
                  f"AT_test (new-node diag block) edges={int(np.count_nonzero(AT_test))}")

    # --- Verify graph construction before any real training happens ---
    set_seed(args.seed)
    probe_model = SpectralGCN(
        in_dim=X_all.shape[1], hidden_dim=args.hidden, num_classes=len(classes),
        num_gcn_layers=args.num_gcn_layers, dropout=args.dropout, use_bn=args.use_bn,
        drop_edge_rate=args.drop_edge_rate, feat_mask_rate=args.feat_mask_rate,
        use_temporal_branch=use_temporal_branch,
        temporal_hidden_dim=args.temporal_branch_hidden,
        temporal_num_layers=args.temporal_branch_layers,
        use_mlp_branch=args.use_mlp_branch,
        use_jk=args.use_jk,
    ).to(device)

    check_outcomes = verify_inductive_graphs(
        A_train, A_train_val, A_train_test, n_tr, n_val, n_te,
        probe_model, device,
        X_tr_tensor, G_train_norm_tensor,
        X_train_val_tensor, G_train_val_norm_tensor,
        X_train_test_tensor, G_train_test_norm_tensor,
        sid_tr, sid_val, sid_te,
        AT_train=AT_train, AT_train_val=AT_train_val, AT_train_test=AT_train_test,
        vm_train=vm_tr_arr, sid_train_arr=sid_tr,
        vm_train_val=vm_train_val_arr, sid_train_val_arr=sid_train_val_arr,
        vm_train_test=vm_train_test_arr, sid_train_test_arr=sid_train_test_arr,
        A_temporal_train_norm_tensor=G_train_temporal_norm_tensor,
        A_temporal_train_val_norm_tensor=G_train_val_temporal_norm_tensor,
        A_temporal_train_test_norm_tensor=G_train_test_temporal_norm_tensor,
    )
    if verbose:
        print(f"[inductive/verify] all checks passed: {check_outcomes}")

    # --- Class weights & loss (identical formula to run_single_fold) ---
    train_labels = y_all[idx_tr]
    cnts = np.bincount(train_labels, minlength=len(classes))
    inv = 1.0 / np.clip(cnts, 1, None)
    wts = (inv ** args.class_weight_exp) / (inv ** args.class_weight_exp).sum() * len(classes)
    w = torch.tensor(wts, dtype=torch.float32, device=device)

    y_val_block = np.concatenate([y_all[idx_tr], y_all[idx_val]])
    val_submask = np.concatenate([np.zeros(n_tr, dtype=bool), np.ones(n_val, dtype=bool)])
    y_test_block = np.concatenate([y_all[idx_tr], y_all[idx_te]])
    test_submask = np.concatenate([np.zeros(n_tr, dtype=bool), np.ones(n_te, dtype=bool)])

    #  Multi-seed ensemble training 
    seeds = [args.seed + i for i in range(num_seeds)]
    ensemble_models: List[SpectralGCN] = []
    all_best_info = []

    for run_i, seed in enumerate(seeds):
        set_seed(seed)
        if verbose:
            print(f"[inductive] TRAINING RUN {run_i + 1}/{num_seeds} (seed={seed}, max {args.epochs} epochs, "
                  f"patience={args.patience}, monitor={args.monitor})")

        model = SpectralGCN(
            in_dim=X_all.shape[1], hidden_dim=args.hidden, num_classes=len(classes),
            num_gcn_layers=args.num_gcn_layers, dropout=args.dropout, use_bn=args.use_bn,
            drop_edge_rate=args.drop_edge_rate, feat_mask_rate=args.feat_mask_rate,
            use_temporal_branch=use_temporal_branch,
            temporal_hidden_dim=args.temporal_branch_hidden,
            temporal_num_layers=args.temporal_branch_layers,
            use_mlp_branch=args.use_mlp_branch,
            use_jk=args.use_jk,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = FocalLoss(alpha=w, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.warmup_epochs,
                                          total_epochs=args.epochs, min_lr=args.min_lr)

        model, best = train_model_inductive(
            model, X_tr_tensor, G_train_norm_tensor, train_labels,
            X_train_val_tensor, G_train_val_norm_tensor, y_val_block, val_submask,
            optimizer, criterion, scheduler=scheduler, max_epochs=args.epochs, patience=args.patience,
            device=device, verbose=verbose, monitor=args.monitor,
            A_temporal_train=G_train_temporal_norm_tensor, A_temporal_train_val=G_train_val_temporal_norm_tensor,
        )

        metric_name = "val_bal" if args.monitor == "val_bal" else "val_loss"
        if verbose:
            print(f"  Run {run_i + 1}: Best epoch={best['epoch']}, Best {metric_name}={best['metric']:.4f}")

        ensemble_models.append(model)
        all_best_info.append({"seed": seed, "epoch": best["epoch"], "metric": best["metric"],
                              "loss_history": best.get("loss_history", {})})

    inductive_dir = None
    if out_dir is not None:
        inductive_dir = Path(out_dir) / "inductive"
        inductive_dir.mkdir(parents=True, exist_ok=True)
        plot_loss_curves(all_best_info, inductive_dir, title_prefix="Inductive ")

    # --- Ensemble evaluation, reusing evaluate/ensemble_evaluate unmodified ---
    train_all_mask = np.ones(n_tr, dtype=bool)
    res_tr = ensemble_evaluate(ensemble_models, X_tr_tensor, G_train_norm_tensor, train_labels,
                               train_all_mask, device,
                               A_temporal=G_train_temporal_norm_tensor)
    res_va = ensemble_evaluate(ensemble_models, X_train_val_tensor, G_train_val_norm_tensor, y_val_block,
                               val_submask, device,
                               A_temporal=G_train_val_temporal_norm_tensor)
    has_test = n_te > 0
    if has_test:
        res_te = ensemble_evaluate(ensemble_models, X_train_test_tensor, G_train_test_norm_tensor, y_test_block,
                                   test_submask, device,
                                   A_temporal=G_train_test_temporal_norm_tensor)
    else:
        res_te = {"balanced_accuracy": float('nan'), "confusion_matrix": [],
                  "classification_report": {}, "probs": np.empty((0, len(classes)))}

    if verbose:
        print_metrics("Inductive Train", res_tr, classes, c2i)
        print_metrics("Inductive Validation", res_va, classes, c2i)
        if has_test:
            print_metrics("Inductive Test", res_te, classes, c2i)

    # Per-seed metrics (for confidence intervals), same pattern as run_single_fold 
    per_seed_results = []
    seed_bal_ci = {}
    seed_per_class_recalls = {}
    if has_test:
        for i, m in enumerate(ensemble_models):
            ind_res = evaluate(m, X_train_test_tensor, G_train_test_norm_tensor, y_test_block, test_submask,
                               device=device, A_temporal=G_train_test_temporal_norm_tensor)
            per_seed_results.append(ind_res)
            if verbose:
                print(f"    Seed {seeds[i]}: Test bal_acc = {ind_res['balanced_accuracy']:.4f}")

        seed_bal_accs = [r['balanced_accuracy'] for r in per_seed_results]
        seed_bal_ci = compute_ci(seed_bal_accs)
        for cls in classes:
            cls_key = str(c2i[cls])
            recalls = [r['classification_report'].get(cls_key, {}).get('recall', 0.0)
                       for r in per_seed_results]
            seed_per_class_recalls[cls] = compute_ci(recalls, clip=(0, 1))

    #  Save results 
    def _strip_probs(d):
        return {k: v for k, v in d.items() if k not in ("probs", "uncertainty")}

    if inductive_dir is not None:
        with open(inductive_dir / "inductive_results.json", "w") as f:
            json.dump({
                "train": _strip_probs(res_tr), "val": _strip_probs(res_va), "test": _strip_probs(res_te),
                "individual_runs": all_best_info,
                "seed_confidence_intervals": {
                    "balanced_accuracy": seed_bal_ci,
                    "per_class_recall": seed_per_class_recalls,
                },
                "classes": classes,
                "class_to_index": c2i,
                "split": {"train": int(n_tr), "val": int(n_val), "test": int(n_te)},
                "args": vars(args),
            }, f, indent=2)

        with open(inductive_dir / "inductive_config.json", "w") as f:
            json.dump({
                "eval_mode": "inductive",
                "frozen_kernel_params": {"kernel": args.kernel, "rbf_gamma": frozen_gamma,
                                         "max_dist": frozen_max_dist},
                "population_temporal_causal_mask": temporal_causal_stats,
                "block_shapes": {
                    "A_train": list(A_train.shape),
                    "A_train_val": list(A_train_val.shape),
                    "A_train_test": list(A_train_test.shape),
                    "AT_train": list(AT_train.shape) if AT_train is not None else None,
                    "AT_train_val": list(AT_train_val.shape) if AT_train_val is not None else None,
                    "AT_train_test": list(AT_train_test.shape) if AT_train_test is not None else None,
                },
                "verify_inductive_graphs_outcomes": check_outcomes,
            }, f, indent=2)

        if verbose:
            print(f"[inductive/done] Results saved to {inductive_dir}/")

    return {
        "train": res_tr, "val": res_va, "test": res_te,
        "models": ensemble_models, "best_info": all_best_info,
        "per_seed_results": per_seed_results,
        "seed_confidence_intervals": {
            "balanced_accuracy": seed_bal_ci,
            "per_class_recall": seed_per_class_recalls,
        },
        "check_outcomes": check_outcomes,
    }


# Nested Cross-Validation

def _build_hp_grid(args):
    """Build hyperparameter search grid for nested CV."""
    from itertools import product
    topk_values = [10, 20, 30]
    rbf_gamma_values = [None, 0.1]
    dropout_values = [0.4, 0.6]
    grid = []
    for topk, gamma, dropout in product(topk_values, rbf_gamma_values, dropout_values):
        grid.append({"topk": topk, "rbf_gamma": gamma, "dropout": dropout})
    return grid


def _override_args(args, hp_dict):
    """Create a copy of args with overridden hyperparameters."""
    new_args = copy.deepcopy(args)
    for key, val in hp_dict.items():
        setattr(new_args, key, val)
    return new_args


def run_nested_cv(df, y_all, edge_feat_cols, classes, c2i,
                  args, device, out_dir):
    """Nested cross-validation: outer folds for evaluation, inner folds for HP tuning.

    """
    subject_ids = df['subject_id'].values
    hp_grid = _build_hp_grid(args)

    outer_cv = StratifiedGroupKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    inner_cv = StratifiedGroupKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed)

    
    print(f"NESTED CROSS-VALIDATION")
    print(f"  Outer folds: {args.outer_folds}, Inner folds: {args.inner_folds}")
    print(f"  Inner seeds: {args.inner_seeds}, Outer seeds: {args.num_seeds}")
    print(f"  HP grid: {len(hp_grid)} combinations")
    for i, hp in enumerate(hp_grid):
        print(f"    [{i}] topk={hp['topk']}, rbf_gamma={hp['rbf_gamma']}, dropout={hp['dropout']}")
   

    outer_results = []
    dummy_X = np.zeros(len(y_all))

    for outer_i, (trainval_idx, test_idx) in enumerate(
        outer_cv.split(dummy_X, y=y_all, groups=subject_ids)
    ):
      
        print(f"OUTER FOLD {outer_i + 1}/{args.outer_folds}  "
              f"(trainval={len(trainval_idx)}, test={len(test_idx)})")


        # Verify no subject leakage
        trainval_subjects = set(subject_ids[trainval_idx])
        test_subjects = set(subject_ids[test_idx])
        assert trainval_subjects.isdisjoint(test_subjects), "Subject leakage in outer fold!"

        # --- Inner CV: HP search ---
        # Subset df to trainval nodes ONLY to prevent data leakage from outer test
        # (outer test nodes must not participate in graph message passing during HP tuning)
        df_inner = df.iloc[trainval_idx].reset_index(drop=True)
        y_inner = y_all[trainval_idx]

        best_hp = None
        best_inner_score = -1.0

        for hp_i, hp in enumerate(hp_grid):
            inner_args = _override_args(args, hp)
            inner_scores = []

            for inner_i, (train_rel_idx, val_rel_idx) in enumerate(
                inner_cv.split(dummy_X[trainval_idx], y=y_inner,
                               groups=subject_ids[trainval_idx])
            ):
                # Masks are relative to df_inner (trainval subset)
                n_inner = len(trainval_idx)
                g_train = np.zeros(n_inner, dtype=bool)
                g_val = np.zeros(n_inner, dtype=bool)
                g_test = np.zeros(n_inner, dtype=bool)  # empty for inner CV
                g_train[train_rel_idx] = True
                g_val[val_rel_idx] = True

                result = run_single_fold(
                    df_inner, y_inner, edge_feat_cols, classes, c2i,
                    g_train, g_val, g_test,
                    inner_args, device, out_dir=None,
                    num_seeds_override=args.inner_seeds,
                    verbose=False, save_models=False,
                    run_explainer=False,
                )
                inner_scores.append(result['val']['balanced_accuracy'])

            mean_score = np.mean(inner_scores)
            print(f"  HP[{hp_i}] topk={hp['topk']}, gamma={hp['rbf_gamma']}, "
                  f"dropout={hp['dropout']}  ->  inner val bal_acc={mean_score:.4f} "
                  f"({' '.join(f'{s:.3f}' for s in inner_scores)})")

            if mean_score > best_inner_score:
                best_inner_score = mean_score
                best_hp = hp

        print(f"\n  Best HP: {best_hp} (inner val bal_acc={best_inner_score:.4f})")

        # --- Outer fold: retrain with best HP on trainval, evaluate on test ---
        outer_args = _override_args(args, best_hp)

        # Split trainval into train/val for early stopping (85/15)
        trainval_subjects_arr = np.array(sorted(trainval_subjects))
        trainval_sub_labels = np.array([
            df[df['subject_id'] == s]["DIAGNOSIS"].mode().iat[0]
            for s in trainval_subjects_arr
        ])
        subs_train, subs_val = train_test_split(
            trainval_subjects_arr, test_size=0.15,
            stratify=trainval_sub_labels, random_state=args.seed + outer_i
        )
        outer_train_mask = np.zeros(len(y_all), dtype=bool)
        outer_val_mask = np.zeros(len(y_all), dtype=bool)
        outer_test_mask = np.zeros(len(y_all), dtype=bool)
        outer_train_mask[np.isin(subject_ids, subs_train)] = True
        outer_val_mask[np.isin(subject_ids, subs_val)] = True
        outer_test_mask[test_idx] = True

        print(f"\n  Outer retrain: train={outer_train_mask.sum()}, val={outer_val_mask.sum()}, "
              f"test={outer_test_mask.sum()}")

        fold_out_dir = out_dir / f"outer_fold_{outer_i}"
        fold_out_dir.mkdir(parents=True, exist_ok=True)

        fold_result = run_single_fold(
            df, y_all, edge_feat_cols, classes, c2i,
            outer_train_mask, outer_val_mask, outer_test_mask,
            outer_args, device, out_dir=fold_out_dir,
            num_seeds_override=args.num_seeds,
            verbose=args.verbose, save_models=True,
            run_explainer=False,
        )

        test_bal = fold_result['test']['balanced_accuracy']
        print(f"\n  Outer fold {outer_i + 1} test balanced accuracy: {test_bal:.4f}")

        outer_results.append({
            "fold": outer_i,
            "best_hp": best_hp,
            "inner_val_score": float(best_inner_score),
            "test_balanced_accuracy": test_bal,
            "test_classification_report": fold_result['test'].get('classification_report', {}),
            "test_auc_ovr_macro": fold_result['test'].get('auc_ovr_macro', float('nan')),
            "test_per_class_ap": fold_result['test'].get('per_class_ap', {}),
        })

    # --- Aggregate results ---
    bal_accs = [r['test_balanced_accuracy'] for r in outer_results]

    per_class_recalls = {cls: [] for cls in classes}
    for r in outer_results:
        report = r.get('test_classification_report', {})
        for cls in classes:
            cls_key = str(c2i[cls])
            if cls_key in report and 'recall' in report[cls_key]:
                per_class_recalls[cls].append(report[cls_key]['recall'])

  
    print(f"NESTED CV RESULTS ({args.outer_folds} outer folds)")
 

    bal_ci = compute_ci(bal_accs)
    print(f"  Balanced Accuracy: {bal_ci['mean']:.4f} +/- {bal_ci['std']:.4f}  "
          f"  95% CI: [{bal_ci['ci_lower']:.4f}, {bal_ci['ci_upper']:.4f}]")
    print(f"  Per-fold: {[f'{a:.4f}' for a in bal_accs]}")

    recall_cis = {}
    for cls in classes:
        recalls = per_class_recalls[cls]
        if recalls:
            ci = compute_ci(recalls, clip=(0, 1))
            recall_cis[cls] = ci
            print(f"  {cls} Recall: {ci['mean']:.4f} +/- {ci['std']:.4f}  "
                  f"  95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]  "
                  f"({[f'{r:.3f}' for r in recalls]})")

    # AUC-OvR macro across folds
    auc_ovrs = [r['test_auc_ovr_macro'] for r in outer_results
                if not math.isnan(r.get('test_auc_ovr_macro', float('nan')))]
    auc_ovr_ci = {}
    if auc_ovrs:
        auc_ovr_ci = compute_ci(auc_ovrs, clip=(0, 1))
        print(f"  AUC-OvR (macro): {auc_ovr_ci['mean']:.4f} +/- {auc_ovr_ci['std']:.4f}  "
              f"  95% CI: [{auc_ovr_ci['ci_lower']:.4f}, {auc_ovr_ci['ci_upper']:.4f}]  "
              f"({[f'{a:.4f}' for a in auc_ovrs]})")

    # Per-class AP across folds
    per_class_ap_cis = {}
    for cls in classes:
        cls_key = str(c2i[cls])
        aps = [r['test_per_class_ap'].get(cls_key, float('nan')) for r in outer_results]
        aps = [a for a in aps if not math.isnan(a)]
        if aps:
            ci = compute_ci(aps, clip=(0, 1))
            per_class_ap_cis[cls] = ci
            print(f"  {cls} AP: {ci['mean']:.4f} +/- {ci['std']:.4f}  "
                  f"  95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]  "
                  f"({[f'{a:.3f}' for a in aps]})")

    print(f"\n  Best Hyperparameters per Outer Fold ")
    for r in outer_results:
        print(f"    Fold {r['fold']}: {r['best_hp']}  (inner val bal_acc={r['inner_val_score']:.4f})")

    # Save results
    # Convert any non-serializable values
    def _clean_for_json(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean_for_json(v) for v in obj]
        return obj

    summary = {
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
        "balanced_accuracy_std": float(np.std(bal_accs)),
        "balanced_accuracy_ci": bal_ci,
        "per_fold_bal_acc": [float(a) for a in bal_accs],
        "per_class_recall_mean": {cls: float(np.mean(v)) for cls, v in per_class_recalls.items() if v},
        "per_class_recall_std": {cls: float(np.std(v)) for cls, v in per_class_recalls.items() if v},
        "per_class_recall_ci": _clean_for_json(recall_cis),
        "auc_ovr_macro_ci": _clean_for_json(auc_ovr_ci),
        "per_fold_auc_ovr": [float(a) for a in auc_ovrs] if auc_ovrs else [],
        "per_class_ap_ci": _clean_for_json(per_class_ap_cis),
        "outer_folds": _clean_for_json(outer_results),
        "hp_grid": hp_grid,
        "args": {k: v for k, v in vars(args).items() if not k.startswith('_')},
    }
    with open(out_dir / "nested_cv_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] Nested CV results saved to {out_dir}/nested_cv_results.json")


# Main

def main():
    ap = argparse.ArgumentParser(description="Spectral GCN with cognitive edge weights")
    ap.add_argument("--single", required=True, help="Pre-merged CSV with features + DIAGNOSIS")
    ap.add_argument("--out_dir", required=True, help="Output directory for results")
    ap.add_argument("--target", choices=["diagnosis", "cdr"], default="diagnosis",
                    help="'diagnosis' (default) or 'cdr' (uses clinical_CDGLOBAL instead)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=256, help="Hidden dimension for GCN + MLP layers")
    ap.add_argument("--num_gcn_layers", type=int, default=2, help="Number of GCN encoder layers")
    ap.add_argument("--dropout", type=float, default=0.6)
    ap.add_argument("--drop_edge_rate", type=float, default=0.2, help="DropEdge rate during training")
    ap.add_argument("--feat_mask_rate", type=float, default=0.05, help="Node feature masking rate during training")
    ap.add_argument("--use_bn", action="store_true", default=True, help="Use BatchNorm (default: True)")
    ap.add_argument("--no_bn", dest="use_bn", action="store_false")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-3, help="AdamW weight decay")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    ap.add_argument("--warmup_epochs", type=int, default=10, help="LR warmup epochs")
    ap.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    ap.add_argument("--topk", type=int, default=25,
                    help="Top-K neighbors per node for graph sparsification")
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="rbf",
                    help="Similarity kernel: 'linear' (1-dist) or 'rbf' (exp(-g*d^2)) (default: rbf)")
    ap.add_argument("--rbf_gamma", type=float, default=None,
                    help="RBF bandwidth. If None, uses median heuristic (default: None)")
    ap.add_argument("--sim_threshold", type=float, default=0.1,
                    help="Prune edges with similarity below this threshold after top-K (default: 0.1)")
    ap.add_argument("--edge_ablation", choices=list(EDGE_ABLATION_GROUPS.keys()),
                    default="cognition",
                    help="Edge feature ablation: cognition, mri, apoe_demo, or all (default: cognition)")
    ap.add_argument("--max_same_subject", type=int, default=1,
                    help="Max neighbors from the same subject per node. 0=block all same-subject edges (default: 1)")
    ap.add_argument("--no_subject_constraint", action="store_true", default=False,
                    help="Disable subject-aware neighbor constraints (original behavior)")
    ap.add_argument("--temporal_lambda", type=float, default=0.2,
                    help="Temporal decay rate for same-subject edge weights (default: 0.2)")
    ap.add_argument("--no_temporal_edges", action="store_true", default=False,
                    help="Disable temporal decay on edge weights")
    ap.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    ap.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing factor")
    ap.add_argument("--class_weight_exp", type=float, default=1.0, help="Class weight exponent")
    ap.add_argument("--monitor", choices=["val_loss", "val_bal"], default="val_bal",
                    help="Early stopping metric (default: val_bal)")
    ap.add_argument("--num_seeds", type=int, default=5, help="Number of seeds for ensemble")
    ap.add_argument("--max_missing", type=float, default=0.8)
    ap.add_argument("--collapse_ge1", action="store_true", default=True)
    ap.add_argument("--no_collapse_ge1", dest="collapse_ge1", action="store_false")
    ap.add_argument("--analyze_graph", action="store_true", default=False,
                    help="Run per-edge feature contribution analysis after graph construction")
    ap.add_argument("--use_temporal_branch", action="store_true", default=True,
                    help="Add parallel temporal GCN branch (same-subject edges, temporal proximity weights)")
    ap.add_argument("--no_temporal_branch", dest="use_temporal_branch", action="store_false",
                    help="Disable the temporal GCN branch")
    ap.add_argument("--use_mlp_branch", action="store_true", default=True,
                    help="Include MLP branch that processes raw features without graph smoothing (default: on)")
    ap.add_argument("--no_mlp_branch", dest="use_mlp_branch", action="store_false",
                    help="Disable the MLP branch (GCN-only mode)")
    ap.add_argument("--use_jk", action="store_true", default=True,
                    help="Use Jumping Knowledge (JK) aggregation: concatenate all GCN layer outputs (default: on)")
    ap.add_argument("--no_jk", dest="use_jk", action="store_false",
                    help="Disable JK aggregation: use only the last GCN layer output")
    ap.add_argument("--temporal_branch_layers", type=int, default=2,
                    help="Number of GCN layers in temporal branch (default: 2)")
    ap.add_argument("--temporal_branch_hidden", type=int, default=None,
                    help="Hidden dim for temporal branch (default: hidden//2)")
    ap.add_argument("--temporal_topk", type=int, default=None,
                    help="Top-K for temporal graph within each subject (default: None, keep all)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no_explain", action="store_true", default=False,
                    help="Skip GNNExplainer and counterfactual explanations (faster runs)")
    ap.add_argument("--load_checkpoints", action="store_true", default=False,
                    help="Skip ensemble training; load saved model_spectral_gcn_seed*.pt "
                         "checkpoints from out_dir instead. Preprocessing, split, and graph "
                         "construction still run unchanged (same --seed/data/args required "
                         "so the loaded checkpoints line up with the rebuilt graph).")
    ap.add_argument("--test_label_free", action="store_true", default=False,
                    help="After neighbor-influence explanation, run test_agreement_is_label_free() "
                         "against the current ensemble and exit before counterfactuals/report stages. "
                         "Requires explanation mode enabled (not --no_explain) and a nonempty test mask.")
    # Nested cross-validation
    ap.add_argument("--nested_cv", action="store_true", default=False,
                    help="Run nested cross-validation instead of single split")
    ap.add_argument("--outer_folds", type=int, default=5,
                    help="Number of outer CV folds (default: 5)")
    ap.add_argument("--inner_folds", type=int, default=3,
                    help="Number of inner CV folds for HP tuning (default: 3)")
    ap.add_argument("--inner_seeds", type=int, default=1,
                    help="Ensemble seeds per inner CV run (default: 1, for speed)")
    ap.add_argument("--counterfactual", action="store_true", default=True,
                    help="Run counterfactual explanation after GNNExplainer (post-hoc, no model change)")
    ap.add_argument("--no_counterfactual", dest="counterfactual", action="store_false",
                    help="Disable counterfactual explanations")
    ap.add_argument("--cf_max_features", type=int, default=2,
                    help="Max features to perturb simultaneously in counterfactual search (default: 2)")
    ap.add_argument("--cf_perturb_steps", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0],
                    help="Perturbation step sizes in units of std dev (default: 0.5 1.0 1.5 2.0)")
    ap.add_argument("--eval_mode", choices=["transductive", "inductive"], default="transductive",
                    help="'transductive' (default): unchanged existing behavior, val/test nodes "
                         "participate in graph message-passing. 'inductive': freeze all graph "
                         "statistics on the training block and build strictly-directed "
                         "train/val/test graphs (sensitivity analysis, not used for reported results "
                         "unless explicitly run) - see run_single_fold_inductive().")
    args = ap.parse_args()

    # temporal branch hidden dim default
    if args.temporal_branch_hidden is None:
        args.temporal_branch_hidden = args.hidden // 2

    # Apply edge feature ablation
    global EDGE_FEATURE_NAMES
    EDGE_FEATURE_NAMES = EDGE_ABLATION_GROUPS[args.edge_ablation]

    set_seed(args.seed)
    device = get_device()
    print(f"[env] device={device}, seed={args.seed}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.single)
    df.columns = [c.strip() for c in df.columns]
    sub_key = find_subject_key(df.columns)
    if sub_key and sub_key != "subject_id":
        df = df.rename(columns={sub_key: "subject_id"})
    if "subject_id" not in df.columns:
        raise KeyError(f"Subject ID column not found. Available: {list(df.columns)}")

    # swap in CDR as the target, then everything below just sees DIAGNOSIS
    if args.target == "cdr":
        if "clinical_CDGLOBAL" not in df.columns:
            raise KeyError(f"clinical_CDGLOBAL not found. Available: {list(df.columns)}")
        if "DIAGNOSIS" in df.columns:
            df = df.drop(columns=["DIAGNOSIS"])
        df = df.rename(columns={"clinical_CDGLOBAL": "DIAGNOSIS"})

    if "DIAGNOSIS" not in df.columns:
        raise KeyError(f"DIAGNOSIS not found. Available: {list(df.columns)}")

    print(f"[load] {args.single} -> {df.shape[0]} rows, {df.shape[1]} cols")

    # Filter and encode target
    df = df[df["DIAGNOSIS"].notna()].copy()

    if args.target == "cdr":
        def collapse_cdr(v):
            try:
                f = float(v)
            except (ValueError, TypeError):
                return str(v)
            if f == 0.0:
                return "0"
            elif f == 0.5:
                return "0.5"
            elif f >= 1.0:
                return "1+"
            return str(v)
        df["DIAGNOSIS"] = df["DIAGNOSIS"].apply(collapse_cdr)
    elif args.collapse_ge1:
        def collapse(v):
            try:
                f = float(v)
            except (ValueError, TypeError):
                return str(v)
            # ADNI integer codes: 1=CN, 2=MCI, 3=AD
            if f == 1.0:
                return "0"
            elif f == 2.0:
                return "0.5"
            elif f >= 3.0:
                return "1+"
            # CDR scores fallback: 0=NC, 0.5=MCI, >=1=AD
            return "1+" if f >= 1.0 else ("0.5" if f == 0.5 else "0")
        df["DIAGNOSIS"] = df["DIAGNOSIS"].apply(collapse)
    else:
        df["DIAGNOSIS"] = df["DIAGNOSIS"].astype(str)

    # Derive temporal features from session ID (only if temporal_lambda > 0)
    if args.temporal_lambda > 0 and "clinical_session_id" in df.columns:
        df["visit_month"] = df["clinical_session_id"].apply(parse_visit_month)
        n_parsed = df["visit_month"].notna().sum()
        n_failed = df["visit_month"].isna().sum()
        if n_failed > 0:
            median_month = df["visit_month"].median()
            df["visit_month"] = df["visit_month"].fillna(median_month)
            print(f"[temporal] Parsed {n_parsed}/{len(df)} session IDs, {n_failed} filled with median={median_month}")
        else:
            print(f"[temporal] Parsed {n_parsed} session IDs -> visit_month: "
                  f"{sorted(df['visit_month'].unique())}")
        if "clinical_entry_age" in df.columns:
            df["visit_age"] = df["clinical_entry_age"] + df["visit_month"] / 12.0
            print(f"[temporal] Derived visit_age = entry_age + visit_month/12")
    elif args.temporal_lambda > 0:
        print("[temporal] No clinical_session_id column — skipping temporal features")

    # Drop sparse columns
    id_cols = [c for c in ["subject_id", "merge_key", "time_key", "clinical_session_id"] if c in df.columns]
    feat = [c for c in df.columns if c not in id_cols + ["DIAGNOSIS"]]
    miss = df[feat].isna().mean()
    keep = [c for c in feat if miss[c] <= args.max_missing]
    df = df[id_cols + keep + ["DIAGNOSIS"]]

    # Identify edge weight features
    edge_feat_cols = get_edge_weight_features(df.columns)
    print(f"[edge] Clinical features for edge weights ({len(edge_feat_cols)}):")
    for col in edge_feat_cols:
        print(f"         {col}")

    if len(edge_feat_cols) == 0:
        raise ValueError("No clinical cognitive features found for edge weight computation!")

    # Class info
    counts = df["DIAGNOSIS"].value_counts().sort_index()
    classes = sorted(df["DIAGNOSIS"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    y_all = np.array([c2i[c] for c in df["DIAGNOSIS"].values], dtype=np.int64)

    print(f"[prep] Classes: {classes}")
    print(f"[prep] Distribution: {counts.to_dict()}")
    print(f"[prep] Subjects: {df['subject_id'].nunique()}, Rows: {len(df)}")

    # Branch: nested CV or single split
    if args.nested_cv:
        run_nested_cv(df, y_all, edge_feat_cols, classes, c2i, args, device, out_dir)
        return

    # Subject-level stratified split: 80 / 10 / 10
    train_mask, val_mask, test_mask = subject_stratified_split(
        df, "DIAGNOSIS", "subject_id",
        val_frac=0.10, test_frac=0.10, seed=args.seed
    )
    print(f"[split] Train={train_mask.sum()} | Val={val_mask.sum()} | Test={test_mask.sum()}")

    # Branch: inductive sensitivity analysis (skips the entire transductive
    # flow below — see run_single_fold_inductive() for the inductive path).
    if args.eval_mode == "inductive":
        run_single_fold_inductive(
            df, y_all, edge_feat_cols, classes, c2i,
            train_mask, val_mask, test_mask,
            args, device, out_dir=out_dir, verbose=args.verbose,
        )
        return

    # Run the full pipeline via run_single_fold
    result = run_single_fold(
        df, y_all, edge_feat_cols, classes, c2i,
        train_mask, val_mask, test_mask,
        args, device, out_dir=out_dir,
        verbose=args.verbose, save_models=True,
        run_explainer=not args.no_explain,
    )

    # Early return if --test_label_free ran the label-permutation regression test —
    # run_single_fold already exited before counterfactuals/save-models; skip this
    # function's later plotting/JSON report-generation stages too.
    if args.test_label_free:
        return

    res_tr = result["train"]
    res_va = result["val"]
    res_te = result["test"]
    ensemble_models = result["models"]
    all_best_info = result["best_info"]
    seeds = [args.seed + i for i in range(args.num_seeds)]

    print(f"\nPOST-HOC VISUALIZATIONS")
    plot_roc_curves(res_te["probs"][test_mask], y_all[test_mask], classes, out_dir)

    X_tensor = result["X_tensor"]
    A_norm_tensor = result["A_norm_tensor"]
    X_edge_tensor = result["X_edge_tensor"]
    edge_index_tensor = result["edge_index_tensor"]
    A_temporal_norm_tensor = result["A_temporal_norm_tensor"]
    plot_umap_embeddings(ensemble_models, X_tensor, A_norm_tensor, y_all, test_mask,
                         classes, c2i, device, out_dir,
                         X_edge=X_edge_tensor, edge_index=edge_index_tensor,
                         A_temporal=A_temporal_norm_tensor)
    plot_seed_confusion_matrix(result.get("per_seed_results", []), seeds, 44, classes, out_dir)
    plot_mci_recall_ablation(out_dir)
    class_importance = result.get("feature_importance", {})
    if class_importance:
        plot_feature_importance(class_importance, classes, out_dir)
    node_results = result.get("neighbor_influence", {})
    if node_results:
        plot_neighbor_influence(node_results, y_all, classes, c2i, out_dir)
    if "uncertainty" in res_te:
        plot_uncertainty_violin(res_te["uncertainty"], y_all, test_mask, classes, out_dir)

    # ---- Save ----
    def _strip_probs(d):
        return {k: v for k, v in d.items() if k not in ("probs", "uncertainty")}

    class_importance = result.get("feature_importance", {})

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "ensemble_train": _strip_probs(res_tr), "ensemble_val": _strip_probs(res_va),
            "ensemble_test": _strip_probs(res_te),
            "individual_runs": all_best_info,
            "seed_confidence_intervals": result.get("seed_confidence_intervals", {}),
            "edge_features": edge_feat_cols,
            "classes": classes,
            "class_to_index": c2i,
            "split": {"train": int(train_mask.sum()), "val": int(val_mask.sum()), "test": int(test_mask.sum())},
            "args": vars(args),
            "feature_importance": {k: v for k, v in class_importance.items()},
            "uncertainty_summary": _build_uncertainty_summary(res_te, y_all, test_mask, classes, c2i),
        }, f, indent=2)

    with open(out_dir / "label_mapping.json", "w") as f:
        json.dump({"classes": classes, "class_to_index": c2i}, f, indent=2)

    # ---- Reproducibility exports ----
    split_labels = np.select(
        [np.asarray(train_mask, dtype=bool), np.asarray(val_mask, dtype=bool), np.asarray(test_mask, dtype=bool)],
        ["train", "val", "test"],
        default="unknown",
    )
    split_subject_df = pd.DataFrame({"subject_id": df["subject_id"].values, "split": split_labels})
    split_subject_df = split_subject_df.drop_duplicates(subset="subject_id").reset_index(drop=True)
    split_subject_df.to_csv(out_dir / "split_subject_ids.csv", index=False)

    per_seed_rows = []
    for seed, r in zip(seeds, result.get("per_seed_results", [])):
        row = {"seed": seed, "balanced_accuracy": r["balanced_accuracy"]}
        for cls in classes:
            recall = r["classification_report"].get(str(c2i[cls]), {}).get("recall", 0.0)
            row[f"{CLASS_DISPLAY.get(cls, cls)}_recall"] = recall
        per_seed_rows.append(row)
    pd.DataFrame(per_seed_rows).to_csv(out_dir / "per_seed_metrics.csv", index=False)

    _save_ensemble_probabilities_csv(res_te, y_all, test_mask, classes, out_dir, df=df)


    print(f"RESULTS SUMMARY (ENSEMBLE of {args.num_seeds} seeds)")

    print(f"  Train Balanced Acc: {res_tr['balanced_accuracy']:.4f}")
    print(f"  Val   Balanced Acc: {res_va['balanced_accuracy']:.4f}")
    print(f"  Test  Balanced Acc: {res_te['balanced_accuracy']:.4f}")

    # Seed-level confidence intervals
    seed_ci = result.get("seed_confidence_intervals", {})
    bal_ci = seed_ci.get("balanced_accuracy", {})
    recall_ci = seed_ci.get("per_class_recall", {})
    if args.num_seeds > 1 and bal_ci:
        print(f"\n   Confidence Intervals (over {args.num_seeds} seeds) ")
        print(f"  Test Bal Acc:  {bal_ci['mean']:.4f} +/- {bal_ci['std']:.4f}  "
              f"  95% CI: [{bal_ci['ci_lower']:.4f}, {bal_ci['ci_upper']:.4f}]")
        for cls in classes:
            ci = recall_ci.get(cls, {})
            if ci:
                print(f"  {cls} Recall:   {ci['mean']:.4f} +/- {ci['std']:.4f}  "
                      f"  95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    print(f"\n  Edge ablation: {args.edge_ablation}")
    print(f"  Edge features: {edge_feat_cols}")
    print(f"  Top-K neighbors: {args.topk}")
    print(f"  Temporal lambda: {args.temporal_lambda}" + (" (disabled)" if args.no_temporal_edges else ""))
    print(f"  GCN layers: {args.num_gcn_layers} (with JK aggregation)")
    if args.use_temporal_branch:
        print(f"  Temporal branch: layers={args.temporal_branch_layers}, "
              f"hidden={args.temporal_branch_hidden}, topk={args.temporal_topk}")
    print(f"  Seeds: {seeds}")
    print(f"\n[done] Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
