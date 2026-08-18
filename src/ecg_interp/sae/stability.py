"""Cross-seed stability analysis for SAEs — the direct test of this project's primary
hypothesis (see docs/research-plan.md): individual feature stability < subspace stability.
"""
from __future__ import annotations

import numpy as np

from ecg_interp.analysis.subspace import decoder_cosine_similarity, pca_subspace, principal_angles
from ecg_interp.sae.topk_sae import TopKSAE


def feature_stability(sae_a: TopKSAE, sae_b: TopKSAE) -> np.ndarray:
    """For each feature in sae_a, its best cosine-similarity match among sae_b's features.
    Low values across many features means individual features aren't reproducible across seeds.
    """
    sims = decoder_cosine_similarity(sae_a.decoder_directions(), sae_b.decoder_directions())
    return np.abs(sims).max(axis=1)  # abs: a sign-flipped feature is still "the same" feature


def subspace_stability(sae_a: TopKSAE, sae_b: TopKSAE, n_components: int = 20) -> np.ndarray:
    """Principal angles (radians) between the top-`n_components` PCA subspaces of the two SAEs'
    decoder directions. Small angles mean the *span* of what was learned is reproducible even
    if individual features (see feature_stability) don't match one-to-one.
    """
    basis_a = pca_subspace(sae_a.decoder_directions(), n_components)
    basis_b = pca_subspace(sae_b.decoder_directions(), n_components)
    return principal_angles(basis_a, basis_b)
