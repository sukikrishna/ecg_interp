"""PCA / low-rank subspace analysis, including subspace principal angles — the main tool for
comparing whether two SAEs (different seeds, or different models) converge on the same
low-dimensional structure even when individual features don't match one-to-one."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def pca_subspace(activations: np.ndarray, n_components: int) -> np.ndarray:
    """Returns an orthonormal basis (n_components, n_features) spanning the top-variance
    subspace of `activations`."""
    pca = PCA(n_components=n_components)
    pca.fit(activations)
    return pca.components_


def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between two subspaces given as orthonormal bases
    of shape (k, n_features). 0 means the subspaces overlap exactly in that direction; pi/2
    means fully orthogonal in that direction. Cosines of the angles are the singular values of
    basis_a @ basis_b.T.
    """
    cosines = np.linalg.svd(basis_a @ basis_b.T, compute_uv=False)
    return np.arccos(np.clip(cosines, -1.0, 1.0))


def decoder_cosine_similarity(decoder_a: np.ndarray, decoder_b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between two SAE decoder-vector sets, shape (n_features_a, dim)
    and (n_features_b, dim) — used to match features across seeds/runs before declaring one
    stable."""
    a = decoder_a / np.linalg.norm(decoder_a, axis=1, keepdims=True)
    b = decoder_b / np.linalg.norm(decoder_b, axis=1, keepdims=True)
    return a @ b.T
