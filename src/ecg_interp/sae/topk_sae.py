"""Top-k sparse autoencoder for decomposing layer activations into sparse features.

Follows the standard top-k SAE recipe (Gao et al., 2024, "Scaling and evaluating sparse
autoencoders"; Makhzani & Frey, 2013, k-sparse autoencoders): a single hidden layer that keeps
only the top-k activations per example and zeroes the rest, trained to reconstruct the input
under MSE. No L1 penalty is needed since top-k directly enforces sparsity.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, dict_size: int, k: int):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.encoder(x - self.pre_bias))
        if self.k < z.shape[-1]:
            _, topk_idx = torch.topk(z, self.k, dim=-1)
            mask = torch.zeros_like(z).scatter_(-1, topk_idx, 1.0)
            z = z * mask
        return z

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decoder(z) + self.pre_bias
        return x_hat, z

    def decoder_directions(self) -> np.ndarray:
        """(dict_size, input_dim) — one unit-norm-ish row per learned feature."""
        return self.decoder.weight.detach().numpy().T


def train_sae(
    activations: np.ndarray,
    dict_size: int,
    k: int,
    seed: int = 0,
    n_epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> TopKSAE:
    torch.manual_seed(seed)
    x = torch.from_numpy(activations).float()
    sae = TopKSAE(input_dim=x.shape[1], dict_size=dict_size, k=k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    n = x.shape[0]
    for _ in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch = x[perm[i : i + batch_size]]
            x_hat, _ = sae(batch)
            loss = ((x_hat - batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Renormalize decoder columns after every step (standard SAE practice) so the
            # reconstruction loss can't be trivially reduced by growing decoder norms instead
            # of actually improving the code.
            with torch.no_grad():
                sae.decoder.weight.div_(sae.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)
    return sae


def explained_variance(sae: TopKSAE, activations: np.ndarray) -> float:
    """Fraction of variance in `activations` recovered by the SAE's reconstruction."""
    x = torch.from_numpy(activations).float()
    with torch.no_grad():
        x_hat, _ = sae(x)
    residual = ((x - x_hat) ** 2).sum()
    total = ((x - x.mean(dim=0, keepdim=True)) ** 2).sum()
    return float(1 - residual / total)


def feature_activations(sae: TopKSAE, activations: np.ndarray) -> np.ndarray:
    """(n_examples, dict_size) sparse code for each example — for concept-correlation checks
    and cross-seed feature matching."""
    x = torch.from_numpy(activations).float()
    with torch.no_grad():
        z = sae.encode(x)
    return z.numpy()
