"""Train a top-k SAE on one ECGFounder layer across multiple seeds, and test this project's
primary hypothesis: individual SAE feature stability < subspace stability (see
docs/research-plan.md). Also checks whether any individual feature correlates with one of the
5 target concepts, the more exploratory "what did the SAE find" angle.

Run after:
    bash scripts/download_ptbxl.sh
    bash scripts/setup_ecgfounder.sh
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pointbiserialr

from ecg_interp.data.ptbxl import PTBXL
from ecg_interp.models.ecgfounder import ECGFounder
from ecg_interp.representations.extract import ActivationExtractor
from ecg_interp.sae.stability import feature_stability, subspace_stability
from ecg_interp.sae.topk_sae import explained_variance, feature_activations, train_sae

import torch

N_RECORDS = 2500
SAMPLE_SEED = 0
LAYER = "stage_list.3"  # a peak layer for LVH/MI in the depth-profile finding
DICT_SIZE_MULTIPLIER = 4
TOP_K = 32
N_SAE_SEEDS = 5


def main() -> None:
    ptbxl = PTBXL.load("data/raw/ptb-xl")
    labels = ptbxl.concept_labels()

    rng = np.random.RandomState(SAMPLE_SEED)
    sample_ids = np.sort(rng.choice(ptbxl.metadata.index.values, size=N_RECORDS, replace=False))
    waveforms = ptbxl.load_waveforms(sample_ids, sampling_rate=100)

    model = ECGFounder(leads=12)
    model.load("weights/ecgfounder/12_lead_ECGFounder.pth")

    x = torch.stack([model.preprocess(w) for w in waveforms]).squeeze(1)
    activations = []
    with ActivationExtractor(model.model, [LAYER]) as extractor:
        with torch.no_grad():
            for i in range(0, len(x), 32):
                model.forward(x[i : i + 32])
                activations.append(extractor.activations[LAYER].mean(dim=-1).numpy())
    activations = np.concatenate(activations, axis=0)
    print(f"activations: {activations.shape} from {LAYER}")

    dict_size = activations.shape[1] * DICT_SIZE_MULTIPLIER
    saes = []
    for seed in range(N_SAE_SEEDS):
        sae = train_sae(activations, dict_size=dict_size, k=TOP_K, seed=seed)
        ev = explained_variance(sae, activations)
        print(f"seed {seed}: explained_variance={ev:.3f}")
        saes.append(sae)

    print(f"\n=== Cross-seed stability ({N_SAE_SEEDS} seeds, {dict_size} features each) ===")
    feature_matches, subspace_angles = [], []
    for i in range(N_SAE_SEEDS):
        for j in range(i + 1, N_SAE_SEEDS):
            feature_matches.append(feature_stability(saes[i], saes[j]))
            subspace_angles.append(subspace_stability(saes[i], saes[j], n_components=20))
    feature_matches = np.concatenate(feature_matches)
    subspace_angles = np.concatenate(subspace_angles)
    print(
        f"individual feature stability (best cosine match across seed pairs): "
        f"mean={feature_matches.mean():.3f}, median={np.median(feature_matches):.3f}, "
        f"frac>0.8={np.mean(feature_matches > 0.8):.3f}"
    )
    print(
        f"subspace stability (principal angles, degrees, across seed pairs): "
        f"mean={np.degrees(subspace_angles).mean():.1f}, "
        f"max={np.degrees(subspace_angles).max():.1f}"
    )

    print(f"\n=== Feature-concept correlation (seed 0, {dict_size} features) ===")
    codes = feature_activations(saes[0], activations)
    labels_sample = labels.loc[sample_ids]
    for concept in labels_sample.columns:
        y = labels_sample[concept].values.astype(float)
        if y.sum() < 20 or (len(y) - y.sum()) < 20:
            continue
        correlations = np.array(
            [pointbiserialr(y, codes[:, f]).correlation for f in range(dict_size)]
        )
        correlations = np.nan_to_num(correlations)
        top = np.argsort(-np.abs(correlations))[:3]
        top_str = ", ".join(f"feature {f} (r={correlations[f]:.2f})" for f in top)
        print(f"  [{concept}] top features: {top_str}")


if __name__ == "__main__":
    main()
