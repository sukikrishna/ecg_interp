"""Cross-model comparison: does ECGFounder's and CLEF's representation of the same ECG signal
look similar at matched depths (linear CKA), and can each independently decode the 5 target
concepts from it (linear probes)?

Both models are fed the literal same Lead-I signal — ECGFounder via its 1-lead checkpoint
(trained specifically on Lead I), CLEF via its (roughly lead-invariant) single-lead input.

IMPORTANT CAVEAT: ECGFounder and CLEF-medium build the identical Net1D configuration (same
filter_list/m_blocks_list at every stage) — see docs/models.md and docs/preliminary-results.md.
This script measures the effect of training objective/data on a fixed architecture, not
architecture-independent representational convergence. Add a third, architecturally distinct
model before drawing conclusions about "shared representations across architectures."

Run after:
    bash scripts/download_ptbxl.sh
    bash scripts/setup_ecgfounder.sh
    bash scripts/setup_clef.sh
"""
from __future__ import annotations

import numpy as np
import torch

from ecg_interp.analysis.probes import linear_probe
from ecg_interp.analysis.subspace import linear_cka
from ecg_interp.data.ptbxl import PTBXL
from ecg_interp.models.clef import CLEF
from ecg_interp.models.ecgfounder import ECGFounder
from ecg_interp.representations.extract import ActivationExtractor

N_RECORDS = 2500
SAMPLE_SEED = 0


def extract_activations(model, waveforms, layer_names, single_lead_slice: bool):
    """`single_lead_slice`: ECGFounder's preprocess() expects (samples, leads) and normalizes
    across whatever leads are passed in, so it needs an explicit Lead-I slice; CLEF's
    preprocess() already takes a `lead` index itself."""
    x = torch.stack(
        [
            model.preprocess(w[:, [0]]) if single_lead_slice else model.preprocess(w)
            for w in waveforms
        ]
    ).squeeze(1)
    per_layer = {name: [] for name in layer_names}
    batch_size = 32
    with ActivationExtractor(model.model, layer_names) as extractor:
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                model.forward(x[i : i + batch_size])
                for name, activation in extractor.activations.items():
                    per_layer[name].append(activation.mean(dim=-1).numpy())
    return {name: np.concatenate(v, axis=0) for name, v in per_layer.items()}


def main() -> None:
    ptbxl = PTBXL.load("data/raw/ptb-xl")
    missing = ptbxl.check_concept_codes()
    if missing:
        raise RuntimeError(f"Fix CONCEPT_DEFINITIONS before trusting labels: {missing}")
    labels = ptbxl.concept_labels()

    rng = np.random.RandomState(SAMPLE_SEED)
    sample_ids = np.sort(rng.choice(ptbxl.metadata.index.values, size=N_RECORDS, replace=False))
    waveforms = ptbxl.load_waveforms(sample_ids, sampling_rate=100)

    ecgfounder = ECGFounder(leads=1)
    ecgfounder.load("weights/ecgfounder/1_lead_ECGFounder.pth")
    clef = CLEF(size="medium")
    clef.load("weights/clef/clef_medium.ckpt")
    layer_names = ecgfounder.layer_names
    assert layer_names == clef.layer_names

    founder_acts = extract_activations(ecgfounder, waveforms, layer_names, single_lead_slice=True)
    clef_acts = extract_activations(clef, waveforms, layer_names, single_lead_slice=False)

    print("=== Layer-wise linear CKA(ECGFounder-1lead, CLEF-medium) ===")
    for name in layer_names:
        print(f"  {name}: CKA={linear_cka(founder_acts[name], clef_acts[name]):.3f}")

    labels_sample = labels.loc[sample_ids]
    for model_name, acts in (("ECGFounder-1lead", founder_acts), ("CLEF-medium", clef_acts)):
        print(f"\n=== Probing {model_name} ===")
        for concept in labels_sample.columns:
            y = labels_sample[concept].values.astype(int)
            if y.sum() < 20 or (len(y) - y.sum()) < 20:
                continue
            aucs = {name: linear_probe(acts[name], y, seed=0)["auc"] for name in layer_names}
            best_layer = max(aucs, key=aucs.get)
            print(f"  [{concept}] best={best_layer} (AUC={aucs[best_layer]:.3f})")


if __name__ == "__main__":
    main()
