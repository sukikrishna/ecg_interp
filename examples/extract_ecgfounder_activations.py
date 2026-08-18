"""End-to-end example: load PTB-XL, run ECGFounder, extract layer activations for the 5 target
clinical concepts, and report a first linear-probe AUC per layer.

Run after:
    bash scripts/download_ptbxl.sh
    bash scripts/setup_ecgfounder.sh
"""
from __future__ import annotations

import numpy as np

from ecg_interp.analysis.probes import linear_probe
from ecg_interp.data.ptbxl import PTBXL
from ecg_interp.models.ecgfounder import ECGFounder
from ecg_interp.representations.extract import ActivationExtractor

N_PER_CLASS = 100  # small sample for a quick first pass, not the full 21,801 records


def main() -> None:
    ptbxl = PTBXL.load("data/raw/ptb-xl")
    missing = ptbxl.check_concept_codes()
    if missing:
        raise RuntimeError(f"Fix CONCEPT_DEFINITIONS before trusting labels: {missing}")

    labels = ptbxl.concept_labels()

    model = ECGFounder(leads=12)
    model.load("weights/ecgfounder/12_lead_ECGFounder.pth")

    for concept in labels.columns:
        positive_ids = labels.index[labels[concept]][:N_PER_CLASS]
        negative_ids = labels.index[~labels[concept]][:N_PER_CLASS]
        if len(positive_ids) < 10 or len(negative_ids) < 10:
            print(f"[{concept}] skipped - too few examples ({len(positive_ids)} positive)")
            continue

        ecg_ids = list(positive_ids) + list(negative_ids)
        y = np.array([1] * len(positive_ids) + [0] * len(negative_ids))
        waveforms = ptbxl.load_waveforms(ecg_ids, sampling_rate=100)

        per_layer_activations = {name: [] for name in model.layer_names}
        with ActivationExtractor(model.model, model.layer_names) as extractor:
            for waveform in waveforms:
                model.forward(model.preprocess(waveform))
                for name, activation in extractor.activations.items():
                    per_layer_activations[name].append(activation.mean(dim=-1).squeeze(0).numpy())

        print(f"\n[{concept}] {len(positive_ids)} positive / {len(negative_ids)} negative")
        for name, activations in per_layer_activations.items():
            result = linear_probe(np.stack(activations), y)
            print(f"  {name}: AUC={result['auc']:.3f}")


if __name__ == "__main__":
    main()
