"""Three-model comparison: ECGFounder (12-lead and 1-lead), CLEF-medium, and ECG-JEPA.

ECGFounder and CLEF are the identical Net1D backbone (see docs/models.md) — ECG-JEPA is a real
transformer, confirmed by inspection to have zero convolutions. Including it is what turns the
CKA comparison into an actual architecture-independence test rather than a same-backbone one.

Processes records in chunks rather than loading the whole dataset into memory at once — at full
scale (21,799 records) the naive approach uses 10GB+ of RAM just for raw waveforms and risks
OOM. Expect this to take HOURS on the full dataset (CPU-only, 4 model configs, full PTB-XL) —
pass a smaller n_records to sanity-check it runs before committing to a full run.

Run after:
    bash scripts/download_ptbxl.sh
    bash scripts/setup_ecgfounder.sh
    bash scripts/setup_clef.sh
    bash scripts/setup_ecgjepa.sh

See docs/preliminary-results.md for what a full run found.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from ecg_interp.analysis.probes import linear_probe
from ecg_interp.analysis.subspace import linear_cka
from ecg_interp.data.ptbxl import PTBXL
from ecg_interp.models.clef import CLEF
from ecg_interp.models.ecgfounder import ECGFounder
from ecg_interp.models.ecgjepa import ECGJEPA
from ecg_interp.representations.extract import ActivationExtractor

CHUNK = 500
N_SEEDS = 5

# Depth points compared across architecturally different layer counts (8 for the CNNs, 14 for
# ECG-JEPA) by matching relative position (early/mid/late) rather than layer index.
DEPTH_POINTS = {
    "ecgfounder_1lead": {"early": "first_conv", "mid": "stage_list.3", "late": "stage_list.6"},
    "clef_medium": {"early": "first_conv", "mid": "stage_list.3", "late": "stage_list.6"},
    "ecgjepa": {"early": "W_P", "mid": "encoder_blocks.blocks.6", "late": "norm"},
}


def main(n_records: int | None = None) -> None:
    t0 = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)

    ptbxl = PTBXL.load("data/raw/ptb-xl")
    all_ids = ptbxl.metadata.index.values
    if n_records is not None:
        all_ids = all_ids[:n_records]
    labels = ptbxl.concept_labels().loc[all_ids]
    log(f"{len(all_ids)} records")

    configs = {
        "ecgfounder_12lead": {
            "model": ECGFounder(leads=12),
            "weights": "weights/ecgfounder/12_lead_ECGFounder.pth",
            "lead_selector": lambda w: w,
            "seq_axis": -1,
        },
        "ecgfounder_1lead": {
            "model": ECGFounder(leads=1),
            "weights": "weights/ecgfounder/1_lead_ECGFounder.pth",
            "lead_selector": lambda w: w[:, [0]],
            "seq_axis": -1,
        },
        "clef_medium": {
            "model": CLEF(size="medium"),
            "weights": "weights/clef/clef_medium.ckpt",
            "lead_selector": lambda w: w,
            "seq_axis": -1,
        },
        "ecgjepa": {
            "model": ECGJEPA(),
            "weights": "weights/ecgjepa/multiblock_epoch100.pth",
            "lead_selector": lambda w: w,
            "seq_axis": 1,
        },
    }
    for cfg in configs.values():
        cfg["model"].load(cfg["weights"])
        cfg["activations"] = {name: [] for name in cfg["model"].layer_names}
    log("all 4 models loaded")

    n_chunks = (len(all_ids) + CHUNK - 1) // CHUNK
    for c in range(n_chunks):
        chunk_ids = all_ids[c * CHUNK : (c + 1) * CHUNK]
        chunk_waveforms = ptbxl.load_waveforms(chunk_ids, sampling_rate=500)  # freed each loop
        for cfg in configs.values():
            model = cfg["model"]
            x = torch.stack(
                [model.preprocess(cfg["lead_selector"](w)) for w in chunk_waveforms]
            ).squeeze(1)
            with ActivationExtractor(model.model, model.layer_names) as extractor:
                with torch.no_grad():
                    for i in range(0, len(x), 64):
                        model.forward(x[i : i + 64])
                        for name, act in extractor.activations.items():
                            cfg["activations"][name].append(act.mean(dim=cfg["seq_axis"]).numpy())
        if c % 5 == 0:
            log(f"chunk {c}/{n_chunks}")

    for tag, cfg in configs.items():
        cfg["activations"] = {
            name: np.concatenate(v, axis=0) for name, v in cfg["activations"].items()
        }
    log("activation extraction done")

    for tag, cfg in configs.items():
        print(f"\n=== Probing {tag} ===")
        for concept in labels.columns:
            y = labels[concept].values.astype(int)
            if y.sum() < 20 or (len(y) - y.sum()) < 20:
                continue
            aucs = {
                name: np.mean([linear_probe(cfg["activations"][name], y, seed=s)["auc"] for s in range(N_SEEDS)])
                for name in cfg["model"].layer_names
            }
            best_layer = max(aucs, key=aucs.get)
            print(f"  [{concept}] best={best_layer} auc={aucs[best_layer]:.3f} final={list(aucs.values())[-1]:.3f}")

    print("\n=== Cross-model CKA (early/mid/late) ===")
    model_pairs = [
        ("ecgfounder_1lead", "clef_medium"),
        ("ecgfounder_1lead", "ecgjepa"),
        ("clef_medium", "ecgjepa"),
    ]
    for model_a, model_b in model_pairs:
        for depth in ("early", "mid", "late"):
            layer_a, layer_b = DEPTH_POINTS[model_a][depth], DEPTH_POINTS[model_b][depth]
            cka = linear_cka(configs[model_a]["activations"][layer_a], configs[model_b]["activations"][layer_b])
            print(f"  {model_a}[{layer_a}] vs {model_b}[{layer_b}] ({depth}): CKA={cka:.3f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n_records=n)
