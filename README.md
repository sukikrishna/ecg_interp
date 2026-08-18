# ecg-interp

Mechanistic interpretability of ECG foundation models: do independently trained models recover a
shared, reproducible representation of the same clinical concepts, or do apparent "clinical
features" depend on architecture, training seed, dataset, and interpretability method?

See [docs/research-plan.md](docs/research-plan.md) for the full research question, hypotheses,
and experimental design, and [docs/related-work.md](docs/related-work.md) for the literature
review this plan is built on.

## Status

Early stage. What's currently wired up end to end: download PTB-XL, load ECGFounder, extract
intermediate-layer activations for 5 clinical concepts, and train a first linear probe per layer
per concept (`examples/extract_ecgfounder_activations.py`). SAE training, seed-reproducibility
analysis, cross-model transfer, and causal ablation (the rest of the plan in
[docs/research-plan.md](docs/research-plan.md)) are designed but not yet implemented.

## Repo structure

```
docs/            research plan, literature review, candidate-model registry
data/            PTB-XL / PTB download scripts and notes (raw data itself is gitignored)
scripts/         download/setup scripts (datasets + model weights)
src/ecg_interp/  library code: data loading, model wrappers, activation extraction, analysis
examples/        end-to-end scripts that tie the library together
external/        cloned third-party model source (gitignored — see scripts/setup_*.sh)
weights/         downloaded pretrained checkpoints (gitignored — see scripts/setup_*.sh)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data

```bash
bash scripts/download_ptbxl.sh   # ~3GB, fully open access, no PhysioNet account needed
bash scripts/download_ptbdb.sh   # optional secondary dataset, same terms
```

See [data/README.md](data/README.md) for dataset details, citations, and a note on what changes
if this project later extends to credentialed datasets like MIMIC-IV.

## Models

```bash
bash scripts/setup_ecgfounder.sh   # clones ECGFounder source + downloads its weights (~740MB)
```

See [docs/models.md](docs/models.md) for the full candidate-model list (CLEF, ECGFounder,
ST-MEM, ECG-JEPA, MERL, ECG-FM, HuBERT-ECG, ...) with license and access notes for each — flags
which ones are permissively licensed vs. which have real usage restrictions (ST-MEM is
proprietary; HuBERT-ECG is non-commercial-only).

## Running the first example

```bash
python examples/extract_ecgfounder_activations.py
```

## License

MIT — see [LICENSE](LICENSE).
