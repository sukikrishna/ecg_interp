# ecg-interp

Mechanistic interpretability of ECG foundation models: do independently trained models recover a
shared, reproducible representation of the same clinical concepts, or do apparent "clinical
features" depend on architecture, training seed, dataset, and interpretability method?

See [docs/research-plan.md](docs/research-plan.md) for the full research question, hypotheses,
and experimental design, and [docs/related-work.md](docs/related-work.md) for the literature
review this plan is built on.

## Status

Early stage. Three models wired up end to end — ECGFounder, CLEF (both the same `Net1D` CNN
backbone), and ECG-JEPA (a genuine transformer, confirmed by inspection) — with intermediate-
layer activation extraction, linear probing for 5 clinical concepts, and cross-model comparison
via linear CKA, all run on the **full PTB-XL dataset** (21,799 records). A first top-k SAE pass
(seed-reproducibility + concept-correlation) has also run, on a smaller sample. See
[docs/preliminary-results.md](docs/preliminary-results.md) for the actual findings, including an
explicit account of which small-sample results held up at full scale and which didn't. SAE
results still need a full-scale rerun; cross-model causal ablation (the rest of the plan in
[docs/research-plan.md](docs/research-plan.md)) isn't implemented yet.

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
bash scripts/setup_clef.sh         # clones CLEF source + downloads its medium checkpoint (~370MB)
bash scripts/setup_ecgjepa.sh      # clones ECG-JEPA source + downloads its checkpoint (~326MB)
```

`setup_ecgjepa.sh` installs `timm`, which by default pulls in a CUDA build of torch and an
incompatible torchvision, silently breaking a CPU-only install — the script guards against this,
but verify `python3 -c "import torch; print(torch.__version__)"` still shows what you expect
after running it.

See [docs/models.md](docs/models.md) for the full candidate-model list (CLEF, ECGFounder,
ST-MEM, ECG-JEPA, MERL, ECG-FM, HuBERT-ECG, ...) with license and access notes for each — flags
which ones are permissively licensed vs. which have real usage restrictions (ST-MEM is
proprietary; HuBERT-ECG is non-commercial-only). It also flags that ECGFounder and CLEF share
the identical backbone architecture, while ECG-JEPA is a genuine transformer — worth reading
before over-interpreting any two-model comparison as "cross-architecture" when it might not be.

## Running the examples

```bash
python examples/extract_ecgfounder_activations.py   # single-model depth profile
python examples/compare_ecgfounder_clef.py           # 2-model CKA + probing (same architecture)
python examples/compare_three_models.py              # 3-model CKA + probing (incl. ECG-JEPA)
```

The 3-model comparison processes data in chunks and is CPU-heavy — expect it to take hours on
the full dataset, not minutes; pass a smaller `n_records` if you just want to sanity-check it
runs. See [docs/preliminary-results.md](docs/preliminary-results.md) for what a full run found.

## License

MIT — see [LICENSE](LICENSE).
