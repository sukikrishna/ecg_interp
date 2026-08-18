# Candidate foundation models

## ECG models being used or evaluated for this project

Access/license notes below were verified directly against each repo (Aug 2026). None of the
seven require a Hugging Face account, an application, or a signed data-use agreement to obtain
weights — the real friction, where it exists, is licensing rather than access-gating.

| Model | Venue | Lab | Code / weights | License | Gated? | Weight size |
|---|---|---|---|---|---|---|
| **CLEF** | arXiv | Nokia Bell Labs | [github.com/Nokia-Bell-Labs/ecg-foundation-model](https://github.com/Nokia-Bell-Labs/ecg-foundation-model) | BSD-3-Clause-Clear | No — direct Zenodo links | Small 448K / Medium 30.7M / Large 296M params |
| **ECGFounder** | NEJM AI 2025 | PKU Digital Health | [github.com/PKUDigitalHealth/ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder) · HF `PKUDigitalHealth/ECGFounder` | MIT | No | ~370MB per checkpoint (12-lead and 1-lead variants) |
| **ST-MEM** | ICLR 2024 | VUNO Inc. | [github.com/vuno/ST-MEM](https://github.com/vuno/ST-MEM) (weights via Google Drive) | **Proprietary — "VUNO Inc., All rights reserved"**, redistribution/derivative use barred without VUNO's permission | No click-through, but license itself is the blocker | Not stated |
| **ECG-JEPA** | arXiv | S. Kim | [github.com/sehunfromdaegu/ECG_JEPA](https://github.com/sehunfromdaegu/ECG_JEPA) (Google Drive, needs `gdown` — plain wget/curl fail on Drive's interstitial) | MIT | No | 326MB per checkpoint (random-masking and multi-block-masking variants; encoder alone is 85.4M params) |
| **MERL** | ICML 2024 | Imperial College London | [github.com/cheliu-computation/MERL-ICML2024](https://github.com/cheliu-computation/MERL-ICML2024) (Google Drive) | MIT | No | Not stated |
| **ECG-FM** | JAMIA Open 2025 | Bo Wang Lab, U. Toronto / Vector Institute | [github.com/bowang-lab/ECG-FM](https://github.com/bowang-lab/ECG-FM) · HF `wanglab/ecg-fm` | MIT | No | ~1.09GB pretrained, ~1.08GB finetuned (90.9M params) — needs their fairseq-based loader, not vanilla `transformers` |
| **HuBERT-ECG** | arXiv | E. Coppola et al. | HF `Edoardo-BS/hubert-ecg-base` | **CC-BY-NC-4.0 — non-commercial only** | No | ~372.5MB (~93.1M params) |
| HeartLang | arXiv | PKU Digital Health | [github.com/PKUDigitalHealth/HeartLang](https://github.com/PKUDigitalHealth/HeartLang) | Not yet checked | — | — |

**Flags for this project:**
- **ST-MEM**'s proprietary license makes it a poor fit for a repo whose results/code we want to
  share openly — usable for private read-only analysis at most, not for anything redistributive.
  Treat as lower priority unless we specifically need it.
- **HuBERT-ECG** is non-commercial only (CC-BY-NC-4.0), which is fine for academic research/
  publication but worth remembering if anything downstream becomes commercial.
- **CLEF and ECGFounder are the identical backbone.** Both instantiate the exact same `Net1D`
  configuration (same `filter_list`/`m_blocks_list` at every stage, verified against both
  repos' source) — they differ in training data/objective (ECGFounder: supervised 150-class
  diagnostic classification; CLEF: SimCLR-style contrastive pretraining on MIMIC-IV-ECG), not
  architecture. Any "shared representation" finding between the two is evidence about
  training-objective effects on a fixed architecture, **not** architecture-independence — see
  [preliminary-results.md](preliminary-results.md). For an architecture-independence claim,
  add a third, structurally different model (ECG-JEPA is the easiest next candidate: MIT,
  ungated).
- CLEF's exact Zenodo record: [10.5281/zenodo.17572734](https://zenodo.org/records/17572734)
  (`clef_small.ckpt` 5.5MB, `clef_medium.ckpt` 368MB, `clef_largel.ckpt` ~3.6GB — the `l` typo
  in "largel" is Zenodo's own filename, not a mistake here).
- Starting pair per the original plan: **CLEF + ECGFounder** — both permissively licensed
  (BSD-3-Clause-Clear / MIT) and fully open, no account needed. ECGFounder is the smaller,
  simpler integration (plain `.pth` + HF hosting), so it's the first one wired up end-to-end in
  [`examples/`](../examples/); CLEF is the natural next model to add.
- **ECG-JEPA is confirmed (by actually instantiating it and checking for `Conv1d`/`Conv2d`) to
  be a genuine transformer/JEPA architecture** — multi-head attention blocks, patch embedding
  via a plain linear layer (`W_P`), no convolutions anywhere. This is what makes it useful here:
  it's the model that can turn the CLEF/ECGFounder comparison into an actual
  architecture-independence test instead of a same-backbone one. It expects a specific 8-lead
  subset (I, II, V1-V6 — III/aVR/aVL/aVF are dropped since they're linearly derivable from I &
  II), 2500 samples (10s at an effective 250Hz), no z-scoring. **Dependency caution**: installing
  its `timm` requirement naively pulls in a CUDA build of torch and an incompatible torchvision,
  silently breaking an existing CPU-only install — `scripts/setup_ecgjepa.sh` pins torch via a
  constraints file and installs the rest with `--no-deps` to avoid this; verify
  `python3 -c "import torch; print(torch.__version__)"` still shows the expected build after
  running it.

## Other modalities noted for a possible follow-on study

| Signal | Model | Venue | Lab | Code / weights |
|---|---|---|---|---|
| PPG | PaPaGei-S | ICLR 2025 | Nokia Bell Labs | github.com/Nokia-Bell-Labs/papagei-foundation-model |
| PPG | Pulse-PPG | arXiv | Xu et al. | github.com/maxxu05/pulseppg |
| PPG | AI-PPG Age | arXiv | Nie et al. | huggingface.co/Ngks03/PPG-VascularAge |
| PPG | SiamQuality | arXiv | C. Ding et al. | github.com/chengding0713/SiamQuality |
| EEG | LaBraM | ICLR 2024 | Shanghai Jiao Tong University | github.com/935963004/LaBraM |
| EEG | CBraMod | ICLR 2025 | Zhejiang University | huggingface.co/weighting666/CBraMod |
| EEG | EEGPT | NeurIPS 2024 | G. Wang et al. | (see workshop index) |
| Sleep (PSG) | SleepFM | ICML 2024 | James Zou Lab, Stanford | github.com/rthapa84/sleepfm-codebase |
| Sleep (PSG) | SleepFM Clinical | Nature Medicine 2026 | James Zou Lab, Stanford | github.com/zou-group/sleepfm-clinical |
| Sleep (PSG) | OSF | ICML 2026 | yang-ai-lab | github.com/yang-ai-lab/OSF-Open-Sleep-FM |

## Interpretability tooling (all MIT, code-only, no weights to worry about)

| Library | Use | Link |
|---|---|---|
| `dictionary_learning` (Marks) | TopK/other SAE trainers, activation buffers, HF-hosted dictionaries | github.com/saprmarks/dictionary_learning |
| OpenAI `sparse_autoencoder` | TopK SAEs + feature visualizer | github.com/openai/sparse_autoencoder |
| SAELens | SAE training/analysis suite | github.com/jbloomAus/SAELens |
| `concept-erasure` (EleutherAI) | Closed-form linear concept erasure (LEACE) | github.com/EleutherAI/concept-erasure |
| `conceptual-constraints` (EleutherAI) | Applying LEACE during training | github.com/EleutherAI/conceptual-constraints |

## Related codebases

- ECG-InterpBench — github.com/JayDuan123/2027-kdd-ECG-InterpBench
- GeoSAE (brain MRI) — github.com/favour-nerrise/GeoSAE
- MammoSAE — github.com/krishnakanthnakka/MammoSAE
- Geneformer atlas — biodyn-ai.github.io/geneformer-atlas
- scGPT atlas — biodyn-ai.github.io/scgpt-atlas
