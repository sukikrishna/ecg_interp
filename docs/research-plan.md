# Research plan

## Question

When different foundation models learn the same clinical concept (starting with ECG), do they
recover a shared, reproducible representation — or do the apparent clinical features instead
depend on model architecture, training seed, database, and interpretability method (i.e. the
representation is less stationary than it looks)?

Related prior finding (Sharkey et al., 2025 line of work on shared interpretable structure):
interpretable symbols are partially shared across models and modalities and can support
cross-model transfer without retraining — but individual SAE features can vary a lot across
runs even when the broader subspace they live in is stable. That distinction (feature-level vs.
subspace-level stability) is the organizing idea for this project.

Sub-questions:
- Are clinical concepts really shared across independently trained models?
- Are individual features stable across seeds/runs, or only low-dimensional subspaces?
- Does stability predict whether a feature is causally important (not just correlated)?

## Primary hypotheses

1. Individual SAE feature stability < subspace stability (subspaces are the more reliable unit).
2. Subspace stability predicts cross-dataset transfer better than feature-level interpretability.
3. Stable representations produce more selective causal interventions (less collateral damage to
   unrelated concepts when ablating).

## Experimental design

Have (at least) two open pretrained models on the same modality. Starting point for ECG: CLEF and
ECGFounder (see [models.md](models.md) for the full candidate list and license/access notes).

**Concepts (5, well-supported in PTB-XL):**
- Atrial fibrillation
- Bundle branch block
- Normal rhythm
- Left ventricular hypertrophy
- Myocardial infarction

**Representation extraction, per model and per layer:**
- Raw neurons
- Linear probe directions (supervised)
- PCA / supervised low-rank subspaces
- Top-k sparse autoencoder (SAE) features

**Reproducibility check:** train SAEs with 5 different seeds per model/layer and measure
within-model reproducibility via:
- Activation correlation across seeds
- Cosine similarity of decoder vectors
- Overlap among top-activating examples
- Optimal-transport / matched-feature comparison
- Subspace principal angles (between the seed-specific subspaces spanned by each SAE's features)

**Cross-model tests:**
- Does a concept direction learned from model A decode the same concept in model B after a
  lightweight (e.g. linear) alignment?
- Ablate / erase a candidate feature or direction and measure: (a) reduction in concept
  decodability, (b) change in disease-classification performance, (c) effect on *unrelated*
  diagnoses (selectivity of the intervention).

Check experimental design against ECG-InterpBench (arxiv.org/abs/2607.27404,
github.com/JayDuan123/2027-kdd-ECG-InterpBench) — that benchmark is more focused on geometry; this
project's angle is stability/reproducibility and causal selectivity across models and seeds.

## Longer-term direction (EHR / bias)

The same stability-vs-instability question generalizes to structured-EHR foundation models and
the "bias direction" literature (see [related-work.md](related-work.md)): if a demographic-proxy
direction is unstable across seeds/sites, that's itself evidence it's a spurious correlation
rather than a robust mechanism — which matters for whether linear erasure (LEACE/RLACE) is a
trustworthy fix. This is a natural second phase once the ECG stability methodology is validated,
not the starting point.

## Candidate venues

- **ML4H 2026** — submission deadline **2026-09-10**, no separate abstract deadline
  (ml4h.ahli.cc/submit/call-for-papers)
- **ICML 2026 workshops** — Mechanistic Interpretability, Structured Data for Health, Trustworthy
  AI for Good, Generative and Agentic AI for Biology (see [related-work.md](related-work.md) for
  which accepted papers at each are most relevant)

## Open to-dos (not yet decided)

- Which second/third model to add after the first working pipeline (CLEF is the natural next
  step alongside ECGFounder; both are ungated and permissively licensed — see models.md)
- Whether to extend beyond ECG to wearables/EEG/sleep, and to EHR/bias as a second study
- Concrete timeline / milestones against the ML4H 2026-09-10 deadline
