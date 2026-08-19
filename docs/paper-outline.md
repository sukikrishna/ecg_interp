# Paper outline (draft)

Target: ML4H 2026, deadline **2026-09-10** (23 days from today, 2026-08-18). No separate abstract
deadline. This is a working outline to shape what experiments still matter, not final prose.

## Honest scope call: what's actually ready vs. not

Three result threads exist right now (see [preliminary-results.md](preliminary-results.md)):

1. **Depth-dependent decodability + CNN-vs-transformer pooling explanation** — full dataset
   (21,799 records), three models, a falsifiable mechanistic story, and a self-caught correction
   (the naive "pooling ratio per stage" correlation doesn't work because ECGFounder pools by a
   constant ratio at every stage — documented as a real limitation, not hidden). **Ready to be
   the paper's spine.**
2. **Cross-model CKA, including the counter-intuitive different-architecture-more-similar
   result** — full dataset, but only 3 depth points per model pair so far (full per-layer sweep
   in progress as of this writing). **Ready pending that sweep landing.**
3. **SAE feature/subspace stability** — 2,500-record sample only, didn't show the hypothesized
   effect, likely under-trained, methodology may not match how the literature does this
   comparison. **Not ready.** Forcing this into the submission either dilutes the paper with an
   inconclusive side-result or requires a substantial new round of work (full-scale retrain,
   better subspace-stability operationalization, held-out validation of the feature-concept
   correlations) that may not fit in 23 days alongside everything else.

**Recommendation: build the ML4H submission around threads 1-2, mention thread 3 as ongoing/
future work in the discussion rather than a result section.** This is a normal, honest scope for
a workshop paper — a single clean, surprising, well-supported finding beats three half-finished
ones. Revisit this call once the full CKA sweep (running now) is in; if a 3-page workshop format
allows a short "preliminary SAE analysis" subsection without diluting the main claim, that's a
judgment call to make once there's a full draft to look at holistically, not before.

## Working title options

- "Depth, Not Architecture, Predicts Concept Decodability in ECG Foundation Models" (leads with
  Finding 1)
- "Same Architecture, Different Objective: What Determines Shared Representations in ECG
  Foundation Models?" (leads with Finding 2)
- "Pooling, Not Architecture, Explains Representational Drift in ECG Foundation Models" (most
  specific to the actual mechanism, but only defensible if the discussion is honest that the
  pooling explanation is well-motivated, not yet a controlled ablation)

Suggest deciding this after the CKA sweep lands — whichever finding ends up better-supported
should be the title's spine, with the other as the second contribution.

## Abstract skeleton (bullet points, not prose)

- Question: do independently trained ECG foundation models converge on shared representations of
  clinical concepts, and does that depend on architecture?
- Setup: 3 pretrained models (ECGFounder, CLEF — same CNN backbone, different objective/data;
  ECG-JEPA — genuine transformer), evaluated on the full PTB-XL dataset (21,799 records), 5
  clinical concepts, linear probing + linear CKA across depth.
- Finding A: concept decodability rises through depth then plateaus/declines for the two CNNs
  but not the transformer — consistent with temporal pooling destroying some linearly-decodable
  signal, a testable mechanism rather than a post-hoc curve description.
- Finding B: the architecturally different model pair (ECGFounder/ECG-JEPA) shows *higher*
  late-layer CKA than the same-architecture pair (ECGFounder/CLEF) — shared architecture is
  neither necessary nor sufficient for representational similarity in this setting.
- Implication: for interpretability claims about "what a model has learned," architecture family
  is a weaker predictor of internal geometry than assumed, and probing/CKA disagree about what
  counts as "shared" — worth stating both rather than picking one.

## Section outline

**1. Introduction** — motivate via the reproducibility/stability framing in
[research-plan.md](research-plan.md): interpretable structure has been shown to be partially
shared across models (cite the OpenReview finding already in research-plan.md), but individual
features vary across runs even when subspaces are stable. This paper asks the architecture
version of that question for ECG foundation models specifically.

**2. Related work** — draw from [related-work.md](related-work.md): the MI-workshop section on
"are concepts directions? geometry of representations" (Section 1.4) and "faithfulness/stability
under shift" (Section 1.5, especially the Gerasimov et al. seed-dependence paper) are the most
directly relevant citations for framing Findings A and B. The EHR-FM and clinical-bias sections
are background for the "why this matters" paragraph, not core citations for this specific paper.

**3. Methods**
- Models: table from [models.md](models.md), with the architecture-sharing caveat stated
  explicitly (this is a strength of the paper's honesty, not just a limitation to bury)
- Data: PTB-XL, concept labeling from SCP codes (from [research-plan.md](research-plan.md))
- Probing: logistic regression protocol (standardization, class balancing, liblinear — cite the
  actual implementation in `src/ecg_interp/analysis/probes.py`)
- CKA: linear CKA (Kornblith et al. 2019), formula and implementation
  (`src/ecg_interp/analysis/subspace.py`)

**4. Results**
- 4.1 Depth profiles (Finding 1) — the three-model figure
  (`docs/figures/full_scale_depth_profiles.png`) and peak-to-final-drop figure
  (`docs/figures/peak_to_final_drop.png`)
- 4.2 Cross-model CKA (Finding 2) — once the full per-layer sweep lands, this should be a
  heatmap (rows = one model's layers, columns = the other's) for each of the 3 pairs, not just
  the 3-point table currently in preliminary-results.md
- 4.3 (optional, scope-dependent) preliminary SAE note as a short paragraph, not a full section,
  if it fits

**5. Discussion** — the two findings in tension: functional agreement (where concepts are best
read out) doesn't require geometric agreement (CKA), and geometric agreement doesn't require
architectural similarity. What does that imply for interpretability work that assumes "same
architecture -> comparable internals" or "found a direction in one model -> expect it in
another"?

**6. Limitations** — state plainly: two of three models share a backbone (partially controlled
for by including ECG-JEPA, not fully); CKA computed on single-lead or model-native inputs, not
byte-identical inputs across models; peak-layer identification is more sample-sensitive than CKA
(demonstrated empirically by the 2,500-vs-21,799-record discrepancy — this is a genuinely useful
methodological point to make explicitly, not just a caveat to bury); the pooling-mechanism story
is well-motivated but not causally tested (no ablation of pooling itself).

## What's still needed before a draft exists

- [ ] Full per-layer CKA sweep (running now) — produces the heatmap figures for 4.2
- [ ] Decide the title/framing once B is confirmed at full resolution
- [ ] Write actual prose for each section (this outline is not a draft)
- [ ] Decide the SAE-section scope call above once the rest of the draft exists
- [ ] Check ML4H's page limit and formatting template before drafting (not yet looked up)
- [ ] A related-work pass specifically checking whether the CKA cross-architecture result has
  independent precedent worth citing/distinguishing from (the [related-work.md](related-work.md)
  document's "geometry of representations" section is the place to check first)
