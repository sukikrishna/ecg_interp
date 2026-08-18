# Related work

Literature review distilled from project notes. Organized by theme; each entry keeps the
original relevance note since that's the part worth preserving, not just the citation.

## EHR foundation models and their representations

| Work | Relevance |
|---|---|
| BEHRT (Li et al., 2020); Med-BERT (Rasmy et al., 2021); CEHR-BERT / CEHR-GPT (2021+) | Canonical structured-EHR backbones; the "model organisms" for this agenda. |
| MedRep (Kim et al., 2025, arXiv:2504.08329) | OMOP concept representations grounded in text and ontology graphs; changes what token embeddings encode. |
| PORTER (2026, arXiv:2606.24102) | Language-grounded, vocabulary-independent event representations; frozen backbone read out by linear probes. |
| Learning Longitudinal Health Representations from EHR and Wearable Data (2026, arXiv:2601.12227) | Multimodal EHR-plus-wearable FM; uses occlusion and counterfactual probes to show what the joint representation weights. |
| AURORA (2026, arXiv:2605.17765) | States the entanglement problem directly; proposes orthogonalized latent geometry. |
| Learning Clinical Representations Under Systematic Distribution Shift (2026, arXiv:2603.07348) | Penalizes environment-sensitive directions tied to workflow and measurement intensity; links representation geometry to shift. |
| Building the EHR Foundation Model via Next Event Prediction (2025, arXiv:2509.25591) | Autoregressive EHR FM; interprets attention against clinical pathways. |

**Takeaway:** the field is converging on frozen-backbone-plus-linear-readout designs — exactly
the setting where probing, activation patching, and concept erasure apply cleanly.

## Bias and fairness in clinical prediction

| Work | Relevance |
|---|---|
| Obermeyer et al. (2019, *Science*) | Landmark: a deployed care-management algorithm underserved Black patients because cost proxied need. The archetypal proxy-discrimination case. |
| Gichoya et al. (2022, *Lancet Digital Health*) | AI recovers patient race from images with no human-visible feature; the mechanism is unresolved. |
| Demographic Predictability in 3D CT Foundation Embeddings (2024, arXiv:2412.00110) | Self-supervised embeddings encode age, sex, race; provides the embedding-probing template. |
| Adversarial Debiasing of 3D CT Foundation Embeddings (2025, arXiv:2502.04386) | VAE-adversarial transform to remove demographic signal from embeddings; requires an added training stage. |
| Debias-CLR (2024, arXiv:2411.10544) | Contrastive in-processing debiasing over ClinicalBERT plus LSTM vitals embeddings. |
| FAME: Fairness-Aware Multimodal Embedding (2025, arXiv:2506.13104) | Subgroup-disparity-weighted fusion over BEHRT and BioClinicalBERT; introduces the EDDI aggregation used for evaluation. |
| SDAE / Fairness at Every Intersection (2024, arXiv:2412.00606) | Intersectional bias across multimodal clinical predictions. |
| A Computational Audit of Demographic Association Encoding in ClinicalBERT (2026, arXiv:2606.14460) | Moves from outcome-level disparity toward model-internal probability structure; notes alignment can suppress overt bias while leaving structural sources intact. |
| EHR data continuity and fairness (2023, arXiv:2309.01935) | Data-level source of disparity; complements representation-level analysis. |
| Integration of fairness-awareness into clinical language processing (2026, *Communications Medicine*) | Fairness-aware loss and subgroup audits across race, sex, age, and intersections. |

**Takeaway:** clinical fairness work is dominated by outcome-level disparity metrics and by pre-,
in-, or post-processing corrections that either retrain the model or transform every
representation. Mechanistic localization of *where and how* bias is represented is largely
absent. The ClinicalBERT audit is the closest existing move toward internal analysis.

## Mechanistic interpretability tooling relevant to bias

| Work | Relevance |
|---|---|
| Sharkey, Chughtai, et al. (2025, TMLR), *Open Problems in Mechanistic Interpretability* | Reference frame for methods (decomposition, probing, causal intervention, validation) and open problems — including that probes detect correlation not causation, and that SAEs have uncovered biases from spurious correlations. |
| Arditi et al. (2024, NeurIPS), *Refusal is mediated by a single direction* | The single-direction, causal-mediation template a "bias direction" study would follow. |
| INLP (Ravfogel et al., 2020, ACL); RLACE (Ravfogel et al., 2022, ICML); LEACE (Belrose et al., 2023, NeurIPS) | The linear concept-erasure toolkit — LEACE gives closed-form global erasure, RLACE gives minimal-rank erasure. |
| Debiasing Without Protected Attributes (2026, arXiv:2606.12088) | Erasure from indirect textual cues without explicit protected labels — important because race and SES are often missing in EHR. |
| Linear socio-demographic representations emerge in LLMs from indirect cues (2025, arXiv:2512.10065) | Models build linear user-demographic directions from indirect signals; directly analogous to proxy encoding in EHR. |
| SAEs to Enhance Mechanistic Interpretability of LLMs in Medicine (2026, JMIR AI e81134) | Position/method survey for SAEs on medical LLMs. |
| Why LLMs' Clinical Reasoning Fails (2026, medRxiv) | Applies SAEs to medical LLM internals to test whether representations are stable and clinically meaningful. |
| Sparse Autoencoders Are Capable LLM Jailbreak Mitigators (2026, ICML MI Workshop) | Demonstrates SAE-latent intervention for a safety objective — the steering-for-safety analog of steering-for-fairness. |
| Fairness is Not Flat: Geometric Phase Transitions Against Shortcut Learning (2026, arXiv:2604.11704) | Isolates low-dimensional data leakages geometrically and links pruning them to reduced demographic bias. |
| The limits of fair medical imaging AI (2024, *Nature Medicine*) | Model performance depends on encoding demographic shortcuts; correcting them reduces out-of-distribution generalization — connects fairness to robustness. |

**Takeaway:** every tool needed for a mechanistic account of clinical bias exists (linear probes,
causal mediation/activation patching, SAEs, closed-form and minimal-rank erasure, steering). None
has been assembled into a fairness-focused study of structured EHR foundation models, and the
correlation-vs-causation distinction remains the field's core validation problem.

## ICML 2026 workshop landscape

### 1. Mechanistic Interpretability workshop

**1.1 Health/clinical FM interpretability** — *Emergent Symbolic Structure in Health Foundation
Models* (Katuwal, Koparkar, Abbaspourazad, Mishra, Kirthivasan) decomposes frozen health FM
embeddings (PPG/accelerometer, ~20M minutes, ~172K participants) into interpretable "symbols"
that associate selectively with health conditions and transfer across modalities. Closest
existing precedent for "what health foundation models learn," and the direct precedent for
extending directional decomposition to structured EHR with fairness as the target concept.

**1.2 Tabular/graph interpretability** (most MI tooling was built for text, EHR FMs are
tabular/temporal) — *Row-Attention Extracts, Column-Attention Projects* (Aßmann): first causal
intervention study of a tabular FM, localizing a linear plug-in classifier to two architectural
axes; a methods template and evidence that activation patching transfers to tabular models.
*Discovering Mechanisms in Tokenized Graph Transformers* (Shin, Jeong, Han): activation patching
+ linear probing on a graph transformer, relevant to disease networks / OMOP ontology graphs.

**1.3 SAEs for fairness/auditing/health** — *Position: Use Sparse Autoencoders to Discover
Unknowns* (Peng, Movva, Kleinberg, Pierson, Garg): SAEs are weaker for known concepts, stronger
for discovering unknown ones; explicitly lists fairness, auditing, safety, health as use cases —
direct justification for an SAE-for-clinical-bias direction, and clarifies when SAEs help vs.
when direction-level methods are better. *Sparse Autoencoders are Capable LLM Jailbreak
Mitigators* (Assogba et al.): Context-Conditioned Delta Steering (CC-Delta) — select
jailbreak-relevant SAE features by comparing paired representations, then apply inference-time
mean-shift steering in SAE latent space; the paired-contrast feature-selection idea transfers
directly to selecting demographic-proxy features.

**1.4 Are concepts directions? Geometry of representations and steering** — *From Directions to
Regions* (Shafran, Ronen, Fahn, Ravfogel, Geiger, Geva): a single global direction assumes linear
separability and can miss nonlinear/multi-dimensional concepts; models activation space as
Gaussian regions with local covariance, often beats SAEs on steering — a direct warning that a
demographic concept (or a clinical one) may be a *region*, not a line. *Manifold Steering Reveals
the Shared Geometry of Neural Network Representation and Behavior* (Wurgaft, Rager, Kowal, et
al.): linear steering can cut through off-manifold regions and produce unnatural outputs;
manifold-respecting steering follows behavior better. *Representational Geometry Reveals How
Context Structures Concept Spaces in LMs* (Hu, Niu, Varma): context moves concepts in a
semantically organized, cross-model-shared way.

**1.5 Faithfulness/stability under shift and reruns (directly this project's reliability
question)** — *Geometry-Adaptive Explainer for Faithful Dictionary-Based Interpretability under
Distribution Shift* (Lim, Kim, Lee, Song): distribution shift rotates the active subspace and
misaligns an in-distribution SAE dictionary; formalizes a faithfulness gap and realigns using
unlabeled OOD activations without gradient updates. **Unstable Features, Reproducible Subspaces:
Understanding Seed Dependence in Sparse Autoencoders** (Gerasimov, Rusalev, Balagansky, Laptev,
Kurochkin, Gavrilov): stable SAE features carry most of the signal; unstable features live in
reproducible low-rank subspaces — **this is the closest existing result to this project's core
hypothesis** and should be treated as a required faithfulness filter before any clinical SAE
feature is claimed real. Also worth citing as methodological caution: *Validating Causal
Abstraction Metrics on Simulated Complex Systems* (Méloux, Pimentel, Portet, Peyrard) — a
ground-truth benchmark template; *Demystifying Variance in Circuit Discovery of LLMs* (Wu, Tonin,
Cevher) — documents resampling/rephrasing/sample-wise variance in circuit discovery.

**1.6 Localizing/editing a concept in weights or features** — *Large Language Models Generate
Harmful Content Using a Distinct, Unified Mechanism* (Orgad et al.): targeted weight pruning as
a causal intervention, finding harmful generation depends on a compact weight set distinct from
benign capabilities, and that alignment compresses rather than removes it — the method transfers
to localizing proxy-discrimination mechanisms. *C-Δθ: Circuit-Restricted Weight Arithmetic for
Selective Refusal* (Kasliwal, Seth, Sankarapu): offline, circuit-restricted weight update (<5%
of parameters) as a deployment path for an edit. *Predictive Concept Decoders* (Huang, Choi,
Johnson, Schwettmann, Steinhardt): concept-bottleneck interpretability assistant that surfaces
latent user attributes — directly the demographic-attribute readout problem. Lower priority but
relevant: *LLMs Know They're Wrong and Agree Anyway* (Pandey) — an orthogonal "opinion-agreement"
direction distinct from a truth direction, masked but not removed by alignment.

**1.7 Multilingual/low-resource interpretability** — *Discovering Cross-Language Reasoning
Invariance in LLMs with Geometry-Invariant SAEs* (Bogdanov, Huang): contrastive SAEs across six
languages, cross-language feature swapping shows sharing is model-dependent — relevant to whether
clinical bias features generalize across languages/settings. *Translation Heads* (Lasnier et
al.): disentangling meaning from language via distinct sparse head sets — a template for
disentangling a nuisance factor (language) from content, analogous to demographic proxy vs.
physiology. *Finding Interpretable Prompt-Specific Circuits in LMs* (Franco, Tassis, Rohr,
Crovella): components reused across languages, signals language-specific.

**1.8 Biology crossover (bridge to GenBio)** — three protein-LM interpretability papers:
*ProtoMech* (Tsui, Talreja, Saeedi, Aghazadeh) — cross-layer transcoders on ESM2, 82–89% of
performance recovered, circuits map to binding/signaling/stability motifs; *Circuit Tracing in
Autoregressive Protein LMs* (Tsui, Deinzer, Saeedi, Aghazadeh) — extends to generative ProGen3;
*Induction Meets Biology* (Pomerants, Nikankin, Reusch, et al.) — two-stage repeat-detection
mechanism. Secondary to a healthcare-centered agenda, but shows the SAE/transcoder/circuit
toolkit already works on biological FMs, supporting that it will work on clinical ones.

**1.9 One causal "AI for good" paper** — *From Tokens to Policy: Causal and Interpretable
Heterogeneous Treatment Effects Identification* (Cadei et al.) reframes HTE identification as
Markov-blanket discovery on a multimodal representation, deployed on two anti-poverty programs in
Africa.

### 2. Structured Data for Health (SD4H) workshop

Scope: tabular EHR, high-frequency physiological time series, disease networks; explicit
**Trust and Reliability** track (explainability, fairness, robustness, privacy) and
**Foundation Models** track (pretraining, scaling, alignment) — the Trust and Reliability track
is the natural home for a mechanistic-fairness EHR paper.

Confirmed accepted papers: *Device Passport: Enabling Spatio-Temporal Pretrained Models to
Generalize Across Input Layouts* (Chau, Liu, Minxha, Cui, Azemi, Zippi, Mahasseni, Sandino) —
cross-device/cross-site robustness, connects to this project's Direction 4 (bias direction
reliability under site/temporal shift). *Knowledge-Informed Kernel State Reconstruction from
Heterogeneous Partial Observations* (Muscarnera, Ruhrberg Estévez, Holt, Saveliev, van der
Schaar) — irregular, partially-observed clinical measurements.

### 3. Trustworthy AI for Good workshop

First-edition workshop connecting AI safety with AI-for-social-good/policy (keynotes: Bengio,
Leibo), special theme on AI for civic discourse. Relevant accepted papers: *Quantifying Risk of
Bias from the Use of AI Surrogates for Social Science Research* (Arif Khan et al.) — bias
quantification when AI stands in for human subjects, methodologically adjacent to auditing bias
in clinical prediction surrogates. *Learning from Self Critique and Refinement for Faithful LLM
Summarization* (Hu, Koppula, Pouransari, Koc, Tuzel, Vemulapalli) — faithfulness of generated
clinical/scientific summaries, a reliability concern for LLM-in-the-loop clinical workflows.

### 4. Generative and Agentic AI for Biology (GenBio) workshop

Scope: generative models for proteins/RNA/cells, agentic hypothesis-generation systems,
foundation/world models for multi-scale biology, benchmarks for autonomous scientific systems,
safety/governance of autonomous biological AI. Accepted list is OpenReview-gated. Relevant
crossover: the three protein-LM interpretability papers above (§1.8) are the concrete
interpretability bridge to GenBio's foundation-model theme. *ELISA: An Interpretable Hybrid
Generative AI Agent for Expression-Grounded Discovery in Single-Cell Genomics* (Coser) — links
scGPT expression embeddings to language via retrieval, making an opaque expression FM queryable;
relevant as a design pattern for making health foundation models interpretable to clinicians.
