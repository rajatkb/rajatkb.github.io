---
title: "The JEPA Ecosystem: A Comprehensive Survey of Variants"
tags: [jepa, survey, self-supervised-learning, representation-learning, world-models, variants]
date: 2026-07-05
draft: false
aliases:
  - jepa-flavors
  - jepa-variants
---

## Overview

This note catalogs every known variant of the Joint-Embedding Predictive Architecture (JEPA) as of mid-2026. The JEPA ecosystem has exploded from two core papers (I-JEPA, V-JEPA) into ~40+ variants spanning theory, world models, audio, vision, language, medical, and beyond.

The taxonomy below groups variants by **research direction** rather than modality, to highlight what each variation actually changes about the core JEPA formula.

For the theoretical foundations of JEPA, see [02-ssl-theory](/notes/jepa_notes/02-ssl-theory). For the historical lineage leading to JEPA, see [01-ssl-history](/notes/jepa_notes/01-ssl-history).

---

## Part 1: Core Papers

The two canonical JEPA papers from Meta/FAIR that define the paradigm.

### I-JEPA (Image)
**Paper:** *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*  
**Authors:** Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas (Meta AI / FAIR)  
**Venue:** ICCV 2023  
[arXiv:2301.08243](https://arxiv.org/abs/2301.08243)

**What it does:** From a single context block in an image, predict representations of multiple target blocks at different scales. Non-generative — predicts in latent space, not pixel space. Uses EMA target encoder + predictor asymmetry to prevent collapse.

**Key innovation:** Data-intrinsic view generation via spatial block masking (no hand-crafted augmentations) + multi-scale prediction heads that create a semantic hierarchy.

**Math:** $\mathcal{L} = \sum_{k=1}^K \mathbb{E}\left[\|g_\phi^{(k)}(s_c) - \bar{s}_t^{(k)}\|_2^2\right]$

### V-JEPA (Video)
**Paper:** *Revisiting Feature Prediction for Learning Visual Representations from Video*  
**Authors:** Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran, Ballas (Meta AI / FAIR)  
**Venue:** NeurIPS 2024  
[arXiv:2404.08471](https://arxiv.org/abs/2404.08471)

**What it does:** Extends I-JEPA's block prediction to the temporal domain — given video context, predict representations of future frames. No reconstruction, no text, no negatives, no pretrained encoders. Trained on 2M videos from public datasets.

**Key innovation:** Temporal prediction in latent space replaces spatial block prediction. Demonstrates that video's temporal structure provides a natural prediction task without curation.

---

## Part 2: LeJEPA & The Anti-Collapse Lineage

The most theoretically important branch — variants that explicitly address the collapse problem by replacing or augmenting the standard EMA + predictor trick.

### LeJEPA
**Paper:** *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*  
**Authors:** Balestriero, Havasi, LeCun, et al.  
**Year:** 2025  
[arXiv:2511.08544](https://arxiv.org/abs/2511.08544)

**What it does:** The most significant JEPA variant since the original. Drops the EMA + stop-gradient + predictor asymmetry entirely and replaces it with **explicit covariance regularization (SIGReg)**. Provides provable guarantees against collapse and removes the architectural heuristics that made JEPA training fragile.

**Key insight:** The trajectory went PMAX's explicit D_l (1992) → BYOL's implicit architectural tricks (2020) → I-JEPA's use of BYOL tricks (2023) → back to explicit D_l in LeJEPA (2025). The field spent five years exploring the ε=1 regime opened up by BYOL, then came home to PMAX's original ε&lt;1 design.

**Anti-collapse mechanism:** SIGReg (Statistical Independent Gaussian Regularization) — explicitly enforces that the representation covariance matrix is full-rank and well-conditioned.

### SIGReg Variants

| Variant | Year | What it changes | Link |
|---------|------|-----------------|------|
| **SIGReg** (in LeJEPA) | 2025 | Explicit covariance regularization replacing EMA + predictor | [arXiv:2511.08544](https://arxiv.org/abs/2511.08544) |
| **Weak-SIGReg** | 2026 | Weaker covariance constraints for stable training without BN | [arXiv:2603.05924](https://arxiv.org/abs/2603.05924) |
| **VISReg** | 2026 | Variance-Invariance-Sketching — sketches the covariance matrix for efficiency | [arXiv:2606.02572](https://arxiv.org/abs/2606.02572) |
| **SPHERE-JEPA** | 2026 | Family of hypersphere-uniformity regularizers for JEPA training | [arXiv:2606.17603](https://arxiv.org/abs/2606.17603) |

---

## Part 3: Theory & Formulation

Variants that analyze, extend, or reformulate the mathematical foundations of JEPA.

### Slow-Features JEPA
**Paper:** *Joint Embedding Predictive Architectures Focus on Slow Features*  
**Authors:** Sobal, Balestriero, et al.  
**Year:** 2022  
[arXiv:2211.10831](https://arxiv.org/abs/2211.10831)

**What it shows:** JEPA's latent prediction objective implicitly learns **slow features** — representations that change slowly over space or time. This explains why JEPA representations are semantically meaningful: semantics change slower than pixels. The temporal/spatial coherence assumption is baked into the objective.

### JEPA Implicit Bias
**Paper:** *How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks*  
**Authors:** Littwin, et al.  
**Year:** 2024  
[arXiv:2407.03475](https://arxiv.org/abs/2407.03475)

**Key result:** In deep linear networks, JEPA's self-distillation dynamics implicitly suppresses features with high variance noise. The EMA target + predictor creates a spectral bias that filters noisy directions before they can corrupt the representation.

### Denoising-JEPA
**Paper:** *Denoising with a Joint-Embedding Predictive Architecture*  
**Authors:** Chen, et al.  
**Year:** 2024  
[arXiv:2410.03755](https://arxiv.org/abs/2410.03755)

**What it does:** Repurposes JEPA for **denoising** — the context encoder sees a noisy image, and the target is the clean image's representation. Shows JEPA's latent prediction can function as a learned denoiser without explicit noise models.

### JEPA + Contrastive
**Paper:** *Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning*  
**Authors:** Mo, et al.  
**Year:** 2024  
[arXiv:2410.19560](https://arxiv.org/abs/2410.19560)

**What it shows:** A formal connection between JEPA's latent prediction loss and contrastive learning objectives. Demonstrates that JEPA can be reinterpreted as a particular form of contrastive learning where positives are different views of the same latent content and negatives are implicitly defined by the predictor's capacity.

### Auxiliary-JEPA
**Paper:** *Why and How Auxiliary Tasks Improve JEPA Representations*  
**Authors:** Yu, et al.  
**Year:** 2025  
[arXiv:2509.12249](https://arxiv.org/abs/2509.12249)

**What it shows:** Adding auxiliary reconstruction or contrastive losses to JEPA improves representation quality. The auxiliary tasks provide additional gradient signal that prevents the predictor from finding degenerate solutions. Establishes a principled framework for choosing auxiliary tasks.

### Gaussian-JEPA
**Paper:** *Gaussian Embeddings: How JEPAs Secretly Learn Your Data Density*  
**Authors:** Balestriero, LeCun  
**Year:** 2025  
[arXiv:2510.05949](https://arxiv.org/abs/2510.05949)

**Key insight:** JEPA's latent space implicitly learns a Gaussian Mixture Model of the data distribution. The predictor's MSE loss in latent space corresponds to a conditional Gaussian density estimation. This provides a probabilistic interpretation of JEPA's representations.

### Intrinsic-Energy JEPA
**Paper:** *Intrinsic-Energy Joint Embedding Predictive Architectures Induce Quasimetric Spaces*  
**Authors:** Kobanda, et al.  
**Year:** 2026  
[arXiv:2602.12245](https://arxiv.org/abs/2602.12245)

**What it shows:** JEPA's latent space naturally forms a **quasimetric** — an asymmetric distance function where the predictor defines a direction-dependent similarity. Connects JEPA to energy-based models and optimal transport theory.

### BiJEPA
**Paper:** *BiJEPA: Bi-directional Joint Embedding Predictive Architecture for Symmetric Representation Learning*  
**Authors:** Huang  
**Year:** 2026  
[arXiv:2603.00049](https://arxiv.org/abs/2603.00049)

**What it does:** Makes JEPA prediction bidirectional — both context→target and target→context prediction losses. The symmetric objective forces representations to be informative about each other in both directions, improving symmetry and reducing the burden on the predictor.

### Reconstruction-vs-JEPA
**Paper:** *Is the reconstruction loss culprit? An attempt to outperform JEPA*  
**Authors:** Potapov, et al.  
**Year:** 2026  
[arXiv:2603.14131](https://arxiv.org/abs/2603.14131)

**What it shows:** A systematic comparison showing that adding pixel-level reconstruction loss to JEPA does NOT hurt — and sometimes helps. Challenges the assumption that generative targets are strictly worse for representation learning.

### Var-JEPA
**Paper:** *Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture*  
**Authors:** Gögl, et al.  
**Year:** 2026  
[arXiv:2603.20111](https://arxiv.org/abs/2603.20111)

**What it does:** Reformulates JEPA as a variational inference problem — the predictor approximates a conditional distribution over target representations. This naturally introduces a KL-divergence regularization term that prevents collapse, providing a principled alternative to SIGReg.

**Key equation:** $\mathcal{L} = \mathbb{E}[-\log p_\phi(z_t | z_c)] + \beta \cdot \text{KL}(q_\theta(z) \| p(z))$

### Sub-JEPA
**Paper:** *Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models*  
**Authors:** Zhao, et al.  
**Year:** 2026  
[arXiv:2605.09241](https://arxiv.org/abs/2605.09241)

**What it does:** Combines JEPA with subspace Gaussian regularization — regularizes the representation covariance only in a learned low-dimensional subspace rather than the full space. Makes end-to-end world model training stable without batch normalization or stop-gradient tricks.

### Factorized Latent Dynamics for Video JEPA
**Paper:** *Factorized Latent Dynamics for Video JEPA: An Empirical Study of Auxiliary Objectives*  
**Authors:** Premi  
**Year:** 2026  
[arXiv:2605.17165](https://arxiv.org/abs/2605.17165)

**What it does:** Factorizes the video JEPA latent space into **content** (static) and **pose** (dynamic) subspaces. An auxiliary slow-feature objective encourages the pose subspace to track temporal changes while the content subspace remains stable.

### BRo-JEPA
**Paper:** *BRo-JEPA: Learning Modular Arithmetic in Latent Space*  
**Authors:** Jha, et al.  
**Year:** 2026  
[arXiv:2606.01372](https://arxiv.org/abs/2606.01372)

**What it does:** A toy-domain probe — trains JEPA to predict modular arithmetic results in latent space. Shows that JEPA learns compositional algebraic structure in its representations, providing evidence for the claim that JEPA extracts abstract relational knowledge.

### JEPA Generalization Theory
**Paper:** *A Generalization Theory for JEPA-Based World Models*  
**Authors:** Cui, et al.  
**Year:** 2026  
[arXiv:2606.27014](https://arxiv.org/abs/2606.27014)

**What it shows:** First formal generalization bounds for JEPA world models. Proves that the gap between training and test prediction error is controlled by the Rademacher complexity of the predictor class and the Lipschitz constant of the encoder. Provides PAC-style guarantees.

---

## Part 4: World Models & Reinforcement Learning

JEPA as the representation backbone for planning, control, and decision-making.

### JEPA-for-RL
**Paper:** *JEPA for RL: Investigating Joint-Embedding Predictive Architectures for Reinforcement Learning*  
**Authors:** Kenneweg, et al.  
**Year:** 2025  
[arXiv:2504.16591](https://arxiv.org/abs/2504.16591)

**What it does:** First systematic study of JEPA as a world model in RL. Encodes states into latent space and predicts future state representations for planning. Shows JEPA world models enable sample-efficient planning in grid-world and MuJoCo tasks, but struggle in visually complex environments.

### DSeq-JEPA
**Paper:** *DSeq-JEPA: Discriminative Sequential Joint-Embedding Predictive Architecture*  
**Authors:** He, et al.  
**Year:** 2025  
[arXiv:2511.17354](https://arxiv.org/abs/2511.17354)

**What it does:** Extends JEPA to sequential decision-making with a **discriminative** objective — the predictor must not only predict the next state representation but also discriminate it from alternatives. Combines JEPA's latent prediction with contrastive learning for RL.

### Value-Guided JEPA
**Paper:** *Value-guided action planning with JEPA world models*  
**Authors:** Destrade, et al.  
**Year:** 2025  
[arXiv:2601.00844](https://arxiv.org/abs/2601.00844)

**What it does:** Integrates value function learning into JEPA's latent space. The predictor not only forecasts future state representations but also the expected return, enabling value-guided planning without separate value network training.

### UWM-JEPA
**Paper:** *UWM-JEPA: Predictive World Models That Imagine in Belief Space*  
**Authors:** Radha, et al.  
**Year:** 2026  
[arXiv:2605.25313](https://arxiv.org/abs/2605.25313)

**What it does:** Introduces **belief-space imagination** — the world model not only predicts future latent states but also maintains a belief distribution over what it *doesn't know*. The predictor outputs uncertainty estimates alongside predicted representations, enabling risk-aware planning.

### EPM-JEPA
**Paper:** *EPM-JEPA: Operator-Side Experience Modulation in JEPA-Family World Models*  
**Authors:** Pandya  
**Year:** 2026  
[arXiv:2606.12979](https://arxiv.org/abs/2606.12979)

**What it does:** Modulates the JEPA world model's behavior based on the **operator** (the agent taking actions). The predictor is conditioned on a policy embedding, allowing the same world model to predict outcomes under different policies — a step toward general-purpose world models.

---

## Part 5: Audio & Multimodal

JEPA applied to non-visual and cross-modal domains.

### A-JEPA (Audio)
**Paper:** *A-JEPA: Joint-Embedding Predictive Architecture Can Listen*  
**Authors:** Fei, et al.  
**Year:** 2023  
[arXiv:2311.15830](https://arxiv.org/abs/2311.15830)

**What it does:** Adapts JEPA to raw audio — masks time-frequency patches in spectrograms and predicts masked region representations. Shows JEPA's block-masking + latent prediction generalizes directly to audio without architectural changes.

### Music-JEPA
**Paper:** *Zero-shot Musical Stem Retrieval with Joint-Embedding Predictive Architectures*  
**Authors:** Riou, et al.  
**Year:** 2024  
[arXiv:2411.19806](https://arxiv.org/abs/2411.19806)

**What it does:** Uses JEPA embeddings for zero-shot retrieval of individual instrument stems from mixed audio. Demonstrates JEPA's representations separate sources in latent space without source separation training.

### Audio-JEPA
**Paper:** *Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning*  
**Authors:** Tuncay, et al.  
**Year:** 2025  
[arXiv:2507.02915](https://arxiv.org/abs/2507.02915)

**What it does:** Full-scale audio representation learning with JEPA on AudioSet and LibriSpeech. Compares block-masking vs temporal-masking strategies for audio and establishes best practices for audio JEPA training.

### VL-JEPA
**Paper:** *VL-JEPA: Joint Embedding Predictive Architecture for Vision-language*  
**Authors:** Chen, et al.  
**Year:** 2025  
[arXiv:2512.10942](https://arxiv.org/abs/2512.10942)

**What it does:** Extends JEPA to vision-language — the context is an image, and the target is a text representation (or vice versa). Predicts cross-modal representations in a joint embedding space. Shows JEPA can learn aligned vision-language representations without contrastive pairs.

### MJEPA (Audio-Visual)
**Paper:** *MJEPA: A Simple and Scalable Joint-Embedding Predictive Architecture for Audio-Visual Learning*  
**Authors:** Teotia, et al.  
**Year:** 2026  
[arXiv:2606.25225](https://arxiv.org/abs/2606.25225)

**What it does:** Multi-modal JEPA that predicts audio representations from video context and vice versa. A single shared encoder processes both modalities. Simple and scalable — matches or exceeds contrastive audio-visual methods.

---

## Part 6: Language & Semantics

JEPA applied to text and semantic representation problems.

### BERT-JEPA
**Paper:** *BERT-JEPA: Reorganizing CLS Embeddings for Language-Invariant Semantics*  
**Authors:** Gillin, et al.  
**Year:** 2026  
[arXiv:2601.00366](https://arxiv.org/abs/2601.00366)

**What it does:** Applies JEPA to BERT embeddings — the context is a sentence in one language, the target is the same sentence's CLS embedding in another language. Learns language-invariant semantic representations without parallel data at training time (uses a pretrained multilingual BERT as the target).

### TextCond-JEPA
**Paper:** *Text-Conditional JEPA for Learning Semantically Rich Visual Representations*  
**Authors:** Huang, et al.  
**Year:** 2026  
[arXiv:2605.03245](https://arxiv.org/abs/2605.03245)

**What it does:** Conditions the JEPA predictor on text — given an image context block and a text description, predict the target block's representation. The text provides semantic guidance that helps the predictor focus on meaningful content rather than spurious correlations.

### LLM-JEPA
**Paper:** *LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures*  
**Authors:** Huang, et al.  
**Year:** 2025  
[arXiv:2509.14252](https://arxiv.org/abs/2509.14252)

**What it does:** Integrates LLM representations as targets for a JEPA-trained vision encoder. A frozen LLM provides semantically rich target representations, while the vision encoder learns to predict them from context. Bridges the semantic gap between vision and language representations.

---

## Part 7: Specialized Applications

Modality-specific and domain-specific JEPA adaptations.

### Point-JEPA
**Paper:** *Point-JEPA: A Joint Embedding Predictive Architecture for Self-Supervised Learning on Point Cloud*  
**Authors:** Saito, et al.  
**Year:** 2024  
[arXiv:2404.16432](https://arxiv.org/abs/2404.16432)

**What it does:** Adapts JEPA to 3D point clouds — masks regions of point cloud space and predicts the representations of masked regions. Handles the irregular, sparse structure of point clouds via a point-wise encoder.

### SparseJEPA
**Paper:** *SparseJEPA: Sparse Representation Learning of Joint Embedding Predictive Architectures*  
**Authors:** Hartman, et al.  
**Year:** 2025  
[arXiv:2504.16140](https://arxiv.org/abs/2504.16140)

**What it does:** Encourages sparsity in JEPA representations via an L1 penalty on the latent codes. Sparse representations are more interpretable, disentangled, and robust. Shows that JEPA can learn truly sparse features without sacrificing downstream performance.

### Graph-JEPA
**Paper:** *Graph-level Representation Learning with Joint-Embedding Predictive Architectures*  
**Authors:** Skenderi, et al.  
**Year:** 2023  
[arXiv:2309.16014](https://arxiv.org/abs/2309.16014)

**What it does:** Adapts JEPA to graph data — predicts representations of masked subgraphs from context subgraphs. The first application of JEPA beyond grid-structured data (images, video, audio).

### Graph-JEPA (PCR)
**Paper:** *Predict, Cluster, Refine: A Joint Embedding Predictive Self-Supervised Framework for Graph Representation Learning*  
**Authors:** Srinivasan, et al.  
**Year:** 2025  
[arXiv:2502.01684](https://arxiv.org/abs/2502.01684)

**What it does:** Adds a clustering + refinement step to Graph-JEPA. After predicting masked subgraph representations, clusters them and refines prototypes — combining JEPA with prototype learning.

### Recurrent-JEPA
**Paper:** *Recurrent Joint Embedding Predictive Architecture with Recurrent Forward Propagation Learning*  
**Authors:** Velarde, et al.  
**Year:** 2024  
[arXiv:2411.16695](https://arxiv.org/abs/2411.16695)

**What it does:** Replaces the feedforward predictor with a recurrent one. For long-range temporal/spatial dependencies, the recurrent predictor can propagate information across multiple steps, enabling JEPA to handle longer horizons.

### SSM-JEPA
**Paper:** *Learning State-Space Models of Dynamic Systems from Arbitrary Data using Joint Embedding Predictive Architectures*  
**Authors:** Ulmen, et al.  
**Year:** 2025  
[arXiv:2508.10489](https://arxiv.org/abs/2508.10489)

**What it does:** Uses JEPA to learn state-space models of physical systems from arbitrary (non-uniformly sampled) trajectory data. The latent space corresponds to the system's state, and the predictor approximates the transition dynamics.

### Video-JEPA (Rep Learning)
**Paper:** *Video Representation Learning with Joint-Embedding Predictive Architectures*  
**Authors:** Drozdov, et al.  
**Year:** 2024  
[arXiv:2412.10925](https://arxiv.org/abs/2412.10925)

**What it does:** Independent implementation of video JEPA (similar to V-JEPA but different masking/augmentation strategies). Provides an open-source baseline for video representation learning with JEPA.

### Video-JEPA (Facial Expression)
**Paper:** *Video Joint-Embedding Predictive Architectures for Facial Expression Recognition*  
**Authors:** Eing, et al.  
**Year:** 2026  
[arXiv:2601.09524](https://arxiv.org/abs/2601.09524)

**What it does:** Fine-tunes JEPA on facial expression recognition. Shows JEPA pretrained representations transfer well to fine-grained facial expression tasks, outperforming contrastive methods.

### Drive-JEPA
**Paper:** *Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving*  
**Authors:** Wang, et al.  
**Year:** 2026  
[arXiv:2601.22032](https://arxiv.org/abs/2601.22032)

**What it does:** Combines video JEPA with multimodal trajectory distillation for autonomous driving. The JEPA world model predicts future driving scenes, and an action head plans trajectories in the latent space. Components: vision encoder, trajectory predictor, action planner — all trained end-to-end.

### MTS-JEPA
**Paper:** *MTS-JEPA: Multi-Resolution Joint-Embedding Predictive Architecture for Time-Series Anomaly Prediction*  
**Authors:** He, et al.  
**Year:** 2026  
[arXiv:2602.04643](https://arxiv.org/abs/2602.04643)

**What it does:** Adapts JEPA to multivariate time series — predicts representations of masked time segments from context at multiple temporal resolutions. Designed for anomaly detection in industrial sensor data.

### JEPA-Guided Diffusion
**Paper:** *Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion*  
**Authors:** Park, et al.  
**Year:** 2026  
[arXiv:2605.24631](https://arxiv.org/abs/2605.24631)

**What it does:** Uses a JEPA latent space to guide diffusion sampling toward minority classes. The JEPA encoder provides a semantic prior that steers the diffusion process away from mode collapse toward rare but meaningful samples.

---

## Part 8: Medical & Healthcare

JEPA applied to clinical and biomedical domains.

### Clin-JEPA
**Paper:** *Clin-JEPA: A Multi-Phase Co-Training Framework for Joint-Embedding Predictive Pretraining on EHR Patient Trajectories*  
**Authors:** Yang, et al.  
**Year:** 2026  
[arXiv:2605.10840](https://arxiv.org/abs/2605.10840)

**What it does:** Applies JEPA to Electronic Health Record (EHR) patient trajectories. A multi-phase co-training framework: Phase 1 predicts masked lab values from context; Phase 2 predicts future diagnosis codes from past trajectory representations. First JEPA for structured clinical data.

### AEGIS (Medical JEPA)
**Paper:** *AEGIS: A Multi-Task Joint-Embedding Predictive Architecture for Mammography*  
**Authors:** Waggener, et al.  
**Year:** 2026  
[arXiv:2607.00277](https://arxiv.org/abs/2607.00277)

**What it does:** Multi-task JEPA for mammography — predicts multi-view breast tissue representations from a single view context. Simultaneously learns to predict future screening outcomes and detect abnormalities in latent space.

---

## Part 9: Visual Taxonomy

```
Original JEPA (FAIR/Meta)
├── I-JEPA (Assran et al., 2023) — Image blocks
└── V-JEPA (Bardes et al., 2024) — Video frames
    │
    ├── THEORY & FORMULATION
    │   ├── Slow-Features JEPA (Sobal, 2022)
    │   ├── JEPA Implicit Bias (Littwin, 2024)
    │   ├── Denoising-JEPA (Chen, 2024)
    │   ├── JEPA+Contrastive (Mo, 2024)
    │   ├── Auxiliary-JEPA (Yu, 2025)
    │   ├── Gaussian-JEPA (Balestriero, 2025)
    │   ├── Intrinsic-Energy JEPA (Kobanda, 2026)
    │   ├── BiJEPA (Huang, 2026)
    │   ├── Reconstruction-vs-JEPA (Potapov, 2026)
    │   ├── Var-JEPA (Gögl, 2026)
    │   ├── Sub-JEPA (Zhao, 2026)
    │   ├── Factorized-LD-VJEPA (Premi, 2026)
    │   ├── BRo-JEPA (Jha, 2026)
    │   └── JEPA Generalization (Cui, 2026)
    │
    ├── ANTI-COLLAPSE (LeJEPA LINEAGE)
    │   ├── LeJEPA / SIGReg (Balestriero, 2025)
    │   ├── Weak-SIGReg (Akbar, 2026)
    │   ├── VISReg (Wu, 2026)
    │   └── SPHERE-JEPA (Nicollier, 2026)
    │
    ├── WORLD MODELS & RL
    │   ├── JEPA-for-RL (Kenneweg, 2025)
    │   ├── DSeq-JEPA (He, 2025)
    │   ├── Value-Guided JEPA (Destrade, 2025)
    │   ├── UWM-JEPA (Radha, 2026)
    │   └── EPM-JEPA (Pandya, 2026)
    │
    ├── AUDIO & MULTIMODAL
    │   ├── A-JEPA / Audio (Fei, 2023)
    │   ├── Music-JEPA (Riou, 2024)
    │   ├── Audio-JEPA (Tuncay, 2025)
    │   ├── VL-JEPA (Chen, 2025)
    │   ├── MJEPA (Teotia, 2026)
    │   └── TextCond-JEPA (Huang, 2026)
    │
    ├── LANGUAGE & SEMANTICS
    │   ├── BERT-JEPA (Gillin, 2026)
    │   ├── LLM-JEPA (Huang, 2025)
    │   └── TextCond-JEPA (Huang, 2026)
    │
    ├── SPECIALIZED APPLICATIONS
    │   ├── Point-JEPA (Saito, 2024)
    │   ├── Graph-JEPA (Skenderi, 2023)
    │   ├── Graph-JEPA / PCR (Srinivasan, 2025)
    │   ├── Recurrent-JEPA (Velarde, 2024)
    │   ├── SSM-JEPA (Ulmen, 2025)
    │   ├── SparseJEPA (Hartman, 2025)
    │   ├── Video-JEPA (Drozdov, 2024)
    │   ├── Video-JEPA / Face (Eing, 2026)
    │   ├── Drive-JEPA (Wang, 2026)
    │   ├── MTS-JEPA (He, 2026)
    │   └── JEPA-Guided Diffusion (Park, 2026)
    │
    └── MEDICAL
        ├── Clin-JEPA (Yang, 2026)
        └── AEGIS (Waggener, 2026)
```

---

## Key Takeaway

The JEPA ecosystem has evolved along clear axes:

1. **Theory (2022-2026):** From "JEPA works" → "why JEPA works" (slow features, implicit bias, Gaussian embeddings) → "can we prove JEPA works" (generalization theory) → "can we replace the heuristics" (LeJEPA, Var-JEPA).

2. **Anti-Collapse (2025-2026):** A distinct sub-lineage that went from "make EMA + predictor work" → "replace them with explicit regularization" → "multiple competing regularizers" (SIGReg, VISReg, SPHERE-JEPA). The field is converging back to PMAX's 1992 design.

3. **World Models (2025-2026):** The most practically important branch — JEPA as the representation backbone for planning, RL, and control. The trajectory from simple RL probes (JEPA-for-RL) → belief-space planning (UWM-JEPA) → operator-conditioned models (EPM-JEPA).

4. **Cross-Modal (2023-2026):** Audio-JEPA → Music-JEPA → VL-JEPA → MJEPA → TextCond-JEPA. The trend is toward increasingly rich conditioning signals (text, language, other modalities) guiding the prediction task.

5. **Domain Adaptation (2023-2026):** Images → Video → Audio → Point Clouds → Graphs → Time Series → EHR. The JEPA framework is modality-agnostic — only the encoder architecture and masking strategy change.

---

## References

### Core Papers
1. Assran, M., et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* ICCV. — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
2. Bardes, A., et al. (2024). *Revisiting Feature Prediction for Learning Visual Representations from Video.* NeurIPS. — [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)

### LeJEPA & Anti-Collapse
3. Balestriero, R., et al. (2025). *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.* — [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
4. Akbar (2026). *Weak-SIGReg: Covariance Regularization for Stable Deep Learning.* — [arXiv:2603.05924](https://arxiv.org/abs/2603.05924)
5. Wu, et al. (2026). *VISReg: Variance-Invariance-Sketching Regularization for JEPA training.* — [arXiv:2606.02572](https://arxiv.org/abs/2606.02572)
6. Nicollier, et al. (2026). *Expanding SPHERE-JEPA: A Family of Statistical Regularizers for the Hypersphere.* — [arXiv:2606.17603](https://arxiv.org/abs/2606.17603)

### Theory & Formulation
7. Sobal, et al. (2022). *Joint Embedding Predictive Architectures Focus on Slow Features.* — [arXiv:2211.10831](https://arxiv.org/abs/2211.10831)
8. Littwin, et al. (2024). *How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks.* — [arXiv:2407.03475](https://arxiv.org/abs/2407.03475)
9. Chen, et al. (2024). *Denoising with a Joint-Embedding Predictive Architecture.* — [arXiv:2410.03755](https://arxiv.org/abs/2410.03755)
10. Mo, et al. (2024). *Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning.* — [arXiv:2410.19560](https://arxiv.org/abs/2410.19560)
11. Yu, et al. (2025). *Why and How Auxiliary Tasks Improve JEPA Representations.* — [arXiv:2509.12249](https://arxiv.org/abs/2509.12249)
12. Balestriero, R., & LeCun, Y. (2025). *Gaussian Embeddings: How JEPAs Secretly Learn Your Data Density.* — [arXiv:2510.05949](https://arxiv.org/abs/2510.05949)
13. Kobanda, et al. (2026). *Intrinsic-Energy Joint Embedding Predictive Architectures Induce Quasimetric Spaces.* — [arXiv:2602.12245](https://arxiv.org/abs/2602.12245)
14. Huang (2026). *BiJEPA: Bi-directional Joint Embedding Predictive Architecture for Symmetric Representation Learning.* — [arXiv:2603.00049](https://arxiv.org/abs/2603.00049)
15. Potapov, et al. (2026). *Is the reconstruction loss culprit? An attempt to outperform JEPA.* — [arXiv:2603.14131](https://arxiv.org/abs/2603.14131)
16. Gögl, et al. (2026). *Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture.* — [arXiv:2603.20111](https://arxiv.org/abs/2603.20111)
17. Zhao, et al. (2026). *Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models.* — [arXiv:2605.09241](https://arxiv.org/abs/2605.09241)
18. Premi (2026). *Factorized Latent Dynamics for Video JEPA: An Empirical Study of Auxiliary Objectives.* — [arXiv:2605.17165](https://arxiv.org/abs/2605.17165)
19. Jha, et al. (2026). *BRo-JEPA: Learning Modular Arithmetic in Latent Space.* — [arXiv:2606.01372](https://arxiv.org/abs/2606.01372)
20. Cui, et al. (2026). *A Generalization Theory for JEPA-Based World Models.* — [arXiv:2606.27014](https://arxiv.org/abs/2606.27014)

### World Models & RL
21. Kenneweg, et al. (2025). *JEPA for RL: Investigating Joint-Embedding Predictive Architectures for Reinforcement Learning.* — [arXiv:2504.16591](https://arxiv.org/abs/2504.16591)
22. He, et al. (2025). *DSeq-JEPA: Discriminative Sequential Joint-Embedding Predictive Architecture.* — [arXiv:2511.17354](https://arxiv.org/abs/2511.17354)
23. Destrade, et al. (2025). *Value-guided action planning with JEPA world models.* — [arXiv:2601.00844](https://arxiv.org/abs/2601.00844)
24. Radha, et al. (2026). *UWM-JEPA: Predictive World Models That Imagine in Belief Space.* — [arXiv:2605.25313](https://arxiv.org/abs/2605.25313)
25. Pandya (2026). *EPM-JEPA: Operator-Side Experience Modulation in JEPA-Family World Models.* — [arXiv:2606.12979](https://arxiv.org/abs/2606.12979)

### Audio & Multimodal
26. Fei, et al. (2023). *A-JEPA: Joint-Embedding Predictive Architecture Can Listen.* — [arXiv:2311.15830](https://arxiv.org/abs/2311.15830)
27. Riou, et al. (2024). *Zero-shot Musical Stem Retrieval with Joint-Embedding Predictive Architectures.* — [arXiv:2411.19806](https://arxiv.org/abs/2411.19806)
28. Tuncay, et al. (2025). *Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning.* — [arXiv:2507.02915](https://arxiv.org/abs/2507.02915)
29. Chen, et al. (2025). *VL-JEPA: Joint Embedding Predictive Architecture for Vision-language.* — [arXiv:2512.10942](https://arxiv.org/abs/2512.10942)
30. Teotia, et al. (2026). *MJEPA: A Simple and Scalable Joint-Embedding Predictive Architecture for Audio-Visual Learning.* — [arXiv:2606.25225](https://arxiv.org/abs/2606.25225)

### Language & Semantics
31. Gillin, et al. (2026). *BERT-JEPA: Reorganizing CLS Embeddings for Language-Invariant Semantics.* — [arXiv:2601.00366](https://arxiv.org/abs/2601.00366)
32. Huang, et al. (2025). *LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures.* — [arXiv:2509.14252](https://arxiv.org/abs/2509.14252)
33. Huang, et al. (2026). *Text-Conditional JEPA for Learning Semantically Rich Visual Representations.* — [arXiv:2605.03245](https://arxiv.org/abs/2605.03245)

### Specialized Applications
34. Saito, et al. (2024). *Point-JEPA: A Joint Embedding Predictive Architecture for Self-Supervised Learning on Point Cloud.* — [arXiv:2404.16432](https://arxiv.org/abs/2404.16432)
35. Hartman, et al. (2025). *SparseJEPA: Sparse Representation Learning of Joint Embedding Predictive Architectures.* — [arXiv:2504.16140](https://arxiv.org/abs/2504.16140)
36. Skenderi, et al. (2023). *Graph-level Representation Learning with Joint-Embedding Predictive Architectures.* — [arXiv:2309.16014](https://arxiv.org/abs/2309.16014)
37. Srinivasan, et al. (2025). *Predict, Cluster, Refine: A Joint Embedding Predictive Self-Supervised Framework for Graph Representation Learning.* — [arXiv:2502.01684](https://arxiv.org/abs/2502.01684)
38. Velarde, et al. (2024). *Recurrent Joint Embedding Predictive Architecture with Recurrent Forward Propagation Learning.* — [arXiv:2411.16695](https://arxiv.org/abs/2411.16695)
39. Ulmen, et al. (2025). *Learning State-Space Models of Dynamic Systems from Arbitrary Data using Joint Embedding Predictive Architectures.* — [arXiv:2508.10489](https://arxiv.org/abs/2508.10489)
40. Drozdov, et al. (2024). *Video Representation Learning with Joint-Embedding Predictive Architectures.* — [arXiv:2412.10925](https://arxiv.org/abs/2412.10925)
41. Eing, et al. (2026). *Video Joint-Embedding Predictive Architectures for Facial Expression Recognition.* — [arXiv:2601.09524](https://arxiv.org/abs/2601.09524)
42. Wang, et al. (2026). *Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving.* — [arXiv:2601.22032](https://arxiv.org/abs/2601.22032)
43. He, et al. (2026). *MTS-JEPA: Multi-Resolution Joint-Embedding Predictive Architecture for Time-Series Anomaly Prediction.* — [arXiv:2602.04643](https://arxiv.org/abs/2602.04643)
44. Park, et al. (2026). *Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion.* — [arXiv:2605.24631](https://arxiv.org/abs/2605.24631)

### Medical
45. Yang, et al. (2026). *Clin-JEPA: A Multi-Phase Co-Training Framework for Joint-Embedding Predictive Pretraining on EHR Patient Trajectories.* — [arXiv:2605.10840](https://arxiv.org/abs/2605.10840)
46. Waggener, et al. (2026). *AEGIS: A Multi-Task Joint-Embedding Predictive Architecture for Mammography.* — [arXiv:2607.00277](https://arxiv.org/abs/2607.00277)
