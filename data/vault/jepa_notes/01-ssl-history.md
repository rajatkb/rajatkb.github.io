---
title: "Self-Supervised Learning: A Historical Arc to JEPA"
tags: [ssl, self-supervised-learning, jepa, representation-learning, deep-learning, predictive-ssl]
date: 2026-06-07
lastmod: 2026-06-07
draft: false
---

## Overview

This note traces the evolution of Self-Supervised Learning (SSL) in computer vision, building a proper **timeline** that distinguishes between **Generative SSL** (predicting pixels/data) and **Predictive SSL** (predicting in latent space). The Joint-Embedding Predictive Architecture (JEPA) belongs firmly to the Predictive SSL lineage — a lineage that goes back much further than is commonly acknowledged.

The goal: understand what theoretical problems each phase of SSL addressed, and how they converge on the questions JEPA claims to solve.

---

## Part 0: A Critical Distinction — Generative vs Predictive SSL

Before tracing the timeline, we must distinguish two fundamentally different families of SSL:

| Aspect | Generative SSL | Predictive SSL |
|--------|---------------|----------------|
| **Target** | Raw data (pixels, tokens, waveforms) | Abstract latent representations |
| **Loss** | Reconstruction error (L1, L2, cross-entropy) | Similarity/predictability in embedding space |
| **Collapse risk** | Low (pixel reconstruction is well-constrained) | High (representations can trivially collapse) |
| **Computational cost** | High (decoder must produce high-d output) | Lower (predict lightweight embeddings) |
| **Examples** | Autoencoders, MAE, BEiT, Colorization | PMAX, BYOL, SimCLR, Barlow Twins, VICReg, JEPA |
| **Theoretical framing** | Information preservation (encode all detail) | Information abstraction (extract what's predictable) |

JEPA is a **predictive** (non-generative) method: it predicts *representations* in latent space, not pixels. This is essential to understanding both its goals and its lineage.

---

## Part 1: The Pre-Deep Learning Roots (Pre-1990s)

### Horace Barlow's Redundancy Reduction (1961)
The conceptual ancestor of all SSL. Barlow proposed that the goal of sensory processing is to **reduce redundancy** in the input — to find a representation where each component is statistically independent. This principle ("Barlow's postulate") directly inspired:
- Infomax (Linsker, 1988; Becker & Hinton, 1992)
- Barlow Twins (Zbontar et al., 2021)
- VICReg (Bardes et al., 2022)

### Werbos, Munro, Nguyen & Widrow (1987-1989) — Early World Models
- **Werbos (1987-1989):** Feedforward neural networks trained to predict future states (system identification via backpropagation). Not called "world models" at the time, but the concept exists.
- **Nguyen & Widrow (1989):** Neural network control using an emulator network — a feedforward model of the system dynamics.

These early "emulator" networks represent the first (feedforward) neural world models, though they operate in fully observable settings and lack the recurrent / partially-observable machinery that Schmidhuber would introduce.

---

## Part 2: The Foundational Era — Predictive SSL Is Born (1990-1993)

This era is almost entirely unknown to modern SSL practitioners. Yet it contains the **core ideas** that JEPA would rediscover 30 years later.

### 2.1 Schmidhuber (1990) — Recurrent Neural World Models
**Paper:** *Making the World Differentiable: On Using Fully Recurrent Self-Supervised Neural Networks for Dynamic Reinforcement Learning and Planning in Non-Stationary Environments* — Technical Report FKI-126-90, TUM, Feb 1990 (revised Nov 1990).

- First paper to use the term **"world model"** for a neural network that learns to predict future states.
- Introduces **planning** with recurrent world models (not just feedforward prediction).
- Also contains the first description of a **generative adversarial** setup: a generator network fighting a predictor network in a minimax game.
- Key insight: a world model learns *predictive representations* of the world that can be used for planning and decision-making, not just perception.

### 2.2 Becker & Hinton (1992) — IMAX / Joint Embedding Architecture
**Paper:** *Self-organizing Neural Network that Discovers Surfaces in Random-Dot Stereograms* — Suzanna Becker, Geoffrey E. Hinton. Nature, 355:161-163, 1992.
**DOI:** 10.1038/355161a0

- **What it does:** Two neural networks receive two different views of the same scene (left and right eye images in a random-dot stereogram). Each produces a latent representation. The objective is to **maximize the mutual information** between these two representations.
- **Why it matters:** This is the earliest instance of a **Joint Embedding Architecture (JEA)** — what LeCun would later call JEA in his response to Schmidhuber's criticism. It also directly inspired Schmidhuber's PMAX.
- **Limitations:** The IMAX approach used a parametric estimate of mutual information that made strong assumptions about the representation distribution (Gaussianity). It was demonstrated on a stereo task but didn't scale to complex datasets.
- **Key omission:** No mechanism to prevent representation collapse (though the mutual information objective naturally penalizes trivial representations to some degree).

### 2.3 Schmidhuber (1991) — PMIN: Predictability MINimization
**Paper:** *Learning Factorial Codes by Predictability Minimization* — Jürgen Schmidhuber. Neural Computation, 4(6):863-879, 1992. Based on TR CU-CS-565-91, Univ. Colorado at Boulder, 1991.
**PDF:** https://sferics.idsia.ch/pub/juergen/factorial.pdf

- A **precursor to adversarial learning** (GANs). Two networks interact:
  1. An **encoder** produces latent representations of the input
  2. A **predictor** tries to predict each unit of the representation from the other units
  3. The encoder *minimizes* this predictability (tries to make each code unit unpredictable from the others)
- **Goal:** Create **factorial codes** (maximally disentangled latent representations) — each dimension carries independent information.
- **Why it matters:** PMIN introduces the adversarial dynamic that later appears in GANs (Goodfellow et al., 2014) and in the collapse-prevention mechanisms of modern SSL.
- **Also predicts parts of latent space from other parts** — but unlike JEPA/PMAX, the goal is to *remove* predictability rather than maximize it.

### 2.4 Schmidhuber & Prelinger (1992/1993) — PMAX: Predictability MAXimization
**Paper:** *Discovering Predictable Classifications* — Jürgen Schmidhuber, Daniel Prelinger. Neural Computation, 5(4):625-635, 1993. Based on TR CU-CS-626-92 (1992).
**PDF:** https://people.idsia.ch/~juergen/predmax1992.pdf

**This paper is essentially JEPA, 30 years earlier.** The core idea:

> Two non-generative neural networks interact: one net tries to create a non-trivial, informative, latent representation of its own input that is **predictable** from the latent representation of the other net's input.

**Architecture (Sec. 2.2 of PMAX):**
- An **autoencoder** (encoder + decoder) encodes input A into a latent representation
- A **predictor network** observes input B (different but related to A) and tries to predict the autoencoder's latent representation
- The autoencoder is trained to make its representation **predictable** from B, while also minimizing reconstruction error on A (preventing it from dropping all information about A)
- The trade-off is controlled by a parameter *ε* where ε=1 means no explicit reconstruction loss — just maximize predictability

**What PMAX already contained (1992):**

| PMAX Section | Modern Equivalent | Year |
|-------------|-------------------|------|
| Sec 2.1: Constrained variance maximization | VICReg variance term | 2022 |
| Sec 2.2: Autoencoder + predictor in latent space | JEPA | 2023 |
| Sec 2.3: Infomax (cross-correlation → identity) | Barlow Twins | 2021 |
| Sec 2.4: PMIN as anti-collapse regularizer | SIGReg, LeJEPA | 2025 |
| Explicit D_l term for collapse prevention | Variance regularization | 2022+ |
| Symmetric (weight-sharing) architecture | Siamese networks | 1993+ |
**Math (Predictability Maximization):**
$$\mathcal{L} = \varepsilon M + (1 - \varepsilon) D_l$$
$T_2$: minimize $\varepsilon M + (1-\varepsilon)D_l$ (predictor). $T_1$: minimize $\varepsilon M$ (encoder, asymmetric).
*Latent learning:* $T_1$ encodes input $A$. $T_2$ sees $B$ (related), predicts $T_1$'s latent. $T_1$ makes representation predictable from $B$ while $D_l$ prevents collapse. Exactly JEPA's mechanism, 30 years earlier. Sec 2.1-2.4 = constrained variance (→VICReg), latent-predictor (→JEPA), Infomax (→Barlow Twins), adversarial anti-collapse (→SIGReg).

**Key theoretical contributions:**
1. **First clear framing of representation learning as predictability in latent space**
2. **Explicit collapse prevention** — the D_l (diversity) term ensures representations don't collapse to a constant
3. **Non-generative** — predicts in latent space, not pixel space
4. **Symmetric and asymmetric** — both weight-sharing and separate encoder/predictor architectures
5. **Empirical demonstration** — outperformed IMAX (Becker & Hinton) on a stereo vision task

**Why this matters for understanding JEPA:**
Michal Valko (leader of the BYOL team, personal communication 2026) explicitly stated:
> *"The JEPA lines are instantiations of PMAX" ... "Barlow Twins: literally Sec 2.3 of PMAX" ... "VICReg: is one section from PMAX."*

### 2.5 LeCun et al. (1993) — Siamese Nets (Contrastive)
**Paper:** *Siamese Neural Networks for One-Shot Image Recognition* (proceedings, 1993) — Yann LeCun, et al.

- First use of **contrastive loss** with Siamese architectures: pull similar pairs together, push dissimilar pairs apart.
- Did NOT cite either Becker & Hinton's IMAX (1992) or Schmidhuber's PMAX (1992) — a point of contention in the priority debate.

---

## Part 3: The Long Gap — Why PMAX Was Forgotten (1993-2015)

Several factors caused the Predictive SSL lineage to go dormant:
1. **Compute constraints** — PMAX's experiments were on tiny problems. The compute to scale these ideas to ImageNet didn't exist yet.
2. **The rise of supervised learning** — ImageNet (2012) made supervised pre-training the dominant paradigm.
3. **Deep learning's focus on discriminative tasks** — classification, detection, segmentation all used supervised labels.
4. **Autoencoders dominated unsupervised learning** — the generative SSL tradition (pixel reconstruction) was the default.

During this period, key developments happened in the **generative SSL** lineage:
- **Denoising Autoencoders** (Vincent et al., 2008)
- **Variational Autoencoders** (Kingma & Welling, 2014)
- **Generative Adversarial Networks** (Goodfellow et al., 2014)

But the Predictive SSL thread was largely dormant.

---

## Part 4: The Pretext Task Revival — SSL's First Modern Wave (2015-2018)

The era that most modern practitioners think of as the "start" of SSL.

### 4.1 Context Prediction — Doersch et al., ICCV 2015
**arXiv:** [1505.05192](https://arxiv.org/abs/1505.05192)
Predict spatial offset between two image patches.
**Math:** Multi-class cross-entropy over 8 spatial offsets:
$$\mathcal{L} = -\mathbb{E}\left[\log \frac{e^{f(P_1, P_2)_r}}{\sum_{k=1}^8 e^{f(P_1, P_2)_k}}\right]$$
*Latent learning:* CNN encoder learns object-part geometry. Latent = penultimate layer of spatial-relation classifier. Forces understanding of object layout. Can be "hacked" via low-level texture cues.

### 4.2 Colorization — Zhang et al., ECCV 2016
**arXiv:** [1603.08511](https://arxiv.org/abs/1603.08511)
Predict color from grayscale. Multi-modal prediction problem (many valid colors) → blurry averages. Generative SSL approach.
**Math:** Multinomial cross-entropy over 313 quantized color bins:
$$\mathcal{L} = -\sum_{h,w} v(Z_{h,w}) \sum_q Z_{h,w,q} \log \hat{Z}_{h,w,q}$$
*Latent learning:* Cross-channel encoder (L → ab). Semantic features emerge because color prediction requires object recognition.

### 4.3 Context Encoders (Inpainting) — Pathak et al., CVPR 2016
**arXiv:** [1604.07379](https://arxiv.org/abs/1604.07379)
Predict missing pixels conditioned on surrounding context. Uses both reconstruction + adversarial loss. Generative SSL.
**Math:** $$\mathcal{L} = \lambda_{\text{rec}} \|M \odot (x - \hat{x})\|_2^2 + \lambda_{\text{adv}} \left(\mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(\hat{x}))]\right)$$
*Latent learning:* Encoder compresses visible context into $z$; decoder reconstructs missing region. $z$ captures both local texture and global semantics.

### 4.4 Jigsaw Puzzles — Noroozi & Favaro, CVPR 2016
**arXiv:** [1603.09246](https://arxiv.org/abs/1603.09246)
Predict permutation of shuffled patches. Forces compositional/relational reasoning.
**Math:** Multi-class over $P$ patch permutations:
$$\mathcal{L} = -\mathbb{E}\left[\log \frac{e^{g([F_1,...,F_9])_p}}{\sum_{k=1}^P e^{g([F_1,...,F_9])_k}}\right],\; F_i = f_\theta(\tilde{P}_i)$$
*Latent learning:* CFN encodes each patch independently. Features $F_i$ capture part semantics for determining spatial arrangement. Spatial SSL (predictive in the spatial layout sense, but classification-based rather than latent-space).

### 4.5 Rotation Prediction — Gidaris et al., ICLR 2018
**arXiv:** [1803.07728](https://arxiv.org/abs/1803.07728)
Predict 2D rotation applied to image. Forces understanding of canonical object orientation. Penalizes rotation invariance.
**Math:** 4-way classification over rotations:
$$\mathcal{L} = -\frac{1}{|X_i|} \sum_{x \in X_i} \log F_y(g(x|y)|\theta)$$
*Latent learning:* Encoder learns canonical orientation. Latent = penultimate layer — discards rotation while retaining semantics needed for "up-ness."

### Theoretical Lessons from the Pretext Task Era
1. **Shortcut learning is pervasive** — models find low-level shortcuts that let them "solve" pretext tasks without learning semantics
2. **Each task teaches a specific bias** — rotation invariance, spatial layout, color statistics. None capture general-purpose understanding.
3. **Transfer is limited** — pretext tasks beat random initialization but still lagged far behind supervised pre-training.
4. **No principled theory** — no framework for designing or comparing pretext tasks. Progress was empirical.
5. **The hidden connection** — most pretext tasks (jigsaw, rotation, context) were actually **classification** problems, not predictive SSL in the latent-space sense. They largely rediscovered, without citing, ideas from the 1990-era literature.

---

## Part 5: The Contrastive Revolution (2018-2021)

### 5.1 CPC & InfoNCE — The Theoretical Breakthrough
**Paper:** *Representation Learning with Contrastive Predictive Coding* — van den Oord et al., 2018
**arXiv:** [1807.03748](https://arxiv.org/abs/1807.03748)

- Introduced **InfoNCE loss**: a contrastive loss that maximizes a lower bound on mutual information between context and future observations
- **Predict future in latent space** using autoregressive models — a conceptual return to the world model idea but with modern machinery
**Math (InfoNCE):**
$$\mathcal{L}_N = -\mathbb{E}\left[\log \frac{\exp(z_{t+k}^\top W_k c_t)}{\sum_{x_j \in X} \exp(z_j^\top W_k c_t)}\right]$$
Optimal critic $f^*(x,c)=p(x|c)/p(x)$ gives $I(x_{t+k};c_t) \geq \log N - \mathcal{L}_N$.
*Latent learning:* Encoder $z_t$ + autoregressive $c_t$. InfoNCE forces $z_t$ to preserve info predictive of future — extracts slow features.
- **Theoretical grounding:** contrastive learning = mutual information maximization between views

### 5.2 MoCo — Momentum Contrast (He et al., CVPR 2020)
**arXiv:** [1911.05722](https://arxiv.org/abs/1911.05722)
- **Dictionary look-up** framing of contrastive learning
- **Momentum encoder** + queue for large, consistent negative sample set
- Does NOT cite PMAX or Becker & Hinton
**Math (InfoNCE with queue + momentum encoder):**
$$\mathcal{L}_q = -\log \frac{\exp(q \cdot k_+ / \tau)}{\sum_{i=0}^{65536} \exp(q \cdot k_i / \tau)}$$
Momentum: $\theta_k \leftarrow 0.999\,\theta_k + 0.001\,\theta_q$.
*Latent learning:* Dictionary look-up with $K=65536$ negatives from queue. Momentum encoder provides stable targets.

### 5.3 SimCLR — Simple Framework (Chen et al., ICML 2020)
**arXiv:** [2002.05709](https://arxiv.org/abs/2002.05709)
- No memory bank, no momentum — just large batch + aggressive augmentation + projection head
- Key findings: 1) Augmentation composition is critical, 2) Projection head improves representations, 3) Larger batches help
- **Theoretical tension:** The quality of representations depends *entirely* on hand-crafted data augmentations — a hidden form of supervision
**Math (NT-Xent):**
$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k\neq i]}\exp(\text{sim}(z_i,z_k)/\tau)}$$
Gradient: $\nabla = \frac{1}{\tau}\frac{e^{u^\top v_+/\tau}}{Z(u)}v_+ - \sum_{v^-}\frac{e^{u^\top v^-/\tau}}{Z(u)}v^-$.
*Latent learning:* Projector $g$ peels off augmentation info so $h=f(x)$ retains semantics. $\tau$ controls negative hardness.

### 5.4 SwAV — Swapped Clustering (Caron et al., NeurIPS 2020)
**arXiv:** [2006.09882](https://arxiv.org/abs/2006.09882)
- Instead of comparing features, compare **cluster assignments**
- **Multi-crop** augmentation strategy — many low-resolution views
- Bridges contrastive learning and prototype-based methods
**Math (Swapped Prediction):**
$$\mathcal{L}(z_t,z_s) = -\sum_k q_s^{(k)}\log p_t^{(k)} - \sum_k q_t^{(k)}\log p_s^{(k)}$$
where $p^{(k)} = \text{softmax}(z^\top c_k/\tau)$, $q = \text{Sinkhorn-Knopp}(p)$.
*Latent learning:* Compares cluster assignments (not features) across views. Prototypes $\{c_k\}$ are learnable centers.

### 5.5 BYOL — Bootstrap Your Own Latent (Grill et al., NeurIPS 2020)
**arXiv:** [2006.07733](https://arxiv.org/abs/2006.07733)
- **Shock to the field:** State-of-the-art WITHOUT negative samples
- Online network predicts target network's representation of a different view
- Target network = slow-moving EMA of online network
- **The critical connection:** BYOL is the first method to **make PMAX work at scale on ImageNet** — specifically in the ε=1 regime (no explicit collapse prevention term)
- BYOL's EMA + stop-gradient + predictor asymmetry = implicit anti-collapse toolkit
- Michal Valko (BYOL lead, 2026): "BYOL first made the PMAX skeleton work at scale on ImageNet"
**Math (MSE + predictor + EMA):**
$$\mathcal{L}_\theta = \left\|\frac{q_\theta(z_\theta)}{\|q_\theta(z_\theta)\|} - \frac{z'_\xi}{\|z'_\xi\|}\right\|_2^2$$
Online $f_\theta\to g_\theta\to q_\theta$, Target $f_\xi\to g_\xi$ (EMA: $\xi\leftarrow \tau\xi+(1-\tau)\theta$). Stop-gradient on target.
*Latent learning:* $h_\theta=f_\theta(x)$ learned by predicting target's output. Predictor $q$ creates asymmetry — without it, collapse is immediate.

### 5.6 SimSiam — The Minimal Siamese (Chen & He, CVPR 2021)
**arXiv:** [2011.10566](https://arxiv.org/abs/2011.10566)
- Minimalist: stop-gradient is the **only** thing preventing collapse
- Without stop-gradient → complete collapse in all experiments
- Proposed **implicit EM interpretation** of stop-gradient
**Math (Neg cosine + stop-gradient):**
$$\mathcal{L} = \frac{1}{2}\mathcal{D}(p_1,\text{sg}(z_2)) + \frac{1}{2}\mathcal{D}(p_2,\text{sg}(z_1)),\; \mathcal{D}(p,z) = -\frac{p}{\|p\|}\cdot\frac{z}{\|z\|}$$
*Latent learning:* Implicit EM: E-step fixes encoder, computes $\eta_x = \text{sg}(f(x_2))$; M-step minimizes $\mathbb{E}\|f_\theta(x_1)-\eta_x\|^2$. Collapse is stationary but not an attractor.

---

## Part 6: Non-Contrastive Methods — Explicit Collapse Prevention (2021-2022)

These methods explicitly regularize the representation to prevent collapse, returning to PMAX's philosophy of explicit D_l terms.

### 6.1 Barlow Twins (Zbontar et al., NeurIPS 2021)
**arXiv:** [2103.03230](https://arxiv.org/abs/2103.03230)
- Cross-correlation matrix between two branches → identity matrix
- Off-diagonal terms penalized (redundancy reduction)
- **PMAX connection:** Michal Valko notes this is "literally Sec 2.3 of PMAX with a different name" (the Infomax section)
**Math (Cross-correlation → Identity):**
$$C_{ij} = \frac{\sum_b z_{b,i}^A z_{b,j}^B}{\sqrt{\sum_b (z_{b,i}^A)^2}\sqrt{\sum_b (z_{b,j}^B)^2}},\; \mathcal{L} = \sum_i (1-C_{ii})^2 + \lambda\sum_{i\neq j} C_{ij}^2$$
*Latent learning:* Diagonal=1 forces invariance. Off-diagonal=0 decorrelates dimensions. Each dimension captures distinct info — redundancy reduction.

### 6.2 VICReg (Bardes et al., ICLR 2022)
**arXiv:** [2105.04906](https://arxiv.org/abs/2105.04906)
- Variance + Invariance + Covariance regularization
- Explicit variance hinge prevents dimensional collapse
- **PMAX connection:** "VICReg: variance hinge (Sec. 2.1) plus covariance penalty (Sec. 2.3) applied simultaneously
**Math (Variance + Invariance + Covariance):**
$$\mathcal{L} = \lambda\frac{1}{N}\sum_i\|z_i-z'_i\|_2^2 + \mu\frac{1}{d}\sum_j\max(0,\gamma-\sqrt{\text{Var}(z_{:,j})+\epsilon}) + \nu\frac{1}{d}\sum_{i\neq j}[C(Z)]_{ij}^2$$
*Latent learning:* Variance hinge forces std$_j\geq\gamma$ (anti-collapse). Covariance decorrelates dims. Each dim independently informative., so indeed that paper is one section from PMAX"

---

## Part 7: The Reconstruction Revival — Generative SSL Returns (2021-2022)

### 7.1 BEiT (Bao et al., ICLR 2022)
**arXiv:** [2106.08254](https://arxiv.org/abs/2106.08254)
- BERT-style masked prediction for vision
- Predict **discrete visual tokens** (from dVAE tokenizer) — a generative SSL approach with discrete latent targets
- First successful adaptation of masked language modeling to vision
**Math (Masked image modeling with discrete tokens):**
$$\mathcal{L} = -\mathbb{E}\left[\sum_{i\in M} \log p(z_i \mid \hat{x}_{I\setminus M})\right]$$
where $z_i$ = dVAE discrete visual tokens.
*Latent learning:* ViT encoder sees only unmasked patches. Predicts discrete token IDs for masked positions (classification, not regression).

### 7.2 MAE — Masked Autoencoders (He et al., CVPR 2022)
**arXiv:** [2111.06377](https://arxiv.org/abs/2111.06377)
- Mask 75% of patches, reconstruct **raw pixels** with asymmetric encoder-decoder
- Masking ratio is the key: 75% prevents trivial interpolation
- Despite predicting pixels (the least abstract target), it learns semantic representations because the task difficulty forces abstraction
- **The paradox:** MAE shows that even a pure generative SSL objective can produce good representations — if the task is hard enough.
**Math (Pixel MSE with 75% masking):**
$$\mathcal{L} = \frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}} \|x_i - \hat{x}_i\|_2^2$$
Asymmetric encoder-decoder: encoder processes only visible patches.
*Latent learning:* 75% masking prevents interpolation — must infer global structure. Despite pixel targets, task difficulty forces abstraction. This challenges the "predictive SSL is fundamentally better" narrative.

### 7.3 data2vec (Baevski et al., 2022)
**arXiv:** [2202.03555](https://arxiv.org/abs/2202.03555)
- Unifies SSL across speech, NLP, vision: mask input, predict **contextualized latent representations** from teacher network
- Not a reconstruction method (predicts latent targets, not raw data)
- Uses self-distillation (teacher-student with EMA)
- **Conceptual bridge** between generative (masked prediction) and predictive (latent targets) SSL
**Math (MSE in latent space, teacher-student):**
$$\mathcal{L} = \frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}} \frac{\|y_i - e(x_i)\|_2^2}{\text{Var}(y)}$$
Teacher (EMA) produces contextualized latent targets $y_i$ from unmasked input.
*Latent learning:* Student predicts teacher's contextualized latents for masked positions — bridges generative masking and predictive SSL.

### 7.4 Theoretical Analysis of Masked Autoencoders & Masked Image Modeling

The success of MAE and MIM spawned a mini-literature trying to understand *why* pixel-space prediction produces good representations. These analyses are critical context for JEPA because they reveal the *mechanisms* by which masking-based SSL works.

#### 7.4.1 MIM as Occlusion-Invariant Feature Learning (Kong & Zhang, 2022)
**Paper:** *Understanding Masked Image Modeling via Learning Occlusion Invariant Feature*
**arXiv:** [2208.04164](https://arxiv.org/abs/2208.04164)

- **Core claim:** MIM implicitly learns **occlusion-invariant features** — representations that are stable under random patch removal
- **Unified formulation:** MIM can be relaxed into equivalent siamese form:
  $$\mathcal{L} = \underbrace{\mathbb{E}\left[\|f_\theta(x \odot M) - f_\xi(x)\|^2\right]}_{\text{occlusion invariance}} + \underbrace{\text{reconstruction term}}_{\text{auxiliary}}$$
- **Unified framework:** By relaxing MIM into an equivalent siamese-like formulation, the paper shows MIM methods can be interpreted in a unified framework with conventional contrastive methods. The only differences are:
  - **(a) Data transformations** — what invariance to learn (occlusion vs color jitter/crop)
  - **(b) Similarity measurement** — how to compare representations
- **Key finding about MAE:** The success of MAE relates *little* to the choice of similarity function (L2 reconstruction works as well as contrastive loss). What matters is the **occlusion-invariant feature introduced by masking** — it turns out to be a favored initialization for Vision Transformers.
- **Important caveat:** The learned occlusion-invariant feature "could be **less semantic**" — it captures low/mid-level structure rather than high-level semantics. This is a critical limitation.

**Why this matters for JEPA:**
This paper reveals that MAE works by learning *occlusion invariance* rather than *semantic invariance*. The masking creates a powerful but narrow inductive bias. JEPA's strategy — predict in representation space, not pixel space — aims to learn *semantic* invariance directly, without relying on the pixel reconstruction bottleneck.

#### 7.4.2 MAE From a Local Contrastive Perspective (Li et al., 2023)
**Paper:** *Understanding Masked Autoencoders From a Local Contrastive Perspective*
**arXiv:** [2310.01994](https://arxiv.org/abs/2310.01994)

- **Core claim:** MAE's reconstructive objective can be **explicitly re-expressed as a local contrastive form** at the patch level
- **What MAE learns:**
  1. **Invariance to random masking** — similar to data augmentation
  2. **Distribution consistency** between learned token embeddings and original image patches
- **Role of the decoder:** The decoder actually learns to *separate* tokens into visible vs. masked categories — it functions as a learned similarity metric
- **Random masking as dual-purpose:** (a) It acts as **data augmentation** for the visible patches, and (b) it restricts the **effective receptive field** of the encoder, encouraging local-to-global reasoning

**Why this matters for JEPA:**
This shows MAE is not "pure" generative SSL — it has a *contrastive character* at the patch level. The distinction between generative and predictive SSL is blurrier than commonly assumed. JEPA makes this contrastive character explicit and operates on entire blocks rather than individual patches.

#### 7.4.3 Complementary Strengths: MIM vs Contrastive Learning
**Paper:** *Contrastive Masked Autoencoders are Stronger Vision Learners* (Huang et al., 2022)
**arXiv:** [2207.13532](https://arxiv.org/abs/2207.13532)

- MIM learns **good local features** (fine-grained, spatially aware) but has **limited discriminability** (instance-level separation)
- Contrastive learning learns **good global discriminability** but can miss fine-grained local structure
- The two paradigms are **complementary** — combining them (as in CMAE) outperforms either alone

**Paper:** *Hybrid Distillation: Connecting Masked Autoencoders with Contrastive Learners* (2023)
**arXiv:** [2306.15876](https://arxiv.org/abs/2306.15876)

- MIM provides **higher diversity and more local attention** across transformer layers
- CL/supervised pre-training provides **long-range global patterns and discrimination**
- The challenge is that naive combination loses the benefits of one or the other — careful distillation is needed

**Why this matters for JEPA:**
JEPA is positioned as a third paradigm that avoids the weaknesses of both:
- It uses **data-intrinsic structure** (masking patterns) like MIM, avoiding hand-crafted augmentations
- It operates in **representation space** like contrastive methods, avoiding pixel-level targets
- Whether it actually captures the *best of both worlds* or sacrifices something from each is a key question for analysis

#### 7.4.4 Target Representations Don't Matter Much (Xingbin Liu et al., 2022)
**Paper:** *Exploring Target Representations for Masked Autoencoders* (dBOT)
**arXiv:** [2209.03735](https://arxiv.org/abs/2209.03735) — *Note: search on Semantic Scholar*

- **Surprising finding:** The choice of target representation (pixels, tokens, HOG features, etc.) is **surprisingly unimportant** — different targets yield similarly behaved models
- This suggests the **masking operation itself**, not the specific reconstruction target, is the primary driver of representation quality
- This further blurs the generative-vs-predictive distinction: what matters is not *what you predict* but the **nature of the prediction task** (masked, incomplete input)

**Why this matters for JEPA:**
If the target representation doesn't matter much for MAE, then JEPA's choice to predict in latent space is not necessarily *superior* to pixel-space prediction — it's simply a different design point that may be more *computationally efficient* and better suited for multi-scale semantic learning.

#### 7.4.5 The MAE Survey (Chen et al., 2022)
**Paper:** *A Survey on Masked Autoencoder for Self-supervised Learning in Vision and Beyond*
**arXiv:** [2208.00173](https://arxiv.org/abs/2208.00173)

- Comprehensive survey tracing the development of masked autoencoders
- Notes that masked prediction has become a "de facto standard SSL practice" — first in NLP (BERT), then in vision (BEiT, MAE)
- Highlights the historical connection to **denoising autoencoders** — masked autoencoding is a specialized form of denoising where the noise is structured patch removal
- Surveys applications beyond vision: video, multimodal, graph, medical imaging

---

## Part 8: The Theoretical Tensions — What Remained Unresolved Before JEPA

### The Three Paradigms and Their Problems

| Paradigm | Example | Strength | Weakness |
|----------|---------|----------|----------|
| **Generative** (reconstruct pixels/tokens) | MAE, BEiT | No collapse, data-intrinsic | Predicts low-level targets, computationally expensive decoder |
| **Contrastive** (pull/push in embedding space) | SimCLR, MoCo | Operates in abstraction space | Depends on hand-crafted views, needs negatives |
| **Non-Contrastive** (regularize embedding statistics) | BYOL, Barlow Twins, VICReg | No negatives, representation-level | Collapse prevention is fragile/theoretically obscure |

### The Five Open Problems

1. **The Hand-Crafted View Problem** — Contrastive methods depend on augmentation recipes engineered by human intuition. The model's invariances come from human assumptions, not from the data's intrinsic structure. This is a hidden form of supervision.

2. **The Abstraction Gap** — Generative methods predict low-level targets (pixels) but want high-level representations. The gap between objective and representation is bridged only by making the task extremely hard (75% masking). Can we instead predict directly at the right level of abstraction?

3. **The Collapse Problem** — Why does BYOL work? Why does SimSiam work with just stop-gradient? The non-contrastive methods are empirically robust but theoretically mysterious. PMAX had an explicit solution (D_l term) that was abandoned in favor of implicit architectural tricks.

4. **The World Model Gap** — SSL for vision is static. Autonomous intelligence requires agents that can **predict and plan** in latent space — learning how the world evolves. This requires moving beyond image representations to predictive world models.

5. **The Information-Theoretic Incompleteness** — The InfoNCE bound is loose. We lack a unified framework that explains when and why different SSL methods work.

---

## Part 9: The Return to Predictive SSL — JEPA (2023-2025)

### 9.1 I-JEPA (Assran et al., ICCV 2023)
**arXiv:** [2301.08243](https://arxiv.org/abs/2301.08243)

- **Core idea:** From a single context block, predict representations of various target blocks in the same image
- **Non-generative:** predicts in representation space, not pixel space
- **Data-intrinsic view generation:** uses image structure (spatial masking), not hand-crafted augmentations
- **Multi-scale prediction heads:** different blocks predict at different scales → semantic hierarchy
- **Key design choice:** Target blocks must be large enough (semantic) and context block spatially distributed
**Math (Multi-scale latent prediction):**
$$\mathcal{L} = \sum_{k=1}^K \mathbb{E}\left[\|g_\phi^{(k)}(s_c) - \bar{s}_t^{(k)}\|_2^2\right]$$
where $s_c = f_\theta(x|_{B_c})$, $\bar{s}_t^{(k)} = \bar{f}_\xi(x|_{B_t^{(k)}})$ (momentum encoder).
*Latent learning:* Context encoder learns representations predictive of target blocks at multiple scales. No negatives — collapse prevented by EMA + predictor asymmetry. Multi-scale heads create semantic hierarchy.

### 9.2 V-JEPA (Bardes et al., 2024)
**arXiv:** [2404.08471](https://arxiv.org/abs/2404.08471) — *Revisiting Feature Prediction for Learning Visual Representations from Video*

- **Temporal prediction:** given a video context, predict representations of future frames
- **No reconstruction, no text, no negatives, no pretrained encoders**
- Learned on 2 million videos from public datasets

### 9.3 LeJEPA / SIGReg (LeCun et al., 2025)
A crucial development: this version **dropped EMA and stop-gradient** entirely and reverted to an explicit anti-collapse regularizer (SIGReg).
> "The trajectory went: PMAX's explicit D_l (1992) → BYOL's implicit architectural tricks (2020) → LeCun's JEPAs using BYOL tricks (2023-2024) → back to explicit D_l in LeJEPA (2025). The field spent five years exploring the ε=1 regime opened up by BYOL, then came home to PMAX's original ε&lt;1 design." — Michal Valko (2026)

---

## The Complete Timeline

```
Pre-1990:
  Barlow (1961)    → Redundancy reduction principle
  Werbos (1987)    → Feedforward neural world models (system ID)
  Nguyen & Widrow  → Neural network control with emulator networks
  (1989)

1990-1993 (The Foundational Era of Predictive SSL):
  Schmidhuber      → First RNN world model + GAN precursor
  (1990)
  Schmidhuber      → PMIN: Predictability MINimization (adversarial SSL)
  (1991)
  Becker & Hinton  → IMAX: Mutual information maximization between views
  (1992)             (First Joint Embedding Architecture)
  Schmidhuber &    → PMAX: Predictability MAXimization
  Prelinger (1992/ →   Sec 2.1: Constrained variance (VICReg)
  1993)              Sec 2.2: Latent predictor (JEPA)
                     Sec 2.3: Infomax (Barlow Twins)
                     Sec 2.4: Anti-collapse with PMIN (SIGReg)
  LeCun (1993)     → First Siamese neural nets (contrastive)

1993-2015 (The Gap):
  DAE, VAE, GAN   → Generative SSL dominates unsupervised learning
  ImageNet era     → Supervised pre-training dominates representation learning

2015-2018 (Pretext Task Revival):
  Context Prediction  → Doersch et al., ICCV 2015
  Colorization        → Zhang et al., ECCV 2016
  Jigsaw Puzzles      → Noroozi & Favaro, CVPR 2016
  Context Encoders    → Pathak et al., CVPR 2016
  Rotation            → Gidaris et al., ICLR 2018

2018-2021 (Contrastive Revolution):
  CPC/InfoNCE       → van den Oord et al., 2018
  MoCo              → He et al., CVPR 2020
  SimCLR            → Chen et al., ICML 2020
  SwAV              → Caron et al., NeurIPS 2020
  BYOL              → Grill et al., NeurIPS 2020 ⬅ first PMAX at scale
  SimSiam           → Chen & He, CVPR 2021
  MoCo v3           → Chen et al., 2021

2021-2022 (Non-Contrastive + Reconstruction):
  Barlow Twins      → Zbontar et al., NeurIPS 2021 ⬅ PMAX Sec 2.3
  BEiT              → Bao et al., ICLR 2022
  VICReg            → Bardes et al., ICLR 2022 ⬅ PMAX Sec 2.1+2.3
  MAE               → He et al., CVPR 2022
  data2vec          → Baevski et al., 2022

2023-2025 (JEPA Era):
  I-JEPA            → Assran et al., ICCV 2023
  V-JEPA            → Bardes et al., 2024
  LeJEPA/SIGReg     → LeCun et al., 2025 ⬅ returns to explicit D_l
```

---

## Key Takeaway

JEPA is not a radical departure from SSL history — it is the **latest expression** of a lineage that began with PMAX in 1992. What JEPA adds to its 30-year-old predecessor:
1. **Vision Transformers** — the architectural backbone that makes latent-space prediction practically effective
2. **Multi-scale prediction** — predicting at multiple block sizes creates a semantic hierarchy
3. **Scaling** — modern compute enables evaluation on ImageNet-scale tasks
4. **Empirical validation** — the strongest demonstration yet that predicting in latent space with data-intrinsic views works at scale

The next note in this series ([02-ssl-theory](/notes/jepa_notes/02-ssl-theory)) dives into the theoretical framework of JEPA itself: its energy-based model framing, its approach to the collapse problem, and the theoretical claims about semantic representation learning.

---

## References

### Foundational Pre-1990
1. Barlow, H. B. (1961). *Possible principles underlying the transformation of sensory messages.* In W. Rosenblith (Ed.), Sensory Communication, pp. 217-234. MIT Press.
2. Werbos, P. J. (1987). *Building and Understanding Adaptive Systems: A Statistical/Numerical Approach to Factory Automation and Brain Research.* IEEE Trans. Systems, Man, and Cybernetics.
3. Nguyen, D. H., & Widrow, B. (1989). *Neural networks for self-learning control systems.* IEEE Control Systems Magazine. — [https://doi.org/10.1109/37.55119](https://doi.org/10.1109/37.55119)

### The Foundational Era (1990-1993)
4. Schmidhuber, J. (1990). *Making the World Differentiable: On Using Fully Recurrent Self-Supervised Neural Networks for Dynamic Reinforcement Learning and Planning in Non-Stationary Environments.* Technical Report FKI-126-90, TUM. — Earliest RNN-based world model + GAN precursor. — [PDF](https://people.idsia.ch/~juergen/FKI-126-90ocr.pdf)
5. Schmidhuber, J. (1992). *Learning Factorial Codes by Predictability Minimization.* Neural Computation, 4(6):863-879. (Based on TR CU-CS-565-91, 1991) — PMIN: adversarial SSL, precursor to GANs. — [https://doi.org/10.1162/neco.1992.4.6.863](https://doi.org/10.1162/neco.1992.4.6.863) — [PDF](https://sferics.idsia.ch/pub/juergen/factorial.pdf)
6. Becker, S., & Hinton, G. E. (1992). *Self-organizing Neural Network that Discovers Surfaces in Random-Dot Stereograms.* Nature, 355:161-163. — IMAX: first Joint Embedding Architecture. — [https://doi.org/10.1038/355161a0](https://doi.org/10.1038/355161a0)
7. Schmidhuber, J., & Prelinger, D. (1993). *Discovering Predictable Classifications.* Neural Computation, 5(4):625-635. (Based on TR CU-CS-626-92, 1992) — PMAX: the original JEPA. — [https://doi.org/10.1162/neco.1993.5.4.625](https://doi.org/10.1162/neco.1993.5.4.625) — [PDF](https://people.idsia.ch/~juergen/predmax1992.pdf)
8. Schmidhuber, J. (1993). *Network Architectures, Objective Functions, and Chain Rule.* Habilitation thesis, TUM. Sec. 5.5 on Predictability Maximization. — [PDF](https://sferics.idsia.ch/pub/juergen/habilitation.pdf)

### Pretext Tasks
9. Doersch, C., Gupta, A., & Efros, A. A. (2015). *Unsupervised Visual Representation Learning by Context Prediction.* ICCV. — [arXiv:1505.05192](https://arxiv.org/abs/1505.05192)
10. Zhang, R., Isola, P., & Efros, A. A. (2016). *Colorful Image Colorization.* ECCV. — [arXiv:1603.08511](https://arxiv.org/abs/1603.08511)
11. Noroozi, M., & Favaro, P. (2016). *Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles.* CVPR. — [arXiv:1603.09246](https://arxiv.org/abs/1603.09246)
12. Pathak, D., Krähenbühl, P., Donahue, J., Darrell, T., & Efros, A. A. (2016). *Context Encoders: Feature Learning by Inpainting.* CVPR. — [arXiv:1604.07379](https://arxiv.org/abs/1604.07379)
13. Gidaris, S., Singh, P., & Komodakis, N. (2018). *Unsupervised Representation Learning by Predicting Image Rotations.* ICLR. — [arXiv:1803.07728](https://arxiv.org/abs/1803.07728)

### Contrastive Learning
14. van den Oord, A., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding.* — [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
15. He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). *Momentum Contrast for Unsupervised Visual Representation Learning.* CVPR. — [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)
16. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* ICML. — [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
17. Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., & Joulin, A. (2020). *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.* NeurIPS. — [arXiv:2006.09882](https://arxiv.org/abs/2006.09882)
18. Grill, J.-B., et al. (2020). *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.* NeurIPS. — [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
19. Chen, X., & He, K. (2021). *Exploring Simple Siamese Representation Learning.* CVPR. — [arXiv:2011.10566](https://arxiv.org/abs/2011.10566)
20. Chen, X., Xie, S., & He, K. (2021). *An Empirical Study of Training Self-Supervised Vision Transformers.* — [arXiv:2104.02057](https://arxiv.org/abs/2104.02057)

### Non-Contrastive / Regularization
21. Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). *Barlow Twins: Self-Supervised Learning via Redundancy Reduction.* NeurIPS. — [arXiv:2103.03230](https://arxiv.org/abs/2103.03230)
22. Bardes, A., Ponce, J., & LeCun, Y. (2022). *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR. — [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)

### Reconstruction / Masked Modeling
23. Bao, H., Dong, L., Piao, S., & Wei, F. (2022). *BEiT: BERT Pre-Training of Image Transformers.* ICLR. — [arXiv:2106.08254](https://arxiv.org/abs/2106.08254)
24. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). *Masked Autoencoders Are Scalable Vision Learners.* CVPR. — [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
25. Baevski, A., Hsu, W.-N., Xu, Q., Babu, A., Gu, J., & Auli, M. (2022). *data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language.* — [arXiv:2202.03555](https://arxiv.org/abs/2202.03555)
26. Chen, Y., et al. (2022). *A Survey on Masked Autoencoder for Self-supervised Learning in Vision and Beyond.* — [arXiv:2208.00173](https://arxiv.org/abs/2208.00173)

### Theoretical Analysis of Masked Image Modeling
27. Kong, X., & Zhang, Y. (2022). *Understanding Masked Image Modeling via Learning Occlusion Invariant Feature.* — [arXiv:2208.04164](https://arxiv.org/abs/2208.04164) — Shows MIM = siamese learning with occlusion invariance; the masking, not the target, drives representation quality.
28. Li, J. et al. (2023). *Understanding Masked Autoencoders From a Local Contrastive Perspective.* — [arXiv:2310.01994](https://arxiv.org/abs/2310.01994) — Re-expresses MAE as local contrastive learning at patch level.
29. Huang, Z., et al. (2022). *Contrastive Masked Autoencoders are Stronger Vision Learners.* — [arXiv:2207.13532](https://arxiv.org/abs/2207.13532) — MIM learns local features, CL learns global discriminability; they are complementary.
30. Liu, X. et al. (2022). *Exploring Target Representations for Masked Autoencoders (dBOT).* — [arXiv:2209.03735](https://arxiv.org/abs/2209.03735) — Different targets yield similarly behaved models; masking itself drives learning.
31. Dong, J. et al. (2023). *Hybrid Distillation: Connecting Masked Autoencoders with Contrastive Learners.* — [arXiv:2306.15876](https://arxiv.org/abs/2306.15876) — MIM provides diversity/local attention, CL provides global discrimination.

### JEPA
32. Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* ICCV. — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
33. Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., & Ballas, N. (2024). *Revisiting Feature Prediction for Learning Visual Representations from Video.* — [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)

### Theoretical & Meta
34. LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence.* OpenReview. — [https://openreview.net/forum?id=BZ5a1r-kVsf](https://openreview.net/forum?id=BZ5a1r-kVsf)
35. Tschannen, M., Djolonga, J., Rubenstein, P. K., Gelly, S., & Lucic, M. (2019). *On Mutual Information Maximization for Representation Learning.* — [arXiv:1907.13625](https://arxiv.org/abs/1907.13625)
36. Schmidhuber, J. (2026). *Who Invented "JEPA"?* Technical Note IDSIA-3-22. — [https://people.idsia.ch/~juergen/who-invented-jepa.html](https://people.idsia.ch/~juergen/who-invented-jepa.html) — Comprehensive priority analysis with LeCun's response.
