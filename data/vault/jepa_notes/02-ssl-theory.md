---
title: "The Mathematics of Latent Prediction & Collapse in SSL"
tags: [ssl, self-supervised-learning, theory, collapse, information-theory, jepa]
created: 2026-06-07
status: draft
aliases: [SSL Theory, Collapse Mathematics, Predictive SSL Theory]
---

# The Mathematics of Latent Prediction & Collapse in Self-Supervised Learning

## Overview

This note develops the mathematical theory underlying Self-Supervised Learning, with a focus on **predictive SSL** — methods that learn representations by predicting one view from another in latent space. We cover:

1. What we are actually estimating and why it is fundamentally difficult
2. The information-theoretic foundation of SSL objectives
3. The collapse problem: complete, dimensional, and its spectral characterization
4. A catalog of mathematical remedies (exact loss functions and why they work)
5. JEPA's mathematical distinctiveness
6. Open theoretical questions

---

## Part 1: The Fundamental Problem Setup

### 1.1 What Are We Estimating?

Let $X \in \mathcal{X}$ be the input space (images) and let $Z \in \mathbb{R}^d$ be a latent representation space. SSL learns an encoder $f_\theta: \mathcal{X} \to \mathbb{R}^d$ such that $Z = f_\theta(X)$ captures **semantic content** — information about objects, scenes, and their relationships — while being **invariant** to nuisance factors (lighting, viewpoint, texture variations).

In predictive SSL specifically, we have two views $x$ and $x^+$ of the same underlying data point (e.g., two crops of the same image, two frames of the same video, two blocks of the same image). The goal is to learn representations $z = f_\theta(x)$ and $z^+ = f_\theta(x^+)$ that are **predictable** from each other — i.e., there exists a predictor $g_\phi$ such that $g_\phi(z) \approx z^+$.

**Formally:** We want to find $f_\theta$ such that the conditional expectation $\mathbb{E}[Z^+ | Z]$ preserves the information in $Z^+$ about the underlying semantic state $S$, while discarding view-specific noise $N$:

$$Z = f_\theta(X) \approx h(S) \quad \text{where } X = T(S, N)$$

Here $T$ is a stochastic transformation (data augmentation, cropping, masking) that produces a view from the latent state $S$ and noise $N$. The representation $Z$ should capture $S$ but not $N$.

### 1.2 Why Is This Difficult?

**Difficulty 1: The invariance-objective contradiction.** The natural objective — make $z$ and $z^+$ close — has a trivial minimum: make $f_\theta$ constant. This is the **collapse problem**.

$$\mathcal{L}_{\text{naive}} = \mathbb{E}\left[\|f_\theta(x) - f_\theta(x^+)\|^2\right] \quad \Rightarrow \quad f_\theta(x) = c \text{ is global minimum}$$

**Difficulty 2: What should be invariant vs. what should be preserved?** There is no ground-truth signal telling us which variations are semantic (should be preserved) and which are nuisances (should be discarded). Data augmentations encode human assumptions about this.

**Difficulty 3: The abstraction hierarchy.** Semantics exist at multiple scales — a car wheel and a car are both "semantic" but at different levels. No single objective automatically discovers this hierarchy.

**Difficulty 4: The noise-information trade-off.** To discard noise, the representation must lose information. But how much and what kind? Too much information loss → collapse. Too little → overfitting to nuisance.

**Difficulty 5: The evaluation gap.** SSL objectives optimize for a surrogate (contrastive, reconstruction, prediction) but we evaluate on downstream tasks (classification, detection). The gap between surrogate and downstream is not formally characterized.

---

## Part 2: Information-Theoretic Foundations

### 2.1 InfoNCE and Mutual Information Lower Bound

**Source:** *Representation Learning with Contrastive Predictive Coding* (van den Oord et al., 2018) — [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)

The InfoNCE loss for a set of $N$ samples $\{x_i\}_{i=1}^N$ with one positive $x^+$ and $N-1$ negatives $\{x_j^-\}_{j=1}^{N-1}$:

$$\mathcal{L}_{\text{NCE}} = -\mathbb{E}\left[\log \frac{f(x^+, c)}{\sum_{j=1}^N f(x_j, c)}\right]$$

where $c$ is a context (e.g., in CPC, this is the representation of the past), and $f(x, c) = \exp(z_x^\top z_c / \tau)$ is a "critic" scoring compatibility.

**Key theoretical result:** The optimal critic satisfies:

$$f^*(x, c) = \frac{p(x|c)}{p(x)}$$

which is a density ratio. Substituting this back:

$$\mathcal{L}_{\text{NCE}}^* = -\mathbb{E}\left[\log \frac{p(x^+|c)/p(x^+)}{p(x^+|c)/p(x^+) + \sum_{j=1}^{N-1} p(x_j^-|c)/p(x_j^-)}\right]$$

This yields a lower bound on mutual information:

$$I(X; C) \geq \log(N) - \mathcal{L}_{\text{NCE}}$$

**Proof sketch:**
- Let $X$ be the positive sample and $\{X_j\}_{j=1}^N$ be the set containing 1 positive + $N-1$ negatives, all drawn independently from $p(x|)$ except the positive which depends on $c$.
- The probability that the critic correctly identifies the positive among $N$ candidates is bounded by $\frac{p(x^+|c)}{p(x^+)}$.
- Jensen's inequality on $\log(1 + \frac{N-1}{N} \mathbb{E}[\frac{p(x^-|c)}{p(x^-)} / \frac{p(x^+|c)}{p(x^+)}])$ gives the bound.

**Important caveat:** The bound tightens as $N \to \infty$, but for finite $N$ it is limited to $\log N$.

### 2.2 Alignment and Uniformity Decomposition

**Source:** *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere* (Wang & Isola, 2020) — [arXiv:2005.10242](https://arxiv.org/abs/2005.10242)

For normalized representations on the unit hypersphere $\mathbb{S}^{d-1}$ (i.e., $\|z\| = 1$), as the number of negatives $K \to \infty$, the InfoNCE loss decomposes into two terms:

$$\mathcal{L}_{\text{contrast}} \xrightarrow{K\to\infty} \underbrace{\mathbb{E}_{x,x^+}\left[\|f(x) - f(x^+)\|^2\right]}_{\text{alignment}} \;+\; \underbrace{\log \mathbb{E}_{x,y}\left[e^{-\|f(x)-f(y)\|^2 / 2\tau^2}\right]}_{\text{uniformity}}$$

**Where this comes from:** For unit vectors, $f(x)^\top f(x^+) = 1 - \frac{1}{2}\|f(x)-f(x^+)\|^2$. As $K \to \infty$, the sum over negatives converges to an expectation, and $\log(K)$ becomes a constant additive shift.

**Alignment** encourages positive pairs to have similar representations. **Uniformity** encourages the overall distribution of representations to be uniform on the hypersphere — this is the force that prevents collapse.

**Properties:**
- Alignment alone is minimized by collapse (all points → same location → alignment = 0)
- Uniformity alone is minimized by a uniform distribution on $\mathbb{S}^{d-1}$
- Together, they balance: representations of the same object should be close, but different objects should be well-separated

**Empirical finding:** Directly optimizing alignment + uniformity (without InfoNCE) achieves comparable or better downstream performance than full contrastive learning, confirming these are the essential properties.

### 2.3 The Tschannen Critique: Is InfoNCE Really Maximizing MI?

**Source:** *On Mutual Information Maximization for Representation Learning* (Tschannen et al., 2019) — [arXiv:1907.13625](https://arxiv.org/abs/1907.13625)

Three fundamental problems with the MI-maximization interpretation:

**1. Bound saturation.** For $N$ negative samples, the bound is $I(X;C) \geq \log(N) - \mathcal{L}_{\text{NCE}}$. Since $\mathcal{L}_{\text{NCE}} \geq 0$, the maximum lower bound is $\log N$. With batch size 256 (SimCLR default), $\log N \approx 5.5$ nats. The actual MI could be much higher — the bound is uninformative.

**2. The critic, not the encoder, is optimized.** The bound is actually on $I(f_\phi(X); C)$ where $f_\phi$ is the critic — not on $I(f_\theta(X); C)$ where $f_\theta$ is the encoder. The encoder representations only influence the bound through the critic. This creates a gap.

**3. Architecture-driven success.** The paper argues that contrastive learning succeeds because of:
- The **data augmentation** design (which defines the prediction task)
- The **projection head** (which peels off unwanted information)
- The **embedding geometry** (hypersphere concentration)

...not because of MI maximization. The MI bound is loose and the actual optimization dynamics are better understood through the alignment-uniformity lens.

### 2.4 Noise Contrastive Estimation (NCE) Connection

InfoNCE is closely related to Noise Contrastive Estimation. In NCE:

$$\mathcal{L}_{\text{NCE}} = -\mathbb{E}\left[\log \frac{p(x|c)}{p(x|c) + p(x)}\right]$$

InfoNCE differs by:
1. **Learned density ratio** — the critic $f(x,c)$ learns $p(x|c)/p(x)$ rather than assuming a parametric form
2. **Self-normalization** — the denominator over all $\{x_j\}$ partitions the space
3. **Multi-class formulation** — $N$-way classification rather than binary

The key distinction: NCE is designed for **density estimation** (the critic is the output), while InfoNCE is designed for **representation learning** (the critic is discarded after training, only the encoder is kept).

---

## Part 3: The Collapse Problem — Mathematical Characterization

### 3.1 Complete Collapse

**Definition:** All inputs map to the same constant vector $c \in \mathbb{R}^d$:

$$\forall x \in \mathcal{X}, \quad f_\theta(x) = c$$

**Properties:**
- Covariance matrix: $\Sigma = \mathbb{E}[(z - \bar{z})(z - \bar{z})^\top] = 0$ (zero matrix)
- All $d$ eigenvalues are zero: $\lambda_1 = \cdots = \lambda_d = 0$
- Rank: $\text{rank}(\Sigma) = 0$
- Entropy: $H(Z) = 0$
- Loss under $\mathcal{L}_{\text{align}}$: $\mathcal{L} = 0$ (global minimum)

**Why it happens:** The invariance objective $\mathcal{L}_{\text{inv}} = \mathbb{E}\|f_\theta(x) - f_\theta(x^+)\|^2$ alone has zero as its global minimum at $f_\theta(x) = c$. There is **no gradient signal** pushing representations apart.

**Linear network dynamics:** For a linear encoder $f_\theta(x) = Wx$:

$$\frac{dW}{dt} = -2\eta \, W \, \underbrace{\mathbb{E}[(x - x^+)(x - x^+)^\top]}_{S}$$

where $S$ is the positive-semidefinite difference covariance. The solution:

$$W(t) = W(0) e^{-2\eta S t}$$

Every singular value of $W$ decays exponentially to zero.

### 3.2 Dimensional Collapse

**Source:** *Understanding Dimensional Collapse in Contrastive Self-supervised Learning* (Jing et al., 2021) — [arXiv:2110.09348](https://arxiv.org/abs/2110.09348)

**Definition:** Representations span a low-dimensional subspace $r \ll d$ but are not all identical. The empirical covariance $\Sigma$ has rank $r$ where $1 \leq r < d$.

**Properties:**
- Eigenvalues: $\lambda_1 \geq \cdots \geq \lambda_r > 0$, $\lambda_{r+1} \approx \cdots \approx \lambda_d \approx 0$
- Effective rank: $\text{eff-rank}(\Sigma) = \exp\left(-\sum_{i=1}^d p_i \log p_i\right)$ where $p_i = \lambda_i / \sum_j \lambda_j$
- Under collapse: $\text{eff-rank}(\Sigma) \ll d$

**Spectral dynamics under InfoNCE:** The eigenvalue $\lambda_k$ evolves as:

$$\frac{d\lambda_k}{dt} = \lambda_k \left(\frac{1}{\tau}\frac{e^{\lambda_k/\tau}}{\sum_j e^{\lambda_j/\tau}} - \frac{N}{\tau} \mathbb{E}[\text{alignment}]\right)$$

**Key insight:** The exponential $e^{\lambda_k/\tau}$ creates a **rich-get-richer** dynamic. Larger eigenvalues get stronger positive gradients (via the softmax numerator), while the negative-pair repulsion is approximately uniform across dimensions. This amplifies dominant modes and suppresses minor ones.

**Three phases of collapse:**
1. **Alignment phase:** All eigenvalues grow as representations spread out from initialization.
2. **Spectral separation:** Top eigenvalues keep growing; smaller ones plateau or begin to shrink.
3. **Dimensional collapse:** Effective rank plummets — the model "gives up" on minor dimensions and concentrates information in a low-dimensional subspace.

**Gradient decomposition for linear encoder $W$:**

$$\nabla_W \mathcal{L} = \underbrace{\frac{1}{\tau}\mathbb{E}_{x,x^+}[(z - z^+)x^\top]}_{\text{pull positive}} - \underbrace{\frac{1}{\tau}\mathbb{E}_{x,x^-}[w_{ij}(z - z^-)x^\top]}_{\text{push negatives}}$$

where $w_{ij} = e^{z_i^\top z_j/\tau} / \sum_k e^{z_i^\top z_k/\tau}$ concentrates on the nearest negatives — which lie along the dominant eigendirections. Thus negatives push hardest along directions where representations already have high variance, further amplifying the imbalance.

### 3.3 Why Invariance Alone Collapses (BUT Non-Contrastive Methods Don't Necessarily)

**Source:** *Understanding Self-supervised Learning Dynamics without Contrastive Pairs* (Tian et al., 2021) — [arXiv:2102.06810](https://arxiv.org/abs/2102.06810)

The core result: in a simplified linear network setting, the **asymmetric architecture** (encoder + predictor + stop-gradient) of methods like BYOL and SimSiam creates dynamics where collapse is **not a fixed point**.

**Setup:** Online encoder $W$, predictor $P$, target encoder $\bar{W}$ (stop-gradient, optionally EMA).

$$\mathcal{L} = \mathbb{E}\|P W x - \bar{W} x^+\|^2$$

**Gradient flow:**

$$\frac{dW}{dt} = -2\eta (P^\top P W A - P^\top \bar{W} C^\top)$$
$$\frac{dP}{dt} = -2\eta (P W A W^\top - \bar{W} C^\top W^\top)$$

where $A = \mathbb{E}[xx^\top]$, $C = \mathbb{E}[x(x^+)^\top]$.

**Why collapse is avoided:** The predictor learns:

$$P^* = \bar{W} C^\top W^\top (W A W^\top)^{-1}$$

This creates an **asymmetry** that prevents the collapse fixed point. The stationary condition $P W A = \bar{W} C^\top$ is **not** satisfied by $W = 0$.

**Per-eigenmode result:** When the cross-correlation $C$ is full rank, the system converges to:

$$W^\top W = C^\top A^{-1} C A^{-1}$$

which is **full rank** ($r = d$). Neither complete nor dimensional collapse occurs.

**Three necessary conditions for non-collapse:**
1. **Predictor present** — without $P$, the loss becomes symmetric and collapse is a fixed point
2. **Stop-gradient applied** — without $sg(.)$, the target $\bar{W}$ chases $W$ and both collapse
3. **Cross-correlation $C$ is full rank** — if views are too similar, $C$ is low-rank and collapse occurs despite asymmetry

---

## Part 4: Mathematical Remedies for Collapse — Catalog with Exact Loss Functions

### 4.1 SimCLR — Temperature-Controlled Negative Repulsion

**Source:** [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)

**Loss (NT-Xent):**

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$

where $\text{sim}(u,v) = u^\top v / (\|u\|\|v\|)$, $z = g(f(x))$ (projection head), and $\tau > 0$ is temperature.

**Why it prevents collapse:** The denominator forces every sample to be distinguishable from every other sample. At collapse ($\text{sim}=1$ for all pairs):

$$\mathcal{L} = -\log\frac{1}{2N-1} > 0$$

This generates gradients that break collapse.

**Role of temperature $\tau$:**
- $\tau \to 0^+$: Softmax → argmax. Hardest negative dominates → strongest uniformity.
- $\tau \to \infty$: All similarities ~1 → no discrimination.
- Optimal $\tau$ (0.07–0.5 for SimCLR): balances alignment strength with uniformity.

### 4.2 VICReg — Variance + Invariance + Covariance

**Source:** [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)

**Loss:**

$$\mathcal{L} = \underbrace{\lambda \cdot \frac{1}{N}\sum_i \|z_i - z'_i\|_2^2}_{\text{invariance}} \;+\; \underbrace{\mu \cdot \frac{1}{d}\sum_j \max(0, \gamma - \sqrt{\text{Var}(z_{:,j}) + \epsilon})}_{\text{variance}} \;+\; \underbrace{\nu \cdot \frac{1}{d}\sum_{i \neq j} [C(Z)]_{ij}^2}_{\text{covariance}}$$

where $C(Z) = \frac{1}{N-1}\sum_i (z_i - \bar{z})(z_i - \bar{z})^\top$, $\gamma = 1$ (target std).

**Why it prevents collapse — three distinct forces:**
1. **Invariance** pulls positive pairs together (would alone collapse).
2. **Variance hinge** forces each feature dimension's standard deviation $\geq \gamma$. This directly prevents dimensional collapse: if one dimension collapses to a constant, that dimension's std = 0, triggering the hinge loss.
3. **Covariance** penalizes off-diagonal correlation matrix entries. This decorrelates features, preventing redundancy and ensuring each dimension captures distinct information.

**Connection to uniform hypersphere:** For uniformly distributed points on $\mathbb{S}^{d-1}$, the expected covariance is approximately $\frac{1}{d} I_d$ — diagonal with equal variance. VICReg enforces exactly this structure, albeit in $\mathbb{R}^d$ rather than on the hypersphere.

### 4.3 Barlow Twins — Cross-Correlation Minimization

**Source:** [arXiv:2103.03230](https://arxiv.org/abs/2103.03230)

**Loss:**

$$C_{ij} = \frac{\sum_b z_{b,i}^A z_{b,j}^B}{\sqrt{\sum_b (z_{b,i}^A)^2} \sqrt{\sum_b (z_{b,j}^B)^2}}$$

$$\mathcal{L} = \sum_i (1 - C_{ii})^2 + \lambda \sum_i \sum_{j \neq i} C_{ij}^2$$

**Why it prevents collapse — redundancy reduction:**
- **Diagonal ($C_{ii} \to 1$):** Each feature dimension must be perfectly correlated across the two views (invariance for that dimension).
- **Off-diagonal ($C_{ij} \to 0$, $i \neq j$):** Different dimensions must be decorrelated (no redundancy).

**Mechanism:** At collapse, all samples map to the same point. After batch normalization (which shifts and scales), each dimension is a constant. The cross-correlation between different dimensions of two constant vectors equals the product of the constants' normalized values, which is generally non-zero. This activates the off-diagonal penalty — which produces gradients that force different dimensions to carry different information. But at a single point, there's no variation to differentiate. The network must learn to spread representations to minimize the off-diagonal term.

### 4.4 BYOL — Predictor Asymmetry + EMA Target

**Source:** [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)

**Architecture:**
- Online: encoder $f_\theta$, projector $g_\theta$, **predictor** $q_\theta$
- Target: encoder $f_\xi$, projector $g_\xi$ — updated by EMA: $\xi \leftarrow \tau \xi + (1 - \tau)\theta$
- No gradients flow through the target (stop-gradient)

**Loss (symmetrized):**

$$\mathcal{L}_\theta = \underbrace{\left\|\frac{q_\theta(z_\theta)}{\|q_\theta(z_\theta)\|_2} - \frac{z'_\xi}{\|z'_\xi\|_2}\right\|_2^2}_{\text{predictor} \to \text{target projection}} \;+\; \text{symmetric term}$$

After L2 normalization, this is equivalent to:

$$\mathcal{L} = 2 - 2 \left\langle \frac{q_\theta(z_\theta)}{\|q_\theta(z_\theta)\|}, \frac{z'_\xi}{\|z'_\xi\|} \right\rangle$$

**Why it prevents collapse (the asymmetry argument):**
1. **Without a predictor:** online and target would be symmetric → MSE has trivial collapse solution.
2. **With a predictor:** the online network must learn to **predict** the target's representation. The predictor is a learnable mapping $q_\theta$ that maps the online representation to the target representation space.
3. **EMA target:** provides stable, slowly-changing regression targets. If collapse tries to happen, the target (from an older, stable encoder) still produces diverse representations, creating a persistent error signal.
4. **Formal result (Tian et al., 2021):** Without a predictor, variance goes to zero. The predictor forces information-preserving representations because it must learn a non-trivial mapping.

**The $\varepsilon$ parameter (from PMAX/JEPA context):** BYOL operates in the $\varepsilon = 1$ regime — no explicit anti-collapse regularizer, relying entirely on architectural asymmetry. This is the "hard" regime that BYOL was the first to make work at ImageNet scale.

### 4.5 SimSiam — Stop-Gradient as Implicit EM

**Source:** [arXiv:2011.10566](https://arxiv.org/abs/2011.10566)

**Loss:**

$$\mathcal{L} = \frac{1}{2} D(p_1, \text{sg}(z_2)) + \frac{1}{2} D(p_2, \text{sg}(z_1))$$

where $\text{sg}(\cdot)$ = stop-gradient, $D(p, z) = -\langle p/\|p\|, z/\|z\|\rangle$ (negative cosine similarity).

**Why it prevents collapse — implicit EM interpretation:**

Without stop-gradient, collapse is immediate — both branches predict each other's constant vector, loss = -1. With stop-gradient, SimSiam implements **alternating optimization**:

- **E-step:** Given encoder parameters $\theta$, compute targets $\eta_x = \mathbb{E}[f_\theta(x^+) | x]$ (expected representation under augmentations). Approximated by $\text{sg}(f_\theta(x_2))$.
- **M-step:** Given targets, update $\theta$ to minimize MSE: $\theta \leftarrow \arg\min_\theta \mathbb{E}\|f_\theta(x_1) - \eta_x\|^2$.

**Why collapse is unstable:** At exact collapse, loss = 0 (stationary point). However, any perturbation away from collapse produces E-step targets that differ across samples. The M-step then tries to match these different targets, pulling representations *apart*. Collapse is a stationary point but **not an attractor** — the EM dynamics naturally increase diversity.

### 4.6 MoCo — Momentum Encoder + Large Queue

**Source:** [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)

**Loss (InfoNCE with queue):**

$$\mathcal{L}_q = -\log \frac{\exp(q \cdot k_+ / \tau)}{\sum_{i=0}^K \exp(q \cdot k_i / \tau)}$$

where $q = f_q(x^{\text{query}})$, $k_+ = f_k(x^{\text{key}})$ (positive), $\{k_0, ..., k_K\}$ are $K$ negatives from the queue.

**Momentum update:** $f_k \leftarrow m f_k + (1 - m) f_q$, with $m = 0.999$.

**Why it prevents collapse:**
1. **Momentum encoder** provides near-consistent targets across training steps. If the query encoder collapses, the positive key from the stable momentum encoder still produces a diverse target — maintaining high loss.
2. **Large queue** ($K = 65536 \gg$ batch size) provides abundant negatives. At collapse, $q \cdot k_i \approx 1$ for all $i$, so $\mathcal{L} \approx \log(K+1) \approx 11.09$ — very high loss driving strong gradients.
3. **Queue decouples dictionary from batch size** — unlike SimCLR's $2N-1$ limit, MoCo provides temporal diversity from past iterations.

### 4.7 Summary: The Collapse Prevention Zoo

| Method | Loss Type | Negatives? | Collapse Prevention | Math Mechanism |
|--------|-----------|------------|---------------------|----------------|
| **SimCLR** | NT-Xent | Yes (batch) | Negative repulsion | Softmax uniformity + temperature |
| **MoCo** | InfoNCE + Queue | Yes (queue) | Large queue + momentum encoder | $K=65536$ negatives, EMA target stability |
| **VICReg** | Var+Inv+Cov | No | Variance hinge | $\max(0, \gamma - \text{std}(z_{:,j}))$ per dimension |
| **Barlow Twins** | Cross-correlation | No | Redundancy reduction | Off-diagonal $C_{ij}^2$ penalty |
| **BYOL** | MSE + Predictor + EMA | No | Architectural asymmetry | Predictor + stop-gradient breaks symmetry |
| **SimSiam** | Neg Cosine + Stop-Grad | No | Implicit EM | E-step/M-step alternation; collapse not attractor |

---

## Part 5: JEPA's Mathematical Distinctiveness

### 5.1 The JEPA Formulation

**Source:** *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (Assran et al., 2023) — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)

JEPA defines a **joint-embedding predictive** objective. Given an image $x$, sample:
- A **context block** $B_c$ (large, spatially distributed)
- **Target blocks** $\{B_t^{(k)}\}_{k=1}^K$ (smaller, semantically meaningful regions)

The encoder $f_\theta$ encodes the context: $s_c = f_\theta(x|_{B_c})$. The predictor $g_\phi$ predicts target representations from the context:

$$\mathcal{L}_{\text{JEPA}} = \sum_{k=1}^K \mathbb{E}\left[\|g_\phi^{(k)}(s_c) - \bar{s}_t^{(k)}\|^2\right]$$

where $\bar{s}_t^{(k)} = \bar{f}_\xi(x|_{B_t^{(k)}})$ is the target representation from a momentum encoder (EMA), and $g_\phi^{(k)}$ are multi-scale prediction heads.

**Key mathematical choices:**
1. **Non-generative** — predicts in latent space, not pixel space. No decoder.
2. **Multi-scale prediction** — different predictor heads operate at different block sizes, naturally creating a semantic hierarchy.
3. **EMA target** — prevents the trivial collapse by decoupling the prediction target from the online encoder.
4. **Block-wise masking** — unlike MAE's random per-patch masking, JEPA masks entire semantic blocks.

### 5.2 Why This Avoids Collapse (Theoretical)

JEPA combines **two** collapse-prevention mechanisms inherited from the BYOL/PMAX lineage:

1. **Architectural asymmetry** — predictor $g_\phi$ + EMA target $\bar{f}_\xi$ creates the same dynamics as BYOL (Section 4.4). Without this, the MSE between context and target representations would collapse.

2. **Multi-task prediction** — the multiple prediction heads $g_\phi^{(k)}$ at different scales create a **variety** of prediction tasks. This naturally constrains the representation: $s_c$ must be informative enough to predict targets at multiple scales, preventing it from discarding too much information.

3. **Spatial masking strategy** — target blocks are sampled at **large semantic scale** (not individual patches). This forces the representation to capture high-level semantics rather than local pixel statistics. The context block being **spatially distributed** ensures it contains sufficient information for non-trivial prediction.

### 5.3 Relation to Energy-Based Models

JEPA can be understood through LeCun's Energy-Based Model (EBM) framework. The energy of a configuration $(x, y)$ is:

$$E(x, y) = \|g_\phi(f_\theta(x)) - \bar{f}_\xi(y)\|^2$$

where $x$ is the context and $y$ is the target block.

The goal is to learn an energy function that assigns **low energy** to compatible (context, target) pairs and **high energy** to incompatible pairs. In JEPA, this is achieved by:
- Pulling down energy for positive pairs (ground-truth context-target combinations)
- The EMA target implicitly prevents pushing-*up* from collapsing

More broadly, JEPA is an instance of a **regularized latent-variable energy model** where the latent variable is the representation itself, and regularization comes from the architectural asymmetry.

### 5.4 Comparison: Predictive vs. Generative vs. Contrastive

| Aspect | **Generative** (MAE) | **Contrastive** (SimCLR) | **Predictive** (JEPA) |
|--------|---------------------|-------------------------|----------------------|
| Target | Raw pixels | Embedding of negative samples | Embedding of positive target |
| Loss | $\|x - \hat{x}\|^2$ | $-\log\frac{e^{\text{sim}(z,z^+)/\tau}}{\sum e^{\text{sim}(z,z^-)/\tau}}$ | $\|g(s_c) - s_t\|^2$ |
| Abstraction level | Low (pixel) | High (embedding) | High (embedding) |
| View generation | Random masking | Hand-crafted augmentations | Spatial block structure |
| Collapse risk | Low (pixel space well-constrained) | Low (negatives push apart) | Moderate (asymmetry + EMA) |

### 5.5 The Multi-Scale Prediction Hierarchy

A unique mathematical aspect of JEPA is its multi-scale prediction heads. Each predictor $g_\phi^{(k)}$ predicts the representation of a target block of a specific size. This creates an implicit curriculum:

- **Small target blocks:** predict fine-grained, local content (textures, edges)
- **Large target blocks:** predict coarse, global content (object shapes, scene layout)

Mathematically, for block sizes $\{s_1, ..., s_K\}$:

$$s_c \xrightarrow{g_\phi^{(1)}} \bar{s}_t^{(1)}(s_1)$$
$$s_c \xrightarrow{g_\phi^{(2)}} \bar{s}_t^{(2)}(s_2)$$
$$\vdots$$
$$s_c \xrightarrow{g_\phi^{(K)}} \bar{s}_t^{(K)}(s_K)$$

The context representation $s_c$ must simultaneously be predictive at all scales. This forces it to encode information that is **simultaneously local and global** — a form of multi-resolution representation learning without explicit multi-resolution architecture.

---

## Part 6: Open Theoretical Questions

### 6.1 Why Does BYOL Work Without Negatives?

This remains the most important open question in SSL theory. BYOL achieves state-of-the-art results with an apparently easy-to-collapse objective (MSE between two views). Current explanations:

- **Predictor asymmetry** (Tian et al., 2021) — the linear network analysis shows collapse is avoided, but the extension to deep nonlinear networks is heuristic.
- **Implicit regularization** — batch normalization, weight decay, and the specific initialization create implicit collapse-prevention forces that are not captured by simplified analyses.
- **Feature diversity via EMA** — the target encoder provides a consistent but drifting target; this prevents the slow collapse seen in symmetric architectures.

### 6.2 Why Does Predictor Strength Affect BYOL and SimSiam Differently?

- In BYOL: a **stronger** predictor (2-layer vs. 1-layer MLP) hurts performance.
- In SimSiam: a **stronger** predictor helps or is neutral.
- The theoretical reason is unknown. Hypothesis: in BYOL, the EMA target is already a good predictor, so an elaborate predictor overfits; in SimSiam (no EMA), the predictor must do more work.

### 6.3 The Relationship Between Batch Size and Collapse

- SimCLR: larger batches → more negatives → stronger uniformity → better representations
- BYOL: batch size has much less effect (no negatives needed)
- Barlow Twins: batch size determines the quality of the cross-correlation estimate
- Why batch size affects some methods dramatically and others barely is not fully characterized.

### 6.4 The Role of Data Augmentation

- Stronger augmentations improve contrastive and non-contrastive methods alike
- But augmentations define the *invariance* the model learns — there's no principled way to choose them
- JEPA's use of spatial structure rather than augmentations is an attempt to avoid this, but it introduces its own hyperparameters (block size, aspect ratio, spatial distribution)

### 6.5 The Spectral Gap Between SSL and Supervised Representations

SSL representations (even from strong methods) differ systematically from supervised ones:
- SSL representations have **higher effective rank** (more dimensions are used)
- SSL representations are **less specialized** — individual dimensions don't align with semantic concepts
- Supervised representations have **sharper spectral decay**
- The theoretical implications of this spectral gap for downstream task performance are not fully understood.

---

## References

### Information-Theoretic Foundations
1. van den Oord, A., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding.* — [arXiv:1807.03748](https://arxiv.org/abs/1807.03748) — InfoNCE loss derivation, MI lower bound, density ratio estimation.
2. Wang, T., & Isola, P. (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML. — [arXiv:2005.10242](https://arxiv.org/abs/2005.10242) — Alignment + uniformity decomposition of contrastive loss.
3. Tschannen, M., Djolonga, J., Rubenstein, P. K., Gelly, S., & Lucic, M. (2019). *On Mutual Information Maximization for Representation Learning.* — [arXiv:1907.13625](https://arxiv.org/abs/1907.13625) — Critique of InfoNCE as MI maximization.

### Collapse Analysis
4. Jing, L., et al. (2021). *Understanding Dimensional Collapse in Contrastive Self-supervised Learning.* — [arXiv:2110.09348](https://arxiv.org/abs/2110.09348) — Mathematical characterization of dimensional collapse, spectral dynamics.
5. Tian, Y., et al. (2021). *Understanding Self-supervised Learning Dynamics without Contrastive Pairs.* — [arXiv:2102.06810](https://arxiv.org/abs/2102.06810) — Linear network analysis of BYOL/SimSiam non-collapse dynamics.

### Mathematical Remedies
6. Chen, T., et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* ICML. — [arXiv:2002.05709](https://arxiv.org/abs/2002.05709) — NT-Xent loss.
7. He, K., et al. (2020). *Momentum Contrast for Unsupervised Visual Representation Learning.* CVPR. — [arXiv:1911.05722](https://arxiv.org/abs/1911.05722) — MoCo momentum encoder + queue.
8. Grill, J.-B., et al. (2020). *Bootstrap Your Own Latent.* NeurIPS. — [arXiv:2006.07733](https://arxiv.org/abs/2006.07733) — BYOL: predictor + EMA.
9. Chen, X., & He, K. (2021). *Exploring Simple Siamese Representation Learning.* CVPR. — [arXiv:2011.10566](https://arxiv.org/abs/2011.10566) — SimSiam: stop-gradient as implicit EM.
10. Zbontar, J., et al. (2021). *Barlow Twins: Self-Supervised Learning via Redundancy Reduction.* NeurIPS. — [arXiv:2103.03230](https://arxiv.org/abs/2103.03230) — Cross-correlation minimization.
11. Bardes, A., et al. (2022). *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR. — [arXiv:2105.04906](https://arxiv.org/abs/2105.04906) — Explicit variance regularization.

### JEPA & Related
12. Assran, M., et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* ICCV. — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) — I-JEPA: multi-scale latent prediction.
13. Bardes, A., et al. (2024). *Revisiting Feature Prediction for Learning Visual Representations from Video.* — [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) — V-JEPA: temporal prediction.
14. LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence.* OpenReview. — [https://openreview.net/forum?id=BZ5a1r-kVsf](https://openreview.net/forum?id=BZ5a1r-kVsf) — EBM and world model vision for JEPA.

### Ancillary
15. Matrix Information Theory for Self-Supervised Learning. — [arXiv:2305.17326](https://arxiv.org/abs/2305.17326) — Unifies SimSiam, Barlow Twins, MEC under maximum entropy encoding.
