---
title: "Transformers & Attention (Serret 2026) — Paper Study Notes"
tags:
  - paper-study
  - transformers
  - attention
  - llm-architecture
date: 2026-08-04
lastmod: 2026-08-04
summary: "Study notes on Serret's intro to Transformers/attention for applied mathematicians — tokenization, kernel view of attention, MHA, encoder/decoder, KV caching, GQA, MLA."
---

# Understanding Transformers and Attention — Paper Study Notes

**Paper:** [Understanding Transformers and Attention Mechanisms: An Introduction for Applied Mathematicians](https://arxiv.org/abs/2604.00965) — Michel Fabrice Serret (Paul Scherrer Institute). arXiv:2604.00965v1 [math.NA], Apr 2026. 13pp, 9 figures. Written for the **IPAM RNLA workshop** (project: "Randomization in Transformer models").

> **The whole paper in one line:** attention = database lookup with a similarity kernel + row normalization. Everything else is engineering around that core.

## Why this paper
- **No new results** — value is a clean, self-consistent **formalism** for attention, aimed at applied mathematicians.
- Walks the full modern stack: tokenization → embeddings → attention → MHA → encoder/decoder → **KV cache → GQA → MLA**.
- Ends on the **memory/compute trade-offs** that drive modern attention compression — the part that matters for LLM inference.

---

## 1. Tokens → Embeddings

### Tokenization
- Text → sequence of substrings (**tokens**); vocabulary $\mathcal{T} = (T_i)_{1\le i\le N_T}$ = the $N_T$ distinct tokens.
- A text becomes a **sequence of token indices**.
- Tokenizer tension: fine enough to capture semantics, small enough to not dilute info or blow up memory.
- In practice: tokens ≈ words (paper takes this WLOG).

### Embeddings (vectorization)
- One trained linear map: matrix $E \in \mathbb{R}^{N_T \times d}$ — token $i$ → row vector $E_i$ (dim $d$).
- In LLMs: $E$ is **trained from scratch as model weights**.
- Optional **feature embeddings** (positional, sentence-id) on top; DeBERTa & DeepSeek V2 keep them **separate by design**.

**Scale check** (Table 3 / §1):

| Model | vocab | embedding dim $d$ |
|---|---|---|
| Llama 3 70B | ~128k | 8,192 |
| Gemma 3 27B | ~262k | 5,376 |

---

## 2. Attention — the core formalism

### The database analogy
A database = set of **(key, value)** tuples; query $q$ returns the value of the matching key. Attention **relaxes "exact match" to "similarity-weighted average"**.

### The math (general kernel view)

**Setup:** queries $X_Q \in \mathbb{R}^{N_Q \times d_{in}}$, keys $X_K \in \mathbb{R}^{N_{KV} \times d_{in}}$, values $X_V \in \mathbb{R}^{N_{KV} \times d_{in}}$ (same $d_{in}$ — all come from the same embeddings). Chat example: $N_Q$ = tokens of the latest question, $N_{KV}$ = the whole conversation.

**Math (general attention):**

$$Q = X_Q W_Q, \quad K = X_K W_K, \quad V = X_V W_V, \qquad W_Q, W_K \in \mathbb{R}^{d_{in}\times d_{QK}}, \ W_V \in \mathbb{R}^{d_{in}\times d_{out}}$$

$$A = \big(\kappa(Q[i,:],\, K[j,:])\big)_{\substack{1\le i\le N_Q \\ 1\le j\le N_{KV}}} \in \mathbb{R}^{N_Q\times N_{KV}}, \qquad Z[i,i] = \sum_{1\le j\le N_{KV}} \kappa(Q[i,:], K[j,:])$$

$$\boxed{Y = Z^{-1} A V \in \mathbb{R}^{N_Q \times d_{out}}}, \qquad d_{out} = d_V$$

*Takeaway:* $Y$ = rows of $V$ **weighted by normalized query–key similarity**. The kernel is almost always $\kappa(v,w) = f_\kappa(\langle v,w\rangle)$ — dot product through a scalar function.

**Cost:** $O(N_Q N_{KV} (d_{QK} + d_V))$ — quadratic in token count; $d_{QK}, d_V$ are small by comparison.

### Softmax = exp kernel + row normalization

**Math (scaled exponential kernel)** [Vas+17]:

$$\kappa(v,w) = \exp\!\left(\frac{\langle v,w\rangle}{\sqrt{d_{QK}}}\right), \qquad A = \exp\!\left(\frac{QK^T}{\sqrt{d_{QK}}}\right) \ \text{(element-wise)}, \qquad (Z^{-1}A)[i,:] = \sigma\!\left(\frac{Q[i,:]K^T}{\sqrt{d_{QK}}}\right)$$

*Why $\sqrt{d_{QK}}$:* entries of $v, w$ i.i.d. mean-0 var-1 ⇒ $\langle v, w\rangle$ has **variance $d_{QK}$** — dividing rescales the dot product to unit variance.

⚠ **Softmax attention is a special case of the kernel view** (exp kernel + row-normalize). The kernel framing is the general object.

### Multi-headed attention (MHA)

**Math (per head $h$):** $Q_h = X_Q W_Q^h$, $K_h = X_K W_K^h$, $V_h = X_V W_V^h$, $A_h = f_\kappa(Q_h K_h^T)$,

$$Y_h = Z_h^{-1} A_h V_h \in \mathbb{R}^{N_Q\times d_{head}}, \qquad Y = \operatorname{concat}_{col}(Y_h)_{1\le h\le N_{heads}} W_O, \quad W_O \in \mathbb{R}^{N_{heads}d_{head}\times d_{out}}$$

*Why:* parallel attention over **different projected subspaces** → attends to several types of semantic info simultaneously.

⚠ **Heads do NOT split the input.** Every head consumes the *same full* $X$; heads differ **only by their weight matrices**:
> "linearly project the queries, keys and values $h$ times with different, learned linear projections" — Vaswani et al. 2017, §3.2.2

The "split into heads" in diagrams/code = **reshape of the output columns** of one big $X W_Q$ matmul — never a partition of $X$.

### Projection sizes (the free-parameter question)
- $d_{QK}$ is a **free parameter** — no constraint vs $d_{in}$. Paper: $d_{QK}$ and $d_V$ "can be different in theory, in practice they are often equal to one another" (§2.1); in MHA, $d_{QK} = d_{head}$ (§2.2).
- Total Q/K width = $N_{heads} \cdot d_{head}$. Table 3: Llama 3 70B → 64×128 = 8,192 = $d_{in}$; Gemma 3 27B → 32×128 = 4,096 < 5,376. **Equal or smaller in practice; never larger.**
- $d_{QK}$ enters the math in exactly one place: the $\sqrt{d_{QK}}$ kernel scaling.

---

## 3. Architecture

### Encoder layer
(1) multi-head **self-attention** ($X_Q = X_K = X_V$, all arrows from the previous layer) → (2) LayerNorm → (3) FFN → (4) **skip connections** around each sublayer (preserve gradient flow, fight vanishing gradients).

- **FFN:** $y = f(Wx + b)$; ReLU family; **GLU variants** [Sha20] dominate recent SOTA (Gemma, Llama 3, Qwen3).

### Normalization — 4 variants

**Math (LayerNorm)** [BKH16]: $\tilde{x} = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}$, then $y = \gamma \odot \tilde{x} + \beta$ — recenter + rescale + **learned gain/shift**.

**Math (RMSNorm)** [ZS19]: $\tilde{x} = x / \mathrm{RMS}(x)$ with $\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2}$, then a learned gain — **no recentering, no bias** ⇒ cheaper. Standard in modern LLMs (Llama, Gemma, DeepSeek).

**Pre-LN vs Post-LN** [Xio+20]: Post-LN = original Transformer (norm *after* each sublayer). Pre-LN = norm *before* → skip connections stay **unnormalized**, gradients skip through unscathed ⇒ easier training, slight quality cost.

**Peri-LN** [Kim+25]: norm **around** each sublayer (before AND after the module) + input/output embedding norms.
- Paper's remark: "state-of-the-art large language models such as **Gemma 3** have made use of both Pre-Layer and Post-Layer Normalization" [Kim+25].
- Peri-LN paper (arXiv 2502.02732, ICML 2025): more **balanced activation-variance growth**, steadier gradient flow (tested ≤3.2B params). Adopters named: **Gemma 2/3 and OLMo 2**.
- ⚠ **Not Gemini, not Kimi.** "[Kim+25]" = Jeonghoon Kim et al. — reads like "Kimi", but neither paper mentions Kimi.

### Decoder layer
Masked causal self-attention → **cross-attention** → FFN.

**Math (causal mask):** query $i$ sees only $j \le i$: $S_i = \{j : j \le i\}$,

$$A = \big(\mathbb{1}_{j\in S_i}\, \kappa(Q[i,:], K[j,:])\big)_{i,j} = \kappa(QK^T + M), \qquad M_{ij} = \begin{cases}0 & j \in S_i \\ -\infty & \text{else}\end{cases}$$

(equivalently $A = \kappa(QK^T) \odot \mathbb{1}_{S_i}(j)$ — mask after the kernel; $-\infty$ → zero probability after softmax.)

**Cross-attention:** keys/values = **encoder's final output** $E_{N_L}$ (source sentence), queries = decoder — how the decoder relates its output to the input. ⚠ Cross-attention / shared embedding spaces = root of **multimodal LLMs** (Flamingo [Ala+22], Latent Diffusion [Rom+22]).

### Encoder-only vs Decoder-only

| | Encoder-only | Decoder-only |
|---|---|---|
| Archetype | BERT [Dev+19] | GPT [Rad+18] |
| Task | extraction / classification | generation (next-token) |
| Pretrain objective | masked completion | next-token prediction |

Paradigm: **pretrain on huge data → fine-tune on small data** [Rad+18].

---

## 4. Efficiency: KV cache → GQA → MLA

### Why KV caching
The last token's output needs **all** K/V:

**Math (last-token output):**

$$Y[N_Q, j] = \sum_{i=1}^{N_{KV}} \frac{\kappa(Q[N_Q,:], K[i,:])}{\sum_{\ell=1}^{N_{KV}} \kappa(Q[N_Q,:], K[\ell,:])} \, V[i,j]$$

- Naive: recompute all K/V per new token → $O(N_{tokens}^3 d)$ total.
- **KV cache:** store K/V; per new token only compute new K/V + the new query row → $O(N_Q N_{tokens} d)$ per step, $O(N_{tokens}^2 d)$ overall.
- Cache memory: $2 \cdot N_L \cdot N_{heads} \cdot N_{KV} \cdot d$ floats → **prohibitive in long context**.
- **Streaming attention** [Han+25]: queries ⊂ KV tokens (chat) → only new queries computed, *if* K/V is cached.

### GQA — Grouped Query Attention

⚠ **It's the KEY/VALUE heads that get shared — queries are NEVER shared.** Q is computed fresh each step and discarded; only K/V get cached — so sharing K/V is what saves memory.

**Math:** $G \le N_{heads}$ KV groups, group size $s = N_{heads}/G$; query head $h$ uses KV head $g(h) = \lfloor (h-1)/s \rfloor + 1$:

$$Q_h = X_Q W_Q^h \ \ \forall h, \qquad K_g = X_K W_K^g, \quad V_g = X_V W_V^g \ \ (g = 1 \dots G), \qquad Y_h = Z_h^{-1} A_h V_{g(h)}, \ \ A_h = f_\kappa(Q_h K_{g(h)}^T)$$

**What it buys:**
- Weights: $N_{heads}$ copies of $W_Q$, but only **$G$ copies** of $W_K, W_V$.
- Cache: $2 G d_{head}$ per token/layer vs $2 N_{heads} d_{head}$ (Table 1: replace $N_{heads}$ with #KV heads in the $W_K$, $W_V$ and cache terms).
- Limits: $G = N_{heads}$ → MHA; $G = 1$ → **MQA** (single shared KV head).

**Table 3 examples:** Llama 3 70B → 64 query heads / 8 KV heads (**8× smaller cache**); Gemma 3 27B → 32 / 16 (**2×**).

**⚠ Never-forget version (why shared K/V ⇒ identical outputs):**
> All heads see the SAME input vector. MHA = same input × different weight matrices. GQA = same input × the SAME weight matrix (within a group) ⇒ the multiplication is literally identical ⇒ identical K/V outputs ⇒ store ONE copy per group. The vectors aren't "similar" — they're the same number, computed twice, stored once.

**Numeric check** ($d_{in}=3$, $d_{head}=2$, $x=(1,2,3)$): $W_K^1 = [[1,0],[0,1],[1,1]]$ → $K_1 = (4,5)$; $W_K^2 = [[0,1],[1,0],[1,0]]$ → $K_2 = (5,1)$ — same $x$, different weights. Share one $W_K^a = W_K^1$ between both heads → $K_1 = K_2 = (4,5)$ exactly. The "head" = a block of **output columns** in $W_K = (W_K^1 \cdots W_K^{N_{heads}})$ — the input is never partitioned.

**Why it works:** queries encode *what to look for* (must stay diverse per head); K/V encode *content of the context* (shareable memory). GQA = quality/cache dial. Origin papers (not in Serret's refs): MQA [Sha19, arXiv 1911.02150], GQA [Ain+23, arXiv 2305.13245 — ~8 groups ≈ MHA quality].

### MLA — Multi-head Latent Attention (DeepSeek-V2) [Dee+24]

**Main idea:** cache **one low-rank latent vector per token, shared across ALL heads** — instead of per-head K/V.

**Column-concat view first:** $Q = (Q_1 \cdots Q_{N_{heads}}) = X_Q W_Q$ with $W_Q = (W_Q^1 \cdots W_Q^{N_{heads}})$, likewise $K, V$. Then low-rank factorize (latent dims $d_{LQ}$ for queries, $d_L$ shared for K/V):

$$W_Q = W_{LQ} W_{LQQ}, \qquad W_K = W_L W_{LK}, \qquad W_V = W_L W_{LV}, \qquad W_L \in \mathbb{R}^{d_{in}\times d_L}$$

**Cached object:** the shared latent $L = X W_L \in \mathbb{R}^{N_{KV}\times d_L}$ — NOT the K/Vs. (DeepSeek-V2 trains this directly; it is not factored out of an existing MHA.)

**Weight merging (the actual trick):**
1. **QK merge** per head: $W_{LQK}^h = W_{LQQ}^h (W_{LK}^h)^T \in \mathbb{R}^{d_{LQ}\times d_L}$ since $Q_h K_h^T = L_Q W_{LQQ}^h (W_{LK}^h)^T L^T$ — attention logits computed **straight on the latent**.
2. **OV merge:** $W_{LO} = \operatorname{blockdiag}(W_{LV}^1, \dots, W_{LV}^{N_{heads}})\, W_O \in \mathbb{R}^{N_{heads} d_L \times d_{out}}$ since $V_h = L (W_{LV}^h)^T$:

$$Y = \operatorname{concat}_{col}\left(Z_h^{-1} A_h L\right)_{1\le h\le N_{heads}} W_{LO}, \qquad A_h = \exp\!\left(L_Q W_{LQK}^h L^T\right)$$

Only weights kept: $W_{LO}, W_L, W_{LQ}, W_{LQK}$ + latent vectors. **$K_h$ / $V_h$ are never materialized.**

**Intuition (why the merging is valid):**
- **Rank constraint:** $W_K = W_L W_{LK} \Rightarrow \operatorname{rank}(W_K) \le d_L$ — all heads' K/V subspaces live in ONE common $d_L$-dim subspace of $\mathbb{R}^{d_{in}}$. That shared-subspace restriction is the *only* approximation; everything else is exact algebra.
- **Queries are never cached:** $L_Q = X_Q W_{LQ}$ has only $N_Q$ rows (current step); only $L$ (the $N_{KV}$ rows) persists across steps.
- **Analogy:** GQA shares one dictionary across heads; MLA compresses the whole dictionary into a bottleneck that every head reads a linear view of.

**Memory comparison** (Tables 1–2, floats):

| | per-token cache per layer |
|---|---|
| MHA | $2 N_{heads} d_{head}$ |
| GQA | $2 G d_{head}$ |
| MLA | $d_L$ |

DeepSeek-V2: 128 heads × 128 dims = **32,768** floats/token/layer → $d_L = 512$ → **64× smaller cache** (60 layers, $d_{in} = 5120$, MoE).

⚠ **RoPE caveat** [Su+23]: without positional embeddings, MLA ≡ low-rank MHA **exactly**. With RoPE, a rotation $R_m \in \mathbb{R}^{d_{head}\times d_{head}}$ is applied to K *after* it's built:

$$Q_h[i,:] R_i R_j^T K_h[j,:]^T = (L_Q[i,:]) W_{LQQ}^h R_i R_j^T (W_{LK}^h)^T (L[j,:])^T$$

$R$ sits *between* the two factors ⇒ $W_{LQQ}^h$ and $W_{LK}^h$ can't be merged (RoPE would need recomputation per evaluation). Fix: keep the latent **rope-free** and append a small **non-latent** K/Q part that carries the position — breaks exactness, keeps the speedup. [MYZ25] (TransMLA) converts pretrained GQA/MHA → MLA, porting DeepSeek's optimization to other models.

---

## Key references (what to look up when a line feels unsupported)

| Ref | Why it matters here |
|---|---|
| [Vas+17] Attention is All You Need | Original Transformer + softmax attention (§2) |
| [BKH16] Layer Normalization | LN definition used in every layer |
| [ZS19] RMS Layer Normalization | Cheaper norm (no recentering/bias) in modern LLMs |
| [Xio+20] On Layer Normalization | Pre-LN variant; unnormalized skip connections |
| [Kim+25] Peri-LN (arXiv 2502.02732, ICML'25) | LN around sublayers (pre + post); Gemma 2/3 + OLMo 2 adopt it |
| [Sha20] GLU Variants | FFN nonlinearity now standard in SOTA |
| [Dee+24] DeepSeek-V2 | Source of MLA; 60-layer MoE, $d_L = 512$ |
| [Su+23] RoFormer / RoPE | Rotary positional embedding; why MLA must break exactness |
| [MYZ25] TransMLA | GQA/MHA → MLA conversion for other models |
| [Han+25] Streaming Attention | Query ⊂ KV; motivates KV caching |
| [Zha+24] Dive into Deep Learning | Source of the database analogy |
| [Dev+19] BERT / [Rad+18] GPT | Encoder-only vs decoder-only + pretrain/finetune paradigm |

## Connections & open threads

- **RNLA angle (why this paper exists):** the kernel + row-normalization framing is exactly where **randomized NLA** enters attention — sketch / row-sample the $QK^T$ similarity or $V$.
- **Cross-attention ↔ latent diffusion** [Rom+22]: same mechanism the video_diffusion notes build on.
- **MLA weight-merge algebra** = a clean worked example of low-rank weights (cf. sparsity_tricks).
- **Self-consistency checks:** softmax = kernel special case; GQA = MHA with tied KV heads; MLA without RoPE = low-rank MHA. All three read off the same formalism.
