# Information-Theoretic Foundation of Self-Supervised Learning

## 1. InfoNCE Loss — CPC Paper (Oord et al., 1807.03748)

### Setup
Sequence x_1..x_T; context c_t = g_enc(x_<=t). Goal: predict future x_{t+k}.

### Core Idea: Density Ratio Estimation
Model the density ratio instead of p(x|c) directly:

    f_k(x_{t+k}, c_t) = p(x_{t+k} | c_t) / p(x_{t+k})

### The InfoNCE Loss
Given 1 positive + K negatives:

    L_NCE = -E[ log( f(x_pos, c) / (f(x_pos,c) + sum_{j=1}^{K} f(x_neg_j,c)) ) ]

### Proof: Lower Bound on MI
Optimal critic: f_k(x,c) = p(x|c)/p(x)
Then: L_NCE >= log(1 + (N-1)/E[r]) where r = p(x_pos|c)/p(x_pos)
Since E[r] = exp(I(x;c)):

    L_NCE >= log(N) - I(x;c)   =>   I(x;c) >= log(N) - L_NCE

### Key Insight
Minimizing InfoNCE maximizes a lower bound on MI. Bound tightens as N -> inf.

---

## 2. Alignment + Uniformity (Wang and Isola, 2005.10242)

### Claim
For normalized embeddings (||z||=1), InfoNCE as K->inf decomposes:

    L -> Alignment + Uniformity

### Alignment Term
    L_align = -E_{(x,x+)}[ ||f(x) - f(x+)||^2 ]

Pulls positive pairs together. Forces invariance to augmentations.

### Uniformity Term
    L_uniform = log( E_{x,y}[ exp(-||f(x)-f(y)||^2 / 2tau^2) ] )

Maximizes entropy on hypersphere. Prevents dimensional collapse.

### Proof Sketch
For unit vectors: f^T f+ = 1 - 0.5||f-f+||^2.
As K->inf, sum over negatives -> expectation.
log(K) is constant w.r.t. f, so optimization decouples.

---

## 3. Tschannen et al. Critique (1907.13625)

### Why MI Bound is Loose
1. Saturation: Bound capped at ~log(N) (~11 nats), true MI is enormous
2. Critic != Encoder: Bound is on I(critic(x);c), not I(encoder(x);c)
3. Critic architecture: Bilinear critics restrict what MI can be captured

### What Actually Drives Performance
- Data augmentation design (dominant factor)
- Architecture (projection heads, encoder design)
- Embedding geometry (alignment + uniformity)
- Performance does NOT correlate with MI estimate

---

## 4. NCE Connection

### Original NCE (Gutmann and Hyvarinen, 2010)
Binary classification: data vs noise. Estimates unnormalized densities.

### InfoNCE = Multi-class NCE with:
1. Learned density ratio p(x|c)/p(x)
2. Self-normalized noise (marginal p(x))
3. (N-1)-way classification

### Key Difference
- NCE: density estimation (learn p(x))
- InfoNCE: representation learning (learn features; discard critic)

---

### References
1. Oord et al. (2018). Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748.
2. Wang and Isola (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. arXiv:2005.10242.
3. Tschannen et al. (2020). On Mutual Information Maximization for Representation Learning. arXiv:1907.13625.
4. Gutmann and Hyvarinen (2010). Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. AISTATS.
