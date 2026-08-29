---
title: "Neural Fourier Spectral Fields: Shapes as Learned Spectra Instead of SDFs"
tags:
  - idea
  - neural-fields
  - fourier
  - spectral
  - sdf
  - implicit-representations
  - collision-detection
  - nerf
date: 2026-08-29
lastmod: 2026-08-29
draft: false
summary: "Idea note — represent neural shapes (SDF/NeRF) directly as 3D Fourier spectra and exploit analytic frequency-domain structure for collision detection. Explores prior art, where the 'analytic collision' claim holds, and concrete research directions."
---

# Neural Fourier Spectral Fields: Shapes as Learned Spectra Instead of SDFs

*Draft idea note — expanded with research. Status: unvalidated, thinking out loud.*

## The core idea

Current implicit neural geometry works pointwise: a coordinate MLP maps $x \in \mathbb{R}^3 \to$ a value — signed distance (DeepSDF, SIREN), occupancy (Occupancy Networks), or radiance + density (NeRF). The network *is* the field; queries cost one forward pass per point.

But any 3D shape is just a signal, and any signal is a superposition of Fourier components. A shape's indicator function (or its SDF) lives in a frequency basis — the "really high component Fourier" of sharp edges included. So the pitch:

> **Instead of learning the SDF value at every point, learn the 3D Fourier spectral representation of the shape: a finite set of (frequency, amplitude, phase) coefficients. The object *is* its spectrum.**

Two consequences that make this attractive:

1. **Unified representation.** Light and geometry in one language: NeRF's radiance field and the SDF are both spectral fields. Rendering becomes evaluating a Fourier series; lighting becomes frequency-domain transport.
2. **Analytic-ish collision detection.** Two spectral objects intersect where their level sets coincide. The sum of two Fourier series is a Fourier series — so the intersection query is a problem about trigonometric polynomials, which have strong analytic structure. The overlap volume of two shapes as a function of relative translation is exactly a cross-correlation of their characteristic functions — and by the convolution theorem that's a pointwise product in the Fourier domain, computable with FFTs.

## Prior art check (this is not a new idea in pieces — the novelty is in the combination)

The "nothing exists" test failed — the components are all real, published work:

| Piece | Reference | What it does |
|---|---|---|
| **Spectral bias** | Rahaman et al., *On the Spectral Bias of Neural Networks* (ICML 2019) | MLPs learn low frequencies first — the theoretical reason Fourier features exist |
| **Fourier features / positional encoding** | Mildenhall et al. (NeRF, 2020); Tancik et al. *Fourier Features Let Networks Learn High Frequency Functions* (NeurIPS 2020) | Injecting a frequency basis into the input fixes the low-frequency bias — the NN learns coefficients over a fixed spectral grid, not the signal itself |
| **Sinusoidal activations** | Sitzmann et al., SIREN (2020) | Sine-activation MLPs that represent signals as compositions of sinusoids |
| **Band-limited coordinate networks** | Lindell et al., *BACON: Band-limited Coordinate Networks for Multiscale Scene Representation* (ICLR 2022) | The closest existing thing: coordinate networks whose Fourier spectrum is *analytically characterized* and controllable per layer/band; demonstrated on images, NeRF, and SDFs. Networks whose behavior is entirely described by their spectrum |
| **Band-limited fields for LOD** | Shabanov et al., BANF (CVPR 2024) | Neural fields that are provably low-pass filtered; frequency decomposition of the signal by construction |
| **Fourier bases for shape INR** | *Learning Spatially Collaged Fourier Bases for INR* (SCONE, 2023) | SOTA 3D shape fitting (98.81% IoU) with learned Fourier bases |
| **Fourier collision detection** | Lysenko, *Fourier Collision Detection*, IJRR 32(4), 2013 | **Direct prior art for the collision claim:** narrowphase collision detection for rigid bodies via Fourier series expansion, grid-free (shapes as countable unions of balls + convolution formulation), accuracy-scalable in the Hausdorff sense for translations |
| **FFT shape correlation / docking** | Katchalski-Katzir et al., *Molecular Surface Recognition by Computer-Vision Techniques* (PNAS, 1992) | The classic: maximize molecular overlap by FFT correlation of shape indicator functions — collision-as-overlap in the frequency domain, proven at industrial scale |
| **2D Fourier descriptors** | Zahn & Roskies (1972) | The ancient ancestor: 2D contours as Fourier coefficient vectors |

So: *learning* spectra (BACON/BANF/SCONE) and *colliding* spectra (Lysenko, docking) each exist. Nobody seems to have fused them — learned coefficient-output networks whose geometry queries are frequency-domain operations. That fusion is the actual idea.

## Where the "analytic collision" claim is true — and where it gets subtle

This is the part I had to think hardest about. "Fourier has an analytical form" is true but the *zero-finding* is not closed-form in general:

- **1D (exact).** The zeros of a band-limited Fourier series $f(x) = \sum c_k e^{ikx}$ can be found analytically-ish: substitute $z = e^{ix}$ and the trigonometric polynomial becomes a Laurent polynomial in $z$ — i.e. an ordinary polynomial. Roots come from a companion matrix. Genuinely algebraic, robust, done.
- **3D (the honest version).** Two band-limited fields $f(x) = 0$, $g(x) = 0$ are algebraic surfaces (3-variable Laurent polynomials). Their intersection is an algebraic space curve — computable in principle via resultants/Gröbner bases, but that's not a real-time algorithm. "Analytical form" ≠ closed-form intersection.

What *is* real and strong in 3D:

1. **Overlap via convolution theorem (exact, fast).** The overlap volume of two shapes as a function of relative translation is the cross-correlation of their indicator functions: $V(t) = \int \mathbb{1}_A(x)\,\mathbb{1}_B(x-t)\,dx = \mathcal{F}^{-1}[\hat{\mathbb{1}}_A \cdot \overline{\hat{\mathbb{1}}_B}](t)$. One FFT multiply per pair, $O(N^3 \log N)$. This is exactly how protein docking works, and it gives you *all* translations at once (penetration depth = gradient of $V$ w.r.t. $t$).
2. **Conservative bounds from band-limitation (no false negatives).** A band-limited series has a Lipschitz bound computable from its coefficient decay ($\sup |\nabla f| \le \sum |k||c_k|$). That turns collision into interval arithmetic / branch-and-bound on $|f - g|$ — rigorous, analytic, and cheap. This is the genuinely under-used trick.
3. **Rotation is exact in $k$-space.** The Fourier transform of a rotated function is the rotated Fourier transform; band-limit is preserved under rotation. Rigid transforms act on spectral shells, so a rotated object stays exactly representable — no resampling, no re-fitting. (This is why SH/rotation is so popular in precomputed rendering.)
4. **No per-query network evaluation.** Once coefficients are learned, collision/rendering queries are arithmetic on a fixed coefficient tensor — no forward passes, no marching cubes, no triangle soup. Fits GPU massively in parallel.

**Pitfalls that will bite:**

- **Gibbs phenomenon.** Indicator/occupancy functions have step edges → algebraic coefficient decay → ringing at truncation. A truncated occupancy spectrum is not watertight. This is *why* SDFs exist: the signed distance is Lipschitz-1 and much smoother than the indicator (only $C^0$ at the medial axis), so its spectrum decays faster. **Learn the SDF's spectrum, not the occupancy spectrum.** Sharp corners still decay slowly (algebraic, not exponential) — sharp features are the enemy of low-rank spectra.
- **Storage.** A full $N \times N \times N$ coefficient tensor is $O(N^3)$ — the same scaling as a dense grid, and worse than hash grids (Instant-NGP). Needs compression: low-rank tensor decompositions, multiscale coefficient banks (BACON-style bands), or sparsity (most shapes need few *dominant* coefficients).
- **"Analytic" ≠ "free".** Root-finding and interval bounds are still numerical work; the win is that they're *provably bounded, parallel, and coefficient-only* — not that they're closed form.

## Concrete research directions

1. **Coefficient-predicting networks.** Encoder (point cloud / voxels / multi-view images) → MLP head → Fourier coefficients (per-axis tensor, or per spherical shell for rotation-equivariance). Train with spectral L2 loss — by Parseval this equals spatial L2, so supervision can come from any existing SDF/occupancy dataset.
2. **Frequency budget as a dial.** Adaptive truncation with error guarantees from tail bounds; progressive LOD by adding coefficient bands (natural multiscale, BACON-style); "buy coefficients where the shape has features" via saliency.
3. **Collision stack on learned spectra.** Broad phase: conservative interval bounds from Lipschitz constants. Narrow phase: convolution-theorem overlap for translations, $k$-space rotation for orientations, penetration depth from $\nabla V$. All of it coefficient-only.
4. **Light + geometry in one field.** Radiance spectrum + SDF spectrum in a single learned tensor; frequency-domain light transport (light × BRDF convolution in Fourier domain) as the renderer.
5. **Benchmark against the obvious baselines.** Instant-NGP hash grids, SIREN, DeepSDF on: fit quality vs parameter count, collision query speed + robustness (false-negative rate), LOD behavior. Lysenko's paper already gives the accuracy-scaling analysis for Fourier collision — reuse his framework for the learned version.

## Open questions

- Does a network that outputs coefficients directly beat a network that implicitly learns a Fourier-feature SDF? The positional-encoding view suggests the MLP is *already* learning coefficients over a fixed frequency grid — is the explicit version just removing the MLP?
- How many coefficients do real shapes need for 1mm-accurate collision? (Theoreticians: relate to surface area × curvature spectrum of the shape.)
- Are there closed-form narrowphase queries for *special* coefficient structures (low-rank, separable per-axis, spherical-shell) even if the general case is algebraic?

## Sources

- Lysenko, *Fourier Collision Detection*, IJRR 2013 — https://journals.sagepub.com/doi/10.1177/0278364913477165 (author's blog: https://0fps.net/category/programming/collision-detection/)
- Lindell et al., *BACON: Band-limited Coordinate Networks* — https://arxiv.org/abs/2112.04645
- Shabanov et al., *BANF: Band-limited Neural Fields for Levels of Detail* (CVPR 2024) — https://theialab.github.io/banf/
- *SCONE: Learning Spatially Collaged Fourier Bases for INR* — https://arxiv.org/abs/2312.17018
- Rahaman et al., *On the Spectral Bias of Neural Networks* — https://arxiv.org/abs/1806.08734
- Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions* — https://arxiv.org/abs/2006.10739
- Sitzmann et al., *SIREN* — https://arxiv.org/abs/2006.09661
- Katchalski-Katzir et al., *Molecular Surface Recognition by Computer-Vision Techniques*, PNAS 1992 — https://www.pnas.org/doi/10.1073/pnas.89.6.2195
- Zahn & Roskies, *Fourier Descriptors for Plane Closed Curves* — https://ieeexplore.ieee.org/document/1671525
