---
title: Alignment Detection for Video Diffusion Models — Literature Landscape
tags:
  - video-diffusion
  - alignment
  - detection
  - research
date: 2026-06-10
---

# Alignment Detection for Video Diffusion Models — What Exists

**Core question:** Given a generated video, can we detect whether it aligns with the user's intended prompt (or broader intent)? This is distinct from *alignment training* (DPO, RLHF) — it's about the *detection/classification* side of alignment, though the two are deeply connected via reward models.

---

## 1. Explicit Misalignment Detection Frameworks

The most directly relevant work: systems that take a generated video + prompt and output *where* and *how* they misalign.

### VideoRepair (Nov 2024)
- **Paper:** *Self-Correcting Text-to-Video Generation with Misalignment Detection and Localized Refinement*
- **Authors:** Daeun Lee, Jaehong Yoon, Jaemin Cho, Mohit Bansal (UNC Chapel Hill)
- **arXiv:** 2411.15115
- **Key idea:** The first training-free, model-agnostic framework that explicitly detects fine-grained text-video misalignments, then performs targeted localized corrections.
- **Misalignment detection mechanism:**
  1. Automatically generates evaluation questions from the prompt using an MLLM (LLaVA-Video)
  2. Queries the MLLM on the rendered video frames to detect which entities/attributes/relations are misaligned
  3. Identifies the specific spatial-temporal regions that need correction
- **Refinement:** Preserves correctly-generated regions, only regenerates misaligned parts via joint optimization
- **Tested on:** EvalCrafter, T2V-CompBench, across 4 T2V backbones
- **Significance for alignment detection:** This is the cleanest "detect then fix" pipeline. The detection stage is essentially a VLM-based alignment classifier.

---

## 2. Reward Models for Video-Prompt Alignment

Reward models implicitly do alignment detection — they score how well a video matches a prompt.

### SG-PVR / Plan-and-Verify (Jun 2026)
- **Paper:** *Plan-and-Verify Video Reward Reasoning with Spatio-Temporal Scene Graph Grounding*
- **Authors:** Hyomin Kim et al.
- **arXiv:** 2606.11838
- **Key idea:** Addresses two weaknesses in existing video reward models: (1) they don't systematically verify every condition in the prompt, (2) visual evidence for judgments is implicit.
- **Architecture:**
  - Decomposes prompt into atomic claims (a "verification plan")
  - Extracts spatio-temporal scene graph from video (entities, attributes, temporally-grounded relations)
  - Each claim is verified against both the video AND the scene graph
  - Scene graph serves as persistent structured visual reference
- **Performance:** Strong on semantic alignment, especially fine-grained temporal semantics. Used as test-time reranker.
- **Significance:** The "plan-and-verify" paradigm is a structured approach to alignment detection — explicitly checking each sub-condition.

### VideoScore & VideoFeedback Dataset (Jun 2024)
- **Paper:** *VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation*
- **Authors:** Xuan He, Dongfu Jiang, Ge Zhang et al.
- **arXiv:** 2406.15252
- **Dataset:** VideoFeedback — 37.6K synthesized videos from 11 video generative models, each with human multi-aspect preference scores
- **Aspects rated:** Temporal consistency, visual quality, dynamics, text-video alignment
- **Approach:** Fine-tune MLLM (LLaVA-based) as a multi-aspect video evaluator
- **Use case:** Can be used as a reward model for alignment detection/scoring

### VideoScore2 (Sep 2025)
- **Paper:** *VideoScore2: Think before You Score in Generative Video Evaluation*
- **Authors:** Xuan He, Dongfu Jiang, Ping Nie et al.
- **arXiv:** 2509.22799
- **Key addition:** Chain-of-thought reasoning before scoring. The model first generates a structured evaluation (identifying strengths/weaknesses) then produces scores.
- **Aspects:** Replaces single opaque score with interpretable analysis
- **Better generalization** than VideoScore to out-of-distribution videos

### VisionReward (Dec 2024)
- **Paper:** *VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation*
- **Authors:** Jiazheng Xu, Yu Huang, Jiale Cheng et al.
- **arXiv:** 2412.21059
- **Approach:** Multi-dimensional preference decomposition → token-level reward aggregation
- **Covers both image AND video** generation
- **Goal:** Interpretable reward model that explains *why* a score was given

### T2VScore (Jan 2024)
- **Paper:** *Towards A Better Metric for Text-to-Video Generation*
- **Authors:** Jay Zhangjie Wu, Guian Fang, Haoning Wu, Xintao Wang et al.
- **arXiv:** 2401.07781
- **Two criteria:**
  1. **Text-Video Alignment** — fidelity of video to text description
  2. **Video Quality** — production caliber (mixture of experts)
- **Dataset:** TVGE — 2,543 videos with human judgments on both criteria
- **Approach:** Fine-tuned CLIP-based model for alignment + quality experts

---

## 3. Hallucination Detection in Video Understanding

Video hallucination detection is adjacent — detecting when a model "sees" things that aren't in the video.

### VideoHallucer (Jun 2024)
- **Paper:** *VideoHallucer: Evaluating Intrinsic and Extrinsic Hallucinations in Large Video-Language Models*
- **Authors:** Yuxuan Wang, Yueqian Wang, Dongyan Zhao et al.
- **arXiv:** 2406.16338
- **First comprehensive benchmark** for hallucination detection in LVLMs
- **Categorization:**
  - **Intrinsic hallucinations** — incorrect assertions about visible content (wrong attributes, actions, counts)
  - **Extrinsic hallucinations** — assertions about content not present in the video at all
- **Methodology:** Fine-grained QA pairs per video, covering object, attribute, action, count, relation, and temporal dimensions
- **Significance for alignment detection:** The intrinsic/extrinsic distinction maps directly to alignment detection — a generated video that *should* contain certain objects but doesn't is similar to an extrinsic hallucination

### PaMi-VDPO (Apr 2025)
- **Paper:** *PaMi-VDPO: Mitigating Video Hallucinations by Prompt-Aware Multi-Instance Video Preference Learning*
- **Authors:** Xinpeng Ding, Kui Zhang, Jianhua Han et al.
- **arXiv:** 2504.05810
- **Approach:** Online preference learning framework for video LLMs
- Uses video augmentations to generate preference pairs (misaligned vs aligned)
- Patch-aware multi-instance learning for fine-grained alignment

---

## 4. Evaluation Benchmarks for Video-Text Alignment

These are the testbeds where alignment detection methods are evaluated.

### VBench Suite (Nov 2023 → Mar 2025)
| Version | arXiv | Focus |
|---------|-------|-------|
| VBench | 2311.17982 | Basic faithfulness: aesthetics, temporal consistency, prompt adherence |
| VBench++ | 2411.13503 | Extended capabilities and versatility |
| VBench-2.0 | 2503.21755 | **Intrinsic faithfulness** — physical laws, commonsense, anatomy, composition |

**VBench-2.0 is the most relevant** for alignment detection: it evaluates "intrinsic faithfulness" across 5 dimensions:
1. **Human Fidelity** — anatomical correctness, natural motion
2. **Controllability** — adherence to specified attributes/actions
3. **Creativity** — novel but plausible compositions
4. **Physics** — adherence to physical laws
5. **Commonsense** — real-world consistency

Uses both generalist VLMs/LLMs and specialist anomaly detectors.

### FETV (Nov 2023)
- **arXiv:** 2311.01813
- **Multi-aspect categorization** of prompts: major content, attributes to control, prompt complexity
- **Temporal-aware** categories specific to video
- **Key finding:** Existing metrics (CLIPScore, FVD) correlate poorly with human evaluation
- Introduced improved CLIPScore and FVD variants with higher human correlation

### T2VQA (Mar 2024)
- **arXiv:** 2403.11956
- Largest T2V quality assessment dataset (T2VQA-DB)
- Human subjective ratings for generated videos
- More about quality than semantic alignment, but overlap exists

### DynamicEval (Oct 2025)
- **arXiv:** 2510.07441
- **Critiques:** VBench & EvalCrafter focus on static/subject-centric prompts, miss dynamic motion evaluation
- Introduces evaluation under camera motion and dynamic scenes
- Proposes video-level (not aggregate model-level) evaluation

### World Consistency Score (Jul 2025)
- **arXiv:** 2508.00144
- **Four sub-components:** object permanence, relation stability, causal compliance, flicker penalty
- Each measures a distinct aspect of temporal world consistency
- Unified single metric for internal consistency of generated videos

### EvalCrafter
- Existing benchmark (no specific arXiv paper identified)
- Referenced as one of the standard T2V evaluation benchmarks
- Used by VideoRepair for evaluation

---

## 5. Related Alignment Techniques

### SARA (May 2026)
- *Semantically Adaptive Relational Alignment for Video Diffusion Models*
- arXiv: 2605.07800
- Improves fine-grained text following by distilling spatio-temporal token relations
- Related to VideoREPA and MoAlign

### LatSearch (Mar 2026)
- *Latent Reward-Guided Search for Faster Inference-Time Scaling in Video Diffusion*
- arXiv: 2603.14526
- Uses reward models (like VideoScore) to guide noise search at inference time
- Optimizes initial noise latent, not the video itself

---

## Key Observations & Gaps

1. **Detection is typically a means, not an end.** Most works detect misalignment to *fix* it (VideoRepair) or to *reward/penalize* it (reward models). Dedicated *detection-only* systems are rare.

2. **VLM-as-judge is the dominant paradigm.** Nearly all approaches use a vision-language model (LLaVA, CLIP-based) to evaluate prompt-video alignment. This means alignment detection quality is bounded by VLM capability.

3. **Scene graphs enable structured verification.** SG-PVR shows that explicit scene graph extraction + claim decomposition outperforms holistic scoring. This suggests detection benefits from structured reasoning.

4. **Intrinsic vs extrinsic / superficial vs intrinsic.** VideoHallucer and VBench-2.0 both make this distinction. The deeper challenge is detecting misalignment that *looks* right but violates physics, commonsense, or causal rules.

5. **No dedicated "alignment detection" benchmark exists.** Work is evaluated on general T2V benchmarks (EvalCrafter, VBench, T2V-CompBench). A benchmark specifically for alignment detection — where the task is to classify/score alignment in a fine-grained way — is missing.

6. **Temporal alignment is under-addressed.** Most evaluation focuses on per-frame or short-clip alignment. Long-range temporal consistency and causal chain verification are open problems.

---

## Papers at a Glance

| Paper | Venue | Year | Core Contribution | Relevance |
|-------|-------|------|-------------------|-----------|
| VideoRepair | arXiv | Nov 2024 | MLLM-based misalignment detection + refinement | ★★★★★ Detection system |
| SG-PVR | arXiv | Jun 2026 | Scene graph + plan-and-verify reward | ★★★★★ Structured verification |
| VideoScore | NeurIPS? | Jun 2024 | Multi-aspect video reward model + 37.6K dataset | ★★★★☆ Reward model as detector |
| VideoScore2 | arXiv | Sep 2025 | Chain-of-thought scoring | ★★★★☆ Interpretable detection |
| VisionReward | arXiv | Dec 2024 | Multi-dimensional preference reward | ★★★★☆ Reward model |
| VideoHallucer | ACL? | Jun 2024 | Hallucination detection benchmark | ★★★★☆ Intrinsic/extrinsic detection |
| VBench-2.0 | arXiv | Mar 2025 | Intrinsic faithfulness benchmark | ★★★★☆ Detection evaluation |
| FETV | arXiv | Nov 2023 | Fine-grained T2V evaluation benchmark | ★★★☆☆ Benchmark |
| T2VScore | arXiv | Jan 2024 | Text-video alignment + quality metric | ★★★☆☆ Alignment metric |
| PaMi-VDPO | arXiv | Apr 2025 | Preference learning for video alignment | ★★★☆☆ Detection through DPO |
| World Consistency Score | arXiv | Jul 2025 | Object permanence + causal compliance | ★★★☆☆ Consistency detection |
