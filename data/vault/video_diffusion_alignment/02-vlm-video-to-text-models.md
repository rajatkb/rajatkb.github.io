---
title: VLM & Video-to-Text Models for Scene Understanding — Capability Catalog
tags: [video-diffusion, alignment, vlm, video-to-text, scene-understanding, research]
date: 2026-06-11
draft: true
---

# VLM & Video-to-Text Models for Scene Understanding

**Core question:** Which VLMs and video-to-text models are capable of providing rich scene descriptions and understanding — the foundational capability that alignment detection methods (VideoRepair, SG-PVR, VideoScore) rely on as their "judge" backbone?

This note catalogs models by their ability to produce:
- **Dense scene descriptions** (what entities, attributes, relations are present)
- **Temporal understanding** (sequencing, action chains, causality over time)
- **Fine-grained alignment verification** (does the video match a specific claim?)
- **Spatial grounding** (where in the frame things happen)

---

## 1. General-Purpose Video Understanding / Dense Captioning

Models that take a video and output natural language descriptions at varying levels of detail.

### LLaVA-Video (Oct 2024)
- **Paper:** *LLaVA-Video: Learning Video Understanding from What We See and What We Say*
- **arXiv:** 2411.01747
- **Architecture:** LLaVA-style (projector-based vision-language alignment) extended to video
- **Key innovation:** Uses a hybrid frame sampling + token compression strategy — 32 frames at reduced resolution + selective high-res patches on key frames
- **Strengths:**
  - Strong general-purpose video understanding — objects, actions, temporal ordering
  - Used as the MLLM backbone in **VideoRepair**'s detection stage
  - Open-source weights available (LLaVA-Video-7B/13B/72B)
- **Limitations:**
  - Frame sampling means it can miss fine-grained temporal details in long videos
  - Not explicitly trained for dense scene description (more QA-oriented)
- **Relevance to alignment detection:** ★★★★★ — Already proven as the detection backbone in VideoRepair

### VideoLLaMA 2 (Jun 2024)
- **Paper:** *VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs*
- **arXiv:** 2406.07476
- **Architecture:** Q-Former connector between video encoder (ViT) and LLM backbone (Zephyr/LLaMA), with STC-Adapter for spatial-temporal modeling
- **Key strengths:**
  - Dual encoder: spatial (image-level) + temporal (3D convolution) branches
  - Audio understanding via ImageBind audio encoder
  - Competitive on VideoChatGPT, MSVD, MSRVTT, TGIF benchmarks
  - Open-source (7B, 13B, 72B)
- **Limitations:**
  - Temporal resolution is limited by fixed frame count
  - Dense captioning not a primary training objective
- **Relevance:** ★★★★☆ — Good general-purpose video VLM with temporal modeling

### Qwen2.5-VL (Jan 2025)
- **Paper:** *Qwen2.5-VL*
- **arXiv:** 2502.13923
- **Architecture:** Qwen2.5 LLM backbone + Vision Transformer with dynamic resolution
- **Key strengths:**
  - Dynamic resolution (can process images at native resolution)
  - Video understanding with temporal dynamic resolution
  - Strong on video QA, captioning, and visual grounding
  - Multi-lingual (strong in both English and Chinese)
- **Limitations:**
  - Video frame count cap limits long-video understanding
  - Less specialized for fine-grained scene graph extraction
- **Relevance:** ★★★★☆ — Strong general VLM, could serve as detection backbone

### InternVideo2 (Dec 2024)
- **Paper:** *InternVideo2: Scaling Video Foundation Models for Multimodal Understanding*
- **arXiv:** 2412.13499
- **Architecture:** Progressive training from image → video → video-text, with masked video modeling + cross-modal contrastive learning
- **Key strengths:**
  - Trained at massive scale (900M video-text pairs after filtering)
  - Strong on both video-level understanding AND frame-level tasks
  - Open-source weights available
  - Top results on Video-MME, LongVideoBench, EgoSchema
- **Limitations:**
  - More of a video encoder than a full video-to-text model (needs a decoder LLM)
  - Primarily evaluated on classification/retrieval, not dense description generation
- **Relevance:** ★★★☆☆ — Strong video representation model, best used as encoder backbone

---

## 2. Scene-Specialized Models (Dense Captioning & Scene Graphs)

Models designed to produce structured scene descriptions, entity-level detail, or scene graphs from video.

### VILA (Jun 2024)
- **Paper:** *VILA: On Pre-training for Visual Language Models*
- **Authors:** Ji Lin, Hongxu Yin, Wei Ping et al. (NVIDIA)
- **arXiv:** 2312.07533 (extended)
- **Architecture:** Interleaved vision-language pre-training, video-capable extension
- **Key strengths:**
  - Strong at interleaved image-text and video-text understanding
  - Can handle multiple images / video frames in context
  - Used as backbone for video captioning and QA
  - Open-source (VILA-7B/13B, VILA2)
- **Limitations:**
  - Primarily image-text interleaved, video is an extension not the core
  - Less specialized for dense scene description
- **Relevance:** ★★★★☆ — Solid general VLM, used in some video pipelines

### InstructVideo / VideoChat (variants)
- Emerging line: instruction-tuned video VLMs that can follow detailed scene description prompts
- **VideoChat (May 2024)** — arXiv: 2305.06355 — Video-centric instruction tuning, supports dense captioning
- **VideoChat2 (Jun 2024)** — arXiv: 2406.08313 — Improved spatial-temporal modeling, stronger on fine-grained tasks

### Osprey (for dense region captioning — image only currently)
- **Paper:** *Osprey: Pixel Understanding with Visual Instruction Tuning*
- **arXiv:** 2312.10032
- **Note:** Image-only, but relevant as a technique — pixel-level mask-guided captioning for fine-grained entity description
- **Video extension potential:** Could be adapted for per-frame dense captioning

---

## 3. VLMs Used Specifically in Alignment / Reward Models

These are models already deployed in the alignment detection pipelines from Stage 01.

### LLaVA-Video (in VideoRepair)
- **Role:** Generates evaluation questions from prompt, then answers them by watching the video
- **Detection workflow:**
  1. Input: prompt → "What entities/attributes/relations should be present?"
  2. LLaVA-Video checks each via QA over video frames
  3. Output: fine-grained alignment scores per aspect
- **Why it works:** QA-based verification is more reliable than asking for a single score — models are better at answering factual questions than producing calibrated scores
- **Key prompt technique:** "Based on the video, answer this question about the prompt's requirements..."

### VideoScore / VideoScore2 (LLaVA-based)
- **Role:** End-to-end multi-aspect scoring model
- **Difference from LLaVA-Video in VideoRepair:** VideoScore is fine-tuned specifically for evaluation (regression head for scores), whereas VideoRepair uses a general-purpose VLM with prompt engineering
- **VideoScore2 improvement:** Chain-of-thought before scoring — model first writes an evaluation, then produces scores
- **Aspects scored:** Temporal consistency, visual quality, dynamics, text-video alignment

### SG-PVR (uses a VLM for scene graph extraction)
- **Role:** Extracts structured spatio-temporal scene graph from video
- **Scene graph components:**
  - Entities (objects with bounding boxes + attributes)
  - Relations (spatial, temporal, interaction)
  - Temporal grounding (when in the video each relation holds)
- **VLM used:** Likely InternVideo or a specialized scene graph model
- **Why it matters for detection:** Structured scene graphs enable claim-by-claim verification against the prompt, which is more robust than holistic scoring

---

## 4. Video Captioning Models (Video → Dense Text)

Models primarily trained for video captioning — converting video to rich text descriptions.

### Videollama 2 (also in §1 — dual use)

### VideoChat2 / Video-ChatGPT
- **Video-ChatGPT (May 2024):** Instruction-tuned video VLM; good at conversation-level video understanding
- **Benchmark:** Video-ChatGPT benchmark (1,000 videos, 5 QA categories)

### VideoPrism (Jun 2024)
- **Paper:** *VideoPrism: A Foundational Visual Encoder for Video Understanding*
- **arXiv:** 2402.13217
- **Approach:** Contrastive + masked video modeling; web-scale pretraining on 36M videos
- **Output:** Video embeddings, not text (needs a language decoder)
- **Relevance:** Could serve as a video encoder in a custom detection pipeline

### LaViLa (Jun 2023)
- **Paper:** *Learning Video Representations from Large Language Models*
- **arXiv:** 2306.02858
- **Approach:** Uses LLM to generate pseudo-captions → trains video-to-text model
- **Dataset:** Ego4D-based with LLM-augmented captions
- **Relevance:** Methodologically interesting for generating training data for detection models

### Video LLaMA (original, Jun 2023)
- arXiv: 2306.02858 — First-generation video LLaMA
- Q-Former connector + image encoder + video encoder
- Largely superseded by VideoLLaMA 2

---

## 5. Open-Source vs API Models

A critical practical concern for building an alignment detection pipeline.

### Open-Source (weights available)
| Model | Size | License | Notes |
|-------|------|---------|-------|
| LLaVA-Video | 7B, 13B, 72B | Apache 2.0 | Best proven option for detection |
| VideoLLaMA 2 | 7B, 13B | Apache 2.0 | Good temporal modelling |
| Qwen2.5-VL | 3B, 7B, 32B, 72B | Apache 2.0 | Strong general VLM |
| InternVideo2 | Various | MIT | Encoder only (needs decoder) |
| VILA | 7B, 13B | Apache 2.0 | Interleaved understanding |

### API-Only / Partial Weights
| Model | Access | Notes |
|-------|--------|-------|
| GPT-4V / GPT-4o | API | Strong video understanding, but costly at scale |
| Gemini 1.5 Pro | API | Very long context (up to 10M tokens for video) |
| Claude 3.5 Sonnet / 4 | API | Image-only (no native video in API) |
| VideoPrism | Weights available | Encoder only |

---

## 6. Practical Capability Matrix for Alignment Detection

| Capability | LLaVA-Video | VideoLLaMA 2 | Qwen2.5-VL | GPT-4o | SG-PVR |
|------------|-------------|-------------|-----------|--------|--------|
| Dense scene description | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★★ | ★★★★★ |
| Temporal ordering | ★★★★☆ | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★☆ |
| Fine-grained entity detection | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★★★★ |
| Attribute verification | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★★ |
| Relation/causality detection | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| Claim-by-claim verification | ★★★★★* | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★★ |
| Open-source / reproducible | ✓ | ✓ | ✓ | ✗ | Partial |
| Reasoning ability | ★★★★☆ | ★★★★☆ | ★★★★★ | ★★★★★ | ★★★☆☆ |
| Cost efficiency (self-host) | ✓ | ✓ | ✓ | ✗ | Depends on backbone |

*\* LLaVA-Video scores ★★★★★ on claim-by-claim because VideoRepair specifically engineered this workflow around it.*

---

## 7. Recommendations for Alignment Detection

### Primary candidates (ranked)

1. **LLaVA-Video (7B/13B)** — Best starting point. Already used in VideoRepair's detection stage, open-source, well-documented. The QA-based verification approach (generate questions from prompt → answer from video) maps directly to alignment detection needs.

2. **Qwen2.5-VL (7B/32B)** — Strongest open-source alternative. Better reasoning than LLaVA-Video in most benchmarks, dynamic resolution allows finer visual detail. Worth testing as a drop-in replacement in VideoRepair's detection pipeline.

3. **SG-PVR style (Scene Graph + Plan-and-Verify)** — Highest potential for fine-grained alignment, but more complex to implement. Requires scene graph extraction model + claim decomposition + verification module. Best for precision-critical cases (e.g., detecting subtle attribute failures).

4. **VideoScore/VideoScore2** — If the goal is a quantitative alignment *score* (not a detection explanation), these are purpose-built. VideoScore2's COT adds interpretability. Limitation: trained on specific score distributions, may not generalize to arbitrary prompts.

### What to test in Stage 03

- LLaVA-Video 7B vs Qwen2.5-VL 7B on alignment-specific prompts (e.g., "given this prompt, which aspects of the video are misaligned?")
- Whether generating "evaluation questions from prompt" (VideoRepair approach) or "scene graph + claim verification" (SG-PVR approach) is more effective for different types of misalignment
- Compare detection accuracy across open-source models for: attribute errors, missing entities, action mismatches, temporal ordering failures
- Cost-speed-accuracy trade-off: API models (GPT-4o, Gemini) vs self-hosted (LLaVA-Video, Qwen2.5-VL)
