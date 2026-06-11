---
title: Video Alignment in Physics Context Alignment
tags: [video-diffusion, alignment, physics, context-alignment, research]
date: 2026-06-11
status: draft
---

# Video Alignment in Physics — Context Alignment

**Core question:** Given a generated video described by a prompt, does the video accurately represent the *physical scenario context* — the setting, objects, materials, environmental conditions, and their implied physical properties? This is distinct from Physics Alignment (Stage 2, which checks whether dynamics obey physical laws).

## The Two-Stage Decomposition

We decompose video physics alignment detection into two hierarchical stages:

```
Prompt → [Context Alignment] → [Physics Alignment] → Alignment verdict
```

| Stage | What it checks | Example failures |
|-------|---------------|-----------------|
| **Context Alignment** | Physical scenario setup: objects present, materials, environmental conditions, spatial layout, implied physical properties | Water in a desert scene, wooden object that behaves like metal, indoor objects in an outdoor scene, impossible material combinations |
| **Physics Alignment** | Physical law compliance: motion, forces, collisions, deformation, causality, object permanence | Object floating without support, impossible trajectories, non-rigid bodies passing through each other, broken cause-effect chains |

**Why decompose?** Context misalignment renders Physics Alignment meaningless — if the scenario itself is physically impossible at the setup level, asking whether the dynamics obey physics is the wrong question. The hierarchy is: get the world right first, then check how it moves.

---

## 1. What Context Alignment Covers

Context Alignment checks whether the generated video matches the **static physical configuration** implied by the prompt. This spans:

### 1.1 Object-Scene Consistency
- Are objects appropriate for the setting? (Fish in water vs fish in desert)
- Are objects correctly categorized by their physical type? (Rigid body vs deformable vs fluid)
- Do object scales match the scene? (Miniature vs full-scale in the same frame)

### 1.2 Material & Surface Properties
- Does the material appearance match the prompt? (Wood grain vs metallic sheen vs fabric weave)
- Are material-implied behaviors consistent? (Glass should be transparent/refractive, cloth should appear flexible)
- Are surface interactions correct? (Water should reflect, matte surfaces shouldn't glare)

### 1.3 Environmental Context
- Lighting: natural vs artificial, time-of-day consistency, shadow direction uniformity
- Weather: rain, snow, fog matching scene setting
- Terrain and geography: indoor vs outdoor, urban vs natural, planetary environment
- Atmospheric conditions: underwater vs air vs vacuum

### 1.4 Implied Physical Properties
Properties that are *inferred* from the context even when not explicitly stated:
- Density: does the object's visual mass match expectations? (Boulder vs balloon at the same scale)
- Temperature: does the scene indicate heat/cold appropriately? (Steam implies hot, frost implies cold)
- State of matter: solid/liquid/gas consistent with material and environment
- Material-specific behavior: wet cloth drapes differently than dry cloth

### 1.5 Spatial & Compositional Physics
- Is the spatial arrangement physically possible? (Objects can't occupy the same 3D space simultaneously unless one is transparent)
- Are occlusions physically plausible? (A small object shouldn't occlude a larger one at the same depth)
- Are support relationships correct? (Table supports cup, not cup supports table)
- Is the camera perspective physically coherent? (No impossible viewing angles)

---

## 2. Why This Is Harder Than Semantic Alignment

Existing alignment detection (VideoRepair, SG-PVR) focuses on *semantic* alignment — does the prompt's entity/attribute/relation set match the video? Context Alignment differs in key ways:

| Dimension | Semantic Alignment (Stage 01) | Context Alignment |
|-----------|------------------------------|-------------------|
| What's checked | Prompt-listed entities/attributes/relations | Implied physical scenario beyond prompt text |
| Failure type | Missing objects, wrong attributes, broken relations | Physically impossible configurations |
| Detection requires | VLM QA (does object X exist?) | Physical world model + commonsense reasoning |
| False positive risk | VLM hallucinates "yes" to a missing object | VLM fails to recognize physical implausibility |
| False negative risk | VLM misses subtle attribute mismatch | VLM accepts impossible scene as plausible |

**The key challenge:** Context Alignment requires *physical commonsense* — knowledge about how materials behave, what environments support what objects, and what spatial configurations are physically realizable. Current VLMs have this only partially and unevenly.

---

## 3. Existing Work That Touches Context Alignment

### Papers at a Glance (Context Alignment)

| # | Paper | arXiv | Year | Core Contribution | Relevance |
|---|-------|-------|------|-------------------|-----------|
| 1 | **PhysGame** | [2412.01800](https://arxiv.org/abs/2412.01800) | Dec 2024 | Physical commonsense violations in gameplay (mechanics, kinematics, optics, materials) | ★★★★★ |
| 2 | **CRONOS** | [2605.23699](https://arxiv.org/abs/2605.23699) | May 2026 | Counterfactual physical consistency with controlled interventions on scene/viewpoint/object | ★★★★☆ |
| 3 | **AV-Phys Bench** | [2605.07061](https://arxiv.org/abs/2605.07061) | May 2026 | Audio-video physics understanding; Steady State, Event, Environment transitions | ★★★★☆ |
| 4 | **VBench-2.0** | [2503.21755](https://arxiv.org/abs/2503.21755) | Mar 2025 | Intrinsic faithfulness — Physics & Commonsense dimensions via VLM + specialist detectors | ★★★★★ |
| 5 | **VBench** | [2311.17982](https://arxiv.org/abs/2311.17982) | Nov 2023 | 16-dim benchmark; spatial relationship, motion smoothness, temporal flickering | ★★★☆☆ |
| 6 | **World Consistency Score** | [2508.00144](https://arxiv.org/abs/2508.00144) | Jul 2025 | Object permanence, relation stability, causal compliance, flicker penalty | ★★★★☆ |
| 7 | **VideoRepair** | [2411.15115](https://arxiv.org/abs/2411.15115) | Nov 2024 | MLLM-based misalignment detection + refinement; extensible with physics QA | ★★★★☆ |
| 8 | **SG-PVR** | [2606.11838](https://arxiv.org/abs/2606.11838) | Jun 2026 | Scene graph + plan-and-verify reward; extensible with physical rule checking | ★★★★☆ |

### 3.1 Physical Commonsense Violation Detection

#### PhysGame (Dec 2024)
- **Paper:** *PhysGame: Uncovering Physical Commonsense Violations in Gameplay Videos*
- **Authors:** Meng Cao, Haoran Tang, Haoze Zhao et al.
- **arXiv:** [2412.01800](https://arxiv.org/abs/2412.01800)
- **What it does:** Designed as a benchmark to evaluate physical commonsense understanding in Video LLMs, specifically for **gameplay videos** containing glitches. 880 videos spanning **4 domains**:
  - Mechanics (rigid body, collision, gravity)
  - Kinematics (motion, trajectory, velocity)
  - Optics (reflection, refraction, shadows)
  - Material properties (elasticity, density, texture)
- **Detection method:** Video LLM evaluated on QA about whether physical violations exist. Also proposes PhysVLM — a fine-tuned video LLM trained on PhysInstruct (140K QA pairs) and PhysDPO (34K preference pairs).
- **Relevance to context alignment:** ★★★★★ — The closest existing work to Context Alignment. The 4-domain taxonomy maps directly to our material and environmental consistency dimensions. However, it focuses on *gameplay glitches* (obvious render errors) rather than *plausible-looking but physically impossible* real-world scene configurations.
- **Key limitation for our framing:** Gameplay videos have different physics from real-world scenes. A character floating in a game is a feature, not a bug. The benchmark doesn't distinguish intentional vs unintentional violations.

#### CRONOS (May 2026)
- **Paper:** *CRONOS: Benchmarking Counterfactual Physical Consistency in Video Models*
- **Authors:** León Begiristain, Olaf Dünkel, Adam Kortylewski
- **arXiv:** [2605.23699](https://arxiv.org/abs/2605.23699)
- **What it does:** Evaluates whether video generation models' predictions respond appropriately to **controlled interventions** on scene context, viewpoint, object appearance, and object category — while keeping the underlying physical event (collision, occlusion, fall) fixed.
- **Built in Unreal Engine** — photorealistic, controlled environment
- **Key finding:** Prediction quality for the same physical event varies substantially with appearance, environment, and *especially viewpoint* changes. Open-source video generators show **substantial failures** in counterfactual physical consistency.
- **Relevance to context alignment:** ★★★★☆ — Directly relevant. The "scene context" intervention tests whether models understand how physical events depend on environment. The finding that viewpoint changes degrade consistency points to a fundamental context representation problem.
- **Limitation:** Focuses on video *prediction* models (future frame prediction), not text-to-video generation. Evaluation is on physical *dynamics* consistency, not static configuration plausibility.

#### AV-Phys Bench (May 2026)
- **Paper:** *Do Joint Audio-Video Generation Models Understand Physics?*
- **Authors:** Zijun Cui, Xiulong Liu, Hao Fang et al.
- **arXiv:** [2605.07061](https://arxiv.org/abs/2605.07061)
- **What it does:** Evaluates physical commonsense in joint audio-video generation across **3 scene categories**:
  - **Steady State** — static physical configuration (most relevant to Context Alignment)
  - **Event Transition** — how a physical event unfolds
  - **Environment Transition** — how physics changes with environment
- **5 evaluation dimensions:** visual semantic, audio semantic, visual physical commonsense, audio physical commonsense, cross-modal physical commonsense
- **Key finding:** All models (including proprietary ones) perform poorly on anti-physics prompts that deliberately request physically inconsistent behavior.
- **Relevance to context alignment:** ★★★★☆ — The "Steady State" and "Anti-AV-Physics" categories directly test context alignment. The 5-dimension evaluation framework is a useful template.
- **Limitation:** Audio-visual focus means many context alignment dimensions (material properties, spatial configuration) are not explicitly tested. They evaluate *intentional* anti-physics prompts rather than *unintentional* context failures.

### 3.2 Benchmark Dimensions for Physical Plausibility

#### VBench (Nov 2023)
- **Paper:** *VBench: Comprehensive Benchmark Suite for Video Generative Models*
- **Authors:** Ziqi Huang, Yinan He, Jiashuo Yu et al. (CVPR 2024)
- **arXiv:** [2311.17982](https://arxiv.org/abs/2311.17982)
- **What it covers:** 16 hierarchical dimensions of video generation quality
- **Context-relevant dimensions:**
  - **Subject identity inconsistency** — do objects maintain identity?
  - **Motion smoothness** — is motion physically natural?
  - **Temporal flickering** — is lighting/color temporally stable?
  - **Spatial relationship** — are object spatial relations correct?
- **Detection method:** CLIP-based scoring + human preference alignment dataset
- **Relevance:** ★★★☆☆ — Foundational benchmark, but the physical context dimensions are basic (CLIP-based, not deep physical reasoning)

#### VBench-2.0 (Mar 2025)
- **Paper:** *VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness*
- **Authors:** Dian Zheng, Ziqi Huang, Hongbo Liu et al.
- **arXiv:** [2503.21755](https://arxiv.org/abs/2503.21755)
- **What it does:** Goes beyond "superficial faithfulness" (aesthetics, temporal consistency) to **intrinsic faithfulness** — does the video adhere to real-world principles?
- **5 key dimensions:**
  1. **Human Fidelity** — anatomical correctness, natural motion
  2. **Controllability** — adherence to specified attributes/actions
  3. **Creativity** — novel but plausible compositions
  4. **Physics** — adherence to physical laws
  5. **Commonsense** — real-world consistency
- **Detection framework:** Integrates both **generalist** VLMs/LLMs and **specialist** anomaly detectors per dimension
- **Relevance:** ★★★★★ — Most directly relevant benchmark for Context Alignment. The Physics and Commonsense dimensions explicitly target what we're trying to detect. The hybrid generalist+specialist evaluation approach validates our "VLM + rules" intuition.
- **Limitation:** Still a benchmark (evaluation of models), not a detection framework. Doesn't distinguish context vs dynamics physics.

#### World Consistency Score (Jul 2025)
- **Paper:** *World Consistency Score: A Unified Metric for Video Generation Quality*
- **Authors:** Akshat Rakheja, Aarsh Ashdhir, Aryan Bhattacharjee, Vanshika Sharma
- **arXiv:** [2508.00144](https://arxiv.org/abs/2508.00144)
- **4 sub-components:**
  - **Object permanence** — objects shouldn't appear/disappear without cause (touches context alignment)
  - **Relation stability** — spatial relations should persist when physics requires it
  - **Causal compliance** — effects should follow causes physically
  - **Flicker penalty** — temporal visual instability
- **Relevance:** ★★★★☆ — Object permanence and relation stability directly measure context-alignment-like properties. The unified score approach is a clean output format.
- **Limitation:** Designed for *consistency over time*, not checking whether the initial static configuration makes physical sense. An impossible scene rendered consistently would score well.

### 3.3 Alignment Detection Methods (from Stage 01)

These are already documented in `01-landscape-alignment-detection.md`. Here we note their relevance to Context Alignment specifically:

#### VideoRepair (Nov 2024) — arXiv: 2411.15115
- **Detection method:** Generate evaluation questions from prompt → answer via MLLM on video frames
- **Extension for Context Alignment:** Could add *physical plausibility questions* that aren't directly from the prompt:
  - *"Does the scene contain contradictions like indoor furniture outdoors without a plausible reason?"*
  - *"Are material properties (wood, metal, glass) visually consistent?"*
- **Key advantage:** Training-free, model-agnostic. The question-generation step can be extended with a physics commonsense knowledge base.

#### SG-PVR (Jun 2026) — arXiv: 2606.11838
- **Detection method:** Extract spatio-temporal scene graph from video → decompose prompt into atomic claims → verify each claim against scene graph
- **Extension for Context Alignment:** The scene graph captures the static configuration. Could add *physical commonsense rules* to the verification step:
  - Material compatibility rules: {glass} → {transparent, fragile}
  - Scene compatibility: {indoor} → {walls, ceiling, furniture}
  - Support physics: {A on B} → size(A) < size(B)
- **Key advantage:** Structured, debuggable, and each rule violation can be attributed to specific scene graph elements.

---

## 4. Detection Approaches for Context Alignment

### Approach A: Physical Plausibility Question Generation (PPQG)
**Inspired by:** VideoRepair's question generation

Extend the evaluation question approach with a *physics-aware question generator*:

1. **Parse prompt** for explicit physical context cues (materials, settings, environments)
2. **Generate context-specific plausibility questions** using a physics commonsense knowledge base:
   - *"Is the glass of water in a desert scene?"* (material-environment contradiction)
   - *"Does the wooden table have a metallic reflection?"* (material property mismatch)
   - *"Are the shadows consistent with a single light source?"* (lighting consistency)
   - *"Is a person visible through a solid wall?"* (occlusion violation)
3. **Answer questions via VLM** observing video frames
4. **Aggregate** into context alignment score

**Key insight:** The questions must go beyond what the prompt explicitly says. A prompt "a glass of water on a table in a room" doesn't say "water should not have metallic sheen" — but context alignment should still catch it.

**Challenge:** Building the question generator. Needs a taxonomy of physical scene violations.

### Approach B: Scene Graph + Physical Rule Verification
**Inspired by:** SG-PVR

1. Extract scene graph from initial keyframe(s) — objects, materials, spatial relations, lighting cues
2. Apply physical commonsense rules to the graph:
   - Material compatibility: {glass} → {transparent, fragile, non-magnetic}
   - Scene compatibility: {indoor setting} → {walls, ceiling, furniture}
   - Support physics: {object A on object B} → size(A) < size(B), A above B
   - Occlusion constraints: {opaque object A occludes B} → A is closer to camera than B
3. Flag any rule violations as context misalignment

**Advantage:** More principled than QA-based — rules are explicit and debuggable
**Challenge:** Need comprehensive rule engine; covering edge cases is difficult

### Approach C: Self-Consistency Across Physical Dimensions
Check cross-dimensional consistency without external rules:

1. Ask VLM to describe the scene from multiple physical perspectives:
   - *"What materials are present?"*
   - *"What's the lighting setup?"*
   - *"What's the spatial arrangement?"*
   - *"What's the environment/weather?"*
2. Check for contradictions across descriptions
3. Flag inconsistency as potential context misalignment

**Insight:** A physically plausible scene will have consistent answers across all dimensions. An implausible one will reveal contradictions (e.g., "indoor room" + "tropical forest humidity").

---

## 5. What We Need to Build

### 5.1 A Taxonomy of Context Violations

```
Context Violations
├── Object-Scene Mismatches
│   ├── Wrong environment (indoor object in outdoor scene without rationale)
│   ├── Impossible object combinations (fire + underwater)
│   └── Scale violations (giant version of normally small object treated as normal)
├── Material Property Violations
│   ├── Incorrect material appearance (wood with metallic reflection)
│   ├── Missing expected material properties (glass that's opaque)
│   └── Material-state errors (melting ice that doesn't produce water)
├── Spatial Configuration Violations
│   ├── Impossible occlusions (foreground object smaller than what it occludes)
│   ├── Support relationship violations (unsupported floating objects)
│   └── Penetration/overlap errors (two solids occupying same space)
├── Environmental Inconsistencies
│   ├── Lighting contradictions (hard shadows + overcast lighting simultaneously)
│   ├── Weather mismatches (rain + no cloud cover)
│   └── Atmospheric errors (visible breath in clearly tropical scene)
└── Physical Commonsense Violations
    ├── Object permanence errors (objects that shouldn't exist in this world)
    ├── Categorical errors (treating a rigid object as fluid)
    └── State-of-matter errors (solid water at room temperature)
```

### 5.2 A Probe Dataset for Context Alignment

To evaluate any detection method, we need a dataset of videos with known context violations. Possible construction:

1. **Take existing T2V benchmarks** (EvalCrafter, VBench)
2. **Inject context violations** via editing (or deliberately choose prompts likely to produce them)
3. **Annotate per violation type** with ground truth labels

Or: **Curate a set of prompts designed to trigger context misalignment** in current T2V models:
- *"A wooden table with glass surface reflection"* (tests material-scene consistency)
- *"A fish swimming in the desert"* (tests object-scene consistency)
- *"Indoor living room with ocean waves"* (tests environment consistency)
- *"A small child carries an elephant"* (tests scale/physics commonsense)

---

## 6. Open Questions for Context Alignment

1. **Where does commonsense end and physics begin?** The boundary between Context and Physics Alignment is fuzzy. Is "a glass table should be transparent" context (material property) or physics (optics)? We define it as context when it's about the *static configuration*.

2. **How much context should a VLM be expected to infer?** A prompt like *"A cozy cabin in the snow"* implies warmth, insulation, snow on the roof. Which implications are "must check" vs "nice to check"?

3. **Can we separate VLM understanding errors from actual context misalignment?** If the VLM says "the scene is physically impossible" but it's actually fine, we have a false alarm. The detection system's ceiling is the VLM's physical reasoning ability.

4. **Is context alignment task-dependent?** A physics simulator for engineering needs strict context alignment. A creative/animated video may intentionally violate context. The strictness should be adjustable.

5. **Do we need scene graph + rules, or is VLM question-answering sufficient?** The trade-off between structured reasoning (SG-PVR-style) and free-form VLM QA (VideoRepair-style) is central to this stage.

---

## 7. Connection to Physics Alignment (Stage 2)

Context Alignment feeds into Physics Alignment as follows:

- **If Context Alignment fails** → report context violation, skip physics check (no point evaluating dynamics of an impossible world)
- **If Context Alignment passes** → proceed to Physics Alignment: check motion, forces, collisions, causality, deformation behavior
- **Partial context alignment** → some context violations allow physics checking on the *consistent sub-regions* of the video

The two-stage pipeline produces a richer diagnosis than a single "physics score":
```
Result = {
  context: {score, violations: [type, location, severity]},
  physics: {score, violations: [type, location, severity]},
  overall_physics_alignment: weighted_combination
}
```

---

## Next Steps for This Stage

1. Build the comprehensive taxonomy of context violations (draft above, needs refinement)
2. Survey existing benchmark datasets for videos with physical impossibilities
3. Test VLM-based detection (LLaVA-Video, Qwen2.5-VL) on context-violation prompts to establish baseline
4. Evaluate whether QA-based or scene-graph-based detection is more effective for context alignment
5. Design a probe dataset for controlled evaluation
