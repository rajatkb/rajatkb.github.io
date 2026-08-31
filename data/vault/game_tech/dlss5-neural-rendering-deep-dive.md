---
title: "DLSS 5 Neural Rendering — Technical Deep Dive"
date: 2026-08-31
tags: [dlss, nvidia, neural-rendering, ngx, renodx, modding, blackwell]
summary: "How DLSS 5 actually works: the one-step pixel-space diffusion transformer model, its NGX contract (inputs/outputs), render-path integration, real performance numbers from the leaked build, and a source-level look at how the renoDX mod stack drives it."
draft: false
---

# DLSS 5 Neural Rendering — Technical Deep Dive

Status as of 2026-08-31: announced at GTC (Mar 16, 2026), detailed at SIGGRAPH 2026 (Jul 20), **shipping fall 2026**. A pre-release runtime (`nvngx_dlssnr.dll` v310.8.0.0, 158 MB) leaked via NBA 2K27's early-access build on Aug 26, 2026 and is already running in several games through the community renoDX stack. Everything below is either from NVIDIA's own announcements/SIGGRAPH session or from hands-on analysis of that leaked build.

## TL;DR

- DLSS 5 is **neural rendering for visual fidelity**, not upscaling. It runs *on top of* the DLSS 4.5 stack (SR + frame gen underneath).
- The model is a **compact one-step pixel-space diffusion transformer**, distilled from larger offline foundation models, ~148M FP8 parameters embedded directly in the DLL, running causally ("one frame in, one frame out").
- Inputs: the rendered color frame + motion vectors (officially also albedo/normals/lighting engine buffers); community path adds depth + a trust mask.
- Output: an enhanced color buffer written back over the frame through a typed UAV — same resolution as its DLAA host evaluate.
- Official integration goes through **NVIDIA Streamline**; the community (leak) path detours `NVSDK_NGX_D3D12_CreateFeature`/`EvaluateFeature` via a ReShade add-on, with the NR pass inserted as an **inline feature (feature 18)** inside a DLSS DLAA evaluate.
- Hardware: officially **RTX 50 (Blackwell) only** — FP8 inference on 5th-gen tensor cores (sm_120 cubins). Ada (sm_89) runs it unofficially via a cubin patch; Ampere can't (no native FP8).
- First consumer perf test on an RTX 5070 Ti at 4K: **71 → 35 FPS (−51%)** — but that's the worst-case placement (full-scene, after upscaling); per-object masking should cut this dramatically.

## What DLSS 5 is (official framing)

NVIDIA announced DLSS 5 on Mar 16, 2026 at GTC, calling it "the company's most significant breakthrough in computer graphics since the debut of real-time ray tracing in 2018," with Jensen Huang calling it "the GPT moment for graphics — blending handcrafted rendering with generative AI." [1][2]

Key official statements, all from the press release and GeForce article [1][2]:

- "DLSS 5 takes a game's **color and motion vectors** for each frame as input, and uses an AI model to infuse the scene with photoreal lighting and materials that are anchored to source 3D content and consistent from frame to frame."
- The model is "trained end to end to understand complex scene semantics such as characters, hair, fabric and translucent skin, along with environmental lighting conditions like front-lit, back-lit or overcast — all by analyzing a single frame," and "generates visually precise images that handle complex elements such as **subsurface scattering on skin, the delicate sheen of fabric and light-material interactions on hair**."
- Runs in real time at up to 4K; developers get controls for **intensity, color grading and masking**; integration is "seamless, using the same NVIDIA Streamline framework used by existing DLSS and NVIDIA Reflex technologies."
- DLSS 4.5 (CES 2026) draws 23 of every 24 pixels; DLSS 5 is positioned as the next step beyond performance, toward fidelity.

At SIGGRAPH 2026 (Jul 20 keynote "Next Era of Graphics: Neural Rendering, World Models, and Simulation"), the session was led by research directors (Neil Ashton — AI physics; Edward Liu — applied deep learning; Ming-Yu Liu; Gabriele Leone — content tech), a deliberate signal that NVIDIA wanted engineering credibility after March criticism [3]. Key framing from that session:

- **"3D-guided AI"**: the model is anchored to what the game actually rendered — "learned generation defines the world and generation enriches the appearance" (Edward Liu); "AI is no longer a separate track in computer graphics — it is the rendering pipeline" (Neil Ashton) [3].
- **Three selectable AI models** (Model A/B/C tiers) let developers trade quality for performance per title [3][4].
- **Single-GPU confirmed** — the dual-RTX 5090 GTC demo was a development setup; SIGGRAPH showed the shipping configuration on one GPU [3].
- Generation is framed as a *third category* of neural rendering alongside reconstruction and function approximation: "the original renderer still owns geometry, camera, lighting and composition; the generative pass runs afterwards on top of a conventionally rendered frame" [5].

## The model

The architectural claim that matters: DLSS 5 is a **compact, one-step pixel-space diffusion transformer distilled from substantially larger foundation models**, processing each image **causally** — "one frame in, one frame out" [4][6]. Video-generative models normally batch frames; a game cannot wait for frames that haven't happened, so DLSS 5 leans on motion vectors and engine data for temporal stability instead [5].

Leaked-DLL forensics (from the NBA 2K27 build) corroborate and quantify this [7][8]:

- `nvngx_dlssnr.dll`, version **310.8.0.0**, **158 MB** — roughly 3× a DLSS 4 SR DLL (20–50 MB) and 2× the DLSS 4.5 Ray Reconstruction DLL (72 MB).
- TechSpot's DLL analysis attributes the bulk to **~148 million FP8 neural weights** embedded in the library — the model runs fully locally, nothing streamed [8].
- The neural network runs in **FP8 precision**, the inference format optimized for Blackwell's 5th-gen tensor cores. The DLL's CUDA binaries (cubins) are compiled **only for sm_120 (Blackwell)** — no sm_89 (Ada), no Ampere [9].
- "DLSS NR" profile entries appeared in GeForce driver releases (via Profile Inspector hidden profiles) earlier in 2026, before any public runtime existed [7][8].

NVIDIA claims a full 4K frame (8.3M pixels) can be processed in **under 16 ms** (60+ FPS) on supported hardware [7].

## Inputs

### Official contract

NVIDIA's official description is minimal: color + motion vectors per frame [1]. The SIGGRAPH detail adds that the model "stays anchored by feeding on **engine buffers (albedo, surface normals, lighting data)** alongside the rendered frame" — the mechanism behind "3D-guided AI" and artistic-intent preservation [5].

### The NGX contract as observed in the wild

The community implementation reads the standard DLSS NGX contract. From the DLSS5-Feeder source and renoDX add-on logs [10][11]:

- **Color** — the rendered backbuffer (shared/typed; the NR add-on samples it in a shader).
- **Depth** — R32F, with ReShade's orientation fixes.
- **Motion vectors** — RG16F in pixels; from the game's own DLSS pipeline when present, or from a ReShade optical-flow provider (LumeniteFX Kernel, iMMERSE Launchpad, VORT, qUINT/DRME convention) when not.
- **Output** — a resource the model writes through a typed UAV; must be a typed format (R16G16B16A16_FLOAT). Typeless resources are rejected: "DLSS output format 9 is not a supported typed codec format (requires shader sampling and typed UAV support)" [12].
- **Jitter / reset / sizes** — standard DLSS evaluate params. The community path runs the host evaluate in **DLAA mode** (render size = output size, no jitter) because NR is a fidelity pass, not a scale [10][11].
- **Bias-current-colour mask** — an optional R8 trust mask: pixels flagged "don't trust history" force the model to favor the current frame (used to mask out bad optical-flow vectors around flickering lights/transparents) [10].
- **Create flags**: the Dying Light: The Beast integration observed `AutoExposure` in feature flags (0x4B), meaning **no exposure texture** is carried in the contract; exposure scalars injected via `DLSS.Exposure.Scale` / `DLSS.Pre.Exposure` are ignored by the NR codec [12].
- **Preset hint** in the contract: `0` default, `5`/`6` = legacy CNN presets E/F (harder history clamping), `10`/`11` = transformer presets J/K [10].

## Outputs

- The model writes an **enhanced color buffer** at the host evaluate's resolution (DLAA: full render resolution), which is then written back over the frame [10][11].
- The output resource must be **typed and UAV-accessible**; the renoDX stack upgrades typeless allocations at creation time (see below) [12].
- The NR path behaves like a **codec**: it "folds HDR to an SDR proxy with a per-channel soft-clip against a *static* paper white," which is why bright scenes lose chroma before the model sees them and why a live `NRPaperWhiteScale` control exists. State is built once at feature creation — live resolution/HDR changes require a restart [12].
- One observed quirk hints at an internal **"guide buffer"** at a fraction of output resolution (odd dimensions seen in a supersampled test), but its exact role is undocumented — treat as an observation, not a spec [12].

## Render path interaction

### Official path (Streamline)

Games integrate DLSS 5 through the **Streamline framework** (same as DLSS/Reflex today) [1]. The pass inserts at the correct pipeline stage — *before* final output, on the pre-upscale buffer — which keeps the pixel count the model processes low, and supports **selective per-object application**: developers can apply NR to specific characters/scene elements with independent intensity per object, or ship with it disabled on characters entirely [5][8]. Per-object masks can be auto-generated (model-driven) or authored in-engine; NVIDIA's demo broke out bottles, grapes and a pitcher as separately controlled objects [5].

### NGX mechanics (as reverse-engineered by the mod community)

- DLSS 5 NR is an **NGX feature (feature ID 18)**, created and evaluated through the standard `NVSDK_NGX_D3D12_CreateFeature` / `EvaluateFeature` path; the add-on logs report `feature 18 created ... inline feature 18 evaluation succeeded` [10][11][12].
- The `renodx-dlss5` add-on **detours those NGX entry points**, reads the DLSS "contract" (Color, Depth, MotionVectors, Output, sizes, jitter, reset), and **inserts its neural pass inside the DLSS evaluate** — it cannot tell whether the contract is genuine (game's own DLSS) or synthetic (fed by another tool) [10][11].
- The leaked community implementation runs the NR pass **after upscaling and after tone mapping** — i.e., on every pixel of the full 4K output (8.3M px). That's the wrong pipeline position (VideoCardz documented this) and inflates cost; a proper SDK integration runs pre-upscale and per-object [8].
- Requires **driver 616.56+** and an RTX 50 GPU for Neural Uplift; `nvngx_dlssnr.dll` sits alongside the existing `nvngx_dlss.dll` (SR), `nvngx_dlssg.dll` (FG), `nvngx_dlssd.dll` (RR) [13].

## Performance (leaked build, first consumer tests)

Dan Mateescu (Compusemble), testing the renoDX injection on character models alone [8]:

| GPU | Resolution | Without NR | With NR | Drop |
|---|---|---|---|---|
| RTX 5070 Ti | 4K | 71 FPS | 35 FPS | ~51% |
| RTX 5090 | 4K | 91 FPS | 50 FPS | ~45% |

Context that matters for reading these numbers:

- This is the **most expensive possible configuration**: full-scene (every pixel of the 4K output), post-upscale/post-tonemap placement, pre-release unoptimized DLL, settings undisclosed [8].
- Cost scales with **output pixel count** — lower resolutions carry smaller impact; 4K is the worst case [8].
- NVIDIA's own GTC demo needed two RTX 5090s (one dedicated to inference); the model was distilled to run on a single GPU for SIGGRAPH. The 45–51% figures are a ceiling estimate for the leak, not a floor for shipping games [8].
- Tom's Hardware's take: DLSS 5 will still demand serious hardware, but per-object selective application should dramatically cut the real-world cost [8].
- VRAM budget and cross-tier scaling (5060/5070/5080) are **undisclosed**; so is whether all three models run across the whole RTX 50 stack [5][8].

## Hardware requirements

- **Official: RTX 50-series (Blackwell) only.** Reason given: FP8 inference on 5th-gen tensor cores; the leaked DLL contains sm_120 cubins exclusively [7][8][9].
- **RTX 40 (Ada): runs unofficially.** Modder "Uncle Burrito" (RTX Remix community) identified the sm_120-only cubins, found Ada's tensor cores have *native FP8* support, and patched/swapped the cubins so the DLL's Ada code path is reachable — the *model itself was not retrained*. Result: Control running on an RTX 4090 at ~82 FPS [9].
- **RTX 30 (Ampere): effectively no.** Ampere lacks the same native FP8 support; OC3D reported "abysmal performance" attempts [9][14].
- Independent tracker dlss5.net: 5090/5080/5070 Ti/5070 "confirmed", 5060 Ti/5060 "expected" (per-model docs pending), RTX 40 "unknown", RTX 20/30 "unlikely" [15].
- The Blackwell-only lock follows the DLSS 3 (FG→RTX 40) and DLSS 4 (MFG→RTX 50) pattern — and, like DLSS 4's transformer SR, could plausibly be walked back to older hardware later [3][9].

## How renoDX works (source-level)

[renoDX](https://github.com/clshortfuse/renodx) ("Renovation Engine for DirectX Games", by Carlos Lopez / clshortfuse) is a toolset that can replace shaders, inject buffers, add overlays, upgrade swapchains, upgrade texture resources, and write user settings to disk. It is built on **ReShade's add-on system**, which "simplifies all the hooks necessary to tap into DirectX without worrying about patching version-specific exe files" [16].

### The NGX hook layer (`src/utils/dlss/nvngx.hpp`)

renoDX detours the full NGX API surface by name, then **unwraps ReShade's proxy devices/command lists to the native ones** before forwarding [16]:

- `NVSDK_NGX_D3D11_Init` / `NVSDK_NGX_D3D12_Init` (+ `_with_ProjectID` variants) — unwraps the proxy `ID3D11Device`/`ID3D12Device`.
- `NVSDK_NGX_D3D11_Shutdown1` / `NVSDK_NGX_D3D12_Shutdown1`.
- `NVSDK_NGX_D3D11_CreateFeature` / `NVSDK_NGX_D3D12_CreateFeature` — unwraps the command list (logs `Using native command list ptr =&gt; ptr`).
- `NVSDK_NGX_D3D11_EvaluateFeature` / `NVSDK_NGX_D3D12_EvaluateFeature` (+ `_C` callback variants).

The same directory carries `DXGIFactoryWrapper.hpp` / `DXGISwapChainWrapper.hpp` (swapchain interception/upgrade) and `streamline_v1.hpp` / `streamline_v2.hpp` (the Streamline SDK headers — renoDX can talk to Streamline-integrating games too) [16].

### Resource upgrading (`utils::resource::upgrade`)

renoDX can rewrite resource properties **at creation time**. The Dying Light: The Beast fix is the cleanest example: Techland allocates the DLSS output as `R16G16B16A16_TYPELESS` (DXGI format 9), which the NR add-on refuses; the fix add-on calls `utils::resource::upgrade::SetUpgradeInfos()` to re-create that resource as typed `R16G16B16A16_FLOAT` (format 10) — same bit layout, only the view typing changes, with strict selection filters (format, aspect ratio vs backbuffer, min size, `usage_include = unordered_access`) to avoid touching anything else. Two hard-won caveats: don't use `mods::swapchain` for this (it force-rewrites the backbuffer to linear float and silently kills SDR color), and a blanket typeless upgrade crashes engines that create `R16G16B16A16_SNORM` views over small typeless LUTs [12].

### The DLSS 5 add-on (`renodx-dlss5.addon64`)

The actual DLSS 5 Neural Rendering add-on is **not in the public GitHub repo** — it's distributed through the RenoDX Discord (`#DLSS5` channel), authored anonymously, and pinned at **v4.55** for compatibility with the feeder tools (newer builds build part of the synthetic contract themselves and conflict; v45+ builds rescans every present and adopt features lazily). It works by detouring `NVSDK_NGX_D3D12_CreateFeature`/`EvaluateFeature` and inserting the neural pass into the DLSS evaluate [10][11].

### DLSS5-Feeder (games with no DLSS at all)

The remaining piece of the puzzle: DLSS 5's add-on only fires when a game makes DLSS calls — a DLSS-less game never does. [DLSS5-Feeder](https://github.com/jlrouzies-fr/DLSS5-Feeder) **makes the calls itself**: it builds a complete synthetic DLAA "contract" from what ReShade already has (frame color, depth, estimated optical-flow motion vectors, plus a validated trust mask), runs a genuine `NGX_D3D12_EVALUATE_DLSS` in DLAA mode on a private same-adapter D3D12 device, lets the DLSS 5 add-on hook that evaluate, and copies the neural result back into the frame [10]:

```
game frame → ReShade effects → [MV provider] → DLSS5_Feed → synthetic DLAA evaluate
                                                               ↓ (NR add-on hooks here)
                                          neural output written back over the frame → present
```

Verified working across D3D11, D3D12, Vulkan (via D3D12 shared-memory interop + timeline semaphores), OpenGL (single-context in-order stream, no locks needed), D3D9 (through dgVoodoo2's D3D9→D3D11 translation), and 32-bit games (via a 64-bit helper host process that shares D3D11 textures cross-process and runs its own ReShade+NGX). The NR add-on cannot tell the contract is synthetic — it reports normal `feature 18 created` / `inline feature 18 evaluation succeeded` logs [10].

## The leak timeline (Aug 2026)

- **Aug 26** — `nvngx_dlssnr.dll` v310.8.0.0 discovered in the NBA 2K27 early-access PC build by Renan Maniero; the game itself has no NR toggle (2K was evidently testing DLSS 5) [7][14].
- **Aug 27** — RenoDX community loads it into **Control** within hours, applied to Jesse Faden's character model; exposed **7 style sliders** (tone, local structure, skin structure, overall NR intensity). Visually: sharper subsurface scattering on skin, light transmission through hair, more pronounced contact shadows — no art-direction deviation. HDR broken (SDR only); NR ran post-upscale/post-tonemap [8]. Control Nexus mod: requires ReShade with add-on support, SSAO on, DLSS on (any mode), recent `nvngx_dlss.dll` + `nvngx_dlssd.dll` (RR) + `nvngx_dlssnr.dll` (NR), RTX 50 + driver 616.56+ [13].
- **~Aug 27–30** — Ports spread: **A Hat in Time** (D3D9 — notable, D3D9 isn't a target API), **Deus Ex: Mankind Divided** (D3D12, motion vectors from DRME; ~22 FPS at 4K DLAA+NR on a 5070 Ti — engine is a 2016 DX11-era title running through D3D12), **Final Fantasy VII Rebirth** (one-click installer repos), **Dying Light: The Beast** (needed the typeless→typed output fix) [8][12][17].
- **Aug 28** — First consumer perf numbers published (5070 Ti 71→35 FPS at 4K) [8].
- **Aug 29–30** — RTX 40 cubin-patch mod (Uncle Burrito) runs the same model on RTX 4090/4080 (~82 FPS in Control); RTX 30 attempts reported as abysmal [9][14].
- NVIDIA has not commented on any of it [8][9].

## Games & publishers

15 titles announced at GTC: AION 2, Assassin's Creed Shadows, Black State, Cinder City, Delta Force, Hogwarts Legacy, Justice, NARAKA: BLADEPOINT, NTE: Neverness to Everness, Phantom Blade Zero, Resident Evil Requiem, Sea of Remnants, Starfield, The Elder Scrolls IV: Oblivion Remastered, Where Winds Meet. Four were demoed live at SIGGRAPH: Starfield, AC Shadows, Hogwarts Legacy, Oblivion Remastered [1][3]. Publishers include Bethesda, CAPCOM, Hotta Studio, NetEase, NCSOFT, S-GAME, Tencent, Ubisoft, Warner Bros. Games [1].

## Reception & open questions

- March GTC drew criticism that NVIDIA described the feature in marketing terms and that the model *generates* detail rather than enhancing it; SIGGRAPH's 3D-guided anchoring, per-object controls and single-GPU confirmation partially defused this. "DLSS 5 controversy" still carries steady search volume; TechRadar: "Nvidia tries again to get gamers to accept DLSS 5 and doesn't entirely succeed" [3].
- **Reproducibility concern**: because DLSS 5 is a generative model applied on top of rendered frames, a future driver/model update could change how a game looks without the game changing — two players on different drivers could see visually different games; no preservation position yet [8].
- VRAM budget, cross-tier model availability, and final shipping performance under proper integration remain undisclosed [5][8].

## Sources

1. NVIDIA Newsroom press release, Mar 16, 2026 — [nvidianews.nvidia.com](https://nvidianews.nvidia.com/news/nvidia-dlss-5-delivers-ai-powered-breakthrough-in-visual-fidelity-for-games)
2. NVIDIA GeForce news, "DLSS 5 Delivers AI-Powered Breakthrough in Visual Fidelity for Games" — [nvidia.com](https://www.nvidia.com/en-us/geforce/news/dlss5-breakthrough-in-visual-fidelity-for-games/)
3. Tech Insider, "Nvidia DLSS 5 Ships This Fall, Skips RTX 40 Cards" (SIGGRAPH 2026 coverage) — [tech-insider.org](https://tech-insider.org/nvidia-dlss-5-siggraph-2026/)
4. PCGamesHardware, "DLSS 5: Three AI models, single-GPU operation and full control" — [pcgameshardware.de](https://www.pcgameshardware.de/Deep-Learning-Super-Sampling-Software-277618/Specials/dlss-5-three-ai-models-single-gpu-control-siggraph-2026-1548907/)
5. HWBusters, "DLSS 5 Gets a Dimmer Switch — NVIDIA Finally Shows the Controls It Promised" — [hwbusters.com](https://hwbusters.com/news/dlss-5-gets-a-dimmer-switch-nvidia-finally-shows-the-controls-it-promised/)
6. 8Bitz (corroborating one-step pixel-space diffusion transformer) — [8bitz.gr](https://8bitz.gr/10915/news/gaming/nvidia-dlss-5-pos-leitoyrgei-ti-allazei-sta-games/)
7. Gaming Ideology, "NVIDIA DLSS 5 Leak: Neural Rendering DLL Hints at Launch" — [gamingideology.com](https://www.gamingideology.com/nvidia-dlss-5-leak-neural-rendering-dll-hints-at-launch/)
8. TechTimes, "DLSS 5 Bleeds Frames in First Consumer Test: 71 to 35 FPS on RTX 5070 Ti" — [techtimes.com](https://www.techtimes.com/articles/325892/20260828/dlss-5-bleeds-frames-first-consumer-test-71-35-fps-rtx-5070-ti.htm)
9. Shattered.io, "DLSS 5 Modded for RTX 4090: Ada Lovelace Mod" (cubin/FP8 analysis) — [shattered.io](https://shattered.io/dlss-5-rtx-4090-ada-lovelace-mod-2026/)
10. DLSS5-Feeder (jlrouzies-fr) — [github.com/jlrouzies-fr/DLSS5-Feeder](https://github.com/jlrouzies-fr/DLSS5-Feeder)
11. DLSS5-Feeder source (`src/dlss5-feed.cpp`) — NGX create/evaluate surface, DLAA contract
12. markitzeroo/dltb-dlss5-fix — [github.com/markitzeroo/dltb-dlss5-fix](https://github.com/markitzeroo/dltb-dlss5-fix) (typeless→typed output upgrade, codec behavior)
13. Nexus Mods, "Control — RR and DLSS 5 (RenoDX)" — [nexusmods.com/control/mods/140](https://www.nexusmods.com/control/mods/140)
14. OC3D, "Nvidia DLSS 5 Neural Rendering DLL files leak" — [overclock3d.net](https://overclock3d.net/news/software/nvidia-dlss-5-neural-rendering-dll-files-leak-launch-incoming/)
15. DLSS 5 Supported Cards & GPU Compatibility Checker — [dlss5.net](https://www.dlss5.net/)
16. renoDX (clshortfuse) — [github.com/clshortfuse/renodx](https://github.com/clshortfuse/renodx), `src/utils/dlss/nvngx.hpp`
17. vgtimes, "NVIDIA DLSS 5 leaked in NBA 2K27 and was already ported into Control" — [vgtimes.com](https://vgtimes.com/tech-and-hardware/165688-nvidia-dlss-5-leaked-in-nba-2k27-and-was-already-ported-into-control.html)
