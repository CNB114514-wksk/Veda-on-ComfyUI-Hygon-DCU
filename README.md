<div align="center">

<img src="https://raw.githubusercontent.com/veda-sparse/Veda-on-ComfyUI/main/assets/icon.svg" width="120" alt="Veda logo">

# Veda Sparse Attention for ComfyUI (MiniMax-H3)

</div>

[Paper](https://arxiv.org/abs/2605.30325) · [Project Page](https://veda-sparse.github.io/) · [Predictor](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview) · [Training Code](https://github.com/veda-sparse/Miowtion) · [中文说明](README.zh-CN.md)

## Introduction

**Veda** is a learned sparse-attention method for video diffusion models.
Attention dominates the cost of a long clip, and a small distilled
predictor can say in advance which tiles of the attention map carry the
result. Veda computes roughly the top 10% of them and skips the rest.

This repository packages Veda as a custom node for ComfyUI's native
MiniMax-H3 (T2VA, FL2VA and R2VA): one node, MODEL in and MODEL out.

### Highlights

- **Faster inference.** 2.9x end to end and 7.1x on attention alone
  against ComfyUI's default attention, measured on an RTX 5070. The gain
  grows with clip length.
- **Plug-and-play.** An attention override rather than a weight patch, so
  Turbo and style LoRAs, fine-tuned and quantized H3 checkpoints,
  first/last-frame and reference conditioning, and other block patches
  keep working.
- **Safe by default.** Every kernel passes a self-test on the GPU before
  it is used. Anything Veda cannot handle runs the model's own attention
  and says so on the node, rather than producing a broken render.
- **Nothing extra to install.** The sparse kernel ships with the node;
  installing from the Comfy Registry pulls in Triton (NVIDIA, SM80 and
  newer) or MLX (Apple silicon).

## Hygon DCU (ROCm / gfx928) port

**Status: verified working (2026-10-07) · benchmark pending.**

Upstream 0.3.0 ships kernels for NVIDIA SM80+ (Triton INT8) and Apple
silicon (MLX) only. For ROCm, `docs/hardware.md:16` says there is no
backend, so the node turns itself off and the model runs its own full
attention. This fork — `CNB114514-wksk/Veda-on-ComfyUI-Hygon-DCU`,
branch `hygon-dcu` — adds that missing backend for Hygon DCU (gfx928)
under ROCm.

### The two changes

Everything else is upstream 0.3.0. The port is 21 added and 3 removed
lines, exported verbatim as a patch from a machine where it was tested.

1. **Backend whitelist** — `veda_comfy/backends/__init__.py`. Upstream
   requires `info.kind == 'cuda' and info.family != 'rocm'`, which hides
   `triton-int8` on HIP. Dropping the `family != 'rocm'` test lets the
   INT8 path be selected. It is safe to expose because the backend still
   runs its per-device self-test against a float reference before any
   layer uses it; a wrong result falls back to full attention.

2. **Pipeline stages** — `veda_comfy/kernels/sage/sparse_int8.py`.
   Upstream's `num_stages = 3` was tuned on SM120. On gfx9 MMAC it is
   much worse: the author's comment records **1294 ms vs 274 ms at 119k
   slots (4.7x)**. A new `ROCM_STAGES = 2` plus `stages_for_device()`
   returns 2 when `torch.version.hip` is set and keeps 3 elsewhere.

The INT8 `tl.dot` reaches Hygon matrix cores because DTK's Triton lowers
it to `v_mmac_*`; no kernel rewrite is involved.

### Evidence

Copied from the ComfyUI console of two complete H3 runs on 2026-10-07
(14:42 and 15:27), on a Hygon DCU gfx928 under ROCm:

```
Veda: Veda ready | Triton INT8 (ROCM) | K500SM_AI
Veda: Veda running | Triton INT8 (ROCM)
Veda: Veda done | Triton INT8 (ROCM)
Attention computed: 12.8% of full attention (87.2% skipped)
```

`K500SM_AI` is the device name the runtime reports. The last line means
the sparse kernel computed about one eighth of the dense attention map.

Before the port the same install reported, and silently used full
attention:

```
Veda: Veda off: no sparse kernel works on K500SM_AI (ROCM); using full attention
```

These runs establish correctness and that the kernel is actually used.
They are **not** a performance measurement: no timing was captured, so
this repository claims no speed-up. See
[docs/benchmark-plan.md](docs/benchmark-plan.md) for what is still
missing.

### Dependencies

* **DTK Triton.** The verified build is
  `triton 3.1.0+das.opt1.dtk2604.torch271` (the `dtk2604` DTK build), with
  `torch 2.7.1+hip6.3.26093`. A stock PyPI Triton is not known to lower
  int8 `tl.dot` to `v_mmac_*` on this target; use the DTK build.
* ComfyUI >= 0.38.0 and MiniMax-H3 weights, as upstream.
* The predictor file (see below). Nothing else.

### Predictor weights

`minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors` — 275 MB
(275,415,648 bytes), sha256
`2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd76573a`,
from `Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview` on Hugging
Face, revision `9a1fd3a4`. Place it in `ComfyUI/models/veda/`.

The weights are under the **MiniMax H3 Community License**, inherited
from MiniMax-H3 — **not MIT**. They are not redistributed in this
repository.

### 中文摘要

本仓库补上 Veda 上游明确没有的一格：Hygon DCU（gfx928 / ROCm）。
`main` 是上游 0.3.0 原样，仅加 `NOTICE`；`hygon-dcu` 在其上只改两处：
放开 `family != 'rocm'` 的白名单、并在 HIP 下把 `num_stages` 从 3 降到 2
（作者注释：119k slots 时 1294 ms vs 274 ms）。2026-10-07 两次完整运行
的日志显示 `Triton INT8 (ROCM)` 生效、跳过 87.2% 的注意力，但**尚未测速**，
因此不给出任何倍数。依赖 DTK 定制 Triton（`3.1.0+das.opt1.dtk2604`）；
打分器权重是 MiniMax H3 Community License，非 MIT，需自行下载。

## Installation

Requires ComfyUI >= 0.38.0. In ComfyUI Manager, search "Veda". Or:

```bash
comfy node install veda-sparse-attention
```

Or clone this repository into `ComfyUI/custom_nodes/` and restart ComfyUI.

## Usage

Open *Workflow -> Browse Templates -> Veda-on-ComfyUI* and pick "Veda
MiniMax H3 T2VA" or "Veda MiniMax H3 R2VA". The missing-model dialog
offers the predictor. Write a prompt and run.

To add Veda to an existing H3 workflow, put **Veda Sparse Attention
(MiniMax H3)** on the MODEL wire after the model and any LoRA loaders,
last before the guider or sampler. Select it and press **Ctrl+B** to
bypass it and render the same seed with full attention.

Do not combine it with ComfyUI's own "Model Sparse Attention" node on H3:
that one replaces the attention blocks outright, so Veda would never be
called. The node says so when it sees both.

Other attention nodes can stay. Veda runs on top of "Patch Sage Attention
KJ" (the calls it does not take go to sage), and KJNodes' "MiniMax H3 Low
VRAM Attention" keeps working. A node that swaps out the whole H3
attention, such as "MiniMax H3 Mem Eff Sage Attention Patch", must come
*before* Veda: Veda then runs the sparse layers and leaves that node the
full-attention ones, and says so on its node. Placed after Veda it would
hide every call from Veda; the node warns about that too.

### Predictor

The node does not download anything itself. The predictor (275 MB)
reaches `models/veda` one of two ways:

* **From a template.** Open one of the Veda templates and ComfyUI offers
  the predictor in its missing-models dialog; one click fetches it.
* **By hand.** Download
  `minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors` from the
  [predictor repository](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview)
  into `ComfyUI/models/veda/` and restart ComfyUI.

Any `.safetensors` in that folder appears in the `predictor` list, so a
predictor trained elsewhere is selected the same way. If the selected
file is not there, the node says so and prints the URL rather than
reaching out on its own.

### Node text

```
Veda done · Triton INT8 (SM120)
Video: 1344x768 · 5.2 s
Attention computed: 10.9% of full attention (89.1% skipped)
```

Before sampling the node names the kernel it will use and the sparsity;
while sampling, the video size and the trained tile plan it matched; at
the end, how much of full attention was actually computed. A line
starting `Veda off` gives the reason Veda is not running, and `nearest
trained size: ...` on the tile plan line means the video is outside the
trained set. `verbose` adds per-phase timing, call counts and predictor
details.

Warnings and errors come in English and Chinese, one after the other;
the node's name and tooltips follow ComfyUI's interface language.

### Options

Only `model` and `predictor` are visible. The rest are advanced inputs
(click "show advanced inputs") and default to the trained values.

| Input | Default | Description |
|---|---|---|
| `generated_sparsity` | `90%` | Sparsity of the generated video's attention. `90%` skips 90% of the key tiles each query tile could attend, which is the trained value; lower is closer to full attention and slower. A whole number such as `24` keeps exactly that many 128-token key tiles instead. |
| `reference_sparsity` | `90%` | The same for references: first/last frames, guide frames, reference images and videos. `0%` gives them full attention. |
| `full_attention_layers` | empty | 0-based DiT blocks that keep full attention, e.g. `0, 1, 47-49`. |
| `full_attention_steps` | empty | 0-based sampling steps that keep full attention, e.g. `0`. |
| `verbose` | off | Also report attention time per phase, call counts and predictor details on the node after each run. |

The released predictor was trained for 1344x768, 768x1344, 768x768 and
1024x768 at 5 / 10 / 14 s with the 8-step Turbo LoRA. Other sizes fall
back to the tile plan of the nearest aspect ratio and duration. Other
sizes, other step counts and R2VA / FL2VA references all work, but are
outside the training data, so compare them against full attention.

## Performance

T2VA at 1344x768, 124 frames (5.2 s), 8-step Turbo LoRA, 90% sparsity,
on an RTX 5070 12 GB under Windows 11. These are upstream's NVIDIA
measurements, kept for reference. No Hygon DCU benchmark has been run
yet — see [docs/benchmark-plan.md](docs/benchmark-plan.md).

| Attention | Per step | 8 steps | Attention per step |
|---|---|---|---|
| ComfyUI default | 40.7 s | 342 s | 31.1 s |
| ComfyUI `--use-sage-attention` | 24.8 s | 231 s | 15.2 s |
| Veda sparse INT8 | **14.0 s** | **130 s** | **4.41 s** |

One attention layer at 104k tokens and 90% sparsity takes 528 ms against
11.8 s for full attention. Veda's INT8 arithmetic is the same code
ComfyUI runs behind `--use-sage-attention`, so the quality is what that
path gives, with sparsity on top. What remains per step is MLP and weight
movement, not attention.

## Hardware

One Triton kernel covers every NVIDIA GPU from SM80 on, Windows and Linux
alike. Older cards and CPU have no kernel, and upstream also excludes
ROCm: the node says so and the model runs its own attention. This fork's
`hygon-dcu` branch adds ROCm support for Hygon DCU; see
[Hygon DCU (ROCm / gfx928) port](#hygon-dcu-rocm--gfx928-port).

| Hardware | Status |
|---|---|
| RTX 30 / A100 / RTX 40 / L40 (sm80-89) | code path ready, not yet verified on this hardware |
| H100 / H200 (sm90) | code path ready, not yet verified on this hardware |
| B200 / B300 (sm100 / sm103) | code path ready, not yet verified on this hardware |
| RTX 50, RTX PRO 6000 Blackwell (sm120) | **verified: RTX 5070, Windows 11** |
| DGX Spark / GB10 (sm121) | code path ready, not yet verified on this hardware |
| Apple silicon (M series) | verified: M3 Pro, macOS 15 |
| Hygon DCU K500 (gfx928, ROCm) | triton-int8 | — | **verified: 2026-10-07** (this fork) |

Per-architecture notes and the full measurement log are in
[docs/hardware.md](https://github.com/veda-sparse/Veda-on-ComfyUI/blob/main/docs/hardware.md).

## License

Code: MIT. The INT8 kernel's arithmetic is derived from SageAttention v1
(BSD-3-Clause). The predictor inherits the MiniMax H3 Community License
from its base model. See
[NOTICE.md](https://github.com/veda-sparse/Veda-on-ComfyUI/blob/main/NOTICE.md).

## Citation

```bibtex
@inproceedings{han2026veda,
  title={Veda: Scalable Video Diffusion via Distilled Sparse Attention},
  author={Han, Shihao and Yang, Hao and Hu, Xinting and Mei, Xiaofeng
          and Jiang, Yi and Qi, Xiaojuan},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
