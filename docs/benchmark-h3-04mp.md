# H3 0.4 MP end-to-end — dense vs Veda 90% on Hygon DCU

Status: **measured, 2026-11-09.** One A/B pair of arms, one card, two runs
each. This is the end-to-end measurement that
[benchmark-plan.md](benchmark-plan.md) asked for; the single-layer and
sparsity-sweep rows of that plan were **not** run.

Everything below is transcribed from the run artifacts in
`h3_veda_20261108/` on the test machine — `report.md` for the timing table,
`11-09_MiniMax-H3-Veda_技术报告.md` for the analysis, `logs/server_{A,B}.log`
for the device and version lines. Where a number appears in both, they agree.

## 1. Setup

| Item | Value | Source |
|---|---|---|
| GPU | Hygon K100-AI, gfx928, one card (card5) visible to the container | `logs/server_A.log` |
| DTK / driver | 26.04 / 6.2.28 | environment record |
| torch | 2.7.1 (ROCm 6.3) | `logs/server_A.log` |
| Python | 3.12.13 | `logs/server_A.log` |
| ComfyUI | 0.38.2, frontend 1.53.6 | `logs/server_A.log` |
| Triton | 3.1.0, `--enable-triton-backend` | `logs/server_A.log` |
| comfy_kitchen | 0.2.36, HIP `_C.abi3.so` md5 `dc534c6c10cfa9dbdeefaaf6e0475001`, unchanged throughout | md5sum at run time |
| Veda node | `custom_nodes/Veda-on-ComfyUI` 0.3.0 (upstream `60bfae6`) + the two DCU patches, see [veda_sparse_attention_dcu.md](veda_sparse_attention_dcu.md) | `git status` at run time |
| Predictor | `minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors`, 275,415,648 B, sha256 `2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd76573a` | `report.md` §2 |

**Workflow.** `Veda MiniMax H3 T2VA (text to audio-video).json`, default
parameters untouched: ResolutionSelector 0.4 MP / 16:9 / multiple=32 →
**864×480**, **124 frames** at **24 fps** (5.167 s), **20 steps**
`res_multistep` + `simple`, `turbo_mode=false` (the turbo LoRA branch is not
selected, but its node still executes). Source: `report.md` §1.

**Arms.** The only difference is the `VedaSparseAttention` node (id 141):

* **A — dense.** Node 141 removed; `BasicGuider.model` wires straight to
  `ComfySwitchNode`, equivalent to Ctrl+B bypass in the UI.
* **B — Veda 90%.** Node 141 kept with the workflow defaults: predictor
  above, `generated_sparsity` 90%, `reference_sparsity` 90%, empty
  `full_attention_layers` / `full_attention_steps`, `verbose` off.

**Pairing and restart.** Both arms use the same two seeds, submitted in the
same order: run 1 = 757358688076613, run 2 = 757358688076614, so A1/B1 and
A2/B2 are same-seed pairs. No restart inside an arm; a full stop and restart
of ComfyUI between arms (A is one process, B is another). Single card, serial
submission. Source: `report.md` §3.

**Which run counts.** Run 1 of each arm is the cold run and carries Triton
compilation in B; run 2 of each arm is the steady-state measurement. All
steady-state deltas below compare A2 against B2.

**Cache.** The `execution_cached` list of both run-2 submissions does not
contain node 125 (`SamplerCustomAdvanced`), so sampling really re-ran. Node
141 is in B2's cache list, but the Veda hooks still fire — the log shows
`Veda running` / `Veda done` / the 22.7% line. Do not read "cached" as "Veda
did not run". Source: `report.md` §3.

## 2. Results

### 2.1 End-to-end and sampling

| Run | Arm | e2e (`/history`, ms delta) | wall (submit → history) | Sampling (tqdm) | Per step | Peak VRAM |
|---|---|---|---|---|---|---|
| 1 | A dense, cold | 520.28 s | 522.41 s | 198 s (03:18) | 9.93 s/it | 37,499 MiB |
| 2 | A dense, steady | **224.02 s** | 228.88 s | **199 s** (03:19) | **9.97 s/it** | 37,545 MiB |
| 1 | B Veda, cold | 540.44 s | 541.69 s | 195 s (03:15) | 9.75 s/it | 37,518 MiB |
| 2 | B Veda, steady | **220.28 s** | 221.51 s | **195 s** (03:15) | **9.77 s/it** | 37,561 MiB |

Steady state, A2 → B2 (`report.md` §5.1):

| Metric | Dense | Veda 90% | Delta |
|---|---|---|---|
| e2e | 224.02 s | 220.28 s | **−3.74 s (−1.67%)** |
| wall | 228.88 s | 221.51 s | −7.37 s (−3.22%) |
| Sampling, 20 steps | 199 s | 195 s | −4.0 s (−2.0%) |
| Per step | 9.97 s/it | 9.77 s/it | −0.20 s/it (−2.0%) |
| Peak VRAM | 37,545 MiB | 37,561 MiB | +16 MiB |

The VRAM difference is 16 MiB — no difference between the arms, both at the
37.5 GB level. ComfyUI reports 65,520 MB total on this card.

### 2.2 Cold start

| Arm | e2e, cold | vs the other arm's cold run |
|---|---|---|
| A dense | 520.28 s | — |
| B Veda | 540.44 s | **+20.16 s (+3.9%)** |

Veda is *slower* on the first run of a process: Triton compiles its kernels
once (~30 s) and `Veda ready` patches the model (~14 s). This is a one-time
cost per process and is excluded from the steady-state comparison above —
and equally, the cold numbers must not be used to compare the two arms.
Source: `report.md` §5.1.

Startup itself (to `/system_stats` answering) was 329.6 s for A and 405.7 s
for B, with 4,864 and 5,433 nodes loaded. B's slower start is ComfyUI-Manager
network retries (`FETCH DATA ... switching to local mode`), not Veda, and it
does not affect the sampling timings.

### 2.3 What the Veda log says

B, both runs (`report.md` §4, `logs/server_B.log`):

```
Veda: Veda ready | Triton INT8 (ROCM) | K100_AI
Sparsity: generated 90% | reference 90%
Veda: Triton INT8 (ROCM): compiling Triton kernels for this GPU (first run only)   # run 1 only
Veda: Veda running | Triton INT8 (ROCM)
Video: 864x480 | 5.2 s
Tile plan: nearest trained size: 1344x768 | 5.2 s (this video: 864x480 | 5.2 s)
Veda: Veda done | Triton INT8 (ROCM)
Attention computed: 22.7% of full attention (77.3% skipped)
```

Both B runs report 77.3% skipped. A has no Veda line at all. No log of the
four contains `Veda off` or `no sparse kernel works`, so the fallback path
never engaged.

### 2.4 Outputs and quality

All four outputs are 864×480, 24 fps, h264 (yuv420p) + AAC 32 kHz stereo,
124 frames / 5.167 s, so the arms are not trading resolution for speed.
Per-frame PSNR/SSIM over 124 frames, same-seed pairs (`report.md` §5):

| Pair | PSNR | SSIM | Read |
|---|---|---|---|
| seed …613, A1 vs B1 | 34.30 dB (17.18–51.18) | 0.796 (0.582–0.961) | content and layout held; differences in high-frequency detail and motion edges |
| seed …614, A2 vs B2 | 7.62 dB (5.15–10.67) | 0.225 (0.146–0.324) | trajectory diverged — same magnitude as a different-seed comparison (dense vs dense at different seeds: 4.65 dB / 0.244) |

The second pair cannot be read as quality loss, and the first pair is a
sample of one. Neither number is an average quality delta.

## 3. Limitations

**Skipping 77.3% of attention bought 1.67% end to end.** The per-step cost
fell by the same ~2% (9.97 → 9.77 s/it), so the saving is in sampling and
nowhere else. At 864×480×124 the attention map is not what the step spends
its time on: QKV projection, MLP and VAE decode/encode are all full-width,
and computing only 22.7% of the map barely moves the total. Any claim that
Veda is "2–3× faster on DCU" is contradicted by this measurement at this
size — upstream's 2.9× was measured at 1344×768 with the 8-step Turbo LoRA,
which is a different regime.

**The tile plan is built for 1344×768 while the video is 864×480.** The log
line `Tile plan: nearest trained size: 1344x768 ... (this video: 864x480)`
says Veda selected the nearest *trained* size, not the actual size, so the
sparse kernel works on a tile grid derived from 1344×768. The number of
tiles actually processed is therefore higher than 864×480 nominally implies,
which eats into the theoretical saving. This is an inference from the log
line, not a verified mechanism — the source was not read line by line for
it — and it is untested: no run has been made at 1344×768 on this box, so
how much of the small gain it explains is unknown.

Two further caveats from the report, kept because they bound the result:

* Steady state was measured **once per arm**. The 2% gap is close to
  run-to-run jitter (the report puts step-to-step variation at ±0.3 s), so
  the ratio is indicative, not settled; three or more repeats would be
  needed to quote it with confidence.
* Only one of the two seed pairs is comparable (see 2.4). The quality
  statement rests on one pair.

## 4. What this does not establish

* Nothing about 1344×768, longer clips, 8-step Turbo, or FL2VA / R2VA.
  Those are the sizes the predictor was trained for and where upstream's
  larger gains live; none were run here.
* Nothing about the single-layer kernel cost or a sparsity sweep — rows 1,
  2 and 4 of [benchmark-plan.md](benchmark-plan.md) remain open.
* No comparison against `--use-sage-attention`; this box ran
  `--use-flash-attention`, and the dense arm is ComfyUI's default attention.

## Sources

| File | Used for |
|---|---|
| `h3_veda_20261108/report.md` | setup, timing table, log excerpt, PSNR/SSIM, predictor hash |
| `h3_veda_20261108/11-09_MiniMax-H3-Veda_技术报告.md` | analysis, cold-start breakdown, caveats |
| `h3_veda_20261108/logs/server_A.log`, `server_B.log` | device, versions, Veda log lines |
| `h3_veda_20261108/logs/history_{A,B}_*.json` | e2e intervals, `execution_cached` |

Raw artifacts (28 MB, four mp4) are not committed. The numbers above are the
whole of the timing result; the mp4s are only needed to re-check the PSNR.
