# Benchmark plan — Hygon DCU (gfx928 / ROCm)

Status: **benchmark pending**. The port is verified as functionally working
(see README, logs from 2026-10-07), but no timing has been captured on this
hardware. Nothing in this repository should be read as a speed-up claim
until the runs below exist.

## What to measure

| # | Measurement | Tool / path | Output |
|---|---|---|---|
| 1 | Single sparse layer, dense vs Veda | `tools/bench_attention.py` | ms/layer at 38k / 74k / 104k tokens |
| 2 | Sparsity sweep | same | ms/layer at 80% / 90% / 95% |
| 3 | H3 end-to-end, dense vs Veda | `tools/e2e_minimax_h3.py` | s/step, s/prompt, peak VRAM |
| 4 | Phase split inside Veda | node `verbose` | kernel / gather / score / scatter / select |

For (1) and (2), use the random scorer so the predictor's cost is constant
and the kernel is isolated. For (3), use the real predictor and real H3
weights. Do not mix the two.

Two dense baselines are needed for (3): ComfyUI's default attention, and
`--use-sage-attention` if it runs on this box. Report Veda against both.

## Fixed conditions

* Model/workflow: MiniMax-H3 T2VA, 1344x768, 124 frames (5.2 s), 8-step
  Turbo LoRA.
* Same prompt, same seed, same step count in every arm of a comparison.
  Seed 42 for T2VA unless a run is invalidated, in which case change it for
  *all* arms.
* Same ComfyUI commit, same torch/Triton/DTK versions, same node version.
  Record them verbatim with the results.
* One card only. This container sees a single DCU; do not aggregate clocks
  across cards. Record the SM clock during the run (median, not max).

## Measurement protocol

1. **Read server-side intervals.** Take `execution_start` to
   `execution_success` from ComfyUI's `/history`, not client wall time and
   not the browser. Exclude model load unless it is reported separately.
2. **Restart ComfyUI between configurations.** Upstream observed that
   residual allocation from a previous arm distorts both speed and peak
   VRAM (1.61 GB vs 3.24 GB in their notes). One process per arm.
3. **Discard a warm-up run.** The first sparse call pays Triton
   compilation/autotune. Time only steady-state steps.
4. **Do not let the node cache fool you.** Submitting the same prompt and
   seed twice can return a cached result in ~0.01 s. Alternate prompts or
   seeds, or clear the queue cache, between timed runs.
5. **Repeat 3 times, report the median** and the min/max. If the spread
   exceeds 5%, find the cause before quoting a ratio.
6. **Report paired ratios per run**, not a ratio of two medians, when the
   arms were not measured back to back.
7. **Peak VRAM** from `torch.cuda.max_memory_allocated()` (or ComfyUI's
   own report), reset before each arm.

## Sparsity sweep

Run 80%, 90%, 95% plus the dense baseline. 90% is the trained value; 80%
and 95% test the trend, not quality. State clearly that 80/95% are outside
what the released predictor was trained for and that image quality is not
being judged here.

## Deliverables

Add `docs/benchmark-results.md` containing:

* a table per measurement, with median and range;
* the exact commands and the fixed-condition block;
* the four startup/attention log lines for each arm;
* the predictor file's sha256 (must match
  `2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd76573a`);
* versions: `triton`, `torch`, DTK, ComfyUI, node commit;
* raw logs under `docs/benchmark-logs/` (or linked externally).

Only then replace "benchmark pending" in the README with numbers, and only
for the hardware and configuration actually measured.

## What is missing right now

* A free GPU window (the card is currently used by a Wan A/B test).
* The predictor file in `models/veda/` (downloadable, see README).
* The H3 weights on the same machine, for the end-to-end arm.
* The runs themselves — the plan above is not a result.
