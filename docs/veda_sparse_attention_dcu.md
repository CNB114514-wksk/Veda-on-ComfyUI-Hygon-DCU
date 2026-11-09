# Veda 稀疏注意力在 Hygon DCU（gfx928）上的适配

> 来源：2026-11-09 本机实测（Hygon K100-AI / gfx928，DTK 26.04，ComfyUI 0.38.2）。
> 上游项目：[Veda-on-ComfyUI](https://github.com/CNB114514-wksk/Veda-on-ComfyUI-Hygon-DCU)
> （分支 `hygon-dcu`，本文两处补丁与之逐字节一致）。
>
> 本文只讲**怎么让 Veda 在 DCU 上跑起来**。全部性能与画质数字集中在
> [benchmark-h3-04mp.md](benchmark-h3-04mp.md)，本文不重复。

插件 `custom_nodes/Veda-on-ComfyUI`（上游 veda-sparse-attention 0.3.0，upstream `60bfae6`）默认只覆盖
CUDA。在 ROCm/HIP 的 gfx9 上要跑起来需要两处改动，均为纯 Python，不改内核源码。

## 1. 补丁一：放开后端白名单对 ROCm 的排除

`veda_comfy/backends/__init__.py`，`candidates()`：

```diff
-    if info.kind == 'cuda' and info.family != 'rocm' and info.cc:
+    # Allow ROCm/Hygon DCU (gfx928) too: this Triton fork lowers int8
+    # tl.dot to the Hygon v_mmac_* matrix instruction, which works on
+    # gfx928 (verified). Upstream excluded ROCm only because it was never
+    # tuned there; the self-test below still guards correctness.
+    if info.kind == 'cuda' and info.cc:
         return ['triton-int8'] if info.cc >= _MIN_CC else []
```

上游排除 ROCm 只是没在那边调过，不是不能跑；`triton-int8` 的自检仍在，失败会自己退回。

## 2. 补丁二：HIP 下 `num_stages=2`

`veda_comfy/kernels/sage/sparse_int8.py`：仓库默认 `STAGES = 3`，那是在 SM120 上调的；
在 gfx9 MMAC 上 3 比 2 慢 4.7×（119k slots：1294 vs 274 ms）。新增按设备选择：

```diff
 WARPS, STAGES = 4, 3
+ROCM_STAGES = 2
+def stages_for_device() -> int:
+    return ROCM_STAGES if getattr(torch.version, 'hip', None) else STAGES
 ...
-    key_block, warps, stages = KEY_BLOCK, WARPS, STAGES
+    key_block, warps, stages = KEY_BLOCK, WARPS, stages_for_device()
```

补丁后日志会自报后端，并给出实际计算/跳过的注意力比例（下为 2026-11-09 一次 B 组运行原文）：

```
[INFO] Veda: Veda ready | Triton INT8 (ROCM) | K100_AI
Sparsity: generated 90% | reference 90%
[WARNING] Veda: Veda running | Triton INT8 (ROCM)
[INFO] Veda: Veda done | Triton INT8 (ROCM)
Attention computed: 22.7% of full attention (77.3% skipped)
```

这三行 `ready / running / done` 是判断「内核真的被选中且执行过」的依据；缺任一行就说明退回了全量
attention（日志会出现 `Veda off: no sparse kernel works ...`）。

## 3. 实测结论

0.4 MP（864×480 / 124 帧 / 20 步）档位上 Veda 90% 确实生效，但**收益很小**：端到端个位数百分比，
且瓶颈不在 attention。完整设置、结果表、冷启动成本、画质对照，以及两条必须一起读的限制
（只跳过 77.3% 注意力却只换来约 2% 提速；tile plan 按训练尺寸 1344×768 生成而视频只有 864×480），
见 **[benchmark-h3-04mp.md](benchmark-h3-04mp.md)**。

一句话：这一档不要默认开 Veda，也不要把它写成「Veda 在 DCU 上快 N 倍」；要验证更大的倍数需要
靠近 predictor 训练分布的尺寸/时长。

## 4. 复现要点

1. predictor `minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors`（275,415,648 B）放到
   `models/veda/`，大小与 sha256 要和上游 `veda_comfy/predictors.py` 的登记值一致；
2. 启动加 `--enable-triton-backend`（Triton INT8(ROCM) 路径要求 triton 3.1.0 后端可用）；
3. 组间必须完全停服重起，否则 B 组的 Triton 编译与模型缓存会污染 A 组计时；
4. A/B 用同 seed 配对，且至少跑两次：第一次含 Triton 编译，只能读第二次的稳态；
5. 判断 Veda 是否执行看日志的 `running` / `done` / `Attention computed`，不要只看
   `execution_cached` —— 节点可能命中缓存而 hooks 仍生效。
