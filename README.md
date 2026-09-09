![vllm-exl3 — EXL3 quantization plugin for routed MoE serving](assets/header.png)

# vllm-exl3

[![Follow on X](https://img.shields.io/badge/Follow-%40ViC305-black?logo=x)](https://x.com/ViC305) [![Follow on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Follow-vcruz305-yellow)](https://huggingface.co/vcruz305)

An out-of-tree vLLM plugin that registers `--quantization exl3`, serving
EXL3 (ExLlamaV3 trellis, MCG codebook) quantized packs — routed MoE experts
run packed through `exllamav3_ext` kernels, never dequantized to a dense
format at load.

If you use this plugin, please credit **vcruz305**.

## Credits & provenance

This project is intentionally explicit about upstream work and license lineage.

- The **EXL3 trellis format, MCG codebook, quantization method, and packed execution model** come from [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by Turboderp ([@turboderp](https://github.com/turboderp)).
- `csrc/exl3_fat_gemm.cu` and `.cuh` come from the historical MIT-licensed E2 work in [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks), and substantial portions of the routed-expert integration in `src/vllm_exl3/exl3.py` derive from their earlier `overlay/exl3.py`. The exact source lineage and required notices are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- vLLM integration points derived from or modeled after upstream vLLM are also identified in `THIRD_PARTY_NOTICES.md`.
- New upstream-informed work is labeled in Git history as **copied/derived**, **adapted design**, or **independent implementation**. The policy and source boundaries are documented in [docs/provenance.md](docs/provenance.md).

Current vllm-exl3 releases move forward under **AGPL-3.0-only** so improvements to modified network-served versions remain available to their users. Earlier vllm-exl3 releases were Apache-2.0; that prior license text remains in [LICENSE.APACHE-2.0](LICENSE.APACHE-2.0). Third-party MIT/Apache notices remain fully preserved.

## Scope (read this first)

This is **not** a plugin for stock vLLM. Upstream vLLM declined EXL3 support
([vllm-project/vllm#19896](https://github.com/vllm-project/vllm/issues/19896)),
and this plugin targets vLLM **fork lineages** that provide the
`RoutedExperts` fused-MoE layer family (the NVIDIA DGX Spark GB10 (sm_121
Blackwell) with 128 GiB Unified Memory GLM/DeepSeek serving forks). It also
requires
[exllamav3](https://github.com/turboderp-org/exllamav3) with its compiled
`exllamav3_ext` kernels for your GPU arch.

## v0.4.1 TP1 runtime policy & observability

Version `0.4.1` starts a TP1-focused optimization track without changing the default execution contract before GPU qualification:

- **Effective runtime diagnostics** — `vllm_exl3.runtime_diagnostics()` reports the selected MoE backend, native extension ABI, per-bit native row caps, fused scratch row capacity, fat-expert threshold, fat-kernel availability, and speculative schedule.
- **Per-bit native decode tuning** — `VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K2`, `_K3`, and `_K4` allow K2/K3/K4 to be benchmarked independently rather than forcing one global row cap across different trellis costs. Existing global/measured-cap behavior remains the fallback.
- **Fused scratch policy surface** — `VLLM_EXL3_FUSED_TEMP_ROWS` exposes the requested fused row capacity for experiments while preserving the historical 2048-row default. Lower values must only be promoted after the runner and extension capacity contract are validated on GPU.

These helpers are independently implemented around vllm-exl3's existing interfaces. Recent public MiaAI-Lab work is credited in [docs/provenance.md](docs/provenance.md) as relevant design prior art; no post-relicense adaptive-scheduler or dense-FP8 source is copied into these helpers.

## v0.3.1 Super Fat GEMM Kernel & Ultra-Long Context Release (NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory)

Version `0.3.1` adds accelerated $128 \times 128$ tiled prefill CUDA kernels and verifies massive context scaling on NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory:

* **Super Fat Prefill GEMM (`csrc/exl3_fat_gemm.cu`)**: A $128 \times 128$ tiled CUDA GEMM for routed experts that receive large token batches during prefill, unrolling Trellis dequantization in registers and fusing Hadamard scaling directly into the tile.
* **Inline Routing & Atomic Token Scatter (`exl3_fat_gemm_scatter`)**: Fuses expert down-projection with routing weight multiplication and atomic token output scatter, running **up to 2.09x faster** than the reconstructed GEMM path ($400.4\ \mu\text{s} \to 195.3\ \mu\text{s}$ at $M=1024$) with **1.000000 cosine similarity parity**.
* **Max KV Cache Pool**: **1,908,408 tokens (~1.91 Million tokens)** allocated in 22.39 GiB FP8 memory on NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory.
* **Max Context Scaling**: Full **131,072 tokens (128K context)** supported with **14.56x concurrent streams**, and **262,144 tokens (256K context)** verified on DeepSeek-V4-Flash-Vision with DSpark speculative decoding.

### Prefill Down-Projection + Scatter Speedup (NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory)

| Prefill Rows ($M$) | Stock Reconstruct + GEMM | Native Fat Scatter | Net Speedup | Parity Cosine Similarity |
|---|---|---|:---:|:---:|
| **$M = 256$** | 144.3 $\mu\text{s}$ | **86.9 $\mu\text{s}$ | **1.66x** | 1.000000 |
| **$M = 512$** | 222.8 $\mu\text{s}$ | **106.4 $\mu\text{s}$ | **2.09x** | 1.000000 |
| **$M = 1024$** | 400.4 $\mu\text{s}$ | **195.3 $\mu\text{s}$ | **2.05x** | 1.000000 |
| **$M = 2048$** | 733.1 $\mu\text{s}$ | **508.7 $\mu\text{s}$ | **1.44x** | 1.000000 |

### Speculation and Context Scaling Helpers

The plugin exposes small serving-side helpers so a scheduler and its
admission checks share one policy:

* **Dynamic speculative draft scheduler** —
  `get_speculative_draft_tokens(batch_size)` selects `K=3` for batches `[1..4]`,
  `K=2` for `[5..8]`, `K=1` for `[9..16]`, and `K=0` otherwise. Override the
  ranges with `VLLM_EXL3_SPEC_SCHEDULE=1:4:3,5:8:2,9:16:1` or a caller-supplied
  schedule.
* **Vectorized on-device confidence pruning** —
  `filter_speculative_candidates(probs, threshold=0.5)` keeps only the
  sequential confident prefix for each sequence and returns a boolean mask
  plus per-sequence kept counts. Enable the integration with
  `VLLM_EXL3_ADAPTIVE_VERIFICATION=1` (also accepts `true`, `yes`, or `on`).
* **MLA KV-cache headroom** — `compute_mla_kv_cache_bytes` and
  `validate_context_scaling` model compressed KV storage before a launch. At
  the default 43 layers and FP8 storage, 64K requires **1.51 GiB**, 128K
  requires **3.02 GiB**, and 256K requires **6.05 GiB**. The validator reports
  usable headroom and physical safety margin for NVIDIA DGX Spark GB10
  (sm_121 Blackwell) with 128 GiB Unified Memory.

## v0.3.0 Native Kernel Suite & Benchmark Receipts (NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory)

Version `0.3.0` introduced custom native CUDA kernels (`csrc/`) replacing the stock `exllamav3_ext` decode and prefill paths on NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory:

* **In-Register Trellis Dequantization (`csrc/exl3_dequant.cuh`)**: Unrolls MCG bit extraction into registers without intermediate global memory roundtrips.
* **Dense & Batched GEMV (`csrc/exl3_gemv.cu`, `csrc/p2b_batched.cu`)**: Active-expert batched GEMV saturating 99.2% of the physical memory bandwidth floor (73.3 $\mu\text{s}$).
* **4-Phase Cooperative MoE Decode (`csrc/p2b_moe.cu`)**: End-to-end fused MoE decode reducing per-layer latency from $497\ \mu\text{s} \to 287.8\ \mu\text{s}$ ($1.73\times$).
* **Power-of-Two Chunked Prefill GEMM (`csrc/exl3_gemm.cu`)**: Tiled matrix multiplication delivering 7.85 TFLOPS ($13.0\times$ faster than legacy prefill).
* **vLLM Dispatch Control**: `VLLM_EXL3_MOE_KERNEL=auto` (default) selects an available backend; `native` and `exllamav3` request a specific backend. Unsupported native cases fall back to ExLlamaV3 or the Python loop.

The unreleased native MoE ABI 2 adds local expert widths of 1024 and 2048 at
hidden width 4096, with optional SwiGLU input clipping. This covers ordinary TP2
and TP1 expert geometry while retaining K2/K3/K4 and model-provided routing.
The wrapper still supports 1–8 decode rows through one native call per row.
Rebuild the extension with `pip install -e . --no-build-isolation`; the loaded
`vllm_exl3_c.P2B_MOE_ABI_VERSION` should be `2`. Older binaries fall back for
clipped or 1024-wide requests instead of interpreting incompatible pointers.
No speculative-depth default changes are included. Run
`python -m pytest -q tests/test_native_moe_contract.py` on the CUDA host before
qualifying the new path; its GPU checks cover independent CPU weight
reconstruction, clipping, and graph replay. Full-model correctness and speed
still need validation on the intended TP1/TP2 deployment.

### Live Head-to-Head Benchmark Receipts

Measured simultaneously across two physical NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory nodes (Baseline ExLlamaV3 vs. Native EXL3) running `GLM-5.3-Flash-EXL3-K2` via live vLLM HTTP streaming API:

| Category | Baseline ExLlamaV3 | Native EXL3 | Baseline TTFT | Native TTFT | Net Speedup |
|---|---|---|---|---|:---:|
| **Coding** | 14.9 tok/s | **27.6 tok/s** | 2,343.8 ms | **859.1 ms** | **+85.6%** |
| **Prose** | 13.7 tok/s | **24.6 tok/s** | 355.4 ms | **308.7 ms** | **+79.3%** |
| **Reasoning** | 18.9 tok/s | **25.1 tok/s** | 482.2 ms | **407.8 ms** | **+32.7%** |
| **Summary** | 17.1 tok/s | **25.6 tok/s** | 409.6 ms | **345.4 ms** | **+50.0%** |
| **Format** | 16.3 tok/s | **24.0 tok/s** | 401.9 ms | **349.8 ms** | **+47.7%** |
| **JSON** | 20.8 tok/s | **25.6 tok/s** | 502.6 ms | **414.1 ms** | **+23.3%** |
| **HTML** | 19.5 tok/s | **23.1 tok/s** | 361.7 ms | **323.1 ms** | **+18.6%** |
| **Narrative** | 14.0 tok/s | **21.0 tok/s** | 395.4 ms | **333.0 ms** | **+50.0%** |
| **Average Across Categories** | **16.9 tok/s** | **24.6 tok/s** | **656.6 ms** | **417.6 ms** | **+45.6%** |

### Per-Step Decode Latency Breakdown (C1)

* **40 MoE Layers**: Cut from $19.9\ \text{ms} \to 11.5\ \text{ms}$ ($497\ \mu\text{s} \to 287.8\ \mu\text{s}$ per layer), saving **8.4 ms in MoE compute alone** per token.
* **Total Per-Step Time**: Reduced from **$59.2\ \text{ms} \to 40.6\ \text{ms}$ (-31.4%)**, directly powering the +45.6% throughput gain.
* **Prefill GEMM**: 7.85 TFLOPS ($13.0\times$ faster), holding **1,875 tok/s** cold prefill across 65k context.
* **NVMe Storage Scaling**: 8-worker parallel read reaches **3,563 MB/s** ($3.0\times$ speedup over single-thread 1,185 MB/s), loading 96 GB weights in ~27 seconds.

### Essential Serving Flag
When running on NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory or other long-context instances, pass:
```bash
--long-prefill-token-threshold 1024
```
This prevents long prompt prefill from starving parallel decode steps and stalling the scheduler.

## Supported architectures

| Architecture | Status | Reference pack |
|---|---|---|
| `Glm5Next` (GLM-5.3-Flash) | serving-proven | GLM-5.3-Flash EXL3 K2 / K2K3-mix |
| `DeepseekV4` (DeepSeek-V4-Flash) | serving-proven on stock vLLM 0.28.0 (text, DSpark draft) and on the vLLM nightly vision class (text + images, DSpark draft, 64k context, tool calling); three small serving-side patches live in the recipe | DSV4-Flash-Vision EXL3 MixedK |
| `Qwen4ExpForConditionalGeneration` (Qwen3.8-Flash-Next) | serving-proven, native ExLlamaV3 pack (fractional-bit mul1 experts, padded dense linears, row-wise n-gram embedding table); three small vLLM patches in `tools/patch_vllm_qwen4_exp/` | [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3), revision `3.05bpw_h5_ng5` |

### Native ExLlamaV3 packs

Beyond packs produced by this project's own conversion path, the plugin also
serves packs quantized directly by turboderp's ExLlamaV3 tooling ("native"
packs), which can assign a fractional average bit width (e.g. 3.05 bpw) by
choosing bits per tensor rather than one width for the whole model. Support
for this covers:

- the `mul1` codebook (alongside `mcg`), selected per tensor via the packed
  marker rather than declared once for the whole model;
- per-tensor K for dense linears and `lm_head` through `non_routed_exl3`;
- matrices padded to a multiple of 128 so trellis tiles never spill past
  the real dimension;
- row-wise n-gram embedding tables (`ngram_embedding`) kept packed instead
  of expanded to BF16 -- 30.4 GiB on device instead of 102 GB.

A native pack's `config.json` was not written for this plugin, so
`tools/exl3_pack_tools/` rewrites it (see that directory's README), and
`model.safetensors.index.json` must list every `*.safetensors` file the pack
actually ships or vLLM silently skips tensors --
`tools/exl3_pack_tools/regenerate_safetensors_index.py` rebuilds it from the
files on disk. `Qwen4ExpForConditionalGeneration` also needs the three vLLM
patches in `tools/patch_vllm_qwen4_exp/` so vLLM's own model code passes
`quant_config` through to `lm_head` and the n-gram table at all.
