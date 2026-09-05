# Changelog

## Unreleased

- Extend the native fused MoE ABI with local intermediate width (1024 or 2048)
  and optional input-clipped SwiGLU. Clamp the gate before SiLU and the up
  projection symmetrically; zero keeps plain SwiGLU. Preserve K2/K3/K4 support.
- Keep legacy native calls compatible and fall back when an older extension
  cannot implement the requested width or clipping. Rebuild `vllm_exl3_c` to
  obtain `P2B_MOE_ABI_VERSION=2`.
- Add CPU dispatch/compatibility tests and CUDA numerical/graph-replay fixtures.
  Spark performance and full-model TP1/TP2 qualification remain required; no
  new throughput result is claimed.

## 0.3.1

- **Super Fat GEMM Prefill Kernel Suite (`csrc/exl3_fat_gemm.cu`, `csrc/exl3_fat_gemm.cuh`)**:
  - Tiled chunked prefill kernel optimized for wide-layer routed expert evaluation during high-context and large-batch prompts.
  - Implements batched matrix multiplication over unquantized and trellis-dequantized states with register-level unrolling.
  - Dispatched automatically via `apply_exl3_batched_fat` in `src/vllm_exl3/exl3.py`.
- **Bug Fix**:
  - Guard `k == 4` in `apply_exl3_batched_fat` dispatch to prevent illegal memory layout indexing when handling 4-bit trellis tiles.
- **Upstream Attribution & Notice Compliance**:
  - Full third-party attribution prominently placed at the top of `README.md` and detailed in `THIRD_PARTY_NOTICES.md`.
  - Credits to @MiaAI-Lab and @plotarmordev for the routed-expert EXL3 serving path and Fat GEMM CUDA kernels (`GLM-5.3-Flash-EXL3-2x-DGX-Sparks`, commit `4b8d3c7`).
  - Credits to @turboderp for the ExLlamaV3 trellis quantization format, MCG codebook, and base dequantization math.
- **Hardware Benchmarks**:
  - Benchmarked on NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory.
- **Dynamic Speculative Draft Scheduler**:
  - `get_speculative_draft_tokens` selects K dynamically by batch size: `[1..4]` → 3, `[5..8]` → 2, `[9..16]` → 1, and larger batches → 0.
  - `VLLM_EXL3_SPEC_SCHEDULE` provides a validated `min:max:k` override.
- **Vectorized On-Device Confidence Pruning**:
  - `filter_speculative_candidates` truncates each candidate stream at its first below-threshold confidence without host-side loops.
  - `VLLM_EXL3_ADAPTIVE_VERIFICATION` enables the opt-in verification path.
- **Context Ceiling Scaling & MLA KV Cache Headroom**:
  - `compute_mla_kv_cache_bytes` and `validate_context_scaling` cover 64K (1.51 GiB), 128K (3.02 GiB), and 256K (6.05 GiB) FP8 MLA KV storage.

## 0.3.0

- **Native EXL3 CUDA Kernel Suite (`csrc/`)**: High-performance native CUDA kernels replacing `exllamav3_ext` decode and prefill paths on NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory:
  - **In-Register Trellis Dequantization (`csrc/exl3_dequant.cuh`)**: Unrolls MCG bit extraction into hardware registers without intermediate global memory roundtrips.
  - **Active-Expert Batched GEMV (`csrc/exl3_gemv.cu`, `csrc/p2b_batched.cu`)**: Saturates 99.2% of the physical memory bandwidth floor (73.3 μs).
  - **4-Phase Cooperative MoE Decode (`csrc/p2b_moe.cu`)**: End-to-end fused MoE decode reducing per-layer latency from 497 μs → 287.8 μs (1.73x speedup).
  - **Power-of-Two Chunked Prefill GEMM (`csrc/exl3_gemm.cu`)**: Tiled matrix multiplication delivering 7.85 TFLOPS (13.0x faster than legacy prefill).
  - **vLLM Dispatch Control**: Environmental toggle `VLLM_EXL3_MOE_KERNEL=native` (default) with zero-cost fallback to `exllamav3`.
- **NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory Live Benchmark Receipts**:
  - Measured across NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory nodes running GLM-5.3-Flash EXL3 K2 via live vLLM HTTP streaming API:
    - Coding: 14.9 tok/s → 27.6 tok/s (+85.6% speedup, TTFT 2,343.8 ms → 859.1 ms)
    - Prose: 13.7 tok/s → 24.6 tok/s (+79.3% speedup)
    - Summary: 17.1 tok/s → 25.6 tok/s (+50.0% speedup)
    - Average Across Categories: 16.9 tok/s → 24.6 tok/s (+45.6% speedup)
  - MoE Compute: Cut from 19.9 ms → 11.5 ms per token (-42.2%), saving 8.4 ms in pure MoE decode compute.
  - Prefill: 1,875 tok/s cold prefill across 65k context.
- **Serving Guidance**:
  - Added `--long-prefill-token-threshold 1024` recommendation to prevent long prompt prefill from starving parallel decode steps.
- **Attribution & Notice Compliance**:
  - Full third-party attribution and notices for Turboderp (@turboderp) and Mia's AI Lab (@MiaAI-Lab) documented in `THIRD_PARTY_NOTICES.md`.

## 0.2.3

- Dense EXL3 for non-routed linears (`quantization_config.non_routed_exl3`): per-module `layers` map with `bits` and `bf16_shards`, `mul1` codebook alongside `mcg`, mixed EXL3/BF16 shards inside one merged linear, stale BF16 `.weight` tensors discarded with a shape check. `tools/dense_overlay.py` assembles an overlay pack from an existing EXL3 checkpoint (it moves tensors, it does not quantize).
- `quantization_config.non_routed_dtype_policy: "bf16_as_stored"`: dense linears go to vLLM's unquantized method instead of the `non_routed_quantization` delegate, which still serves source-format MTP experts. Fixes silent empty output when BF16 dense weights met an fp8 delegate (DeepSeek-V4-Flash on stock vLLM 0.28).
- Fix: `mul1` codebook marker constant (0x83DCD12D as signed int32 is -2082680531).

## 0.2.2

- Mixed-format packs: `quantization_config.mtp_experts: "source"` routes MTP/draft-block routed experts through the declared `non_routed_quantization` method (e.g. MXFP4) instead of EXL3, enabling MTP speculative serving for packs that keep drafter experts in the source format. Default (`"exl3"`) is unchanged.

## 0.2.1

- Fix: the `glm53_exl3_plugin` compatibility shim now provides a real `glm53_exl3_plugin.exl3` submodule (submodule imports bypassed the lazy alias).

## 0.2.0

First standalone release. Renamed from `glm53_exl3_plugin` 0.1.1 — identical
behavior plus the package rename; the old import path remains as a
deprecated shim.

## 0.1.1 (as glm53_exl3_plugin, shipped in the GLM recipe)

- Raise the fused-MoE per-expert row cap (`TEMP_ROWS_FUSED` 128 → 2048),
  fixing the >163k-token prefill stall where fat experts fell back to a
  slow per-expert reconstruction path.
- Delegate non-routed layers to a pack-declared source-format quant method
  (`quantization_config.non_routed_quantization`).

## 0.1.0 (as glm53_exl3_plugin)

Initial in-recipe release: EXL3/MCG routed-expert quantization method for
vLLM fork runtimes, per-layer `layer_bits` support.
