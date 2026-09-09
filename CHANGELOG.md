# Changelog

## Unreleased

### Added

- `runtime_diagnostics()` exposes the effective EXL3 serving policy in a JSON-friendly form: selected MoE backend, native extension availability and ABI, per-bit native decode-row caps, fused scratch row capacity, fat-expert threshold, fat-kernel availability, and speculative schedule. This gives recipes a machine-checkable record of what actually loaded rather than relying on requested environment variables alone.
- `src/vllm_exl3/runtime_policy.py` adds independently implemented TP1-oriented policy helpers. Per-bit overrides (`VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K2/K3/K4`) make it possible to A/B K2 separately from K3/K4 instead of forcing one global native-row threshold across different trellis costs. `VLLM_EXL3_FUSED_TEMP_ROWS` exposes the requested fused scratch capacity while preserving the historical 2048-row default.
- `docs/provenance.md` records exact source/design provenance and establishes commit-message labels for copied/derived work, adapted designs, and independent implementations.
- `VLLM_EXL3_PREFILL_SYNC=<max_rows>`: synchronizes the device before each EXL3 dense or routed-expert call whose row count is between 2 and max_rows (never during CUDA graph capture, never for single-row decode). Workaround for a vLLM nightly V2 model runner wedge on Qwen3.8-Flash-Next where prefills of roughly 33 to 144 tokens never complete; decode speed is unchanged. Recommended value 256 on that runner.

### Licensing

- vllm-exl3 moves forward under **AGPL-3.0-only** so improvements to modified network-served versions remain available to their users. The prior Apache-2.0 license text is retained in `LICENSE.APACHE-2.0`; all existing MIT/Apache third-party notices remain in `THIRD_PARTY_NOTICES.md` and `NOTICE`.
- The newer TP1 runtime-policy helpers were independently written around this project's APIs. Recent public MiaAI-Lab GLM serving work is credited as relevant design prior art; no post-relicense scheduler or dense-FP8 source was copied into these helpers.

### Fixed

- Dense EXL3 calls no longer launch exllamav3's cooperative trellis GEMM. Rows 17 to 144 take exllamav3's reconstruct+hgemm path (exact); rows up to 16 keep exllamav3's own dispatch. On the vLLM nightly V2 model runner the cooperative GEMM wedged the engine; a 4-worker MTP stress that wedged within 3 minutes ran clean for 45 minutes once the routing was in place, with decode speed unchanged. `VLLM_EXL3_COOP_GEMM=1` restores the old dispatch; `VLLM_EXL3_RECONSTRUCT_MIN_ROWS` moves the reconstruct threshold.
- Fat-expert prefill path: the branch for packs whose gate and up projections carry distinct `suh` rotations handed column slices of the shared `gate_up` scratch buffer to kernels requiring contiguous row-major operands. The branch now runs on contiguous fp32 temporaries. Regression test: `tests/test_fat_distinct_suh.py`.
- Native fused-MoE decode-row cap measured on GB10 (K2: 8 rows, K3/K4: 1) ships as an opt-in (`VLLM_EXL3_NATIVE_MOE_MEASURED_CAP=1`, or `VLLM_EXL3_NATIVE_MOE_MAX_ROWS=<n>`); the default keeps the dispatch contract of up to 8 rows. Receipt: `tools/receipts/ab_moe_gb10.json`.
- Add `Exl3EmbeddingMethod` for row-wise n-gram embedding tables (`ngram_embedding`), decoded through the compiled `exllamav3_ext.ngram_dequant` kernel or a pure-torch fallback (`VLLM_EXL3_NGRAM_KERNEL=ext|torch`).
- Add the `ngram_embedding` config spec and its checkpoint layout; tensor parallel size 1 only.
- Extend `get_quant_method` with branches for `ParallelLMHead` and `VocabParallelEmbedding`.
- Accept tuple and `None` shard-id spans in the dense linear weight loader.
- Generalize `_check_moe_codebook_markers` to validate `mul1` markers for routed experts alongside `mcg`.
- Add `_exl3_routed_experts_loader`, a per-expert `load_weights` path for routed-experts layers that mirrors vLLM's checkpoint-name resolution.
- Pad dense EXL3 linear geometry to multiples of 128 so trellis tiles never spill past the real matrix dimension.
- Thread per-tensor codebook flags through the fused `exl3_moe` launch arguments.
- Register the EXL3 custom ops so they trace opaquely under `torch.compile(fullgraph=True)` instead of breaking graph capture.
- Restore the CUDA-graph-safe fat-expert-sync guard and the MTP/draft-expert quant-method delegate.
- Add `tools/patch_vllm_qwen4_exp`, `tools/exl3_pack_tools`, and `tools/verify_native_pack`.
- Add attribution notices for ExLlamaV3's n-gram embedding codec and for vLLM's routed-experts loader/custom-op registration.
- Extend the native fused MoE ABI with local intermediate width (1024 or 2048) and optional input-clipped SwiGLU while preserving K2/K3/K4 support.
- Keep legacy native calls compatible and fall back when an older extension cannot implement the requested width or clipping. Rebuild `vllm_exl3_c` to obtain `P2B_MOE_ABI_VERSION=2`.
- Add CPU dispatch/compatibility tests and CUDA numerical/graph-replay fixtures. Spark performance and full-model TP1/TP2 qualification remain required; no new throughput result is claimed.

## 0.3.1

- **Super Fat GEMM Prefill Kernel Suite (`csrc/exl3_fat_gemm.cu`, `csrc/exl3_fat_gemm.cuh`)**:
  - Tiled chunked prefill kernel optimized for wide-layer routed expert evaluation during high-context and large-batch prompts.
  - Implements batched matrix multiplication over unquantized and trellis-dequantized states with register-level unrolling.
  - Dispatched automatically via `apply_exl3_batched_fat` in `src/vllm_exl3/exl3.py`.
- **Bug Fix**:
  - Guard `k == 4` in `apply_exl3_batched_fat` dispatch to prevent illegal memory layout indexing when handling 4-bit trellis tiles.
- **Upstream Attribution & Notice Compliance**:
  - Full third-party attribution documented in `THIRD_PARTY_NOTICES.md` and the README Credits & provenance section.
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
  - `compute_mla_kv_cache_bytes` and `validate_context_scaling` cover 64K, 128K, and 256K FP8 MLA KV storage estimates.

## 0.3.0

- **Native EXL3 CUDA Kernel Suite (`csrc/`)**: high-performance native CUDA kernels replacing `exllamav3_ext` decode and prefill paths on NVIDIA DGX Spark GB10.
- **Serving Guidance**: added `--long-prefill-token-threshold 1024` recommendation to prevent long prompt prefill from starving parallel decode steps.
- **Attribution & Notice Compliance**: full third-party attribution and notices for Turboderp and Mia's AI Lab documented in `THIRD_PARTY_NOTICES.md`.

## 0.2.3

- Dense EXL3 for non-routed linears (`quantization_config.non_routed_exl3`): per-module `layers` map with `bits` and `bf16_shards`, `mul1` codebook alongside `mcg`, mixed EXL3/BF16 shards inside one merged linear, stale BF16 `.weight` tensors discarded with a shape check. `tools/dense_overlay.py` assembles an overlay pack from an existing EXL3 checkpoint.
- `quantization_config.non_routed_dtype_policy: "bf16_as_stored"`: dense linears go to vLLM's unquantized method instead of the `non_routed_quantization` delegate, which still serves source-format MTP experts.
- Fix: `mul1` codebook marker constant.

## 0.2.2

- Mixed-format packs: `quantization_config.mtp_experts: "source"` routes MTP/draft-block routed experts through the declared `non_routed_quantization` method instead of EXL3, enabling MTP speculative serving for packs that keep drafter experts in the source format.

## 0.2.1

- Fix: the `glm53_exl3_plugin` compatibility shim now provides a real `glm53_exl3_plugin.exl3` submodule.

## 0.2.0

First standalone release. Renamed from `glm53_exl3_plugin` 0.1.1; the old import path remains as a deprecated shim.

## 0.1.1 (as glm53_exl3_plugin, shipped in the GLM recipe)

- Raise the fused-MoE per-expert row cap (`TEMP_ROWS_FUSED` 128 → 2048), fixing the >163k-token prefill stall where fat experts fell back to a slow per-expert reconstruction path.
- Delegate non-routed layers to a pack-declared source-format quant method.

## 0.1.0 (as glm53_exl3_plugin)

Initial in-recipe release: EXL3/MCG routed-expert quantization method for vLLM fork runtimes, per-layer `layer_bits` support.
