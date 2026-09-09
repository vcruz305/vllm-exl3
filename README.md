![vllm-exl3 — EXL3 quantization plugin for routed MoE serving](assets/header.png)

# vllm-exl3

[![Follow on X](https://img.shields.io/badge/Follow-%40ViC305-black?logo=x)](https://x.com/ViC305) [![Follow on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Follow-vcruz305-yellow)](https://huggingface.co/vcruz305)

An out-of-tree vLLM plugin that registers `--quantization exl3`, serving EXL3 (ExLlamaV3 trellis, MCG codebook) quantized packs. Routed MoE experts remain packed at load time and execute through ExLlamaV3/native kernels.

If you use this plugin, please credit **vcruz305**.

## Credits & provenance

This project is intentionally explicit about upstream work and license lineage.

- The **EXL3 trellis format, MCG codebook, quantization method, and packed execution model** come from [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by Turboderp ([@turboderp](https://github.com/turboderp)).
- `csrc/exl3_fat_gemm.cu` and `.cuh` come from the historical MIT-licensed E2 work in [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks), and substantial portions of the routed-expert integration in `src/vllm_exl3/exl3.py` derive from their earlier `overlay/exl3.py`. Exact source lineage and notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- vLLM-derived integration points are identified in `THIRD_PARTY_NOTICES.md`.
- New upstream-informed work is labeled in Git history as **copied/derived**, **adapted design**, or **independent implementation**. See [docs/provenance.md](docs/provenance.md).

Current releases move forward under **AGPL-3.0-only** so improvements to modified network-served versions remain available to their users. Earlier vllm-exl3 releases were Apache-2.0; that text remains in [LICENSE.APACHE-2.0](LICENSE.APACHE-2.0). Third-party MIT/Apache notices are preserved.

## Scope (read this first)

This is **not** a plugin for stock vLLM. It targets vLLM fork lineages that expose the `RoutedExperts` fused-MoE family and requires ExLlamaV3 with its compiled extension for the target GPU architecture.

## v0.4.1 TP1 runtime policy & observability

Version `0.4.1` begins a TP1-focused optimization track without changing the default kernel contract before GPU qualification:

- `vllm_exl3.runtime_diagnostics()` reports selected backend, native ABI, per-bit native row caps, fused scratch row capacity, fat-expert threshold, fat-kernel availability, and speculative schedule.
- `VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K2`, `_K3`, and `_K4` allow controlled per-bit A/B testing instead of applying one row limit to all trellis widths.
- `VLLM_EXL3_FUSED_TEMP_ROWS` exposes the requested fused scratch capacity while preserving the historical 2048-row default.

These helpers are independently implemented around vllm-exl3's existing interfaces. Recent MiaAI-Lab serving work is credited in [docs/provenance.md](docs/provenance.md) as relevant design prior art; no post-relicense adaptive-scheduler or dense-FP8 source is copied into these helpers.

## v0.3.1 Super Fat GEMM Kernel & Ultra-Long Context Release

Version `0.3.1` adds accelerated 128x128 tiled prefill kernels and long-context helpers on DGX Spark GB10:

- **Super Fat Prefill GEMM (`csrc/exl3_fat_gemm.cu`)**: tiled routed-expert prefill path.
- **Inline routing/scatter**: fused down-projection weighting and scatter for eligible K4/MCG fat experts.
- **Context helpers**: `compute_mla_kv_cache_bytes` and `validate_context_scaling` provide planning estimates.
- **Speculation helpers**: `get_speculative_draft_tokens`, `parse_speculative_schedule`, and `filter_speculative_candidates` expose reusable scheduling primitives.

The existing measured K4 fat-kernel microbench receipts remain in Git history and release documentation. K2/K3 grouped-prefill acceleration is a separate future qualification target and is not claimed by this release.

## v0.3.0 Native Kernel Suite

Version `0.3.0` introduced the custom CUDA kernel suite in `csrc/`:

- in-register trellis dequantization,
- dense and batched GEMV,
- cooperative fused MoE decode,
- chunked prefill GEMM,
- backend selection with `VLLM_EXL3_MOE_KERNEL=auto|native|exllamav3`.

Native MoE ABI 2 supports local expert widths 1024 and 2048 at hidden width 4096, K2/K3/K4, and optional SwiGLU clipping. Run `python -m pytest -q tests/test_native_moe_contract.py` on the CUDA host before qualification.

## Supported architectures

| Architecture | Status | Reference pack |
|---|---|---|
| `Glm5Next` | serving-proven | GLM-5.3-Flash EXL3 K2 / K2K3-mix |
| `DeepseekV4` | serving-proven on supported fork lineages | DeepSeek-V4-Flash-Vision EXL3 MixedK |
| `Qwen4ExpForConditionalGeneration` | serving-proven with required model plumbing | Qwen3.8-Flash-Next native EXL3 pack |

## Native ExLlamaV3 packs

The plugin also supports native ExLlamaV3 packs with per-tensor bit width, `mcg`/`mul1` codebooks, padded dense linears, and row-wise n-gram embedding tables. `tools/exl3_pack_tools/` contains config/index utilities for packs whose metadata was not authored for this plugin.

## Installation

```bash
pip install vllm-exl3
```

For CUDA/native builds, use the repository build flow appropriate for your architecture. DGX Spark/aarch64 users should follow the matching model recipe so the vLLM fork, PyTorch/CUDA build, ExLlamaV3 extension, and plugin version are qualified together.

## Runtime diagnostics

```python
import json
import vllm_exl3
print(json.dumps(vllm_exl3.runtime_diagnostics(), indent=2, sort_keys=True))
```

For TP1 experiments, prefer changing one variable at a time and keeping a baseline receipt. In particular, do not reduce `VLLM_EXL3_FUSED_TEMP_ROWS` until the runner and native extension are proven never to exceed that row capacity.

## Validation

CPU/unit CI checks package integrity, attribution, config parsing, and non-GPU behavior. Native CUDA changes still require real GPU qualification. Recommended checks include:

```bash
python -m pytest -q tests/test_runtime_policy.py
python -m pytest -q tests/test_native_moe_contract.py
```

For performance changes, record TTFT, decode tok/s, acceptance where applicable, peak memory, effective runtime diagnostics, model revision, plugin revision, vLLM revision, CUDA/PyTorch versions, and exact serve flags.

## License

Current project code is licensed under **GNU AGPL-3.0-only**; see [LICENSE](LICENSE). Earlier vllm-exl3 releases were Apache-2.0 and that prior license text is preserved in [LICENSE.APACHE-2.0](LICENSE.APACHE-2.0). Third-party material remains under its original licenses and notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [NOTICE](NOTICE).
