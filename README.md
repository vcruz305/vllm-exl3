![vllm-exl3 — EXL3 quantization plugin for routed MoE serving](assets/header.png)

# vllm-exl3

[![Follow on X](https://img.shields.io/badge/Follow-%40ViC305-black?logo=x)](https://x.com/ViC305) [![Follow on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Follow-vcruz305-yellow)](https://huggingface.co/vcruz305)

An out-of-tree vLLM plugin registering `--quantization exl3` for EXL3 (ExLlamaV3 trellis) packs. It serves routed MoE experts and declared dense EXL3 tensors through ExLlamaV3 and optional native CUDA kernels. This is a **serving plugin, not a quantizer**.

**Use a compatible model recipe, not a stock vLLM installation.** The current integration targets fork/specialized runtimes exposing the required `RoutedExperts` fused-MoE interfaces. Installing this plugin alone does not add a missing model architecture to vLLM or ExLlamaV3.

## Credits and provenance

Please credit **vcruz305** and the upstream work this project builds on:

- **Turboderp / [ExLlamaV3](https://github.com/turboderp-org/exllamav3)**: EXL3 trellis format, MCG/mul1 codebooks, quantization math, packed execution, and reused extension kernels/headers.
- **Mia's AI Lab / [GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)**, including @plotarmordev: substantial routed-expert integration lineage and the historical MIT-licensed E2 fat-GEMM sources.
- **vLLM**: integration and checkpoint-loading interfaces identified in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Exact copied/derived files, historical notices, and the distinction between adapted design and independent implementation are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [docs/provenance.md](docs/provenance.md). The new policy/planner helpers do not copy the newer upstream scheduler or dense-FP8 implementation.

## Current development candidate: 0.4.2

The package metadata on this development line is `0.4.2`; this is **not a claim that a 0.4.2 wheel has been published or GPU-qualified**. The executable candidate for the GLM TP1 test protocol is commit `d3cfd394920360d69f820d2dc96f8292a9e10283`. Its CPU/source/packaging CI passed; end-to-end GB10 qualification is still required.

| Change | What is implemented | Qualification boundary |
|---|---|---|
| Per-bit native row caps | K2/K3/K4 overrides wrap the real native resolver during `register()` | Opt-in experiments; defaults and unsupported-shape guards remain |
| Gate/up SUH compatibility cache | Normal fused-state construction caches whether the input rotations match; direct fat-path callers cache on first use | Rotations must remain immutable after construction; new speedup not yet measured |
| Runtime diagnostics | Backend preference, extension ABI, row caps, registration state, scratch and grouped-planner status | Describes the calling process, not a remote serving worker |
| K2/K3 grouped-prefill planner | Default-off eligibility checks and bounded candidate row-window estimates | **No grouped CUDA executor; `execution_available` is false** |
| Fused scratch request | Reports the requested row capacity separately from the actual capacity | **Override inactive; actual `TEMP_ROWS_FUSED` remains 2048** |
| CPU-offload planner | Fail-closed metadata/architecture eligibility plan for an external ExLlamaV3 CPU-MoE experiment | **No CPU executor is implemented in `vllm-exl3`** |

Neither enabling the grouped planner nor requesting fewer fused rows currently changes serving allocations or provides a kernel speedup. Planner scratch estimates are not a measurement or bound on every existing runtime allocation.

Release history and earlier kernel work belong in [CHANGELOG.md](CHANGELOG.md). Historical K4 fat-GEMM microbenchmarks are not evidence of K2/K3 grouped-prefill acceleration.

## Compatibility and execution

Existing integrations include `Glm5Next`, `DeepseekV4`, DeepSeek-V4.1 compatibility helpers for a V4.1-capable vLLM runtime, and `Qwen4ExpForConditionalGeneration`, each requiring its matching model plumbing. The plugin does not make an unsupported model architecture appear in either vLLM or standalone ExLlamaV3.

This candidate's primary qualification target remains **one GB10, TP=1, GLM-5.3-Flash K2 and K2/K3-mix** unless a model-specific recipe says otherwise.

Start with the [GLM single-Spark recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe) and its [TP1 qualification protocol](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe/blob/main/docs/TP1_POLICY_AB.md).

### DeepSeek-V4.1-Flash

DeepSeek-V4.1 support has a strict ownership boundary:

- the dedicated V4.1 **vLLM** runtime owns `DeepseekV41ForCausalLM`, CED/CSA2 attention, Engram, routing and DSpark;
- `vllm-exl3` bridges EXL3 metadata and routed-expert execution into those vLLM layers;
- this plugin does **not** implement a standalone V4.1 model class;
- current upstream ExLlamaV3 does not provide the forward-correct V4.1/CED/Engram model graph required to load the checkpoint standalone.

The public compiled checkpoint under active qualification is [`vcruz305/DSV4.1-Flash-EXL3-4.75bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw). Use the [DeepSeek-V4.1 DGX Spark recipe](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe), which now includes a fail-closed checkpoint compatibility check and an explicit small-GPU/large-host-RAM validation protocol.

The plugin also exposes a policy-only CPU-offload planner:

```python
from vllm_exl3 import plan_exllamav3_cpu_offload

plan = plan_exllamav3_cpu_offload(
    architecture="DeepseekV41ForCausalLM",
    codebooks="mul1",
    max_k=4,
    uniform_expert_biases=True,
    exllamav3_architecture_available=False,
)
print(plan.to_dict())
```

`plan.vllm_exl3_execution_available` is intentionally `False`. The helper records the current upstream ExLlamaV3 CPU-MoE eligibility contract and prevents tooling from implying that `vllm-exl3` has a CPU backend when it does not. See [`docs/CPU_OFFLOAD.md`](docs/CPU_OFFLOAD.md).

### Qwen3.8-Flash-Next

`Qwen4ExpForConditionalGeneration` serves from turboderp's native ExLlamaV3 pack
([Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3), revision `3.05bpw_h5_ng5`),
including its row-wise n-gram embedding table through `Exl3EmbeddingMethod`. It needs the three vLLM patches in
`tools/patch_vllm_qwen4_exp/` and the one-time pack rewrites described in the
[Qwen single-Spark recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe).

Measured on one GB10 at TP=1 on plugin revision `6b26e5c` against vLLM `0.28.1rc1.dev324`, MTP k=2, 65,536-token
context, one request in flight: greedy decode 47.6 tok/s at p50 with 0.300 s TTFT p50, 38.4 tok/s at vendor
thinking settings, and 1,122 tok/s prefill on a 9,483-token prompt. Weights occupy 79.96 GiB resident with the
n-gram table included, leaving 11.04 GiB of KV cache, which is 303,951 tokens at 64k context, for 102 to 103 GiB
of system memory in use and 79.4 GiB on disk. On sixcat-eval v0.5.1 under vendor policy it scores 86.7 on first
attempt and 90.0 best-of-attempts; the Q4_K_M GGUF of the same model on llama.cpp scores 89.2 and 92.5 on the
same two bases while decoding 1.45x slower and prefilling at roughly half the rate, because its 95.4 GiB BF16
embedding table is paged from NVMe rather than held resident.

These figures are a record of that revision on that workload. They are not part of the 0.4.2 qualification target
above, and they were not re-measured on 0.4.2.

Routed expert weights remain packed at load time. Some fallback/prefill paths reconstruct **temporary FP16 weights for an expert**; packed loading does not mean zero reconstruction or zero scratch memory. The existing tiled fat-GEMM fast path is gated to eligible **K4/MCG**, non-mul1 projections with compatible gate/up input rotations. K2/K3 and distinct-rotation cases retain their applicable fallback paths.

Native MoE ABI 2 includes hidden width 4096 and local intermediate widths 1024/2048 with K2/K3/K4 and optional SwiGLU clipping. These are kernel contract dimensions, not a promise that every model or row count uses the native path. Unsupported cases may fall back. Verify actual dispatch, not only the requested backend or presence of an extension symbol.

## CPU-offload boundary

`vllm-exl3` currently has **no host-resident CPU expert executor**. Large-host-memory deployments must not assume that the presence of EXL3 expert weights implies CPU execution support.

Current upstream ExLlamaV3's experimental CPU-MoE path is a separate runtime feature. Its current eligibility contract requires `mul1` experts, K <= 8, uniform expert-bias presence, and a model architecture that ExLlamaV3 can instantiate. For DeepSeek-V4.1 the architecture requirement is presently the first blocker for standalone ExLlamaV3.

Use `plan_exllamav3_cpu_offload()` for policy/preflight tooling, then qualify the external runtime separately. More detail: [`docs/CPU_OFFLOAD.md`](docs/CPU_OFFLOAD.md).

## Pack metadata

For the routed-expert packs used by the GLM recipe, the basic declaration is:

```json
{
  "quantization_config": {
    "quant_method": "exl3",
    "bits": 2,
    "codebook": "mcg"
  }
}
```

Mixed packs additionally declare per-layer overrides in `layer_bits`. Preserve the checkpoint's existing config, index, tensor shapes and bit-width metadata; do not flatten a mixed pack to the base `bits` value.

Other supported metadata includes `non_routed_quantization` for delegating source-format non-routed weights and `non_routed_exl3` for declared dense EXL3 linears. Native ExLlamaV3 packs can also use per-tensor widths, mul1 codebooks, padding and row-wise n-gram embeddings. These capabilities do not make every specialized kernel eligible. See [AGENTS.md](AGENTS.md), `src/vllm_exl3/exl3.py`, and `tools/exl3_pack_tools/` for the serving-side contract.

For CPU-offload planning, never infer a codebook from average bpw or the repository name. Inspect the checkpoint's actual metadata/tensor suffixes first.

## Installation

Use the model recipe to establish the compatible vLLM fork, Python, PyTorch/CUDA and ExLlamaV3 build first. **Do not upgrade a working fork to stock vLLM to resolve an import error.** A published package version, a Git tag and a locally built wheel are different artifacts; retain the exact artifact and hash used by a working baseline.

For a native candidate build from an exact checked-out source revision, in the prepared runtime environment:

```bash
# Requires PyTorch, ExLlamaV3 extension headers, the CUDA toolkit,
# setuptools >= 77, wheel, and the build tools required by the runtime.
unset VLLM_EXL3_NO_CUDA
python -m pip install --no-build-isolation --no-deps .
python -c "import torch, vllm_exl3_c; print(vllm_exl3_c.P2B_MOE_ABI_VERSION)"
```

For GLM/GB10, prefer the recipe's exact-ref `scripts/install_candidate_plugin.sh` after preserving the baseline. A `VLLM_EXL3_NO_CUDA=1` build is a Python-only distribution and **does not qualify native CUDA behavior**. Never mix candidate Python files with an unverified old native extension.

## Runtime controls and diagnostics

Set controls **before starting the serving process**, and restart that process between variants.

| Control | Meaning |
|---|---|
| `VLLM_EXL3_MOE_KERNEL=auto\|native\|exllamav3` | Backend preference; shape/format checks and fallback still apply |
| `VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K2`, `_K3`, `_K4` | Per-bit native row ceiling; absent/invalid values fall back to the existing resolver, zero disables eligibility for that width |
| `VLLM_EXL3_FAT_THRESHOLD` | Existing fat-expert routing threshold; default 256 |
| `VLLM_EXL3_FUSED_TEMP_ROWS` | Requested capacity only; no active allocation override |
| `VLLM_EXL3_GROUPED_PREFILL` | Requests experimental planner eligibility only; default off |
| `VLLM_EXL3_GROUPED_PREFILL_MAX_ROWS` | Planner row-window budget, not a live kernel allocation setting |

In a **fresh process in the intended runtime environment**, explicitly register before inspecting the policy:

```python
import json
import vllm_exl3

vllm_exl3.register()
print(json.dumps(vllm_exl3.runtime_diagnostics(), indent=2, sort_keys=True))
```

Check `per_bit_native_policy_installed`, `native_available`, `native_abi`, `native_row_caps`, `fused_temp_rows_actual`, `fused_temp_rows_override_active`, `grouped_prefill.execution_available`, and `cpu_offload.execution_available`. A standalone diagnostic is a preflight, not proof of what an already-running server loaded. Capture worker-side configuration and dispatch evidence for benchmarks.

Speculation/context helpers are callable utilities. Their existence does not establish that a vLLM runner uses an adaptive schedule, or that a generic MLA memory estimate models GLM's hybrid caches, draft state, workspace and graph pools.

## Validation

From the matching source checkout:

```bash
python -m pytest -q tests/test_runtime_policy.py tests/test_prefill_policy.py
python -m pytest -q tests/test_native_moe_contract.py tests/test_fat_distinct_suh.py
python -m pytest -q tests/test_cpu_offload_plan.py
```

GPU tests that skip are **not passes for hardware qualification**. Run the full suite as well and retain the skip/failure report. Native tests, real-checkpoint parity, changed-input/routing graph replay, memory behavior and end-to-end serving are separate gates.

Compare exact revisions on the same checkpoint and workload. Record cold-prefix TTFT separately from warm prefix-cache hits, output-token accounting including reasoning, server-side runtime identity, actual dispatch, acceptance, peak allocated/reserved memory, host memory headroom and exact launch flags. Keep kernel microbenchmarks separate from full-model performance, and aggregate throughput separate from per-request speed.

## License

Current project work is distributed under **AGPL-3.0-only**; see [LICENSE](LICENSE). The historical Apache-2.0 text is retained in [LICENSE.APACHE-2.0](LICENSE.APACHE-2.0). Third-party material retains its applicable notices in [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights and other runtime dependencies have separate licenses.
