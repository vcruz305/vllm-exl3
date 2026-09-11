# DeepSeek V4.1 Flash: TP4 + EP4 EXL3 bring-up

This document defines the first-boot contract for serving `deepseek-ai/DeepSeek-V4.1-Flash` with `vllm-exl3` on four DGX Sparks.

The plugin does **not** reimplement DeepSeek V4.1. The dedicated vLLM V4.1 runtime owns the architecture, sparse attention, Lightning Indexer, Engram, Hyper-Connections, vision, DSpark, tokenizer, reasoning parser and tool parser. `vllm-exl3` owns the EXL3 expert storage/execution boundary and the mixed-checkpoint delegation contract.

## Recommended parallel layout

Use the model at tensor parallel size 4 **with expert parallelism enabled**.

Under vLLM EP, the routed MoE changes from tensor-sharded experts to whole-expert ownership across the original TP group:

| Layout | Experts/rank | Hidden | Intermediate/rank | EXL3 path |
|---|---:|---:|---:|---|
| TP4 only | 384 | 5120 | 576 | Not recommended; exceeds the current ExLlamaV3 fused expert-count envelope and leaves a 64-element tail in 128-wide native tiles |
| **TP4 + EP4** | **96** | **5120** | **2304** | **Recommended; full experts, dimensions aligned to 128, within ExLlamaV3's 128-local-expert fused envelope** |

The safe first V4.1 boot remains the ExLlamaV3 fused expert executor. This branch also generalizes the native cooperative p2b kernel to positive 128-aligned hidden/intermediate dimensions and bumps its ABI to **3**. V4.1 native dispatch is intentionally opt-in until GB10 parity and throughput are measured:

```bash
export VLLM_EXL3_V41_NATIVE_MOE=1
```

The Python compatibility layer refuses the V4.1 native geometry unless the loaded extension reports `P2B_MOE_ABI_VERSION >= 3`, preventing an older ABI-2 `.so` from being used accidentally.

## Required EXL3 pack metadata

For a V4.1 pack that EXL3-quantizes the **main routed expert stack** while preserving official source formats for the rest of the checkpoint, the outer config must retain enough source quantization metadata for vLLM's V4.1 architecture-level probes:

```json
{
  "quantization_config": {
    "quant_method": "exl3",
    "bits": 2,
    "codebook": "mcg",
    "scope": "deepseek_v41_routed_experts",
    "non_routed_quantization": {
      "quant_method": "deepseek_v4_fp8",
      "weight_block_size": [32, 32],
      "activation_scheme": "dynamic"
    },
    "mtp_experts": "source"
  }
}
```

Mixed-K packs should keep their existing `layer_bits` ledger. Do not flatten a mixed K2/K3 pack to the base `bits` value.

`vllm-exl3` now surfaces `non_routed_quantization.weight_block_size` through the outer `Exl3Config`. This is required because the V4.1 model inspects the global quantization config before individual dense layers request their delegated quantization method.

### DSpark

For the first boot, keep all three DSpark draft stages in their official source format:

```json
"mtp_experts": "source"
```

Existing packs may continue to set:

```json
"mtp_experts_start_layer": 40
```

For V4.1 packs that omit the numeric boundary, the plugin can infer DSpark source delegation only when all of the following are true:

- `mtp_experts == "source"`
- the source delegate is `deepseek_v4_fp8`
- its block shape is `[32, 32]`
- the routed block has 128 global experts

The 384-expert main stack remains EXL3.

## Runtime baseline

Start from the dedicated vLLM DeepSeek V4.1 image/runtime rather than stock pip vLLM. Pin the exact image digest and plugin commit used for every measurement.

Minimum logical launch shape:

```bash
vllm serve /models/DeepSeek-V4.1-Flash-EXL3 \
  --quantization exl3 \
  --tokenizer-mode deepseek_v41 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel
```

Do not treat that minimal command as the final DGX Spark launch recipe. SM121 sparse-attention, CUDA-graph and Engram placement must be qualified against the exact pinned V4.1 image.

## First-boot progression

Run these gates in order. Never change EXL3, Engram, speculative decoding and CUDA-graph behavior in the same A/B.

1. **Preflight / no checkpoint**
   - `plan_deepseek_v41()` reports 96 local experts and `preferred_first_boot_backend=exllamav3`.
   - `runtime_diagnostics()` reports `deepseek_v41.compat_installed=true`.
   - Source quantization metadata resolves `weight_block_size=[32,32]`.

2. **TP4 + EP4, source DSpark disabled**
   - Main routed experts load as EXL3.
   - Each rank owns 96 expert packs.
   - Dense/attention/shared weights use the V4.1 source MXFP8 delegate.
   - No persistent BF16 reconstruction of routed experts.
   - Start eager for the first correctness pass.

3. **Native p2b A/B**
   - Rebuild `vllm_exl3_c` from this branch and verify ABI 3.
   - Hold the checkpoint, prompts, TP4+EP4 topology and sampling constant.
   - Baseline with `VLLM_EXL3_V41_NATIVE_MOE` unset.
   - Candidate with `VLLM_EXL3_V41_NATIVE_MOE=1`.
   - Require output/parity checks before interpreting throughput.

4. **DSpark-5**
   - Keep DSpark experts source-native.
   - Use five speculative tokens, matching the trained V4.1 block size.
   - Begin with adaptive verification disabled on SM121 until the pinned sparse-MLA stack proves padded/variable graph shapes safe.

5. **CUDA graphs**
   - Capture only after all runtime JIT kernels are prebuilt/warmed.
   - Record exact capture sizes and verify no decode batch is silently padded into an unsupported sparse-MLA shape.

6. **Context scaling**
   - Validate 64K, 128K, 300K, then longer contexts.
   - Record KV token capacity, graph pool, peak unified memory, host headroom and per-rank model residency.

7. **Vision/tools**
   - Text correctness first, then enable the vision encoder and V4.1 tool/reasoning parsers.
   - Run at least one image request and one complete tool-call round trip.

## Engram strategy

Engram is a separate storage problem from EXL3 expert execution. For the first distributed boot, use the already-proven Spark-compatible disk/node-local Engram path if the pinned vLLM image still cannot hold Engram safely in unified memory.

After EXL3 expert residency is measured, retry with resident Engram. The goal is to use the memory saved by EXL3 to remove or reduce disk-backed Engram without sacrificing KV/cache headroom.

Do not claim resident Engram until all four ranks report safe peak unified-memory headroom during long-context prefill and DSpark decode.

## DeepSelect

DeepSelect is relevant to V4.1's Lightning Indexer TopK, but it is **not** a first-boot dependency. Its current published build targets SM100/SM103, not SM121. The DGX Spark baseline should first use the best working vLLM SM12x TopK path.

A later SM121 DeepSelect experiment must compare against that optimized baseline rather than against `torch.topk`.

## DeepJIT

DeepJIT is a promising future backend for shape-specialized EXL3 kernels, but it is not required to boot V4.1. DeepGEMM already embeds DeepJIT in the V4.1 runtime stack.

If `vllm-exl3` adopts DeepJIT later:

- compile before CUDA graph capture;
- cache CUBINs by EXL3 format + K + hidden + intermediate + SM architecture + plugin ABI;
- allow a shared read-only cache across Spark nodes;
- never trigger an unbounded parallel NVCC build during model startup;
- keep the AOT/native backend available as a fallback.

## Qualification evidence to retain

For every meaningful A/B, capture:

- vLLM image digest and source commit;
- `vllm-exl3` commit;
- ExLlamaV3 revision;
- native p2b ABI;
- CUDA/Torch/FlashInfer/DeepGEMM versions;
- TP/EP topology and rank-to-expert ownership;
- EXL3 K per layer;
- actual expert backend dispatch, not just requested backend;
- DSpark proposed/accepted tokens and acceptance length;
- TTFT, TPOT, output tok/s and aggregate tok/s;
- peak allocated/reserved/unified memory per rank;
- KV token capacity;
- Engram placement;
- CUDA graph mode/capture sizes;
- GB10 clocks/power during the run.

A skipped GPU test is not a hardware qualification pass.
