# Host-memory offload compatibility boundary

There are now **two different host-memory paths** and they must not be conflated:

1. **CPU compute:** routed experts live in host RAM and execute on the CPU.
2. **UVA zero-copy:** routed experts live in pinned host RAM, are exposed as mapped CUDA views, and the existing GPU EXL3 kernels dereference them across PCIe.

`vllm-exl3` still does **not** implement a CPU-compute backend for routed EXL3 experts. It now has an opt-in placement guard for the second path so a vLLM UVA experiment can fail closed instead of silently running a mixed or ordinary-CPU placement.

## Why UVA is interesting for DeepSeek-V4.1

Current upstream vLLM has two features that materially change the fastest path for a small-GPU / large-host-RAM V4.1 machine:

- generic selective weight offload through `--offload-backend uva`, `--cpu-offload-gb` and `--cpu-offload-params`;
- a DeepSeek-V4.1-aware `EngramConfig` whose `cpu_offload` mode keeps the huge FP8 Engram shard in pinned CPU memory and reads it through UVA.

The generic offloader leaves mapped parameters as accelerator tensors and marks them with `_vllm_is_uva_offloaded`. That shape is potentially compatible with EXL3's packed CUDA execution because the EXL3 handles and pointer tables are constructed after model placement and can point at the mapped accelerator views.

That is a hypothesis requiring a real GPU parity/performance run, not a published throughput claim.

## Experimental EXL3 UVA contract

Set:

```bash
export VLLM_EXL3_REQUIRE_UVA_EXPERTS=1
```

and configure vLLM to offload all six large EXL3 expert payload segments:

```text
w13_trellis
w13_suh
w13_svh
w2_trellis
w2_suh
w2_svh
```

For current vLLM CLI syntax, the intended shape is:

```bash
--offload-backend uva \
--cpu-offload-gb <enough-for-the-packed-expert-payload> \
--cpu-offload-params \
  w13_trellis w13_suh w13_svh \
  w2_trellis w2_suh w2_svh
```

DeepSeek-V4.1 Engram can be requested independently with:

```bash
--engram-config '{"cpu_offload": true}'
```

The plugin's guard validates the expert payload during `process_weights_after_loading` before EXL3 handles/pointer tables are accepted for the qualification path. It rejects:

- ordinary CPU tensors from vLLM's non-UVA fallback;
- partial UVA placement caused by an insufficient offload budget;
- a run where UVA was required but none of the packed expert parameters carry vLLM's mapped-offload marker.

The codebook-marker tensors are tiny and intentionally are not required to live in host RAM.

Inspect the process policy with:

```python
import json
import vllm_exl3

vllm_exl3.register()
print(json.dumps(vllm_exl3.runtime_diagnostics()["uva_expert_offload"], indent=2))
```

A successful placement guard is only a **placement gate**. It does not prove that every ExLlamaV3/native kernel has acceptable performance over PCIe, that CUDA graph capture is safe, or that the model output matches the source runtime.

## Qualification order for a 16 GB Blackwell + 1 TiB host

This is now the preferred first experiment because it preserves vLLM's already-working V4.1 model graph and avoids porting CED/Engram solely to reach the host-memory topology:

1. use a current V4.1-capable vLLM revision with `sm_120` support;
2. enable DeepSeek-V4.1 Engram UVA offload;
3. selectively UVA-offload the six EXL3 expert payload parameters;
4. enable `VLLM_EXL3_REQUIRE_UVA_EXPERTS=1` so partial/fallback placement is fatal;
5. start text-only, batch 1, short context, DSpark off and eager mode;
6. prove greedy output/logit parity against the known-good source harness;
7. measure PCIe reads, GPU utilization, pinned-host footprint and tok/s;
8. only then enable DSpark and longer context.

If the mapped EXL3 kernels are correct but too PCIe-bound, the next alternatives are:

- reuse ExLlamaV3's CPU-compute MoE path after a forward-correct V4.1 standalone architecture exists; or
- implement a dedicated CPU expert executor in `vllm-exl3` / vLLM.

## Upstream ExLlamaV3 CPU-compute path

Current upstream ExLlamaV3 also has an experimental CPU-MoE path. That path actually computes experts on CPU and currently requires:

- `mul1` codebook experts;
- K <= 8;
- uniform expert-bias presence for an eligible layer;
- a model architecture that ExLlamaV3 can instantiate.

Those prerequisites are separate from the vLLM UVA path. In particular, the vLLM UVA experiment can be useful even when standalone ExLlamaV3 still lacks the V4.1 graph.

Use the policy helper to inspect the external CPU-compute route:

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

`plan.vllm_exl3_execution_available` remains intentionally `False`: that helper describes ExLlamaV3's external CPU-compute route, not a hidden vLLM backend.

## Current V4.1 architecture boundary

Current upstream ExLlamaV3 still does not provide a forward-correct standalone `DeepseekV41ForCausalLM` loader at the revision used by the Spark recipe. The existing `DeepseekV4ForCausalLM` class is not a safe alias because V4.1 has different compression/source relationships and Engram/CED semantics.

That standalone port remains useful for CPU-compute experiments, but it is no longer a prerequisite for trying the faster-to-integrate **vLLM V4.1 + Engram UVA + EXL3 expert UVA** path.

## Upstream implementation references

The experiment is based on current public vLLM interfaces, not undocumented assumptions:

- `vllm/config/offload.py` — UVA zero-copy weight offload and selective parameter segments;
- `vllm/model_executor/offloader/uva.py` — pinned-host parameters exposed as accelerator views with `_vllm_is_uva_offloaded`;
- `vllm/config/engram.py` — DeepSeek-V4.1-aware Engram CPU-offload configuration;
- `vllm/models/deepseek_v4_1/common/engram.py` — pinned-host FP8 Engram tables read through UVA.

Pin the exact vLLM revision used for every result because these interfaces are moving quickly.
