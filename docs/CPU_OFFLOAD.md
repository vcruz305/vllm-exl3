# CPU-offload compatibility boundary

`vllm-exl3` does **not** currently implement a host-resident CPU executor for routed EXL3 experts.

The DeepSeek-V4.1 integration in this project is intentionally split this way:

- vLLM owns the `DeepseekV41ForCausalLM` model graph, CED/CSA2 attention, Engram, routing and DSpark;
- `vllm-exl3` owns the EXL3 storage/execution integration used by supported vLLM routed-expert layers;
- CPU-MoE is not silently delegated to ExLlamaV3 by this plugin.

This distinction matters for machines with a small GPU and very large host RAM. A 16 GB GPU plus hundreds of GiB of DDR5 can run a topology where experts live and execute on CPU only if the chosen runtime explicitly supports that placement.

## Upstream ExLlamaV3 CPU-MoE

Current upstream ExLlamaV3 has an experimental CPU-MoE path. Its current format contract requires:

- `mul1` codebook experts;
- K <= 8;
- uniform expert-bias presence for an eligible layer;
- a model architecture that ExLlamaV3 can instantiate.

Those prerequisites are separate from `vllm-exl3`'s GPU/native kernels. In particular, an MCG pack is not made CPU-offload-compatible just because the plugin can execute MCG on a GPU path.

Use the policy helper to make this boundary explicit in tooling:

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

`plan.vllm_exl3_execution_available` is intentionally `False`. The helper is a preflight/planning API, not a backend switch.

## DeepSeek-V4.1

The current upstream ExLlamaV3 architecture registry does not provide a forward-correct standalone `DeepseekV41ForCausalLM` loader at the pinned revision used by the Spark recipe. The existing `DeepseekV4ForCausalLM` class is not a safe alias: V4.1 has a different compression schedule/source relationship and Engram/CED semantics.

For a CPU-heavy V4.1 deployment, the recommended development order is:

1. complete and qualify a standalone V4.1 ExLlamaV3 forward port;
2. inspect the actual EXL3 codebook/K metadata;
3. if the pack is `mul1` and otherwise eligible, reuse upstream ExLlamaV3 CPU-MoE first;
4. compare against a source/reference implementation before publishing performance or quality claims;
5. only build a separate vLLM CPU expert backend if the ExLlamaV3 route cannot meet the deployment requirement.

This keeps model-graph correctness, packed-weight execution and CPU placement as separately testable gates.
