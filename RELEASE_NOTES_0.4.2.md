# vllm-exl3 0.4.2

Serving EXL3 trellis-quantized checkpoints under vLLM. This release closes two defects that changed served
output, adds native ExLlamaV3 pack support, and moves the project to AGPL-3.0-only.

## Two correctness fixes

**Fat-expert prefill was numerically wrong for packs with distinct gate/up rotations.** `apply_exl3_batched_fat`
handed column slices of a shared scratch buffer to kernels that index contiguous row-major operands, so any
prefill routing more than 256 tokens to one expert produced wrong hidden states. Short prompts never reach that
path, so generation stayed fluent while long-prompt output was wrong: prompt log-likelihood over a fixed
6,000-token corpus read mean NLL 4.21 against the exllamav3 reference 0.94 on the same weights, with top-1
agreement collapsing from 74.7% to 30.8%. Packs whose gate and up projections share one `suh` were unaffected.
The branch now runs on contiguous fp32 temporaries. Regression test: `tests/test_fat_distinct_suh.py`.
Workaround for older builds: `VLLM_EXL3_FAT_THRESHOLD=1000000000`.

**Dense EXL3 calls wedged the vLLM nightly V2 model runner.** exllamav3 dispatches dense calls by row count:
up to 2 rows use the non-cooperative GEMV, 3 to 144 rows use the cooperative trellis GEMM, and above 144 rows it
reconstructs the weight and runs hgemm. The cooperative GEMM is the kernel that wedges: EngineCore pinned at
100% CPU with the GPU busy at idle power, deterministically on 33 to 144-token prefills and on MTP evaluation
after 30 to 60 minutes. Dense calls with 17 to 144 rows now take the reconstruct path, which is exact; rows up
to 16 keep exllamav3's own dispatch. `VLLM_EXL3_RECONSTRUCT_MIN_ROWS` moves the threshold and
`VLLM_EXL3_COOP_GEMM=1` restores the old dispatch for A/B work.

Worth knowing what that fix costs and why it is still the right trade. On a real K=5 layer the reconstruct path
runs about 0.64 to 0.72 ms per call regardless of row count, against 0.088 ms for the cooperative GEMM at 17
rows and 0.433 ms at 144, so it is never the faster path in that band. It is also the only path that runs at
all: with `VLLM_EXL3_COOP_GEMM=1` the server does not finish starting, hanging in prefill CUDA graph capture
until killed.

## Added

Native ExLlamaV3 pack support, including row-wise n-gram embedding tables through `Exl3EmbeddingMethod`, the
`ngram_embedding` config spec, `mul1` codebooks alongside `mcg`, 128-padded dense geometry, a per-expert
routed-experts loader, and opaque custom-op registration so the paths trace under `torch.compile(fullgraph=True)`.
Tooling ships in `tools/patch_vllm_qwen4_exp/`, `tools/exl3_pack_tools/` and `tools/verify_native_pack/`.
`runtime_diagnostics()` reports the effective serving policy, and `runtime_policy.py` and `prefill_policy.py`
add TP1-oriented policy helpers with per-bit overrides. Grouped prefill planning is present and off by default
(`VLLM_EXL3_GROUPED_PREFILL`).

## Changed

Gate/up input rotations are compared once at load and cached on the expert pack rather than per expert per
prefill chunk, removing a CUDA `torch.equal` from the fat-prefill hot loop. The native fused-MoE decode-row cap
measured on GB10 ships as an opt-in.

## Licensing

The project moves forward under AGPL-3.0-only. The historical Apache-2.0 text is retained in
`LICENSE.APACHE-2.0`, and third-party material keeps its own notices.

## Validation

One GB10, TP=1, vLLM 0.28.1rc1.dev324 unless stated.

| check | result |
|---|---|
| CPU test suite | 329 passed, 4 failed; the same 4 fail on the previous revision (fixture drift, not product) |
| Qwen corpus NLL vs exllamav3 reference | 0.9436 against 0.9422, top-1 74.51% against 74.70% |
| Qwen NLL, this release vs previous revision | +0.0005 nats scored in one forward, +0.003 chunked, numerically neutral |
| GLM-5.3-Flash K2/K3-mix, KV pinned at 5.5 GiB | decode p50 17.0 tok/s on both revisions; aggregate 37.9 both at 4 streams, 41.1 against 40.9 at 8 |
| DeepSeek-V4-Flash-Vision ablit | serves coherently, 62.0 tok/s aggregate over 8 streams, speculative acceptance 2.72 |
| Qwen3.8-Flash-Next single stream | 27.9 tok/s greedy decode, 0.303 s TTFT, 979 tok/s prefill on 1,218 tokens, no draft head |
| MTP k=2 batch serving | 44.9, 73.5 and 129.2 tok/s aggregate at 1, 8 and 32 streams |
| Wedge stress, 4 workers, MTP k=2 | clean for 45 minutes, 1,085 requests, against a wedge in 161 s before the fix |

Score the corpus in a single forward when comparing against the exllamav3 reference. The scorer's default
4,096-token chunking splits the corpus and scores the second piece without preceding context, which costs
0.0925 nats on a healthy build and looks like a regression.

## Known issues

Routed-expert calls with small row counts still dispatch the cooperative `exl3_moe` kernel, the same family
whose dense counterpart wedged the V2 runner. It did not fire in any run behind this release, including the
45-minute stress and two full 120-item evaluation suites, but the dense fix does not cover that path.
`VLLM_EXL3_MOE_KERNEL` selects the backend.

At 8 concurrent sequences against a small KV pool the scheduler can hold one request back for roughly 16
seconds while the others decode. It is pool-size dependent and reproduces identically on revisions before and
after this release. Pin the pool with `--kv-cache-memory-bytes` when comparing revisions.

Tensor-parallel size 1 only for the native n-gram path.
