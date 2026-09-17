# SAGE 3.30 two-Spark runtime contribution

This directory supplies the working runtime source behind the September 12–14
two-Spark experiments: the first bounded loader, grouped and mixed-K execution,
asynchronous expert reads, Engram row caching, and the later native service and
CUDA graph fixes. It is an experimental source handoff for review and porting.
It does not change the installed plugin or qualify current main for this mode.

The campaign is now closed. [FINAL_NOTES.md](FINAL_NOTES.md) records the last
qualified result, the rejected follow-up experiments, and the limits that remain.

The earlier component PR supplied reusable pieces without the loader and kernel
integration. This bundle includes those dependencies and the historical versions
needed to inspect the progression. The existing safetensors repair remains a
separate contribution. Related: [components #19](https://github.com/vcruz305/vllm-exl3/pull/19),
[repair #18](https://github.com/vcruz305/vllm-exl3/pull/18), and
[initial recipe findings #10](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/pull/10).

## Review the fixes

| Problem observed | Implementation and behavior | Historical evidence |
|---|---|---|
| Resident experts exceeded the two-node memory budget | `runtime/vllm_disk.py`, `sage_plugin.py`, `native_stream.py`: stream native tensors, authorize expert metadata placeholders, require every owned tensor, admit an 80 GiB arena per rank | Full-model quality/decode runs with the original 3.30 checkpoint |
| Separate launches for many routed experts | `grouped_plan.py`, `grouped_cache.py`: group physical `(w1,w3,w2)` K triplets, preserve all routes and whole-expert ownership | Matched 32K TTFT: 181.65 → 172.47 s |
| Serial demand reads blocked execution | `async_store.py`, `prefetch_cache.py`: two bounded record reads, hash before exposure, leases/cancellation and resident-first ordering | 172.47 → 157.25 s; two active reads maximum |
| Repeated Engram row reads | `engram_rows.py`, `cached_engram_table.py`, `history/engram-image-r1/engram_prefetch.py`: deduplicate/cache raw rows and prefetch before consumption | 157.25 → 137.99 s; original FP8 weights/scales |
| Grouping still launched each K triplet separately | `native/mixed_k/`, `heterogeneous_cache.py`: fused packed dispatch over different physical K values | K2–K8 numerical checks; 139.19 s TTFT, so no TTFT improvement at this step |
| Small prefill chunks repeatedly incurred dispatch overhead | `history/decoder-tail-image-r4/` plus CLI chunk size 512, tail disabled | 139.19 → 73.46 s TTFT; same 32,631-token prompt |
| CPU-built route descriptors prevented graph replay | `gpu_decode_plan.py`, `native/dynamic/`: device-built routes and physical resident-grid partitioning | Changing routes/expert counts, K2–K8 comparisons |
| Python miss servicing stalled on the GIL | `native/service/`, `native_shared_cache.py`: native coordinator and two readers own read/hash/MUL1 validation/eviction/copy/descriptors/acknowledgment | 152 comparisons per rank and a six-second held-GIL miss test |
| Unused reserved KV page zero could contain NaNs | `null_kv_page.py`, `vllm_overlay/cudagraph_utils.py`: zero only the reserved page segments around dummy capture | Page-zero NaNs removed; active-page bytes preserved |
| Disabling warmup prevented graph capture; the active runner was V2 | `bootstrap/sitecustomize.py`, `graph_runtime.py`, `vllm_overlay/`: bounded C1 capture in the actual V2 runner, stable buffers, ownership fences and graph retirement | 2,513 full graph replays per rank; full-model quality passed |
| Engram callbacks were unsuitable for full graph replay | `native/engram/`, `native_rows.py`, `graph_rows.py`: native row service, original row IDs/ownership and GPU dequantization | 32 bit-exact comparisons per rank |

The final `runtime/` is the native graph candidate. `history/` contains earlier
image overlays, not alternative modules to add to `PYTHONPATH` simultaneously.
The original CUDA arena path remains in `expert_cache.py`; the final cache derives
through `shared_mapped_cache.py` and `native_shared_cache.py`. The native service
producer performs no Python callbacks or CUDA submissions.

## Exact measured configuration

| Item | Recorded value |
|---|---|
| Model | `vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw` |
| Model revision | `e831e9e4d6bfeafa6d630848296417b1393404a3` |
| Quantization | Original routed MUL1 K2–K8 bytes; native MXFP8/FP8 non-routed tensors; no requantization |
| Hardware/topology | Two DGX Sparks, GB10 SM121, TP2 + EP2, 192 whole experts/rank, 40 layers |
| vLLM | Dedicated V4.1 image, `0.1.dev20904+g179dd0fa9` |
| Torch / CUDA | `2.13.0+cu130` / 13.0 |
| Historical plugin | `8f4517e80416466fa4a3ad2eb28685021d39e95f` |
| ExLlamaV3 | `be57335b087e4f001c5caae061544df3c06ba01e` |
| Expert / KV budgets | 80 GiB arena + 4 GiB KV per rank |
| Container / host limits | 104 GiB memory and memory+swap ceiling; zero host swap; at least 8 GiB host reserve |
| Context / concurrency | 32,768 capacity, one sequence; decode prompts were short |
| Final prefill / graphs | 512 tokens; `FULL_DECODE_ONLY`, capture sizes `[1]`, compilation mode 0 |
| Attention / MoE | Native V4.1 FlashInfer SM120 attention; bundled packed heterogeneous EXL3 kernels |
| Engram | Local disk; bounded raw-row cache/prefetch; native row service in final graph run |
| Speculation / prefix reuse / tail | Disabled / disabled / disabled |

`--kernel-config '{"moe_backend":"triton"}'` was a native-runtime CLI setting;
routed EXL3 execution used the bundled custom kernels. It must not be reported
as a Triton EXL3 throughput result.

## Measurements and limits

One fresh deployment per mode, identical request order, original checkpoint,
80 GiB cache and a 32,631-token context probe:

| Stage | Coding tokens/s | Writing tokens/s | Context TTFT (s) |
|---|---:|---:|---:|
| Original eager, chunk 128 | 2.731 | 4.893 | 181.65 |
| Grouped K triplets | 2.657 | 5.019 | 172.47 |
| Bounded async expert reads | 2.545 | 4.896 | 157.25 |
| Engram row cache/prefetch | 2.625 | 4.957 | 137.99 |
| Mixed-K fused, chunk 128 | 2.654 | 5.365 | 139.19 |
| Mixed-K fused, chunk 512, tail off | 2.629 | 5.243 | 73.46 |

These are sequential mechanism probes, not statistical confirmation. Decode
values here include the cache state from that workload order. The following
separate sustained-decode comparison ran eight quality requests as warmup and
then three alternating coding/writing pairs in each fresh deployment:

| Runtime | Median coding tokens/s | Median writing tokens/s |
|---|---:|---:|
| CUDA arena eager baseline | 4.400203 | 5.680964 |
| System arena eager | 3.967238 | 5.010387 |
| Native service + full decode graphs | 4.648543 | 6.117337 |

The final graph result improved these medians by 5.64% and 7.68% over the CUDA
arena baseline. The intervening system-arena result regressed; it is retained
here because component parity alone did not predict full-model speed. The final
run maintained zero swap and minimum available memory of 12.99/12.84 GiB.

Coding generated 135 tokens; writing was capped at 512. Decode TPS uses returned
token IDs and client arrival times: `(completion_tokens - 1) / (last - first)`.
This is not end-to-end request throughput. Three requests inside one deployment
are not three independent deployment confirmations. The earlier approximate
decoder-tail experiment is preserved for inspection but rejected by the final
bootstrap; its quality and semantics require a separate review.

Nothing here establishes 30 tokens/s, 600K context, DSpark acceleration,
concurrent serving, or TP4 qualification. The 32K prefill measurement belongs
to the earlier eager configuration; it is not a 32K graph-prefill measurement.

`evidence/results.json` contains allowlisted numeric records and hashes of their
original local artifacts. `evidence/decode-requests.json` contains the synthetic
decode requests. Original machine inventories and full raw logs are not included;
the recorded hashes identify those local artifacts, not public download URLs.

## Build and reproduction boundary

The retained historical images were local builds, not pullable registry releases.
The final rank-0 image ID was
`sha256:fda0fd530a93a347d2ed6fe5e54195f1379975f770c7a30d1939456925018b20`.
The parent system-cache image ID was
`sha256:6efa34f3a352b6d873e541558bfad63cdef1c0b727212c57e5ffc8378f5a51a7`.

To rebuild the extensions and overlay **with a retained compatible parent**:

```bash
python3 -B experiments/dsv41_runtime/run_cpu_tests.py
docker build --build-arg BASE_IMAGE="$RETAINED_V41_IMAGE" \
  -t sage330-runtime:review experiments/dsv41_runtime
```

The Dockerfile checks hashes of the preexisting V2 runner, graph utilities and
bootstrap before any replacement. It also requires the base's V4.1 `sm120_page`
helper and the recorded Torch/CUDA combination. Do not bypass a mismatch by
editing the hashes: port the changes to the new base, build on GB10, and repeat
the numerical, memory, quality and graph checks. These selected preimage checks
are compatibility guards, not a complete attestation of the base image.

The new Docker build wrapper has been checked on CPU only; its CUDA build and
full-model launch have not been rerun for this contribution. The native sources
and installed Python snapshots match the historical image/component receipts.
All four native outputs must be built locally; binary artifacts are not bundled.

For source extraction, `provenance.json` maps every copied file to its frozen
artifact and SHA256. The exporter verified all files in 29 contributing artifact
manifests. Only transitive native serving headers are included, with upstream
licenses retained. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The companion recipe contribution provides rank-package construction, runtime
configuration and launch settings. Package construction moves original tensor
bytes into serving files and does not produce new quants. Fresh current-main
integration, an independently buildable public base and new hardware qualification
remain required before treating this as a supported recipe.
