# Changelog

## Unreleased

### Added

- Per-layer chunk rotation for the aligned MoE TP split (`VLLM_EXL3_MOE_TP_ROTATE=1`, #40). Without
  it the uneven split's larger chunks land on the same ranks in every layer — 70.1 vs 56.8 GiB of
  weights per rank on DSV4.1-Flash, where the heavy ranks cap the KV cache for the whole TP group.
  With rotation, rank `r` of layer `L` takes chunk `(r + L) % tp`. The MoE all-reduce sums rank
  partials, so outputs are unchanged and only ownership moves. The contributor measures 63.4 GiB on
  every rank and KV cache raised from 8 to 17 GiB per rank (2.45M to 5.22M tokens). Off by default.

## 0.5.0 (2026-09-25)

Everything merged since 0.4.2, plus the vLLM 0.30.0 compatibility audit. The MoE kernel work in this
release is **not on by default**: no serving path selects it unless a recipe asks for it. Numbers
below are the contributors' measurements on their own hardware unless a section says otherwise.

### Added

**Routed-expert MoE kernels**

- Multi-K fused MoE (`p2b_fused_moe_mk`): per-expert K tables are read on device, so a mixed-K layer
  takes one cooperative launch instead of one launch per expert (#31).
- Fixed-shape padded MoE (`p2b_fused_moe_padded`): fixed shapes and no host sync, so the MoE can be
  captured into a CUDA graph. The original single cooperative launch cannot be captured at all
  (`cudaLaunchCooperativeKernel` is illegal under default capture mode, and relaxed mode records
  nothing), so it is split into nine ordinary same-stream stage kernels. A silent bug found while
  splitting is fixed with it: the prologue's `accum` zeroing had replicated into all nine stages, so
  stage 8 zeroed `accum` and wrote the zeros back, making the output independent of the inputs
  (#33).
- Grouped padded MoE: live slots are grouped by expert, up to 16 per group, so each trellis tile is
  decoded once per expert rather than once per slot. `P2B_GROUPED=0` reverts (#39).
- Opt-in decode routing through exllamav3's cooperative `exl3_moe` kernel for decode-shaped batches
  on exllamav3 >= 1.5.0 (#27).

**TP geometry**

- Hadamard-aligned uneven TP split for routed experts (`VLLM_EXL3_MOE_TP_ALIGN=128`). `2304 / 4 =
  576` is not 128-aligned, and EXL3's Hadamard block requires it, so an even TP4 split cuts a block
  and decodes every shard against the wrong transform. The router's columns are now cut in whole
  128-blocks — 640 / 640 / 512 / 512 — from the pack that already exists, so MoE-TP4 does not need a
  re-quantized 640-wide pack. Off unless the variable is set, and unset behaviour is unchanged. The
  contributor measures +16% single-stream decode on four-Spark DSV4.1-Flash TP4 (#36).

**Engine, load path and host memory**

- DeepSeek-V4.1 TP4+EP4 EXL3 compatibility (#8), per-expert mixed-K routed weights (#10) with the
  qualification hardening that followed (#11), physical-trellis K derivation (#12), V4.1 cache math
  and a TP-aware mixed-K prescan (#13), FP8 `weight_block_size` propagation with EP-aware expert
  loading (#16), and runtime plugin state for the SAGE TP2 pair — tensor metadata, mixed-K store,
  draft/MTP plan tolerance (#30).
- Routed-expert trellis arenas placed in pinned host memory for UVA runs (#26), EXL3 trellises copied
  directly into their final arena slots (#21), and a UMA-safe load path that direct-fills the arena
  and applies MADV after the H2D (#14).
- GB10 load path: pinned bounce buffers, `POPULATE_READ`, batched markers and a prescan cache (#28).
  Reported: the four-rank load drops from about 8.7 to about 5 minutes and fill from 7.5 to 1.6
  minutes, with decode unchanged.
- The bounded SAGE NVMe and Engram cache components are shared rather than duplicated (#19), and a
  gapped-safetensors repair rebuilds a compressed shard without changing tensor bytes (#18).

- Unsharded n-gram tables. Packs from exllamav3 1.5.0 onward ship the table as one `<root>.trellis` tensor instead of `shard_<i>.trellis` pieces; the scan tool now recognises that layout (a `trellis` whose root also carries `head_offsets`), the config tool emits `ngram_embedding.sharded: false` with `num_shards: 1`, and the loader registers the matching `trellis` parameter. No pack rewrite is needed any more for turboderp's `4.05bpw_h6_ng6` revision.
- `VLLM_EXL3_NGRAM_TABLE=disk`. The packed n-gram table stays in the checkpoint: the loader keeps vLLM's memory-mapped safetensors views instead of copying into a resident int16 tensor, and each lookup gathers its unique rows on the host, uploads them, and decodes on the device. The table then costs page cache rather than 32 to 36 GiB of device memory. The host gather is a synchronization point, so this mode needs `--compilation-config '{"cudagraph_mode": "PIECEWISE", "splitting_ops": [...attention ops..., "vllm::exl3_ngram_lookup_out"]}'`; the loader refuses FULL graph modes with that message. Default stays `resident`. Tests: `tests/test_ngram_layouts.py` (CPU layouts plus in-image coverage of the registered `vllm::exl3_ngram_lookup*` ops, resident/unsharded/disk lookup parity and the FULL-graph refusal), `tests/test_pack_tools_ngram_fixture.py` (synthetic MoE pack through the scan and config tools).
- The caller-allocated lookup op (`vllm::exl3_ngram_lookup_out`) is reachable only with `VLLM_EXL3_NGRAM_TABLE=disk`; `resident` serving keeps the existing returning lookup op.

- exllamav3 1.5.0 support for the fused routed-expert launch. 1.5.0 appended five positional arguments to `exl3_moe` (`output_scratch`, `fused_base`, `count_lo`, `count_hi`, `m_tile`) for its deterministic-accumulation and row-tile modes; the plugin now reads the binding's arity from its pybind signature and, on 1.5.0, passes the values that reproduce the 1.4.x all-fused atomic launch (`None, None, 1, <temp rows>, 16`). 1.4.x bindings are called exactly as before. `runtime_diagnostics()` records the detected arity as `exllamav3_exl3_moe_arity`. Measured on one GB10 with Qwen3.8-Flash-Next 3.05 bpw at the recipe's envelope config: 52.05 tok/s at MTP k=3 on 1.5.0 against 52.22 on 1.4.7, 28.54 against 27.77 without a draft, prefill unchanged. Kernel version is not a speed lever on this hardware; the change is about running on current upstream. Tests: `tests/test_exl3_moe_arity.py`.

- `tools/gb10_exl3_moe_parity.py`: on-hardware probe for the same launch — one launch per per-layer K against the native `exl3_gemv` map, plus the check that a 1.5.0 binding rejects the 1.4.x arity. `--dry-run` exercises the argument assembly on CPU.

### Fixed

- The native-MoE dispatch gate treated an *unset* codebook flag as a mismatch, so `main` failed 103
  tests (100 in `test_routing_parity.py`, 3 in `test_activation_parity.py`). The default now resolves
  to the MCG tuple, and a layer carrying a genuinely non-MCG codebook still fails closed (#34).
- The expert codebook became a build-time parameter. A pack carrying the **mul1** codebook
  (`0x83DCD12D`) decoded by an MCG-built kernel does not crash — it produces a plausible-looking
  wrong vector, which is worse. Build with `-DP2B_CB=2` for mul1 packs; all three codebook checks
  fail closed (#32).
- The loader's pre-read EP weight filter was sized once from the main stack's 384 experts and applied
  to the DSpark draft's 128, so EP ranks 1 to 3 silently loaded none of the draft experts they owned.
  The contributor measures +9% to +20% decode on the four-Spark DSV4.1 configuration, with acceptance
  about 1.7 to 2.0 (#35).
- `_narrow_tp` fell back to the even split when a requested 128-block alignment could not be
  satisfied — the silent wrong-transform case the aligned path exists to remove. It now raises at
  load time, with tests (#36).
- Disabled-CPU symbol behaviour preserved on arm (#20); the redundant CUDA sync after the blocking
  trellis `copy_` dropped (#29); the aarch64 patch verified against exllamav3 v1.5.0 in CI (#24).

### Compatibility

- **vLLM v0.30.0 audit** in [docs/VLLM_COMPATIBILITY.md](docs/VLLM_COMPATIBILITY.md), re-runnable
  with `tools/check_vllm_compat.py <tree>`. Verified present: the `vllm.general_plugins`
  entry-point group, every `from vllm...` import in `src/` and `tools/`, and `FusedMoEMethodBase`'s
  abstract surface (`create_weights`, `get_fused_moe_quant_config`), which `Exl3MoEMethod`
  satisfies.
- The Qwen4Exp model tree moved from `vllm/model_executor/models/qwen4_exp/` to `vllm/models/`. The
  patch tools already targeted the current layout; their README did not, and now does.
- PLE n-gram tables: current vLLM builds the table with a quant config itself, so the PLE half of
  `patch_vllm_qwen4_ple.py` is obsolete and now reports itself as such instead of failing its anchor
  check. The EXL3 path itself does **not** carry over, because the layer selects its storage format
  through `Qwen4ExpPLEEmbeddingMethod.from_quant_config`, which raises for any config that is not
  ModelOpt or `Fp8Config`. An adapter is required; the compatibility page records the two options.
  DeepSeek-V4.1 serving does not touch this code.
- Corrected the Engram path `docs/CPU_OFFLOAD.md` cites:
  `vllm/models/deepseek_v4_1/common/engram.py` is now `.../deepseek_v41/...`.

### Qualification boundary

Nothing here changes a default serving path: the new MoE kernels are exported, not selected, and
`apply_exl3_fused_moe` dispatches exactly as it did. Enabling them, and building with `-DP2B_CB=2`
for a **mul1** pack, is a recipe decision. The speed figures above are the contributors' measurements
on their own GB10 configurations and were not re-measured for this tag. The kernel batch needs a
GB10 qualification run — parity against the Python `LinearEXL3` reference, then the standard gate —
before a recipe adopts it.

## 0.4.2 (2026-09-09)

### Fixed

- Dense EXL3 calls no longer launch exllamav3's cooperative trellis GEMM. Rows 17 to 144 take exllamav3's reconstruct+hgemm path (exact); rows up to 16 keep exllamav3's own dispatch. On the vLLM nightly V2 model runner the cooperative GEMM wedged the engine; a 4-worker MTP stress that wedged within 3 minutes ran clean for 45 minutes once the routing was in place, with decode speed unchanged. `VLLM_EXL3_COOP_GEMM=1` restores the old dispatch; `VLLM_EXL3_RECONSTRUCT_MIN_ROWS` moves the reconstruct threshold.
- Fat-expert prefill path: the branch for packs whose gate and up projections carry distinct `suh` rotations handed column slices of the shared `gate_up` scratch buffer to kernels requiring contiguous row-major operands. The branch now runs on contiguous fp32 temporaries. Regression test: `tests/test_fat_distinct_suh.py`.

### Added

- `runtime_diagnostics()` exposes the effective EXL3 serving policy in a JSON-friendly form: selected MoE backend, native extension availability and ABI, per-bit native decode-row caps, fused scratch row capacity, fat-expert threshold, fat-kernel availability, and speculative schedule. This gives recipes a machine-checkable record of what actually loaded rather than relying on requested environment variables alone.
- `src/vllm_exl3/runtime_policy.py` adds independently implemented TP1-oriented policy helpers. Per-bit overrides (`VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K2/K3/K4`) make it possible to A/B K2 separately from K3/K4 instead of forcing one global native-row threshold across different trellis costs. `VLLM_EXL3_FUSED_TEMP_ROWS` exposes the requested fused scratch capacity while preserving the historical 2048-row default.
- `docs/provenance.md` records exact source/design provenance and establishes commit-message labels for copied/derived work, adapted designs, and independent implementations.
- `VLLM_EXL3_PREFILL_SYNC=<max_rows>`: synchronizes the device before each EXL3 dense or routed-expert call whose row count is between 2 and max_rows (never during CUDA graph capture, never for single-row decode). Workaround for a vLLM nightly V2 model runner wedge on Qwen3.8-Flash-Next where prefills of roughly 33 to 144 tokens never complete; decode speed is unchanged. Recommended value 256 on that runner.
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

### Changed

- Native fused-MoE decode-row cap measured on GB10 (K2: 8 rows, K3/K4: 1) ships as an opt-in (`VLLM_EXL3_NATIVE_MOE_MEASURED_CAP=1`, or `VLLM_EXL3_NATIVE_MOE_MAX_ROWS=<n>`); the default keeps the dispatch contract of up to 8 rows. Receipt: `tools/receipts/ab_moe_gb10.json`.
- Gate/up input rotations are compared once at load and cached on the expert pack (`_exl3_gate_up_shared_suh`) instead of per expert per prefill chunk, removing a CUDA `torch.equal` from the fat-prefill hot loop. Behaviour is unchanged; a fallback recomputes the flag for callers that bypass `build_exl3_fused_state`.

### Licensing

- vllm-exl3 moves forward under **AGPL-3.0-only** so improvements to modified network-served versions remain available to their users. The prior Apache-2.0 license text is retained in `LICENSE.APACHE-2.0`; all existing MIT/Apache third-party notices remain in `THIRD_PARTY_NOTICES.md` and `NOTICE`.
- The newer TP1 runtime-policy helpers were independently written around this project's APIs. Recent public MiaAI-Lab GLM serving work is credited as relevant design prior art; no post-relicense scheduler or dense-FP8 source was copied into these helpers.

### Known issues

- Routed-expert calls with small row counts still dispatch exllamav3's cooperative `exl3_moe` kernel, the same kernel family whose dense counterpart wedged the vLLM nightly V2 model runner (see the dense routing fix above). It did not fire in any run behind this release, including a 45-minute 4-worker MTP stress and two full 120-item evaluation suites, but the dense fix does not cover this path. `VLLM_EXL3_MOE_KERNEL` selects the backend if you need to move off it.
- At 8 concurrent sequences against a small KV pool the scheduler can hold one request back for roughly 16 seconds while the others decode. This is pool-size dependent, reproduces identically on builds before and after this release, and is listed here so it is not mistaken for a regression: pin the pool with `--kv-cache-memory-bytes` when comparing revisions.

### Validation

- One GB10, TP=1. GLM-5.3-Flash K2/K3-mix before and after this release with the KV pool pinned to 5.5 GiB: decode p50 17.0 tok/s both, aggregate 37.9 tok/s both at 4 concurrent and 41.1 against 40.9 at 8, so the release is performance-neutral there. DeepSeek-V4-Flash-Vision ablit serves coherently at 62.0 tok/s aggregate over 8 streams with speculative acceptance 2.72. Qwen3.8-Flash-Next, the only pack that exercises the distinct-`suh` fat-expert branch, scores corpus mean NLL 0.9436 against the exllamav3 reference 0.9422 with top-1 agreement 74.51% against 74.70%, and sits +0.0005 nats from the previous revision on an identical procedure. It serves at 27.9 tok/s greedy decode with 0.303 s TTFT and 979 tok/s prefill on a 1,218-token prompt without a draft head. Score the corpus in one forward: the scorer's default 4,096-token chunking scores the tail without preceding context and costs 0.09 nats on a healthy build.

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
