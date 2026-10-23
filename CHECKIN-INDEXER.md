# CHECKIN: MiniMax-M3 indexer MXFP8 retention (VLLM_EXL3_INDEXER_MXFP8)

Branch: `feat/minimax-indexer-mxfp8` (base 3f80e67), worktree
`lanes/plugin/wt-indexer-mxfp8`. NOT pushed.

## What changed

The MiniMax-M3 MSA lightning-indexer projections (`index_q`/`index_k`,
shards 3/4 of the fused `MinimaxM3QKVParallelLinearWithIndexer`) can now
stay in their native MXFP8 checkpoint form (F8_E4M3 payload + E8M0
block-32 scales) on device instead of being dequantized to BF16 at load,
behind opt-in env `VLLM_EXL3_INDEXER_MXFP8` (default OFF = today's BF16
behavior, byte-identical). When ON, their GEMM runs vLLM's own
`FlashInferCutlassMxfp8LinearKernel` — the sm_121 backend the image
already selects for the DSpark draft's ModelOpt MXFP8 layers — through
the kernel's real API (`process_weights_after_loading` for its F8_128x4
scale swizzle, `apply_weights` for the W8A8 GEMM). Output dtype/shape are
identical to the BF16 path (bf16 slices of the fused
`[q | k | v | index_q | index_k]` output). If the kernel is unavailable
at runtime, load fails closed with a clear error naming the switch —
never a silent BF16 fallback. q/k/v EXL3 paths are untouched.

### Files and evidence

- `src/vllm_exl3/exl3.py`
  - `:1896` `_indexer_mxfp8_enabled()` — env parsing, default OFF.
  - `:1932` `indexer_mxfp8_retain()` — bit-preserving retention (payload
    bitcast to `float8_e4m3fn`, scales stay uint8 row-major
    `[N, K/32]`, the layout vLLM's kernel load step expects).
  - `:1962` `indexer_mxfp8_kernel()` — fail-closed resolver: imports
    vLLM's kernel API, requires both `is_supported()` (platform) and
    `can_implement()` (shape) to admit the shard, else `RuntimeError`
    naming `VLLM_EXL3_INDEXER_MXFP8=1`.
  - `:2003` `make_indexer_mxfp8_shim()` — the parameter shim the
    kernel's process/apply read (`weight` fp8 `[N,K]`,
    `weight_scale` uint8, `params_dtype` bf16).
  - `:5805` create_weights: the BF16 staging union is gated on the
    switch — with the switch ON the indexer rows are never allocated
    (the staging weight param stays 0-row), so the load-time transient
    drops too.
  - `:6270-6274` marker checks skip retained-MXFP8 indexer shards
    (they are not EXL3 and carry no codebook markers).
  - `:6314` shard coalescing is disabled on this path (one kernel shim
    per shard must keep its slot).
  - `:6374-6377` LinearEXL3 build skips the two indexer shards (None
    placeholders, as the BF16 shards already had).
  - `:6441` `_process_indexer_shards()` — the load-end step. Default
    branch is the pre-switch dequant code moved verbatim (byte-identical
    staging); switch-on branch retains, binds the kernel, runs the
    kernel's own load step, frees the stash either way.
  - `:6560-6567` `_apply_impl`: the two indexer slices route through
    `kernel.apply_weights(shim, x_2d)` and are accumulated in fp32 like
    every other slice; fused output layout/dtype unchanged.
- `tools/gb10_minimax_m3_parity.py`
  - `:164-167` case G gates (`G_COS_MIN=0.9999`, `G_ABS_FACTOR=4`,
    `G_MS=(1,64)`).
  - `:1453` `_case_g()` — real layer-0 `index_q`/`index_k` range-read
    from the pinned Hub revision (same pinned fixture/shard identity as
    case E), through the plugin's production glue, vs the BF16-dequant
    reference computed from the same bytes; gates cosine >= 0.9999,
    max-abs within [1/4, 4] of the reference, 0 NaN, payload bits
    unchanged by the kernel's load step; a fail-closed resolver is a FAIL
    (recorded in the receipt), never a skip.
  - `:2101-2107` wired into `main()`'s verdict; `:1983-2012` CPU
    self-test bars covering the case G structure and the plugin-side
    fail-closed contract.
- `tests/test_indexer_mxfp8_retention.py` (new, 13 tests) and
  `tests/test_minimax_m3_parity_tool.py` (+3 case G tests).

## Test counts

- Before (3f80e67): 571 passed, 39 skipped, 8 failed — the 8 failures
  are pre-existing in `tests/test_compact_safetensors*.py` /
  `tests/test_compact_source_verification_window.py` (not ours, left
  as-is).
- After: **587 passed, 39 skipped, 8 failed** (the same 8 pre-existing;
  +16 new tests, all passing). Parity tool `--cpu-selftest`:
  `GB10_MINIMAX_M3_PARITY SELFTEST PASS`.
- New CPU tests cover: (a) default-off byte-identity (the dequant staging
  equals the pre-switch reference bit-for-bit and no MXFP8 state appears
  on the layer; `"0"` behaves like unset); (b) the retained pair is
  bit-preserving and the emulated MXFP8 GEMM semantics (through the
  F8_128x4 scale swizzle round trip) match a pure-torch dequant+matmul
  reference exactly on random data, and the fused apply output is
  byte-identical to the BF16 path with identical shape/dtype;
  (c) memory accounting at the real pack shapes; (d) switch-on fails
  closed with no vLLM kernel present, binds one kernel shim per shard
  (fp8 + row-major uint8 scales) and frees the stash when a kernel is
  bound.

## Memory saving

Per layer the two shards are `(512+128, 6144)`: BF16 staging
`640*6144*2` = 7.5 MiB; retained `640*6144 + 640*192` (payload + E8M0
block) = 3.87 MiB. The kernel's F8_128x4 swizzle does not grow either
block at these shapes (512 and 128 are already 128-row tiles; 192 scale
columns are already 4-col tiles).

- 60 indexer layers: **450.0 MiB -> 232.0 MiB resident**, saving
  **218 MiB (~0.21 GiB)** of the 121.69 GiB GB10 UMA pool.
- The switch-on path also never allocates the 450 MiB BF16 staging
  transient at load.

## Risks

- **W8A8 semantics.** The kernel quantizes the activation to MXFP8
  in-kernel, so indexer outputs deviate from the BF16-dequant path even
  when the weight handling is perfect. Case G gates this at cosine
  >= 0.9999 per shard at m in {1, 64}. Prior GB10 evidence for this same
  kernel on the DSpark draft's random synthetic shapes scored 0.99987 —
  below this case's bar — so the real-pack gate may land close to the
  threshold. **If case G fails, do not enable the switch**; the receipt
  records the per-shard cosines.
- **index_k sits at the kernel's minimum N.** `can_implement` admits
  N=128 on SM12x (N >= 128, N % 32 == 0), but the underlying
  `mm_mxfp8` may carry stricter undocumented tile constraints at exactly
  128. Case G would surface that as a hard error (fail-closed), never a
  silent fallback.
- **Decode-size arm.** The m=1 arm of case G exists because the kernel
  pads its activation-scale grid to 128 rows internally; a small-M
  limitation would be a production decode failure and must be caught by
  the case, not in the serve.
- TP > 1 remains unsupported for indexer shards (unchanged, checked at
  create_weights).
- The CPU swizzle round-trip test uses a test-local reference of the
  F8_128x4 layout; the real-kernel agreement is only proven by case G on
  the GB10.

## Exact GPU command (case G, for the parent to run later)

Inside the dev image on the DGX Spark (GB10, sm_121), with this worktree
bind-mounted at `/plugin` and the Hub token at `/run/hf_token`:

```
cd /plugin && PYTHONPATH=/plugin/src python3 tools/gb10_minimax_m3_parity.py \
    --json receipt_gb10.json --hf-token-file /run/hf_token
```

Case G runs as part of the full harness (cases A-G) and its per-shard
results are in `receipt_gb10.json` under `case_g`
(`shards[].per_m.m{1,64}.cosine/max_abs_ratio/nan` and the overall
`case_g.pass`). CPU-only structural check on any box:

```
cd /plugin && PYTHONPATH=/plugin/src python3 tools/gb10_minimax_m3_parity.py --cpu-selftest
```

## Reroute to the weight-only Marlin kernel (W8A16) after the run-1 case G failure

Run 1 (receipts in `/root/work/minimax-m31/receipts/val1/`) failed case G
on both shards: index_q m1 cos 0.999615 / m64 0.999673, index_k m1
0.999586 / m64 0.999659, against the 0.9999 gate.

### What the 607c94e path selected on sm_121 and why

The resolver hardcoded vLLM's
`FlashInferCutlassMxfp8LinearKernel`. On GB10 (sm_121) its gates admit
the shards (`has_device_capability(100)` is true for capability 121 and
its `can_implement` takes the SM12x `mm_mxfp8` route when N % 32 == 0),
so it executed -- but it is a W8A8 backend: it quantizes the ACTIVATION
to MXFP8 inside the kernel. Run-1's case D isolated the loss to exactly
that activation quantization: the weight-only `MarlinMxfp8LinearKernel`
scored cos 0.999999 on the same device (activations stay bf16), while
every W8A8 candidate landed at ~0.9998. The indexer drives top-k token
selection, so its activations must stay bf16.

### The fix

- `indexer_mxfp8_kernel()` now binds vLLM's own
  `MarlinMxfp8LinearKernel` (W8A16) and nothing else; the fail-closed
  contract is unchanged (missing import, platform refusal, or shape
  refusal each raise a clear `RuntimeError` naming
  `VLLM_EXL3_INDEXER_MXFP8`; the kernel's own constructor re-asserts
  both gates). The bound kernel class names are recorded on the layer
  (`_exl3_indexer_mxfp8_kernels`).
- The shim now carries the fields Marlin's load/apply steps read:
  `output_size_per_partition` / `input_size_per_partition` (the shard's
  true N/K) and `bias=None`; retention (fp8 payload bitcast + row-major
  uint8 E8M0 scales) is unchanged.
- Case G in `tools/gb10_minimax_m3_parity.py` now RECORDS the kernel
  class the resolver bound (per shard, `kernel`) and requires it to be
  `MarlinMxfp8LinearKernel`; a W8A8 re-point can never pass silently.
  The numeric gates are unchanged (cos >= 0.9999 at m in {1, 64},
  max_abs_ratio in [1/4, 4], 0 NaN). The old FlashInfer-specific
  payload-bits/scale-swizzle checks are replaced by an honest
  byte-accounting check: the Marlin load step replaces payload+scales
  with its repacked tensors, and the receipt verifies
  `repacked_bytes_preserved` (the repack must not grow the footprint).

### Honest memory accounting (Marlin path)

Per layer the shards are (512, 6144) and (128, 6144):

- Marlin repack: `marlin_padded_nk` needs NO tile padding at these
  shapes (both already satisfy the 64/128 thread-tile families), the
  repacked qweight is (K/16, 4N) int32 == exactly the payload bytes, and
  the permuted scale block comes back as 1-byte e8m0fnu == the scale
  bytes. Resident per layer is unchanged: 3.094 (index_q) + 0.773
  (index_k) = 3.867 MiB.
- Marlin workspace: ONE shared persistent `marlin_locks` tensor of
  num_compute_units x 4 int32 (~2 KiB total for the whole model, via
  vLLM's workspace manager), not per-layer.
- Load-time transient (honest): the repack staging (an int32 copy of
  the payload plus the repack output, and a short-lived fp32 scale
  staging) roughly doubles-to-triples the shard bytes for the duration
  of the load step and is freed immediately (stash freed +
  `replace_parameter`). Peak per layer during load ~9.7 MiB transient --
  still BELOW the default path's own load peak (the 3.87 MiB stash plus
  the 7.5 MiB BF16 staging rows, ~11.4 MiB), and the switch-on path
  never allocates the persistent 7.5 MiB staging at all.

Bottom line (unchanged by the reroute): 60 indexer layers
**450.0 MiB BF16 -> 232.0 MiB resident, saving 218 MiB (~0.21 GiB)** of
the 121.69 GiB GB10 UMA pool.

### CPU tests

`tests/test_indexer_mxfp8_retention.py` rewritten where the backend
changed: the F8_128x4 swizzle round-trip tests are gone (that layout
belonged to the W8A8 kernel); new tests pin (i) the resolver binds the
weight-only Marlin kernel and fails closed on platform/shape refusal
(through a fake vllm kernel API), (ii) the resolver source names no W8A8
backend, (iii) the Marlin tile rule needs no padding at the pack shapes
and the repack is byte-preserving, (iv) the shim carries the partition
sizes Marlin reads. Default-off byte-identity tests are untouched.
Suite: 591 passed / 39 skipped / 8 failed -- the same 8 pre-existing
`test_compact_safetensors_attestation.py` /
`test_compact_source_verification_window.py` failures, nothing new.
Parity tool `--cpu-selftest`: `GB10_MINIMAX_M3_PARITY SELFTEST PASS`.

### Exact GPU command (the parent runs this on the GB10)

Same harness, same gates; only case G's backend changed. With this
worktree at `<tree>` and the Hub token at `/run/hf_token`, inside the
dev image:

```
docker run --rm --gpus all \
    -v <tree>:/plugin:ro \
    -v /root/work/minimax-m31/receipts:/receipts \
    -v /run/hf_token:/run/hf_token:ro \
    -e PYTHONPATH=/plugin/src \
    <dev-image> \
    python3 /plugin/tools/gb10_minimax_m3_parity.py \
        --json /receipts/parity.json --hf-token-file /run/hf_token
```

`<tree>` is `/root/work/minimax-m31/lanes/plugin/wt-indexer-mxfp8` and
`<dev-image>` is the same dev image that produced the run-1 receipt.
Case G's per-shard `kernel` field must read `MarlinMxfp8LinearKernel`
and `repacked_bytes_preserved` must be true; the numeric gates are
unchanged. CPU-only structural check on any box:

```
cd <tree> && PYTHONPATH=src python3 tools/gb10_minimax_m3_parity.py --cpu-selftest
```

### Unverified without the GPU

- The real GB10 case G verdict under the Marlin backend (this box has no
  GPU). Case D proved Marlin at (12288, 6144) on sm_121; the shard
  shapes (512/128, 6144) pass `can_implement` and need no tile padding
  by arithmetic, but only the rerun proves the marlin_gemm execution at
  those N.
- The m=1 decode arm under Marlin (case D ran m=64 only).

## Checkin 2: real-serve load fix (boot14) -- the retained shards were on
the CPU when the Marlin repack ran

### Root cause (from the loader code, not a guess)

The real serve boots through vLLM's
`model_executor/model_loader/utils.py::process_weights_after_loading`,
which wraps every quant method's hook in `device_loading_context`: the
layer's REGISTERED parameters are moved onto the CUDA target for
load-time processing and restored to their pinned-CPU/UVA placement
afterwards. The retained indexer payload/scales are stashed as PLAIN
list attributes (`_exl3_indexer_payloads/_scales`) -- never parameters --
so nothing ever moves them: at `_process_indexer_shards` time they are
still the CPU mmap source views the loaders built. The DEFAULT path
never hit this because its dequant output goes through
`layer.weight.data[rows].copy_(weight_bf16)`: the staging parameter IS
on the target device at that point, so the copy performs the H2D
transfer and the rows survive as `_exl3_bf16_weight`. The switch-on path
handed the CPU stash straight to the kernel, and
`gptq_marlin_repack` only has a CUDA backend (receipts/val2/
boot14-container.log ~260-300).

### Fix

- `src/vllm_exl3/exl3.py`
  - `_indexer_mxfp8_target_device()` -- resolves the ONE device the
    fused layer's parameters agree on at load processing and requires
    it to be CUDA; anything else (CPU params, disagreement, no params)
    fails closed naming `VLLM_EXL3_INDEXER_MXFP8`.
  - `indexer_mxfp8_move_to_device()` -- moves one shard's payload and
    scales onto that device before any repack: no-op if already there,
    refuses mixed devices, verifies BOTH tensors landed exactly on the
    target (else fails closed naming the switch), then releases the
    consumed CPU source views (`_madv_dontneed_cpu_tensor`, the same
    reclaim the loader uses after H2D).
  - `_process_indexer_shards()` (switch-on branch) now resolves the
    target device once, moves each shard, and clears the stash slots and
    the loop locals BEFORE retention/kernel binding, so no reference to
    the CPU tensors survives (GB10 UMA: a lingering CPU copy would erase
    the 218 MiB saving). The repack transient stays per shard: the
    kernel's load step `replace_parameter`s the payload/scales with the
    repacked tensors, so the retained fp8 form is released in the same
    step.
  - Default (switch-off) branch untouched: byte-identical staging.
- `tools/gb10_minimax_m3_parity.py`
  - New case G2: the boot14 real-serve placement. The REAL layer-0
    index_q/index_k bytes are fetched into CPU-resident tensors exactly as
    the loader leaves them (never `.to(device)`), the layer carries only
    the 0-row switch-on staging parameter on CUDA (as
    `device_loading_context` leaves it), and the case drives the
    plugin's PRODUCTION load step
    (`Exl3LinearMethod._process_indexer_shards`), not a tool-side move.
    Per shard it gates: repacked tensors on CUDA, the layer holds no
    reference to the CPU sources (stash cleared, no pointer alias),
    byte-preserving repack, and case G's numeric gates (cos >= 0.9999,
    max-abs within 4x, 0 NaN) against the BF16-dequant reference of the
    same bytes. Fail-closed is a FAIL, never a skip; wired into the
    verdict next to case G. `--cpu-selftest` gained 5 G2 structure bars
    plus 2 plugin-side bars (move-before-repack ordering, fail-closed
    mover with source release); selftest still PASS.
- `tests/test_indexer_mxfp8_retention.py` -- the switch-on tests now
  simulate the real-serve placement (CPU stash + patched device seams);
  new tests pin: move BEFORE the kernel load step in shard order, the
  shim holds the moved tensors never the CPU stash, no reachable layer
  reference to the CPU tensors, resolver fails closed on CPU/disagreeing
  parameters, mover no-op on-device / real move + source release /
  fail-closed on off-device landing and mixed devices (the real transfer
  arm uses the meta device; the CUDA transfer is gated by case G2).
- `tests/test_minimax_m3_parity_tool.py` -- new case G2 tests (CPU
  sources, production load step, device/release/numeric gates,
  fail-closed FAIL, main() wiring) and a source pin that the plugin's
  load step moves the stash before the kernel binding.

### Test counts

- Before this checkin: 591 passed, 39 skipped, 8 failed (the same 8
  pre-existing `test_compact_safetensors*.py` /
  `test_compact_source_verification_window.py` failures).
- After: **602 passed, 39 skipped, 8 failed** (identical failure set,
  +11 new tests, all passing). Parity tool `--cpu-selftest`:
  `GB10_MINIMAX_M3_PARITY SELFTEST PASS`.

### Unverified without the GPU

- The real GB10 case G2 verdict (this box has no GPU): the on-hardware
  move, the Marlin repack at the moved tensors, and the m in {1, 64}
  numeric gates must be proven by the parent's rerun; case G's numerics
  are already settled at cos 0.999999, but G2 additionally proves the
  production load step's placement path.
- The real serve boot with `VLLM_EXL3_INDEXER_MXFP8=1` (needs the GB10
  dev image; the boot14 crash path is what case G2 exercises).
