# CHECKIN-Q4KV-1 — q4 (nvfp4-layout) main KV cache, milestone 1

Branch `feat/minimax-q4kv` (base `3f80e67`), worktree
`/root/work/minimax-m31/lanes/plugin/wt-q4kv`. Scope per
`lanes/engine/Q4KV-SCOPE.md`: main MSA KV only; the indexer cache stays
fp8 e4m3; the DSpark draft pool is untouched. Design: 4-bit e2m1 data
nibbles + e4m3 per-group-of-16 scales in the vendor nvfp4 pool layout,
pool sized at **576 B/token/layer** for 4 KV heads (73728 B per
128-token page), dispatched only under `VLLM_EXL3_Q4KV=1`.

## What is implemented

1. **Byte-level reference** — `src/vllm_exl3/minimax_q4kv_reference.py`
   (795 lines, pure torch; no vllm, no triton imports, runs on the
   CPU-only red venv):
   - e2m1/e4m3 encode/decode; e4m3 encoder is byte-equal to torch's
     satfinite `float8_e4m3fn` cast; e2m1 ladder `{0, .5, 1, 1.5, 2, 3,
     4, 6}` with sign bit, ties-to-even RNE, and **code 7 = ±6.0
     saturation (no NaN)**, matching the vendor csrc `e2m1Code` LUT /
     `cvt.rn.satfinite.e2m1x2` (csrc lines 350-440).
   - Pool layout per vendor csrc (stride checks `(2)==72`, `(1)==9216`):
     container uint8 `[2*nkv, 128, 72]`; each 9216-byte slot region is
     flat `[8192 B data][1024 B scales]`; slot id `2*head + side`
     (K=0, V=1); data byte j = elements (2j, 2j+1) as (low, high)
     nibbles. K scales linear (`token*8 + g`); V scales use the
     SM100 trtllm-gen 4x4 token-quad swizzle
     (`((t//4)*4 + g//2)*8 + (g%2)*4 + (t%4)`); both offset tables
     bijective (1024 unique entries each).
   - Pool arithmetic: `q4_pool_page_bytes` = 73728 B/page (nkv=4),
     `q4_pool_bytes_per_token` = 576 B/token/layer.
   - Quantization: bf16 pre-round; per-group scale
     `s = max(amax/6, 2^-9)` as saturating e4m3; **unit global scale**
     (matches the vendor's shipped nvfp4 KV path for this model);
     zeros stored as +0.
   - Full W1 reference (Gemma QK-norm with a precisely-rounded 1/sqrt —
     deliberately not the vendor kernel's approximate hardware rsqrt —
     plus partial NeoX RoPE, pack, and slot scatter), fail-closed
     `validate_w1_call`, vectorized `dequant_pool_pages`,
     `build_prefill_staging` (prefill fallback: dequantize the selected
     blocks to bf16 and reuse vLLM's existing prefill kernel), and a
     dense fp32 block-sparse attention oracle.
2. **Triton kernels** — `src/vllm_exl3/minimax_q4kv_kernels.py`
   (681 lines; triton imported lazily via `vllm.triton_utils`,
   CUDA-only):
   - **W1** `q4_w1_insert` (kernels.py:500): fused Gemma QK-norm +
     partial NeoX RoPE + e2m1/e4m3 pack + scatter insert, with the
     processed K and index_k rows written back into qkv in place
     (mirrors the vendor op); nibble pairing via a weighted-sum
     reduction (Triton has no tensor slicing); RoPE partner computed
     from a permuted load times the (permutation-invariant) norm
     reciprocal.
   - **R2** `q4_sparse_decode` (kernels.py:592): q4-dequant split-K
     sparse decode producing partials for vLLM's existing merge
     kernel, which is reused unchanged; shape-constant chunk count
     (CUDA-graph safe), as in the vendor wrapper.
3. **Overlay** — `src/vllm_exl3/minimax_q4kv.py` (426 lines) plus the
   gate in `src/vllm_exl3/__init__.py:50` (`register()` installs the
   overlay only when `os.environ["VLLM_EXL3_Q4KV"] == "1"`; default
   off, nothing patched, byte-identical behavior).

## Hook evidence (all verified against vllm-image at the cited lines)

| Upstream hook (vllm-image/vllm) | Overlay (wt-q4kv) |
|---|---|
| `models/minimax_m3/common/sparse_attention.py:525` `select_main_backend_and_impl_cls` (nvfp4 raise on non-SM100) | rerouted by `_make_select_main` (minimax_q4kv.py:150, installed :404); overlay backend publishes the upstream nvfp4 spec arithmetic (2 K/V slots per head, 72 B/slot, LBHNC) so the pool is sized 576 B/token/layer for 4 KV heads; vendor Triton impl retained; block size stays 128 |
| `_custom_ops.py:2711` `fused_minimax_m3_qknorm_rope_kv_insert` (fp8 write call sites `nvidia/model.py:722,1055`) | wrapped by `_make_w1_wrapper` (minimax_q4kv.py:218, installed :411): q4 pool → our Triton W1; fp8/bf16 pools and the DSpark draft (separate op attribute) → original CUDA op unchanged |
| `common/ops/sparse_attn.py:240` `_gqa_sparse_decode_kernel` via `minimax_m3_sparse_attn_decode` | wrapped by `_make_decode_wrapper` (minimax_q4kv.py:288, installed :417-425): q4 pool → our R2 + unchanged vLLM merge kernel; else original |
| `common/ops/sparse_attn.py:52` `_gqa_sparse_fwd_kernel` via `minimax_m3_sparse_attn` (prefill) | wrapped by `_make_prefill_wrapper` (minimax_q4kv.py:335): q4 pool → `build_prefill_staging` dequantizes the selected blocks to bf16 and reuses vLLM's existing prefill kernel on a remapped block table (correct, not fast); else original |
| `common/indexer.py:155` `get_kv_cache_spec` (indexer), `:510` `select_indexer_impl_cls` | NOT patched: the indexer cache stays fp8 e4m3 in milestone 1; nvfp4 indexer requests are refused (`index_cache_dtype` "uint8" → fail closed) |
| `nvidia/model.py:926` `get_kv_cache_spec` (main) | reached only through the rerouted selector above; spec arithmetic is upstream's own nvfp4 path |

Pool detection `_is_q4_pool` (minimax_q4kv.py:45) matches only the q4
container (uint8, dim 4, `[., 2*nkv, 128, 72]`, strides (2)==72 /
(1)==9216); fp8, bf16, indexer and draft pools never match and fall
through to the original ops.

## Test counts (red venv, `/root/work/minimax-m31/lanes/plugin/.venv-red`)

Command: `python -m pytest -q -p no:cacheprovider tests`

- Baseline `3f80e67`: 571 passed, 39 skipped, 8 failed (all 8
  pre-existing: `tests/test_compact_safetensors*.py`,
  `tests/test_compact_source_verification_window.py`).
- After milestone 1: **592 passed, 39 skipped, 8 failed** — +21 passed
  (the new suite `tests/test_minimax_q4kv.py`, 21/21 green in 1.96 s),
  the same 8 pre-existing failures, skips unchanged.
- New suite covers: pool byte arithmetic (576/73728), scale-offset
  layout + bijectivity, encoder oracles (e4m3 == torch satfinite cast;
  e2m1 ladder incl. code-7 saturation), pack/unpack roundtrip +
  byte-exact repack + error bounds, full W1 reference vs independent
  math, fail-closed dispatch (`validate_w1_call` + all overlay
  wrappers), pool detection, staging dequant == per-slot oracle,
  prefill remap correctness, and default-off `register()`.

## Interpreter results

`triton` is not importable anywhere on this VPS (system python and the
red venv; `TRITON_INTERPRET=1` needs the package, so the interpreter
route is unavailable) and there is no GPU. **W1/R2 were therefore
never executed here.** Their validation is (a) bit-level agreement of
the algorithms with the pure-torch reference by construction (same
ladders, same offset tables, pinned in tests) and (b) the GB10 parity
harness below, which must be run before any serving claim.

## CPU-reference accuracy numbers (measured, this venv)

- e4m3 encoder: byte-equal to `torch.Tensor.to(torch.float8_e4m3fn)`
  on probed boundary values (0, subnormals, 2^-9 boundaries, 448,
  saturation, inf/NaN).
- Quantization error bound: max `|dq − x| / group_scale` = **1.0**
  (the pinned worst-case ladder half-gap, between 4 and 6); mean
  relative error 0.18 on N(0,1) bf16 data.
- Sparse-decode attention accuracy (exact fp32 oracle vs quantized
  pools, 64 q-heads/4 kv-heads/4 pages, exact-oracle attention):
  - cos(q4 attention, exact) = **0.9946** (≥ 0.99 required)
  - cos(fp8 attention, exact) = 0.9996
  - cos(q4 attention, fp8 attention) = **0.9943**
  i.e. the q4 path retains the ≥ 0.99 cosine floor against both the
  fp8 baseline and the exact oracle on this probe.

## GB10 parity (NOT run on this VPS — no GPU, no triton)

Exact command, from the worktree root, on a DGX Spark GB10 with the
MiniMax-M3.1-Flash vLLM image:

```
python tools/gb10_q4kv_parity.py
```

Cases (all must pass, non-zero exit otherwise):
1. W1 byte-exact vs the torch reference packer (pool bytes, q_out,
   index bytes; cross-checked CPU-vs-GPU; unmapped pages stay zero).
2. R2 decode cosine ≥ 0.99 vs the fp8 baseline AND vs the fp32 oracle
   on real-shaped random data (64 q-heads, 4 kv-heads, 64 pages,
   topk 8).
3. Pool bytes measured: 73728 B/page, 576 B/token/layer.
4. CUDA-graph capture of R2 reproduces the eager output bit-identically.

## Open risks

- W1/R2 Triton kernels are **unexecuted**: compile errors, sm_121 code
 gen limits, and register pressure are unverified until the parity
  harness runs on GB10. The R2 chunk-count constant and grid sizes
  mirror the vendor wrapper but are unproven on hardware.
- The prefill fallback dequantizes selected blocks to bf16 (correct,
  slow); prefill throughput under q4 will regress until a native q4
  prefill reader (R1) exists.
- The Gemma-norm 1/sqrt choice (precisely rounded, not the vendor
  hardware rsqrt) makes reference and Triton bit-agreeable but differs
  numerically from the vendor CUDA op's approximate rsqrt; downstream
  parity is against our reference, not the vendor op.
- Unit global scale is pinned to the vendor's shipped nvfp4 KV path;
  if the vendor path ships non-unit scales, `validate_w1_call` fails
  closed (refusal, never silent misuse).
- vLLM upgrade drift: hook lines cited above are from the current
  vllm-image; any upstream refactor moves them.

## What milestone 2 needs

- Run `tools/gb10_q4kv_parity.py` on GB10 and fix any Triton
  compile/numerics fallout; measure real W1/R2 latency.
- R1: native q4 prefill reader (replace the bf16 staging fallback).
- R3/R4: nvfp4 indexer cache (currently refused; `index_cache_dtype`
  "uint8" support), including the indexer score kernels.
- End-to-end serve with `VLLM_EXL3_Q4KV=1` on GB10: pool sizing
  observed at 576 B/token/layer in the engine logs, draft (DSpark)
  still on its own fp8 pool, quality gates vs the fp8 baseline.
- CUDA-graph capture of the full decode step (not just R2 in
  isolation).

## Post-val1 fix: fp8 pointers at the kernel boundary (this commit)

The first real GB10 run (receipts/val1/q4kv.log, image vLLM
0.1.dev23341+g8744dd2fb) died in case 1 with
`CompilationError: cannot cast uint8[constexpr[128]] to fp8e4nv`:
`_q4_w1_kernel` stores raw e4m3 *bytes* into the fp8 e4m3 index
buffers (`index_q_out`, `index_cache`), and Triton's frontend refuses
to cast the stored uint8 values to the pointer's fp8 element type.

**Convention adopted (all q4 KV kernels, not just the failing line):**
fp8 e4m3 buffers NEVER cross the kernel boundary as fp8 pointers. The
kernels are pure byte math, so the host wrapper hands each of them a
`uint8` byte view of the same storage (`_fp8_bytes` =
`tensor.view(torch.uint8)`, applied to `index_q_out` and
`index_cache` in `q4_w1_insert`). R2 takes no fp8 tensors at all (the
pool is uint8 by design); its wrapper is pinned to hand the pool over
as-is. `validate_w1_call` still validates the ORIGINAL fp8 tensors
before the view is taken, so fail-closed semantics are unchanged.

**Kernel audit (frontend type errors, both kernels, every pointer):**
- W1: qkv/q_out loads+stores bf16-to-bf16; pool data/scale stores
  uint8-to-uint8; norm/cos_sin loads are value-cast to fp32 (legal for
  fp32 or bf16 host weights); positions/slot-map loads stay integer
  with the page offset computed in int64 (`page.to(tl.int64) *
  pool_page_stride`); index-byte stores now uint8-to-uint8. No other
  fp8 pointer exists in the signature.
- R2: topk loads are normalized `.to(tl.int32)` and block-table page
  indices `.to(tl.int64)` (required for the pool offset); the split-K
  loop bounds are now pinned to int32 (`seq_lens` load normalized
  `.to(tl.int32)`) so `tl.range` sees uniform bound types whether the
  host tensor is int32 (vLLM serving) or int64 (the parity harness);
  sequence lengths fit int32 by construction, so this is
  value-preserving. o/lse stores match their bf16/fp32 buffers.
- `_expand16`/`_nibbles_to_bytes`/`_dequant_side` reshapes all match
  element counts (join-based repeat-16 and nibble pairing verified
  against the reference layout).

**Frontend check without a GPU:** `tools/q4kv_triton_frontend_check.py`
compiles the REAL W1/R2 kernel objects via
`triton.compile(ASTSource(...), target=GPUTarget("cuda", 121, 32))`
for a matrix of host dtype variants (parity-harness int64 index
tensors, vLLM int32 serving tensors, fp32 and bf16 norm/cos_sin
weights, PROCESS_INDEX on/off) and refuses any signature containing
an fp8 pointer type. `tools/gb10_q4kv_parity.py` now runs it as step
0, before case 1, so this failure class dies fast with a clear
message. NOTE: no Python on this VPS has triton (checked: system
python, the red venv, every other local venv, pip caches), so the
check tool has NEVER been executed here; its first execution must be
the GB10 run below. The tool also depends on the Triton version's
ASTSource API shape (constants vs constexprs kwarg) and adapts by
introspection; if the image's Triton renames the API again, the tool
fails loudly instead of skipping.

**Tests (red venv):** 4 new CPU tests in `tests/test_minimax_q4kv.py`
(25/25 green): the real `q4_w1_insert` wrapper hands the fp8 index
buffers to the kernel as uint8 views of the same storage (data_ptr
equality) and nothing fp8-typed crosses the boundary; the None
fallbacks pass the documented placeholders un-viewed; the real
`q4_sparse_decode` wrapper hands the uint8 pool over as-is and
produces bf16/fp32 split-K partials consumed by the unchanged vLLM
merge kernel; and the GB10 harness source is pinned to run the
frontend check before case 1. Full suite: 596 passed, 39 skipped,
the same 8 pre-existing failures
(test_compact_safetensors_attestation.py,
test_compact_source_verification_window.py) — failure set identical
to the baseline.

**Exact GB10 command for the re-run (parent executes; the worktree is
mounted read-only, the toolchain lives in the image):**

```
docker run <minimax-m3.1-flash vllm image, vLLM 0.1.dev23341+g8744dd2fb> \
  -v <this worktree>:/plugin:ro -w /plugin \
  -e PYTHONPATH=/plugin/src \
  python3 tools/gb10_q4kv_parity.py
```

Unverified on this VPS (no GPU, no triton): the actual sm_121
compilation of W1/R2, the fix's effect on the val1 failure, and all
GB10 parity numerics. The frontend check + case 1 re-run on GB10 is
the acceptance gate for this fix.

## Post-val2 fix: harness input-data consistency + CPU dry run (this commit)

GB10 val2 result: step 0 passed (W1 + R2 compile for sm_121 in every
host dtype variant — the val1 fix works), then case 1 crashed in the
harness's own test data: positions 1000..1031 against a 256-row
cos_sin cache -> IndexError inside the reference packer. No q4 kernel
numerics were checked.

**Root causes audited in the harness data (both fixed):**

1. The rotary cache was 256 random rows while positions reached 1031.
   The case now builds the REAL partial-RoPE table — vLLM's
   `RotaryEmbedding._compute_cos_sin_cache` formula (NeoX halved
   `[cos(pos*inv_freq) | sin(pos*inv_freq)]` rows) with the
   MiniMax-M3 configuration the overlay serves (rope_theta 5e6,
   rotary_dim 64 = partial_rotary_factor 0.5 of head 128), sized to
   cover max(positions)+1 rows.
2. Case 1 passed `index_cache=None`: the wrapper then hands the POOL
   as the index_k scatter placeholder, so the kernel's index_k writes
   would corrupt pool page 0 and break the byte-exact comparison. The
   case now allocates a real fp8 e4m3 index cache (2 pages; the slot
   mappings deliberately cross a page boundary on both pools) and
   compares the full scattered index cache byte-for-byte against the
   reference's index_k bytes — a fourth byte-exact output (pool, q,
   index_q, index_k scatter).

Cases 2/4 data was audited and is consistent: topk entries are
logical block-table indices within the table width, block_table
entries are physical pages within the pool, seq_lens is pinned to
TOPK*BLOCK so the kernel's n_blocks mask, the fp8 baseline's causal
mask and the fp32 oracle's query_pos select the same token set (the
CPU dry run proves the baseline and the oracle agree at cosine
0.999629 before any kernel runs).

**Fail-closed input validation (never a kernel-looking failure again):**
`tools/q4kv_harness_inputs.py` (pure torch, red-venv-safe) builds
every case's inputs and validates them BEFORE any kernel or reference
call; every index must be in range for its tensor (positions vs cache
rows, slot mappings vs pool/index pages, topk vs block-table width,
block_table vs pool pages, seq_lens vs accessible tokens and the
fixture invariant) and any violation raises a ValueError naming the
input. The reference `reference_w1_write` now raises a named
ValueError (not an IndexError) when positions exceed the cache.

**CPU dry run (new):** `python tools/gb10_q4kv_parity.py --cpu-dry-run`
builds every case's inputs, runs the validators, runs the full
pure-torch reference paths on CPU (W1 packer incl. scatter-offset
dequantization round-trip and index_k byte check; fp8 baseline and
fp32 oracle attention to completion) and the case-3 pool arithmetic —
no triton, no GPU. PASSED on this VPS (red venv, 3.3 s). The GPU
cases (Triton W1/R2 byte-exactness and cosine, CUDA-graph capture)
remain GB10-only.

**Tests (red venv):** 6 new CPU tests (31/31 green in
`tests/test_minimax_q4kv.py`): the reference's position guard
(ValueError, never IndexError, in-range batches still pack); the
shipped builders pass their validators; the rotary cache is the real
partial-RoPE table (theta 5e6, vLLM row form, sized from positions);
the exact val2 bug (256-row cache) is caught by the validator; every
out-of-range mutation (slot mapping, index slot mapping, index_cache
None, positions, topk, block_table, seq_lens capacity and invariant)
fails closed with the named message; and every harness case body is
pinned to validate its inputs before its kernel call, with main()
routing --cpu-dry-run. Full suite: 602 passed, 39 skipped, the same 8
pre-existing failures as the unmodified tree (verified by stash/run/pop)
— failure set identical to the baseline.

Unverified on this VPS (no GPU, no triton): the actual GB10 re-run of
cases 1, 2 and 4 — W1 byte-exactness against the now-real rotary
cache and index cache, R2 cosine on real-shaped data, and CUDA-graph
capture. The exact GB10 command is unchanged except nothing; the CPU dry
run must precede it.

## Post-val3 fix: gate what matters, run everything, JSON receipt (this commit)

GB10 val3 result: step 0 passed (W1+R2 frontend-compile for sm_121),
then case 1 died at `assert torch.equal(q_out, ...)` — the Triton W1
kernel's post-QK-norm+RoPE Q differs from the pure-torch reference by
last-bit fp32 noise, and the harness stopped there: nothing about K/V
pool bytes, index bytes, R2 cosine, pool bytes or CUDA-graph capture
was measured. The CHECKIN already documented that bit-exact bf16
Q/K after norm+RoPE is NOT a realistic gate (the reference uses a
precisely-rounded 1/sqrt, the kernel the hardware `sqrt.approx.f32`
like the vendor op's `rsqrtf` — verified in the image csrc
`fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu:141` — and the
sum-of-squares reduction order differs). The gate was wrong, not the
kernel.

**Analysis first (what can legitimately differ, and what cannot):**
the two noise sources are bounded at a handful of fp32 ulps (~2^-21
relative). After bf16 rounding such a perturbation either rounds to
the same bf16 number or to the adjacent one, so kernel-vs-reference
bf16 rows differ by at most 1 bf16 ulp on a small fraction of
elements. Everything else in the kernel was re-derived against the
reference and the vendor csrc: partial NeoX RoPE pairing for rotary
dim 64 (partner d±32, halved cos/sin indexing) is bit-identical to
the reference; the norm is Gemma-style `x*rsqrt(mean+eps)*(1+w)` with
the same eps, the same (1+w) and reduction length 128 in both; and
the vendor op encodes index_q/index_k e4m3 bytes from the fp32
in-register values (no bf16 materialization), exactly as the reference
and our kernel do.

**Real bug found and fixed — the payload quotient.** The reference
spec pins the payload quotient as RZ at fp32 precision
(`rz(x * rn(1/s))`, `rz_rounded_product`), but the Triton packer
computed the plain RN product (`q = xb * (1/s_vec)`): at the e2m1
decision boundaries the kernel emitted payload bytes the spec'd packer
does not. Worse, auditing the reference's own emulation (now factored
into `rz_rounded_product`) showed its correction condition was
SIGNED (`p > exact`), which is true RZ only for positive products:
negative quotients that had rounded away from zero kept a magnitude
ABOVE the exact product. Both are fixed: the quotient steps one
representable value toward zero iff the RN product landed on the
away-from-zero side of the exact (f64) product, for BOTH signs — the
literal definition of RZ. The Triton `_rz_product` mirrors the
reference bit-for-bit (f64 product compare + sign-magnitude bit
pattern decrement); the fp64 mul is two slow ops on a write path,
carried for exactness. The CPU tests pin the property differentially:
the shipped implementation never exceeds the exact magnitude on
either sign, the correction triggers on both signs, and the PRE-FIX
condition is reproduced violating the property.

**New gate module `tools/q4kv_gates.py` (pure torch):**
- `gate_bf16_numerics` (gate a): max bf16 ulp distance (monotone-key
  bit distance), max_abs, mismatched count/fraction, cosine; PASS iff
  max ulp <= 2 AND cos >= 0.99999 AND finiteness agrees. The 2-ulp
  limit carries a full 2x margin over the <= 1 ulp the two legitimate
  noise sources can produce; a uniform 1-ulp shift (a systematic
  2^-8-relative bias no legitimate noise source produces) fails the
  cosine leg (~1 - 2^-15 < 0.99999).
- `gate_bytes_exact` (gate b): byte-identical or fail, with the exact
  mismatch count and first mismatching flat offset.
- `gate_e4m3_codes` (gate c): for the fp8 index bytes (no bf16
  readback exists — vendor and reference encode fp32 directly): max
  distance in representable-value code steps (sign-magnitude monotone
  key) and decoded cosine; PASS iff <= 2 code steps AND cos >=
  0.99999 AND no one-sided NaN code. The fp32 noise can move a byte by
  at most 1 code step with probability ~2^-18 per element.
- `bf16_ulp_step` / `e4m3_code_step`: synthetic-perturbation helpers
  for the dry run and tests.

**Harness restructure (`tools/gb10_q4kv_parity.py`):** every unit runs
to completion inside its own try/except — one PASS/FAIL line per unit,
full per-gate diagnostics, JSON receipt via `--json PATH`, final
verdict, exit non-zero iff any unit failed. Units: 0 frontend check;
1a norm+RoPE numerics (q_out + the processed K/index_k rows read back
from qkv, gate a); 1b packing byte-exactness DECOUPLED from the norm
rounding — the reference packer run on the kernel's OWN processed
rows (read back after the call; V is not normed) must reproduce the
kernel's pool bytes byte-for-byte (bf16 rounding is idempotent, so
both packers see identical fp32 inputs there; any differing byte is a
real packing bug), plus unmapped-pages-stay-zero; 1c index_q/index_k
bytes (gate c — byte-exactness is impossible by design, and the
receipt says so); 1d rewrite scope (only K/index_k rows of qkv change;
Q/V/index_q rows bit-identical — nothing downstream consumes an
in-place Q on the q4 path, the sparse layer reads its query from the
dedicated q_out buffer, image model.py:642); 2 R2 cosine >= 0.99 vs
fp8 baseline AND fp32 oracle, NaN-free; 3 pool bytes; 4 CUDA-graph
replay bit-identical to eager (case 4 now runs its own eager pass, so
a case-2 failure no longer blanks case 4).

**CPU dry run extended:** the reference paths still run to completion
(mapped pages/slots, dequant round-trip, index_k bytes,
baseline-vs-oracle cosine 0.999628) and every gate is self-tested on
synthetic perturbations: sparse 1-ulp passes, uniform 1-ulp and gross
corruption fail; a single flipped nibble and a single flipped scale
byte fail; sparse 1-code-step passes, 3-code-step and one-sided NaN
fail. VERDICT PASS (6/6 units) on this VPS.

**Tests:** 8 new (39/39 green in `tests/test_minimax_q4kv.py`): gate
semantics (a/b/c, exact metrics and first-mismatch offsets), the rz
property + pre-fix differential, the kernel's `_rz_product` source
pin, the harness's per-unit completion/JSON/non-zero-exit pins, the
kernel's-own-rows pin for unit 1b, and the dry-run gate self-test
pins. Full suite: 610 passed, 39 skipped, the same 8 pre-existing
failures as the unmodified tree — failure set identical to the
baseline.

**Exact GB10 command (parent executes; the worktree is mounted
read-only, the toolchain lives in the image, and the receipt must be
written to the mounted receipts volume):**

```
docker run <minimax-m3.1-flash vllm image, vLLM 0.1.dev23341+g8744dd2fb> \
  -v <this worktree>:/plugin:ro -v <receipts dir>:/receipts \
  -w /plugin -e PYTHONPATH=/plugin/src \
  python3 tools/gb10_q4kv_parity.py --json /receipts/val4/q4kv.json
```

Unverified on this VPS (no GPU, no triton): the sm_121 compile of the
new `_rz_product` (fp64 ops in Triton — if the frontend refuses fp64
on sm_121, unit 0 fails loudly and the fallback is an fma-based
two-product emulation of the same RZ condition), and every GPU unit's
actual verdict. One GB10 run now yields the full picture: the receipt
will contain per-gate ulp/code-distance/mismatch/cosine numbers for
q_out, processed K, index_k, the packing byte comparison, both R2
cosines, the pool byte counts and the CUDA-graph replay result.

## Post-val4 fix: device placement at the kernel boundary (this commit)

The 4th GB10 run (receipts/val4) passed the frontend check and every
W1 unit (1a within 1 bf16 ulp, 1b/1c byte- and code-exact, 1d clean)
and case 3, but cases 2 and 4 both died with
`ValueError: Pointer argument cannot be accessed from Triton (cpu
tensor?)`. **Root cause: the harness, not the kernels.** `_run_r2`
moved the five R2 inputs to CUDA but the caller passed the OUTPUT
buffer `torch.zeros_like(inp["q"])` — a CPU tensor — straight into
`q4_sparse_decode`, where it reached vLLM's unchanged merge kernel.
The R2 Triton kernel itself never executed; nothing about R2 numerics
was wrong.

**Harness fix — one mover, explicit placement everywhere:**

- `_bundle_to_device` is the single recursive mover (dicts/tuples/
  lists; non-tensors pass through unchanged). Every case now names its
  kernel-facing keys (`_W1_KERNEL_KEYS`, `_R2_KERNEL_KEYS`) and moves
  the ENTIRE bundle with the one mover; per-tensor `.cuda()` calls are
  gone. Everything not in those key sets is a reference-path input and
  stays on the CPU bundle by construction.
- `_r2_static_bundle` allocates the R2 output buffer ON the target
  device (`torch.zeros_like(gpu_q)`), never inherits it from the CPU
  bundle; `_launch_r2` is the only call site of the R2 wrapper
  (eager, warmup and capture all go through it).
- The pure-torch reference paths are pinned CPU-only: `_fp8_baseline`
  (and the oracle) refuse any non-CPU tensor with a named error, so a
  GPU-side tensor can never leak into baseline math either.

**Case 4 rewritten to the static-buffer rules.** Every input is a
preallocated CUDA tensor from `_r2_static_bundle`; the check is
two-stage: (a) replay on the captured bytes must equal the eager run
bit-identically, and (b) after NEW pool bytes are copied into the
static buffer (whole pages rotated), replay must equal the eager run on
THOSE bytes and must differ from the captured-bytes output — a stale
replay of captured data, or a vacuous mutation, both fail loudly.

**Src fix — fail-closed device gate (both wrappers).**
`minimax_q4kv_kernels.py` now has `_device_gate`/`_off_device`: the
query tensor must be on a CUDA device and every kernel-facing argument
must be on the query's device, else the wrapper raises BEFORE any
validation or triton import, naming the offending argument(s) and
device(s) (`q4 KV refused: 'output' is on cpu, expected cuda:0`). A
serving-side placement bug now dies with a precise message instead of
an opaque Triton pointer error. The internal split-K partials already
follow `q.device`; the gate makes the whole boundary symmetric. The
CPU-only test suite simulates the GPU boundary by patching
`_REQUIRED_DEVICE_TYPE` (the gate itself is tested unpatched and must
fire).

**Dry run extended (dry/4-device-audit, meta simulation).** For the
GPU path it asserts, with the meta device standing in for CUDA: the
mover puts EVERY tensor of both cases' kernel-facing bundles on the
target (count/shape/dtype preserved, nothing dropped), the R2 output
buffer is allocated on the target, non-tensors pass through unchanged,
`_off_device` names every offender (and only the offender in a mixed
bundle), the real wrappers refuse an all-CPU bundle before any triton
import naming `'q'`/`'qkv'`, and the fp8 baseline refuses a
target-device tensor. VERDICT PASS (7/7 units) on this VPS.

**Tests:** 3 new (42/42 green in `tests/test_minimax_q4kv.py`): the
device gate names every off-device argument and refuses all-CPU bundles
pre-triton; the gate does not shadow the geometry/fail-closed
validation; and the dry-run device audit + static-buffer case-4 pins.
Full suite: **613 passed, 39 skipped, the same 8 pre-existing
failures** (failure set verified identical to the unmodified tree by
re-running it: 610 passed there, +3 = the new tests).

**Unverified on this VPS (no GPU, no triton):** whether cases 2 and 4
now pass on GB10 — the R2/merge kernels have still never executed; the
next val run must show both R2 cosines >= 0.99 and the two-stage
graph-replay result. The device gate and mover are proven only in
simulation (meta) and by source pins; their first hardware exercise is
that same run.

## Post-boot17 fix: serve wiring — every hook binding patched, fail-closed self-check (this commit)

The first real serve attempt with `VLLM_EXL3_Q4KV=1` +
`--kv-cache-dtype nvfp4` (receipts/boot17/container.log) died at model
init: `model.py:866 -> sparse_attention.py:567` — the ORIGINAL
SM100-only selector ran, and the log showed no overlay line at all.

**Root cause (by-value name binding).** `models/minimax_m3/nvidia/model.py:128`
does `from ...sparse_attention import (..., select_main_backend_and_impl_cls,
...)`. Patching the `sparse_attention` module attribute (the old install)
never reaches `model.py` once it is imported: its local name holds the
original function object. The identical problem exists at other sites.

**Binding audit (every site in the image that holds one of the four
callables, verified by grep of the vllm-image):**

- `select_main_backend_and_impl_cls`: defined
  `common/sparse_attention.py:525`; **by-value importers**
  `nvidia/model.py:128` (the boot17 site) and `amd/model.py:133`.
  `select_main_impl_cls` (`:602`) delegates to the module-global name,
  so it picks up the patched attribute; it has no by-value importers.
- `fused_minimax_m3_qknorm_rope_kv_insert`: defined
  `vllm/_custom_ops.py:2711`; every call site resolves the attribute on
  the `vllm._custom_ops` MODULE OBJECT at call time
  (`nvidia/model.py:22/722/1055`, `amd/model.py:26`, `dspark.py:16` —
  all `from vllm import _custom_ops as ops`), so the single
  module-attribute patch covers them; nothing imports it by value.
- `minimax_m3_sparse_attn_decode` / `minimax_m3_sparse_attn`:
  defined `common/ops/sparse_attn.py:616/:529`; **by-value bindings** in
  `common/ops/__init__.py:11` (package re-export),
  `common/sparse_attention.py:35/:40` (the impl's forward calls these
  local names at :489/:507), `nvidia/sparse_attention_msa.py:29` (SM100
  Triton fallback decode, used at :399) and `amd/ops/sparse_attn.py:18`
  (ROCm re-export). The AMD AITER variant
  (`minimax_m3_sparse_attn_decode_aiter`) is a different callable, not a
  hook.

**install() rewrite (`src/vllm_exl3/minimax_q4kv.py`):**

1. A binding-site registry (`_REQUIRED_BINDING_SITES` /
   `_OPTIONAL_BINDING_SITES`) enumerates every (module, attribute, hook).
   install imports each module ITSELF (so the patch is robust to import
   order: a module imported earlier, by the plugin or by vLLM, is
   patched on its own attribute), then patches the attribute on EVERY
   importable site with ONE wrapper per hook.
2. Required sites (defining modules, the `common.ops` re-export,
   `nvidia/model.py`): import failure or a missing attribute raises a
   named RuntimeError — a partial overlay would silently reproduce
   boot17. Platform-specific sites (`nvidia/sparse_attention_msa.py`,
   `amd/model.py`, `amd/ops/sparse_attn.py`) are patched when
   importable and otherwise logged as left-to-upstream: the q4 overlay
   never routes on SM100 or ROCm (`_route_q4` returns False there).
3. One INFO log line lists every patched `module.attribute` (so the next
   boot log positively shows the overlay) plus the skipped ones.
4. Fail-closed self-check at the end: every importable binding site must
   carry the overlay marker, else a RuntimeError names the unpatched
   site. Install is idempotent (a marked wrapper is never re-wrapped;
   the upstream original is preserved as the delegation target).
5. The overlay backend class is now built lazily at first q4 routing
   (install no longer imports `vllm.utils.torch_utils` /
   `vllm.v1.kv_cache_interface` eagerly).

**Second serve blocker found and fixed — the impl's fp8 view.**
`MiniMaxM3SparseImpl.__init__` sets
`use_fp8_kv = is_quantized_kv_cache("nvfp4")` = True
(`common/sparse_attention.py:465` with `v1/kv_cache_interface.py:116`),
and the Triton impl's forward views the pool as fp8 before calling the
reader hooks (`:480`). The q4 container would therefore arrive at the
R2/prefill wrappers typed `float8_e4m3fn` — the exact val1
frontend-failure class at the kernel boundary. The overlay now re-views
a q4-geometry pool to raw uint8 bytes of the same storage
(`_q4_pool_bytes`, 1-byte dtypes only; wider dtypes refused) before
dispatch, and the R2 kernel boundary independently refuses non-uint8
pools. The W1 path receives the raw uint8 container (model.py hands
`self.kv_cache` over un-viewed; `bind_kv_cache` stores it as-is,
`attention_layer_base.py:27`), but the same normalization applies.

**Other nvfp4/SM100 init branches audited (item 5, file:line):**

- `nvidia/model.py:169` `_supports_qkv_fused_epilogue`: returns False off
  SM100 — the fused MXFP8 epilogue is disabled on GB10 and the layer
  takes the W1 op path we wrap. No action needed; this is the supported
  path.
- `nvidia/model.py:844` `kv_cache_dtype_str_to_dtype("nvfp4")` ->
  `torch.uint8` (`utils/torch_utils.py:54`): the pool container is
  uint8, matching `_is_q4_pool`'s byte-level contract.
- `nvidia/model.py:1078-79`: the W1 call passes `_k_scale`/`_v_scale`
  only when `use_nvfp4_kv`; `set_default_quant_scales`
  (`layers/attention/attention.py`) pins them to 1.0, so the overlay's
  unit-global-scale gate passes by construction.
- Indexer: `indexer_kv_dtype` does NOT mirror `--kv-cache-dtype` —
  `resolve_indexer_kv_dtype("bf16")` (`config/attention.py:187-191`)
  only substitutes "auto", so with nvfp4 main KV the indexer keeps its
  own configured dtype (bf16 default / fp8) and its Triton impl. An
  explicit `indexer_kv_dtype="nvfp4"` request raises upstream's own
  named NotImplementedError (`common/indexer.py:534-545`, SM100/103/107
  only) — fail-closed by upstream, matching milestone-1 scope (the
  indexer cache stays fp8/bf16). No overlay action.
- Pool sizing: the spec arithmetic is consumed generically with no
  platform check (`v1/kv_cache_interface.py:526-560` num_head_slots /
  state_content_bytes; `v1/worker/utils.py:474-481` calls the SELECTED
  backend's `customize_spec`, i.e. ours) and the dtype gate
  (`v1/attention/backend.py:111-116`) passes because the overlay backend
  publishes `supported_kv_cache_dtypes=["nvfp4"]`. Block size stays 128
  (`get_supported_kernel_block_sizes` is unconditional `[128]`).
- `topk_blocks`: the q4 route ignores topk at selection (any width is
  served by the same wrapper); the only topk==16 assumption in the image
  is the SM100 MSA indexer path we never touch.
- Env reach: EngineCore is started via multiprocessing spawn
  (`v1/engine/utils.py:325` `proc.start()`), which inherits
  `os.environ` wholesale; the Ray path copies every `VLLM_`-prefixed
  variable (`ray/ray_env.py:40` `DEFAULT_ENV_VAR_PREFIXES`). The
  APIServer's "Unknown vLLM environment variable detected" warning is
  the envs allowlist, not a propagation failure. register() calls
  install() before every other installer (`__init__.py:50-53`); after
  this commit the order is irrelevant because install imports the
  binding modules itself.

**Tests (red venv):** 9 new (51/51 green in
`tests/test_minimax_q4kv.py`): the real import path fails closed naming
`vllm._custom_ops` when vllm is absent; a full fake-module simulation of
the pre-install by-value state (every binding holds the upstream
original) is patched at every required site including the boot17
`nvidia.model` site, one wrapper object per hook, originals preserved as
delegation targets, the INFO log lists every patched and skipped site,
and non-q4 traffic still delegates to the original; the self-check
raises naming the site on upstream drift (missing attribute) and on an
unimportable required module; install is idempotent; the fp8-view
normalization is pinned at the helper, wrapper (W1 and R2, same
storage/data_ptr, real fp8 pools still fall through) and kernel-boundary
(non-uint8 pool refused, nothing launched) levels. Default-off is
unchanged: with the env unset, register() imports and patches nothing
(pinned by the pre-existing default-off tests). Full suite: **622
passed, 39 skipped, the same 8 pre-existing failures** — failure set
identical to the baseline.

**Unverified on this VPS (no GPU, no triton, no vLLM install):** the
real-container import of `vllm.models.minimax_m3.nvidia.model` at install
time (it was already imported in the boot17 process per the root cause,
so the import is expected to be a no-op there); whether the SM100 MSA /
AMD modules import on GB10 (either way is handled and logged); the real
allocator's pool tensor shape/strides matching `_is_q4_pool`'s geometry
(the harness pinned the layout from the vendor csrc, but the
allocator-produced container is only observable at a real boot); and the
whole end-to-end serve — the next GB10 boot with
`VLLM_EXL3_Q4KV=1 --kv-cache-dtype nvfp4` must show the new install INFO
line (every binding listed), the q4 selector line, and 576 B/token/layer
pool sizing, or fail with one of the named refusals.
