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
