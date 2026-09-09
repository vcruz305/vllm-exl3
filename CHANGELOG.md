# Changelog

## Unreleased

### Added

- `runtime_diagnostics()` exposes the effective EXL3 serving policy in a JSON-friendly form: selected MoE backend, native extension availability and ABI, per-bit native decode-row caps, fused scratch row capacity, fat-expert threshold, fat-kernel availability, and speculative schedule. This gives recipes a machine-checkable record of what actually loaded rather than relying on requested environment variables alone.
- `src/vllm_exl3/runtime_policy.py` adds independently implemented TP1-oriented policy helpers. Per-bit overrides (`VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K2/K3/K4`) make it possible to A/B K2 separately from K3/K4 instead of forcing one global native-row threshold across different trellis costs. `VLLM_EXL3_FUSED_TEMP_ROWS` exposes the requested fused scratch capacity while preserving the historical 2048-row default.
- `docs/provenance.md` records exact source/design provenance and establishes commit-message labels for copied/derived work, adapted designs, and independent implementations.

### Licensing

- vllm-exl3 moves forward under **AGPL-3.0-only** so improvements to modified network-served versions remain available to their users. The prior Apache-2.0 license text is retained in `LICENSE.APACHE-2.0`; all existing MIT/Apache third-party notices remain in `THIRD_PARTY_NOTICES.md` and `NOTICE`.
- The newer TP1 runtime-policy helpers were independently written around this project's APIs. Recent public MiaAI-Lab GLM serving work is credited as relevant design prior art; no post-relicense scheduler or dense-FP8 source was copied into these helpers.

### Fixed

- Dense EXL3 calls no longer launch exllamav3's cooperative trellis GEMM. Rows 17 to 144 take exllamav3's reconstruct+hgemm path (exact); rows up to 16 keep exllamav3's own dispatch (GEMV up to 2 rows, cooperative GEMM for 3 to 16 rows, which ran clean through the same 45-minute stress and where reconstruct would cost about 10x per call). On the vLLM nightly V2 model runner the cooperative GEMM wedged the engine (EngineCore at 100% CPU, GPU busy at idle power): deterministically on 33 to 144-token prefills and after 30 to 60 minutes of MTP k=2 decoding. A 4-worker MTP stress that wedged within 3 minutes ran clean for 45 minutes (1085 requests, 77 tok/s aggregate) once the routing was in place, with decode speed unchanged and TTFT on 32-token prompts about 0.25 s. `VLLM_EXL3_COOP_GEMM=1` restores the old dispatch; `VLLM_EXL3_RECONSTRUCT_MIN_ROWS` moves the reconstruct threshold. `VLLM_EXL3_PREFILL_SYNC` is no longer needed for this and stays as an optional knob.

- Fat-expert prefill path (`apply_exl3_batched_fat`, experts with more than `VLLM_EXL3_FAT_THRESHOLD` routed rows in a chunk): the branch for packs whose gate and up projections carry distinct `suh` rotations handed column slices of the shared `gate_up` scratch buffer to `ext.hgemm` and `ext.had_r_128`. Those kernels index contiguous row-major operands, so the expert output was uncorrelated with the reference (relative error 1.3, cosine 0.01 on a Qwen3.8-Flash-Next expert) while short prompts, which never reach that path, looked normal. Prompt log-likelihood over a 6000-token corpus was mean NLL 4.21 through vLLM against 0.94 through exllamav3 on the same pack; with the fix it is 0.943. The branch now runs on contiguous fp32 temporaries. Packs with a shared gate/up `suh` (fused gate_up quantization) were not affected. Regression test: `tests/test_fat_distinct_suh.py`.

- Native fused-MoE decode-row cap measured on GB10 (K2: 8 rows, K3/K4: 1) ships as an opt-in
  (`VLLM_EXL3_NATIVE_MOE_MEASURED_CAP=1`, or `VLLM_EXL3_NATIVE_MOE_MAX_ROWS=<n>`); the default keeps
  the dispatch contract of up to 8 rows. Receipt: `tools/receipts/ab_moe_gb10.json`.
- Add `Exl3EmbeddingMethod` for row-wise n-gram embedding tables
  (`ngram_embedding`), decoded through the compiled
  `exllamav3_ext.ngram_dequant` kernel or a pure-torch fallback
  (`VLLM_EXL3_NGRAM_KERNEL=ext|torch`).
- Add the `ngram_embedding` config spec (`bits`, `num_shards`,
  `rows_per_shard`, `num_heads`, `modules`) and its checkpoint layout
  (`shard_<i>.trellis`, `head_bias`, `head_offsets`, `head_vocab_sizes`,
  `layer_multipliers`); tensor parallel size 1 only.
- Extend `get_quant_method` with branches for `ParallelLMHead` and
  `VocabParallelEmbedding`, so EXL3 `lm_head` and n-gram tables resolve
  once a model passes `quant_config=` through to those layers.
- Accept tuple and `None` shard-id spans in the dense linear weight
  loader, for fused modules with more than one contiguous shard and for
  full-tensor (non-sharded) loads.
- Generalize `_check_moe_codebook_markers` to validate `mul1` markers
  for routed experts alongside `mcg`.
- Add `_exl3_routed_experts_loader`, a per-expert `load_weights` path for
  routed-experts layers that mirrors vLLM's own checkpoint-name
  resolution for one-tensor-per-expert EXL3 checkpoints.
- Pad dense EXL3 linear geometry to multiples of 128 so trellis tiles
  never spill past the real matrix dimension.
- Thread per-tensor codebook flags (mcg/mul1) through the fused
  `exl3_moe` launch arguments.
- Register the EXL3 custom ops so they trace opaquely under
  `torch.compile(fullgraph=True)` instead of breaking graph capture.
- Restore the CUDA-graph-safe fat-expert-sync guard (skip the device
  sync entirely when the row count cannot produce a fat expert) and the
  MTP/draft-expert quant-method delegate (prefer MXFP4, matching DSV4's
  own fp4 draft experts, before falling back to the non-routed delegate).
- Add `tools/patch_vllm_qwen4_exp` (vLLM quant_config plumbing for
  `Qwen4ExpForConditionalGeneration`), `tools/exl3_pack_tools`
  (pack scanning, config rewriting, and
  `regenerate_safetensors_index.py`), and `tools/verify_native_pack`
  (four pre-boot GPU correctness gates).
- Add attribution notices for ExLlamaV3's n-gram embedding codec and for
  vLLM's routed-experts loader / custom-op registration; thank
  turboderp for the Qwen3.8-Flash-Next-exl3 pack used to validate this
  release.

- Extend the native fused MoE ABI with local intermediate width (1024 or 2048)
  and optional input-clipped SwiGLU. Clamp the gate before SiLU and the up
  projection symmetrically; zero keeps plain SwiGLU. Preserve K2/K3/K4 support.
- Keep legacy native calls compatible and fall back when an older extension
  cannot implement the requested width or clipping. Rebuild `vllm_exl3_c` to
  obtain `P2B_MOE_ABI_VERSION=2`.
- Add CPU dispatch/compatibility tests and CUDA numerical/graph-replay fixtures.
  Spark performance and full-model TP1/TP2 qualification remain required; no
  default speculative-depth changes are made by the ABI update.
