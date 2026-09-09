# Provenance and adaptation policy

vllm-exl3 is community engineering built on prior open-source work. This file records where important ideas and code originated, what was copied or derived, and what was independently implemented for this plugin.

## Core upstreams

### ExLlamaV3

The EXL3 trellis format, codebooks, quantization math, and native packed execution model originate in [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) by Turboderp. The applicable MIT notice is reproduced in `THIRD_PARTY_NOTICES.md`.

### MiaAI-Lab GLM-5.3 serving work

The original routed-expert vLLM integration and the E2 fat-prefill kernel lineage came from [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks). Existing copied/derived files and the exact historical MIT commit used are documented in `THIRD_PARTY_NOTICES.md`.

The upstream repository later moved to AGPL-3.0. vllm-exl3 now also moves forward under AGPL-3.0-only, while retaining the original third-party MIT and Apache notices and preserving its earlier Apache-2.0 license text in `LICENSE.APACHE-2.0`.

## How new work is handled

For each upstream-informed change we use one of three labels in code review and commit messages:

1. **Copied/derived** — source-level reuse. The source repository, source commit, license, and local modifications must be identified.
2. **Adapted design** — the upstream design materially informs our implementation, but the code is independently written around vllm-exl3's own APIs and requirements. The commit identifies the upstream idea and explains what differs.
3. **Independent** — discovered and implemented from our own profiling/testing. Related prior art may still be cited for context.

Attribution is not used as a substitute for license compliance. When code is reused, its license terms and notices travel with it.

## Current TP1 optimization direction

Recent MiaAI-Lab work demonstrated the value of reducing host-driven expert dispatch, keeping effective serving configuration observable, and adapting speculative verification to workload behavior. vllm-exl3 is applying those lessons to a different target:

- single DGX Spark / TP1 as a first-class geometry,
- EXL3 K2 and mixed K2/K3 routed experts,
- reusable behavior for GLM, DeepSeek, Qwen, and other compatible vLLM fork models,
- bounded scratch-memory use because TP1 has less memory headroom than a two-node deployment,
- per-bit decode dispatch rather than assuming one row threshold is optimal for K2/K3/K4.

### Bounded K2/K3 grouped-prefill planning

MiaAI-Lab's E3 grouped fat-expert work (public commit `1a0feb0`) is prior art for the high-level observation that a large routed-MoE prefill can benefit from grouping work on-device rather than driving one fat expert at a time from the host. A verified historical snapshot containing the grouped implementation under MIT is `9cdf84570117a6afc203125a5c01ec61978c4e60`.

`src/vllm_exl3/prefill_policy.py` is **adapted-design / independently written code**, not copied or translated CUDA. It differs intentionally:

- it targets K2/K3 MCG candidates first rather than assuming the upstream K4 packed layout;
- it defines bounded row windows instead of sizing a persistent workspace for every possible routed row;
- it exposes conservative scratch accounting as an admission contract;
- it is architecture-neutral and does not claim that a grouped GPU executor exists;
- execution remains default-off until a local kernel passes parity, graph-replay, memory, and end-to-end TP1 qualification.

If a future CUDA executor actually reuses or derives from the historical MIT source, that commit must identify the exact upstream files/snapshot and enumerate the local K2/K3 changes in both Git history and `THIRD_PARTY_NOTICES.md`.

The earlier `runtime_policy.py` implementation is likewise independently written and contains no copied scheduler or dense-FP8 code from newer MiaAI-Lab commits.

## Commit-message standard

Commits that materially use outside work should include a provenance paragraph such as:

```text
Upstream provenance:
- Idea/source: <repository + commit/PR>
- License at source: <license>
- Relationship: copied/derived | adapted design | independent
- Local changes: <what we changed for TP1/EXL3/this plugin>
```

This keeps attribution visible in Git history even when README sections move over time.
