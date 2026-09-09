# Provenance and adaptation policy

vllm-exl3 is community engineering built on prior open-source work. This file records where important ideas and code originated, what was copied or derived, and what was independently implemented for this plugin.

## Core upstreams

### ExLlamaV3

The EXL3 trellis format, codebooks, quantization math, and native packed execution model originate in [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) by Turboderp. The applicable MIT notice is reproduced in `THIRD_PARTY_NOTICES.md`.

### MiaAI-Lab GLM-5.3 serving work

The original routed-expert vLLM integration and the E2 fat-prefill kernel lineage came from [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks). Existing copied/derived files and the exact historical MIT commit used are documented in `THIRD_PARTY_NOTICES.md`.

The upstream repository later moved to AGPL-3.0. vllm-exl3 now also moves forward under AGPL-3.0, while retaining the original third-party MIT and Apache notices and preserving its earlier Apache-2.0 license text in `LICENSE.APACHE-2.0`.

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

The initial `runtime_policy.py` implementation is independently written and contains no copied scheduler or dense-FP8 code from the newer MiaAI-Lab commits. Future grouped-prefill work may intentionally derive from the historical MIT snapshot where appropriate; if so, the exact source commit and modifications will be recorded here and in `THIRD_PARTY_NOTICES.md`.

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
