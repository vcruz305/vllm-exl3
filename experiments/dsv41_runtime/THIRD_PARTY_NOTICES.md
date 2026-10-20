# Source provenance and licenses

This contribution redistributes serving source, not model weights. Original
contribution code follows the repository's AGPL-3.0-only license. Existing root
notices and attribution remain in force.

## ExLlamaV3

`native/mixed_k/` and `native/dynamic/` contain adapted serving kernels and their
transitive headers from
[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3/tree/be57335b087e4f001c5caae061544df3c06ba01e),
revision `be57335b087e4f001c5caae061544df3c06ba01e`, under MIT. Each directory retains
`LICENSE-exllamav3`. The original paths are `exllamav3/exllamav3_ext/` followed by
the path beneath each bundled `native/` directory. Local heterogeneous bindings,
per-expert K dispatch and the dynamic resident-grid changes are recorded by the
artifact hashes in `provenance.json`; they are not claimed as unchanged upstream
files. Only the headers needed by these serving kernels are included.

## vLLM and the dedicated V4.1 image

[vllm-project/vllm](https://github.com/vllm-project/vllm) is Apache-2.0. The copied
files retain their SPDX and contributor copyright headers; the license is
included as `LICENSE.APACHE-2.0`.

The copied/adapted files are:

- `vllm_overlay/model_runner.py`: image path `vllm/v1/worker/gpu/model_runner.py`;
- `vllm_overlay/cudagraph_utils.py`: image path `vllm/v1/worker/gpu/cudagraph_utils.py`;
- `history/decoder-tail-image-r4/model.py`: image path `vllm/models/deepseek_v4_1/nvidia/model.py`.

Their source was the dedicated `vllm/vllm-openai:deepseekv41-flash-0909` runtime
line, reporting `0.1.dev20904+g179dd0fa9`, through the retained experimental images
identified in the README. The full upstream commit corresponding to that image
has not been independently resolved; no full SHA is invented. Exact installed
file hashes and original graph preimages are in `provenance.json`. Local changes
add bounded graph ownership/capture, input-buffer lifetime retention, reserved
KV-page clearing and the optional historical tail hooks.

## vllm-exl3 and recipe compatibility work

Historical plugin:
[vcruz305/vllm-exl3 at 8f4517e80416466fa4a3ad2eb28685021d39e95f](https://github.com/vcruz305/vllm-exl3/tree/8f4517e80416466fa4a3ad2eb28685021d39e95f),
AGPL-3.0-only with that repository's third-party notices.
`runtime/deepseek_v41_compat.py` preserves its compatibility helper;
`bootstrap/sitecustomize.py` adapts the retained recipe bootstrap. It depends on
the base image's `sm120_page.py`, which is not replaced by this bundle.

The mixed-K loader groundwork from @Blackwellboy and the disk-backed Engram
recipe contribution remain prerequisites; this bundle adds bounded expert
residency, fused runtime execution and native graph servicing to that work.
See the original recipe's
[third-party notices](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/blob/c3e6566/THIRD_PARTY_NOTICES.md).

DeepSeek checkpoint/model licensing is supplied with the original model. NVIDIA
CUDA, drivers and container components are external prerequisites and are not
redistributed here.
