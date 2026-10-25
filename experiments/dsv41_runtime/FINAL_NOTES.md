# Final SAGE 3.30 two-Spark campaign notes

The campaign is closed without promoting a serving configuration. The exact
SAGE 3.30 checkpoint served successfully on two DGX Sparks and passed the full
quality, identity, source-integrity, restoration, memory, and zero-swap gates at
approximately 262K actual input tokens. Decode remained at 8.90–9.06 tokens/s.
The target of at least 20 tokens/s, a result near 30 tokens/s, and qualification
near 600K actual input tokens were not reached.

## Last qualified result

One fresh TP2/EP2 deployment used the original checkpoint revision
`e831e9e4d6bfeafa6d630848296417b1393404a3`, an 80 GiB expert arena per rank,
2.5 GiB KV per rank, one sequence, and disabled host swap.

| Workload | Actual input tokens | Decode tokens/s | Time to first token |
|---|---:|---:|---:|
| Retrieval 1 | 262,294 | 9.049765 | 1,090.07 s |
| Retrieval 2 | 262,294 | 9.064167 | 1,126.56 s |
| Reasoning 1 | 262,232 | 8.913270 | 1,168.06 s |
| Reasoning 2 | 262,232 | 8.904144 | 1,163.61 s |

The model endpoint became ready 102.50 seconds after launch. Minimum host
available memory was 14,084,935,680 and 14,115,414,016 bytes on the two ranks.
No swap was used. All eight quality cases and all four long-context cells passed.
The frozen artifact manifest is
`d5f002c537d6f78287bc66c20781064d08a4ec8bb168152cd525f99827cc2c07`;
the deployment manifest is
`1c96c5dffb5557ca61eb056c1a7aa761351dca881d51bad33125b3bd89f14a31`.
This is one deployment, so it does not meet the three-deployment confirmation
gate.

## Follow-up results

The original three-token speculative configuration with four-row target
verification completed the historical 64-token comparison exactly. Three
256-token diagnostic requests measured 9.28908, 9.19855, and 8.09612 tokens/s,
with acceptance rates of 56.84%, 59.42%, and 47.17%. Long-output repeatability
failed, so the result was rejected. Its artifact manifest is
`3ddd49c7c460de94abea750f7123c895bb788eb340aaaa31e9957e64c634bc08`.

A resident-first expert planner passed its component tests and reduced one
controlled mixed-cache transaction. Its full-model requests measured 8.95151,
9.33811, and 8.64922 tokens/s. The 64-token comparison passed, but long-output
repeatability failed and there was no clear speed gain. Its artifact manifest is
`dd271b75f17ac4021299c32d73e6ce63315e4e7a5b8185c4183ab57eede75af1`.

A process-local `MADV_HUGEPAGE` component produced more than 99% huge-page
coverage on a 708,968,448-byte arena on both ranks while retaining exact extent,
zero swap, source integrity, and restoration. The full-model version timed out
on its first 4,096-input/64-output request before generation or post-request
arena evidence. It has no TPS result and was rejected. Component and full-model
artifact manifests are
`8ee7dddff1732fbd328d718f3dbbff380efdfc656659175e10c7d1e572a05a41`
and `7a48ac50a6f54b8f38217f6a09ae38f51f869dbf59969d8be86b7641f2f7416d`.

## Findings worth retaining

- First expert misses accounted for 94.0% and 93.3% of measured read wait on
  the two ranks. Second misses were usually ready, so increasing the existing
  two-slot queue did not address the main delay.
- Reusing native read buffers reduced the median whole-read/four-hash stage by
  roughly 32–50%. Splitting retained-buffer reads and hashes across four tasks
  added only about 0–4% in the same component measurement.
- Best-fit allocation replay preserved all 42 recorded phase-state hashes but
  predicted only 0.523% and 0.349% fewer loaded bytes. Fragmentation fixes alone
  were too small for the target.
- A six-row native verification component passed the original 152 cases plus 24
  graph panels per rank. Full-model profiling then showed target verification,
  service gates, and collectives dominating the speculative path. Optimizing the
  draft graph alone was unlikely to produce the required gain.
- Cache-first prefill changed floating-point addition order even when individual
  expert contributions matched. Fixed route-slot reduction remained a component
  candidate; its first full fixture was rejected after the test attempted a
  graph lease in prefill mode. It never received serving qualification.

## Reproduction boundary

These notes extend the historical bundle in this directory. The bundled source
still depends on the retained V4.1 image and recorded Torch/CUDA/ExLlamaV3
revisions described in [README.md](README.md). Current main, a public base image,
600K context, concurrent serving, and a repeated 256K deployment have not been
qualified. The original checkpoint bytes were never requantized or modified.
