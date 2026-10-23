#!/usr/bin/env python3
"""PROVISIONAL cold-expert alias selection for the MiniMax-M3.1 EXL3 pack.

The engine memory model needs part of the routed experts streamed from NVMe
via the ATS zero-copy alias path (ats_alias.py + tools/ats_build_alias.py):
~5.3 GiB at 32K context and ~27 GiB at 262K (parent arithmetic, to be
re-measured on the first real boot).  A real cold set is ranked by measured
routing frequency (route_stats.py histogram -> ats_build_alias.py
--select-npz).  No routing statistics exist until the first real serve, so
this tool writes a PROVISIONAL selection from the pack census bytes only:

* spread evenly across the 57 MoE layers (round-robin, layer 3..59 in
  order, one candidate per layer per turn), and
* within each layer prefer the (2,2,2) experts first (smallest trellis,
  finest budget granularity), then (2,3,2), then (3,3,3), ties broken by
  expert id, and
* deterministic: same census + budget -> byte-identical selection.

The output is PROVISIONAL: regenerate from a real route histogram after the
first boot (see GPU_JOB-2.md / CHECKIN notes for the exact
VLLM_EXL3_ROUTE_STATS recipe) and rebuild the side file.

Inputs/outputs
--------------
``--census`` is the pack header census (per-tensor dtype/shape/bytes), NOT
weights; the VPS copy lives at /root/work/minimax-m31/pack/census_tensors.json.
Only ``.trellis`` bytes are counted (that is what the alias side file gathers;
suh/svh/mul1 stay resident).

Writes (under ``--out``):
  minimax_provisional_alias_selection.json      pure {"layers.N": [eids]},
      the exact format ats_build_alias.py --experts-json consumes;
  minimax_provisional_alias_selection.meta.json provenance (budget, achieved
      bytes, per-layer spread, per-K-triple counts, PROVISIONAL banner).

    python3 tools/minimax_provisional_alias.py \
        --census /root/work/minimax-m31/pack/census_tensors.json \
        --budget-gib 5.5 --out /tmp/alias5.5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

# Real-pack constants (parent census; CAMPAIGN_STATE): 57 MoE layers (3..59),
# 128 routed experts each, per-expert trellis K triples (w1, w2, w3) drawn from
# {(2,2,2), (2,3,2), (3,3,3)}.
REAL_LAYERS = 57
REAL_EXPERTS = 128
SUPPORTED_K = (2, 3)

_NAME_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\."
    r"(w[123])\.trellis$"
)


def parse_census(census: dict) -> dict[int, dict[int, dict[str, dict]]]:
    """``{layer: {eid: {proj: {"bytes": n, "k": k}}}}`` for routed trellises."""
    out: dict[int, dict[int, dict[str, dict]]] = {}
    for name, ent in census.items():
        m = _NAME_RE.match(str(name))
        if not m:
            continue
        layer, eid, proj = int(m.group(1)), int(m.group(2)), m.group(3)
        shape = list(ent.get("shape") or [])
        if len(shape) != 3:
            raise ValueError(f"{name}: unexpected trellis shape {shape}")
        k = int(shape[-1]) // 16
        if k not in SUPPORTED_K:
            raise ValueError(f"{name}: trellis K {k} outside {SUPPORTED_K}")
        out.setdefault(layer, {}).setdefault(eid, {})[proj] = {
            "bytes": int(ent.get("bytes") or 0),
            "k": k,
        }
    if not out:
        raise ValueError("census has no routed-expert trellis tensors")
    return out


def validate(experts: dict[int, dict[int, dict[str, dict]]],
             expect_layers: int, expect_experts: int) -> None:
    layers = sorted(experts)
    if len(layers) != expect_layers:
        raise ValueError(
            f"census has {len(layers)} MoE layers, expected {expect_layers}"
        )
    if layers != list(range(layers[0], layers[0] + expect_layers)):
        raise ValueError(f"MoE layer ids are not contiguous: {layers[:5]}...")
    for layer in layers:
        eids = experts[layer]
        if len(eids) != expect_experts:
            raise ValueError(
                f"layer {layer} has {len(eids)} experts, expected {expect_experts}"
            )
        if sorted(eids) != list(range(expect_experts)):
            raise ValueError(f"layer {layer} expert ids are not 0..{expect_experts-1}")
        for eid, projs in eids.items():
            if sorted(projs) != ["w1", "w2", "w3"]:
                raise ValueError(f"layer {layer} expert {eid}: projections {sorted(projs)}")


def expert_table(experts: dict[int, dict[int, dict[str, dict]]]) -> list[dict]:
    """Flat candidate table with per-expert trellis bytes and K triple."""
    table = []
    for layer in sorted(experts):
        for eid in sorted(experts[layer]):
            projs = experts[layer][eid]
            nbytes = sum(p["bytes"] for p in projs.values())
            ktriple = (projs["w1"]["k"], projs["w2"]["k"], projs["w3"]["k"])
            table.append(
                {"layer": layer, "eid": eid, "bytes": nbytes, "k": ktriple}
            )
    return table


def select_provisional(table: list[dict], budget_bytes: int) -> dict:
    """Round-robin across layers, smallest-experts-first within each layer.

    Deterministic: candidates are ordered by (bytes, eid) inside each layer
    and layers are visited in ascending order, one pick per visit.
    """
    by_layer: dict[int, list[dict]] = {}
    for cand in table:
        by_layer.setdefault(cand["layer"], []).append(cand)
    for layer in by_layer:
        by_layer[layer].sort(key=lambda c: (c["bytes"], c["eid"]))

    sel: dict[int, list[int]] = {}
    used = 0
    exhausted = set()
    while len(exhausted) < len(by_layer):
        for layer in sorted(by_layer):
            if layer in exhausted:
                continue
            picked = False
            while by_layer[layer]:
                cand = by_layer[layer][0]
                if used + cand["bytes"] > budget_bytes:
                    # Smallest first: nothing later in this layer can fit
                    # either, but other layers may still have smaller experts.
                    by_layer[layer] = [
                        c for c in by_layer[layer] if used + c["bytes"] <= budget_bytes
                    ]
                    if not by_layer[layer]:
                        exhausted.add(layer)
                    break
                used += cand["bytes"]
                sel.setdefault(layer, []).append(cand["eid"])
                by_layer[layer].pop(0)
                picked = True
                break
            if not picked and not by_layer[layer]:
                exhausted.add(layer)
    return {"experts": {
                f"layers.{k}": sorted(v) for k, v in sorted(sel.items())
            },
            "used_bytes": used}


def _layer_index(key: str) -> int:
    """'layers.N' -> N (ats_build_alias.py --experts-json key format)."""
    return int(str(key).rsplit(".", 1)[1])


def summarize(sel: dict, table: list[dict]) -> dict:
    chosen = {
        (_layer_index(l), e) for l, eids in sel["experts"].items() for e in eids
    }
    per_layer = {
        l: len(eids) for l, eids in sorted(sel["experts"].items())
    }
    per_k: dict[tuple, int] = {}
    for cand in table:
        if (cand["layer"], cand["eid"]) in chosen:
            per_k[cand["k"]] = per_k.get(cand["k"], 0) + 1
    return {
        "n_experts": len(chosen),
        "used_bytes": sel["used_bytes"],
        "layers": len(per_layer),
        "per_layer": per_layer,
        "per_layer_min": min(per_layer.values()) if per_layer else 0,
        "per_layer_max": max(per_layer.values()) if per_layer else 0,
        "per_k_triple": {"x".join(str(k) for k in kk): n for kk, n in sorted(per_k.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="PROVISIONAL MiniMax-M3 cold-expert alias selection "
        "(census bytes only; regenerate from route stats after first serve)"
    )
    ap.add_argument("--census", required=True, help="census_tensors.json (headers only)")
    ap.add_argument("--budget-gib", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-layers", type=int, default=REAL_LAYERS)
    ap.add_argument("--expect-experts", type=int, default=REAL_EXPERTS)
    args = ap.parse_args()

    if args.budget_gib <= 0:
        print("--budget-gib must be positive", file=sys.stderr)
        return 2

    with open(args.census, encoding="utf-8") as fh:
        census = json.load(fh)
    experts = parse_census(census)
    validate(experts, args.expect_layers, args.expect_experts)
    table = expert_table(experts)
    total = sum(c["bytes"] for c in table)
    budget = int(args.budget_gib * 2**30)
    if budget > total:
        print(
            f"budget {budget/2**30:.2f} GiB exceeds the routed trellis total "
            f"{total/2**30:.2f} GiB",
            file=sys.stderr,
        )
        return 2

    sel = select_provisional(table, budget)
    s = summarize(sel, table)

    os.makedirs(args.out, exist_ok=True)
    sel_path = os.path.join(args.out, "minimax_provisional_alias_selection.json")
    meta_path = os.path.join(args.out, "minimax_provisional_alias_selection.meta.json")
    with open(sel_path, "w", encoding="utf-8") as fh:
        json.dump(sel["experts"], fh, indent=1, sort_keys=True)
    meta = {
        "provisional": True,
        "warning": (
            "PROVISIONAL census-bytes-only cold set (even layer spread, "
            "(2,2,2) experts preferred). NOT routing-frequency ranked. "
            "Regenerate with ats_build_alias.py --select-npz from a real "
            "route_stats.npz after the first serve, then rebuild the side file."
        ),
        "census": os.path.abspath(args.census),
        "budget_gib": args.budget_gib,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "routed_trellis_total_bytes": total,
        **s,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)

    spread = ", ".join(f"{l}:{n}" for l, n in list(s["per_layer"].items())[:6])
    print(
        f"PROVISIONAL selection: {s['n_experts']} experts over {s['layers']} layers, "
        f"{s['used_bytes']/2**30:.3f} GiB of {args.budget_gib} GiB budget"
    )
    print(f"per-K-triple: {s['per_k_triple']}")
    print(f"per-layer spread min/max: {s['per_layer_min']}/{s['per_layer_max']} ({spread} ...)")
    print(f"wrote {sel_path}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
