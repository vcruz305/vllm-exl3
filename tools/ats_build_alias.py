#!/usr/bin/env python3
"""Build the 64-byte-aligned ATS alias side file for a DSV4.1 EXL3 pack.

The v2 pack lays every expert trellis on a ``12 mod 16`` file offset, which the
int16 trellis kernels reject for a zero-copy device pointer.  This gathers the
selected tensors, in checkpoint order, into one file where every tensor starts
on a 64-byte boundary, and writes the manifest the vllm-exl3 ATS path reads.

    python3 ats_build_alias.py --model-dir ~/models/dsv41-1.62bpw-v2 \
        --out ~/dsv41-vllm-exl3/alias --layers 38,39 [--mtp 0,1,2] \
        [--align 64]

    # or, per-expert cold set from a routing histogram:
    python3 ats_build_alias.py --model-dir ~/models/dsv41-1.62bpw-v3 \
        --out ~/dsv41-vllm-exl3/alias-cold \
        --select-npz route_stats.npz --budget-gib 7.43 [--rows 7]

Read-only on the pack; writes only under ``--out``.  Safe to re-run: finished
output is verified and kept, a failed build leaves ``.tmp`` behind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

CHUNK = 8 << 20
_PROJ = ("w1", "w2", "w3")


def header(path: str):
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        return n, json.loads(fh.read(n))


def _mtp_layer_offset(model_dir: str) -> int | None:
    """``num_hidden_layers`` of the main stack: mtp.<i> == layers.<offset + i>."""
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        return None
    for holder in (cfg.get("text_config") or {}, cfg):
        n = holder.get("num_hidden_layers")
        if isinstance(n, int):
            return n
    return None


# Checkpoint expert-projection families: the historical layout is
# layers.N.ffn.experts (DSV4.1); MiniMax-M3 stores its routed experts under
# layers.N.block_sparse_moe.experts. Both must resolve for the
# routing-histogram cold-set path (--select-npz) and the byte scan.
_TRELLIS_RE = re.compile(
    r"(?:^|\.)((?:layers|mtp)\.\d+)\.(ffn|block_sparse_moe)\.experts\.(\d+)\.(w1|w2|w3)\.trellis$"
)


def _scan_expert_bytes(model_dir: str, weight_map: dict[str, str]):
    """``{(layer_key, eid): (bytes, [tensor keys])}`` for every routed expert.

    Read-only; only the safetensors headers are parsed, not the payload.
    """
    by_shard: dict[str, list[str]] = {}
    for k in weight_map:
        m = _TRELLIS_RE.search(k)
        if m:
            by_shard.setdefault(weight_map[k], []).append(k)
    out: dict[tuple[str, int], list] = {}
    for shard, keys in by_shard.items():
        n, hdr = header(os.path.join(model_dir, shard))
        for k in keys:
            m = _TRELLIS_RE.search(k)
            assert m is not None, k
            b, e = hdr[k]["data_offsets"]
            ent = out.setdefault((m.group(1), int(m.group(3))), [0, []])
            ent[0] += e - b
            ent[1].append(k)
    return {k: (v[0], v[1]) for k, v in out.items()}


def select_cold(
    npz_path: str,
    budget_bytes: int,
    expert_bytes: dict[tuple[str, int], tuple[int, list[str]]],
    rows_per_step: int = 7,
    log=print,
) -> dict[str, list[int]]:
    """Coldest experts first until ``budget_bytes`` of trellis are aliased.

    Per-token routing probability is ``g_e = counts[row, e] / calls[row]``
    (``counts`` sums the ``top_k`` entries of every token, ``calls`` counts
    tokens), so the probability that expert ``e`` is touched by one step of
    ``rows`` tokens is ``q_e = 1 - (1 - g_e)**rows``.  Ranking is by ``q_e``,
    tie-broken by ``g_e`` then (layer, eid) for determinism.
    """
    import numpy as np

    z = np.load(npz_path, allow_pickle=False)
    counts = z["counts_0"] if "counts_0" in z else z["counts"]
    calls = z["calls_0"] if "calls_0" in z else z.get("calls")
    rows = z["rows_0"] if "rows_0" in z else z.get("rows")
    meta = json.loads(str(z["meta"])) if "meta" in z else {}
    offset = int(meta.get("mtp_offset", 40))
    n_rows = counts.shape[0]

    def layer_key(row: int) -> str:
        return f"mtp.{row - offset}" if row >= offset else f"layers.{row}"

    cand: list[tuple[float, float, str, int, int]] = []
    for row in range(n_rows):
        # ``rows`` is the TOKEN count for this row (counts sums top_k per
        # token), ``calls`` is the number of module invocations.  g must be a
        # per-token probability: counts/tokens.  (The lane's original
        # counts/calls divides by invocations and yields g = 16-59 for a top-6
        # router, i.e. it ranks prefill-heavy rows as hot.)
        n_tok = float(rows[row]) if rows is not None else 0.0
        if n_tok <= 0 and calls is not None:
            n_tok = float(calls[row])
        if n_tok <= 0:
            continue
        lay = layer_key(row)
        for eid in range(counts.shape[1]):
            if (lay, eid) not in expert_bytes:
                continue
            g = float(counts[row, eid]) / n_tok
            if g <= 0.0:
                g = 1e-12
            q = 1.0 - (1.0 - g) ** rows_per_step
            cand.append((q, g, lay, eid, expert_bytes[(lay, eid)][0]))
    cand.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

    sel: dict[str, list[int]] = {}
    used = 0
    n_layers = 0
    for q, p, lay, eid, nb in cand:
        if used + nb > budget_bytes:
            continue
        used += nb
        sel.setdefault(lay, []).append(eid)
        if len(sel[lay]) == 1:
            n_layers += 1
    log(
        f"selected {sum(len(v) for v in sel.values())} experts over {n_layers} "
        f"layers, {used / 2**30:.3f} GiB (budget {budget_bytes / 2**30:.3f} GiB, "
        f"{len(cand)} candidates)"
    )
    for lay in list(sel)[:6]:
        eids = sel[lay]
        qs = [c[0] for c in cand if c[2] == lay and c[3] in set(eids)]
        log(f"  {lay}: {len(eids)} experts, q_e {min(qs):.4f}..{max(qs):.4f}")
    for lay in sorted(sel):
        sel[lay].sort()
    return sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="", help="comma list of layers.N to alias")
    ap.add_argument("--mtp", default="", help="comma list of mtp.N to alias")
    ap.add_argument(
        "--embed",
        action="store_true",
        help=(
            "also gather the text embedding (embed.weight) here.  v3 puts it on "
            "an 8-mod-16 file offset, so it cannot be ATS-aliased in place; a "
            "1.23 GiB copy on a 64-byte boundary removes that CUDA allocation."
        ),
    )
    ap.add_argument("--align", type=int, default=64)
    ap.add_argument("--file", default="experts.bin")
    ap.add_argument(
        "--experts-json",
        default="",
        help='explicit per-expert set, JSON {"layers.38": [0, 7, ...], "mtp.1": [...]}',
    )
    ap.add_argument(
        "--select-npz",
        default="",
        help="routing histogram (route_stats.npz from VLLM_EXL3_ROUTE_STATS=1); "
        "alias the COLDEST experts until --budget-gib",
    )
    ap.add_argument("--budget-gib", type=float, default=0.0)
    ap.add_argument(
        "--rows",
        type=int,
        default=7,
        help="rows per decode step used for the touch probability (default 7)",
    )
    ap.add_argument(
        "--select-only",
        action="store_true",
        help="write selection.json and exit without building the side file",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    align = a.align
    assert align >= 16 and (align & (align - 1)) == 0, "align must be a power of two >= 16"

    idx_path = os.path.join(a.model_dir, "model.safetensors.index.json")
    with open(idx_path, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]

    sel: dict[str, list[int]] | None = None
    if a.experts_json:
        with open(a.experts_json, encoding="utf-8") as fh:
            sel = {
                str(k): sorted({int(x) for x in v})
                for k, v in json.load(fh).items()
                if v
            }
        print(f"expert set from {a.experts_json}: {sum(len(v) for v in sel.values())} experts")
    elif a.select_npz:
        if a.budget_gib <= 0:
            print("--select-npz needs --budget-gib > 0", file=sys.stderr)
            return 2
        expert_bytes = _scan_expert_bytes(a.model_dir, weight_map)
        print(f"scanned {len(expert_bytes)} routed experts in the pack")
        sel = select_cold(
            a.select_npz,
            int(a.budget_gib * 2**30),
            expert_bytes,
            rows_per_step=a.rows,
        )
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "selection.json"), "w", encoding="utf-8") as fh:
            json.dump(sel, fh, indent=1, sort_keys=True)
        if a.select_only:
            print(f"wrote {os.path.join(a.out, 'selection.json')}")
            return 0

    prefixes = [f"layers.{p.strip()}." for p in a.layers.split(",") if p.strip()]
    prefixes += [f"mtp.{p.strip()}." for p in a.mtp.split(",") if p.strip()]

    # ``embed.weight`` is a lone tensor, not a per-layer prefix: it goes through
    # the same gather so that its offset lands on the alignment grid (the v3
    # pack has it at 8 mod 16 and the ATS view rejects that).
    embed_key = None
    if a.embed:
        from vllm_exl3.ats_alias import MINIMAX_EMBED_KEY

        if "embed.weight" in weight_map:
            embed_key = "embed.weight"
        elif MINIMAX_EMBED_KEY in weight_map:
            # MiniMax-M3 VL wrapper: the text embedding lives under its
            # language_model.model.* checkpoint name.
            embed_key = MINIMAX_EMBED_KEY
        else:
            print("--embed: no embedding tensor found in the index", file=sys.stderr)
            return 2

    if not prefixes and embed_key is None and sel is None:
        print("nothing selected", file=sys.stderr)
        return 2

    if sel is not None:
        from vllm_exl3.ats_alias import EXPERT_CKPT_PREFIXES

        keys = []
        missing = []
        for lay, eids in sel.items():
            for eid in eids:
                for p in _PROJ:
                    k = None
                    for fam in EXPERT_CKPT_PREFIXES:
                        cand = f"{lay}.{fam}.{eid}.{p}.trellis"
                        if cand in weight_map:
                            k = cand
                            break
                    if k is None:
                        missing.append(f"{lay}.ffn.experts.{eid}.{p}.trellis")
                    else:
                        keys.append(k)
        if missing:
            print(
                f"{len(missing)} selected tensors absent from the pack, e.g. {missing[:3]}",
                file=sys.stderr,
            )
            return 2
        keys.sort()
    else:
        keys = sorted(
            k
            for k in weight_map
            if k.endswith(".trellis") and any(p in k for p in prefixes)
        )
    if embed_key is not None:
        keys.append(embed_key)
        keys.sort()
    if not keys:
        print("no trellis tensors matched", file=sys.stderr)
        return 2

    hdrs: dict[str, tuple] = {}
    by_shard: dict[str, list[str]] = {}
    for k in keys:
        by_shard.setdefault(weight_map[k], []).append(k)

    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, a.file)
    total = 0
    plan = []
    for shard in sorted(by_shard):
        path = os.path.join(a.model_dir, shard)
        if shard not in hdrs:
            hdrs[shard] = header(path)
        n, hdr = hdrs[shard]
        for k in by_shard[shard]:
            b, e = hdr[k]["data_offsets"]
            nb = e - b
            plan.append((shard, 8 + n + b, nb, k, hdr[k]["dtype"], hdr[k]["shape"]))
            total += nb
    total = (total + align - 1) // align * align
    print(f"{len(plan)} tensors, {total / 2**30:.3f} GiB payload, align={align}")

    if a.dry_run:
        for shard, off, nb, k, dt, sh in plan[:5]:
            print(f"  {k} {dt}{sh} {nb} abs={off} {off % align}")
        return 0

    free = os.statvfs(a.out).f_bavail * os.statvfs(a.out).f_frsize
    if free < total * 1.02:
        print(f"not enough space: need {total / 2**30:.1f} GiB, have {free / 2**30:.1f}", file=sys.stderr)
        return 3

    tensors: dict[str, list] = {}
    t0 = time.time()
    with open(out_path + ".tmp", "wb") as out:
        cur = 0
        for i, (shard, abs_off, nb, k, dt, sh) in enumerate(plan):
            off = (cur + align - 1) // align * align
            if off > cur:
                out.write(b"\0" * (off - cur))
            with open(os.path.join(a.model_dir, shard), "rb") as src:
                src.seek(abs_off)
                left = nb
                while left:
                    buf = src.read(min(CHUNK, left))
                    if not buf:
                        raise RuntimeError(f"short read {k}")
                    out.write(buf)
                    left -= len(buf)
            tensors[k] = [off, nb, dt, list(sh)]
            cur = off + nb
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(plan)} {cur / 2**30:.2f} GiB {cur / 2**20 / el:.0f} MiB/s", flush=True)
        pad = (cur + align - 1) // align * align - cur
        if pad:
            out.write(b"\0" * pad)
        out.flush()
        os.fsync(out.fileno())

    os.replace(out_path + ".tmp", out_path)

    # verify: every offset on the grid, and a byte-exact spot check per shard
    bad = [k for k, v in tensors.items() if v[0] % align]
    if bad:
        print(f"grid violation: {bad[:3]}", file=sys.stderr)
        return 4
    checked = 0
    with open(out_path, "rb") as out:
        for shard, abs_off, nb, k, dt, sh in plan[:: max(1, len(plan) // 32)]:
            off = tensors[k][0]
            with open(os.path.join(a.model_dir, shard), "rb") as src:
                src.seek(abs_off)
                a_bytes = src.read(min(nb, 1 << 20))
            out.seek(off)
            b_bytes = out.read(len(a_bytes))
            if hashlib.sha256(a_bytes).digest() != hashlib.sha256(b_bytes).digest():
                print(f"verify mismatch {k}", file=sys.stderr)
                return 5
            checked += 1

    man = {
        "model_dir": os.path.abspath(a.model_dir),
        "file": a.file,
        "align": align,
        "bytes": os.path.getsize(out_path),
        "tensors": tensors,
        "prefixes": prefixes,
        # Per-expert cold set: when present the runtime takes this as the sole
        # authority on what is aliased (ats_alias.layer_aliases).
        **({"selection": "expert_list", "experts": sel} if sel is not None else {}),
        # DSpark builds its draft layers as ``model.layers.<num_hidden_layers + i>``
        # while the checkpoint names them ``mtp.<i>``; record the offset so the
        # runtime can resolve one to the other.
        "mtp_layer_offset": _mtp_layer_offset(a.model_dir),
        "embed_key": embed_key,
        "verified_tensors": checked,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh)
    print(
        f"wrote {out_path} {man['bytes'] / 2**30:.3f} GiB, {len(tensors)} tensors, "
        f"{checked} spot-checked, {time.time() - t0:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())