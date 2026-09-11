#!/usr/bin/env python3
"""Preflight DeepSeek V4.1 EXL3 metadata and the recommended TP4+EP4 layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from vllm_exl3.deepseek_v41 import (
    is_deepseek_v41_source_quant,
    plan_deepseek_v41,
    source_weight_block_size,
)


def _load_quant_config(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config must contain a JSON object")
    quant = raw.get("quantization_config", raw)
    if not isinstance(quant, dict):
        raise ValueError("quantization_config must be an object")
    return quant


def inspect_config(quant: dict) -> dict[str, object]:
    config = SimpleNamespace(**quant)
    source = quant.get("non_routed_quantization")
    source = source if isinstance(source, dict) else {}
    issues: list[str] = []

    if str(quant.get("quant_method", "")).lower() != "exl3":
        issues.append("outer quant_method is not exl3")
    if quant.get("mtp_experts") != "source":
        issues.append("mtp_experts should be 'source' for the first V4.1 DSpark boot")
    if not is_deepseek_v41_source_quant(config):
        issues.append(
            "non_routed_quantization must identify deepseek_v4_fp8 with weight_block_size [32, 32]"
        )

    return {
        "ok": not issues,
        "issues": issues,
        "outer_quant_method": quant.get("quant_method"),
        "bits": quant.get("bits"),
        "codebook": quant.get("codebook"),
        "scope": quant.get("scope"),
        "mtp_experts": quant.get("mtp_experts"),
        "mtp_experts_start_layer": quant.get("mtp_experts_start_layer"),
        "source_quant_method": source.get("quant_method"),
        "source_weight_block_size": source_weight_block_size(config),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="model config.json or a JSON file containing quantization_config",
    )
    parser.add_argument(
        "--pure-tp",
        action="store_true",
        help="show the non-recommended pure-TP4 MoE layout instead of TP4+EP4",
    )
    args = parser.parse_args()

    report: dict[str, object] = {
        "layout": plan_deepseek_v41(expert_parallel=not args.pure_tp).to_dict(),
    }
    if args.config is not None:
        report["pack"] = inspect_config(_load_quant_config(args.config))

    print(json.dumps(report, indent=2, sort_keys=True))
    pack = report.get("pack")
    return 0 if not isinstance(pack, dict) or bool(pack.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
