"""Drop the split vision q/k/v tensors an EXL3 pack ships next to the fused bf16 attn.qkv.

EXL3 packs for Qwen3.5-MoE / Ornith-1.5 store `model.visual.blocks.N.attn.{q,k,v}_proj.*`
trellis tensors alongside the original fused `attn.qkv.{weight,bias}` (bf16). vLLM's
`Qwen3_VisionTransformer` in `vllm/model_executor/models/qwen3_vl.py` holds only `attn.qkv`,
so `AutoWeightsLoader` raises `ValueError: There is no module or parameter named
'blocks.0.attn.k_proj'` unless `.attn.{q,k,v}_proj.` are ignored in `hf_to_vllm_mapper`.
Backup <file>.orig, idempotent, compile-checked.

usage: python3 patch_vllm_qwen3_5_vision_split.py <venv_or_site_packages_vllm>
"""
import glob
import os
import shutil
import sys

OLD = """class Qwen3_VisionTransformer(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            "attn.q.": ("attn.qkv.", "q"),
            "attn.k.": ("attn.qkv.", "k"),
            "attn.v.": ("attn.qkv.", "v"),
        }
    )
"""
NEW = """class Qwen3_VisionTransformer(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            "attn.q.": ("attn.qkv.", "q"),
            "attn.k.": ("attn.qkv.", "k"),
            "attn.v.": ("attn.qkv.", "v"),
        },
        orig_to_new_substr={
            ".attn.q_proj.": None,
            ".attn.k_proj.": None,
            ".attn.v_proj.": None,
        },
    )
"""


def _resolve_qwen3_vl_path(arg: str) -> str:
    if os.path.isfile(arg):
        return arg
    candidates = [
        os.path.join(arg, "model_executor", "models", "qwen3_vl.py"),
        os.path.join(arg, "vllm", "model_executor", "models", "qwen3_vl.py"),
    ]
    candidates.extend(
        sorted(
            glob.glob(
                os.path.join(
                    arg,
                    "lib",
                    "python*",
                    "site-packages",
                    "vllm",
                    "model_executor",
                    "models",
                    "qwen3_vl.py",
                )
            )
        )
    )
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0]


def main():
    if len(sys.argv) < 2:
        print("usage: python3 patch_vllm_qwen3_5_vision_split.py <venv_or_site_packages_vllm>")
        return 2
    path = _resolve_qwen3_vl_path(sys.argv[1])
    src = open(path, encoding="utf-8").read()
    if NEW in src or '".attn.k_proj.": None' in src:
        print(f"already patched: {path}")
        return 0
    n = src.count(OLD)
    if n != 1:
        print(f"ERROR: anchor found {n} times in {path}")
        return 1
    out = src.replace(OLD, NEW)
    compile(out, path, "exec")
    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
    open(path, "w", encoding="utf-8").write(out)
    print(f"patched {path} (backup {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
