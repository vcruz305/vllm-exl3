"""Build the four historical serving extensions inside the pinned CUDA image."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from install_overlay import SOURCE, verify_sources


def build(output):
    manifest = verify_sources()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["MAX_JOBS"] = "2"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1"
    import torch
    from torch.utils.cpp_extension import load

    if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
        raise ValueError("Use the recorded Torch 2.13.0+cu130 / CUDA 13.0 image")
    fast = ["-lineinfo", "-O3", "--use_fast_math", "-Xcudafe",
            "--diag_suppress=177", "-Xcudafe", "--diag_suppress=20012"]
    for name, folder, filenames, cflags, cuda, ldflags in [
        ("sage_heterogeneous_ext", "mixed_k", ["heterogeneous_bindings.cpp", "heterogeneous_moe.cu"], ["-O3"], fast, []),
        ("sage_heterogeneous_dynamic", "dynamic", ["heterogeneous_bindings.cpp", "heterogeneous_moe.cu"], ["-O3"], fast, []),
        ("sage_shared_miss", "service", ["shared_miss.cu", "native_service.cpp"], ["-O2", "-pthread"], ["-O2", "-lineinfo"], ["-ldl", "-pthread"]),
    ]:
        directory = output / (name + "-build")
        directory.mkdir()
        module = load(name=name, sources=[str(SOURCE / "native" / folder / f) for f in filenames],
                      extra_cflags=cflags, extra_cuda_cflags=cuda, extra_ldflags=ldflags,
                      build_directory=str(directory), verbose=True)
        shutil.copyfile(module.__file__, output / (name + ".so"))
    subprocess.run(["c++", "-std=c++17", "-O2", "-fPIC", "-shared", "-pthread",
                    str(SOURCE / "native/engram/row_service.cpp"), "-o",
                    str(output / "libsg_rows.so")], check=True)
    receipt = {"torch": torch.__version__, "cuda": torch.version.cuda, "sm": "12.1",
               "sources": {k: v["sha256"] for k, v in manifest["files"].items() if k.startswith("native/")},
               "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.so")}}
    (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    build(parser.parse_args().output)
