"""Check the historical image before applying its explicitly requested overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

SOURCE = Path(__file__).resolve().parent
EXTENSIONS = ("sage_heterogeneous_ext.so", "sage_heterogeneous_dynamic.so",
              "sage_shared_miss.so", "libsg_rows.so")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(source=SOURCE):
    manifest = json.loads((source / "provenance.json").read_text())
    for name, record in manifest["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or digest(path) != record["sha256"]:
            raise ValueError(f"Bundled source differs: {name}")
    return manifest


def validate_base(root=Path("/"), source=SOURCE):
    manifest = verify_sources(source)
    for name, expected in manifest["base_preimages"].items():
        path = root / name.lstrip("/")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Historical runtime preimage differs: {name}; port explicitly")
    helper = root / "opt/dsv41-patch/sm120_page.py"
    if not helper.is_file():
        raise ValueError("The dedicated V4.1 base must supply sm120_page.py")
    return manifest


def install(extensions, root=Path("/"), source=SOURCE):
    # Validate every input before the first copy. Docker layer failure rolls back
    # partial writes; this tool is not a general host-environment installer.
    manifest = validate_base(root, source)
    native = [extensions / name for name in EXTENSIONS]
    if any(not p.is_file() or p.stat().st_size == 0 for p in native):
        raise ValueError("Build all four native extensions before installation")
    receipt_path = extensions / "build-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    expected_sources = {k: v["sha256"] for k, v in manifest["files"].items()
                        if k.startswith("native/")}
    if receipt.get("sources") != expected_sources:
        raise ValueError("Native build receipt belongs to different sources")
    if (receipt.get("torch"), receipt.get("cuda"), receipt.get("sm")) != (
            "2.13.0+cu130", "13.0", "12.1"):
        raise ValueError("Native build receipt has a different runtime identity")
    for path in native:
        if digest(path) != receipt["outputs"][path.name]:
            raise ValueError(f"Built extension changed: {path.name}")
    plan = []
    for name, record in manifest["files"].items():
        if name.startswith("runtime/"):
            target = "/opt/sage-offload/" + Path(name).name
        elif "installed_path" in record:
            target = record["installed_path"]
        else:
            continue
        plan.append((source / name, root / target.lstrip("/")))
    plan.extend((p, root / "opt/sage-offload" / p.name) for p in native)
    for src, target in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    pth = root / "usr/local/lib/python3.12/dist-packages/sage_offload.pth"
    pth.write_text("/opt/sage-offload\n")
    shutil.copyfile(receipt_path, root / "opt/sage-offload/build-receipt.json")
    return len(plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-in-container", action="store_true")
    parser.add_argument("--extensions", type=Path)
    args = parser.parse_args()
    if args.apply_in_container:
        if not Path("/.dockerenv").exists() or args.extensions is None:
            parser.error("Apply only in a disposable Docker build with --extensions")
        print(f"Installed {install(args.extensions)} files")
    else:
        validate_base()
        print("Historical preimages and bundled sources match; no files changed")
