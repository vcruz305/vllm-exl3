import ast
import json
from pathlib import Path
import re
import tempfile
import unittest

from install_overlay import EXTENSIONS, SOURCE, digest, install, validate_base, verify_sources


class BundleTests(unittest.TestCase):
    def make_install_fixture(self, tmp):
        source, root, extensions = [Path(tmp) / name for name in ("source", "root", "extensions")]
        for path in (source / "runtime", root, extensions):
            path.mkdir(parents=True)
        module = source / "runtime/example.py"
        module.write_text("# synthetic runtime\n")
        base = root / "base.py"
        base.write_text("# historical preimage\n")
        helper = root / "opt/dsv41-patch/sm120_page.py"
        helper.parent.mkdir(parents=True)
        helper.write_text("# fixture\n")
        (root / "usr/local/lib/python3.12/dist-packages").mkdir(parents=True)
        (source / "provenance.json").write_text(json.dumps({
            "files": {"runtime/example.py": {"sha256": digest(module)}},
            "base_preimages": {"/base.py": digest(base)}}))
        for name in EXTENSIONS:
            (extensions / name).write_bytes(b"synthetic binary fixture")
        receipt = {"sources": {}, "torch": "2.13.0+cu130", "cuda": "13.0", "sm": "12.1",
                   "outputs": {name: digest(extensions / name) for name in EXTENSIONS}}
        (extensions / "build-receipt.json").write_text(json.dumps(receipt))
        return source, root, extensions

    def test_install_uses_verified_inputs_and_preserves_unrelated_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, root, extensions = self.make_install_fixture(tmp)
            self.assertEqual(install(extensions, root, source), 5)
            self.assertEqual((root / "opt/sage-offload/example.py").read_bytes(),
                             (source / "runtime/example.py").read_bytes())
            self.assertEqual((root / "base.py").read_text(), "# historical preimage\n")
            self.assertEqual((root / "usr/local/lib/python3.12/dist-packages/sage_offload.pth").read_text(),
                             "/opt/sage-offload\n")

    def test_changed_binary_is_rejected_before_installation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, root, extensions = self.make_install_fixture(tmp)
            (extensions / EXTENSIONS[0]).write_bytes(b"different build")
            with self.assertRaisesRegex(ValueError, "extension changed"):
                install(extensions, root, source)
            self.assertFalse((root / "opt/sage-offload").exists())

    def test_wrong_source_receipt_is_rejected_before_installation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, root, extensions = self.make_install_fixture(tmp)
            path = extensions / "build-receipt.json"
            receipt = json.loads(path.read_text())
            receipt["sources"] = {"native/other.cpp": "0" * 64}
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "different sources"):
                install(extensions, root, source)
            self.assertFalse((root / "opt/sage-offload").exists())

    def test_frozen_sources_match_recorded_bytes(self):
        self.assertGreater(len(verify_sources()["files"]), 80)

    def test_python_syntax_and_native_header_closure(self):
        for path in SOURCE.rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
        for path in (SOURCE / "native").rglob("*"):
            if path.suffix not in (".h", ".cpp", ".cu", ".cuh"):
                continue
            for name in re.findall(r'^\s*#\s*include\s*"([^"]+)"', path.read_text(), re.M):
                self.assertTrue((path.parent / name).is_file(), f"{path}: {name}")

    def test_runtime_dependencies_are_explicit(self):
        local = {p.stem for p in (SOURCE / "runtime").glob("*.py")}
        import sys
        external = {"torch", "triton", "vllm", "vllm_exl3", "exllamav3", "exllamav3_ext",
                    "sage_shared_miss", "sage_heterogeneous_ext", "sage_heterogeneous_dynamic"}
        for path in (SOURCE / "runtime").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [n.name for n in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    self.assertIn(name.split(".")[0], local | external | sys.stdlib_module_names, str(path))

    def test_base_mismatch_refuses_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "preimage differs"):
                validate_base(root)
            self.assertEqual(list(root.iterdir()), [])

    def test_matching_preimages_require_base_helper_and_detect_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            source = Path(tmp) / "source"
            source.mkdir()
            target = root / "a.py"
            target.parent.mkdir()
            target.write_text("historical fixture")
            (source / "provenance.json").write_text(json.dumps({"files": {}, "base_preimages": {"/a.py": digest(target)}}))
            with self.assertRaisesRegex(ValueError, "sm120_page"):
                validate_base(root, source)
            helper = root / "opt/dsv41-patch/sm120_page.py"
            helper.parent.mkdir(parents=True)
            helper.write_text("fixture")
            validate_base(root, source)
            target.write_text("different runtime")
            with self.assertRaisesRegex(ValueError, "preimage differs"):
                validate_base(root, source)
