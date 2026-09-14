"""Test ARM-specific CPU stubs in patch_exllamav3_aarch64.py.

Validates that the ARM patcher correctly generates CPU-disabled stubs for:
  - exl3_moe_cpu_has_avx512_bw(): must return false
  - exl3_moe_cpu_pool_stress(): must throw std::runtime_error (catchable exception)
    NOT std::abort() (process termination) or return 0

These are ARM-only additions to the patcher, not modifications to exl3.py.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path


def test_patch_generates_avx512_bw_stub() -> None:
    """ARM patcher must generate exl3_moe_cpu_has_avx512_bw() returning false."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create minimal test structure with all subdirs patcher expects
        ext_dir = tmpdir / "exllamav3" / "exllamav3_ext"
        ext_dir.mkdir(parents=True)
        (ext_dir / "bindings.cpp").write_text("// minimal binding")
        # Create subdirs that patcher will write to
        (ext_dir / "cpu").mkdir(exist_ok=True)
        (ext_dir / "parallel").mkdir(exist_ok=True)

        # Run patcher with --force to cross-compile on x86
        patcher_path = Path(__file__).parent.parent / "tools" / "patch_exllamav3_aarch64.py"
        result = subprocess.run(
            [
                "python",
                str(patcher_path),
                str(ext_dir),
                "--force",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Patcher failed: {result.stderr}"

        # Verify moe_mul1.cpp contains the new stub
        moe_mul1_path = ext_dir / "cpu" / "moe_mul1.cpp"
        assert moe_mul1_path.is_file(), f"moe_mul1.cpp not created at {moe_mul1_path}"

        content = moe_mul1_path.read_text()
        assert (
            "bool exl3_moe_cpu_has_avx512_bw() { return false; }" in content
        ), "exl3_moe_cpu_has_avx512_bw stub not found in moe_mul1.cpp"


def test_pool_stress_throws_catchable_exception() -> None:
    """Validate the actual function body, not matching text elsewhere."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ext_dir = Path(tmpdir) / "exllamav3" / "exllamav3_ext"
        ext_dir.mkdir(parents=True)
        (ext_dir / "bindings.cpp").write_text("// fixture")
        (ext_dir / "cpu").mkdir()
        (ext_dir / "parallel").mkdir()
        patcher = Path(__file__).resolve().parents[1] / "tools/patch_exllamav3_aarch64.py"
        result = subprocess.run(
            [sys.executable, str(patcher), str(ext_dir), "--force"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        content = (ext_dir / "cpu/moe_mul1.cpp").read_text()
        assert "#include <stdexcept>" in content
        match = re.search(
            r"int64_t exl3_moe_cpu_pool_stress\([^)]*\)\s*\{([^{}]*)\}",
            content,
        )
        assert match, "Missing or unexpectedly structured pool_stress function"
        body = " ".join(match.group(1).split())
        assert body == 'throw std::runtime_error("CPU MoE is disabled in this ARM64 build");'


def test_patch_preserves_other_cpu_stubs() -> None:
    """ARM patcher must preserve existing CPU stubs while adding new ones."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        ext_dir = tmpdir / "exllamav3" / "exllamav3_ext"
        ext_dir.mkdir(parents=True)
        (ext_dir / "bindings.cpp").write_text("// minimal binding")
        # Create subdirs that patcher will write to
        (ext_dir / "cpu").mkdir(exist_ok=True)
        (ext_dir / "parallel").mkdir(exist_ok=True)

        result = subprocess.run(
            [
                "python",
                "tools/patch_exllamav3_aarch64.py",
                str(ext_dir),
                "--force",
            ],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0

        moe_mul1_path = ext_dir / "cpu" / "moe_mul1.cpp"
        content = moe_mul1_path.read_text()

        # All existing stubs must still be present
        existing_stubs = [
            "bool exl3_moe_cpu_has_avx2() { return false; }",
            "bool exl3_moe_cpu_has_avx512_vnni() { return false; }",
            "bool exl3_moe_cpu_has_avx512_vbmi() { return false; }",
        ]
        for stub in existing_stubs:
            assert stub in content, f"Missing existing stub: {stub}"


def test_patch_cpu_stubs_count() -> None:
    """Verify total CPU stubs: 7 capability checks across all files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        ext_dir = tmpdir / "exllamav3" / "exllamav3_ext"
        ext_dir.mkdir(parents=True)
        (ext_dir / "bindings.cpp").write_text("// minimal binding")
        # Create subdirs that patcher will write to
        (ext_dir / "cpu").mkdir(exist_ok=True)
        (ext_dir / "parallel").mkdir(exist_ok=True)

        patcher_path = Path(__file__).parent.parent / "tools" / "patch_exllamav3_aarch64.py"
        result = subprocess.run(
            [
                "python",
                str(patcher_path),
                str(ext_dir),
                "--force",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Count all "return false" stubs across all generated files
        total_capability_checks = 0

        # avx2_target.cpp: is_avx2_supported, is_f16c_supported
        avx2_path = ext_dir / "avx2_target.cpp"
        if avx2_path.is_file():
            total_capability_checks += avx2_path.read_text().count("{ return false; }")

        # avx512_target.cpp: is_avx512_supported
        avx512_path = ext_dir / "avx512_target.cpp"
        if avx512_path.is_file():
            total_capability_checks += avx512_path.read_text().count("{ return false; }")

        # moe_mul1.cpp: has_avx2, has_avx512_vnni, has_avx512_vbmi, has_avx512_bw
        moe_mul1_path = ext_dir / "cpu" / "moe_mul1.cpp"
        if moe_mul1_path.is_file():
            total_capability_checks += moe_mul1_path.read_text().count("{ return false; }")

        assert total_capability_checks == 7, (
            f"Expected 7 total capability stubs (2 avx2 + 1 avx512 + 4 in moe_mul1), "
            f"found {total_capability_checks}"
        )


def test_x86_without_force_no_mutation() -> None:
    """An explicit x86 host must reject and preserve the entire fixture."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ext_dir = Path(tmpdir) / "exllamav3_ext"
        ext_dir.mkdir()
        (ext_dir / "bindings.cpp").write_text("// original binding")
        (ext_dir / "cpu").mkdir()
        (ext_dir / "parallel").mkdir()
        (ext_dir / "cpu/moe_mul1.cpp").write_text("// original native source")
        before = {str(f.relative_to(ext_dir)): f.read_bytes()
                  for f in ext_dir.rglob("*") if f.is_file()}
        patcher = Path(__file__).resolve().parents[1] / "tools/patch_exllamav3_aarch64.py"
        runner = (
            "import platform,runpy,sys; "
            "platform.machine=lambda:'x86_64'; "
            "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        result = subprocess.run(
            [sys.executable, "-c", runner, str(patcher), str(ext_dir)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, "Unforced x86 patch unexpectedly succeeded"
        assert "refusing ARM64 patch on host architecture 'x86_64'" in result.stderr
        after = {str(f.relative_to(ext_dir)): f.read_bytes()
                 for f in ext_dir.rglob("*") if f.is_file()}
        assert before == after, "Rejected x86 patch changed fixture contents"
