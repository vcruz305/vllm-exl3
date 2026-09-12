from pathlib import Path


def test_setup_uses_relative_cuda_extension_sources() -> None:
    setup_py = Path(__file__).parents[1] / "setup.py"
    text = setup_py.read_text(encoding="utf-8")

    assert "def _src(" in text
    assert 'sources=[' in text
    assert 'str(ROOT / "csrc" / "bindings.cpp")' not in text
    for name in (
        "bindings.cpp",
        "exl3_gemv.cu",
        "p2b_batched.cu",
        "p2b_moe.cu",
        "exl3_gemm.cu",
        "exl3_fat_gemm.cu",
    ):
        assert f'_src("{name}")' in text
