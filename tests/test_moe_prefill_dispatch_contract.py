"""CPU dispatch-contract tests for the routed-MoE prefill paths (GPU_JOB-2 run 4).

Run 4 showed the >256-slot (prefill) dispatch of the production act-3 entry
point at cosine ~0.10 while the decode coop path matched at 1.0. These tests
mock-record the exact arguments of every native launch on both shapes and
prove the PYTHON side is faithful on the prefill path: the expert-batched
exl3_moe launch receives the SAME activation index (3, swiglu_oai), the SAME
loader codebook flags ((mcg, mul1) = (False, True) per projection), the SAME
K tuple and the SAME pointer tables as the qualified coop launch. With that
established, the divergence has to live in the batched kernel's own
activation switch -- which is what the fork branch feat/moe-act3-batched
fixes (see test_fork_batched_kernel_instantiates_act3 below).

The mixed-K tests additionally pin the real pack's per-K-group dispatch: the
default (unified exl3_moe_mixedk) launch and the VLLM_EXL3_MIXEDK_MODE=
grouped launches must both carry act 3 and the mul1 codebook, and every inner
path must stamp its own dispatch label.
"""

import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3

HIDDEN, INTER = 256, 128
N_EXP = 6
# The real pack's three per-expert (K_gate, K_up, K_down) tuples (layer 3 of
# vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw: w1->gate, w3->up, w2->down).
K_GROUPS = ((2, 2, 2), (2, 2, 3), (3, 3, 3))
PTR_KEYS = (
    "gate_trellis", "gate_suh", "gate_svh",
    "up_trellis", "up_suh", "up_svh",
    "down_trellis", "down_suh", "down_svh",
)


def _pybind_doc(name: str, n_args: int) -> str:
    params = ", ".join(f"arg{i}: torch.Tensor" for i in range(n_args))
    return f"{name}({params}) -> None\n\n{name}"


class _Recorder:
    """A pybind-shaped binding that records every launch's exact args."""

    def __init__(self, name: str, n_args: int) -> None:
        self.__name__ = name
        self.__doc__ = _pybind_doc(name, n_args)
        self.calls: list[tuple] = []

    def __call__(self, *args):
        self.calls.append(args)
        return None


def _mock_ext(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    ext = SimpleNamespace(
        exl3_moe=_Recorder("exl3_moe", 35),
        exl3_moe_coop=_Recorder("exl3_moe_coop", 36),
        exl3_moe_mixedk=_Recorder("exl3_moe_mixedk", 36),
        exl3_moe_max_concurrency=lambda _idx: 1,
    )
    monkeypatch.setattr(exl3, "load_exllamav3_ext", lambda: ext)
    monkeypatch.setitem(sys.modules, "exllamav3_ext", ext)
    monkeypatch.setattr(exl3, "get_moe_kernel_backend", lambda: "exllamav3")
    return ext


def _ptr_tables(device: torch.device) -> dict:
    return {
        key: torch.arange(N_EXP, dtype=torch.int64, device=device) + i
        for i, key in enumerate(PTR_KEYS)
    }


def _layer(
    device: torch.device, *, mixed_k: bool, gateless: bool = False
) -> SimpleNamespace:
    """The layer exactly as the loader leaves it after weight load."""
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            activation=SimpleNamespace(value="swigluoai_uninterleave")
        ),
        swiglu_alpha=exl3.MINIMAX_SWIGLU_ALPHA_DEFAULT,
        swiglu_beta=1.0,
        swiglu_limit=exl3.MINIMAX_SWIGLU_LIMIT_DEFAULT,
        _exl3_inners=[{} for _ in range(N_EXP)],
        _exl3_ptrs=_ptr_tables(device),
        _exl3_fused_temps=tuple(
            torch.zeros(1, exl3.TEMP_ROWS_FUSED, w, dtype=torch.float16)
            for w in (HIDDEN, HIDDEN, INTER, INTER)
        ),
        _exl3_codebook_flags=(False, True) * 3,
        _exl3_gateless=gateless,
        _exl3_k_groups=None,
        _exl3_kdesc=None,
        _exl3_krank=None,
    )
    if mixed_k:
        # build_exl3_fused_state's mixed-K state for the real pack's tuples:
        # two experts per distinct (K_gate, K_up, K_down) tuple.
        layer._exl3_mixedk_unified = True
        layer._exl3_k = None
        layer._exl3_k_tuple = None
        layer._exl3_coop_compat = False
        ids = list(range(N_EXP))
        groups = [
            (kt, ids[i : i + 2]) for i, kt in enumerate(K_GROUPS)
        ]
        layer._exl3_k_groups = groups
        g_index = torch.full((N_EXP + 1,), -1, dtype=torch.long)
        for gi, (_kt, g_ids) in enumerate(groups):
            g_index[g_ids] = gi
        layer._exl3_group_index = g_index
        layer._exl3_k_group_ids = [
            torch.tensor(g_ids, dtype=torch.long) for _kt, g_ids in groups
        ]
        karr = []
        for j in range(3):
            vals = []
            for _kt, g_ids in groups:
                vals.extend([_kt[j]] * len(g_ids))
            karr.append(torch.tensor(vals, dtype=torch.int32))
        layer._exl3_K_gate_arr, layer._exl3_K_up_arr, layer._exl3_K_down_arr = karr
        layer._exl3_k_max_launch = max(max(kt) for kt in K_GROUPS)
    else:
        layer._exl3_mixedk_unified = False
        layer._exl3_k = 2
        layer._exl3_k_tuple = (2, 2, 2)
        layer._exl3_coop_compat = not gateless
    return layer


def _ids_weights(tokens: int, topk: int):
    # Round-robin routing: every expert sees the same row count and no expert
    # reaches the fat threshold, so the launch under test is the standard one.
    ids = torch.arange(tokens * topk) % N_EXP
    ids = ids.reshape(tokens, topk).long()
    weights = torch.ones(tokens, topk)
    return ids, weights


def _apply(ext, layer, tokens, topk):
    ids, weights = _ids_weights(tokens, topk)
    x = torch.zeros(tokens, HIDDEN, dtype=torch.float16)
    out = exl3.apply_exl3_fused_moe(
        x, ids, weights, layer, layer._exl3_inners, None,
        exl3.MINIMAX_SWIGLU_LIMIT_DEFAULT,
    )
    return out, getattr(layer, "_exl3_last_apply", "unknown")


def test_uniform_prefill_launch_carries_the_coop_launchs_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root-cause evidence: the prefill (expert-batched) launch receives the
    same act index, codebook flags, K tuple and pointer tables as the coop
    launch -- the python side does not drop any of them."""
    ext = _mock_ext(monkeypatch)
    layer = _layer(torch.device("cpu"), mixed_k=False)

    # Decode shape: 64 tokens x topk 2 = 128 slots <= the 256-slot cap.
    _apply(ext, layer, 64, 2)
    assert len(ext.exl3_moe_coop.calls) == 1
    assert len(ext.exl3_moe.calls) == 0
    coop = ext.exl3_moe_coop.calls[0]
    assert layer._exl3_last_apply == "coop"

    # Prefill shape: 192 tokens x topk 2 = 384 slots > the cap.
    _apply(ext, layer, 192, 2)
    assert len(ext.exl3_moe.calls) == 1
    batched = ext.exl3_moe.calls[0]
    assert layer._exl3_last_apply == "batched"

    # Same activation index (act 3, swiglu_oai) on both paths.
    assert coop[23] == exl3.MOE_ACT_SWIGLU_OAI
    assert batched[9] == exl3.MOE_ACT_SWIGLU_OAI
    # Same loader codebook flags: (mcg, mul1) = (False, True) per projection.
    assert (coop[21], coop[22]) == (False, True)
    assert batched[22:28] == (False, True) * 3
    # Same K tuple and the SAME pointer-table tensors.
    assert coop[18:21] == (2, 2, 2)
    assert batched[10:13] == (2, 2, 2)
    for i, key in enumerate(PTR_KEYS):
        assert coop[6 + i] is layer._exl3_ptrs[key]
        assert batched[13 + i] is layer._exl3_ptrs[key]
    # The MiniMax clamp limit rides along on both paths.
    assert coop[24] == exl3.MINIMAX_SWIGLU_LIMIT_DEFAULT
    assert batched[28] == exl3.MINIMAX_SWIGLU_LIMIT_DEFAULT


def test_mixedk_default_launch_is_unified_and_carries_act3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real pack's default dispatch (VLLM_EXL3_MIXEDK_MODE unset) is the
    unified exl3_moe_mixedk launch, and it must carry act 3, the mul1
    codebook and the real per-expert K arrays."""
    monkeypatch.delenv("VLLM_EXL3_MIXEDK_MODE", raising=False)
    ext = _mock_ext(monkeypatch)
    layer = _layer(torch.device("cpu"), mixed_k=True)

    _apply(ext, layer, 64, 2)
    assert layer._exl3_last_apply == "mixedk_unified"
    assert len(ext.exl3_moe_mixedk.calls) == 1
    call = ext.exl3_moe_mixedk.calls[0]
    assert call[9] == exl3.MOE_ACT_SWIGLU_OAI
    assert call[22:28] == (False, True) * 3
    assert call[28] == exl3.MINIMAX_SWIGLU_LIMIT_DEFAULT
    # The per-expert K arrays are the loader's own tensors, covering the
    # pack's three tuples.
    assert call[10] is layer._exl3_K_gate_arr
    assert call[11] is layer._exl3_K_up_arr
    assert call[12] is layer._exl3_K_down_arr
    assert call[10].tolist() == [2, 2, 2, 2, 3, 3]
    assert call[11].tolist() == [2, 2, 2, 2, 3, 3]
    assert call[12].tolist() == [2, 2, 3, 3, 3, 3]


def test_mixedk_grouped_launches_one_per_k_tuple_with_act3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VLLM_EXL3_MIXEDK_MODE=grouped: one uniform-K exl3_moe launch per
    distinct K tuple, each carrying act 3 and the mul1 codebook."""
    monkeypatch.setenv("VLLM_EXL3_MIXEDK_MODE", "grouped")
    ext = _mock_ext(monkeypatch)
    layer = _layer(torch.device("cpu"), mixed_k=True)

    _apply(ext, layer, 64, 2)
    assert layer._exl3_last_apply == "mixedk_grouped"
    assert len(ext.exl3_moe.calls) == len(K_GROUPS)
    assert len(ext.exl3_moe_mixedk.calls) == 0
    for call, kt in zip(ext.exl3_moe.calls, K_GROUPS):
        assert call[9] == exl3.MOE_ACT_SWIGLU_OAI
        assert call[22:28] == (False, True) * 3
        assert call[10:13] == kt


def _fake_proj(k: int) -> SimpleNamespace:
    """A projection exactly as the loader leaves it: the stored 3-D trellis
    whose width is 16*K (K = true bit width), plus its suh/svh scales."""
    return SimpleNamespace(
        trellis=torch.empty(4, 4, 16 * k, dtype=torch.int16),
        suh=torch.empty(4, 16 * k, dtype=torch.float16),
        svh=torch.empty(4, 16 * k, dtype=torch.float16),
    )


def _real_builder_layer() -> tuple[SimpleNamespace, list[dict[str, Any]]]:
    """Layer + inners exactly as the loader leaves a mixed-K pack: two experts
    per distinct (K_gate, K_up, K_down) tuple of the pack's census."""
    tuples = ((2, 2, 2), (2, 2, 2), (2, 2, 3), (2, 2, 3), (3, 3, 3), (3, 3, 3))
    inners = [
        {
            "gate": _fake_proj(t[0]),
            "up": _fake_proj(t[1]),
            "down": _fake_proj(t[2]),
        }
        for t in tuples
    ]
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            activation=SimpleNamespace(value="swigluoai_uninterleave")
        ),
        swiglu_alpha=exl3.MINIMAX_SWIGLU_ALPHA_DEFAULT,
        swiglu_beta=1.0,
        swiglu_limit=exl3.MINIMAX_SWIGLU_LIMIT_DEFAULT,
        w13_suh=torch.zeros(1, dtype=torch.float16),
        _exl3_inners=inners,
        _exl3_hidden_size=HIDDEN,
        _exl3_intermediate_local=INTER,
        _exl3_mixed_k=True,
    )
    return layer, inners


def test_real_builder_karrays_are_true_bits_and_match_the_grouped_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU_JOB-2 run 5's root cause, pinned: the REAL build_exl3_fused_state
    must build the unified kernel's per-expert K arrays with the TRUE bit
    width (trellis.shape[-1] // 16), and the unified launch's arrays must
    equal the grouped launches' scalar K for every expert -- two spellings of
    the same load-time fact. Commit 3275819 shipped // 8 = 2*K here, the
    kernel consumed the doubled value as bits (moe_gemm_tile's runtime switch
    has cases 1..8 only), and the mixedk_unified path scored cosine ~0 on
    real and synthetic packs while the grouped path was exact."""
    monkeypatch.delenv("VLLM_EXL3_MIXEDK_MODE", raising=False)
    monkeypatch.delenv("VLLM_EXL3_MIXEDK_PACKED", raising=False)
    ext = _mock_ext(monkeypatch)
    layer, inners = _real_builder_layer()
    exl3.build_exl3_fused_state(layer, inners)

    # The grouped spelling: the true (K_gate, K_up, K_down) per expert.
    grouped_k: dict[int, tuple[int, ...]] = {}
    for kt, ids in layer._exl3_k_groups:
        for e in ids:
            grouped_k[e] = kt
    assert sorted(grouped_k) == list(range(N_EXP))
    assert layer._exl3_k_max_launch == max(max(t) for t in grouped_k.values())

    # The unified arrays carry the SAME true bit width per expert (run 5's
    # bug doubled every entry: [4, 4, 4, 4, 6, 6] instead of [2, 2, 2, 2, 3, 3]).
    for arr, j in (
        (layer._exl3_K_gate_arr, 0),
        (layer._exl3_K_up_arr, 1),
        (layer._exl3_K_down_arr, 2),
    ):
        assert arr.tolist() == [grouped_k[e][j] for e in range(N_EXP)]

    # Mock-record BOTH dispatches and diff them per expert: the unified
    # launch's per-expert K must equal the grouped launch's scalar K.
    layer._exl3_codebook_flags = (False, True) * 3
    _apply(ext, layer, 64, 2)
    assert layer._exl3_last_apply == "mixedk_unified"
    unified = ext.exl3_moe_mixedk.calls[0]
    assert unified[9] == exl3.MOE_ACT_SWIGLU_OAI
    monkeypatch.setenv("VLLM_EXL3_MIXEDK_MODE", "grouped")
    _apply(ext, layer, 64, 2)
    assert layer._exl3_last_apply == "mixedk_grouped"
    assert len(ext.exl3_moe.calls) == len(layer._exl3_k_groups)
    for call, (kt, ids) in zip(ext.exl3_moe.calls, layer._exl3_k_groups):
        assert call[10:13] == kt
        for e in ids:
            got = tuple(int(unified[10 + j][e]) for j in range(3))
            assert got == kt, (
                f"expert {e}: unified K arrays say {got}, grouped launch says "
                f"{kt} -- the unified kernel would run the wrong bits instance"
            )


def test_load_guard_rejects_karray_group_divergence() -> None:
    """The fail-closed load guard: any divergence between the unified K arrays
    and the grouped tuples is a load-time ValueError, never a silent
    wrong-bits launch (this is what let the 3275819 regression through)."""
    device = torch.device("cpu")
    groups = [((2, 2, 2), [0, 1]), ((2, 2, 3), [2, 3]), ((3, 3, 3), [4, 5])]
    good = [
        torch.tensor(v, dtype=torch.int32, device=device)
        for v in ([2, 2, 2, 2, 3, 3], [2, 2, 2, 2, 3, 3], [2, 2, 3, 3, 3, 3])
    ]
    exl3._assert_mixedk_karrays_match_groups(
        good[0], good[1], good[2], groups, N_EXP, device
    )
    # The 3275819 spelling (2*K): must raise.
    doubled = [2 * t for t in good]
    with pytest.raises(ValueError, match="disagree with the per-K-group"):
        exl3._assert_mixedk_karrays_match_groups(
            doubled[0], doubled[1], doubled[2], groups, N_EXP, device
        )
    # A single tampered entry: must raise.
    tampered = [t.clone() for t in good]
    tampered[2][4] = 2
    with pytest.raises(ValueError, match="disagree with the per-K-group"):
        exl3._assert_mixedk_karrays_match_groups(
            tampered[0], tampered[1], tampered[2], groups, N_EXP, device
        )


def _fork_candidates() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.get("VLLM_EXL3_FORK_SRC")
    cands = []
    if env:
        cands.append(env)
    cands.extend(
        [
            os.path.join(os.path.dirname(here), os.pardir, "exllamav3-fix"),
            "/root/work/minimax-m31/src/exllamav3",
        ]
    )
    return [
        c
        for c in cands
        if os.path.isfile(
            os.path.join(c, "exllamav3", "exllamav3_ext", "quant", "hadamard_inner.cuh")
        )
    ]


def test_fork_batched_kernel_instantiates_act3() -> None:
    """The fork-side regression bar: the expert-batched kernels' activation
    helper must instantiate act 3. On 909ad85 the guad switch fell through to
    'default: break' for act 3 (raw gate x up, no activation) -- the cosine
    ~0.10 of GPU_JOB-2 run 4. The fix branch feat/moe-act3-batched adds the
    ACT_SILU_OAI case and skips the generic clamp + gate multiply for it.
    Skipped when no exllamav3 fork source tree is discoverable."""
    cands = _fork_candidates()
    if not cands:
        pytest.skip("no exllamav3 fork source tree discovered")
    path = os.path.join(
        cands[0], "exllamav3", "exllamav3_ext", "quant", "hadamard_inner.cuh"
    )
    src = open(path, encoding="utf-8").read()
    guad = src.split("had_hf_r_128_guad_inner", 1)[1]
    assert "#define ACT_SILU_OAI 3" in src
    assert "case ACT_SILU_OAI:" in guad
    # The generic clamp + gate multiply must be guarded OFF for act 3: the
    # up clamp and the (up + 1) shift are part of the activation there.
    assert "if (act_function != ACT_SILU_OAI)" in guad
    assert "__expf(-1.702f * g)" in guad
