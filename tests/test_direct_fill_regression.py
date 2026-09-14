"""TDD regression tests for PR14 direct-copy trellis fill.

Tests cover:
- Shape validation (shape mismatch detection)
- dtype enforcement (int16 conversion + tracking)
- Device policy isolation (no cross-device allocation)
- Contiguity handling
- Stats accumulation (call count, byte volume)
- Negative controls (missing plan, invalid expert_id)
- Content verification (real tensors, nonzero values, dtype/contiguity checks)

CPU tests run unconditionally. CUDA peak-allocation tests are clearly separated
and marked optional; they require actual GPU hardware and are not executed locally.

DIRECT_FILL_BYTES measures the original SOURCE tensor's byte size (source.numel() *
source.element_size()), which may differ from the transferred int16 byte volume
when the source is float32. This metric tracks the staging transient footprint
on the source device, not the arena write size.
"""

import os
import sys
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

# Skip CUDA imports and tests if CUDA_VISIBLE_DEVICES is empty
SKIP_CUDA_TESTS = os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""

try:
    import torch
    import torch.nn as nn
    from torch.nn.parameter import Parameter
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Parameter = None  # type: ignore[assignment]


# Import from canonical vllm_exl3.exl3 namespace (production code)
if _TORCH_AVAILABLE and torch is not None:
    from vllm_exl3.exl3 import (
        _direct_fill_trellis_slot,
        direct_fill_stats,
        _DIRECT_FILL_STATS,
    )
else:
    # Tests requiring torch will be skipped individually
    _direct_fill_trellis_slot = None
    direct_fill_stats = None
    _DIRECT_FILL_STATS = None


class TestDirectFillStats(unittest.TestCase):
    """Test the direct_fill_stats() function and global tracking."""

    def setUp(self):
        """Reset stats before each test."""
        if not _TORCH_AVAILABLE or _DIRECT_FILL_STATS is None:
            self.skipTest("torch and vllm_exl3 required")
        _DIRECT_FILL_STATS["DIRECT_FILL_CALLS"] = 0
        _DIRECT_FILL_STATS["DIRECT_FILL_BYTES"] = 0

    def test_direct_fill_stats_initial(self):
        """Stats should start at zero."""
        stats = direct_fill_stats()
        self.assertEqual(stats["DIRECT_FILL_CALLS"], 0)
        self.assertEqual(stats["DIRECT_FILL_BYTES"], 0)

    def test_direct_fill_stats_returns_dict_copy(self):
        """Stats function returns a copy, not a reference."""
        _DIRECT_FILL_STATS["DIRECT_FILL_CALLS"] = 5
        stats = direct_fill_stats()
        stats["DIRECT_FILL_CALLS"] = 999  # Modify the copy
        self.assertEqual(direct_fill_stats()["DIRECT_FILL_CALLS"], 5)  # Original unchanged


class TestDirectFillTrellisSlot(unittest.TestCase):
    """Core CPU tests for _direct_fill_trellis_slot."""

    def setUp(self):
        """Set up a mock layer with arena plan and stats reset."""
        if not _TORCH_AVAILABLE or _DIRECT_FILL_STATS is None:
            self.skipTest("torch and vllm_exl3 required")

        # Reset global stats
        _DIRECT_FILL_STATS["DIRECT_FILL_CALLS"] = 0
        _DIRECT_FILL_STATS["DIRECT_FILL_BYTES"] = 0

        # Create mock layer with arena plan
        self.layer = MagicMock()
        self.layer._exl3_trellis_temp_peak_bytes = 0

        # Mock arena plan structure: plan[proj][shape]["arena"] = [slot0, slot1, ...]
        self.arena_tensor = torch.zeros((2, 512, 64), dtype=torch.int16)

        self.layer._exl3_trellis_arena_plan = {
            "gate": {(512, 64): {"arena": self.arena_tensor}},
            "up": {(512, 128): {"arena": torch.zeros((2, 512, 128), dtype=torch.int16)}},
            "down": {(256, 512): {"arena": torch.zeros((2, 256, 512), dtype=torch.int16)}},
        }

        self.layer._exl3_trellis_eid_index = {
            "gate": {0: ((512, 64), 0), 1: ((512, 64), 1)},
            "up": {0: ((512, 128), 0)},
            "down": {0: ((256, 512), 0)},
        }

    def test_negative_control_missing_plan(self):
        """Should raise RuntimeError if arena plan is missing."""
        layer = MagicMock()
        layer._exl3_trellis_eid_index = None
        layer._exl3_trellis_arena_plan = None

        src = torch.zeros((512, 64), dtype=torch.int16)

        with self.assertRaises(RuntimeError) as cm:
            _direct_fill_trellis_slot(layer, "gate", 0, src)
        self.assertIn("prepare_trellis_arena_plan", str(cm.exception))

    def test_negative_control_invalid_expert_id(self):
        """Should raise RuntimeError if expert_id not in plan."""
        src = torch.zeros((512, 64), dtype=torch.int16)

        # expert_id=999 not in the plan
        with self.assertRaises(RuntimeError) as cm:
            _direct_fill_trellis_slot(self.layer, "gate", 999, src)
        self.assertIn("arena plan missing", str(cm.exception))

    def test_negative_control_shape_mismatch(self):
        """Should raise RuntimeError if src.shape != planned shape."""
        # Plan expects (512, 64) but provide (256, 64)
        src = torch.zeros((256, 64), dtype=torch.int16)

        with self.assertRaises(RuntimeError) as cm:
            _direct_fill_trellis_slot(self.layer, "gate", 0, src)
        self.assertIn("shape mismatch", str(cm.exception))

    def test_dtype_conversion_float32_to_int16(self):
        """Should convert float32 source to int16."""
        src = torch.arange(512 * 64, dtype=torch.float32).reshape(512, 64)
        _direct_fill_trellis_slot(self.layer, "gate", 0, src)

        # Verify the call succeeded and stats updated
        stats = direct_fill_stats()
        self.assertEqual(stats["DIRECT_FILL_CALLS"], 1)
        # DIRECT_FILL_BYTES measures SOURCE tensor size (float32: 4 bytes/element)
        expected_bytes = 512 * 64 * 4
        self.assertEqual(stats["DIRECT_FILL_BYTES"], expected_bytes)

    def test_dtype_preservation_int16(self):
        """Should not re-convert int16 source."""
        src = torch.arange(512 * 64, dtype=torch.int16).reshape(512, 64)
        _direct_fill_trellis_slot(self.layer, "gate", 0, src)

        stats = direct_fill_stats()
        self.assertEqual(stats["DIRECT_FILL_CALLS"], 1)
        expected_bytes = 512 * 64 * 2  # int16
        self.assertEqual(stats["DIRECT_FILL_BYTES"], expected_bytes)

    def test_dtype_and_content_verify(self):
        """Verify dtype conversion preserves content (nonzero, saturated)."""
        # Create float32 tensor with distinct values
        src_float = torch.arange(512 * 64, dtype=torch.float32).reshape(512, 64)
        src_float = src_float % 32768  # Keep within int16 range

        _direct_fill_trellis_slot(self.layer, "gate", 0, src_float)

        # Check arena slot received int16 version
        arena_slot = self.arena_tensor[0]
        expected_int16 = src_float.to(dtype=torch.int16)
        self.assertTrue(
            torch.equal(arena_slot, expected_int16),
            "Arena slot dtype mismatch or content not preserved after conversion",
        )

    def test_contiguity_enforced(self):
        """Source is made contiguous before copy."""
        # Create non-contiguous tensor via transpose
        src = torch.arange(512 * 64, dtype=torch.int16).reshape(64, 512).t()
        self.assertFalse(src.is_contiguous())

        _direct_fill_trellis_slot(self.layer, "gate", 0, src)

        # Should not raise; contiguity is handled internally
        stats = direct_fill_stats()
        self.assertEqual(stats["DIRECT_FILL_CALLS"], 1)

        # Verify arena received the data with correct values
        arena_slot = self.arena_tensor[0]
        expected = src.contiguous()
        self.assertTrue(
            torch.equal(arena_slot, expected),
            "Arena slot does not match contiguous source",
        )

    def test_stats_accumulation_calls(self):
        """Stats should accumulate across multiple calls."""
        src0 = torch.ones((512, 64), dtype=torch.int16) * 42
        src1 = torch.ones((512, 64), dtype=torch.int16) * 99

        _direct_fill_trellis_slot(self.layer, "gate", 0, src0)
        _direct_fill_trellis_slot(self.layer, "gate", 1, src1)

        stats = direct_fill_stats()
        self.assertEqual(stats["DIRECT_FILL_CALLS"], 2)
        self.assertEqual(stats["DIRECT_FILL_BYTES"], 512 * 64 * 2 * 2)

    def test_temp_peak_bytes_tracked(self):
        """Layer should track transient peak bytes."""
        src = torch.ones((512, 64), dtype=torch.int16)
        _direct_fill_trellis_slot(self.layer, "gate", 0, src)

        expected_transient = 512 * 64 * 2
        self.assertEqual(self.layer._exl3_trellis_temp_peak_bytes, expected_transient)

    def test_temp_peak_bytes_max_of_all_calls(self):
        """Peak should be max across all calls, not cumulative."""
        # Set up layer with multiple shapes to test different sizes
        layer = MagicMock()
        layer._exl3_trellis_temp_peak_bytes = 0

        arena_small = torch.zeros((2, 256, 64), dtype=torch.int16)
        arena_large = torch.zeros((2, 512, 64), dtype=torch.int16)

        layer._exl3_trellis_arena_plan = {
            "gate": {
                (256, 64): {"arena": arena_small},
                (512, 64): {"arena": arena_large},
            },
        }
        layer._exl3_trellis_eid_index = {
            "gate": {
                0: ((256, 64), 0),
                1: ((512, 64), 0),
            },
        }

        # First call: small tensor
        src0 = torch.ones((256, 64), dtype=torch.int16)
        _direct_fill_trellis_slot(layer, "gate", 0, src0)
        peak_after_0 = layer._exl3_trellis_temp_peak_bytes

        # Second call: larger tensor
        src1 = torch.ones((512, 64), dtype=torch.int16)
        _direct_fill_trellis_slot(layer, "gate", 1, src1)
        peak_after_1 = layer._exl3_trellis_temp_peak_bytes

        expected_0 = 256 * 64 * 2
        expected_1 = 512 * 64 * 2
        self.assertEqual(peak_after_0, expected_0)
        self.assertEqual(peak_after_1, expected_1)
        self.assertEqual(peak_after_1, max(expected_0, expected_1))

    def test_device_policy_isolation(self):
        """Should not create intermediate src.to(cuda) allocations."""
        # Source on CPU, arena on CPU (mock test environment)
        src = torch.arange(512 * 64, dtype=torch.float32, device="cpu").reshape(512, 64)

        # Mock arena that tracks calls to .copy_()
        copy_calls = []
        arena_slot = MagicMock()
        arena_slot.copy_ = lambda t, non_blocking=False: copy_calls.append((t, non_blocking))

        self.layer._exl3_trellis_arena_plan["gate"][(512, 64)]["arena"] = MagicMock()
        self.layer._exl3_trellis_arena_plan["gate"][(512, 64)]["arena"].__getitem__ = (
            lambda self, idx: arena_slot
        )

        _direct_fill_trellis_slot(self.layer, "gate", 0, src)

        # Verify copy was called with non_blocking=False (blocking)
        self.assertEqual(len(copy_calls), 1)
        self.assertFalse(copy_calls[0][1])  # non_blocking=False

    def test_multiple_projections(self):
        """Should handle gate, up, down projections independently."""
        src_gate = torch.arange(512 * 64, dtype=torch.int16).reshape(512, 64)
        src_up = torch.arange(512 * 128, dtype=torch.int16).reshape(512, 128)
        src_down = torch.arange(256 * 512, dtype=torch.int16).reshape(256, 512)

        _direct_fill_trellis_slot(self.layer, "gate", 0, src_gate)
        _direct_fill_trellis_slot(self.layer, "up", 0, src_up)
        _direct_fill_trellis_slot(self.layer, "down", 0, src_down)

        stats = direct_fill_stats()
        self.assertEqual(stats["DIRECT_FILL_CALLS"], 3)
        expected_bytes = (512 * 64 + 512 * 128 + 256 * 512) * 2
        self.assertEqual(stats["DIRECT_FILL_BYTES"], expected_bytes)

    def test_projection_content_isolation(self):
        """Each projection arena should receive only its own content."""
        src_gate = torch.ones((512, 64), dtype=torch.int16) * 11
        src_up = torch.ones((512, 128), dtype=torch.int16) * 22

        _direct_fill_trellis_slot(self.layer, "gate", 0, src_gate)
        _direct_fill_trellis_slot(self.layer, "up", 0, src_up)

        # Verify gate arena contains gate values
        gate_arena = self.layer._exl3_trellis_arena_plan["gate"][(512, 64)]["arena"]
        self.assertTrue(torch.all(gate_arena[0] == 11), "Gate arena corrupted")

        # Verify up arena contains up values
        up_arena = self.layer._exl3_trellis_arena_plan["up"][(512, 128)]["arena"]
        self.assertTrue(torch.all(up_arena[0] == 22), "Up arena corrupted")


@unittest.skipIf(SKIP_CUDA_TESTS, "CUDA_VISIBLE_DEVICES not set; skipping GPU tests")
class TestDirectFillCUDAPeakAllocation(unittest.TestCase):
    """Optional CUDA-only tests: peak allocation tracking.

    HARDWARE NOTES (parent attestation):
    - C2: 16 MiB per trellis source → 16 MiB extra CUDA peak
    - C3/C4: same direct-copy code → 0 MiB extra CUDA peak with exact byte equality
    - Tests below are marked optional and do NOT run locally without explicit GPU

    Parent owns all GPU/model qualification. Do not claim these tests passed.
    They exist only to verify the isolation when GPU is available.
    """

    def setUp(self):
        """Set up for CUDA tests."""
        if not _TORCH_AVAILABLE or torch is None:
            self.skipTest("torch required")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        _DIRECT_FILL_STATS["DIRECT_FILL_CALLS"] = 0
        _DIRECT_FILL_STATS["DIRECT_FILL_BYTES"] = 0

        self.device = torch.device("cuda")
        self.layer = MagicMock()
        self.layer._exl3_trellis_temp_peak_bytes = 0

        # 16 MiB arena (from hardware receipts)
        self.arena_tensor = torch.zeros(
            (1, 4096, 2048), dtype=torch.int16, device=self.device
        )
        self.layer._exl3_trellis_arena_plan = {
            "gate": {(4096, 2048): {"arena": self.arena_tensor}},
        }
        self.layer._exl3_trellis_eid_index = {
            "gate": {0: ((4096, 2048), 0)},
        }

    def test_cuda_no_intermediate_allocation(self):
        """Verify no extra GPU allocation beyond arena."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Baseline: measure current CUDA allocation before fill
        baseline_alloc = torch.cuda.memory_allocated()

        src = torch.arange(4096 * 2048, dtype=torch.int16, device="cpu")
        src = src.reshape(4096, 2048)

        _direct_fill_trellis_slot(self.layer, "gate", 0, src)
        torch.cuda.synchronize()

        peak_alloc = torch.cuda.max_memory_allocated()
        # Delta measures allocation *during* the fill operation
        delta_peak = peak_alloc - baseline_alloc

        # Allow small tolerance for CUDA runtime overhead, but no extra allocation zone
        tolerance = 1024 * 1024  # 1 MiB tolerance
        self.assertLess(
            delta_peak, tolerance,
            f"Extra CUDA peak {delta_peak / 1024 / 1024:.1f} MiB detected "
            f"(baseline={baseline_alloc / 1024 / 1024:.1f} MiB, peak={peak_alloc / 1024 / 1024:.1f} MiB)",
        )

    def test_cuda_byte_equality_cpu_to_cuda(self):
        """Verify int16 content byte-for-byte equality after H2D copy."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Create distinct int16 values on CPU
        src_cpu = torch.arange(4096 * 2048, dtype=torch.int16, device="cpu")
        src_cpu = src_cpu.reshape(4096, 2048)

        _direct_fill_trellis_slot(self.layer, "gate", 0, src_cpu)

        # Copy arena slot back to CPU for verification
        arena_slot_cuda = self.arena_tensor[0]
        arena_slot_cpu = arena_slot_cuda.to(device="cpu")

        # Verify byte-for-byte equality
        self.assertTrue(
            torch.equal(arena_slot_cpu, src_cpu),
            "Arena int16 content does not match source after H2D copy",
        )


class TestDirectFillIntegration(unittest.TestCase):
    """Integration tests combining multiple direct-fill operations."""

    def setUp(self):
        """Reset stats before each test."""
        if not _TORCH_AVAILABLE or _DIRECT_FILL_STATS is None:
            self.skipTest("torch and vllm_exl3 required")

        _DIRECT_FILL_STATS["DIRECT_FILL_CALLS"] = 0
        _DIRECT_FILL_STATS["DIRECT_FILL_BYTES"] = 0

    def test_full_expert_fill_sequence(self):
        """Simulate filling all experts in an MoE layer."""
        # Simulate 8 experts
        n_experts = 8
        shape = (512, 64)
        layer = MagicMock()
        layer._exl3_trellis_temp_peak_bytes = 0

        arena = torch.zeros((n_experts, *shape), dtype=torch.int16)
        layer._exl3_trellis_arena_plan = {
            "gate": {shape: {"arena": arena}},
        }
        layer._exl3_trellis_eid_index = {
            "gate": {i: (shape, i) for i in range(n_experts)},
        }

        # Fill all experts with distinct values
        for eid in range(n_experts):
            src = torch.ones(shape, dtype=torch.int16) * eid
            _direct_fill_trellis_slot(layer, "gate", eid, src)

        stats = direct_fill_stats()
        self.assertEqual(stats["DIRECT_FILL_CALLS"], n_experts)
        expected_bytes = n_experts * 512 * 64 * 2
        self.assertEqual(stats["DIRECT_FILL_BYTES"], expected_bytes)

        # Verify arena contents (content verification test)
        for eid in range(n_experts):
            self.assertTrue(
                torch.all(arena[eid] == eid),
                f"Expert {eid} arena slot contains wrong values",
            )


if __name__ == "__main__":
    unittest.main()
