"""K7/K8 configuration compatibility for EXL3.

ExLlamaV3's generic LinearEXL3 path supports the full K2-K8 trellis family,
while vllm-exl3's native cooperative MoE kernels intentionally remain limited
to the bit widths they have been qualified for. Historically Exl3Config rejected
K7/K8 before dispatch could fall back to the generic path.

This narrow installer widens *configuration acceptance* to K2-K8 without
claiming native-kernel support for K5-K8. It does not add tensor-level mixed-K
inside one RoutedExperts layer; current routed-MoE allocation still expects one
K per layer.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

_ALLOWED = frozenset(range(2, 9))
_LEGACY_ALLOWED = frozenset(range(2, 7))


def _safe_legacy_k(value: object) -> object:
    try:
        k = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return value
    return 6 if k in (7, 8) else value


def _validate_k(value: object, label: str) -> int:
    try:
        k = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer K2-K8, got {value!r}") from exc
    if k not in _ALLOWED:
        raise ValueError(f"unsupported EXL3 {label}={k}; expected K2-K8")
    return k


def _sanitize_non_routed(raw: object) -> tuple[object, dict[str, Any] | None]:
    if not isinstance(raw, dict):
        return raw, None
    original = deepcopy(raw)
    safe = deepcopy(raw)
    if "bits" in safe:
        _validate_k(safe["bits"], "non_routed_exl3 bits")
        safe["bits"] = _safe_legacy_k(safe["bits"])
    layer_bits = safe.get("layer_bits")
    if isinstance(layer_bits, dict):
        for key, value in list(layer_bits.items()):
            _validate_k(value, f"non_routed_exl3 layer_bits[{key}]")
            layer_bits[key] = _safe_legacy_k(value)
    layers = safe.get("layers")
    if isinstance(layers, dict):
        for key, spec in layers.items():
            if isinstance(spec, dict) and "bits" in spec:
                _validate_k(spec["bits"], f"non_routed_exl3 layers[{key}] bits")
                spec["bits"] = _safe_legacy_k(spec["bits"])
    return safe, original


def install_k78_config_compat(exl3_module: object) -> bool:
    """Allow K7/K8 config values while preserving native dispatch guards.

    The underlying Exl3Config currently validates K2-K6. During its constructor
    only, K7/K8 declarations are temporarily represented as K6 so all unrelated
    validation/setup runs unchanged. The original K values are restored on the
    resulting config before any layer quant method is created.
    """
    if bool(getattr(exl3_module, "_vllm_exl3_k78_compat_installed", False)):
        return False

    config_cls = getattr(exl3_module, "Exl3Config")
    original_init = config_cls.__init__
    if bool(getattr(original_init, "_vllm_exl3_k78_wrapped", False)):
        setattr(exl3_module, "_vllm_exl3_k78_compat_installed", True)
        return False

    def init_k78(self, bits=4, codebook="mcg", scope="glm53_routed_experts_only", **kwargs):
        original_bits = _validate_k(bits, "bits")

        original_layer_bits = kwargs.get("layer_bits")
        safe_layer_bits = None
        if original_layer_bits is not None:
            if not isinstance(original_layer_bits, dict):
                raise ValueError("layer_bits must be a mapping")
            restored_layer_bits: dict[int, int] = {}
            safe_layer_bits = {}
            for key, value in original_layer_bits.items():
                k = _validate_k(value, f"layer_bits[{key}]")
                restored_layer_bits[int(key)] = k
                safe_layer_bits[key] = _safe_legacy_k(k)
        else:
            restored_layer_bits = {}

        safe_nr, original_nr = _sanitize_non_routed(kwargs.get("non_routed_exl3"))

        safe_kwargs = dict(kwargs)
        if safe_layer_bits is not None:
            safe_kwargs["layer_bits"] = safe_layer_bits
        if "non_routed_exl3" in safe_kwargs:
            safe_kwargs["non_routed_exl3"] = safe_nr

        original_init(
            self,
            bits=_safe_legacy_k(original_bits),
            codebook=codebook,
            scope=scope,
            **safe_kwargs,
        )

        self.bits = original_bits
        if original_layer_bits is not None:
            self.layer_bits = restored_layer_bits
        if original_nr is not None:
            self.non_routed_exl3 = original_nr

    init_k78._vllm_exl3_k78_wrapped = True  # type: ignore[attr-defined]
    init_k78.__name__ = getattr(original_init, "__name__", "__init__")
    init_k78.__doc__ = getattr(original_init, "__doc__", None)
    config_cls.__init__ = init_k78
    setattr(exl3_module, "_vllm_exl3_k78_compat_installed", True)
    return True


def supported_config_bits() -> tuple[int, ...]:
    return tuple(sorted(_ALLOWED))


def native_qualified_bits() -> tuple[int, ...]:
    # Kept explicit: widening config acceptance must never imply native support.
    return (2, 3, 4)
