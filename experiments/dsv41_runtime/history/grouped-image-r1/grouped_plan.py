"""Pure routing plan for byte-preserving mixed-K EXL3 dispatch."""
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class RoutedExpert:
    key: str
    bits: tuple
    positions: tuple


def routing_groups(layer, rows, records, *, max_routes=2048):
    """Each owned route appears exactly once, including duplicate IDs per row.

    Unowned/out-of-range IDs never acquire a pointer. Oversized experts are
    split into separate waves so the kernel cannot silently skip their rows.
    """
    if max_routes < 1:
        raise ValueError('Positive route bound required')
    width = len(rows[0]) if rows else 0
    if any(len(row) != width for row in rows):
        raise ValueError('Ragged routing rows')
    positions = OrderedDict()
    for row, values in enumerate(rows):
        for column, expert in enumerate(values):
            if type(expert) is not int:
                raise ValueError('Integral expert IDs required')
            key = f'{layer}:{expert}'
            if key in records:
                positions.setdefault(key, []).append(row * width + column)
    groups = OrderedDict()
    for key, routes in positions.items():
        tensors = records[key]['tensors']
        bits = tuple(tensors[p + '.trellis']['shape'][-1] // 16 for p in ('w1', 'w3', 'w2'))
        if any(k not in range(2, 9) for k in bits):
            raise ValueError('Unsupported packed K geometry')
        for wave, start in enumerate(range(0, len(routes), max_routes)):
            groups.setdefault((bits, wave), []).append(RoutedExpert(key, bits, tuple(routes[start:start+max_routes])))
    return list(groups.values())


def packed_routing(batch, *, tokens, topk):
    """Return counts, token rows and weight offsets with a full sentinel tail."""
    positions = [p for item in batch for p in item.positions]
    length = tokens * topk
    if len(positions) > length or any(p < 0 or p >= length for p in positions):
        raise ValueError('Invalid bounded route positions')
    if len(set(positions)) != len(positions):
        raise ValueError('A route was dispatched twice in one wave')
    padding = length - len(positions)
    counts = [len(item.positions) for item in batch] + [padding]
    token_rows = [p // topk for p in positions] + [0] * padding
    offsets = positions + [0] * padding
    return counts, token_rows, offsets
