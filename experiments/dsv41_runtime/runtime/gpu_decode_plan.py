"""Pinned-resident C1/top-six routing fixture, not a full-model miss protocol.

All routing and mixed-K dispatch metadata stays on GPU after construction.
An owned nonresident route sets the device miss flag and suppresses the entire
expert computation. Such output MUST be rejected; this class is deliberately
not installed in the serving runtime. Leases pin every published pointer until
all plan work completes, including graph replays on the declared stream.
"""
from contextlib import ExitStack
import math
import torch
import triton
import triton.language as tl
import sage_heterogeneous_ext


@triton.jit
def _routes(IDS, WEIGHTS, TABLE, COUNTS, TOKENS, SORTED_WEIGHTS, BITS, POINTERS, MISS,
            BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    eid = tl.load(IDS + lane, lane < 6, other=-1)
    valid = (lane < 6) & (eid >= 0) & (eid < 384)
    state = tl.load(TABLE + eid * 13, valid, other=0)
    missing = tl.sum(((state == 1) & valid).to(tl.int32), 0) > 0
    tl.store(MISS, missing.to(tl.int32))
    eq = (eid[:, None] == eid[None, :]) & valid[None, :]
    earlier = lane[None, :] < lane[:, None]
    duplicate = tl.sum((eq & earlier).to(tl.int32), 1) > 0
    count = tl.where(valid & (state == 2) & ~duplicate & ~missing,
                     tl.sum(eq.to(tl.int32), 1), 0)
    tl.store(COUNTS + lane, count.to(tl.int64), lane < 6)
    tl.store(COUNTS + 6, (6 - tl.sum(count, 0)).to(tl.int64))
    tl.store(TOKENS + lane, 0, lane < 6)
    first = tl.min(tl.where(eq, lane[None, :], BLOCK), 1)
    start = tl.sum(tl.where(lane[None, :] < first[:, None], count[None, :], 0), 1)
    within = tl.sum((eq & earlier).to(tl.int32), 1)
    destination = start + within
    weight = tl.load(WEIGHTS + lane, lane < 6, other=0).to(tl.float32)
    active = valid & (state == 2) & ~missing
    matches = (lane[:, None] == destination[None, :]) & active[None, :]
    ordered_weight = tl.sum(tl.where(matches, weight[None, :], 0.0), 1)
    tl.store(SORTED_WEIGHTS + lane, ordered_weight, lane < 6)
    for projection in tl.static_range(3):
        value = tl.load(TABLE + eid * 13 + 1 + projection, valid & (state == 2), other=2)
        tl.store(BITS + lane * 3 + projection, value, lane < 6)
    for pointer in tl.static_range(9):
        value = tl.load(TABLE + eid * 13 + 4 + pointer, valid & (state == 2), other=0)
        tl.store(POINTERS + pointer * 6 + lane, value, lane < 6)


class PinnedDecodePlan:
    """One stream, immutable resident set and original packed tensor leases."""
    def __init__(self, cache, layer, keys, *, limit=10.0):
        if not keys or len(keys) > 16 or len(set(keys)) != len(keys):
            raise ValueError('One to sixteen distinct pinned experts required')
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError('Positive activation limit required')
        self.cache = cache
        self.device = cache.device
        self.stream = torch.cuda.current_stream(self.device)
        self.closed = False
        self._graph = None
        self._graph_inputs = None
        self._graph_input_layout = None
        self.limit = float(limit)
        self.leases = ExitStack()
        self.keys = tuple(keys)
        try:
            table = [[0] * 13 for _ in range(384)]
            for key in cache.store.records:
                owner_layer, eid = map(int, key.split(':'))
                if owner_layer == layer:
                    if not 0 <= eid < 384:
                        raise ValueError('Invalid original expert ID')
                    table[eid][0] = 1
            for key in keys:
                owner_layer, eid = map(int, key.split(':'))
                if owner_layer != layer or key not in cache.store.records:
                    raise ValueError('Pinned expert is not owned in this layer')
                entry = self.leases.enter_context(cache.lease(key))
                bits, pointers = [], []
                for projection in ('w1', 'w3', 'w2'):
                    p = entry.projections[projection]
                    expected = (2304, 5120) if projection == 'w2' else (5120, 2304)
                    if p.K not in range(2, 9) or p.mcg or not p.mul1 or (p.in_features, p.out_features) != expected:
                        raise ValueError('Unqualified original packed projection')
                    bits.append(p.K)
                    pointers.extend(getattr(p, attr).data_ptr() for attr in ('trellis', 'suh', 'svh'))
                table[eid] = [2, *bits, *pointers]
            self.table = torch.tensor(table, dtype=torch.int64, device=self.device)
            self.counts = torch.empty(7, dtype=torch.int64, device=self.device)
            self.tokens = torch.empty(6, dtype=torch.int64, device=self.device)
            self.weights = torch.empty(6, dtype=torch.float16, device=self.device)
            self.bits = torch.empty((6, 3), dtype=torch.int64, device=self.device)
            self.pointers = torch.empty((9, 6), dtype=torch.int64, device=self.device)
            self.miss = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.output = torch.empty((1, 5120), dtype=torch.float32, device=self.device)
            self.temps = tuple(torch.empty((cache._concurrency, 16, n), dtype=torch.float16,
                                          device=self.device) for n in (5120, 5120, 2304, 2304))
            self.locks = torch.zeros(cache.LOCK_INTS, dtype=torch.int32, device=self.device)
        except BaseException:
            self.stream.synchronize()
            self.leases.close()
            self.closed = True
            for name in ('table', 'counts', 'tokens', 'weights', 'bits', 'pointers',
                         'miss', 'output', 'temps', 'locks'):
                setattr(self, name, None)
            raise

    def run(self, x, ids, weights):
        if self.closed or self.cache.closed:
            raise RuntimeError('Resident plan/cache is closed')
        if torch.cuda.current_stream(self.device) != self.stream:
            raise ValueError('Pinned plan is restricted to its declared stream')
        if x.shape != (1, 5120) or ids.shape != (1, 6) or weights.shape != (1, 6):
            raise ValueError('C1/top-six fixed geometry required')
        if x.dtype != torch.float16 or ids.dtype != torch.int64 or weights.dtype != torch.float16:
            raise ValueError('Preallocated FP16 activation/weights and INT64 IDs required')
        if any(t.device != self.device or not t.is_contiguous() for t in (x, ids, weights)):
            raise ValueError('Same-device contiguous inputs required')
        _routes[(1,)](ids, weights, self.table, self.counts, self.tokens,
                     self.weights, self.bits, self.pointers, self.miss, BLOCK=8, num_warps=1)
        self.output.zero_()
        sage_heterogeneous_ext.moe(x, self.output, self.counts, self.tokens, self.weights,
                                  *self.temps, self.bits, self.pointers, self.locks, self.limit)
        return self.output, self.miss

    def capture(self, x, ids, weights):
        if self._graph is not None:
            raise RuntimeError('A graph is already captured')
        self.run(x, ids, weights)
        self.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            self.run(x, ids, weights)
        self._graph = graph
        self._graph_inputs = (x, ids, weights)
        self._graph_input_layout = tuple((t.data_ptr(), t.shape, t.stride(), t.dtype, t.device)
                                         for t in self._graph_inputs)

    def replay(self):
        if self.closed or self._graph is None:
            raise RuntimeError('No live captured plan')
        if torch.cuda.current_stream(self.device) != self.stream:
            raise ValueError('Replay must use the declared stream')
        if tuple((t.data_ptr(), t.shape, t.stride(), t.dtype, t.device)
                 for t in self._graph_inputs) != self._graph_input_layout:
            raise ValueError('Captured input storage or geometry changed')
        self._graph.replay()
        return self.output, self.miss

    def close(self):
        if self.closed:
            return
        # Includes graph replays on the declared stream; published pointers
        # cannot be released until this succeeds. No cross-stream use admitted.
        with torch.cuda.stream(self.stream):
            self.stream.synchronize()
            if self._graph is not None:
                self._graph.reset()
                self._graph = None
            self._graph_inputs = None
            self._graph_input_layout = None
            self.leases.close()
            self.closed = True
            for name in ('table', 'counts', 'tokens', 'weights', 'bits', 'pointers',
                         'miss', 'output', 'temps', 'locks'):
                setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
