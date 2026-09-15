"""Eager grouped packed EXL3 execution over the existing bounded expert cache.

Uses the pinned ExLlamaV3 exl3_moe ABI (be57335b...), retaining original K2–K8
and MUL1 tensors. Thin routes use fused MoE; fat routes explicitly use the
existing packed GEMM policy. Neither path reconstructs whole weight matrices.
Disk reads remain the independently validated synchronous implementation.
"""
from contextlib import ExitStack
import math

import torch
import exllamav3_ext
from expert_cache import ExpertCache, CacheBusy
from grouped_plan import routing_groups, packed_routing


class GroupedExpertCache(ExpertCache):
    MAX_EXPERTS = 16
    MAX_ROUTES = 2048
    FUSED_ROUTE_THRESHOLD = 8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temps = None
        self._temp_rows = 0
        self._last_apply = None
        self._concurrency = int(exllamav3_ext.exl3_moe_max_concurrency(self.device.index or 0))
        if self._concurrency < 1:
            raise ValueError('Device has no supported fused expert concurrency')
        self.stats.update(grouped_launches=0, grouped_routes=0, grouped_experts=0,
                          grouped_max_batch=0, grouped_cache_flushes=0, grouped_apply_calls=0,
                          packed_fat_experts=0, packed_fat_routes=0)

    def _scratch(self, rows, stream):
        rows = (rows + 15) // 16 * 16
        if not 0 < rows <= self.MAX_ROUTES:
            raise ValueError('Fused scratch row bound exceeded')
        if self._temp_rows < rows:
            self._guard()
            self._temps = tuple(torch.empty((self._concurrency, rows, size),
                dtype=torch.float16, device=self.device) for size in (5120, 5120, 2304, 2304))
            self._temp_rows = rows
        for tensor in self._temps:
            tensor.record_stream(stream)
        return self._temps

    def _execute(self, batch, entries, x, weights, result, limit, stream):
        bits = batch[0].bits
        counts, token_rows, offsets = packed_routing(batch, tokens=x.shape[0], topk=weights.shape[1])
        pointers = []
        for projection, k in zip(('w1', 'w3', 'w2'), bits):
            for entry in entries:
                p = entry.projections[projection]
                if p.K != k or p.mcg or not p.mul1:
                    raise ValueError('Live projection differs from packed dispatch signature')
                expected = (2304, 5120) if projection == 'w2' else (5120, 2304)
                if (p.in_features, p.out_features) != expected:
                    raise ValueError('Unsupported fused expert dimensions')
            for attribute in ('trellis', 'suh', 'svh'):
                pointers.extend(getattr(e.projections[projection], attribute).data_ptr() for e in entries)
        # One transfer contains all route metadata and all nine pointer tables.
        metadata = torch.tensor(counts + token_rows + offsets + pointers,
                                dtype=torch.int64, device=self.device)
        n, length = len(entries), len(token_rows)
        cursor = n + 1
        expert_counts = metadata[:cursor]
        sorted_tokens = metadata[cursor:cursor+length]
        cursor += length
        sorted_weights = weights.reshape(-1).index_select(0, metadata[cursor:cursor+length]).contiguous()
        cursor += length
        tables = metadata[cursor:].view(9, n)
        if min(counts[:-1]) > self.FUSED_ROUTE_THRESHOLD:
            # Large per-expert batches measured slower in the fused kernel.
            # The original packed GEMM is selected deliberately and counted,
            # using the shared route plan instead of GPU nonzero per expert.
            offset = 0
            for entry, count in zip(entries, counts[:-1]):
                rows = sorted_tokens[offset:offset+count]
                weight = sorted_weights[offset:offset+count]
                p = entry.projections
                h = x.index_select(0, rows).contiguous()
                gate = p['w1'].forward(h, {}, out_dtype=torch.float32)
                up = p['w3'].forward(h, {}, out_dtype=torch.float32)
                if limit is not None:
                    gate = gate.clamp(max=limit)
                    up = up.clamp(-limit, limit)
                activation = (torch.nn.functional.silu(gate)*up).half().contiguous()
                down = p['w2'].forward(activation, {}, out_dtype=torch.float32)
                result.index_add_(0, rows, down*weight[:, None].float())
                offset += count
            self.stats['packed_fat_experts'] += n
            self.stats['packed_fat_routes'] += offset
            return
        if max(counts[:-1]) > self.FUSED_ROUTE_THRESHOLD:
            raise ValueError('Mixed thin/fat dispatch batch')
        temps = self._scratch(max(counts[:-1]), stream)
        exllamav3_ext.exl3_moe(x, result, expert_counts, sorted_tokens, sorted_weights.half(),
            *temps, 0, *bits, *tables.unbind(0),
            False, True, False, True, False, True,
            0.0 if limit is None else float(limit), n)
        self.stats['grouped_launches'] += 1
        self.stats['grouped_routes'] += sum(counts[:-1])
        self.stats['grouped_experts'] += n
        self.stats['grouped_max_batch'] = max(self.stats['grouped_max_batch'], n)

    def apply(self, layer, x, ids, weights, *, limit=10.0):
        if (x.ndim != 2 or x.shape[1] != 5120 or ids.ndim != 2 or
            ids.shape != weights.shape or ids.shape[0] != x.shape[0]):
            raise ValueError('Invalid routing or activation geometry')
        if any(t.device != self.device for t in (x, ids, weights)):
            raise ValueError('Routing and activations must use the cache device')
        if ids.dtype not in (torch.int32, torch.int64) or not x.is_floating_point() or not weights.is_floating_point():
            raise ValueError('Invalid input dtypes')
        if limit is not None and (not math.isfinite(limit) or limit <= 0):
            raise ValueError('Fused limit must be positive or None')
        with self.lock, torch.cuda.device(self.device):
            if self.closed:
                raise RuntimeError('Cache is closed')
            self._guard()
            stream = torch.cuda.current_stream(self.device)
            if self._last_apply is not None:
                stream.wait_event(self._last_apply)
            try:
                result = torch.zeros_like(x, dtype=torch.float32)
                xh = x.half().contiguous()
                # One routing synchronization per layer, replacing per-expert
                # GPU nonzero/gather barriers. Graph capture is a later phase.
                groups = routing_groups(layer, ids.cpu().tolist(), self.store.records,
                                        max_routes=self.MAX_ROUTES)
                dispatch_groups = []
                for group in groups:
                    for thin in (True, False):
                        selected = [r for r in group if (len(r.positions) <= self.FUSED_ROUTE_THRESHOLD) == thin]
                        if selected:
                            dispatch_groups.append(selected)
                for group in dispatch_groups:
                    offset = 0
                    while offset < len(group):
                        with ExitStack() as leases:
                            batch, entries = [], []
                            while offset < len(group) and len(batch) < self.MAX_EXPERTS:
                                item = group[offset]
                                try:
                                    entry = leases.enter_context(self.lease(item.key))
                                except CacheBusy:
                                    if not batch:
                                        raise
                                    self.stats['grouped_cache_flushes'] += 1
                                    break
                                batch.append(item)
                                entries.append(entry)
                                offset += 1
                            self._execute(batch, entries, xh, weights, result, limit, stream)
                            # Leases record completion events after launch;
                            # eviction must wait before these pointers can move.
                self.stats['grouped_apply_calls'] += 1
                return result
            finally:
                self._last_apply = torch.cuda.Event()
                self._last_apply.record(stream)

    def close(self):
        with self.lock, torch.cuda.device(self.device):
            if self._last_apply is not None:
                self._last_apply.synchronize()
            super().close()
            self._temps = None
            self._last_apply = None
            self._temp_rows = 0
