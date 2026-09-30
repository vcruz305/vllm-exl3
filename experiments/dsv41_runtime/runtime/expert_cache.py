"""Demand-only packed EXL3 cache with explicit CUDA lifetimes.

Experimental C1/eager path. It never allocates the full expert checkpoint.
"""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import threading
import torch
from exllamav3.modules.quant.exl3 import LinearEXL3
from expert_store import ALIGN, Regions


def host_memory():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {name: int(values[name].split()[0])*1024 for name in ('MemAvailable','SwapTotal','SwapFree')}


class CacheBusy(RuntimeError):
    pass


@dataclass
class Entry:
    key: str
    offset: int
    size: int
    generation: int
    projections: dict
    users: int = 0
    events: list = field(default_factory=list)


class ExpertCache:
    def __init__(self, store, capacity_bytes, *, layer_quotas=None, device='cuda:0', reserve_bytes=8*2**30):
        if capacity_bytes <= 0 or capacity_bytes % ALIGN:
            raise ValueError('Cache capacity must be aligned')
        self.store = store
        self.device = torch.device(device)
        self.reserve = reserve_bytes
        initial = host_memory()
        if initial['SwapTotal'] != initial['SwapFree'] or initial['MemAvailable'] < capacity_bytes+reserve_bytes+2*2**30:
            raise RuntimeError('Insufficient idle RAM or swap already in use')
        layers = sorted({int(key.split(':')[0]) for key in store.records})
        if layer_quotas is None:
            if len(layers) != 1:
                raise ValueError('Multiple layers require explicit byte quotas')
            layer_quotas = {layers[0]: capacity_bytes}
        if set(layer_quotas) != set(layers) or sum(layer_quotas.values()) != capacity_bytes:
            raise ValueError('Layer quotas must exactly partition the cache')
        self.regions = {}
        base = 0
        for layer, size in sorted(layer_quotas.items()):
            self.regions[layer] = Regions(size, base)
            largest = max(r['bytes'] for key,r in store.records.items() if int(key.split(':')[0]) == layer)
            if largest > size:
                raise ValueError('Layer quota cannot fit its largest expert')
            base += size
        self.arena = torch.empty(capacity_bytes, dtype=torch.uint8, device=self.device)
        self.entries = OrderedDict()
        self.lock = threading.RLock()
        self.generation = 0
        self.stats = dict(hits=0, misses=0, evictions=0, failures=0, peak_resident_bytes=0)
        self.resident_bytes = 0
        self.closed = False

    def _guard(self):
        m = host_memory()
        if m['MemAvailable'] < self.reserve or m['SwapTotal'] != m['SwapFree']:
            raise RuntimeError('RAM reserve or zero-swap gate failed')

    def _evict(self, key):
        entry = self.entries[key]
        if entry.users:
            raise CacheBusy('Cannot evict an expert in use')
        for event in entry.events:
            event.synchronize()
        entry.events.clear()
        # Dropping handles is necessary: LinearEXL3.unload() is a no-op.
        entry.projections.clear()
        self.regions[int(key.split(':')[0])].release(entry.offset)
        self.resident_bytes -= entry.size
        del self.entries[key]
        self.stats['evictions'] += 1

    def _load(self, key, cancel):
        self._guard()
        record = self.store.records[key]
        layer = int(key.split(':')[0])
        region = self.regions[layer]
        offset = region.allocate(record['bytes'])
        while offset is None:
            candidate = next((k for k,e in self.entries.items()
                              if int(k.split(':')[0]) == layer and e.users == 0), None)
            if candidate is None:
                raise CacheBusy('All eviction candidates are in use')
            self._evict(candidate)
            offset = region.allocate(record['bytes'])
        projections = {}
        try:
            with self.store.read(key, cancel=cancel) as data:
                cpu = torch.frombuffer(data, dtype=torch.uint8)
                target = self.arena.narrow(0, offset, record['bytes'])
                target.copy_(cpu, non_blocking=False)
                torch.cuda.current_stream(self.device).synchronize()
                del cpu
            if cancel is not None and cancel.is_set():
                raise InterruptedError('Expert load cancelled')
            dtype = {'I16':torch.int16, 'F16':torch.float16, 'I32':torch.int32}
            for p in ('w1','w2','w3'):
                tensors = {}
                for suffix in ('trellis','suh','svh','mul1'):
                    spec = record['tensors'][f'{p}.{suffix}']
                    tensors[suffix] = target.narrow(0,spec['offset'],spec['bytes']).view(dtype[spec['dtype']]).view(spec['shape'])
                if tensors['mul1'].item() != -2082680531:
                    raise ValueError('MUL1 marker mismatch')
                inner = LinearEXL3(config=None, in_features=tensors['suh'].numel(),
                    out_features=tensors['svh'].numel(), out_dtype=torch.float16,
                    transformers_fix=True, **tensors)
                inner.config.infer_params.no_reconstruct = True
                projections[p] = inner
            self.generation += 1
            entry = Entry(key,offset,record['bytes'],self.generation,projections)
            self.entries[key] = entry
            self.resident_bytes += record['bytes']
            self.stats['peak_resident_bytes'] = max(self.stats['peak_resident_bytes'],self.resident_bytes)
            return entry
        except BaseException:
            torch.cuda.current_stream(self.device).synchronize()
            projections.clear()
            region.release(offset)
            self.stats['failures'] += 1
            raise

    @contextmanager
    def lease(self, key, *, cancel=None):
        with torch.cuda.device(self.device), self.lock:
            if self.closed:
                raise RuntimeError('Cache is closed')
            if key not in self.store.records:
                raise KeyError(key)
            if cancel is not None and cancel.is_set():
                raise InterruptedError('Expert lease cancelled')
            entry = self.entries.get(key)
            if entry is None:
                self.stats['misses'] += 1
                entry = self._load(key,cancel)
            else:
                self.stats['hits'] += 1
                entry.events[:] = [e for e in entry.events if not e.query()]
                self.entries.move_to_end(key)
            entry.users += 1
        try:
            yield entry
        finally:
            with torch.cuda.device(self.device), self.lock:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(self.device))
                entry.events.append(event)
                entry.users -= 1

    def apply(self, layer, x, ids, weights, *, limit=10.0):
        self._guard()
        if x.ndim != 2 or ids.ndim != 2 or ids.shape != weights.shape or ids.shape[0] != x.shape[0]:
            raise ValueError('Invalid routing shapes')
        if x.device != self.device or ids.device != self.device or weights.device != self.device:
            raise ValueError('Routing and activations must use the cache device')
        result = torch.zeros_like(x,dtype=torch.float32)
        for eid in torch.unique(ids).tolist():
            key = f'{layer}:{int(eid)}'
            if key not in self.store.records:
                # EP rank owns only its manifest's experts.
                continue
            rows, positions = (ids == eid).nonzero(as_tuple=True)
            with self.lease(key) as entry:
                h = x.index_select(0,rows).half().contiguous()
                p = entry.projections
                gate = p['w1'].forward(h,{},out_dtype=torch.float32)
                up = p['w3'].forward(h,{},out_dtype=torch.float32)
                if limit is not None:
                    gate = gate.clamp(max=limit)
                    up = up.clamp(-limit,limit)
                act = (torch.nn.functional.silu(gate)*up).half().contiguous()
                down = p['w2'].forward(act,{},out_dtype=torch.float32)
                result.index_add_(0,rows,down*weights[rows,positions,None].float())
        return result

    def clear(self):
        with self.lock:
            if any(e.users for e in self.entries.values()):
                raise CacheBusy('Cache has active leases')
            for key in list(self.entries):
                self._evict(key)

    def close(self):
        with self.lock:
            self.clear()
            self.arena = None
            self.closed = True
