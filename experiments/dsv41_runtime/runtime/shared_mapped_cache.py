"""One mapped cache for prefill waves and GPU-directed C1 decode misses.

Opt-in adapter; original HeterogeneousExpertCache dispatch/kernel code is kept.
CPU service never creates/destroys GPU views or events. Mode transitions drain
the one owning stream and release those handles on the model thread first.
"""
from collections import OrderedDict, deque
from contextlib import contextmanager, nullcontext
import ctypes
import math
import struct
import threading
import time

import torch
import exllamav3_ext
import sage_heterogeneous_dynamic as sage_heterogeneous_ext
from exllamav3.modules.quant.exl3 import LinearEXL3
from expert_cache import Entry, CacheBusy, host_memory
from expert_store import ALIGN, Regions
from heterogeneous_cache import HeterogeneousExpertCache
from gpu_decode_plan import _routes


class SharedMappedCache(HeterogeneousExpertCache):
    EXTENSION = None

    def __init__(self, store, capacity_bytes, *, layer_quotas=None, device='cuda:0',
                 reserve_bytes=8 * 2**30, extension=None):
        self.extension = extension or self.EXTENSION
        if self.extension is None:
            raise ValueError('Verified shared-mailbox extension required')
        if not 0 < capacity_bytes <= 80 * 2**30 or capacity_bytes % ALIGN:
            raise ValueError('Invalid admitted mapped cache capacity')
        self.store, self.device, self.reserve = store, torch.device(device), reserve_bytes
        initial = host_memory()
        if initial['SwapTotal'] != initial['SwapFree'] or initial['MemAvailable'] < capacity_bytes + reserve_bytes + 2 * 2**30:
            raise RuntimeError('Mapped cache admission failed')
        layers = sorted({int(k.split(':')[0]) for k in store.records})
        if layer_quotas is None and len(layers) == 1:
            layer_quotas = {layers[0]: capacity_bytes}
        if not layer_quotas or set(layer_quotas) != set(layers) or sum(layer_quotas.values()) != capacity_bytes:
            raise ValueError('Exact original per-layer cache partition required')
        self.regions, base = {}, 0
        for layer, size in sorted(layer_quotas.items()):
            if size <= 0 or size % ALIGN:
                raise ValueError('Aligned layer quota required')
            largest = sorted((r['bytes'] for k, r in store.records.items()
                              if int(k.split(':')[0]) == layer), reverse=True)[:6]
            # With at most five protected records, freeing all other entries
            # leaves at most six holes.12*largest guarantees one hole fits the
            # next record without moving an active pointer or needing staging
            # for compaction. The original80GiB proportional quotas exceed this.
            if size < 12 * largest[0]:
                raise ValueError('Layer quota lacks the complete-batch fragmentation bound')
            self.regions[layer] = Regions(size, base); base += size
        # Initialize the original cache/dispatch state without its CUDA arena.
        # This intentionally mirrors the pinned base initializers; the new
        # allocation is the sole expert cache and remains physically accounted.
        self.entries = OrderedDict(); self.lock = threading.RLock()
        self.generation = self.resident_bytes = 0; self.closed = False
        self.stats = dict(hits=0, misses=0, evictions=0, failures=0, peak_resident_bytes=0,
            grouped_launches=0, grouped_routes=0, grouped_experts=0, grouped_max_batch=0,
            grouped_cache_flushes=0, grouped_apply_calls=0, packed_fat_experts=0,
            packed_fat_routes=0, heterogeneous_launches=0, heterogeneous_experts=0,
            heterogeneous_distinct_k_triplets=0, mapped_capacity_bytes=capacity_bytes,
            mapped_loads=0, mapped_read_wait_seconds=0., mapped_cpu_copy_seconds=0.,
            decode_calls=0, service_requests=0, service_publications=0,
            ownership_transitions=0)
        self._temps = None; self._temp_rows = 0; self._last_apply = None
        self._heterogeneous_locks = None
        self._concurrency = int(exllamav3_ext.exl3_moe_max_concurrency(self.device.index or 0))
        if self._concurrency < 1: raise ValueError('Unqualified packed kernel concurrency')
        self._arena_pair = self.extension.allocate_system(capacity_bytes)
        self.cpu_arena, self.arena = self._arena_pair
        table_size = 40 * 384 * 13 * 8
        self._table_pair = self.extension.allocate((table_size + ALIGN - 1) // ALIGN * ALIGN)
        self.table_cpu = self._table_pair[0][:table_size].view(torch.int64).view(40, 384, 13)
        self.table_gpu = self._table_pair[1][:table_size].view(torch.int64).view(40, 384, 13)
        self._used_pair = self.extension.allocate(40 * 384 * 8)
        self.used_cpu = self._used_pair[0].view(torch.int64)
        self.used_gpu = self._used_pair[1].view(torch.int64)
        self.table_cpu.zero_(); self.used_cpu.zero_()
        self._table = (ctypes.c_int64 * (40 * 384 * 13)).from_address(self.table_cpu.data_ptr())
        self._used = (ctypes.c_int64 * (40 * 384)).from_address(self.used_cpu.data_ptr())
        self._cpu_base, self._gpu_base = self.cpu_arena.data_ptr(), self.arena.data_ptr()
        self.descriptors = {}
        for key, record in store.records.items():
            layer, eid = map(int, key.split(':')); index = layer * 384 + eid
            bits, offsets, markers = [], [], []
            for p in ('w1', 'w3', 'w2'):
                t = record['tensors']; k = t[p + '.trellis']['shape'][-1] // 16
                dims = (t[p + '.suh']['shape'][0], t[p + '.svh']['shape'][0])
                if k not in range(2, 9) or dims != ((2304, 5120) if p == 'w2' else (5120, 2304)):
                    raise ValueError('Original packed geometry differs')
                bits.append(k)
                offsets.extend(t[p + '.' + a]['offset'] for a in ('trellis', 'suh', 'svh'))
                markers.append(t[p + '.mul1']['offset'])
            self.descriptors[key] = (index, bits, offsets, markers)
            self._table[index * 13] = 1
        self.stream = self.mailbox = self.worker = None
        self.mode = 'prefill'; self.host_epoch = self.last_request = 0
        self.stop = threading.Event(); self.error = None
        self._protected = set(); self.events = deque(maxlen=256)
        self._decode = None
        self.graph_leases = {}

    def _bind(self):
        current = torch.cuda.current_stream(self.device)
        if self.stream is not None:
            if self.stream != current:
                raise ValueError('Shared cache requires one owning model stream')
            return
        self.stream = current
        self.mailbox = self.extension.Mailbox(self.table_cpu, self.table_gpu,
                                              self.used_cpu, self.used_gpu, 10_000_000_000)
        self.worker = threading.Thread(target=self._serve, name='sage-shared-miss', daemon=False)
        self.worker.start()

    def _switch(self, mode):
        self._bind()
        if mode == self.mode: return
        # Never hold self.lock while waiting for a GPU miss that needs the
        # handler to acquire that lock. One transition fence, not one per layer.
        self.stream.synchronize()
        with self.lock:
            if any(e.users for e in self.entries.values()):
                raise CacheBusy('Cannot transfer cache ownership with active leases')
            if mode == 'prefill':
                self.host_epoch = self.mailbox.pause()
            else:
                for entry in self.entries.values():
                    entry.events.clear(); entry.projections.clear()
                self._last_apply = None
                self.mailbox.resume(self.host_epoch)
            self.mode = mode
            self.stats['ownership_transitions'] += 1

    def _evict(self, key):
        entry = self.entries[key]
        if entry.users or key in self._protected:
            raise CacheBusy('Current expert use prevents eviction')
        if self.mode == 'decode':
            if entry.events or entry.projections:
                raise RuntimeError('GPU handles must be released before CPU service ownership')
        else:
            for event in entry.events: event.synchronize()
            entry.events.clear(); entry.projections.clear()
        index = self.descriptors[key][0]
        self._table[index * 13] = 1
        for column in range(1, 13): self._table[index * 13 + column] = 0
        self.regions[int(key.split(':')[0])].release(entry.offset)
        self.resident_bytes -= entry.size; del self.entries[key]
        self.stats['evictions'] += 1
        self.events.append(dict(kind='evict', key=key, offset=entry.offset, generation=entry.generation))

    def _load(self, key, cancel):
        self._guard()
        record = self.store.records[key]; layer = int(key.split(':')[0])
        region = self.regions[layer]; offset = region.allocate(record['bytes'])
        while offset is None:
            candidates = [k for k, e in self.entries.items() if int(k.split(':')[0]) == layer
                          and not e.users and k not in self._protected]
            if not candidates: raise CacheBusy('No safe region for the complete expert record')
            victim = min(candidates, key=lambda k: (self._used[self.descriptors[k][0]], k)) if self.mode == 'decode' else candidates[0]
            self._evict(victim); offset = region.allocate(record['bytes'])
        try:
            index, bits, offsets, markers = self.descriptors[key]
            before = time.perf_counter()
            with self.store.read(key, cancel=cancel) as data:
                self.stats['mapped_read_wait_seconds'] += time.perf_counter() - before
                if any(struct.unpack_from('<i', data, p)[0] != -2082680531 for p in markers):
                    raise ValueError('Original MUL1 marker differs')
                source = (ctypes.c_char * record['bytes']).from_buffer(data)
                before = time.perf_counter()
                try: ctypes.memmove(self._cpu_base + offset, source, record['bytes'])
                finally: del source
                self.stats['mapped_cpu_copy_seconds'] += time.perf_counter() - before
            if cancel is not None and cancel.is_set(): raise InterruptedError('Mapped load cancelled')
            self.generation += 1
            entry = Entry(key, offset, record['bytes'], self.generation, {})
            row = [2, *bits, *[self._gpu_base + offset + p for p in offsets]]
            for column in range(1, 13): self._table[index * 13 + column] = row[column]
            self._table[index * 13] = 2
            self.entries[key] = entry; self.resident_bytes += record['bytes']
            self.stats['mapped_loads'] += 1
            self.stats['peak_resident_bytes'] = max(self.stats['peak_resident_bytes'], self.resident_bytes)
            self.events.append(dict(kind='load', key=key, offset=offset, generation=entry.generation))
            return entry
        except BaseException:
            region.release(offset); self.stats['failures'] += 1
            raise

    def _materialize(self, entry):
        if entry.projections: return
        target = self.arena.narrow(0, entry.offset, entry.size)
        dtype = {'I16': torch.int16, 'F16': torch.float16, 'I32': torch.int32}
        for p in ('w1', 'w2', 'w3'):
            tensors = {}
            for suffix in ('trellis', 'suh', 'svh', 'mul1'):
                spec = self.store.records[entry.key]['tensors'][p + '.' + suffix]
                tensors[suffix] = target.narrow(0, spec['offset'], spec['bytes']).view(dtype[spec['dtype']]).view(spec['shape'])
            inner = LinearEXL3(config=None, in_features=tensors['suh'].numel(),
                out_features=tensors['svh'].numel(), out_dtype=torch.float16,
                transformers_fix=True, **tensors)
            inner.config.infer_params.no_reconstruct = True
            entry.projections[p] = inner

    @contextmanager
    def lease(self, key, *, cancel=None):
        if self.mode != 'prefill': raise RuntimeError('Host leases require explicit prefill ownership')
        with super().lease(key, cancel=cancel) as entry:
            self._materialize(entry)
            self.host_epoch += 1; self._used[self.descriptors[key][0]] = self.host_epoch
            yield entry

    def _serve(self):
        try:
            while not self.stop.is_set():
                request = self.mailbox.poll(20)
                if request == [-1]:
                    print({'sage_service_failure':self.mailbox.cancellation_code(),'last_request':self.last_request},flush=True)
                    break
                if not request: continue
                with self.lock:
                    sequence, layer, *ids = request
                    if self.mode != 'decode' or sequence <= self.last_request or len(ids) != 6:
                        raise RuntimeError('Invalid cache ownership or miss sequence')
                    self._protected = {f'{layer}:{eid}' for eid in ids if f'{layer}:{eid}' in self.store.records}
                    missing = [k for k in dict.fromkeys(f'{layer}:{eid}' for eid in ids)
                               if k in self._protected and k not in self.entries]
                    if not missing: raise RuntimeError('Request has no owned miss')
                    try:
                        self.store.plan(missing)
                        for key in missing:
                            if self.stop.is_set() or self.mailbox.cancellation_code(): raise InterruptedError('Service cancelled')
                            self.stats['misses'] += 1; self._load(key, self.stop)
                        if not self._protected <= self.entries.keys(): raise RuntimeError('Partial owned miss batch')
                        self.last_request = sequence
                        self.stats['service_requests'] += 1; self.stats['service_publications'] += 1
                        self.stats.update({'prefetch_' + k: v for k, v in self.store.async_stats.items()})
                        self.stats['prefetch_staging_bytes'] = self.store.staging_bytes
                        self.stats['prefetch_peak_staging_bytes'] = self.store.peak_staging_bytes
                    finally:
                        self._protected.clear()
                    # Last shared-state mutation in this transaction. GPU can
                    # start the next consumer immediately after this release.
                    self.mailbox.commit(sequence)
        except BaseException as error:
            self.error = repr(error)
            print({'sage_service_error':self.error,'cause':self.mailbox.cancellation_code(),'last_request':self.last_request},flush=True)
            self.stop.set(); self.mailbox.cancel()

    def _decode_buffers(self):
        if self._decode is None:
            d = self.device
            self._decode = dict(counts=torch.empty(7, dtype=torch.int64, device=d),
                tokens=torch.empty(6, dtype=torch.int64, device=d),
                weights=torch.empty(6, dtype=torch.float16, device=d),
                bits=torch.empty((6, 3), dtype=torch.int64, device=d),
                pointers=torch.empty((9, 6), dtype=torch.int64, device=d),
                miss=torch.empty(1, dtype=torch.int32, device=d),
                locks=torch.zeros(self.LOCK_INTS, dtype=torch.int32, device=d),
                temps=tuple(torch.empty((self._concurrency, 16, n), dtype=torch.float16, device=d)
                            for n in (5120, 5120, 2304, 2304)))
        return self._decode

    def apply(self, layer, x, ids, weights, *, limit=10.0):
        if self.closed or self.error: raise RuntimeError(self.error or 'Shared cache closed')
        if layer not in self.regions or x.ndim != 2 or x.shape[1] != 5120 or ids.shape != weights.shape or ids.shape != (x.shape[0], 6):
            raise ValueError('Original layer/top-six geometry required')
        if any(t.device != self.device for t in (x, ids, weights)) or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('Original same-device routing required')
        if not x.is_floating_point() or not weights.is_floating_point(): raise ValueError('Floating inputs required')
        if limit is not None and (not math.isfinite(limit) or limit <= 0): raise ValueError('Invalid activation limit')
        if torch.cuda.is_current_stream_capturing() and (not self.graph_leases or self.mode != 'decode' or self._decode is None or x.shape[0] != 1):
            raise RuntimeError('Captured decode requires prepared C1 storage and graph ownership')
        if x.shape[0] != 1:
            self._switch('prefill')
            return super().apply(layer, x, ids, weights, limit=limit)
        self._switch('decode')
        b = self._decode_buffers()
        xh, rid, rw = x.half().contiguous(), ids.long().contiguous(), weights.half().contiguous()
        result = torch.zeros_like(x, dtype=torch.float32)
        self.mailbox.enqueue(layer, rid)
        _routes[(1,)](rid, rw, self.table_gpu[layer], b['counts'], b['tokens'], b['weights'],
                     b['bits'], b['pointers'], b['miss'], BLOCK=8, num_warps=1)
        sage_heterogeneous_ext.moe(xh, result, b['counts'], b['tokens'], b['weights'],
            *b['temps'], b['bits'], b['pointers'], b['locks'], 0. if limit is None else float(limit))
        for t in (xh, rid, rw): t.record_stream(self.stream)
        self.stats['decode_calls'] += 1
        return result

    def prepare_forward(self, tokens):
        if self.closed or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare ownership outside capture/replay')
        current = torch.cuda.current_stream(self.device)
        if self.stream is not None and current != self.stream:
            # Drain on the old stream without holding the service lock, then
            # rebind the paused mailbox before allowing work on the new stream.
            with torch.cuda.stream(self.stream):
                self._switch('prefill')
            self.mailbox.rebind()
            self.stream = current
        self._switch('decode' if tokens == 1 else 'prefill')
        if tokens == 1:
            self._decode_buffers()

    def lease_graph(self, graph):
        if self.closed or id(graph) in self.graph_leases or self.mode != 'decode':
            raise RuntimeError('Invalid C1 graph cache lease')
        self.graph_leases[id(graph)] = graph

    def release_graph(self, graph):
        if self.graph_leases.get(id(graph)) is not graph:
            raise RuntimeError('Unknown expert graph lease')
        self.stream.synchronize()
        graph.reset()
        self.stream.synchronize()
        del self.graph_leases[id(graph)]

    def clear(self):
        if self.graph_leases: raise RuntimeError('Cannot clear expert storage held by a graph')
        if self.closed: return
        if self.stream is not None: self._switch('prefill')
        super().clear()

    def close(self):
        if self.graph_leases: raise RuntimeError('Cannot close expert storage held by a graph')
        try:
            with torch.cuda.stream(self.stream) if self.stream is not None else nullcontext():
                self._close_owned()
        except BaseException:
            # A failed CUDA stream cannot certify pointer reclamation, but it
            # must not leave a non-daemon CPU producer waiting indefinitely.
            # Retain arena/metadata handles for owned-worker termination.
            self._stop_service()
            raise

    def _stop_service(self):
        self.stop.set()
        if self.mailbox is not None: self.mailbox.cancel()
        if self.worker is not None:
            self.worker.join(timeout=15)
            if self.worker.is_alive(): raise RuntimeError('CPU service still holds mapped storage')

    def _close_owned(self):
        if self.graph_leases: raise RuntimeError('Cannot close expert storage held by a graph')
        if self.closed: return
        # Drain and transfer ownership before taking the inherited close lock.
        if self.stream is not None: self._switch('prefill')
        self._stop_service()
        super().close()
        if self.mailbox is not None: self.mailbox.close()
        self._decode = self.mailbox = None
        self._table = self._used = None
        self.table_cpu = self.table_gpu = self.used_cpu = self.used_gpu = None
        self._table_pair = self._used_pair = None
        self.cpu_arena = self._arena_pair = None
