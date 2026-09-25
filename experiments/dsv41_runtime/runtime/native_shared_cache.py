"""Original Python prefill cache with an independent native decode service.

Only drained ownership boundaries exchange entry/region state. The native
service retains the same arena and descriptor addresses used by CUDA graphs.
"""
from collections import OrderedDict
import torch
from expert_cache import Entry, CacheBusy
from expert_store import Regions
from shared_mapped_cache import SharedMappedCache as PythonSharedCache


class _WorkerView:
    def __init__(self, service):
        self.service = service

    def is_alive(self):
        return self.service is not None and self.service.is_alive()

    def join(self, timeout=None):
        milliseconds = 15000 if timeout is None else min(15000, int(timeout * 1000))
        if not self.service.close(milliseconds):
            raise RuntimeError('Native reader still owns cache storage')


class NativeSharedCache(PythonSharedCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_service = None
        self._native_seen = {}
        self._native_snapshot = {}
        self._quotas = {
            layer: sum(region.allocated.values()) + sum(n for _, n in region.free)
            for layer, region in self.regions.items()
        }
        # SelectedStore is only the fixture's catalog filter. Its original
        # AsyncExpertStore remains the sole Python staging allocator.
        self._reader = getattr(self.store, 'store', self.store)

    def _bind(self):
        current = torch.cuda.current_stream(self.device)
        if self.stream is not None:
            if self.stream != current:
                raise ValueError('Shared cache requires one owning model stream')
            return
        self.stream = current
        self.mailbox = self.extension.Mailbox(
            self.table_cpu, self.table_gpu, self.used_cpu, self.used_gpu,
            10_000_000_000)
        rows = []
        for key, record in self.store.records.items():
            _, bits, offsets, markers = self.descriptors[key]
            rows.append((key, record['offset'], record['bytes'], record['sha256'],
                         bits, offsets, markers))
        if not self._reader.direct:
            raise ValueError('Native expert service requires original direct I/O')
        self.native_service = self.extension.NativeService(
            self.mailbox, self.cpu_arena, self.arena, self._reader.fd,
            rows, self._quotas, self.reserve, 2)
        self.worker = _WorkerView(self.native_service)

    def _state_for_native(self):
        return (
            [(e.key, e.offset, e.size, e.generation) for e in self.entries.values()],
            {layer: {'free': list(region.free),
                     'allocated': list(region.allocated.items())}
             for layer, region in self.regions.items()},
            self.generation, self.resident_bytes, self.last_request)

    def _accept_native_state(self, state):
        entries = OrderedDict()
        for key, offset, size, generation in state['entries']:
            if key in entries or self.store.records[key]['bytes'] != size:
                raise ValueError('Invalid native entry export')
            entries[key] = Entry(key, offset, size, generation, {})
        regions = {}
        for layer, row in state['regions'].items():
            layer = int(layer)
            region = Regions(self._quotas[layer])
            region.free = [tuple(v) for v in row['free']]
            region.allocated = dict(row['allocated'])
            if sum(region.allocated.values()) + sum(n for _, n in region.free) != self._quotas[layer]:
                raise ValueError('Native region export lost capacity')
            regions[layer] = region
        if set(regions) != set(self._quotas):
            raise ValueError('Native region export changed layers')
        if state['resident_bytes'] != sum(e.size for e in entries.values()):
            raise ValueError('Native residency export differs')
        self.entries, self.regions = entries, regions
        self.generation = state['generation']
        self.resident_bytes = state['resident_bytes']
        self.last_request = state['last_request']
        self.refresh_native_stats(state['stats'])

    def _check_native(self):
        if self.native_service is not None:
            error = self.native_service.error()
            code = self.mailbox.cancellation_code()
            if code and not error:
                error = 'Native GPU mailbox cancelled with code '+str(code)
            if error:
                self.error = error
                raise RuntimeError(error)

    def _switch(self, mode):
        if mode not in ('prefill', 'decode'):
            raise ValueError('Unqualified cache ownership')
        # Inherited close calls clear again after the drained native close.
        # That second prefill clear owns no GPU work and must not interpret
        # the normal close cancellation as a live-service failure.
        if self.stop.is_set() and mode == self.mode == 'prefill':
            if self.native_service is not None:
                if self.native_service.is_alive() or self.native_service.error():
                    raise RuntimeError('Native service did not close cleanly')
            return
        self._bind()
        self._check_native()
        if mode == self.mode:
            return
        # This fence releases the GIL. Never hold the Python cache lock while
        # waiting for the independent native service to finish a GPU miss.
        self.stream.synchronize()
        with self.lock:
            if any(e.users for e in self.entries.values()):
                raise CacheBusy('Cannot transfer cache ownership with active leases')
            if mode == 'prefill':
                self.host_epoch = self.mailbox.pause()
                if not self.native_service.stop(15000, False):
                    raise RuntimeError('Native ownership handoff timed out; retain storage')
                self._check_native()
                self._accept_native_state(self.native_service.export_state())
            else:
                # Ready/prefetched Python buffers count toward the same two
                # staging slots. Drain them before creating native read work.
                self.store.plan([])
                with self._reader.control:
                    if (self._reader.pending or self._reader.queue or
                            self._reader.staging_bytes or self._reader.active_reads):
                        raise CacheBusy('Python reader still owns staging storage')
                for entry in self.entries.values():
                    entry.events.clear()
                    entry.projections.clear()
                self._last_apply = None
                self.native_service.import_state(*self._state_for_native())
                self.native_service.start()
                self.mailbox.resume(self.host_epoch)
            self.mode = mode
            self.stats['ownership_transitions'] += 1

    def refresh_native_stats(self, snapshot=None):
        if self.native_service is None:
            return {}
        snapshot = self.native_service.stats() if snapshot is None else snapshot
        # Native counters are cumulative across ownership epochs. Add only
        # each new delta; original prefill counters remain independently owned.
        mapping = {'loads': 'mapped_loads', 'misses': 'misses',
                   'evictions': 'evictions', 'requests': 'service_requests',
                   'publications': 'service_publications', 'failures': 'failures'}
        for source, destination in mapping.items():
            current = int(snapshot.get(source, 0))
            prior = self._native_seen.get(source, 0)
            if current < prior:
                raise RuntimeError('Native counters moved backward')
            self.stats[destination] += current - prior
            self._native_seen[source] = current
        self.stats['peak_resident_bytes'] = max(
            self.stats['peak_resident_bytes'], int(snapshot.get('peak_resident_bytes', 0)))
        self._native_snapshot = snapshot
        if self.mode == 'decode':
            self.resident_bytes = int(snapshot.get('resident_bytes', self.resident_bytes))
        return snapshot

    def _stop_service(self):
        self.stop.set()
        if self.native_service is not None:
            # Native close marks stopping before mailbox cancellation, so a
            # normal shutdown cannot be mistaken for an unexpected GPU fatal.
            if not self.native_service.close(15000):
                raise RuntimeError('Native CPU service still holds mapped storage')
        elif self.mailbox is not None:
            self.mailbox.cancel()

    def _close_owned(self):
        super()._close_owned()
        # Only after the original stream/graph and service lifetime guards pass.
        if self.worker is not None:
            self.worker.service = None
        self.native_service = None
