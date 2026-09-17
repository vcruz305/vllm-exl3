"""Cumulative per-layer offload observations; leaves computation unchanged."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time

def install(session):
    if getattr(session, '_telemetry_installed', False): return
    cache = session.get_cache()
    session._telemetry_installed = True
    lock = threading.RLock()
    layers = {}
    engram = {'calls': 0, 'rows': 0, 'requested_bytes': 0, 'wall_seconds': 0.0}
    last = [time.monotonic()]
    original_lease = cache.lease
    original_apply = cache.apply

    def stats(layer):
        return layers.setdefault(str(layer), dict(hits=0, misses=0, hit_bytes=0, miss_bytes=0,
            load_seconds=0.0, apply_seconds=0.0, apply_calls=0, rows=0, evictions=0))

    @contextmanager
    def lease(key, **kwargs):
        layer = int(key.split(':')[0]); row = stats(layer)
        with cache.lock:
            hit = key in cache.entries
            before = cache.stats['evictions']
            started = time.monotonic()
            with original_lease(key, **kwargs) as entry:
                row['hits' if hit else 'misses'] += 1
                row['hit_bytes' if hit else 'miss_bytes'] += entry.size
                row['evictions'] += cache.stats['evictions']-before
                if not hit: row['load_seconds'] += time.monotonic()-started
                yield entry

    def snapshot():
        import torch
        from expert_cache import host_memory
        return {'sage_offload': 'telemetry', 'rank': session.rank, 'time': time.time(),
            'identity': getattr(session, 'runtime_identity', None),
            'layers': layers, 'engram': engram, 'engram_cache': __import__('engram_prefetch').snapshot(), 'cache': dict(cache.stats),
            'resident_expert_bytes': cache.resident_bytes, 'expert_disk_bytes': session.store.bytes_read,
            'cuda_allocated': torch.cuda.memory_allocated(), 'cuda_reserved': torch.cuda.memory_reserved(),
            'host': host_memory(), 'timing_scope': 'host wall time; GPU apply is not forcibly synchronized'}

    def emit(force=False):
        with lock:
            if not force and time.monotonic()-last[0] < 30: return
            payload = snapshot()
            target = os.environ.get('SAGE_TELEMETRY_DIR')
            if target:
                directory = Path(target); directory.mkdir(exist_ok=True, parents=True)
                temporary = directory/f'rank{session.rank}.json.tmp'
                temporary.write_text(json.dumps(payload, sort_keys=True)+'\n')
                os.replace(temporary, directory/f'rank{session.rank}.json')
            print(json.dumps(payload, sort_keys=True), flush=True)
            last[0] = time.monotonic()

    def apply(layer, *args, **kwargs):
        started = time.monotonic()
        result = original_apply(layer, *args, **kwargs)
        with lock:
            row = stats(layer); row['apply_calls'] += 1; row['rows'] += int(args[0].shape[0])
            row['apply_seconds'] += time.monotonic()-started
        if layer == max(session.quotas): emit()
        return result

    cache.lease = lease; cache.apply = apply
    session.telemetry_snapshot = snapshot; session.telemetry_emit = emit
    # Existing Engram dequantization, ownership and read strategy stay intact.
    from vllm.models.deepseek_v4_1.common.engram_disk import DiskEngramTable
    original_gather = DiskEngramTable.gather_dequant
    def gather(table, rel, owned):
        started = time.monotonic()
        result = original_gather(table, rel, owned)
        with lock:
            engram['calls'] += 1; engram['rows'] += int(rel.numel())
            engram['requested_bytes'] += int(rel.numel())*(table.dim+table.sb)
            engram['wall_seconds'] += time.monotonic()-started
        return result
    DiskEngramTable.gather_dequant = gather
