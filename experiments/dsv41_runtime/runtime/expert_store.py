"""Byte-preserving expert records and strict, bounded disk reads.

Serving layout only: this module does not quantize or alter model weights.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import struct
import tempfile
import threading

ALIGN = 4096
WIDTH = {'I16': 2, 'F16': 2, 'I32': 4}
SUFFIXES = ('trellis', 'suh', 'svh', 'mul1')
MODEL = 'vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw'
REVISION = 'e831e9e4d6bfeafa6d630848296417b1393404a3'


def aligned(n, alignment=ALIGN):
    return (n + alignment - 1) // alignment * alignment


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_tensors(tensors):
    expected = {f'{p}.{s}' for p in ('w1', 'w2', 'w3') for s in SUFFIXES}
    if set(tensors) != expected:
        raise ValueError('Expected exactly twelve MUL1 expert tensors')
    dimensions = {}
    for p in ('w1', 'w2', 'w3'):
        t, u, v, m = (tensors[f'{p}.{s}'] for s in SUFFIXES)
        if [t['dtype'], u['dtype'], v['dtype'], m['dtype']] != ['I16', 'F16', 'F16', 'I32']:
            raise ValueError('Invalid expert dtypes')
        if len(u['shape']) != 1 or len(v['shape']) != 1 or m['shape'] not in ([], [1]):
            raise ValueError('Invalid scale or marker shape')
        ni, no = u['shape'][0], v['shape'][0]
        if ni <= 0 or no <= 0 or ni % 128 or no % 128:
            raise ValueError('Invalid EXL3 dimensions')
        shape = t['shape']
        if len(shape) != 3 or shape[:2] != [ni//16, no//16] or shape[2] not in range(32, 129, 16):
            raise ValueError('Invalid EXL3 K2-K8 geometry')
        dimensions[p] = (ni, no)
    if dimensions['w1'] != dimensions['w3'] or dimensions['w2'] != dimensions['w1'][::-1]:
        raise ValueError('Inconsistent expert projections')


def build_bank(source, destination, prefixes, *, model=MODEL, revision=REVISION):
    """Build a new bank directory atomically from existing safetensors bytes."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if not prefixes or len(set(prefixes)) != len(prefixes):
        raise ValueError('Expert selection must be nonempty and unique')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.expert-bank-', dir=destination.parent))
    try:
        with source.open('rb') as src, (temporary/'experts.bin').open('xb') as out:
            raw = src.read(8)
            if len(raw) != 8:
                raise ValueError('Missing source header')
            size = struct.unpack('<Q', raw)[0]
            if not 2 <= size <= 32*2**20:
                raise ValueError('Invalid source header size')
            header = json.loads(src.read(size))
            source_size = source.stat().st_size
            records = {}
            for prefix in sorted(prefixes):
                parts = prefix.split('.')
                if len(parts) != 5 or parts[0] != 'layers' or parts[2:4] != ['ffn','experts']:
                    raise ValueError('Invalid expert prefix')
                key = f'{int(parts[1])}:{int(parts[4])}'
                if key in records:
                    raise ValueError('Duplicate expert identity')
                tensors = {}
                record = bytearray()
                for p in ('w1', 'w2', 'w3'):
                    for suffix in SUFFIXES:
                        name = f'{prefix}.{p}.{suffix}'
                        spec = header[name]
                        start, end = spec['data_offsets']
                        shape = spec['shape']
                        if any(type(d) is not int or d < 0 for d in shape):
                            raise ValueError('Invalid source shape')
                        nbytes = math.prod(shape)*WIDTH[spec['dtype']]
                        if end-start != nbytes or not 0 <= start < end <= source_size-8-size:
                            raise ValueError('Invalid source extent')
                        src.seek(8+size+start)
                        data = src.read(nbytes)
                        if len(data) != nbytes:
                            raise OSError('Truncated source tensor')
                        if suffix == 'mul1' and struct.unpack('<i', data)[0] != -2082680531:
                            raise ValueError('Wrong MUL1 marker')
                        offset = aligned(len(record), 256)
                        record.extend(bytes(offset-len(record)))
                        record.extend(data)
                        tensors[f'{p}.{suffix}'] = dict(dtype=spec['dtype'], shape=shape,
                            offset=offset, bytes=nbytes, sha256=hashlib.sha256(data).hexdigest())
                validate_tensors(tensors)
                record.extend(bytes(aligned(len(record))-len(record)))
                records[key] = dict(offset=out.tell(), bytes=len(record), tensors=tensors,
                                    sha256=hashlib.sha256(record).hexdigest())
                out.write(record)
            out.flush()
            os.fsync(out.fileno())
        manifest = dict(version=1, model=model, revision=revision, alignment=ALIGN,
                        source_sha256=digest_file(source), bank_bytes=(temporary/'experts.bin').stat().st_size,
                        records=records)
        with (temporary/'manifest.json').open('x') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.rename(temporary, destination)
        return digest_file(destination/'manifest.json')
    except BaseException:
        # Only files created by this invocation are eligible for cleanup.
        for name in ('experts.bin', 'manifest.json'):
            (temporary/name).unlink(missing_ok=True)
        temporary.rmdir()
        raise


class ExpertStore:
    def __init__(self, directory, expected_manifest_sha256, *, direct=True, max_record_bytes=64*2**20):
        directory = Path(directory)
        if digest_file(directory/'manifest.json') != expected_manifest_sha256:
            raise ValueError('Expert manifest identity mismatch')
        self.manifest = json.loads((directory/'manifest.json').read_text())
        m = self.manifest
        if (m['version'], m['model'], m['revision'], m['alignment']) != (1, MODEL, REVISION, ALIGN):
            raise ValueError('Unsupported expert store identity')
        self.records = m['records']
        end = 0
        for key, record in sorted(self.records.items(), key=lambda x: x[1]['offset']):
            layer, expert = map(int, key.split(':'))
            if not 0 <= layer < 40 or not 0 <= expert < 384:
                raise ValueError('Invalid expert identity')
            if record['offset'] != end or record['bytes'] % ALIGN or not 0 < record['bytes'] <= max_record_bytes:
                raise ValueError('Invalid expert record bounds')
            tensors = record['tensors']
            validate_tensors(tensors)
            tensor_end = 0
            for name, tensor in sorted(tensors.items(), key=lambda x: x[1]['offset']):
                offset, nbytes = tensor['offset'], tensor['bytes']
                if (offset < tensor_end or offset % 256 or nbytes != math.prod(tensor['shape'])*WIDTH[tensor['dtype']]
                        or offset+nbytes > record['bytes']):
                    raise ValueError('Invalid tensor extent')
                tensor_end = offset+nbytes
            end += record['bytes']
        if not self.records or end != m['bank_bytes'] or end != (directory/'experts.bin').stat().st_size:
            raise ValueError('Expert bank is incomplete')
        if direct and (not hasattr(os, 'O_DIRECT') or not hasattr(os, 'preadv')):
            raise RuntimeError('Direct I/O unavailable; no automatic buffered fallback')
        self.direct = direct
        self.fd = os.open(directory/'experts.bin', os.O_RDONLY | (os.O_DIRECT if direct else 0))
        self.lock = threading.RLock()
        self.reads = self.bytes_read = self.active_reads = self.peak_staging_bytes = 0

    @contextmanager
    def read(self, key, *, cancel=None):
        """One staging record at a time; its memory expires on context exit."""
        with self.lock:
            if self.fd is None:
                raise RuntimeError('Expert store is closed')
            record = self.records[key]
            if cancel is not None and cancel.is_set():
                raise InterruptedError('Expert read cancelled')
            with mmap.mmap(-1, record['bytes']) as buffer:
                self.active_reads += 1
                self.peak_staging_bytes = max(self.peak_staging_bytes, record['bytes'])
                try:
                    got = 0
                    while got < record['bytes']:
                        if cancel is not None and cancel.is_set():
                            raise InterruptedError('Expert read cancelled')
                        try:
                            if self.direct:
                                view = memoryview(buffer)[got:]
                                try:
                                    n = os.preadv(self.fd, [view], record['offset']+got)
                                finally:
                                    view.release()
                            else:
                                data = os.pread(self.fd, min(2**20, record['bytes']-got), record['offset']+got)
                                n = len(data)
                                buffer[got:got+n] = data
                        except InterruptedError:
                            if cancel is not None and cancel.is_set():
                                raise
                            continue
                        if n <= 0:
                            raise OSError('Short expert read')
                        got += n
                        if self.direct and got < record['bytes'] and got % ALIGN:
                            raise OSError('Unaligned partial direct read')
                    self.reads += 1
                    self.bytes_read += got
                    if hashlib.sha256(buffer).hexdigest() != record['sha256']:
                        raise ValueError('Expert record hash mismatch')
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError('Expert read cancelled')
                    yield buffer
                finally:
                    self.active_reads -= 1

    def close(self):
        with self.lock:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


class Regions:
    """Exact, aligned allocation accounting inside a fixed arena."""
    def __init__(self, size, base=0):
        if size <= 0 or size % ALIGN or base % ALIGN:
            raise ValueError('Arena must be positive and aligned')
        self.free = [(base, size)]
        self.allocated = {}

    def allocate(self, nbytes):
        if nbytes <= 0:
            raise ValueError('Invalid allocation size')
        size = aligned(nbytes)
        for i, (offset, length) in enumerate(self.free):
            if length >= size:
                self.free[i:i+1] = [(offset+size, length-size)] if length > size else []
                self.allocated[offset] = size
                return offset
        return None

    def release(self, offset):
        size = self.allocated.pop(offset)
        self.free.append((offset, size))
        merged = []
        for a, b in sorted(self.free):
            if merged and merged[-1][0]+merged[-1][1] == a:
                merged[-1] = (merged[-1][0], merged[-1][1]+b)
            else:
                merged.append((a,b))
        self.free = merged
