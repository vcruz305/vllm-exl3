"""Graph-capturable original Engram rows with explicit graph leases.

Native hash calculation and native embed collectives remain outside this class.
A caller must prepare storage before capture and keep every graph registered
until reset plus completion. CPU callbacks never call CUDA or enter Python.
"""
import ctypes as C
import glob
from pathlib import Path
import torch
import triton
import triton.language as tl
from native_rows import Work, load, create, stats


@triton.jit
def _dequant(W, S, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N * 256
    values = tl.load(W + offsets, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(S + offsets // 32, mask=valid, other=0)
    factor = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    tl.store(OUT + offsets, (values * factor).to(tl.bfloat16), mask=valid)


def cudart():
    paths = glob.glob('/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so*')
    paths += glob.glob('/usr/local/cuda/lib64/libcudart.so*')
    for path in paths:
        try:
            lib = C.CDLL(path)
            lib.cudaLaunchHostFunc.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
            lib.cudaLaunchHostFunc.restype = C.c_int
            return lib
        except OSError:
            continue
    raise RuntimeError('Installed CUDA runtime unavailable')


class GraphRows:
    def __init__(self, embedding, library, max_tokens, *, capacity=131072):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Allocate row stage before capture')
        if embedding.dp_size != 1 or embedding.dim != 256:
            raise ValueError('Initial graph stage requires native DP1 rows')
        self.embedding = embedding
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.stream = torch.cuda.current_stream(self.device)
        self.closed = False
        self.graphs = {}
        self.works = {}
        self.lib = load(library)
        self.cuda = cudart()
        self.max_tokens = int(max_tokens)
        cap = self.max_tokens * embedding.part_n_hash_cols
        if not 0 < cap <= 8192:
            raise ValueError('Row staging exceeds admitted 8192-row bound')
        self.host_ids = torch.empty(cap, dtype=torch.int64, device='cpu', pin_memory=True)
        self.host_w = torch.empty((cap,256), dtype=torch.uint8, device='cpu', pin_memory=True)
        self.host_s = torch.empty((cap,8), dtype=torch.uint8, device='cpu', pin_memory=True)
        self.dev_w = torch.empty_like(self.host_w, device=self.device)
        self.dev_s = torch.empty_like(self.host_s, device=self.device)
        self.handle = create(self.lib, embedding.disk, embedding.vocab_start_idx,
                             embedding.vocab_end_idx, capacity)

    def prepare_shape(self, count):
        if self.closed or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare shape on a live stage outside capture')
        if not 0 <= count <= self.max_tokens:
            raise ValueError('Unqualified token count')
        rows = count * self.embedding.part_n_hash_cols
        if rows not in self.works:
            self.works[rows] = Work(self.handle, self.host_ids.data_ptr(),
                                   self.host_w.data_ptr(), self.host_s.data_ptr(), rows)

    def transfer_stream(self):
        if self.closed or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Stream ownership changes require an external fence')
        # All previously queued reads/copies/consumers finish before new use.
        self.stream.synchronize()
        self.stream = torch.cuda.current_stream(self.device)

    def lease_graph(self, graph):
        if self.closed or id(graph) in self.graphs:
            raise RuntimeError('Invalid graph lease')
        self.graphs[id(graph)] = graph

    def release_graph(self, graph):
        if self.graphs.get(id(graph)) is not graph:
            raise RuntimeError('Unknown graph lease')
        self.stream.synchronize()
        graph.reset()
        self.stream.synchronize()
        del self.graphs[id(graph)]

    def lookup(self, indices, out):
        if self.closed or torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError('Row stage requires its live owning stream')
        e = self.embedding
        count = indices.shape[0]
        rows = count * e.part_n_hash_cols
        if indices.ndim != 2 or indices.shape[1] != e.n_hash_cols or indices.dtype not in (torch.int32,torch.int64):
            raise ValueError('Native hash geometry required')
        if indices.device != self.device or out.device != self.device or out.dtype != torch.bfloat16:
            raise ValueError('Native GPU hash/output storage required')
        if out.shape != (count,e.part_n_hash_cols,256) or not out.is_contiguous() or rows not in self.works:
            raise ValueError('Prepare exact stable output shape before use')
        if torch.cuda.is_current_stream_capturing() and not self.graphs:
            raise RuntimeError('Captured callbacks require an explicit graph lease')
        if not rows:
            return
        end = min(e.head_start + e.part_n_hash_cols, e.n_hash_cols)
        ids = indices[:,e.head_start:end]
        if ids.shape[1] < e.part_n_hash_cols:
            ids = torch.cat((ids, ids.new_full((count,e.part_n_hash_cols-ids.shape[1]),-1)),dim=1)
        ids = ids.reshape(-1).to(dtype=torch.int64)
        self.host_ids[:rows].copy_(ids, non_blocking=True)
        rc = self.cuda.cudaLaunchHostFunc(self.stream.cuda_stream,
            C.cast(self.lib.sg_lookup, C.c_void_p), C.addressof(self.works[rows]))
        if rc:
            raise RuntimeError('CUDA row callback enqueue failed: ' + str(rc))
        self.dev_w[:rows].copy_(self.host_w[:rows], non_blocking=True)
        self.dev_s[:rows].copy_(self.host_s[:rows], non_blocking=True)
        _dequant[(triton.cdiv(rows*256,256),)](
            self.dev_w.view(torch.float8_e4m3fn),self.dev_s,out,N=rows,BLOCK=256)
        ids.record_stream(self.stream)

    def snapshot(self):
        if self.closed:
            return {'closed': True}
        return {**stats(self.lib,self.handle), 'closed':False,'graphs':len(self.graphs),
                'staging_rows':self.host_ids.numel()}

    def close(self):
        if self.closed:
            return
        if self.graphs:
            raise RuntimeError('Destroy all graph leases before freeing callback storage')
        self.stream.synchronize()
        self.lib.sg_destroy(self.handle)
        self.handle = None
        self.works.clear()
        self.closed = True
