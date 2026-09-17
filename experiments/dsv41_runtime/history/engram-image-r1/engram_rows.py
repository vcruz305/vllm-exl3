"""Bounded raw Engram row cache and one-batch asynchronous prefetch.

This module performs no dequantization or hashing. Callers retain the native
global row IDs, ownership mask, FP8 weights and E8M0 scales. No unowned row is
read. The row count and per-request cap bound payload and Python metadata.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from functools import wraps
import os
import threading


def serialized(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        with self.control:return method(self,*args,**kwargs)
    return call


class EngramRows:
    MAX_ROWS=131072
    MAX_REQUEST=8192

    def __init__(self,w_fd,w_offset,s_fd,s_offset,rows,dim,scale_bytes,*,capacity_rows=131072,workers=4):
        if type(capacity_rows) is not int or type(workers) is not int or not 0<capacity_rows<=self.MAX_ROWS or not 1<=workers<=4:
            raise ValueError('Engram cache capacity or worker bound exceeded')
        if rows<=0 or (dim,scale_bytes)!=(256,8) or min(w_offset,s_offset)<0:
            raise ValueError('Invalid Engram geometry')
        self.w_fd=os.dup(w_fd)
        try:self.s_fd=os.dup(s_fd)
        except BaseException:
            os.close(self.w_fd);raise
        self.w_offset,self.s_offset=w_offset,s_offset
        self.rows,self.dim,self.sb=rows,dim,scale_bytes
        self.capacity=capacity_rows
        self.cache=OrderedDict()
        self.lock=threading.RLock()
        self.control=threading.RLock()
        self.pending=None
        self.closed=False
        self.read_pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='sage-engram-row')
        self.prefetch_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='sage-engram-prefetch')
        self.identity=(self._identity(self.w_fd),self._identity(self.s_fd))
        if self.identity[0][2]<w_offset+rows*dim or self.identity[1][2]<s_offset+rows*scale_bytes:
            self.close();raise ValueError('Engram table extent exceeds file')
        self.stats=dict(requests=0,owned_rows=0,distinct_rows=0,unique_hits=0,unique_misses=0,
            disk_weight_bytes=0,disk_scale_bytes=0,evictions=0,peak_cached_rows=0,
            prefetch_submitted=0,prefetch_consumed=0,prefetch_unused=0)

    @staticmethod
    def _identity(fd):
        stat=os.fstat(fd)
        return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns

    def _validate(self,rows,owned):
        rows,owned=tuple(rows),tuple(owned)
        if self.closed:raise RuntimeError('Engram row cache is closed')
        if len(rows)!=len(owned) or len(rows)>self.MAX_REQUEST:
            raise ValueError('Engram request shape or row bound exceeded')
        if any(type(r) is not int or not 0<=r<self.rows for r in rows) or any(type(v) is not bool for v in owned):
            raise ValueError('Invalid Engram row ID or ownership mask')
        return rows,owned

    @staticmethod
    def _read(fd,offset,size):
        data=bytearray(size);got=0
        while got<size:
            try:chunk=os.pread(fd,size-got,offset+got)
            except InterruptedError:continue
            if not chunk:raise OSError('Short Engram row read')
            data[got:got+len(chunk)]=chunk;got+=len(chunk)
        return bytes(data)

    def _fetch(self,keys):
        if (self._identity(self.w_fd),self._identity(self.s_fd))!=self.identity:
            raise ValueError('Engram table identity changed')
        def chunk(batch):
            return [(row,self._read(self.w_fd,self.w_offset+row*self.dim,self.dim)+
                         self._read(self.s_fd,self.s_offset+row*self.sb,self.sb)) for row in batch]
        futures=[]
        try:
            for start in range(0,len(keys),16):futures.append(self.read_pool.submit(chunk,keys[start:start+16]))
            result=[item for future in futures for item in future.result()]
            if (self._identity(self.w_fd),self._identity(self.s_fd))!=self.identity:
                raise ValueError('Engram table changed during read')
            self.stats['disk_weight_bytes']+=len(keys)*self.dim
            self.stats['disk_scale_bytes']+=len(keys)*self.sb
            return result
        finally:
            for future in futures:future.cancel()
            wait(futures)
            # Preserve the existing table-only page-release policy. Hot rows
            # live in this explicitly bounded cache, not an unbounded page cache.
            if hasattr(os,'posix_fadvise'):
                for fd in (self.w_fd,self.s_fd):os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)

    def _payloads(self,signature):
        rows,owned=signature
        with self.lock:
            if self.closed:raise RuntimeError('Engram row cache is closed')
            if (self._identity(self.w_fd),self._identity(self.s_fd))!=self.identity:
                raise ValueError('Engram table identity changed')
            keys=list(dict.fromkeys(r for r,keep in zip(rows,owned) if keep))
            payloads={};missing=[]
            for key in keys:
                if key in self.cache:
                    payloads[key]=self.cache[key];self.cache.move_to_end(key)
                else:missing.append(key)
            # Retain this request's bytes separately: a request may exceed the
            # cache capacity and evict early rows before its output is assembled.
            for key,data in self._fetch(missing) if missing else ():
                payloads[key]=data
                if len(self.cache)>=self.capacity:
                    self.cache.popitem(last=False);self.stats['evictions']+=1
                self.cache[key]=data
            self.stats['requests']+=1
            self.stats['owned_rows']+=sum(owned)
            self.stats['distinct_rows']+=len(keys)
            self.stats['unique_misses']+=len(missing)
            self.stats['unique_hits']+=len(keys)-len(missing)
            self.stats['peak_cached_rows']=max(self.stats['peak_cached_rows'],len(self.cache))
            return [payloads[r] if keep else None for r,keep in zip(rows,owned)]

    @serialized
    def prefetch(self,rows,owned):
        signature=self._validate(rows,owned)
        # Scheduling control is separate from the cache lock held during I/O,
        # so this call returns while the bounded worker batch runs.
        if self.pending is not None:raise RuntimeError('One Engram prefetch batch may be outstanding')
        future=self.prefetch_pool.submit(self._payloads,signature)
        self.pending=(signature,future)
        self.stats['prefetch_submitted']+=1

    @serialized
    def gather_into(self,rows,owned,weight,scale):
        signature=self._validate(rows,owned)
        count=len(signature[0])
        if len(weight)!=count*self.dim or len(scale)!=count*self.sb or weight.readonly or scale.readonly:
            raise ValueError('Invalid Engram output storage')
        if self.pending is not None:
            expected,future=self.pending
            payloads=future.result()
            self.pending=None
            if expected==signature:self.stats['prefetch_consumed']+=1
            else:
                self.stats['prefetch_unused']+=1
                payloads=self._payloads(signature)
        else:payloads=self._payloads(signature)
        for i,data in enumerate(payloads):
            weight[i*self.dim:(i+1)*self.dim]=data[:self.dim] if data is not None else bytes(self.dim)
            scale[i*self.sb:(i+1)*self.sb]=data[self.dim:] if data is not None else bytes(self.sb)

    @serialized
    def close(self):
        if self.closed:return
        error=None
        if self.pending is not None:
            try:self.pending[1].result()
            except BaseException as caught:error=caught
            self.pending=None
        self.prefetch_pool.shutdown(wait=True,cancel_futures=True)
        self.read_pool.shutdown(wait=True,cancel_futures=True)
        with self.lock:
            self.closed=True
            self.cache.clear()
            os.close(self.w_fd);os.close(self.s_fd)
        if error:raise error
