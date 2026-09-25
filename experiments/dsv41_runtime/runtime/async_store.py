"""Bounded verified expert prefetch; original packed bytes and direct I/O.

At most two records, including completed and leased records, own staging RAM.
No CUDA work occurs on reader threads. The existing cache owns upload and
CUDA lifetimes; disk read/hash work overlaps its uploads and expert execution.
"""
from collections import OrderedDict, deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import mmap
import os
import threading

from expert_store import ALIGN, ExpertStore


@dataclass
class Ticket:
    key: str
    stop: threading.Event = field(default_factory=threading.Event)
    consumer_cancel: object = None
    future: object = None
    claimed: bool = False

    def cancelled(self):
        return self.stop.is_set() or (self.consumer_cancel is not None and self.consumer_cancel.is_set())


class AsyncExpertStore(ExpertStore):
    def __init__(self,*args,slots=2,**kwargs):
        if type(slots) is not int or not 1<=slots<=2:
            raise ValueError('Only one or two staging slots are permitted')
        super().__init__(*args,**kwargs)
        self.slots=slots
        self.control=threading.RLock()
        self.pool=ThreadPoolExecutor(max_workers=slots,thread_name_prefix='sage-expert-read')
        self.pending=OrderedDict()
        self.queue=deque()
        self.closed=False
        self.staging_bytes=0
        self.async_stats={'submitted':0,'prefetched_consumed':0,'demand_submitted':0,
            'cancelled':0,'failed':0,'peak_active_reads':0,'peak_pending':0}

    def _fetch(self,ticket):
        record=self.records[ticket.key]
        buffer=None
        with self.lock:
            self.active_reads+=1
            self.async_stats['peak_active_reads']=max(self.async_stats['peak_active_reads'],self.active_reads)
        try:
            if ticket.cancelled():raise InterruptedError('Expert prefetch cancelled')
            buffer=mmap.mmap(-1,record['bytes'])
            with self.lock:
                self.staging_bytes+=record['bytes']
                self.peak_staging_bytes=max(self.peak_staging_bytes,self.staging_bytes)
            got=0
            while got<record['bytes']:
                if ticket.cancelled():raise InterruptedError('Expert prefetch cancelled')
                try:
                    if self.direct:
                        view=memoryview(buffer)[got:]
                        try:n=os.preadv(self.fd,[view],record['offset']+got)
                        finally:view.release()
                    else:
                        data=os.pread(self.fd,min(2**20,record['bytes']-got),record['offset']+got)
                        n=len(data);buffer[got:got+n]=data
                except InterruptedError:
                    if ticket.cancelled():raise
                    continue
                if n<=0:raise OSError('Short expert read')
                got+=n
                if self.direct and got<record['bytes'] and got%ALIGN:
                    raise OSError('Unaligned partial direct read')
            with self.lock:
                self.reads+=1;self.bytes_read+=got
            if hashlib.sha256(buffer).hexdigest()!=record['sha256']:
                raise ValueError('Expert record hash mismatch')
            if ticket.cancelled():raise InterruptedError('Expert prefetch cancelled')
            return buffer
        except BaseException as error:
            with self.lock:
                self.async_stats['cancelled' if isinstance(error,InterruptedError) else 'failed']+=1
            if buffer is not None:self._release_buffer(buffer)
            raise
        finally:
            with self.lock:self.active_reads-=1

    def _release_buffer(self,buffer):
        size=len(buffer)
        buffer.close()  # Refuse release if a consumer still exports a view.
        with self.lock:self.staging_bytes-=size

    def _check(self):
        if self.closed or self.fd is None:raise RuntimeError('Expert store is closed')

    def _submit(self,key):
        if len(self.pending)>=self.slots or key in self.pending:
            raise RuntimeError('Staging bound or duplicate submission')
        ticket=Ticket(key)
        self.pending[key]=ticket
        ticket.future=self.pool.submit(self._fetch,ticket)
        self.async_stats['submitted']+=1
        self.async_stats['peak_pending']=max(self.async_stats['peak_pending'],len(self.pending))
        return ticket

    def _fill(self):
        while self.queue and len(self.pending)<self.slots and not self.closed:
            key=self.queue.popleft()
            if key not in self.pending:self._submit(key)

    def _discard(self,ticket):
        if ticket.claimed:raise RuntimeError('Cannot discard a leased staging record')
        ticket.stop.set()
        ticket.future.cancel()
        try:
            buffer=ticket.future.result()
        except (InterruptedError,CancelledError):
            self.pending.pop(ticket.key,None)
        except BaseException:
            self.pending.pop(ticket.key,None)
            raise
        else:
            self._release_buffer(buffer)
            self.pending.pop(ticket.key,None)

    def plan(self,keys):
        keys=list(dict.fromkeys(keys))
        if len(keys)>384 or any(key not in self.records for key in keys):
            raise ValueError('Invalid bounded expert read plan')
        with self.control:
            self._check()
            for ticket in list(self.pending.values()):
                if ticket.key not in keys and not ticket.claimed:self._discard(ticket)
            self.queue=deque(k for k in keys if k not in self.pending)
            self._fill()

    @contextmanager
    def read(self,key,*,cancel=None):
        with self.control:
            self._check()
            if key not in self.records:raise KeyError(key)
            if cancel is not None and cancel.is_set():raise InterruptedError('Expert read cancelled')
            ticket=self.pending.get(key)
            if ticket is None:
                # Unexpected demand takes priority without allocating a third
                # record. Return displaced work to the bounded future plan.
                if len(self.pending)>=self.slots:
                    victim=next((t for t in reversed(self.pending.values()) if not t.claimed),None)
                    if victim is None:raise RuntimeError('All staging slots are leased')
                    self._discard(victim)
                    self.queue.appendleft(victim.key)
                self.queue=deque(k for k in self.queue if k!=key)
                ticket=self._submit(key)
                self.async_stats['demand_submitted']+=1
            else:
                self.async_stats['prefetched_consumed']+=1
            if ticket.claimed:raise RuntimeError('Expert staging record already leased')
            ticket.claimed=True
            ticket.consumer_cancel=cancel
        buffer=None
        try:
            buffer=ticket.future.result()
            if ticket.cancelled():raise InterruptedError('Expert read cancelled')
            yield buffer
        finally:
            with self.control:
                if buffer is not None:
                    try:self._release_buffer(buffer)
                    except BufferError:
                        ticket.claimed=False
                        raise
                self.pending.pop(key,None)
                self._fill()

    def close(self):
        with self.control:
            if self.closed:return
            if any(t.claimed for t in self.pending.values()):
                raise RuntimeError('Cannot close expert store with active staging leases')
            self.closed=True
            self.queue.clear()
            errors=[]
            for ticket in list(self.pending.values()):
                try:self._discard(ticket)
                except BaseException as error:errors.append(error)
            self.pool.shutdown(wait=True,cancel_futures=True)
            super().close()
            if self.staging_bytes or self.active_reads:
                raise RuntimeError('Staging storage was not fully released')
            if errors:raise errors[0]
