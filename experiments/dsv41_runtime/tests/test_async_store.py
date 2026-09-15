"""Synthetic byte-transport/control tests, not model numerical evidence."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from expert_store import build_bank
from async_store import AsyncExpertStore


class AsyncStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        root=Path(cls.temp.name)
        header={};data=bytearray();prefixes=[]
        for eid in (1,2,3,4):
            prefix=f'layers.0.ffn.experts.{eid}';prefixes.append(prefix)
            for projection in ('w1','w2','w3'):
                for suffix,dtype,shape,nbytes in (
                    ('trellis','I16',[8,8,(eid+1)*16],8*8*(eid+1)*16*2),
                    ('suh','F16',[128],256),('svh','F16',[128],256),('mul1','I32',[],4)):
                    start=len(data)
                    data.extend(struct.pack('<i',-2082680531) if suffix=='mul1' else bytes([eid*17])*nbytes)
                    header[prefix+'.'+projection+'.'+suffix]={'dtype':dtype,'shape':shape,'data_offsets':[start,len(data)]}
        encoded=json.dumps(header).encode()
        source=root/'synthetic.safetensors';source.write_bytes(struct.pack('<Q',len(encoded))+encoded+data)
        cls.bank=root/'bank';cls.digest=build_bank(source,cls.bank,prefixes)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def store(self,**kwargs):return AsyncExpertStore(self.bank,self.digest,direct=False,**kwargs)

    def test_two_reads_overlap_and_completed_buffers_remain_bounded(self):
        store=self.store();real=os.pread;barrier=threading.Barrier(2);seen=set();lock=threading.Lock()
        def overlapped(*args):
            with lock:
                first=threading.get_ident() not in seen
                seen.add(threading.get_ident())
            if first:barrier.wait(timeout=3)
            return real(*args)
        try:
            with patch('async_store.os.pread',overlapped):
                store.plan(list(store.records))
                for key,record in store.records.items():
                    with store.read(key) as data:
                        self.assertEqual(hashlib.sha256(data).hexdigest(),record['sha256'])
                self.assertEqual(store.reads,4)
                self.assertEqual(store.async_stats['peak_active_reads'],2)
                self.assertEqual(store.async_stats['peak_pending'],2)
                self.assertLessEqual(store.peak_staging_bytes,2*max(r['bytes'] for r in store.records.values()))
                self.assertEqual(store.staging_bytes,0)
        finally:store.close()
        self.assertIsNone(store.fd)

    def test_unplanned_demand_replaces_only_unleased_work(self):
        store=self.store()
        try:
            store.plan(['0:1','0:2'])
            with store.read('0:1') as first:
                expected=bytes(first)
                with store.read('0:3') as third:
                    self.assertEqual(hashlib.sha256(third).hexdigest(),store.records['0:3']['sha256'])
                    self.assertEqual(bytes(first),expected)
                    with self.assertRaises(RuntimeError):
                        with store.read('0:4'):pass
                    with self.assertRaises(RuntimeError):store.close()
                self.assertEqual(bytes(first),expected)
        finally:store.close()
        self.assertEqual(store.staging_bytes,0)

    def test_exported_view_prevents_recycling_and_can_be_released(self):
        store=self.store()
        with self.assertRaises(BufferError):
            with store.read('0:1') as data:view=memoryview(data)
        self.assertGreater(store.staging_bytes,0)
        view.release()
        store.close()
        self.assertEqual(store.staging_bytes,0)

    def test_hash_failure_and_short_read_never_yield_bytes(self):
        for failure in ('corrupt','eof'):
            store=self.store();real=os.pread
            def broken(*args):
                if failure=='eof':return b''
                data=bytearray(real(*args));data[0]^=1;return bytes(data)
            try:
                with patch('async_store.os.pread',broken):
                    with self.assertRaises((ValueError,OSError)):
                        with store.read('0:1'):self.fail('Unverified bytes were exposed')
                self.assertEqual(store.staging_bytes,0)
                self.assertEqual(store.active_reads,0)
                self.assertEqual(store.async_stats['failed'],1)
            finally:store.close()

    def test_cancel_and_replan_release_unused_records(self):
        store=self.store();cancel=threading.Event();cancel.set()
        try:
            store.plan(['0:1','0:2'])
            with self.assertRaises(InterruptedError):
                with store.read('0:1',cancel=cancel):pass
            store.plan(['0:3','0:4'])
            for key in ('0:3','0:4'):
                with store.read(key) as data:self.assertEqual(hashlib.sha256(data).hexdigest(),store.records[key]['sha256'])
            self.assertEqual(store.staging_bytes,0)
        finally:store.close()

    def test_interrupted_syscall_retries_and_closed_store_rejects_work(self):
        store=self.store();real=os.pread;count=[0]
        def interrupted(*args):
            count[0]+=1
            if count[0]==1:raise InterruptedError()
            return real(*args)
        with patch('async_store.os.pread',interrupted):
            with store.read('0:1') as data:self.assertEqual(hashlib.sha256(data).hexdigest(),store.records['0:1']['sha256'])
        store.close()
        with self.assertRaises(RuntimeError):store.plan(['0:1'])
        with self.assertRaises(RuntimeError):
            with store.read('0:1'):pass

    def test_invalid_capacity_and_identity_are_rejected(self):
        for slots in (0,3,True):
            with self.assertRaises(ValueError):self.store(slots=slots)
        with self.assertRaises(ValueError):AsyncExpertStore(self.bank,'0'*64,direct=False)


if __name__=='__main__':unittest.main()
