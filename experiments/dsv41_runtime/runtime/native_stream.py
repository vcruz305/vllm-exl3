"""Typed native tensor views with bounded hashing and explicit mapped-page release."""
import hashlib
import json
import math
import mmap
import os
import struct
import torch

DTYPES={'BF16':torch.bfloat16,'F16':torch.float16,'F32':torch.float32,'F8_E4M3':torch.float8_e4m3fn,
        'F8_E8M0':torch.float8_e8m0fnu,'I64':torch.int64,'I32':torch.int32,'I16':torch.int16,'U8':torch.uint8,'I8':torch.int8}
CHUNK=4*2**20

def stream_native(path,receipts):
    with open(path,'rb') as file:
        size=os.fstat(file.fileno()).st_size
        n=struct.unpack('<Q',file.read(8))[0]
        if not 2<=n<=32*2**20:raise ValueError('Invalid native header size')
        header=json.loads(file.read(n));base=8+n
        if set(header)!=set(receipts):raise ValueError('Native tensor catalog differs')
        mapping=mmap.mmap(file.fileno(),0,access=mmap.ACCESS_COPY)

        def release(start,length):
            lo=start//4096*4096;hi=min(size,((start+length+4095)//4096)*4096)
            mapping.madvise(mmap.MADV_DONTNEED,lo,hi-lo)
            os.posix_fadvise(file.fileno(),lo,hi-lo,os.POSIX_FADV_DONTNEED)

        # A tensor may briefly remain referenced by the consumer between yields.
        # MADV_DONTNEED releases clean pages while keeping its address valid; any
        # later access reloads the original bytes. Never close a live tensor view.
        try:
            end=0
            for name,spec in sorted(header.items(),key=lambda item:item[1]['data_offsets'][0]):
                lo,hi=spec['data_offsets'];dtype=DTYPES[spec['dtype']]
                length=math.prod(spec['shape'])*torch.empty((),dtype=dtype).element_size()
                if lo!=end or hi-lo!=length or base+hi>size or length!=receipts[name]['bytes']:raise ValueError('Invalid native tensor extent')
                start=base+lo;h=hashlib.sha256()
                for offset in range(start,start+length,CHUNK):
                    stop=min(offset+CHUNK,start+length)
                    view=memoryview(mapping)[offset:stop];h.update(view);view.release()
                    release(offset,stop-offset)
                if h.hexdigest()!=receipts[name]['sha256']:raise ValueError('Native tensor differs: '+name)
                tensor=torch.frombuffer(mapping,dtype=dtype,count=math.prod(spec['shape']),offset=start).reshape(spec['shape'])
                yield name,tensor
                release(start,length);del tensor
                end=hi
            if base+end!=size:raise ValueError('Trailing native payload')
        finally:
            release(0,size)
            # torch.frombuffer retains the mapping until its last tensor is gone.
            del mapping
