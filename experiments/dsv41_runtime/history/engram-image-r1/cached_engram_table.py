"""Native Engram dequantization over bounded cached original row bytes."""
import os
import torch
from vllm.models.deepseek_v4_1.common.engram_disk import DiskEngramTable
from engram_rows import EngramRows


class CachedDiskEngramTable(DiskEngramTable):
    def __init__(self,*args,capacity_rows=131072,**kwargs):
        super().__init__(*args,**kwargs)
        self._closed=False
        try:
            self.row_cache=EngramRows(self.w_fd,self.w_off,self.s_fd,self.s_off,
                min(self.w_shape[0],self.s_shape[0]),self.dim,self.sb,
                capacity_rows=capacity_rows,workers=min(self.threads,4))
        except BaseException:
            self.pool.shutdown(wait=True)
            os.close(self.w_fd);os.close(self.s_fd)
            self._closed=True
            raise

    def _validate_request(self,rel,owned):
        if self._closed:raise RuntimeError('Engram table is closed')
        if rel.ndim!=1 or owned.shape!=rel.shape or rel.device.type!='cpu' or owned.device.type!='cpu':
            raise ValueError('Invalid Engram row request')
        if rel.dtype!=torch.int64 or owned.dtype!=torch.bool or rel.numel()>8192:
            raise ValueError('Engram request exceeds qualified shape or dtype')
        if rel.numel() and (int(rel.min())<0 or int(rel.max())>=min(self.w_shape[0],self.s_shape[0])):
            raise ValueError('Engram row outside the original table')

    def prefetch(self,rel,owned):
        self._validate_request(rel,owned)
        self.row_cache.prefetch(rel.tolist(),owned.tolist())

    def gather_dequant(self,rel,owned):
        self._validate_request(rel,owned)
        r=int(rel.numel())
        if r==0:
            self.row_cache.gather_into([],[],memoryview(bytearray()),memoryview(bytearray()))
            return torch.empty((0,self.dim),dtype=torch.bfloat16)
        w=torch.empty((r,self.dim),dtype=torch.uint8)
        s=torch.empty((r,self.sb),dtype=torch.uint8)
        self.row_cache.gather_into(rel.tolist(),owned.tolist(),
            memoryview(w.numpy()).cast('B'),memoryview(s.numpy()).cast('B'))
        # Preserve the exact installed native conversion, including rounding
        # order and the final ownership mask. Only row acquisition changed.
        vals=w.view(torch.float8_e4m3fn).to(torch.float32).view(r,self.sb,-1)
        scale=(s.to(torch.int32)<<23).view(torch.float32)
        out=(vals*scale[:,:,None]).reshape(r,self.dim)
        out[~owned]=0
        return out.to(torch.bfloat16)

    def close(self):
        if self._closed:return
        try:self.row_cache.close()
        finally:
            self.pool.shutdown(wait=True)
            os.close(self.w_fd);os.close(self.s_fd)
            self._closed=True
