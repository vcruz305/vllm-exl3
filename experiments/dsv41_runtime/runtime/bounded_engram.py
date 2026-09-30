"""Bound existing Engram requests and release this reader's clean file pages."""
import os

def install():
    from vllm.models.deepseek_v4_1.common.engram_disk import DiskEngramTable
    if getattr(DiskEngramTable, '_sage_bounded', False): return
    original_read = DiskEngramTable._read_rows
    original_gather = DiskEngramTable.gather_dequant

    def read(table, fd, base, rel, row_bytes, buf):
        try: return original_read(table, fd, base, rel, row_bytes, buf)
        finally:
            # All bounded row tasks have completed before the normal return.
            # No global cache flushing: only clean pages of this table inode.
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def gather(table, rel, owned):
        import torch
        if rel.ndim != 1 or owned.shape != rel.shape or rel.device.type != 'cpu' or owned.device.type != 'cpu':
            raise ValueError('Invalid Engram row request')
        if rel.dtype != torch.int64 or owned.dtype != torch.bool or rel.numel() > 8192:
            raise ValueError('Engram request exceeds qualified shape or dtype')
        if rel.numel() == 0: return torch.empty((0, table.dim), dtype=torch.bfloat16)
        if int(rel.min()) < 0 or int(rel.max()) >= min(table.w_shape[0], table.s_shape[0]):
            raise ValueError('Engram row outside the original table')
        return original_gather(table, rel, owned)

    DiskEngramTable._read_rows = read
    DiskEngramTable.gather_dequant = gather
    DiskEngramTable._sage_bounded = True
