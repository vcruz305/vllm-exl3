"""Initialize the native reserved null KV block during startup only.

Pinned BlockPool removes logical block 0 from its free list. Sparse MLA may
read that block for masked indices, so its packed FP8 bytes must stay finite.
Native KVBlockZeroer handles overlaid views, strides and virtual block splits.
"""
import torch


def initialize_null_kv(runner):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Null KV initialization must be outside graph capture')
    if runner.kv_cache_config.num_blocks <= 1:
        raise ValueError('Native cache requires a reserved null block and data blocks')
    with torch.inference_mode():
        if runner.kv_block_zeroer is None:
            runner._init_kv_zero_meta()
        zeroer = runner.kv_block_zeroer
        if zeroer is None or zeroer._meta is None:
            raise RuntimeError('Native null KV zeroing has no attention segments')
        zeroer.zero_block_ids([0])
        torch.cuda.synchronize()
    return {'block_id': 0, 'segments': zeroer._meta[-1]}
