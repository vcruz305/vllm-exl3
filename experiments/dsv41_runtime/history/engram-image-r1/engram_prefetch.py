"""Eager, one-ubatch native Engram staging with bounded raw-row prefetch.

Only row acquisition moves. Native hash history, ownership, dequantization,
output storage, head collectives, sequence-parallel selection and gating remain.
No CUDA operations are submitted by reader threads.
"""
import atexit
import time
import weakref
import torch
from cached_engram_table import CachedDiskEngramTable

_TABLES=weakref.WeakSet()
_STAGES=weakref.WeakSet()


def file_request(embedding,hash_ids):
    if hash_ids.ndim!=2 or hash_ids.shape[1]!=embedding.n_hash_cols or hash_ids.dtype not in (torch.int32,torch.int64):
        raise ValueError('Invalid native Engram hash geometry')
    t=int(hash_ids.shape[0]);heads=embedding.part_n_hash_cols
    ids=hash_ids.detach().to('cpu',dtype=torch.int64)
    end=min(embedding.head_start+heads,embedding.n_hash_cols)
    local=ids[:,embedding.head_start:end]
    if local.shape[1]<heads:
        local=torch.cat((local,torch.full((t,heads-local.shape[1]),-1,dtype=torch.int64)),dim=1)
    rows=local.reshape(-1)
    owned=(rows>=embedding.vocab_start_idx)&(rows<embedding.vocab_end_idx)
    return torch.where(owned,rows,torch.zeros_like(rows)),owned


class PendingStage:
    def __init__(self):
        self.pending=None
        self.stats=dict(scheduled=0,consumed=0,peak_pending=0,schedule_seconds=0.0,consume_seconds=0.0)
        _STAGES.add(self)

    def prepare(self,engram,hash_ids):
        if self.pending is not None:raise RuntimeError('Unconsumed Engram stage would be overwritten')
        # This prototype is deliberately limited to the qualified eager C1
        # DP1 / PP1 launch. Extra ubatches require independent table futures.
        if engram.embed_tokens.dp_size!=1 or engram._extra_staged_rows:
            raise ValueError('Engram prefetch requires DP1 and one ubatch')
        if hash_ids.device.type=='cuda' and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Engram prefetch is not yet graph qualified')
        started=time.monotonic()
        out=engram._staged_rows_for_ubatch()[:hash_ids.shape[0]]
        if out.shape[0]!=hash_ids.shape[0]:raise ValueError('Engram staging buffer too small')
        rel,owned=file_request(engram.embed_tokens,hash_ids)
        table=engram.embed_tokens.disk
        table.prefetch(rel,owned)
        self.pending=(rel,owned,out,int(hash_ids.shape[0]))
        self.stats['scheduled']+=1;self.stats['peak_pending']=1
        self.stats['schedule_seconds']+=time.monotonic()-started

    def consume(self,engram,hash_ids):
        if self.pending is None:raise RuntimeError('Native Engram embed called without a prepared stage')
        rel,owned,out,count=self.pending
        if hash_ids.shape!=(count,engram.embed_tokens.n_hash_cols):
            raise ValueError('Engram hash shape changed between preparation and consumption')
        started=time.monotonic()
        deq=engram.embed_tokens.disk.gather_dequant(rel,owned)
        out.copy_(deq.view(count,engram.embed_tokens.part_n_hash_cols,engram.embed_tokens.dim))
        self.pending=None
        self.stats['consumed']+=1
        self.stats['consume_seconds']+=time.monotonic()-started


def snapshot():
    tables=[]
    for table in list(_TABLES):
        cache=table.row_cache
        tables.append({'layer':table._sage_layer_id,'closed':cache.closed,
            'capacity_rows':cache.capacity,'cached_rows':len(cache.cache),
            'pending_batches':int(cache.pending is not None),**dict(cache.stats)})
    stages=[{**stage.stats,'pending':int(stage.pending is not None)} for stage in list(_STAGES)]
    return {'tables':sorted(tables,key=lambda row:row['layer']),'stages':stages,
        'scope':'raw-row read overlap; native dequant and H2D at consumption; eager DP1 one ubatch'}


def close_tables():
    errors=[]
    for table in list(_TABLES):
        try:table.close()
        except Exception as error:errors.append(error)
    if errors:raise RuntimeError('Engram cache close failed') from errors[0]


def install():
    import vllm.models.deepseek_v4_1.common.engram_disk as disk_module
    import vllm.models.deepseek_v4_1.common.engram as engram_module
    if getattr(engram_module.Engram,'_sage_row_prefetch',False):return
    original_table=disk_module.DiskEngramTable
    if engram_module.DiskEngramTable is not original_table:
        raise ValueError('Native Engram table aliases differ')
    if not issubclass(CachedDiskEngramTable,original_table):
        raise ValueError('Unexpected native Engram class replacement')

    class TrackedCachedTable(CachedDiskEngramTable):
        def __init__(self,model_dir,layer_id,dim,block_size):
            super().__init__(model_dir,layer_id,dim,block_size)
            self._sage_layer_id=int(layer_id)
            _TABLES.add(self)

    original_prepare=engram_module.Engram.prepare_embeddings
    original_embed=engram_module.Engram.embed

    def prepare(engram,hash_ids):
        if not isinstance(engram.embed_tokens.disk,CachedDiskEngramTable):
            return original_prepare(engram,hash_ids)
        stage=getattr(engram,'_sage_pending_stage',None)
        if stage is None:
            stage=PendingStage();engram._sage_pending_stage=stage
        stage.prepare(engram,hash_ids)

    def embed(engram,hash_ids):
        if isinstance(engram.embed_tokens.disk,CachedDiskEngramTable):
            stage=getattr(engram,'_sage_pending_stage',None)
            if stage is None:raise RuntimeError('Missing native Engram prefetch stage')
            stage.consume(engram,hash_ids)
        return original_embed(engram,hash_ids)

    disk_module.DiskEngramTable=engram_module.DiskEngramTable=TrackedCachedTable
    engram_module.Engram.prepare_embeddings=prepare
    engram_module.Engram.embed=embed
    engram_module.Engram._sage_row_prefetch=True
    atexit.register(close_tables)
