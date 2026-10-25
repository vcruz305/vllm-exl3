"""Original native Engram hashes and collectives with graph row acquisition."""
import atexit
import os
import weakref
import torch
from graph_rows import GraphRows
_STAGES=weakref.WeakSet()
_TABLES=weakref.WeakSet()
LIBRARY='/opt/sage-offload/libsg_rows.so'


def prepare_model(model, tokens):
    from vllm.models.deepseek_v4_1.common.engram import Engram,NgramHashState
    stages=[];history=[]
    for module in model.modules():
        if isinstance(module,NgramHashState):
            module.ensure_cache()
            history.extend(x for x in (module._cache,module.swa_cache_module.kv_cache) if x is not None)
        if isinstance(module,Engram):
            if module._extra_staged_rows:raise ValueError('Extra ubatches are unqualified')
            stage=getattr(module,'_sage_graph_rows',None)
            if stage is None:
                stage=GraphRows(module.embed_tokens,LIBRARY,module.staged_rows.shape[0])
                module._sage_graph_rows=stage;stage.layer_id=module.embed_tokens.disk._sage_layer_id
                _STAGES.add(stage)
            if stage.stream!=torch.cuda.current_stream(stage.device):stage.transfer_stream()
            stage.prepare_shape(tokens);stages.append(stage);history.append(module.staged_rows)
    if len(stages)!=2:raise ValueError('Expected exactly two original Engram stages')
    return stages,history


def snapshot():
    return {'tables':[{'layer':s.layer_id,**s.snapshot()} for s in list(_STAGES)],
            'scope':'native hash and collectives; graph CPU row service and GPU dequantization'}


def close_tables():
    # Graph owner must run first. Reject freeing any live callback resource.
    for stage in list(_STAGES):stage.close()
    for table in list(_TABLES):table.close()


def install():
    from vllm.models.deepseek_v4_1.common import engram,engram_disk
    if getattr(engram.Engram,'_sage_graph_rows_installed',False):return
    original=engram_disk.DiskEngramTable
    if engram.DiskEngramTable is not original:raise ValueError('Native table aliases differ')
    class TrackedTable(original):
        def __init__(self,model_dir,layer_id,dim,block_size):
            super().__init__(model_dir,layer_id,dim,block_size)
            self._sage_layer_id=int(layer_id);self._sage_closed=False;_TABLES.add(self)
        def close(self):
            if self._sage_closed:return
            self.pool.shutdown(wait=True)
            os.close(self.w_fd);os.close(self.s_fd);self._sage_closed=True
    def prepare(module,hash_ids):
        stage=getattr(module,'_sage_graph_rows',None)
        if stage is None:raise RuntimeError('Prepare graph row storage outside model execution')
        stage.lookup(hash_ids,module._staged_rows_for_ubatch()[:hash_ids.shape[0]])
    engram_disk.DiskEngramTable=engram.DiskEngramTable=TrackedTable
    engram.Engram.prepare_embeddings=prepare
    engram.Engram._sage_graph_rows_installed=True
    atexit.register(close_tables)
