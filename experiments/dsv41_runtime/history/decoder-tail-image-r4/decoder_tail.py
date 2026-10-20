"""Opt-in eager C1 decoder-tail approximation; full global KV is preserved.

The paired experiment must evaluate this mode against the same larger chunk
with tail disabled. Full prompt hidden states/logprobs and speculation are not
supported in this prototype. Native model source calls slice_decoder_tail at21.
"""
from contextvars import ContextVar
import os
import torch
from decoder_tail_metadata import TAIL,tail_swa,tail_compressed,tail_indexer

_FRAME=ContextVar('sage_decoder_tail_frame',default=None)
_STATS=dict(enabled=False,tail_batches=0,full_chunk_tokens=0,retained_tokens=0,
    discarded_late_layer_token_rows=0,restored_contexts=0)


def snapshot():
    frame=_FRAME.get()
    return {**_STATS,'active_tail_frame':bool(frame is not None and frame.total is not None)}


def validate_configuration(model,context):
    cfg=model._sage_tail_vllm_config;parallel=cfg.parallel_config
    if not cfg.model_config.enforce_eager or cfg.speculative_config is not None:
        raise ValueError('Tail prototype requires eager and speculation disabled')
    # This native version uses num_ubatches=0 for disabled microbatching.
    # use_ubatching is the execution predicate, not the nominal buffer count.
    topology=(parallel.data_parallel_size,parallel.pipeline_parallel_size,
        parallel.decode_context_parallel_size,parallel.prefill_context_parallel_size)
    if topology!=(1,1,1,1) or parallel.use_ubatching or context.ubatch_slices is not None:
        raise ValueError('Tail prototype requires DP1 PP1 with no active microbatch or context splitting')
    if model.use_sequence_parallel or model._mtp_hidden_buffer is not None or model.aux_hidden_state_layers:
        raise ValueError('Tail prototype excludes sequence parallelism and prompt hidden-state capture')
    if (model.start_layer,model.end_layer,len(model.layers))!=(0,40,40):raise ValueError('Unexpected main-layer range')


class TailFrame:
    def __init__(self,model):
        self.model=model;self.context=None;self.original_metadata=None;self.total=None

    def begin(self,positions):
        if self.total is not None:raise RuntimeError('Decoder tail boundary entered twice')
        if positions.shape[0]<=TAIL:return False
        from vllm.forward_context import get_forward_context,is_forward_context_available
        if not is_forward_context_available():return False
        context=get_forward_context()
        if context.attn_metadata is None:return False
        model=self.model
        validate_configuration(model,context)
        total=int(positions.shape[0]);metadata=context.attn_metadata
        if not isinstance(metadata,dict):raise ValueError('Tail prototype requires one metadata dictionary')
        if not TAIL<total<=512:raise ValueError('Tail prototype chunk exceeds admitted 512-token limit')
        selected=dict(metadata);transformed={}
        def narrow(prefix,fn):
            old=metadata[prefix];key=(fn,id(old))
            if key not in transformed:transformed[key]=fn(old,total)
            selected[prefix]=transformed[key]
        for layer in model.layers[21:40]:
            attn=layer.attn
            if attn.is_kv_source or attn.kv_source_layer_id!=20 or attn.compress_ratio!=1:
                raise ValueError('Late layers must consume source20 full global KV')
            if type(attn).__name__!='DeepseekV4FlashInferSM120Attention':raise ValueError('Unqualified tail backend')
            narrow(attn.swa_cache_layer.prefix,tail_swa)
        source=model.layers[20].attn
        narrow(source.prefix,tail_compressed)
        narrow(source.indexer.k_cache.prefix,tail_indexer)
        # The source20 producer used full token coordinates. Every consumer
        # retains the same buffer object/address but reads the retained rows at0.
        for buffer in (model.topk_indices_buffer,model.candidate_block_buffer):
            if buffer is None or buffer.shape[0]<total:raise ValueError('Missing source20 reuse buffer')
            buffer[:TAIL].copy_(buffer[total-TAIL:total].clone())
        self.context=context;self.original_metadata=metadata;self.total=total
        context.attn_metadata=selected
        _STATS['tail_batches']+=1;_STATS['full_chunk_tokens']+=total
        _STATS['retained_tokens']+=TAIL;_STATS['discarded_late_layer_token_rows']+=(total-TAIL)*19
        return True

    def finish(self,result):
        if self.total is None:return result
        if not isinstance(result,torch.Tensor) or result.shape[0]!=TAIL:
            raise ValueError('Tail prototype expects normalized tail states only')
        # vLLM's selected logit positions still use the original batch indices.
        # Discarded prompt rows are intentionally unavailable; this mode is only
        # qualified for last-position generation, never full prompt logprobs.
        output=result.new_zeros((self.total,*result.shape[1:]))
        output[-TAIL:]=result
        return output

    def restore(self):
        if self.context is not None:
            self.context.attn_metadata=self.original_metadata
            _STATS['restored_contexts']+=1


def slice_decoder_tail(model,*values):
    frame=_FRAME.get()
    if frame is None or frame.model is not model:return values
    # values follow the native decoder-layer call exactly:
    # hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual,
    # engram_hashes, engram_mask.
    if not frame.begin(values[1]):return values
    total=frame.total
    result=[]
    for value in values:
        if value is None:result.append(None);continue
        if not isinstance(value,torch.Tensor) or value.shape[0]!=total:
            raise ValueError('Token-dependent state does not match the original chunk')
        result.append(value[-TAIL:].contiguous())
    return tuple(result)


def install():
    from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4Model
    if getattr(DeepseekV4Model,'_sage_tail_installed',False):return
    original_init=DeepseekV4Model.__init__
    def initialize(model,*,vllm_config,prefix=''):
        original_init(model,vllm_config=vllm_config,prefix=prefix)
        # vLLM's configuration context is construction-scoped. Keep the exact
        # admitted object for forward-time guards and the allocation audit.
        model._sage_tail_vllm_config=vllm_config
    DeepseekV4Model.__init__=initialize
    original=DeepseekV4Model.forward
    def forward(model,*args,**kwargs):
        from vllm.forward_context import is_forward_context_available,get_forward_context
        if is_forward_context_available() and isinstance(get_forward_context().attn_metadata,dict):
            from kv_layout_audit import audit
            audit(model)
        enabled=os.environ.get('SAGE_DECODER_TAIL','0')=='1'
        _STATS['enabled']=enabled
        if not enabled:return original(model,*args,**kwargs)
        if _FRAME.get() is not None:raise RuntimeError('Reentrant decoder-tail forward')
        frame=TailFrame(model);token=_FRAME.set(frame)
        try:return frame.finish(original(model,*args,**kwargs))
        finally:frame.restore();_FRAME.reset(token)
    DeepseekV4Model.forward=forward
    DeepseekV4Model._sage_tail_installed=True
