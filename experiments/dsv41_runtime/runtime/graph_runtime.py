"""External C1 full-graph preparation, pointer guards and shared ownership."""
import dataclasses
import functools
import threading
import torch
from graph_lifetime import GraphLease
_LOCAL=threading.local()
_OWNERS={}
_STATS={'prepared':0,'captured':0,'replayed':0,'released':0}


def tensor_signature(value):
    signature={};retained=[];seen=set()
    def visit(x,path,depth):
        if isinstance(x,torch.Tensor):
            if x.device.type=='cuda':
                signature[path]=(x.data_ptr(),str(x.dtype),str(x.device),tuple(x.shape),tuple(x.stride()))
                retained.append(x)
            return
        if depth>12:raise ValueError('Unexpected graph metadata nesting')
        if id(x) in seen:return
        seen.add(id(x))
        if isinstance(x,dict):
            for key,item in sorted(x.items(),key=lambda kv:str(kv[0])):visit(item,path+'/'+str(key),depth+1)
        elif isinstance(x,(list,tuple)):
            for i,item in enumerate(x):visit(item,path+'/'+str(i),depth+1)
        elif dataclasses.is_dataclass(x):
            for f in dataclasses.fields(x):visit(getattr(x,f.name),path+'/'+f.name,depth+1)
        elif hasattr(x,'__dict__') and not isinstance(x,torch.nn.Module):
            for key,item in sorted(vars(x).items()):visit(item,path+'/'+key,depth+1)
    visit(value,'root',0)
    return signature,retained


def prepare(model,args,kwargs,context):
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('Graph preparation entered capture')
    from vllm.utils.torch_utils import current_stream
    if current_stream()!=torch.cuda.current_stream():raise RuntimeError('vLLM and CUDA stream ownership differ')
    ids=kwargs.get('input_ids',args[0] if args else None)
    if ids is None or ids.ndim!=1 or not 0<ids.shape[0]<=512:raise ValueError('Unqualified native token input')
    from sage_plugin import get_session
    cache=get_session().get_cache();cache.prepare_forward(ids.shape[0])
    from engram_prefetch import prepare_model
    if not getattr(_LOCAL,'dummy',False) and isinstance(context.attn_metadata,dict):
        from kv_layout_audit import audit
        language=getattr(model,'language_model',model)
        audit(language.model, metadata=context.attn_metadata)
    stages,history=prepare_model(model,ids.shape[0])
    signature,retained=tensor_signature({'args':args,'kwargs':kwargs,'attention':context.attn_metadata,'slots':context.slot_mapping,'padding':context.is_padding,'history':history})
    _STATS['prepared']+=1
    return dict(cache=cache,stages=stages,signature=signature,retained=[model,*retained],tokens=ids.shape[0])


def context_for(metadata,slots,padding):
    from types import SimpleNamespace
    return SimpleNamespace(attn_metadata=metadata,slot_mapping=slots,is_padding=padding)


def prepare_capture(manager,desc,model,inputs,metadata,slots,padding):
    state=prepare(model,(),inputs,context_for(metadata,slots,padding))
    if state['tokens']!=1 or desc.num_tokens!=1:raise ValueError('Only C1 graph storage is qualified')
    if not hasattr(manager,'_sage_states'):manager._sage_states={}
    manager._sage_states[desc]=state


def lease_current_graph(manager,graph,desc):
    state=manager._sage_states[desc]
    owner=_OWNERS.setdefault(id(manager),{'manager':manager,'graphs':{}})
    lease=GraphLease(graph,state['cache'],state['stages'],state['retained'])
    owner['graphs'][id(graph)]={'lease':lease,'signature':state['signature']}
    lease.acquire();_STATS['captured']+=1


def prepare_replay(manager,desc,model,inputs,metadata,slots,padding):
    state=prepare(model,(),inputs,context_for(metadata,slots,padding))
    graph=manager.graphs[desc]
    record=_OWNERS[id(manager)]['graphs'][id(graph)]
    if record['signature']!=state['signature']:
        changed=[k for k in set(record['signature'])|set(state['signature']) if record['signature'].get(k)!=state['signature'].get(k)]
        raise RuntimeError('Captured storage changed at: '+repr(sorted(changed)))
    manager._sage_replay_ready=desc


def clear(manager):
    owner=_OWNERS.get(id(manager))
    if owner is not None:
        for record in owner['graphs'].values():
            record['lease'].close();_STATS['released']+=1
        del _OWNERS[id(manager)]
    manager.graphs.clear()
    if hasattr(manager,'_sage_states'):manager._sage_states.clear()
    manager.hidden_states=None;manager.aux_hidden_states=[];manager.intermediate_tensors=None


def close_all():
    for owner in list(_OWNERS.values()):clear(owner['manager'])


def snapshot():return dict(_STATS,live_graphs=sum(len(o['graphs']) for o in _OWNERS.values()),runner='V2')


def install():
    import os,time
    from vllm.v1.worker.gpu import cudagraph_utils as cg
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.worker_base import CompilationTimes
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context,is_forward_context_available
    from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV41LLMForCausalLM
    if getattr(cg.CudaGraphManager,'_sage_graph_owner',False):return
    capture=cg.ModelCudaGraphManager.capture;replay=cg.CudaGraphManager.run_fullgraph
    teardown=cg._teardown_profiling_state;shutdown=GPUModelRunner.shutdown
    dummy_run=GPUModelRunner._dummy_run;forward=DeepseekV41LLMForCausalLM.forward
    @functools.wraps(capture)
    def captured(manager,*args,**kwargs):
        previous=getattr(_LOCAL,'dummy',False);_LOCAL.dummy=True
        previous_capture=getattr(_LOCAL,'capturing_manager',None);_LOCAL.capturing_manager=manager
        try:return capture(manager,*args,**kwargs)
        except BaseException:clear(manager);raise
        finally:_LOCAL.dummy=previous;_LOCAL.capturing_manager=previous_capture
    @functools.wraps(replay)
    def replayed(manager,desc):
        if getattr(manager,'_sage_replay_ready',None)!=desc:raise RuntimeError('External V2 replay preparation missing')
        manager._sage_replay_ready=None
        result=replay(manager,desc);_STATS['replayed']+=1
        from sage_plugin import get_session
        get_session().telemetry_emit()
        return result
    @functools.wraps(forward)
    def eager(model,*args,**kwargs):
        if not torch.cuda.is_current_stream_capturing() and getattr(_LOCAL,'capturing_manager',None) is None and is_forward_context_available():
            prepare(model,args,kwargs,get_forward_context())
        return forward(model,*args,**kwargs)
    @functools.wraps(dummy_run)
    def dummy(runner,*args,**kwargs):
        previous=getattr(_LOCAL,'dummy',False);_LOCAL.dummy=True
        try:return dummy_run(runner,*args,**kwargs)
        finally:_LOCAL.dummy=previous
    @functools.wraps(teardown)
    def release_profile(runner):
        if runner.cudagraph_manager is not None:clear(runner.cudagraph_manager)
        return teardown(runner)
    @functools.wraps(shutdown)
    def stop(runner):
        close_all()
        from sage_plugin import close
        close()
        return shutdown(runner)
    def warmup_and_capture(worker):
        # The inherited startup shim replaced this whole method with a no-op.
        # Restore only the qualified native C1 warmup/capture, not its unrelated
        # mixed-attention autotuning path that this SM121 image disabled.
        if not worker.use_v2_model_runner:raise ValueError('Pinned V2 runner required')
        start=time.monotonic()
        from null_kv_page import initialize_null_kv
        print({'sage_null_kv_before_capture':initialize_null_kv(worker.model_runner)},flush=True)
        if os.environ.get('SAGE_FULL_GRAPH')=='1':
            manager=worker.model_runner.cudagraph_manager
            if manager.cudagraph_mode!=CUDAGraphMode.FULL_DECODE_ONLY or manager.use_breakable_cg:
                raise ValueError('Graph mode was silently downgraded')
            descs=manager._capture_descs
            if len(descs.get(CUDAGraphMode.FULL,[]))!=1 or CUDAGraphMode.PIECEWISE in descs:
                raise ValueError('Only one full C1 capture is admitted')
            worker.model_runner.capture_model()
            print({'sage_null_kv_after_capture':initialize_null_kv(worker.model_runner)},flush=True)
            if len(manager.graphs)!=1:raise RuntimeError('Native full capture did not occur')
        from vllm.utils.torch_utils import set_random_seed
        set_random_seed(worker.model_config.seed)
        return CompilationTimes(0.0,time.monotonic()-start)
    cg.ModelCudaGraphManager.capture=captured;cg.CudaGraphManager.run_fullgraph=replayed
    cg._teardown_profiling_state=release_profile;GPUModelRunner.shutdown=stop
    GPUModelRunner._dummy_run=dummy;DeepseekV41LLMForCausalLM.forward=eager
    Worker.compile_or_warm_up_model=warmup_and_capture
    cg.CudaGraphManager._sage_graph_owner=True
