"""Explicit image-local bootstrap for the exact staged disk-offload candidate."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import types

_installed=False
_session=None
_package=None
_lock=threading.RLock()

def get_session():
    global _session,_package
    with _lock:
        if _session is not None:return _session
        from expert_store import MODEL,REVISION,digest_file
        from vllm_disk import DiskSession
        from vllm.distributed import get_tp_group
        from vllm.config import get_current_vllm_config
        config=get_current_vllm_config();parallel=config.parallel_config
        if (parallel.tensor_parallel_size,parallel.enable_expert_parallel,parallel.data_parallel_size,
            parallel.pipeline_parallel_size,parallel.enable_eplb)!=(2,True,1,1,False):raise ValueError('Unqualified runtime topology')
        from vllm.config import CUDAGraphMode
        from vllm import envs
        graph_mode=os.environ.get('SAGE_FULL_GRAPH')=='1'
        if os.environ.get('SAGE_DECODER_TAIL','0')!='0':raise ValueError('Decoder tail must be disabled')
        if config.speculative_config is not None:raise ValueError('DSpark remains unqualified')
        if envs.VLLM_USE_BREAKABLE_CUDAGRAPH or parallel.use_ubatching:raise ValueError('Breakable graphs and ubatches are unqualified')
        if graph_mode:
            c=config.compilation_config
            if config.model_config.enforce_eager or c.cudagraph_mode!=CUDAGraphMode.FULL_DECODE_ONLY or c.cudagraph_capture_sizes!=[1] or int(c.mode)!=0:
                raise ValueError('Exact C1 FULL_DECODE_ONLY configuration required')
        elif not config.model_config.enforce_eager:raise ValueError('Explicit eager comparison required')
        if config.scheduler_config.max_num_seqs!=1 or config.scheduler_config.max_num_batched_tokens>512:raise ValueError('Disk candidate requires C1 and bounded prefill')
        if os.environ.get('DSV41_ENGRAM_DISK')!='1':raise ValueError('Engram must remain on local disk')
        spec=json.loads(Path(os.environ['SAGE_DISK_CONFIG']).read_text())
        if spec.get('version')!=1 or spec.get('model')!=MODEL or spec.get('revision')!=REVISION:raise ValueError('Runtime model identity differs')
        rank=get_tp_group().rank_in_group
        root=Path(config.model_config.model).resolve()
        if root!=Path(spec['model_path']).resolve():raise ValueError('Unexpected model package path')
        lock=spec['packages'][rank]
        if digest_file(root/'package.json')!=lock['package_sha256']:raise ValueError('Rank package identity differs')
        package=json.loads((root/'package.json').read_text())
        if (package['model'],package['revision'],package['rank'],package['qualification_only'])!=(MODEL,REVISION,rank,False):raise ValueError('Unqualified or incorrect package')
        for name,metadata in package['files'].items():
            file=(root/name).resolve()
            if not file.is_relative_to(root) or file.stat().st_size!=metadata['bytes']:raise ValueError('Incomplete serving package')
        for name in ('config.json','native-tensors.json','model.safetensors.index.json','experts/manifest.json'):
            if digest_file(root/name)!=package['files'][name]['sha256']:raise ValueError('Package metadata changed')
        capacity=int(spec['cache_bytes'])
        if capacity!=80*2**30:raise ValueError('Only the admitted 80 GiB initial arena is enabled')
        _session=DiskSession(root/'experts',package['files']['experts/manifest.json']['sha256'],rank,capacity)
        _session.package_root=root;_session.native_receipts=json.loads((root/'native-tensors.json').read_text())
        _session.runtime_identity={'model':MODEL,'revision':REVISION,'rank':rank,
            'package_sha256':lock['package_sha256'],'expert_manifest_sha256':package['files']['experts/manifest.json']['sha256']}
        _package=package
        from runtime_telemetry import install as install_telemetry
        install_telemetry(_session)
        atexit.register(close)
        print(json.dumps({'sage_offload':'session_admitted',**_session.runtime_identity,'cache_bytes':capacity,'owned_experts':len(_session.store.records)}),flush=True)
        return _session

def close():
    global _session
    if _session is not None:
        from graph_runtime import close_all
        close_all()
        from engram_prefetch import close_tables
        close_tables()
        if hasattr(_session,'telemetry_emit'):_session.telemetry_emit(force=True)
        _session.close();_session=None

def native_and_catalog(source):
    """Native tensor views plus authorized expert metadata; no Engram/full expert read."""
    from native_stream import stream_native
    session=get_session();root=session.package_root
    if Path(source.model_or_path).resolve()!=root or source.subfolder is not None:raise ValueError('Secondary source is unqualified')
    for name,tensor in stream_native(root/'native.safetensors',session.native_receipts):yield source.prefix+name,tensor
    del tensor
    for name,tensor in session.catalog_weights():yield source.prefix+name,tensor
    # Native CPU tensor views have now been consumed and replaced by catalog
    # markers in the caller. Release their file-cache pages before KV admission.
    fd=os.open(root/'native.safetensors',os.O_RDONLY)
    try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    finally:os.close(fd)

def install():
    global _installed
    if _installed or 'SAGE_DISK_CONFIG' not in os.environ:return
    from vllm_disk import attach_to_config,install_image_compat
    from vllm_exl3.exl3 import Exl3Config
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    install_image_compat()
    from bounded_engram import install as install_bounded_engram
    install_bounded_engram()
    from engram_prefetch import install as install_engram_prefetch
    install_engram_prefetch()
    from graph_runtime import install as install_graph_runtime
    install_graph_runtime()
    from decoder_tail import install as install_decoder_tail
    install_decoder_tail()
    original_quant=Exl3Config.get_quant_method
    def quant(self,layer,prefix):
        if not getattr(self,'_sage_disk_attached',False):
            # Preserve the native delegate as the fallback; avoid a recursive
            # class-hook -> instance-hook -> class-hook chain.
            self.get_quant_method=types.MethodType(original_quant,self)
            attach_to_config(self,get_session())
        return self.get_quant_method(layer,prefix)
    Exl3Config.get_quant_method=quant
    def iterator(self,source):
        if self.counter_before_loading_weights==0.0:self.counter_before_loading_weights=time.perf_counter()
        return native_and_catalog(source)
    DefaultModelLoader._get_weights_iterator=iterator
    original_load=DefaultModelLoader.load_weights
    def load(self,model,model_config):
        original_model_load=model.load_weights
        def tracked(weights):
            loaded=original_model_load(weights)
            if loaded is None:raise ValueError('Model did not report loaded parameters')
            required={name for name,_ in model.named_parameters()}
            if required-set(loaded):raise ValueError(f'Uninitialized model parameters: {sorted(required-set(loaded))}')
            return loaded
        model.load_weights=tracked
        try:return original_load(self,model,model_config)
        finally:model.load_weights=original_model_load
    DefaultModelLoader.load_weights=load
    _installed=True
