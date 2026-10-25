"""Observe real native cache ownership, bindings and allocation geometry once."""
import dataclasses
_REPORT=None


def tensor_info(tensor):
    return {'pointer':tensor.data_ptr(),'shape':list(tensor.shape),'stride':list(tensor.stride()),
        'dtype':str(tensor.dtype),'view_content_bytes':tensor.numel()*tensor.element_size(),
        'storage_pointer':tensor.untyped_storage().data_ptr(),'storage_bytes':tensor.untyped_storage().nbytes()}


def slot_regions(info,slots,element_bytes,owner):
    """Actual byte intervals, accounting for group-overlay pool/block strides."""
    shape=info['shape'];stride=info['stride']
    if len(shape)!=3 or stride[2]!=1 or stride[1]<shape[2] or stride[0]<shape[1]*stride[1]:
        raise ValueError('Unqualified native [block, token, channel] cache geometry: '+repr(info))
    regions=[]
    for slot in sorted(set(int(s) for s in slots if s>=0)):
        block,token=divmod(slot,shape[1])
        if block>=shape[0]:raise ValueError('KV slot exceeds allocated pool')
        address=info['pointer']+(block*stride[0]+token*stride[1])*element_bytes
        end=address+shape[2]*element_bytes
        if address<info['storage_pointer'] or end>info['storage_pointer']+info['storage_bytes']:
            raise ValueError('KV write exceeds backing storage')
        regions.append((address,end,owner))
    if not regions:raise ValueError('No real writable slots for '+owner)
    return regions


def require_disjoint(regions):
    ordered=sorted(regions)
    for previous,current in zip(ordered,ordered[1:]):
        if current[0]<previous[1]:
            raise ValueError('Logical KV owners overlap active bytes: '+repr((previous,current)))
    return len(ordered)


def snapshot():return _REPORT


def audit(model, metadata=None):
    global _REPORT
    if _REPORT is not None:return _REPORT
    cfg=model._sage_tail_vllm_config;hf=cfg.model_config.hf_config
    if tuple(hf.kv_source_layer_ids)!=(2,8,14,20):raise ValueError('Unexpected compressed-KV source topology')
    if tuple(hf.index_source_layer_ids)!=(2,8,14,20,24,28,32,36):raise ValueError('Unexpected indexer topology')
    if len(model.layers)!=40:raise ValueError('Expected exact 40-layer main model')
    from vllm.forward_context import get_forward_context
    if metadata is None:metadata=get_forward_context().attn_metadata
    rows=[];owners={'main':set(),'index':set(),'window':set()};regions=[];storages={}
    def owner(kind,module,cache):
        if module.prefix in owners[kind]:raise ValueError('Duplicate logical cache owner')
        if cfg.compilation_config.static_forward_context[module.prefix] is not module:
            raise ValueError('Cache owner is not the native registered module')
        owners[kind].add(module.prefix)
        info=tensor_info(cache);storages[info['storage_pointer']]=info['storage_bytes']
        slots=metadata[module.prefix].slot_mapping.tolist()
        regions.extend(slot_regions(info,slots,cache.element_size(),module.prefix))
        return {'prefix':module.prefix,'active_write_slots':len(set(s for s in slots if s>=0)),**info}
    for number,layer in enumerate(model.layers):
        attn=layer.attn
        if type(attn).__name__!='DeepseekV4FlashInferSM120Attention':raise ValueError('Unqualified attention backend')
        swa=attn.swa_cache_layer.kv_cache
        if swa.numel()==0:return None
        entry={'layer':number,'backend':type(attn).__name__,'kv_cache_dtype':attn.kv_cache_dtype,
            'compression_ratio':attn.compress_ratio,'swa':owner('window',attn.swa_cache_layer,swa)}
        if attn.compress_ratio:
            source=model.layers[attn.kv_source_layer_id].attn
            cache=attn._compressed_kv_cache()
            if tensor_info(cache)!=tensor_info(source.kv_cache) or attn.compressed_cache_prefix!=source.prefix:
                raise ValueError('Compressed-KV consumer is not bound to its native source')
            entry.update(kv_source=attn.kv_source_layer_id,compressed=tensor_info(cache))
            if attn.is_kv_source:
                entry['compressed']=owner('main',attn,cache)
                spec=attn.get_kv_cache_spec(cfg)
                entry['compressed_spec']={**dataclasses.asdict(spec),'page_size_bytes':spec.page_size_bytes}
        if attn.indexer is not None:
            source=model.layers[attn.kv_source_layer_id].attn.indexer
            if attn.indexer.k_cache is not source.k_cache:raise ValueError('Indexer key cache is not shared with its KV source')
            cache=attn.indexer.k_cache.kv_cache
            if cache.numel()==0:return None
            entry['indexer']={'owns_k':attn.indexer.owns_k,'fp4':attn.indexer.use_fp4_kv,
                'kv_source':attn.kv_source_layer_id,**tensor_info(cache)}
            if attn.indexer.owns_k:
                entry['indexer'].update(owner('index',attn.indexer.k_cache,cache))
        rows.append(entry)
    if tuple(len(owners[k]) for k in ('main','index','window'))!=(4,4,40):
        raise ValueError('Native logical cache ownership count differs')
    checked=require_disjoint(regions)
    # Avoid exposing pointers in normal result summaries; raw telemetry remains
    # private and binds the actual storage equality checks above.
    _REPORT={'passed':True,'main_cache_owners':4,'index_cache_owners':4,'window_cache_owners':40,
        'physical_storage_allocations':len(storages),'physical_storage_bytes':sum(storages.values()),
        'active_write_regions_disjoint':True,'active_write_regions_checked':checked,
        'layers':rows,'scope':'Once-per-worker native binding and active-slot byte-isolation audit. '
        'Groups may overlay a physical pool with distinct allocated blocks; not a whole-run or 1M request proof.'}
    import json
    _REPORT=json.loads(json.dumps(_REPORT,default=str))
    return _REPORT
