"""C1 text-prefill metadata for explicitly approximate decoder tail replay.

Global compressed-KV visibility stays unchanged. Only late-layer local-window
visibility begins at the retained tail boundary. Original metadata is immutable.
"""
from dataclasses import replace
import torch

TAIL=128


def _prefill_only(meta,total):
    if (meta.num_decodes,meta.num_prefills,meta.num_decode_tokens,meta.num_prefill_tokens)!=(0,1,0,total):
        raise ValueError('Decoder tail requires a single unpadded prefill request')


def tail_swa(meta,total):
    _prefill_only(meta,total)
    if total<=TAIL:raise ValueError('Tail transformation requires a larger chunk')
    start=total-TAIL
    if meta.query_start_loc_cpu.tolist()!=[0,total] or meta.slot_mapping.shape!=(total,):
        raise ValueError('Unexpected native SWA token geometry')
    if meta.is_valid_token is None or not bool(meta.is_valid_token.all()):
        raise ValueError('Padded or invalid SWA tokens are not qualified for tail replay')
    for visible in (meta.prefill_left_visible,meta.prefill_right_visible):
        if visible is not None and bool(visible.any()):raise ValueError('Image visibility is not supported by text tail replay')
    old=meta.prefill_swa_indices
    lens=meta.prefill_swa_lens
    if old is None or lens is None or old.shape[0]!=total or lens.shape!=(total,):
        raise ValueError('Missing native paged SWA indices')
    if old.ndim!=3 or old.shape[1]!=1 or old.shape[2]<TAIL:
        raise ValueError('Unexpected native paged SWA index width')
    rows=old[start:,0,:]
    old_lens=lens[start:]
    allowed=torch.arange(1,TAIL+1,dtype=old_lens.dtype,device=old_lens.device)
    if bool((old_lens<allowed).any()):raise ValueError('Native SWA window does not contain the retained tail')
    columns=torch.arange(rows.shape[1],dtype=torch.int64,device=rows.device)[None,:]
    source_columns=(old_lens-allowed).long()[:,None]+columns
    indices=rows.gather(1,source_columns.clamp(max=rows.shape[1]-1))
    indices=torch.where(columns<allowed[:,None],indices,-1).unsqueeze(1).contiguous()
    changes=dict(slot_mapping=meta.slot_mapping[start:].contiguous(),
        query_start_loc=meta.query_start_loc.new_tensor([0,TAIL]),
        query_start_loc_cpu=meta.query_start_loc_cpu.new_tensor([0,TAIL]),
        is_valid_token=meta.is_valid_token[start:].contiguous(),
        token_to_req_indices=meta.token_to_req_indices[start:].contiguous(),
        prefill_swa_indices=indices,prefill_swa_lens=allowed,
        num_prefill_tokens=TAIL,prefill_query_lens_cpu=meta.prefill_query_lens_cpu.new_tensor([TAIL]),
        prefill_gather_lens=meta.prefill_gather_lens.new_tensor([TAIL]),
        flashinfer_sparse_index_cache={})
    for name in ('prefill_left_visible','prefill_right_visible'):
        value=getattr(meta,name)
        if value is not None:changes[name]=value[start:].contiguous()
    return replace(meta,**changes)


def tail_compressed(meta,total):
    if meta.num_reqs!=1 or meta.num_actual_tokens!=total or meta.max_query_len!=total:
        raise ValueError('Unexpected compressed-cache prefill geometry')
    start=total-TAIL
    return replace(meta,num_actual_tokens=TAIL,max_query_len=TAIL,
        query_start_loc=meta.query_start_loc.new_tensor([0,TAIL]),
        slot_mapping=meta.slot_mapping[start:].contiguous(),
        req_id_per_token=meta.req_id_per_token[start:].contiguous())


def tail_indexer(meta,total):
    _prefill_only(meta,total)
    if meta.prefill is None or meta.decode is not None:raise ValueError('Native prefill index metadata required')
    start=total-TAIL;chunks=[];covered=[]
    for chunk in meta.prefill.chunks:
        lo=max(start,chunk.token_start);hi=min(total,chunk.token_end)
        if lo>=hi:continue
        if chunk.num_reqs!=1:raise ValueError('Indexer tail requires C1 chunk metadata')
        offset=lo-chunk.token_start;count=hi-lo
        # Bounds still refer to the original full global KV. The first retained
        # query chunk must gather KV even when earlier discarded chunks would
        # have done so in the original execution.
        chunks.append(replace(chunk,token_start=lo-start,token_end=hi-start,
            cu_seqlen_ks=chunk.cu_seqlen_ks[offset:offset+count].contiguous(),
            cu_seqlen_ke=chunk.cu_seqlen_ke[offset:offset+count].contiguous(),
            skip_kv_gather=bool(chunks)))
        covered.extend(range(lo,hi))
    if covered!=list(range(start,total)):raise ValueError('Indexer chunks do not cover each retained query exactly once')
    return replace(meta,slot_mapping=meta.slot_mapping[start:].contiguous(),num_prefill_tokens=TAIL,
        prefill=replace(meta.prefill,chunks=chunks))
