"""Mixed-K thin experts in one native packed kernel; existing fat GEMM path."""
import torch
import sage_heterogeneous_ext
from prefetch_cache import PrefetchExpertCache
from grouped_plan import packed_routing


class HeterogeneousExpertCache(PrefetchExpertCache):
    LOCK_INTS=1024*1024+2*1024+2+64

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._heterogeneous_locks=None
        self.stats.update(heterogeneous_launches=0,heterogeneous_experts=0,heterogeneous_distinct_k_triplets=0)

    def _routing_groups(self,layer,rows):
        groups=super()._routing_groups(layer,rows)
        thin=[];fat=[]
        for group in groups:
            thin.extend(r for r in group if len(r.positions)<=self.FUSED_ROUTE_THRESHOLD)
            selected=[r for r in group if len(r.positions)>self.FUSED_ROUTE_THRESHOLD]
            if selected:fat.append(selected)
        # Fat groups retain the native shared-K validation and packed GEMM.
        # Thin experts can now carry different original K triplets in one wave.
        return ([thin] if thin else [])+fat

    def _execute(self,batch,entries,x,weights,result,limit,stream):
        counts,token_rows,offsets=packed_routing(batch,tokens=x.shape[0],topk=weights.shape[1])
        if min(counts[:-1])>self.FUSED_ROUTE_THRESHOLD:
            return super()._execute(batch,entries,x,weights,result,limit,stream)
        if not batch or max(counts[:-1])>self.FUSED_ROUTE_THRESHOLD or len(batch)>self.MAX_EXPERTS:
            raise ValueError('Unqualified heterogeneous expert batch')
        pointers=[]
        for projection_index,projection in enumerate(('w1','w3','w2')):
            for item,entry in zip(batch,entries):
                p=entry.projections[projection];k=item.bits[projection_index]
                if k not in range(2,9) or p.K!=k or p.mcg or not p.mul1:
                    raise ValueError('Live mixed-K descriptor differs from original projection')
                expected=(2304,5120) if projection=='w2' else (5120,2304)
                if (p.in_features,p.out_features)!=expected:raise ValueError('Unexpected projection geometry')
            for attribute in ('trellis','suh','svh'):
                pointers.extend(getattr(e.projections[projection],attribute).data_ptr() for e in entries)
        bits=[k for item in batch for k in item.bits]
        metadata=torch.tensor(counts+token_rows+offsets+bits+pointers,dtype=torch.int64,device=self.device)
        n,length=len(entries),len(token_rows)
        cursor=n+1;expert_counts=metadata[:cursor]
        sorted_tokens=metadata[cursor:cursor+length];cursor+=length
        sorted_weights=weights.reshape(-1).index_select(0,metadata[cursor:cursor+length]).half().contiguous();cursor+=length
        descriptors=metadata[cursor:cursor+3*n].view(n,3);cursor+=3*n
        tables=metadata[cursor:].view(9,n)
        temps=self._scratch(max(counts[:-1]),stream)
        if self._heterogeneous_locks is None:
            self._guard()
            self._heterogeneous_locks=torch.zeros(self.LOCK_INTS,dtype=torch.int32,device=self.device)
        self._heterogeneous_locks.record_stream(stream)
        sage_heterogeneous_ext.moe(x,result,expert_counts,sorted_tokens,sorted_weights,*temps,
            descriptors,tables,self._heterogeneous_locks,0.0 if limit is None else float(limit))
        self.stats['grouped_launches']+=1
        self.stats['grouped_routes']+=sum(counts[:-1])
        self.stats['grouped_experts']+=n
        self.stats['grouped_max_batch']=max(self.stats['grouped_max_batch'],n)
        self.stats['heterogeneous_launches']+=1
        self.stats['heterogeneous_experts']+=n
        self.stats['heterogeneous_distinct_k_triplets']+=len({r.bits for r in batch})

    def close(self):
        super().close()
        self._heterogeneous_locks=None
