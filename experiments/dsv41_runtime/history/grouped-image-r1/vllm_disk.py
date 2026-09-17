"""Opt-in vLLM EXL3 disk-expert method; native V4.1 owns the model graph."""
import hashlib
import math
import re
import types
import torch
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm_exl3.exl3 import _exl3_routed_experts_loader
from deepseek_v41_compat import is_deepseek_v41_source_quant,install_deepseek_v41_compat,source_weight_block_size
from expert_store import ALIGN, ExpertStore
from grouped_cache import GroupedExpertCache as ExpertCache


class DiskSession:
    def __init__(self, directory, manifest_hash, rank, capacity_bytes, *, qualification_keys=None):
        if rank not in (0,1):raise ValueError('Only contiguous EP2 is qualified for this candidate')
        self.rank=rank
        self.store=ExpertStore(directory,manifest_hash,direct=True)
        if self.store.manifest.get('qualification_only') and qualification_keys is None:
            self.store.close();raise ValueError('Qualification-only bank cannot serve a model')
        full={f'{l}:{e}' for l in range(40) for e in range(rank*192,(rank+1)*192)}
        expected=full if qualification_keys is None else set(qualification_keys)
        if not expected or not expected<=full or set(self.store.records)!=expected:
            self.store.close();raise ValueError('Owned expert catalog is incomplete or includes another rank')
        self.qualification_only=qualification_keys is not None
        self.token=object()
        self.cache=None
        self.capacity_bytes=capacity_bytes
        payload={}
        for key,r in self.store.records.items():
            layer=int(key.split(':')[0]);payload[layer]=payload.get(layer,0)+r['bytes']
        total=sum(payload.values())
        quotas={l:(capacity_bytes*b//total)//ALIGN*ALIGN for l,b in payload.items()}
        quotas[max(quotas)]+=capacity_bytes-sum(quotas.values())
        self.quotas=quotas

    def get_cache(self):
        if self.cache is None:
            self.cache=ExpertCache(self.store,self.capacity_bytes,layer_quotas=self.quotas)
        return self.cache

    def catalog_weights(self):
        """Authorized metadata sent through the real loader, without reading payloads.

        The session-only token distinguishes these from missing checkpoint data.
        Every owned tensor still crosses name/shape/ownership/completeness gates.
        """
        dtype={'I16':torch.int16,'I32':torch.int32,'F16':torch.float16}
        for key,r in sorted(self.store.records.items()):
            layer,expert=key.split(':')
            for name,spec in sorted(r['tensors'].items()):
                tensor=torch.empty(0,dtype=dtype[spec['dtype']],device='cpu')
                tensor._sage_catalog=(self.token,key,name)
                yield f'layers.{layer}.ffn.experts.{expert}.{name}',tensor

    def close(self):
        if self.cache is not None:self.cache.close()
        self.store.close()


class DiskExl3MoEMethod(FusedMoEMethodBase):
    def __init__(self,moe,session,layer_index):
        super().__init__(moe)
        self.session=session;self.layer_index=layer_index
        self.seen=set();self.finalized=False

    def maybe_roundup_sizes(self,hidden_size,intermediate_size_per_partition,act_dtype,moe_parallel_config):
        if (hidden_size,intermediate_size_per_partition)!=(5120,2304):
            raise ValueError('Disk EXL3 requires the pinned V4.1 dimensions')
        if (moe_parallel_config.tp_size,moe_parallel_config.ep_size,moe_parallel_config.use_ep)!=(1,2,True):
            raise ValueError('Disk EXL3 requires whole experts under EP2')
        if moe_parallel_config.enable_eplb:raise ValueError('Dynamic expert placement is unsupported')
        return hidden_size,intermediate_size_per_partition

    def get_fused_moe_quant_config(self,layer):return None

    def create_weights(self,layer,num_experts,hidden_size,intermediate_size_per_partition,params_dtype,**extra):
        if num_experts!=192 or (hidden_size,intermediate_size_per_partition)!=(5120,2304):
            raise ValueError('Unexpected expert dimensions/count')
        if self.layer_index not in self.session.quotas:raise ValueError('Layer missing from expert catalog')
        actual={e for e in range(384) if layer._map_global_expert_id_to_local_expert_id(e)>=0}
        expected=set(range(self.session.rank*192,(self.session.rank+1)*192))
        if actual!=expected:raise ValueError('Runtime expert ownership disagrees with disk package')
        for group in ('w13','w2'):
            for suffix,dtype in (('trellis',torch.int16),('suh',torch.float16),('svh',torch.float16),('mul1',torch.int32)):
                name=f'{group}_{suffix}'
                param=torch.nn.Parameter(torch.empty(0,dtype=dtype),requires_grad=False)
                param.weight_loader=self._load
                param._sage_slot=(group,suffix)
                layer.register_parameter(name,param)
        layer.load_weights=_exl3_routed_experts_loader(layer)
        # Create only the bounded arena. No full expert-sized Parameters exist.
        self.session.get_cache()

    def _load(self,param,loaded_weight,weight_name,shard_id='w1',expert_id=0,return_success=False):
        if self.finalized:raise RuntimeError('Reload into finalized disk method is unsupported')
        if not 0<=expert_id<384:raise ValueError('Invalid global expert ID')
        if expert_id//192!=self.session.rank:return False if return_success else None
        key=f'{self.layer_index}:{expert_id}'
        if key not in self.session.store.records:raise ValueError('Missing owned disk expert')
        group,suffix=param._sage_slot
        if shard_id not in ('w1','w2','w3') or group!=('w2' if shard_id=='w2' else 'w13'):
            raise ValueError('Projection mapping mismatch')
        if not weight_name.endswith(f'{group}_{suffix}'):raise ValueError('Parameter mapping mismatch')
        name=f'{shard_id}.{suffix}';identity=(key,name)
        if identity in self.seen:raise ValueError('Duplicate owned expert tensor')
        spec=self.session.store.records[key]['tensors'][name]
        token=getattr(loaded_weight,'_sage_catalog',None)
        if token is not None:
            if token!=(self.session.token,key,name) or loaded_weight.numel()!=0:
                raise ValueError('Invalid catalog authorization')
        else:
            dtype={'I16':torch.int16,'I32':torch.int32,'F16':torch.float16}[spec['dtype']]
            if loaded_weight.dtype!=dtype or list(loaded_weight.shape)!=spec['shape']:
                raise ValueError('Loaded tensor does not match the expert catalog')
            data=loaded_weight.detach().cpu().contiguous().numpy().tobytes()
            if hashlib.sha256(data).hexdigest()!=spec['sha256']:
                raise ValueError('Source tensor differs from expert bank')
        self.seen.add(identity)
        return True if return_success else None

    def process_weights_after_loading(self,layer):
        expected={(key,name) for key,r in self.session.store.records.items()
                  if int(key.split(':')[0])==self.layer_index for name in r['tensors']}
        if self.seen!=expected:raise ValueError(f'Incomplete disk expert load: {len(self.seen)}/{len(expected)}')
        self.finalized=True

    def apply(self,layer,x,topk_weights,topk_ids,shared_experts,shared_experts_input):
        if not self.finalized:raise RuntimeError('Disk experts have not completed loading')
        limit=self.moe.swiglu_limit
        if limit is not None and (not math.isfinite(limit) or limit<=0):
            raise ValueError('Invalid SwiGLU clipping limit')
        return self.session.get_cache().apply(self.layer_index,x,topk_ids,topk_weights,limit=limit).to(x.dtype)


def install_image_compat():
    """Install the pinned helper with the retained image's metadata setter."""
    # Retained image has the pinned exl3.py but lacks this recipe helper.
    # The helper is copied unchanged from the locked plugin revision.
    import vllm_exl3.exl3 as exl3_module
    # This retained image's sitecustomize copies the declared block size onto
    # each new config. Accept that assignment only if it agrees with metadata;
    # the upstream read-only property otherwise breaks repeated construction.
    if 'weight_block_size' not in exl3_module.Exl3Config.__dict__:
        def validate_block_size(self,value):
            if list(value)!=source_weight_block_size(self):
                raise ValueError('Image block-size shim disagrees with source metadata')
        exl3_module.Exl3Config.weight_block_size=property(source_weight_block_size,validate_block_size)
    install_deepseek_v41_compat(exl3_module)


def attach_to_config(config,session):
    """Explicit instance-level opt-in; all other quantization dispatch is kept."""
    install_image_compat()
    if not is_deepseek_v41_source_quant(config):raise ValueError('Native V4.1 FP8 metadata missing')
    if getattr(config,'_sage_disk_attached',False):raise ValueError('Config already has a disk session')
    original=config.get_quant_method
    def dispatch(self,layer,prefix):
        if isinstance(layer,RoutedExperts) and layer.global_num_experts==384:
            match=re.search(r'(?:^|\.)layers\.(\d+)\.ffn\.experts(?:\.routed_experts)?$',prefix)
            if not match:raise ValueError('Unexpected V4.1 expert prefix')
            return DiskExl3MoEMethod(layer.moe_config,session,int(match.group(1)))
        return original(layer,prefix)
    config.get_quant_method=types.MethodType(dispatch,config)
    config._sage_disk_attached=True
    return config
