"""Read-back verification for one complete rank package; no CUDA allocation."""
import argparse
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from build_packages import digest_dropped
from expert_store import MODEL, REVISION, ExpertStore, digest_file

def verify(root,rank):
    root=Path(root).resolve();started=time.monotonic()
    package=json.loads((root/'package.json').read_text())
    assert (package['version'],package['model'],package['revision'],package['rank'],package['qualification_only'])==(1,MODEL,REVISION,rank,False)
    assert package['bytes']<330*10**9 and package['expert_count']==7680 and package['expert_tensor_count']==92160
    assert package['native_tensor_count']==1257
    native=json.loads((root/'native-tensors.json').read_text())
    assert len(native)==1257 and {'image_start','image_end','image_newline'}<=native.keys()
    files=package['files'];verified={}
    def verify_file(item):
        name,spec=item
        path=(root/name).resolve()
        assert path.is_relative_to(root) and path.is_file() and path.stat().st_size==spec['bytes']
        sha=digest_dropped(path)
        assert sha==spec['sha256'],name
        return name,{'sha256':sha,'bytes':spec['bytes']}
    with ThreadPoolExecutor(3) as pool:
        for name,spec in pool.map(verify_file,files.items()):
            verified[name]=spec
            print(json.dumps({'verified_package_file':name,'bytes':spec['bytes']}),flush=True)
    assert sum(r['bytes'] for r in verified.values())==package['bytes']
    store=ExpertStore(root/'experts',files['experts/manifest.json']['sha256'],direct=True)
    try:
        expected={f'{layer}:{expert}' for layer in range(40) for expert in range(rank*192,(rank+1)*192)}
        assert set(store.records)==expected
        assert not store.manifest.get('qualification_only')
        assert store.manifest['source_shards_sha256']==package['source_shards_sha256']
    finally:store.close()
    original=json.loads((root/'config.original.json').read_text())
    overlay=json.loads((root/'config.json').read_text())
    quant=overlay.pop('quantization_config');old=original.pop('quantization_config')
    assert overlay==original
    assert quant=={**old,'non_routed_quantization':{'quant_method':'deepseek_v4_fp8','weight_block_size':[32,32],'activation_scheme':'dynamic'},'mtp_experts':'source'}
    return {'passed':True,'model':MODEL,'revision':REVISION,'rank':rank,'package_sha256':digest_file(root/'package.json'),
        'verified_files':verified,'bytes':package['bytes'],'seconds':time.monotonic()-started}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--rank',type=int,choices=(0,1),required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    out=Path(a.receipt)
    if out.exists():raise FileExistsError(out)
    result=verify(a.directory,a.rank);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
