"""Pinned full tensor catalog and expert record layout, without weight downloads."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from expert_store import MODEL,REVISION,ALIGN,SUFFIXES,aligned,validate_tensors,digest_file

WIDTH={'I8':1,'I16':2,'I32':4,'F16':2,'BF16':2,'F32':4,'F8_E4M3':1,'F8_E8M0':1,'U8':1,'I64':8,'F8_E5M2':1}
EXPERT=re.compile(r'layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(trellis|suh|svh|mul1)')

def catalog(root):
    root=Path(root);api=json.loads((root/'model-api.json').read_text())
    if api['sha']!=REVISION or api['id']!=MODEL:raise ValueError('Source identity differs')
    shards={s['rfilename']:s for s in api['siblings'] if s['rfilename'].endswith('.safetensors')}
    tensors={};headers={}
    for file,meta in sorted(shards.items()):
        path=root/'headers'/(file+'.json');d=json.loads(path.read_text());headers[file]=digest_file(path)
        if d['file_size']!=meta['size']:raise ValueError('Shard size disagrees')
        end=0
        for name,spec in sorted(d['header'].items(),key=lambda x:x[1].get('data_offsets',[0])[0]):
            if name=='__metadata__':continue
            if name in tensors:raise ValueError('Duplicate tensor')
            lo,hi=spec['data_offsets'];n=math.prod(spec['shape'])*WIDTH[spec['dtype']]
            if not end<=lo<hi<=meta['size']-8-d['header_length'] or hi-lo!=n:raise ValueError('Invalid tensor extent')
            end=hi
            tensors[name]={**spec,'bytes':n,'file':file,'start':8+d['header_length']+lo}
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    if {n:s['file'] for n,s in tensors.items()}!=index:raise ValueError('Header catalog disagrees with checkpoint index')
    return tensors,shards,headers

def layout(tensors,rank):
    records={};end=0
    for layer in range(40):
        for expert in range(rank*192,(rank+1)*192):
            specs={};n=0
            for p in ('w1','w2','w3'):
                for s in SUFFIXES:
                    name=f'layers.{layer}.ffn.experts.{expert}.{p}.{s}';source=tensors[name]
                    offset=aligned(n,256)
                    specs[f'{p}.{s}']=dict(dtype=source['dtype'],shape=source['shape'],offset=offset,bytes=source['bytes'],sha256='0'*64)
                    n=offset+source['bytes']
            validate_tensors(specs);n=aligned(n)
            records[f'{layer}:{expert}']=dict(offset=end,bytes=n,tensors=specs,sha256='0'*64)
            end+=n
    return records,end

def make_qualification(root,destination):
    """Metadata and sparse holes only; explicit guard prevents model serving."""
    tensors,shards,headers=catalog(root);destination=Path(destination)
    destination.mkdir(exist_ok=False,parents=True)
    for rank in (0,1):
        out=destination/f'rank{rank}';out.mkdir()
        records,size=layout(tensors,rank)
        manifest=dict(version=1,model=MODEL,revision=REVISION,alignment=ALIGN,qualification_only=True,
            scope='metadata-only sparse fixture; all expert payload hashes intentionally invalid',
            source_header_sha256=headers,bank_bytes=size,records=records)
        (out/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n')
        with (out/'experts.bin').open('xb') as f:f.truncate(size)
    return {'tensors':len(tensors),'experts':15360,'qualification_only':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('sources');p.add_argument('destination');a=p.parse_args()
    print(json.dumps(make_qualification(a.sources,a.destination)))
