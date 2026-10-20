"""Byte-preserving full checkpoint to rank-owned disk serving packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from concurrent.futures import ThreadPoolExecutor
from catalog_layout import catalog,layout,EXPERT
from expert_store import MODEL,REVISION,ALIGN,digest_file,validate_tensors

CHUNK=4*2**20

def native_tensor_names(tensors):
    # The multimodal wrapper registers these small image-marker Parameters even
    # with vision disabled. Preserve their source bytes for complete loading.
    return [name for name in sorted(tensors)
            if not EXPERT.fullmatch(name)
            and not name.startswith(('mtp.','vision.','aligner.'))
            and '.engram.embed.' not in name]

def digest_dropped(path):
    """Bounded production hashing without retaining hundreds of GB in page cache."""
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        offset=0
        while block:=stream.read(CHUNK):
            h.update(block)
            os.posix_fadvise(stream.fileno(),offset,len(block),os.POSIX_FADV_DONTNEED)
            offset+=len(block)
    return h.hexdigest()

def sync_drop(stream,start=0):
    stream.flush();os.fsync(stream.fileno())
    os.posix_fadvise(stream.fileno(),start,0,os.POSIX_FADV_DONTNEED)

class OriginalReader:
    def __init__(self,root,tensors):
        self.root=Path(root);self.tensors=tensors;self.fds={}
    def chunks(self,name):
        spec=self.tensors[name]
        if spec['file'] not in self.fds:
            self.fds[spec['file']]=os.open(self.root/spec['file'],os.O_RDONLY)
        fd=self.fds[spec['file']];offset=spec['start'];left=spec['bytes']
        while left:
            block=os.pread(fd,min(left,CHUNK),offset)
            if not block:raise OSError('Original checkpoint truncated')
            yield block
            os.posix_fadvise(fd,offset,len(block),os.POSIX_FADV_DONTNEED)
            left-=len(block);offset+=len(block)
    def close(self):
        for fd in self.fds.values():os.close(fd)

def expert_record(reader,layer,expert,record):
    data=bytearray(record['bytes'])
    for name,spec in record['tensors'].items():
        original=f'layers.{layer}.ffn.experts.{expert}.{name}'
        offset=spec['offset'];h=hashlib.sha256()
        for block in reader.chunks(original):
            data[offset:offset+len(block)]=block;offset+=len(block);h.update(block)
        if name.endswith('.mul1') and struct.unpack_from('<i',data,spec['offset'])[0]!=-2082680531:
            raise ValueError('Original MUL1 marker invalid')
        spec['sha256']=h.hexdigest()
    validate_tensors(record['tensors'])
    record['sha256']=hashlib.sha256(data).hexdigest()
    return data

def write_native(reader,names,path):
    header={};offset=0;receipts={}
    for name in names:
        spec=reader.tensors[name];end=offset+spec['bytes']
        header[name]={'dtype':spec['dtype'],'shape':spec['shape'],'data_offsets':[offset,end]};offset=end
    encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
    with path.open('xb') as out:
        out.write(struct.pack('<Q',len(encoded)));out.write(encoded);last=out.tell()
        for name in names:
            h=hashlib.sha256()
            for block in reader.chunks(name):
                out.write(block);h.update(block)
                if out.tell()-last>=64*2**20:sync_drop(out,last);last=out.tell()
            receipts[name]={'sha256':h.hexdigest(),'bytes':reader.tensors[name]['bytes']}
        sync_drop(out)
    if path.stat().st_size!=8+len(encoded)+offset:raise ValueError('Native file extent differs')
    return receipts

def write_experts(reader,tensors,shards,headers,destination):
    expected={name:s['lfs']['sha256'] for name,s in shards.items()}
    for rank in (0,1):
        directory=Path(destination)/f'rank{rank}'/'experts';directory.mkdir(parents=True)
        records,size=layout(tensors,rank)
        with (directory/'experts.bin').open('xb') as out:
            last=0
            for key,record in records.items():
                layer,expert=map(int,key.split(':'))
                if out.tell()!=record['offset']:raise ValueError('Expert layout drift')
                out.write(expert_record(reader,layer,expert,record))
                if out.tell()-last>=64*2**20:sync_drop(out,last);last=out.tell()
                if expert%192==191:print(json.dumps({'rank':rank,'built_layer':layer,'bytes':out.tell()}),flush=True)
            sync_drop(out)
        manifest=dict(version=1,model=MODEL,revision=REVISION,alignment=ALIGN,bank_bytes=size,
            source_shards_sha256=expected,source_header_sha256=headers,records=records)
        (directory/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n')

def build(original,sources,destination,prepared=None):
    original=Path(original);destination=Path(destination)
    if destination.exists():raise FileExistsError('Do not overwrite a serving package')
    tensors,shards,headers=catalog(sources)
    completed=json.loads((original/'download-complete.json').read_text())
    if (completed['model'],completed['revision'])!=(MODEL,REVISION):raise ValueError('Unverified source staging')
    expected={s['rfilename']:s['lfs']['sha256'] for s in shards.values()}
    receipts={s['file']:s for s in completed['files']}
    if not all(receipts[n]['sha256']==h for n,h in expected.items()):raise ValueError('Download receipts differ from pinned originals')
    def verify_original(item):
        n,h=item
        if digest_dropped(original/n)!=h:raise ValueError(f'Original source changed: {n}')
        return n
    with ThreadPoolExecutor(3) as pool:
        for n in pool.map(verify_original,expected.items()):print(json.dumps({'verified_original':n}),flush=True)
    # Hard links share original Engram shards and native text across head packages.
    # The worker receives only its package through an ordinary verified copy.
    if shutil.disk_usage(original).free<(20 if prepared else 260)*10**9:raise RuntimeError('Derived-bank disk allowance unavailable')
    temporary=Path(tempfile.mkdtemp(prefix='.building-packages-',dir=destination.parent))
    (temporary/'BUILDING.json').write_text(json.dumps({'model':MODEL,'revision':REVISION}))
    reader=OriginalReader(original,tensors)
    try:
        rank_dirs=[temporary/f'rank{r}' for r in (0,1)]
        if prepared:
            prepared=Path(prepared);admission=json.loads((prepared/'prepared.json').read_text())
            if (admission['model'],admission['revision'])!=(MODEL,REVISION):raise ValueError('Prepared expert source differs')
            expert_sources={s['file'] for n,s in tensors.items() if EXPERT.fullmatch(n)}
            if admission['source_shards_sha256']!={n:expected[n] for n in expert_sources}:raise ValueError('Prepared expert shard identity differs')
            for rank,directory in enumerate(rank_dirs):
                (directory/'experts').mkdir(parents=True)
                for name in ('experts.bin','manifest.json'):
                    source=prepared/f'rank{rank}'/'experts'/name
                    if digest_dropped(source)!=admission['files'][f'rank{rank}/experts/{name}']['sha256']:raise ValueError('Prepared expert data changed')
                    os.link(source,directory/'experts'/name)
        else:write_experts(reader,tensors,shards,headers,temporary)
        native_names=native_tensor_names(tensors);engram_names=[]
        for name in sorted(tensors):
            if EXPERT.fullmatch(name) or name.startswith(('mtp.','vision.','aligner.')):continue
            if '.engram.embed.' in name:engram_names.append(name)
        native=rank_dirs[0]/'native.safetensors';native_receipts=write_native(reader,native_names,native)
        os.link(native,rank_dirs[1]/native.name)
        for directory in rank_dirs:
            weight_map={name:native.name for name in native_names}
            for name in engram_names:
                source=tensors[name]['file'];target=directory/source
                if not target.exists():os.link(original/source,target)
                weight_map[name]=source
            # Preserve source config separately; overlay only delegate metadata.
            for source in original.iterdir():
                if (source.suffix in ('.json','.jinja','.model','.md') or source.name=='LICENSE') and not source.name.startswith(('download-','model.safetensors.index')):
                    shutil.copy2(source,directory/source.name)
            shutil.copy2(original/'config.json',directory/'config.original.json')
            config=json.loads((directory/'config.json').read_text())
            config['quantization_config']={**config['quantization_config'],
                'non_routed_quantization':{'quant_method':'deepseek_v4_fp8','weight_block_size':[32,32],'activation_scheme':'dynamic'},
                'mtp_experts':'source'}
            (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
            (directory/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':sum(tensors[n]['bytes'] for n in weight_map)},'weight_map':weight_map},indent=2)+'\n')
            (directory/'native-tensors.json').write_text(json.dumps(native_receipts,sort_keys=True)+'\n')
        for rank,directory in enumerate(rank_dirs):
            # Read back every derived file; original Engram hashes were checked above.
            files={}
            for path in sorted(directory.rglob('*')):
                if not path.is_file():continue
                name=str(path.relative_to(directory))
                sha=expected[path.name] if path.name in expected else digest_dropped(path)
                files[name]={'bytes':path.stat().st_size,'sha256':sha}
            payload=sum(s['bytes'] for s in files.values())
            if payload>=330*10**9:raise RuntimeError('Rank package exceeds admitted storage budget')
            package=dict(version=1,model=MODEL,revision=REVISION,rank=rank,expert_ids=[rank*192,(rank+1)*192-1],
                layers=40,expert_count=7680,files=files,bytes=payload,source_shards_sha256=expected,
                native_tensor_count=len(native_names),expert_tensor_count=92160,qualification_only=False)
            (directory/'package.json').write_text(json.dumps(package,indent=2)+'\n')
        (temporary/'BUILDING.json').unlink()
        os.rename(temporary,destination)
        return {'packages':[str(destination/f'rank{r}') for r in (0,1)],'passed':True}
    finally:reader.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('original');p.add_argument('sources');p.add_argument('destination');p.add_argument('--prepared');a=p.parse_args()
    print(json.dumps(build(a.original,a.sources,a.destination,a.prepared)),flush=True)
