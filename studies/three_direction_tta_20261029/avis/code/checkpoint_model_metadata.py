"""Parse tensor *metadata only* with PyTorch's restricted weights-only unpickler.
No pickle globals outside the native safe allowlist are enabled. Tensor rebuild
functions are replaced, within this standalone process, by inert metadata builders.
"""
import pathlib,json,io,struct,hashlib,torch
import torch._weights_only_unpickler as safe
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=ROOT/'checkpoints/AVISM_R50_IN.pth';entries=json.loads((ROOT/'downloads/checkpoint_zip.json').read_text());member=next(x for x in entries if x['name']=='archive/data.pkl')
bitmap=json.loads(p.with_suffix('.pth.ranges.json').read_text());assert 0 in bitmap['done']
with p.open('rb') as f:
 f.seek(member['header_offset']);header=f.read(30);h=struct.unpack('<IHHHHHIIIHH',header);assert h[0]==0x04034b50 and h[3]==0
 data_offset=member['header_offset']+30+h[-2]+h[-1];f.seek(data_offset);payload=f.read(member['size'])
assert len(payload)==member['size']
class MetadataUnpickler(safe.Unpickler):
 def persistent_load(self,sid):
  assert isinstance(sid,tuple) and sid[0] in ['storage',b'storage'] and len(sid)>=5,sid
  _,storage_type,key,location,numel,*_=sid
  dtype=str(storage_type.dtype).replace('torch.','')
  assert dtype in ['float32','float64','float16','bfloat16','int64','int32','int16','int8','uint8','bool']
  assert str(key).isdigit() and isinstance(numel,int) and 0<=numel<10**9
  return {'key':str(key),'dtype':dtype,'numel':numel}
def tensor_spec(storage,offset,size,stride,*_):
 assert isinstance(storage,dict) and isinstance(offset,int) and offset>=0
 assert len(size)==len(stride) and all(isinstance(x,int) and x>=0 for x in size+stride)
 return {'__tensor_spec__':True,'storage':storage,'offset':offset,'size':list(size),'stride':list(stride)}
allowed=safe._get_allowed_globals()
for key in ['torch._utils._rebuild_tensor','torch._utils._rebuild_tensor_v2','torch._utils._rebuild_tensor_v3']:
 if key in allowed:allowed[key]=tensor_spec
obj=MetadataUnpickler(io.BytesIO(payload),encoding='utf-8').load()
assert 'model' in obj
model=obj['model'];assert all(isinstance(v,dict) and v.get('__tensor_spec__') for v in model.values())
unique={v['storage']['key']:v['storage'] for v in model.values()};zipentries={x['name']:x for x in entries};needed=set()
for key,s in unique.items():
 e=zipentries['archive/data/'+key]
 # Header padding is at most a few hundred bytes for these released ZIP members.
 first=e['header_offset'];last=first+30+len(e['name'].encode())+256+e['size']
 needed.update(range(first//bitmap['chunk'],(last+bitmap['chunk']-1)//bitmap['chunk']))
out={'metadata_parser':'PyTorch weights-only restricted unpickler, tensor rebuild -> inert metadata','data_pkl_sha256':hashlib.sha256(payload).hexdigest(),'checkpoint_keys':list(obj),'tensor_count':len(model),'storage_count':len(unique),'model':dict(model),'priority_chunks':sorted(needed),'zip_members':{k:zipentries['archive/data/'+k] for k in unique}}
(ROOT/'downloads/model_metadata.json').write_text(json.dumps(out,indent=2));(ROOT/'downloads/checkpoint_priority_chunks.json').write_text(json.dumps(sorted(needed)))
print(json.dumps({k:v for k,v in out.items() if k not in ['model','zip_members','priority_chunks']},indent=2));print('model chunks',len(needed),'complete',sum(i in bitmap['done'] for i in needed))
