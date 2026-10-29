"""Materialize only model tensors from verified ranges of the official PyTorch ZIP.
Reads no pickle here. Metadata was produced by restricted weights-only unpickling.
Every selected uncompressed ZIP member must pass the official ZIP CRC32 before use.
"""
import pathlib,json,struct,zlib,hashlib,torch,collections,argparse
ROOT=pathlib.Path(__file__).resolve().parents[1];p=ROOT/'checkpoints/AVISM_R50_IN.pth'
spec=json.loads((ROOT/'downloads/model_metadata.json').read_text());bitmap=json.loads(p.with_suffix('.pth.ranges.json').read_text());done=set(bitmap['done']);entries={e['name']:e for e in json.loads((ROOT/'downloads/checkpoint_zip_crc.json').read_text())}
need=spec['priority_chunks'];missing=[i for i in need if i not in done]
if missing:raise RuntimeError(f'{len(missing)} required model chunks still missing of {len(need)}')
state=collections.OrderedDict();checks={}
def verified_read(entry):
 start=entry['header_offset']
 with p.open('rb') as f:
  f.seek(start);h=struct.unpack('<IHHHHHIIIHH',f.read(30));assert h[0]==0x04034b50 and h[3]==0
  begin=start+30+h[-2]+h[-1];end=begin+entry['size']
  assert all(i in done for i in range(begin//bitmap['chunk'],(end+bitmap['chunk']-1)//bitmap['chunk']))
  f.seek(begin);raw=f.read(entry['size'])
 assert len(raw)==entry['size'] and zlib.crc32(raw)&0xffffffff==entry['crc32'],entry['name']
 return raw
# Verify the tensor metadata itself against the independent central directory CRC.
meta_raw=verified_read(entries['archive/data.pkl']);assert hashlib.sha256(meta_raw).hexdigest()==spec['data_pkl_sha256']
storages={}
for name,t in spec['model'].items():
 s=t['storage'];key=s['key'];dtype=getattr(torch,s['dtype'])
 if key not in storages:
  entry=entries['archive/data/'+key];raw=verified_read(entry);itemsize=torch.empty((),dtype=dtype).element_size();assert len(raw)==s['numel']*itemsize
  storages[key]=torch.frombuffer(bytearray(raw),dtype=dtype).clone()
  checks[key]={'member':entry['name'],'crc32':entry['crc32'],'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
 storage=storages[key];bound=t['offset']+sum((size-1)*stride for size,stride in zip(t['size'],t['stride']))+1
 assert bound<=storage.numel()
 state[name]=storage.as_strided(t['size'],t['stride'],t['offset']).clone()
out=ROOT/'checkpoints/AVISM_R50_IN.model_only_derived.pth';torch.save({'model':state},out)
# Safe reload ensures the new artifact needs no non-tensor globals.
loaded=torch.load(out,weights_only=True,map_location='cpu',mmap=True)['model'];assert all(torch.equal(loaded[k],v) for k,v in state.items())
sha=hashlib.file_digest(out.open('rb'),'sha256').hexdigest()
receipt={'state':'completed','derived':True,'source_full_file_complete':(p.with_suffix('.pth.download.json').exists() and json.loads(p.with_suffix('.pth.download.json').read_text()).get('state')=='completed'),'source_public_url':bitmap['url'],'source_expected_bytes':bitmap['total'],'source_data_pkl_sha256':spec['data_pkl_sha256'],'source_zip_directory_sha256':hashlib.sha256((ROOT/'downloads/checkpoint_zip_crc.json').read_bytes()).hexdigest(),'tensor_keys':len(state),'storage_members':checks,'bytes':out.stat().st_size,'sha256':sha,'derivation_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'safety':'source model metadata parsed via restricted weights-only unpickler; no unknown globals; model raw members CRC32 checked; derived artifact weights_only reload verified'}
out.with_suffix('.pth.download.json').write_text(json.dumps(receipt,indent=2));print(out,out.stat().st_size,sha)
