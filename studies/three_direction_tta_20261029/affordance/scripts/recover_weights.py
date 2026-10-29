"""Recover source model tensors by archive member size+CRC grouping.
No checkpoint globals execute; storage/tensor descriptions are inert.
All retrieved raw members must match archive CRC and byte size. A receipt records
aliases so this derived weights-only artifact is distinguishable from original.
"""
import pathlib,pickle,io,struct,collections,json,math,hashlib,argparse,concurrent.futures,threading,zlib,binascii,time
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1];CHUNK=2*1024*1024;TOTAL=1208734804
URL='https://drive.usercontent.google.com/download?id=1yOV4KdRVWXRjXlXbYZ4_H9peoZIIKc0Y&export=download&confirm=t'
PARTS=ROOT/'downloads/checkpoints/best_unseen_aff.pt.parts2m'
class Tensor:
 def __init__(self,storage,offset,size,stride,*args):self.shape=tuple(size);self.storage=storage;self.offset=offset;self.stride=tuple(stride)
class Reader(pickle.Unpickler):
 def find_class(self,module,name):
  if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
  if (module,name) in [('torch._utils','_rebuild_tensor_v2'),('torch._utils','_rebuild_tensor')]:return Tensor
  if module=='torch' and name.endswith('Storage'):return name
  raise pickle.UnpicklingError(f'Forbidden global {module}.{name}')
 def persistent_load(self,pid):return {'type':pid[1],'key':pid[2],'device':pid[3],'count':pid[4]}
def description():
 raw=(PARTS/'0').read_bytes();h=struct.unpack('<4s5H3I2H',raw[:30]);start=30+h[-2]+h[-1];end=raw.find(b'PK\x07\x08',start)
 return Reader(io.BytesIO(raw[start:end])).load()['model']
def spans(start,end):return set(range(start//CHUNK*CHUNK,min(TOTAL,end)//CHUNK*CHUNK+1,CHUNK))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 sd=description();idx=json.loads((ROOT/'checkpoint_zip_index.json').read_text());prefix=next(iter(idx)).split('/')[0];groups=collections.defaultdict(list)
 for name,info in idx.items():
  if '/data/' in name:groups[(info['size'],info['crc'])].append(name)
 have={int(p.name) for p in PARTS.iterdir() if p.name.isdigit()};need=set();aliases={};chosen={}
 for key in sorted({v.storage['key'] for v in sd.values()},key=lambda k:-idx[f'{prefix}/data/{k}']['size']):
  name=f'{prefix}/data/{key}';info=idx[name];matches=groups[(info['size'],info['crc'])]
  def required(n):
   i=idx[n];return spans(i['offset'],i['offset']+256+i['compressed'])
  representative=min(matches,key=lambda n:(len(required(n)-have-need),n))
  aliases[name]=representative;chosen[representative]=idx[representative];need|=required(representative)-have
 receipt={'source_index_sha256':hashlib.sha256((ROOT/'checkpoint_zip_index.json').read_bytes()).hexdigest(),'source_metadata_first_chunk_sha256':hashlib.sha256((PARTS/'0').read_bytes()).hexdigest(),'source_size':TOTAL,'model_tensor_entries':len(sd),'unique_representatives':len(chosen),'missing_chunks':len(need),'missing_bytes_upper_bound':len(need)*CHUNK,'aliases':aliases,'basis':'source ZIP member size+CRC; all selected member contents checked before reconstruction','state':'prepared'}
 (ROOT/'weights_recovery_plan.json').write_text(json.dumps(receipt,indent=2));print('remaining unique model chunks',len(need),'bytes',len(need)*CHUNK,flush=True)
 if not a.execute:return
 local=threading.local()
 def fetch(start):
  p=PARTS/str(start);end=min(TOTAL,start+CHUNK)-1
  if p.exists() and p.stat().st_size==end-start+1:return
  if not hasattr(local,'session'):local.session=requests.Session()
  for attempt in range(5):
   try:
    r=local.session.get(URL,headers={'Range':f'bytes={start}-{end}'},timeout=(20,40));r.raise_for_status()
    assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{TOTAL}' and len(r.content)==end-start+1
    tmp=p.with_suffix('.recover_tmp');tmp.write_bytes(r.content);tmp.replace(p);return
   except Exception:
    if attempt==4:raise
    time.sleep(2**attempt)
 with concurrent.futures.ThreadPoolExecutor(24) as ex:
  for i,f in enumerate(concurrent.futures.as_completed([ex.submit(fetch,n) for n in sorted(need)])):
   f.result()
   if i%8==0:print('recovery chunks',i+1,'/',len(need),flush=True)
 def read_at(start,size):
  result=[];pos=start
  while pos<start+size:
   off=pos//CHUNK*CHUNK;raw=(PARTS/str(off)).read_bytes();lo=pos-off;piece=raw[lo:min(len(raw),lo+start+size-pos)];assert piece;result.append(piece);pos+=len(piece)
  return b''.join(result)
 import torch
 torch.set_num_threads(4);storages={};member_hashes={}
 for name,info in chosen.items():
  header=read_at(info['offset'],30);h=struct.unpack('<4s5H3I2H',header);assert h[0]==b'PK\x03\x04'
  filename=read_at(info['offset']+30,h[-2]).decode();assert filename==name
  dataoffset=info['offset']+30+h[-2]+h[-1];raw=read_at(dataoffset,info['compressed']);raw=zlib.decompress(raw,-15) if info['method']==8 else raw
  assert len(raw)==info['size'] and binascii.crc32(raw)&0xffffffff==info['crc'],name
  storages[name]=bytearray(raw);member_hashes[name]=hashlib.sha256(raw).hexdigest()
 dtypes={'FloatStorage':torch.float32,'LongStorage':torch.int64,'DoubleStorage':torch.float64,'IntStorage':torch.int32,'HalfStorage':torch.float16,'ByteStorage':torch.uint8,'BoolStorage':torch.bool}
 result=collections.OrderedDict()
 for name,v in sd.items():
  raw=storages[aliases[f'{prefix}/data/{v.storage["key"]}']];base=torch.frombuffer(raw,dtype=dtypes[v.storage['type']]);assert base.numel()==v.storage['count'];result[name]=base.as_strided(v.shape,v.stride,v.offset).clone();assert torch.isfinite(result[name]).all()
 path=ROOT/'downloads/checkpoints/best_unseen_aff.model_only_crc_recovered.pt';torch.save({'model':result},path)
 receipt.update(state='recovered',derived_path=str(path),derived_sha256=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest(),member_sha256=member_hashes,all_selected_member_crc_valid=True);(ROOT/'weights_recovery_receipt.json').write_text(json.dumps(receipt,indent=2));print('recovered',path,flush=True)
if __name__=='__main__':main()
