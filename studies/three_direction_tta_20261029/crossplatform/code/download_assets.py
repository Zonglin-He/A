import requests,concurrent.futures,json,hashlib,zlib,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parent
INDEX=json.loads((ROOT/'downloads/dataset_zip_index.json').read_text())
selected=[]
for platform in ['waymo','drone','quad']:
 val=set((ROOT/f'upstream/data/splits/{platform}_val.txt').read_text().splitlines())
 names=[a for a in INDEX if a['name'].startswith(f'3eed_dataset/{platform}/') and a['name'].endswith('/meta_info.json') and a['name'].split('/')[2] in val]
 names.sort(key=lambda a:hashlib.sha256(a['name'].encode()).hexdigest())
 # one frame per sequence to avoid temporal pseudoreplication, max 4.
 chosen=[];scenes=set()
 for a in names:
  scene=a['name'].split('/')[2]
  if scene in scenes:continue
  chosen.append(a);scenes.add(scene)
  if len(chosen)==4:break
 for a in chosen:
  prefix=a['name'].rsplit('/',1)[0]+'/'
  selected.extend(x for x in INDEX if x['name'].startswith(prefix) and x['name'].endswith(('meta_info.json','lidar.bin','lidar.npy')))
(ROOT/'downloads/subset_names.json').write_text(json.dumps([a['name'] for a in selected],indent=2))
print('Selected',len(selected),'files',sum(a['compressed'] for a in selected),flush=True)
def fetch(a):
 n=a['name'];dest=ROOT/'assets'/n
 if dest.exists() and dest.stat().st_size==a['size']:return n
 url='https://huggingface.co/datasets/RRRong/3EED/resolve/main/3eed_dataset.zip'
 start=a['offset'];end=start+30+len(n.encode())+65536+a['compressed']
 for attempt in range(4):
  try:
   r=requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=120);r.raise_for_status()
   if r.status_code!=206:raise RuntimeError('No range')
   buf=r.content
   if buf[:4]!=b'PK\x03\x04':raise RuntimeError('Bad ZIP header')
   nl,el=struct.unpack_from('<HH',buf,26);offset=30+nl+el
   b=buf[offset:offset+a['compressed']];b=zlib.decompress(b,-15) if a['type']==8 else b
   if len(b)!=a['size']:raise RuntimeError('Wrong uncompressed size')
   dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(b)
   return n
  except Exception:
   if attempt==3:raise
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
 for n in ex.map(fetch,selected):print(n,flush=True)
