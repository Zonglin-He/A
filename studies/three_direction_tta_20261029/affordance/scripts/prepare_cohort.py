"""Deterministic metadata-only cohort selection, remote extraction, isolated xyz/GT exports."""
import argparse,pathlib,json,hashlib,zipfile,sys,io,concurrent.futures,struct,zlib,binascii
import numpy as np
from remote_zip import RemoteFile,safe_write

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);p.add_argument('--limit',type=int,default=12);a=p.parse_args();root=a.root
 url='https://drive.usercontent.google.com/download?id=1rkWAZcIEfj5Rk7E3EOlh0gwDr4Hg14pp&export=download&confirm=t'
 z=zipfile.ZipFile(RemoteFile(url));names=z.namelist()
 infos={n:{'offset':z.getinfo(n).header_offset,'compressed':z.getinfo(n).compress_size,'size':z.getinfo(n).file_size,'crc':z.getinfo(n).CRC,'method':z.getinfo(n).compress_type} for n in names}
 (root/'data/archive_index.json').write_text(json.dumps(infos))
 folder=root/'data/Unseen_aff';read=lambda n:(folder/n).read_text().splitlines()
 images,points,human,obj=[read(n) for n in ['Img_test.txt','Point_test.txt','hk_test.txt','ok_test.txt']]
 assert len(images)==len(points)==len(human)==len(obj)
 # Hash ranks independent physical point paths; avoid repeated point cloud parents.
 chosen=[];seen=set()
 for i in sorted(range(len(images)),key=lambda i:hashlib.sha256((images[i]+'|'+points[i]).encode()).hexdigest()):
  if points[i] in seen:continue
  chosen.append(i);seen.add(points[i])
  if len(chosen)==a.limit:break
 (root/'data/selected_indices.json').write_text(json.dumps(chosen))
 def resolve(v):
  options=[v.replace('\\','/').lstrip('./')]
  for n in names:
   if any(v.endswith('/'+n) or n.endswith('/'+v) for v in options):return n
  suffix='/'.join(options[0].split('/')[-5:])
  matches=[n for n in names if n.endswith(suffix)]
  if len(matches)!=1:
   suffix='/'.join(options[0].split('/')[-4:]);matches=[n for n in names if n.endswith(suffix)]
  if len(matches)!=1:raise ValueError((v,matches[:5]))
  return matches[0]
 required=sorted({resolve(v) for i in chosen for v in [images[i],points[i]]})
 # Separate reader per worker: zipfile seeking is stateful.
 def get(n):
  output=root/'data'/n
  if output.exists():return
  info=infos[n];r=RemoteFile(url);r.seek(info['offset']);header=r.read(30);h=struct.unpack('<4s5H3I2H',header);assert h[0]==b'PK\x03\x04';r.seek(h[-2]+h[-1],1);compressed=r.read(info['compressed'])
  raw=zlib.decompress(compressed,-15) if info['method']==8 else compressed
  assert len(raw)==info['size'] and binascii.crc32(raw)&0xffffffff==info['crc'];safe_write(root/'data',n,raw);print(n,flush=True)
 with concurrent.futures.ThreadPoolExecutor(4) as ex:list(ex.map(get,required))
 inference=[];scoring=[];(root/'data/geometry').mkdir(exist_ok=True);(root/'data/targets').mkdir(exist_ok=True)
 for j,i in enumerate(chosen):
  ip,pp=resolve(images[i]),resolve(points[i]);raw=np.load(root/'data'/pp,allow_pickle=False);assert raw.shape==(2048,4),raw.shape
  geo=root/f'data/geometry/{j:04d}.npy';gt=root/f'data/targets/{j:04d}.npy';np.save(geo,raw[:,:3]);np.save(gt,raw[:,3:])
  identity=hashlib.sha256((images[i]+'|'+points[i]).encode()).hexdigest()
  inference.append({'id':identity,'image':str(root/'data'/ip),'geometry':str(geo),'human':human[i],'object':obj[i]})
  scoring.append({'id':identity,'target':str(gt),'object_class':pp.split('/')[-4],'affordance':ip.split('/')[-2]})
 (root/'data/inference_manifest.json').write_text(json.dumps(inference,indent=2));(root/'data/scoring_manifest.json').write_text(json.dumps(scoring,indent=2));print('cohort ready',len(inference),flush=True)
if __name__=='__main__':main()
