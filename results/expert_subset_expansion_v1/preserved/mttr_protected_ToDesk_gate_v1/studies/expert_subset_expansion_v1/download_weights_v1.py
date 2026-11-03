"""Download official public checkpoint once; preserve partial file if transfer fails."""
from pathlib import Path
import urllib.request,json,hashlib,time,os
D=Path(__file__).resolve().parents[2]/'outputs/expert_subset_expansion_v1'
def download(name,url):
 dest=D/'vendor'/name;assert not dest.exists();part=dest.with_suffix(dest.suffix+'.partial');assert not part.exists();t=time.perf_counter();h=hashlib.sha256();n=0
 with urllib.request.urlopen(url,timeout=180)as response,part.open('wb')as f:
  length=response.headers.get('Content-Length');ctype=response.headers.get('Content-Type');assert 'text/html'not in(ctype or ''),ctype
  while True:
   b=response.read(4*1024**2)
   if not b:break
   f.write(b);h.update(b);n+=len(b)
  f.flush();os.fsync(f.fileno())
 assert n>10000000 and(length is None or n==int(length));os.replace(part,dest)
 receipt=dict(status='official_checkpoint_downloaded_length_SHA',url=url,bytes=n,sha256=h.hexdigest(),elapsed_wall_seconds=time.perf_counter()-t,new_model=0,new_GPU=0,new_Source=0,new_expert=0)
 (D/(name+'.download.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':download('referformer_r101_refytvos.pth','https://drive.usercontent.google.com/download?id=1QJU8HC8xk_3gMcj-Yi4vkUHaeHeP8n82&export=download&confirm=t')
