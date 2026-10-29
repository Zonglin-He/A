import requests,sys,concurrent.futures,time,os,json
from pathlib import Path
url,p=sys.argv[1],Path(sys.argv[2]);total=int(sys.argv[3]);start=p.stat().st_size if p.exists() else 0
chunk=8*1024*1024;parts=p.parent/(p.name+'.parts');parts.mkdir(exist_ok=True)
def task(a):
 b=min(a+chunk,total)-1;dest=parts/str(a)
 if dest.exists() and dest.stat().st_size==b-a+1:return a
 for i in range(5):
  try:
   r=requests.get(url,headers={'Range':f'bytes={a}-{b}'},timeout=(30,120));r.raise_for_status()
   if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {a}-{b}/{total}':raise RuntimeError('Range mismatch')
   if len(r.content)!=b-a+1:raise RuntimeError('Bad size')
   dest.write_bytes(r.content);return a
  except Exception:
   if i==4:raise
   time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("DOWNLOAD_WORKERS","6"))) as ex:
 for k,a in enumerate(ex.map(task,range(start,total,chunk))):print(p.name,start+min((k+1)*chunk,total-start),total,flush=True)
with p.open('ab') as f:
 for a in range(start,total,chunk):f.write((parts/str(a)).read_bytes())
print('COMPLETE',p,p.stat().st_size,flush=True)
