import requests,sys,time,os,json
from pathlib import Path
p=Path(sys.argv[2]); p.parent.mkdir(parents=True,exist_ok=True)
s=p.stat().st_size if p.exists() else 0
r=requests.get(sys.argv[1],headers={"Range":f"bytes={s}-"} if s else {},stream=True,timeout=(30,120));r.raise_for_status()
if s and r.status_code!=206: raise RuntimeError("server did not accept resume")
t=time.time();n=s
with p.open("ab" if s else "wb") as f:
 for c in r.iter_content(4*1024*1024):
  f.write(c);n+=len(c)
  if time.time()-t>15: print(json.dumps({"path":str(p),"bytes":n}),flush=True);t=time.time()
print(json.dumps({"path":str(p),"bytes":n,"completed":True}),flush=True)
