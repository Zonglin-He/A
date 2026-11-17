from pathlib import Path
import json,hashlib,datetime,sys
R=Path(__file__).resolve().parents[2];H=R/'research/round143';B=R/'crossplatform/round143'
def rd(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def check():
 for p,h in rd(H/'experiment/DEPENDENCIES_LOCK.json')['files'].items():assert sha(R/p)==h,p
 if (H/'experiment/CODE_LOCK.json').exists():
  for p,h in rd(H/'experiment/CODE_LOCK.json')['files'].items():assert sha(R/p)==h,p
