from pathlib import Path
import json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[3];H=Path(__file__).resolve().parent;P=ROOT/'research/round135';B=ROOT/'crossplatform/round135/formal'
def rd(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def check():
 assert not rd(P/'STATE.json')['execution_open']
 for p,h in rd(H/'PLAN_LOCK.json')['source_files'].items():assert sha(ROOT/p)==h,p
