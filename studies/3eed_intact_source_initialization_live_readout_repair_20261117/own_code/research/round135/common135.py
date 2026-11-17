from pathlib import Path
import json,hashlib,datetime
R135=Path(__file__).resolve().parents[2];H135=R135/'research/round135';B135=R135/'crossplatform/round135'
def rd(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def basecheck():
 for p,h in rd(H135/'experiment/BASELINE_PIN.json')['files'].items():assert sha(R135/p)==h,p
