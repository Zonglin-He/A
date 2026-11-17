from pathlib import Path
import json,hashlib,datetime
R136=Path(__file__).resolve().parents[2];H136=R136/'research/round136';B136=R136/'crossplatform/round136'
def rd(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def check():
 for p,h in rd(H136/'experiment/BASELINE_PIN.json')['files'].items():assert sha(R136/p)==h,p
