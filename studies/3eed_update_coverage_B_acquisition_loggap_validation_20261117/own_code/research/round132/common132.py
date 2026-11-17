from pathlib import Path
import json,hashlib,datetime
R132=Path(__file__).resolve().parents[2];H132=R132/'research/round132';B132=R132/'crossplatform/round132'
def rd(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def basecheck():
 for p,h in rd(H132/'experiment/BASELINE_PIN.json')['files'].items():assert sha(R132/p)==h,p
