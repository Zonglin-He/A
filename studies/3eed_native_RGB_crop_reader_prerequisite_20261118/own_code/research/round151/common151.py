from pathlib import Path
import json,hashlib,datetime,sys
R=Path(__file__).resolve().parents[2];H=R/'research/round151';B=R/'crossplatform/round151'
def rd(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def check():
 for name in ['DEPENDENCIES_LOCK','CROP_CODE_LOCK','FROZEN_CODE_LOCK','SOURCE_CODE_LOCK','PREPARATION_CODE_LOCK','TTA_CODE_LOCK']:
  p=H/'experiment'/(name+'.json')
  if p.exists():
   for f,h in rd(p)['files'].items():assert sha(R/f)==h,f
