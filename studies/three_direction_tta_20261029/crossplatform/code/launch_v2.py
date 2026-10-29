import json,hashlib,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
p=R/'PLAN_LOCK_v6.json';raw=p.read_bytes();lock=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()==(R/'PLAN_LOCK_v6.sha256').read_text().strip(),'Plan hash mismatch'
assert hashlib.sha256((R/'manifest.json').read_bytes()).hexdigest()==lock['manifest_sha256'],'Manifest mismatch'
for name,h in lock['source_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,('Source mismatch',name)
patch=subprocess.check_output(['git','-C',str(R/'upstream'),'diff']);assert hashlib.sha256(patch).hexdigest()==lock['upstream_patch_sha256'],'Upstream patch mismatch'
assert (R/'assets_verified.json').exists(),'Official assets not verified'
print('PLAN AND ASSET PREFLIGHT PASSED',hashlib.sha256(raw).hexdigest(),flush=True)
subprocess.run([sys.executable,str(R/'run_v2.py'),*sys.argv[1:]],check=True)
