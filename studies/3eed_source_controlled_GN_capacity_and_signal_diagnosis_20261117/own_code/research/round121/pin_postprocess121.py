from pathlib import Path
import json,hashlib,datetime
H=Path(__file__).resolve().parent;R=H.parents[1];files=[H/n for n in ['audit121.py','score121.py','audit_scores121.py']];o=H/'experiment/POSTPROCESS_CODE_LOCK.json';assert not o.exists();o.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_formal_metric_results_read':True,'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
