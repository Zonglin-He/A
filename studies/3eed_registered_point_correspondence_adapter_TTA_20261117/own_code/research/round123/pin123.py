from pathlib import Path
import json,hashlib,datetime
H=Path(__file__).resolve().parent;R=H.parents[1];files=[H/x for x in ['audit123.py','score123.py','audit_scores123.py']];p=H/'experiment/POSTPROCESS_CODE_LOCK.json';assert not p.exists();p.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before_task_scores':True,'files':{str(x.relative_to(R)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files}},indent=2)+'\n')
