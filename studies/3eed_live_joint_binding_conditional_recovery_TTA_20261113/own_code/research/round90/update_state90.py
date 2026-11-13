from pathlib import Path
import json,sys,datetime
H=Path(__file__).resolve().parent;s=json.loads((H/'STATE.json').read_text());s.update(current_stage=sys.argv[1],updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(H/'STATE.json').write_text(json.dumps(s,indent=2)+'\n')
