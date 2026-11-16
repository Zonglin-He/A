"""Administrative base64 transport of independently audited public bytes only."""
from pathlib import Path
import json,sys,base64
R=Path(__file__).resolve().parents[2];idx=json.loads((R/'research/round109/publication/UPLOAD_INDEX_PRIVATE.json').read_text());i,offset=map(int,sys.argv[1:]);f=idx['files'][i];p=R/f['local_path'];assert p.is_relative_to(R/'research/round109/publication/public_export_attempt002');b=p.read_bytes();chunk=b[offset:offset+24576];print(json.dumps({'n':len(chunk),'base64':base64.b64encode(chunk).decode()}))
