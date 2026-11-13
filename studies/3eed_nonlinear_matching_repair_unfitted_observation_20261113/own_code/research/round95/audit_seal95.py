"""Verify fixed method/input pins and every saved prediction/state byte, no GT/model read."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[2];H=R/'research/round95';rd=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();l=rd(H/'experiment/DESIGN_LOCK.json')
for f,s in {**l['programs_sha256'],**l['inputs_sha256']}.items():assert sha(R/f)==s,f
P=R/'crossplatform/round95/prediction336_attempt001';se=rd(P/'SEALED.json')
for f,s in se['files'].items():assert sha(P/f)==s,f
out={'state':'PASS_R95_ALL_LOCKED_INPUT_AND_PREDICTION_STATE_BYTES','pinned_programs':len(l['programs_sha256']),'pinned_inputs':len(l['inputs_sha256']),'saved_files':len(se['files']),'prediction_seal_sha256':sha(P/'SEALED.json'),'GT_access':False,'FINAL_access':False};(H/'SEALED_BYTES_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
