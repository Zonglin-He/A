"""Pin completed public evidence and own export code after scalar audit."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round98';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert rd(H/'SCALAR_AUDIT.json')['state'].startswith('PASS_')
for f,s in rd(H/'experiment/CODE_LOCK.json')['code_sha256'].items():assert sha(R/f)==s,f
files=set(H.glob('*.py'));files.update(H/x for x in ['REPORT.md','PROTOCOL.md','ITERATION_RECORD.md','DECISION.json','NATIVE_AUDIT.json','SCALAR_AUDIT.json','LEGACY_R95_PARITY_AUDIT.json','FIT_SUMMARY_ROWS.json','experiment/DESIGN_LOCK.json','experiment/CODE_LOCK.json']);files.update((H/'scores_attempt001').glob('*.json'));files.update(R/'crossplatform/round98'/d/'SEALED.json' for d in rd(R/'crossplatform/round98/MATRIX_SEALED.json')['directions'])
p=H/'experiment/PUBLIC_EXPORT_LOCK.json';assert not p.exists();p.write_text(json.dumps({'inputs_sha256':{str(f.relative_to(R)):sha(f) for f in sorted(files)}},indent=2)+'\n')
