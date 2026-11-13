"""Sequential postseal scoring, independent scalar checks and descriptive analysis."""
from pathlib import Path
import subprocess,sys,json
H=Path(__file__).resolve().parent;assert json.loads((H/'independent_native_audit_attempt001/AUDIT_RESULT.json').read_text())['state'].startswith('PASS_')
for f in ['score89.py','audit_scalar89.py','analyze89.py','historical_abc89.py','step_effect89.py','routing_diagnostic89.py','render89.py','plot89.py','write_report89.py','append_context89.py','audit_report89.py','audit_context89.py','normalize_public89.py']:
 print('POSTSCORE89',f,flush=True);subprocess.run([sys.executable,str(H/f)],check=True)
