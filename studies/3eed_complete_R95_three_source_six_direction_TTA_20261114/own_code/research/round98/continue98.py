"""Watch this one finite prediction job, then execute its already-authorized audit chain."""
from pathlib import Path
import json,subprocess,time
R=Path(__file__).resolve().parents[2];H=R/'research/round98'
while True:
 s=json.loads((H/'STATE.json').read_text())
 if s['current_stage']=='all_six_R95_predictions_sealed_await_native_audit':break
 if 'FAILED' in s['state']:raise RuntimeError(s)
 time.sleep(50)
subprocess.run([str(R/'crossplatform/env/bin/python'),str(H/'postprocess98.py')],cwd=R,check=True)
