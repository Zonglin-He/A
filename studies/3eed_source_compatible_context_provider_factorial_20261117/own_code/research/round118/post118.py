from pathlib import Path
import subprocess
R=Path(__file__).resolve().parents[2];H=R/'research/round118'
for s in ['audit118.py','score118.py','audit_scores118.py']:
 with (H/s.replace('.py','.log')).open('w') as log:subprocess.run([str(R/'crossplatform/env/bin/python'),str(R/'tools/exploration_guard_round118.py'),'--resource','cpu','--timeout-seconds','3600','--',str(R/'crossplatform/env/bin/python'),str(H/s)],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
print('POST118_COMPLETE',flush=True)
