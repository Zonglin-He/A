from pathlib import Path
import subprocess,sys,time
R=Path(__file__).resolve().parents[2];H=R/'research/round108'
while not (R/'crossplatform/round108/MATRIX_SEALED.json').exists():time.sleep(5)
for n in ['audit108','score108']:
 with (H/(n+'.log')).open('w') as f:subprocess.run([sys.executable,str(R/'tools/exploration_guard_round108.py'),'--resource','cpu','--timeout-seconds','3600','--',sys.executable,str(H/(n+'.py'))],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('POST108',n,'complete',flush=True)
