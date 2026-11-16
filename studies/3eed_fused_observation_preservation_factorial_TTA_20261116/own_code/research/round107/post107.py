from pathlib import Path
import subprocess,sys,time
R=Path(__file__).resolve().parents[2];H=R/'research/round107'
while not (H/'NATIVE_AUDIT.json').exists():time.sleep(5)
for n in ['score107','report107']:
 with (H/(n+'.log')).open('w') as f:subprocess.run([sys.executable,str(R/'tools/exploration_guard_round107.py'),'--resource','cpu','--timeout-seconds','3600','--',sys.executable,str(H/(n+'.py'))],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('POST107',n,'complete',flush=True)
