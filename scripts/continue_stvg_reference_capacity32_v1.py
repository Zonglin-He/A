"""One bounded GT-assisted decoder intervention and CPU summary scoring."""
from pathlib import Path
import subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]
for stage in ['measure','score']:
 subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_stvg_reference_capacity32_v1.py'),stage],cwd=ROOT,check=True,timeout=1100 if stage=='measure'else 900,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'if stage=='measure'else''))
