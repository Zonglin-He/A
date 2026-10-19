"""Exactly one bounded GPU measurement then independent CPU offline scoring."""
import subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for stage in ['measure','score']:
 subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_stvg_sparse_state_route32_v1.py'),stage],cwd=ROOT,check=True,timeout=1100 if stage=='measure'else 900,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'if stage=='measure'else''))
