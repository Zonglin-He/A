"""One bounded cached decomposition -> offline diagnosis chain."""
from pathlib import Path
import sys,subprocess,os
ROOT=Path(__file__).resolve().parents[1]
for stage in ['seal','diagnose']:
 subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_stvg_pair_direction_dev64_v1.py'),stage],check=True,timeout=900,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),cwd=ROOT)
