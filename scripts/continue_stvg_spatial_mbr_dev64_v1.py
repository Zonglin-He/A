"""One finite CPU readout -> global seal -> offline scoring chain."""
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for stage in ['measure','score']:
 subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_stvg_spatial_mbr_dev64_v1.py'),stage],cwd=ROOT,check=True,timeout=900,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
