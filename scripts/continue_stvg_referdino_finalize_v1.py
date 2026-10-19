"""Finite serial stages only; no scheduler or autonomous continuation."""
from pathlib import Path
import subprocess,os,sys,json,time
R=Path(__file__).resolve().parents[1];D=R/'artifacts/stvg_referdino_finalize_v1'
for i in range(8):
 with (D/f'Live{i}.log').open('ab')as f:
  p=subprocess.run([str(R/'.conda/tubedetr/bin/python'),'-u','-B','scripts/run_stvg_referdino_finalize_v1.py','measure','--chunk',str(i)],cwd=R,stdout=f,stderr=subprocess.STDOUT,timeout=900,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))
 assert p.returncode==0,(i,p.returncode)
(D/'DEVELOPMENT_WORKER_COMPLETE.json').write_text(json.dumps(dict(status='all128Live_predictions_sealed',time=time.time())))
