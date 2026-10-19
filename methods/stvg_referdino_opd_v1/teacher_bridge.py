"""Isolate the two upstream packages' conflicting `models` module names."""
import subprocess,tempfile,time,os
from pathlib import Path
import torch
from .runtime import ROOT

def refer_native(frames,caption):
 t=time.perf_counter()
 with tempfile.TemporaryDirectory(prefix='stvg-refer-')as d:
  a=Path(d)/'request.pt';b=Path(d)/'reply.pt';torch.save(dict(frames=frames,caption=caption),a)
  x=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','-m','methods.stvg_referdino_opd_v1.teacher_worker',str(a),str(b)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  if x.returncode:raise RuntimeError('ReferDINO subprocess failed: '+x.stdout.decode(errors='replace')[-4000:])
  out=torch.load(b,map_location='cpu',weights_only=False)
 out['process_wall_seconds']=time.perf_counter()-t;return out
