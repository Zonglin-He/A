"""One native ReferDINO process; its returned evidence is used twice."""
import os,subprocess,tempfile,time
from pathlib import Path
import torch
from stvg.runtime import ROOT
def refer_native(frames,caption):
    start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='single-ref-')as d:
        inp,out=Path(d)/'request.pt',Path(d)/'reply.pt';torch.save(dict(frames=frames,caption=caption),inp)
        p=subprocess.run([str(Path(__import__('sys').executable)),'-B','-m','stvg.teacher_worker',str(inp),str(out)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        if p.returncode:raise RuntimeError(p.stdout.decode(errors='replace')[-5000:])
        z=torch.load(out,map_location='cpu',weights_only=False)
    z['process_wall_seconds']=time.perf_counter()-start;return z
