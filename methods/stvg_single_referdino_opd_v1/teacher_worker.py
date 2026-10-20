"""Isolated one-expert inference with detector-checkpoint access denied."""
import sys,time
from pathlib import Path
import numpy as np,torch
from methods.stvg_referdino_opd_v1.runtime import setup
from .teacher import SingleRefer

def deny_detector(event,args):
    if event=='open'and args and isinstance(args[0],(str,bytes,Path)):
        p=str(args[0]).lower()
        if '/checkpoints/groundingdino' in p or 'grounding-dino-tiny' in p:
            raise PermissionError('Single-expert method forbids independent detector checkpoint: '+p)

def main():
    setup();sys.addaudithook(deny_detector);torch.manual_seed(42);np.random.seed(42)
    inp,out=sys.argv[1:];x=torch.load(inp,map_location='cpu',weights_only=False);assert set(x)=={'frames','caption'}
    start=time.perf_counter();teacher=SingleRefer();torch.cuda.synchronize();elapsed=time.perf_counter()-start
    z=teacher(x['frames'],x['caption']);z.update(model_load_seconds=elapsed,peak_GPU_bytes=torch.cuda.max_memory_allocated(),load_audit=teacher.load_audit)
    torch.save(z,out)
if __name__=='__main__':main()
