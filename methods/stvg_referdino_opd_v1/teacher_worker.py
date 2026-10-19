"""Isolated official ReferDINO runtime, no dataset, scoring, or GT access."""
import sys,time
from pathlib import Path
import numpy as np,torch
from .runtime import setup
from .experts import NativeRefer

def main():
 setup();torch.manual_seed(42);np.random.seed(42)
 inp,out=sys.argv[1:];x=torch.load(inp,map_location='cpu',weights_only=False);assert set(x)=={'frames','caption'}
 t=time.perf_counter();teacher=NativeRefer();torch.cuda.synchronize();load_s=time.perf_counter()-t
 z=teacher(x['frames'],x['caption']);z['model_load_seconds']=load_s;z['peak_GPU_bytes']=torch.cuda.max_memory_allocated();torch.save(z,out)
if __name__=='__main__':main()
