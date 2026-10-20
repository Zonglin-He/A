"""Isolated one-expert inference with detector-checkpoint access denied."""
import sys,time
from pathlib import Path
import numpy as np,torch
from stvg.runtime import setup
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
def serve():
    """One frozen expert; release all model GPU tensors between requests."""
    import contextlib,json,gc,random,traceback
    setup();sys.addaudithook(deny_detector)
    teacher=None;rng=None
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        try:
            with contextlib.redirect_stdout(sys.stderr):
                x=torch.load(request['input'],map_location='cpu',weights_only=False)
                if set(x)!={'frames','caption'}:raise ValueError('Only RGB and caption are accepted')
                elapsed=0.
                if teacher is None:
                    torch.manual_seed(42);np.random.seed(42)
                    start=time.perf_counter();teacher=SingleRefer();torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate())
                teacher.model.cuda()
                torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);np.random.set_state(rng[2]);random.setstate(rng[3])
                torch.cuda.reset_peak_memory_stats()
                try:
                    z=teacher(x['frames'],x['caption'])
                    z.update(model_load_seconds=elapsed,peak_GPU_bytes=torch.cuda.max_memory_allocated(),load_audit=teacher.load_audit)
                finally:
                    teacher.model.cpu();gc.collect();torch.cuda.empty_cache()
                torch.save(z,request['output'])
            print(json.dumps(dict(ok=True)),flush=True)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps(dict(ok=False,error=traceback.format_exc()[-3000:])),flush=True)
            break

if __name__=='__main__':
    if sys.argv[1:]==['--serve']:serve()
    else:main()
