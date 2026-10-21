"""Same frozen native teacher; read-only feature hook, no alternate checkpoint."""
import sys,json,time,contextlib,gc,random,traceback
import torch,numpy as np
from pathlib import Path
from stvg.runtime import setup
from stvg.teacher import SingleRefer
from stvg.teacher_worker import deny_detector

def serve():
    setup();sys.addaudithook(deny_detector);teacher=None;rng=None
    for line in sys.stdin:
        req=json.loads(line)
        if req.get('stop'):break
        try:
            with contextlib.redirect_stdout(sys.stderr):
                x=torch.load(req['input'],map_location='cpu',weights_only=False)
                assert set(x)=={'frames','caption'}
                elapsed=0.
                if teacher is None:
                    torch.manual_seed(42);np.random.seed(42);start=time.perf_counter();teacher=SingleRefer();torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate())
                teacher.model.cuda();torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);np.random.set_state(rng[2]);random.setstate(rng[3]);torch.cuda.reset_peak_memory_stats();features=[];masks=[]
                def capture(module,args):
                    srcs,ms=args[0],args[1];assert srcs[1].ndim==4
                    features.append(srcs[1].detach().float().cpu().clone());masks.append(ms[1].detach().cpu().clone())
                hook=teacher.model.transformer.register_forward_pre_hook(capture)
                try:
                    z=teacher(x['frames'],x['caption']);assert len(features)==1 and features[0].shape[0]==8
                    z.update(layout_features=features[0],layout_mask=masks[0],feature_definition='transformer input srcs[1], projected backbone stride16, before text fusion',model_load_seconds=elapsed,peak_GPU_bytes=torch.cuda.max_memory_allocated(),load_audit=teacher.load_audit)
                finally:hook.remove();teacher.model.cpu();gc.collect();torch.cuda.empty_cache()
                torch.save(z,req['output'])
            print(json.dumps(dict(ok=True)),flush=True)
        except Exception:
            traceback.print_exc(file=sys.stderr);print(json.dumps(dict(ok=False,error=traceback.format_exc()[-3000:])),flush=True);break

if __name__=='__main__':serve()
