"""One isolated resident ReferDINO; native full boxes and selected raw mask."""
import json,sys,time,hashlib,contextlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from stvg.runtime import setup
from stvg.teacher import SingleRefer
from stvg.teacher_worker import deny_detector


def main():
    setup();sys.addaudithook(deny_detector)
    with contextlib.redirect_stdout(sys.stderr):teacher=SingleRefer()
    for line in sys.stdin:
        x=json.loads(line)
        if x.get('stop'):break
        try:
            request=torch.load(x['input'],map_location='cpu',weights_only=False);seen=[];original=teacher.model.infer
            def tap(*a,**kw):
                z=original(*a,**kw);seen.append(z);return z
            teacher.model.infer=tap;torch.manual_seed(42);np.random.seed(42);tick=time.perf_counter()
            try:
                with contextlib.redirect_stdout(sys.stderr):z=teacher(request['frames'],request['caption'])
            finally:teacher.model.infer=original
            assert len(seen)==1;logits=seen[0]['pred_masks'][0,:,z['selected_query']].detach().float().cpu();assert len(logits)==8 and torch.isfinite(logits).all()
            z.update(mask_logits=logits,teacher_image_size=z['input_shape'][-2:],RGB_sha256=hashlib.sha256(request['frames'].tobytes()).hexdigest(),teacher_seconds=time.perf_counter()-tick)
            torch.save(z,x['output']);del seen,request,logits,z
            torch.cuda.empty_cache();print(json.dumps(dict(ok=True)),flush=True)
        except Exception as e:print(json.dumps(dict(ok=False,error=type(e).__name__+': '+str(e))),flush=True);raise

if __name__=='__main__':main()
