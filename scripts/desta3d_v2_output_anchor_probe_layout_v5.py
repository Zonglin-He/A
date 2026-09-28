"""Isolated memory repair with compact teacher evidence and memory trace."""
import argparse, json, subprocess, sys, time, gc
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import desta3d_v2_output_anchor_probe as base
from scripts.desta3d_v2_p0 import read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once


def register(out):
    base.register(out)
    paths=[Path(__file__), ROOT/'vg_tta/desta3d_v2_output_anchor_checkpoint_v4.py', ROOT/'vg_tta/desta3d_v2_output_anchor_layout_v5.py', ROOT/'tests/test_desta3d_v2_output_anchor_layout.py',
           ROOT/'vg_tta/desta3d_v2_output_anchor_offload_v3.py',
           ROOT/'tests/test_desta3d_v2_output_anchor_checkpoint.py']
    save_once(out/'MEMORY_REPAIR.json',{
        'previous':'probe004 both native distributions match exactly; actual event backward fails SDPA bias stride21810 alignment4. FFN recompute removed forward OOM. No output backward passed yet.',
        'change':'retain v4 FFN recompute; allocate attention-bias storage padded to8 then slice to exact original shape, values/support unchanged; preserve strides across bounded CPU activation offload' ,
        'evidence_storage':'save complete actual teacher logits/token/cache trace and native readouts; omit duplicate large adapter feature tensors and updated_tokens from teacher injection diagnostics only; old raw files preserved',
        'scientific_configuration':'same source key10016/B1/teacher support/3step perturbation/tolerances; no changes to objective, values or state',
        'pins':{str(p):sha(p) for p in paths}})


def run(out,name):
    for p,h in read(out/'MEMORY_REPAIR.json')['pins'].items():assert sha(Path(p))==h,p
    import vg_tta.desta3d_v2_output_anchor as anchor
    from vg_tta.desta3d_v2_output_anchor_layout_v5 import replay_branch
    original_save=base.save_pt
    def memory(event):
        if not torch.cuda.is_initialized():return
        row={'event':event,'allocated':torch.cuda.memory_allocated(),
             'reserved':torch.cuda.memory_reserved(),'time':time.time()}
        with (out/'MEMORY_TRACE.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def compact_save(p,obj):
        if p.name=='TEACHER.pt':
            native=dict(obj['native'])
            for branch in ('event','spatial'):
                key=branch+'_injection'
                if native.get(key) is not None:
                    native[key]={k:v for k,v in native[key].items() if k not in ('fields','updated_tokens')}
            obj={'native':native,'trace':obj['trace'],
                 'storage_note':'complete output/teacher-forcing trace; duplicate dense injection fields omitted'}
        original_save(p,obj)
        memory('saved:'+p.name)
    def replay(*args,**kwargs):
        gc.collect();torch.cuda.empty_cache();memory('start:'+args[5])
        try:
            result=replay_branch(*args,**kwargs)
            memory('finish:'+args[5]);return result
        finally:memory('finally:'+args[5])
    with patch.object(anchor,'replay_branch',replay),patch.object(base,'save_pt',compact_save):
        base.run(out,name)


def launch(out,name):
    assert not (out/'STARTED.json').exists()
    start=time.monotonic();child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'run','--name',name],cwd=ROOT)
    wall=time.monotonic()-start;receipt=base.PARENT/f'receipts/output_anchor_{name}.json'
    worker=read(receipt)['seconds'] if receipt.exists() else 0.
    save_once(base.PARENT/f'receipts/output_anchor_{name}_wrapper.json',{'status':'completed' if child.returncode==0 else 'failed',
        'seconds':max(0.,wall-worker),'worker_seconds':worker,'child_wall_seconds':wall,'returncode':child.returncode})
    raise SystemExit(child.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','launch']);p.add_argument('--name',required=True)
    a=p.parse_args();assert a.name.replace('_','').isalnum()
    out=base.BASE/'output_anchor_probe_v1'/a.name
    if a.action=='register':register(out)
    elif a.action=='run':run(out,a.name)
    else:launch(out,a.name)
