"""Isolated lossless memory repair; old OOM attempt and pins are preserved."""
import argparse,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import desta3d_v2_output_anchor_probe as base
from scripts.desta3d_v2_p0 import read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once


def register(out):
    base.register(out)
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v2_output_anchor_offload.py']
    save_once(out/'MEMORY_REPAIR.json',{'previous':'probe001 OOM at first differentiable cached branch replay',
        'change':'lossless saved-tensor CPU offload at most8GiB; frozen model weight storage remains on GPU',
        'scientific_configuration':'same query/B1/teacher support/3step perturbation/tolerances, no loss or state change',
        'pins':{str(p):sha(p) for p in paths}})


def run(out,name):
    for p,h in read(out/'MEMORY_REPAIR.json')['pins'].items():assert sha(Path(p))==h,p
    import vg_tta.desta3d_v2_output_anchor as anchor
    from vg_tta.desta3d_v2_output_anchor_offload import replay_branch
    with patch.object(anchor,'replay_branch',replay_branch):base.run(out,name)


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
