"""Isolated memory-capacity repair; no science or prior pins changed."""
import argparse,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import desta3d_v2_tta8_output_anchor as base
from scripts.desta3d_v2_p0 import read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once
OUT=base.BASE/'target8_B1_output_anchor_v2'
RECEIPT=base.PARENT/'receipts/tta8_output_anchor_v2.json'


def register():
    with patch.object(base,'OUT',OUT):base.register()
    assert read(OUT/'CONFIG.json')==read(base.OUT/'CONFIG.json')
    assert read(OUT/'INPUTS.json')==read(base.OUT/'INPUTS.json')
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v2_output_anchor_memory_v6.py']
    save_once(OUT/'MEMORY_REPAIR.json',{'previous':str(base.OUT/'FAILURE.json'),
        'failure':'first HC13 physical grid32x22x28 vs source31x16x28 requires more memory; OOM before first Adam step',
        'change':'only saved activation host capacity8GiB to16GiB, initial host reserve6GiB; same stride-preserving v5 values/cache/backward and MLP checkpoint',
        'scientific_configuration_identical':True,'target_GT_read':False,
        'pins':{str(p):sha(p) for p in paths}})


def run():
    for p,h in read(OUT/'MEMORY_REPAIR.json')['pins'].items():assert sha(Path(p))==h,p
    import vg_tta.desta3d_v2_output_anchor_tta as tta
    from vg_tta.desta3d_v2_output_anchor_memory_v6 import replay_branch
    original=tta.adapt_output_anchor
    def adapt(*args,**kwargs):return original(*args,**kwargs,replay=replay_branch)
    with patch.object(base,'OUT',OUT),patch.object(base,'RECEIPT',RECEIPT),patch.object(tta,'adapt_output_anchor',adapt):base.run()


def launch():
    assert not (OUT/'STARTED.json').exists()
    start=time.monotonic();child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=ROOT)
    wall=time.monotonic()-start;worker=read(RECEIPT)['seconds'] if RECEIPT.exists() else 0.
    save_once(base.PARENT/'receipts/tta8_output_anchor_wrapper_v2.json',{'status':'completed' if child.returncode==0 else 'failed',
        'seconds':max(0.,wall-worker),'worker_seconds':worker,'child_wall_seconds':wall,'returncode':child.returncode})
    raise SystemExit(child.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','launch']);args=p.parse_args()
    {'register':register,'run':run,'launch':launch}[args.action]()
