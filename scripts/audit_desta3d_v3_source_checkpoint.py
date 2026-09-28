"""Independent CPU readback of a committed v3 source-fit checkpoint."""
import argparse,json,os,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.desta3d_v2_p0 import read,sha,adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
from vg_tta.desta3d_v3_data import balanced_epoch
from vg_tta.desta3d_v2 import Desta3DAdapterV2
from vg_tta.desta3d_v2_training import make_source_optimizer,warmup_cosine
from vg_tta.optimizer_checkpoint import restore_optimizer,validate_serialized_optimizer
from vg_tta.desta3d_v3_training import optimizer_counters
OUT=ROOT/'artifacts/desta3d_v3/full_source_fit_v1';ROSTER=ROOT/'artifacts/desta3d_v3/full_source_roster_v1'
def main(name):
    torch.set_num_threads(4);p=OUT/'audit_points'/f'{name}.pt';p.parent.mkdir(parents=True,exist_ok=True);os.link(OUT/'LATEST.pt',p)
    s=torch.load(p,map_location='cpu',weights_only=False);cfg=read(OUT/'CONFIG.json');assert s['lock_sha']==sha(OUT/'LOCK.json')
    for f,h in read(OUT/'LOCK.json')['pins'].items():assert sha(f)==h,f
    validate_serialized_optimizer(s['optimizer']);m=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False)
    m.load_state_dict(s['adapter']);o=make_source_optimizer(m,'repaired','B');restore_optimizer(o,s['optimizer'])
    counts=optimizer_counters(m,o)
    assert all(torch.isfinite(v).all() for v in s['adapter'].values())
    assert all(torch.isfinite(v).all() for st in o.state.values() for v in st.values() if isinstance(v,torch.Tensor))
    assert s['cpu_rng'].numel()>0 and s['cuda_rng'] and all(v.numel()>0 for v in s['cuda_rng'])
    rows=[r for r in read(ROSTER/'INPUTS.json') if r['split']=='train'];orders={};windows=[];cursor=0;epoch=0
    for step in range(1,s['step']+1):
        e=(step-1)//cfg['source_counts']['updates_per_epoch'];h=read(OUT/'history'/f'E{e}_S{step:07d}.json')
        if e not in orders:orders[e]=balanced_epoch(rows,e,cfg['seed'])
        if e!=epoch:cursor=0;epoch=e
        order=orders[e];stop=min(cursor+cfg['accum'],len(order))
        assert h['step']==step and h['cursor']==stop and h['divisor']==stop-cursor
        assert [r['key'] for r in h['queries']]==[rows[order[i]]['key'] for i in range(cursor,stop)]
        assert h['lr_scale']==warmup_cosine(step-1,cfg['total_horizon'])
        assert all(v<=step and v>0 for v in h['actual_counters'].values());cursor=stop;windows.append(h)
    if windows:
        assert windows[-1]==s['last_window'] and counts==windows[-1]['actual_counters']
        assert s['cursor']==cursor and s['epoch']==epoch
    result=dict(time=time.time(),status='committed_state_CPU_independently_verified',snapshot_sha=sha(p),
      epoch=s['epoch'],cursor=s['cursor'],step=s['step'],stage=s['stage'],adapter_sha=adapter_sha256(m),
      optimizer_int_keys=True,all_optimizer_states_live_Parameter_bound_exact=True,actual_counters=counts,
      checked_complete_windows=len(windows),query_order_exact=True,RNG_present=True,all_finite=True,
      labels_or_predictions_read=False,limits='CPU binding readback plus worker actual counters; continuing GPU update requires separate next-window verification')
    save_once(OUT/f'{name}.json',result);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');main(p.parse_args().name)
