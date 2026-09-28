"""Frozen B1 query-conditioned source moments; no labels or target inputs."""
from __future__ import annotations
import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.desta3d_v2_p0 import read, sha, adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once, state_sha
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts, tensor_sha256
from vg_tta.optimizer_checkpoint import cpu_clone
from vg_tta.desta3d_tta_v1 import SourceFeatureStatsAccumulator

PARENT = ROOT / 'artifacts/desta3d_v2'
REF = PARENT / 'aux_backflow_v1/final_B1_reference_v1'
OUT = PARENT / 'tta_v2/source_moments_B1_v1'
RECEIPT = PARENT / 'receipts/source_moments_B1_v1.json'


def jsonable(x):
    if isinstance(x, torch.Tensor): return x.tolist()
    if isinstance(x, dict): return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    return x


def register():
    assert not OUT.exists()
    assert read(REF / 'SCORING_COMPLETE.json')['status'] == 'completed_and_crosschecked'
    audit = read(REF / 'ROOT_SCORE_CROSSCHECK.json'); assert audit['status'] == 'passed'
    score_path = REF / 'B1_final001/independent_cpu_readback/INDEPENDENT_SCORE.json'
    assert audit['report_sha'] == sha(score_path)
    scored = read(score_path)
    assert scored['arms']['shared_reference_time']['summary']['parent_macro']['vIoU'] >= scored['frozen_reference']['parent_macro']['vIoU']
    old = read(REF / 'B1_final001/CONFIG.json')
    ck = old['checkpoint']; assert sha(Path(ck['checkpoint'])) == ck['sha256']
    rows = sorted([r for r in read(PARENT / 'source_fit/INPUTS.json') if r['split'] == 'train'], key=lambda r:r['key'])
    assert len(rows) == 618 and len({r['source'] for r in rows}) == 95
    c = {'checkpoint': ck, 'queries': 618, 'parents': 95, 'condition': 'clean',
        'feature_definition': 'v2_query_conditioned_readers', 'feature_keys': ['branch_features_spatial', 'branch_features_event'],
        'aggregation': 'THW moments within query, equal query within parent, equal parent; total second-moment variance',
        'optimizer_steps': 0, 'GT_read': False, 'target_data_read': False, 'seed': 20260927,
        'phase_seconds': 1800, 'free_disk_bytes': 8*2**30, 'cumulative_cap_seconds': None}
    save_once(OUT / 'CONFIG.json', c); save_once(OUT / 'INPUTS.json', rows)
    paths = [Path(__file__), Path(ck['checkpoint']), OUT / 'CONFIG.json', OUT / 'INPUTS.json',
        REF / 'SCORING_COMPLETE.json', REF / 'ROOT_SCORE_CROSSCHECK.json', score_path,
        PARENT / 'source_fit/LOCK.json', ROOT / 'vg_tta/desta3d_v2.py', ROOT / 'vg_tta/desta3d_v2_ptd.py',
        ROOT / 'vg_tta/desta3d_tta_v1.py', ROOT / 'vg_tta/optimizer_checkpoint.py', ROOT / 'scripts/ptd_spatial_adapter_ab_v1.py',
        ROOT / 'scripts/desta3d_tta_run_v1.py', ROOT / 'methods/CURRENT_METHOD.json']
    save_once(OUT / 'LOCK.json', {'pins': {str(p):sha(p) for p in paths}})
    save_once(OUT / 'REGISTRATION.json', {'status':'registered_before_inference', 'time':time.time(),
        'purpose':'new source moments for the authorized v2 view-only versus +alignment TTA control; no v1 feature reuse',
        'training_or_target_predictions':False, 'queries':618, 'parents':95})
    print('REGISTERED', OUT, flush=True)


def run():
    c = read(OUT / 'CONFIG.json'); assert not (OUT / 'STARTED.json').exists()
    for path, digest in read(OUT / 'LOCK.json')['pins'].items(): assert sha(Path(path)) == digest, path
    for path, digest in read(PARENT / 'source_fit/LOCK.json')['pins'].items(): assert sha(Path(path)) == digest, path
    lease = (ROOT / 'artifacts/spatial_tta_research_v2/gpu.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    began = time.monotonic(); status='running'; failure=None
    prior = sum(scan_nested_gpu_receipts(p)[0] for p in [ROOT / 'artifacts/desta3d_v1', PARENT])
    try:
        save_once(OUT / 'STARTED.json', {'time':time.time(), 'pid':os.getpid(), 'prior_seconds':prior})
        def guard():
            assert shutil.disk_usage(ROOT).free >= c['free_disk_bytes'], 'disk reserve'
            assert time.monotonic()-began <= c['phase_seconds'], 'engineering review interval, partials preserved'
        guard(); torch.set_num_threads(4); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
        torch.cuda.reset_peak_memory_stats()
        from scripts.ptd_spatial_adapter_ab_v1 import model_load, processor_load, frames_for, inputs_for
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields
        pr=processor_load(); model=model_load(); model.requires_grad_(False); model.eval()
        adapter=Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False).cuda().eval()
        state=torch.load(c['checkpoint']['checkpoint'], map_location='cpu', weights_only=False)['adapter']
        adapter.load_state_dict(state); adapter.set_train_stage('frozen')
        digest=adapter_sha256(adapter); assert digest==c['checkpoint']['adapter_sha256']
        accumulator=SourceFeatureStatsAccumulator(source_split='train'); raw=[]
        for i,row in enumerate(read(OUT / 'INPUTS.json')):
            guard(); frames,ids=frames_for(row,'clean'); assert ids==row['input']['frame_ids']
            prompt,prep=inputs_for(row,pr,frames)
            f=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
            with torch.no_grad(): fields=adapter(f['visual_grid'],f['query_tokens'],query_mask=f['query_mask'],frame_times=f['frame_times'])
            mapped={'referent':fields['branch_features_spatial'], 'event':fields['branch_features_event']}
            accumulator.add(parent_id=row['source'], query_id=row['key'], branch_features=mapped)
            moments={b:{'mean':accumulator._rows[b][row['key']][0], 'second_moment':accumulator._rows[b][row['key']][1]} for b in mapped}
            record={'key':row['key'],'source':str(row['source']),'frame_ids':ids,'video_sha':row['input']['video_sha256'],
                'preprocess':prep,'query_tokens_sha':tensor_sha256(f['query_tokens']), 'visual_grid_sha':tensor_sha256(f['visual_grid']),
                'frame_times_sha':tensor_sha256(f['frame_times']),'moments':moments,'GT_read':False}
            file=OUT / 'queries' / f'{i:03}.pt'; file.parent.mkdir(exist_ok=True)
            temp=file.with_suffix('.tmp'); torch.save(cpu_clone(record),temp); os.replace(temp,file); raw.append(file)
            assert adapter_sha256(adapter)==digest
            if (i+1)%20==0 or i+1==618: print('SOURCE_MOMENTS',i+1,618,flush=True)
            del frames,prompt,f,fields,mapped; gc.collect()
        result=accumulator.finalize()
        payload={'feature_definition':'v2_query_conditioned_readers','source_split':'train','GT_read':False,
            'adapter_sha':digest,'checkpoint_sha':c['checkpoint']['sha256'],'queries':618,'parents':95,
            'aggregation':result['aggregation'],'parent_moments':result['parents']}
        for dst,src in [('spatial','referent'),('event','event')]:
            g=result['parents'][src]['__global__']; assert g['query_count']==618 and g['parent_count']==95
            assert g['mean'].numel()==128 and torch.isfinite(g['mean']).all() and torch.isfinite(g['std']).all() and (g['std']>=0).all()
            payload[dst]={'mean':g['mean'],'std':g['std'],'second_moment':g['second_moment']}
        assert not any(p.requires_grad or p.grad is not None for p in adapter.parameters())
        assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
        save_once(OUT / 'MOMENTS.json', jsonable(payload))
        save_once(OUT / 'COMPLETE.json', {'status':'completed_source_moments','queries':618,'parents':95,
            'pins':{str(p):sha(p) for p in raw+[OUT / 'MOMENTS.json']}, 'adapter_unchanged':digest,
            'GT_read':False,'target_data_read':False,'optimizer_steps':0})
        status='completed'
    except BaseException:
        status='failed'; failure=traceback.format_exc(); save_once(OUT / 'FAILURE.json',{'failure':failure,'time':time.time()}); raise
    finally:
        seconds=time.monotonic()-began
        save_once(RECEIPT,{'status':status,'seconds':seconds,'prior_seconds':prior,'cumulative_seconds':prior+seconds,
            'failure':failure,'cumulative_cap_seconds':None,'peak_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None})
        lease.close()


def launch():
    assert not RECEIPT.exists() and not (OUT / 'STARTED.json').exists()
    began=time.monotonic(); child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=ROOT)
    wall=time.monotonic()-began; worker=read(RECEIPT)['seconds'] if RECEIPT.exists() else 0.
    save_once(PARENT / 'receipts/source_moments_B1_wrapper_v1.json', {'status':'completed' if child.returncode==0 else 'failed',
        'seconds':max(0.,wall-worker),'child_wall_seconds':wall,'worker_seconds':worker,'returncode':child.returncode,
        'scope':'incremental launch overhead excluding worker receipt','cumulative_cap_seconds':None})
    raise SystemExit(child.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','launch']);a=p.parse_args()
    {'register':register,'run':run,'launch':launch}[a.action]()
