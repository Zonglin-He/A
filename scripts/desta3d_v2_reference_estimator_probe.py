"""Single reference-standard-deviation intervention, source-only, no optimizer."""
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
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts, tensor_sha256, make_mild_photometric_view
from vg_tta.optimizer_checkpoint import cpu_clone

BASE = ROOT / 'artifacts/desta3d_v2'
SOURCE = BASE / 'tta_v2/source_moments_B1_v1'
OLD = BASE / 'tta_v2/view_alignment_signal_source_v1'
OUT = BASE / 'tta_v2/source_reference_estimator_v1'
RECEIPT = BASE / 'receipts/source_reference_estimator_v1.json'


def save_tensor(path, value):
    assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(cpu_clone(value), temp)
    os.replace(temp, path)


def register():
    assert not OUT.exists()
    import numpy as np
    from vg_tta.desta3d_v2_query_std_reference import aggregate_query_std
    old=read(SOURCE / 'MOMENTS.json');sealed=read(SOURCE / 'COMPLETE.json')['pins']
    assert sha(SOURCE / 'MOMENTS.json')==sealed[str(SOURCE / 'MOMENTS.json')]
    allrows=[]
    for path in sorted((SOURCE / 'queries').glob('*.pt')):
        assert sha(path)==sealed[str(path)]
        allrows.append(torch.load(path,map_location='cpu',weights_only=False))
    result=aggregate_query_std(allrows);assert result['queries']==618 and result['parents']==95
    parents=sorted({r['source'] for r in allrows})
    alternative={'feature_definition':'v2_query_conditioned_readers','source_split':'train',
        'queries':618,'parents':95,'adapter_sha':old['adapter_sha'],'checkpoint_sha':old['checkpoint_sha'],
        'std_definition':'equal parent then equal query expected within-query population std',
        'variance_floor':1e-12,'mean_definition':'original global mean copied exactly',
        'GT_read':False,'target_data_read':False,'original_moments_sha':sha(SOURCE / 'MOMENTS.json')}
    errors={};changes={}
    for src,dst in [('referent','spatial'),('event','event')]:
        reference=result['branches'][dst]
        parent_std=[];parent_mean=[]
        for parent in parents:
            qs=[r for r in allrows if r['source']==parent]
            mu=np.array([r['moments'][src]['mean'].numpy() for r in qs],dtype=np.float64)
            sec=np.array([r['moments'][src]['second_moment'].numpy() for r in qs],dtype=np.float64)
            parent_std.append(np.sqrt(np.maximum(sec-mu*mu,1e-12)).mean(0));parent_mean.append(mu.mean(0))
        independent=np.stack(parent_std).mean(0);mean=np.stack(parent_mean).mean(0)
        errors[dst]={'expected_std':float(np.max(np.abs(independent-reference['expected_query_std'].numpy()))),
            'mean':float(np.max(np.abs(mean-np.asarray(old[dst]['mean'])))),
            'old_population_std':float(np.max(np.abs(reference['population_std'].numpy()-np.asarray(old[dst]['std']))))}
        assert max(errors[dst].values())<1e-12
        alternative[dst]={'mean':old[dst]['mean'],'std':reference['expected_query_std'].tolist()}
        oldstd=np.asarray(old[dst]['std']);newstd=independent
        changes[dst]={'old_std':old[dst]['std'],'expected_query_std':newstd.tolist(),
            'within_query_rms_std':reference['within_query_rms_std'].tolist(),
            'relative_L2_change':float(np.linalg.norm(newstd-oldstd)/np.linalg.norm(oldstd)),
            'new_to_old_ratio_quantiles':np.quantile(newstd/oldstd,[0,.25,.5,.75,1]).tolist()}
    save_once(OUT / 'ALTERNATIVE_MOMENTS.json',alternative)
    rows=read(OLD / 'INPUTS.json');assert len(rows)==4 and len({r['source'] for r in rows})==4
    cfg=dict(read(OLD / 'CONFIG.json'))
    cfg.update(student_views=['brightness1.05_contrast0.95'],reference_arms=['population','expected_query_std'],
        mean_unchanged=True,phase_seconds=300,
        near_same_resource_rule={'all_total_cosine_at_least':.995,'all_relative_L2_at_most':.15},
        control_repeat_tolerance={'atol':1e-8,'rtol':1e-5},new_target_inference=False)
    save_once(OUT / 'CONFIG.json',cfg);save_once(OUT / 'INPUTS.json',rows)
    save_once(OUT / 'CPU_PREFLIGHT.json',{'status':'passed','synthetic_tests':3,
        'test_command':'.venv-ptd-audit/bin/python -B -m pytest -q tests/test_desta3d_v2_query_std_reference.py',
        'scope':'unequal parents, queries and grid sizes; expected std vs RMS/population; floor and invalid support',
        'source_aggregation_errors':errors,'reference_changes':changes,'quantile_levels':[0,.25,.5,.75,1],
        'source_queries':618,'source_parents':95,'GPU_used':False,'GT_read':False})
    paths=[Path(__file__),ROOT / 'vg_tta/desta3d_v2_query_std_reference.py',ROOT / 'tests/test_desta3d_v2_query_std_reference.py',
        ROOT / 'protocols/desta3d_v2_source_reference_estimator_v1.md',ROOT / 'vg_tta/desta3d_v2_signal_probe.py',
        ROOT / 'vg_tta/desta3d_v2_tta_pilot.py',ROOT / 'vg_tta/desta3d_v2_tta_objective.py',
        ROOT / 'vg_tta/desta3d_v2.py',ROOT / 'vg_tta/desta3d_v2_ptd.py',ROOT / 'vg_tta/optimizer_checkpoint.py',
        ROOT / 'scripts/ptd_spatial_adapter_ab_v1.py',ROOT / 'scripts/desta3d_tta_run_v1.py',
        Path(cfg['checkpoint']['checkpoint']),SOURCE / 'MOMENTS.json',SOURCE / 'COMPLETE.json',
        SOURCE / 'INPUTS.json',SOURCE / 'ROOT_FULL_READBACK.json',OLD / 'COMPLETE.json',
        OUT / 'ALTERNATIVE_MOMENTS.json',OUT / 'CPU_PREFLIGHT.json',OUT / 'CONFIG.json',OUT / 'INPUTS.json',ROOT / 'methods/CURRENT_METHOD.json']
    for i,row in enumerate(rows):
        paths += [OLD / 'queries' / f'{i:02}_mild.pt',OLD / 'queries' / f'{i:02}_IDENTITY.json',
            SOURCE / 'queries' / f"{row['source_moments_index']:03}.pt"]
    save_once(OUT / 'LOCK.json',{'pins':{str(p):sha(p) for p in paths}})
    save_once(OUT / 'REGISTRATION.json',{'status':'registered_before_source_inference','time':time.time(),
        'purpose':'one source reference std estimator intervention, no optimizer or targets',
        'source_queries':[r['key'] for r in rows],'parents':[r['source'] for r in rows],
        'optimizer_steps':0,'source_GT_read':False,'target_data_read':False})
    print('REGISTERED', OUT, flush=True)


def run():
    cfg=read(OUT / 'CONFIG.json'); assert not (OUT / 'STARTED.json').exists()
    for path,digest in read(OUT / 'LOCK.json')['pins'].items(): assert sha(Path(path))==digest, path
    lease=(ROOT / 'artifacts/spatial_tta_research_v2/gpu.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX | fcntl.LOCK_NB)
    began=time.monotonic(); status='running'; failure=None
    prior=sum(scan_nested_gpu_receipts(ROOT / 'artifacts' / n)[0] for n in ['desta3d_v1','desta3d_v2'])
    try:
        save_once(OUT / 'STARTED.json', {'time':time.time(),'pid':os.getpid(),'prior_seconds':prior})
        def guard():
            assert shutil.disk_usage(ROOT).free>=cfg['free_disk_bytes'], 'disk reserve'
            assert time.monotonic()-began<cfg['phase_seconds'], 'finite engineering allocation; partials retained'
        guard(); torch.set_num_threads(4); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
        torch.cuda.reset_peak_memory_stats()
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields
        from vg_tta.desta3d_v2_signal_probe import measure_signals
        processor=processor_load(); model=model_load(); model.eval().requires_grad_(False)
        adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        initial=torch.load(cfg['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter']
        adapter.load_state_dict(initial); adapter.set_train_stage('frozen')
        digest=adapter_sha256(adapter); assert digest==cfg['checkpoint']['adapter_sha256']
        moments=read(SOURCE / 'MOMENTS.json'); alternative=read(OUT / 'ALTERNATIVE_MOMENTS.json'); outputs=[]
        for index,row in enumerate(read(OUT / 'INPUTS.json')):
            guard(); frames,ids=frames_for(row,'clean'); assert ids==row['input']['frame_ids']
            prompt,pre=inputs_for(row,processor,frames)
            view_prompt,view_pre=inputs_for(row,processor,make_mild_photometric_view(frames))
            args=(row['input']['caption'],ids,row['input']['fps'])
            fields=capture_stock_fields(model,processor,prompt,*args)
            view=capture_stock_fields(model,processor,view_prompt,*args)
            assert pre['grid']==view_pre['grid']
            assert torch.equal(fields['frame_times'],view['frame_times'])
            assert fields['visual_grid'].shape==view['visual_grid'].shape
            source_record=torch.load(SOURCE / 'queries' / f"{row['source_moments_index']:03}.pt",map_location='cpu',weights_only=False)
            identity={'key':row['key'],'source':row['source'],'frame_ids':ids,'video_sha':row['input']['video_sha256'],
                'preprocess':pre,'view_preprocess':view_pre,'adapter_sha':digest,'source_moments_sha':sha(SOURCE / 'MOMENTS.json')}
            for key in ['query_tokens','visual_grid','frame_times']:
                identity[key+'_sha']=tensor_sha256(fields[key]); identity['view_'+key+'_sha']=tensor_sha256(view[key])
                a=fields[key].float(); b=view[key].float(); assert a.shape==b.shape
                identity[key+'_relative_L2']=float((b-a).double().norm()/a.double().norm().clamp_min(1e-30))
                identity[key+'_shape']=list(a.shape)
                assert source_record[key+'_sha']==identity[key+'_sha'], key
            assert source_record['preprocess']==pre and source_record['frame_ids']==ids
            assert source_record['key']==row['key'] and str(source_record['source'])==str(row['source'])
            path=OUT / 'queries' / f'{index:02}_IDENTITY.json'; save_once(path,identity); outputs.append(path)
            previous=read(OLD / 'queries' / f'{index:02}_IDENTITY.json')
            assert previous==identity
            for label,reference in [('population',moments),('expected_query_std',alternative)]:
                record=measure_signals(adapter,fields,view,reference)
                if label=='population':
                    control=torch.load(OLD / 'queries' / f'{index:02}_mild.pt',map_location='cpu',weights_only=False)
                    assert control['ordered_names']==record['ordered_names']
                    repeat={}
                    for name,g in record['raw_gradients'].items():
                        assert torch.allclose(g,control['raw_gradients'][name],**cfg['control_repeat_tolerance'])
                        repeat[name]={'exact':torch.equal(g,control['raw_gradients'][name]),
                            'max_abs':float((g-control['raw_gradients'][name]).abs().max())}
                    record['control_repeat']=repeat
                assert adapter_sha256(adapter)==digest
                assert all(not p.requires_grad and p.grad is None for p in model.parameters())
                assert all(not p.requires_grad and p.grad is None for p in adapter.parameters())
                record.update({'key':row['key'],'source':row['source'],'view':'mild','reference':label,'adapter_sha':digest,
                    'backbone_grad_free':True,'GT_read':False,'target_data_read':False})
                path=OUT / 'queries' / f'{index:02}_{label}.pt'; save_tensor(path,record); outputs.append(path)
            print('REFERENCE_QUERY',index+1,4,row['key'],row['source'],'NO_UPDATE',flush=True)
            del fields,view,prompt,view_prompt,frames,record; gc.collect()
        save_once(OUT / 'COMPLETE.json',{'status':'completed_reference_sensitivity_no_update','queries':4,'parents':4,
            'signal_cases':8,'pins':{str(p):sha(p) for p in outputs},'optimizer_steps':0,
            'adapter_unchanged':digest,'source_GT_read':False,'target_data_read':False})
        status='completed'
    except BaseException:
        status='failed'; failure=traceback.format_exc(); save_once(OUT / 'FAILURE.json',{'time':time.time(),'failure':failure}); raise
    finally:
        seconds=time.monotonic()-began
        save_once(RECEIPT,{'status':status,'seconds':seconds,'prior_seconds':prior,'cumulative_seconds':prior+seconds,
            'failure':failure,'cumulative_cap_seconds':None,'peak_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None})
        lease.close()


def launch():
    assert not RECEIPT.exists() and not (OUT / 'STARTED.json').exists()
    began=time.monotonic(); child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=ROOT)
    wall=time.monotonic()-began; worker=read(RECEIPT)['seconds'] if RECEIPT.exists() else 0.
    save_once(BASE / 'receipts/source_reference_estimator_wrapper_v1.json',{'status':'completed' if child.returncode==0 else 'failed',
        'seconds':max(0.,wall-worker),'child_wall_seconds':wall,'worker_seconds':worker,'returncode':child.returncode,
        'scope':'nonoverlapping launch/import/finalization overhead','cumulative_cap_seconds':None})
    raise SystemExit(child.returncode)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['register','run','launch'])
    {'register':register,'run':run,'launch':launch}[parser.parse_args().action]()
