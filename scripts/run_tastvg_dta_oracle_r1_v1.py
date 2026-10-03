"""Finite CPU R1 oracle run. Target labels are deliberately supervised inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys, time, json, hashlib, collections, traceback
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, sha, save, status
from vg_tta.tastvg_dta_oracle_v1 import *
BASE=ROOT/'artifacts/tastvg_dta_oracle_r1_v1'
PUB=ROOT/'results/tastvg_dta_oracle_r1/2026-10-04'
LAT=ROOT/'artifacts/tastvg_temporal_latent_quality_v1'
PRE=ROOT/'artifacts/tastvg_temporal_boundary_support_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
VIEW=ROOT/'artifacts/tastvg_current_correction_views_v1'
DATASETS=['vidstg','hc2']
OWN=['vg_tta/tastvg_dta_oracle_v1.py','scripts/run_tastvg_dta_oracle_r1_v1.py',
     'scripts/test_tastvg_dta_oracle_r1_v1.py','protocols/tastvg_dta_oracle_r1_v1.md',
     'docs/tastvg_dta_oracle_r1_v1/EXECUTION.md']

def st(state, **extra):
    status(BASE/'STATUS.json',dict(status=state,time=time.time(),pid=os.getpid(),**extra))

def prefix(c): return f'{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}'
def key(c): return '/'.join(str(c[k]) for k in ['dataset','split','condition','order','arrival'])
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def checked(f):
    receipt=read(f.with_suffix('.json')); assert sha(f)==receipt['sha256'],f
    return torch.load(f,map_location='cpu',weights_only=False,mmap=True)

def verify():
    lock=read(BASE/'RUNTIME_LOCK.json')
    for f,h in {**lock['code'],**lock['inputs']}.items(): assert sha(ROOT/f)==h,f
    return lock

def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists(), 'Do not repeat the experiment'
    torch.set_num_threads(2); torch.manual_seed(20261004)
    BASE.mkdir(parents=True,exist_ok=True); PUB.mkdir(parents=True,exist_ok=True)
    inputs={}; cellrows=read(LAT/'COHORT.json')['cells']
    assert len(cellrows)==1152 and sum(c['scheduled'] for c in cellrows)==288
    source=read(LAT/'SOURCE_INPUTS.json'); selected_sources={}; checkpoints={}
    for ds in DATASETS:
        selected_sources[ds]=[r for r in source[ds] if r['split']=='validation']
        assert len(selected_sources[ds])==({'vidstg':31,'hc2':16}[ds])
        ck=ROOT/'checkpoints'/('TASTVG_VidSTG.pth' if ds=='vidstg' else 'TASTVG_HCSTVG2.pth')
        checkpoints[ds]=dict(path=str(ck.relative_to(ROOT)),sha256=sha(ck),key='model_ema')
        state=torch.load(ck,map_location='cpu',weights_only=False,mmap=True)['model_ema']
        head={k[len('temp_embed.layers.'):]:v.detach().cpu().float().clone()
              for k,v in state.items() if k.startswith('temp_embed.layers.')}
        assert set(head)=={'0.weight','0.bias','1.weight','1.bias'}
        assert sum(head[k].numel() for k in ['1.weight','1.bias'])==514
        save(BASE/ds/'HEAD.pt',head); inputs[str(ck.relative_to(ROOT))]=checkpoints[ds]['sha256']
        inputs[str((BASE/ds/'HEAD.pt').relative_to(ROOT))]=sha(BASE/ds/'HEAD.pt')
        for r in selected_sources[ds]:
            f=LAT/ds/'source_features'/f'{r["index"]:04}.pt'; z=checked(f)
            assert z['split']=='validation' and z['head_exact'] and z['GT_read'] is False
            inputs[str(f.relative_to(ROOT))]=sha(f);inputs[str(f.with_suffix('.json').relative_to(ROOT))]=sha(f.with_suffix('.json'))
        for c in cellrows:
            if c['dataset']!=ds: continue
            f=PRE/ds/'predictions'/f'{prefix(c)}.json';inputs[str(f.relative_to(ROOT))]=sha(f)
            if c['scheduled']:
                f=LAT/ds/'target_features'/f'{prefix(c)}.pt'; z=checked(f)
                assert z['GT_read'] is False and z['A_spatial_bitwise_parity'] and z['source_A_temporal_bitwise_parity']
                inputs[str(f.relative_to(ROOT))]=sha(f);inputs[str(f.with_suffix('.json').relative_to(ROOT))]=sha(f.with_suffix('.json'))
                rf=POOL/ds/'capture'/c['condition']/f'{c["parent"]:05}.json'; rec=read(rf)
                cf=POOL/ds/rec['cache'];assert rec['pixel_sha256']==c['pixel_sha256'] and sha(cf)==rec['sha256']
                inputs[str(rf.relative_to(ROOT))]=sha(rf);inputs[str(cf.relative_to(ROOT))]=rec['sha256']
        for f in [VIEW/ds/'PLAN.json',POOL/ds/'GT_LABELS_search.json',POOL/ds/'GT_LABELS_confirm.json',
                  LAT/ds/'SOURCE_FEATURE_BARRIER.json',LAT/ds/'TARGET_FEATURE_BARRIER.json']:
            inputs[str(f.relative_to(ROOT))]=sha(f)
    for f in [LAT/'SOURCE_INPUTS.json',LAT/'SOURCE_GT.json',LAT/'COHORT.json',ROOT/'methods/CURRENT_METHOD.json']:
        inputs[str(f.relative_to(ROOT))]=sha(f)
    write(BASE/'COHORT.json',dict(cells=cellrows,source_validation_indices={d:[r['index'] for r in selected_sources[d]] for d in DATASETS}))
    inputs[str((BASE/'COHORT.json').relative_to(ROOT))]=sha(BASE/'COHORT.json')
    cfg=dict(version='tastvg_dta_oracle_r1_v1',stage='R1 supervised oracle only',predecessor_commit='e6037633c10773a57cf3983d5060b56f08fc96cc',
        checkpoints=checkpoints,head_parameters=514,effective_bias_invariance=True,scope='original temp_embed.layers.1 weight+bias',
        K=STEPS,beta=BETA,lr_grid=LR_GRID,lr_selection='per dataset source-validation mean physical tIoU after step3; exact tie lowest lr',
        source_validation_queries={d:len(selected_sources[d]) for d in DATASETS},source_training_queries_used=0,
        target_design_sources=dict(search=32,confirm=16),target_design_arrivals=1152,
        target_adaptation_cells=288,target_expert_sources=dict(vidstg=dict(search=16,confirm=8),hc2=dict(search=14,confirm=7)),
        nonexpert_cached_A_cells=864,conditions=['clean','frame_drop_5','frame_freeze_5','motion_blur_5','occlusion_5','exposure_5'],
        schedule_fraction=.25,sampling='unchanged original Paper48, two offsets',GT_used_in_target_teacher=True,
        GT_used_for_lr_selection='official-train-derived source validation only',target_selection=False,
        sigma='median physical spacing of each offset; one observed cell, unclipped GT coordinates',
        probabilities='full strict i<j joint for each original offset; averaged two losses',
        readout='original offset MAP FP32 logsoftmax and physical envelope',spatial_A_fixed=True,
        temporal_state='reset for each current query, after exactly three SGD steps, discard',
        beta_is_soft_penalty_not_hard_trust_region=True,lambda_R2_unused=.5,head_arithmetic='CPU FP32; FP64 probability/loss',
        original_CUDA_CPU_logit_atol=1e-4,exact_native_interval_parity_required=True,
        new_GPU_calls=0,new_backbone_calls=0,new_suffix_replays=0,new_expert_calls=0,
        history_exposure=True,production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        bootstrap_draws=10000,bootstrap_seed=20261004,time=time.time())
    write(PUB/'CONFIG.json',cfg)
    write(BASE/'RUNTIME_LOCK.json',dict(code={f:sha(ROOT/f) for f in OWN},inputs=inputs,time=time.time()))
    st('prepared_pending_source_lr_selection',target_GT_teacher_started=False)

def source_selection():
    verify();tick=time.monotonic();source=read(LAT/'SOURCE_INPUTS.json'); labels=read(LAT/'SOURCE_GT.json')
    rows=[];choices={}; summary={}
    for ds in DATASETS:
        head=torch.load(BASE/ds/'HEAD.pt',map_location='cpu',weights_only=False)
        queries=[r for r in source[ds] if r['split']=='validation']
        for r in queries:
            z=checked(LAT/ds/'source_features'/f'{r["index"]:04}.pt'); span=labels[ds][str(r['index'])]
            for lr in LR_GRID:
                a=fit_query(z['hidden'],z['frame_ids'],head,span,lr)
                assert a['before']['indices']==z['candidates'][0]['indices'],('source_native_parity',ds,r['index'])
                save(BASE/ds/'source_runs'/f'{r["index"]:04}_{lr:g}.pt',a)
                rows.append(dict(dataset=ds,source_id=r['index'],lr=lr,before_t=tiou(a['before']['physical_interval'],span),
                    after_t=tiou(a['after']['physical_interval'],span),teacher_MAP_t=tiou(a['teacher_MAP']['physical_interval'],span),
                    objective_before=a['trace'][0]['loss'],objective_after=a['trace'][-1]['after_loss'],
                    native_indices_exact=True,initial_head_sha256=a['head_initial_sha256'],final_head_sha256=a['head_final_sha256']))
            print('SOURCE_R1',ds,r['index'],flush=True)
        path=[dict(lr=lr,queries=len(queries),mean_after_tIoU=float(np.mean([r['after_t'] for r in rows if r['dataset']==ds and r['lr']==lr])),
                   mean_before_tIoU=float(np.mean([r['before_t'] for r in rows if r['dataset']==ds and r['lr']==lr]))) for lr in LR_GRID]
        at=min(range(len(path)),key=lambda j:(-path[j]['mean_after_tIoU'],path[j]['lr']))
        choices[ds]=path[at]['lr'];summary[ds]=dict(path=path,selected_index=at,selected_lr=choices[ds])
    write(PUB/'SOURCE_VALIDATION_ROWS.json',rows);write(PUB/'SOURCE_SELECTION.json',summary)
    barrier=dict(status='both source choices frozen',choices=choices,time=time.time(),target_GT_teacher_started=False,
        source_rows_sha256=sha(PUB/'SOURCE_VALIDATION_ROWS.json'),selection_sha256=sha(PUB/'SOURCE_SELECTION.json'),
        CPU_wall_seconds=time.monotonic()-tick,query_configurations=len(rows),backward_calls=len(rows)*STEPS)
    write(BASE/'SOURCE_SELECTION_BARRIER.json',barrier);write(PUB/'SOURCE_SELECTION_BARRIER.json',barrier)
    st('source_selection_sealed_pending_target',choices=choices,target_GT_teacher_started=False)

def target_adapt():
    verify();tick=time.monotonic();bar=read(BASE/'SOURCE_SELECTION_BARRIER.json');cohort=read(BASE/'COHORT.json')['cells']
    rows=[];files={}; maxerr=0.;count=0
    for ds in DATASETS:
        head=torch.load(BASE/ds/'HEAD.pt',map_location='cpu',weights_only=False)
        cfg=read(PUB/'CONFIG.json');labels={sp:read(POOL/ds/f'GT_LABELS_{sp}.json') for sp in ['search','confirm']}
        for c in [c for c in cohort if c['dataset']==ds and c['scheduled']]:
            f=LAT/ds/'target_features'/f'{prefix(c)}.pt';z=checked(f)
            rf=read(POOL/ds/'capture'/c['condition']/f'{c["parent"]:05}.json')
            data=torch.load(POOL/ds/rf['cache'],map_location='cpu',weights_only=False,mmap=True)
            assert z['frame_ids']==data['frame_ids'] and c['pixel_sha256']==data['pixel_sha256']
            assert c['pre_sha']==z['A_state_pre_sha256'] and c['post_sha']==z['A_state_post_sha256']
            g=labels[c['split']][str(c['parent'])]
            a=fit_query(z['hidden'],z['frame_ids'],head,g['span'],bar['choices'][ds],data['records'])
            parts=offsets(z['frame_ids'],data['records']);assert a['before']['indices']==data['prediction']['indices']
            for p, old in zip(parts,data['prediction']['logits']):
                err=float((a['logits'][0][p]-old.reshape(-1,2).float()).abs().max());maxerr=max(maxerr,err)
                assert err<cfg['original_CUDA_CPU_logit_atol'],('CPU head parity',err)
            prior=read(PRE/ds/'predictions'/f'{prefix(c)}.json')
            assert a['before']['indices']==prior['native_indices']
            a.update(cell_key=key(c),dataset=ds,split=c['split'],source_id=c['parent'],frame_ids=z['frame_ids'],
                records=data['records'],feature_sha256=sha(f),A8_indices=prior['A_indices'],
                A_state_pre_sha256=z['A_state_pre_sha256'],A_state_post_sha256=z['A_state_post_sha256'],
                pixel_sha256=c['pixel_sha256'],source_selection_barrier_sha256=sha(BASE/'SOURCE_SELECTION_BARRIER.json'))
            out=BASE/ds/'target_runs'/f'{prefix(c)}.pt';save(out,a);files[str(out.relative_to(BASE))]=sha(out)
            trace=[]
            for t in a['trace']:
                q={k:v for k,v in t.items() if k!='prediction'}
                q['interval_normalized']=[(v-z['frame_ids'][0])/(z['frame_ids'][-1]+1-z['frame_ids'][0]) for v in t['prediction']['physical_interval']]
                trace.append(q)
            rows.append(dict(cell_key=key(c),dataset=ds,split=c['split'],source_id=c['parent'],condition=c['condition'],order=c['order'],arrival=c['arrival'],
                GT_supervised=True,lr=a['lr'],trace=trace,head_initial_sha256=a['head_initial_sha256'],head_final_sha256=a['head_final_sha256'],
                original_native_indices_exact=True,original_CUDA_CPU_logit_max_error=err,episodic_reset=True,
                A_state_pre_sha256=z['A_state_pre_sha256'],A_state_post_sha256=z['A_state_post_sha256'],pixel_sha256=c['pixel_sha256']))
            count+=1
            if count%12==0:
                st('target_oracle_running',dataset=ds,done=count,total=288,GT_supervised=True)
                print('TARGET_R1',ds,count,288,flush=True)
        del head
    assert count==288
    write(PUB/'ADAPTATION_TRACES.json',rows)
    barrier=dict(status='sealed',time=time.time(),GT_used_in_teacher=True,GT_used_for_target_selection=False,
        source_selection_time=bar['time'],files=files,adaptation_cells=count,max_CUDA_CPU_logit_error=maxerr,
        source_selection_barrier_sha256=sha(BASE/'SOURCE_SELECTION_BARRIER.json'),
        adaptation_traces_sha256=sha(PUB/'ADAPTATION_TRACES.json'),CPU_wall_seconds=time.monotonic()-tick,
        backward_calls=STEPS*count,spatial_state_unchanged=True,expert_calls=0,backbone_calls=0,suffix_replays=0)
    write(BASE/'PREDICTION_BARRIER.json',barrier)
    write(PUB/'PREDICTION_BARRIER.json',{k:v for k,v in barrier.items() if k!='files'})
    st('predictions_sealed_pending_dense_score',done=288,GT_supervised=True)

FIELDS=['N_v','A8_v','R1_v','N_t','A8_t','R1_t','R1_minus_N_v','R1_minus_N_t',
    'R1_minus_A8_v','R1_minus_A8_t','teacher_MAP_v','teacher_MAP_t','GT_time_v','GT_time_t',
    'R1_gross_gain_N','R1_gross_loss_N','R1_gross_gain_A8','R1_gross_loss_A8']

def score():
    verify();tick=time.monotonic();bar=read(BASE/'PREDICTION_BARRIER.json'); assert bar['status']=='sealed'
    from scripts.tastvg_correction_views_common_v1 import oldcell
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube, official
    cohort=read(BASE/'COHORT.json')['cells'];traces={r['cell_key']:r for r in read(PUB/'ADAPTATION_TRACES.json')};allrows=[];err=0.
    for ds in DATASETS:
        plan=read(VIEW/ds/'PLAN.json');labels={sp:read(POOL/ds/f'GT_LABELS_{sp}.json') for sp in ['search','confirm']}
        for c in [c for c in cohort if c['dataset']==ds]:
            old=oldcell(ds,c['split'],c['condition'],c['order'],c['arrival']);row=plan['rows'][c['parent']]
            assert old['pre_sha']==c['pre_sha'] and old['post_sha']==c['post_sha']
            boxes=old['slow']['boxes'];g=labels[c['split']][str(c['parent'])];truth={int(k):v for k,v in g['truth'].items()}
            dense=DenseTube(boxes,row,truth,g['span'],ds=='hc2');p=read(PRE/ds/'predictions'/f'{prefix(c)}.json');ids=row['frame_ids']
            A8=[ids[p['A_indices'][0]],ids[p['A_indices'][1]]+1]
            if c['scheduled']:
                f=BASE/ds/'target_runs'/f'{prefix(c)}.pt';assert sha(f)==bar['files'][str(f.relative_to(BASE))]
                a=torch.load(f,map_location='cpu',weights_only=False,mmap=True)
                assert a['A8_indices']==p['A_indices'] and a['A_state_pre_sha256']==c['pre_sha']
                intervals=dict(N=a['before']['physical_interval'],A8=A8,R1=a['after']['physical_interval'],teacher_MAP=a['teacher_MAP']['physical_interval'],GT_time=g['span'])
                trace=traces[key(c)]['trace']
            else:
                # Cached full-flow emulation, without rerunning online A.
                intervals=dict(N=A8,A8=A8,R1=A8,teacher_MAP=A8,GT_time=g['span']);trace=[]
            r=dict(cell_key=key(c),dataset=ds,split=c['split'],source_id=c['parent'],condition=c['condition'],order=c['order'],arrival=c['arrival'],
                expert_scheduled=c['scheduled'],GT_supervised=c['scheduled'],A_spatial_unchanged=True,
                A_state_pre_sha256=c['pre_sha'],A_state_post_sha256=c['post_sha'])
            for arm, interval in intervals.items():
                v=dense.score(interval);m=official(boxes,row,truth,g['span'],interval,ds)
                for k in ['v','t','s']:
                    error=abs(v[k]-m[k]);assert error<1e-10;err=max(err,error)
                r.update({f'{arm}_{k}':v[k] for k in ['v','t']})
            for baseline in ['N','A8']:
                for k in ['v','t']:r[f'R1_minus_{baseline}_{k}']=r['R1_'+k]-r[baseline+'_'+k]
                dv=r[f'R1_minus_{baseline}_v'];r[f'R1_gross_gain_{baseline}']=max(dv,0.);r[f'R1_gross_loss_{baseline}']=max(-dv,0.)
            if c['scheduled']:
                for t,pt in zip(trace,a['trace']):
                    m=dense.score(pt['prediction']['physical_interval']);t['v']=m['v'];t['t']=m['t']
                r.update(loss_before=a['trace'][0]['loss'],loss_after=a['trace'][-1]['after_loss'],
                         GT_KL_before=a['trace'][0]['GT_KL'],GT_KL_after=a['trace'][-1]['after_GT_KL'],
                         native_interval_changed=a['before']['indices']!=a['after']['indices'],
                         arrival_displacement=a['trace'][-1]['arrival_displacement'])
            allrows.append(r)
    write(PUB/'ROWS.json',allrows);write(PUB/'ADAPTATION_TRACES_SCORED.json',list(traces.values()))
    summary={}
    for ds in DATASETS:
        summary[ds]={}
        for sp in ['search','confirm']:
            rr=[r for r in allrows if r['dataset']==ds and r['split']==sp];panels={}
            for name,predicate in [('expert_corrupt',lambda r:r['expert_scheduled'] and r['condition']!='clean'),
              ('expert_clean',lambda r:r['expert_scheduled'] and r['condition']=='clean'),
              ('expert_all',lambda r:r['expert_scheduled']),('flow_corrupt',lambda r:r['condition']!='clean'),
              ('flow_clean',lambda r:r['condition']=='clean'),('nonexpert_corrupt',lambda r:not r['expert_scheduled'] and r['condition']!='clean')]:
                q=[r for r in rr if predicate(r)];ss=paired_summary(q,FIELDS)
                ss['negative_tails']={b:dict(severe_harm=sum(r[f'R1_minus_{b}_v']<-.05 for r in q),
                   harm=sum(r[f'R1_minus_{b}_v']<-1e-12 for r in q),gain=sum(r[f'R1_minus_{b}_v']>1e-12 for r in q),
                   baseline_good_destroyed_at_03=sum(r[b+'_v']>=.3 and r['R1_v']<.3 for r in q),
                   baseline_bad_rescued_at_03=sum(r[b+'_v']<.3 and r['R1_v']>=.3 for r in q)) for b in ['N','A8']}
                ss['orders']={o:paired_summary([r for r in q if r['order']==o],FIELDS) for o in ['order1','order2']}
                ss['objective_task_disagreement']=sum(r.get('loss_after',0)<r.get('loss_before',0) and r['R1_minus_N_v']<-1e-12 for r in q)
                panels[name]=ss
            summary[ds][sp]=panels
    write(PUB/'SUMMARY.json',summary)
    resources=dict(source_CPU_wall_seconds=read(BASE/'SOURCE_SELECTION_BARRIER.json')['CPU_wall_seconds'],
       target_adaptation_CPU_wall_seconds=bar['CPU_wall_seconds'],dense_score_CPU_wall_seconds=time.monotonic()-tick,
       source_queries=47,source_query_lr_configurations=235,target_query_configurations=288,
       source_backward_calls=705,target_backward_calls=864,new_backbone_calls=0,new_suffix_replays=0,new_expert_calls=0,
       GPU_initialized=torch.cuda.is_initialized(),fixed_head_parameters=514,spatial_updates=0,temporal_persistence_writes=0,
       worker_wall_is_not_kernel_time=True,private_run_bytes=sum(f.stat().st_size for f in BASE.glob('*/*_runs/*.pt')),
       independent_dense_max_error=err,time=time.time())
    write(PUB/'RESOURCES.json',resources)
    write(BASE/'SCORE_COMPLETION.json',dict(status='completed_pending_root_audit_publication',time=time.time(),rows=1152,expert=288,
        prediction_barrier_sha256=sha(BASE/'PREDICTION_BARRIER.json'),rows_sha256=sha(PUB/'ROWS.json'),summary_sha256=sha(PUB/'SUMMARY.json')))
    st('completed_pending_root_audit_publication',rows=1152,expert=288,GT_supervised=True)

def main():
    torch.set_num_threads(2);torch.manual_seed(20261004)
    if len(sys.argv)>1:
        {'prepare':prepare,'source':source_selection,'target':target_adapt,'score':score}[sys.argv[1]]()
    else:
        prepare();source_selection();target_adapt();score()

if __name__=='__main__':
    try:main()
    except Exception:
        st('engineering_failed',traceback=traceback.format_exc());raise
