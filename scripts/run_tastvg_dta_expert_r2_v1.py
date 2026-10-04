"""Finite matched R2; deploy runs in its own GT-guarded CPU process."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,time,json,traceback,subprocess,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,save,status
from vg_tta.tastvg_dta_oracle_v1 import fit_query,paired_summary,tiou,STEPS,BETA
from vg_tta.tastvg_dta_expert_r2_v1 import deploy_choice,oracle_choice,validate_support,read_guard
BASE=ROOT/'artifacts/tastvg_dta_expert_r2_v1';PUB=ROOT/'results/tastvg_dta_expert_r2/2026-10-04'
R1=ROOT/'artifacts/tastvg_dta_oracle_r1_v1';R1PUB=ROOT/'results/tastvg_dta_oracle_r1/2026-10-04'
LAT=ROOT/'artifacts/tastvg_temporal_latent_quality_v1';POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
VIEW=ROOT/'artifacts/tastvg_current_correction_views_v1';PRE=ROOT/'artifacts/tastvg_temporal_boundary_support_v1'
OWN=['vg_tta/tastvg_dta_expert_r2_v1.py','vg_tta/tastvg_dta_oracle_v1.py',
     'scripts/run_tastvg_dta_expert_r2_v1.py','scripts/test_tastvg_dta_expert_r2_v1.py',
     'protocols/tastvg_dta_expert_r2_v1.md','docs/tastvg_dta_expert_r2_v1/EXECUTION.md']
ARMS=['EDeploy','EOracle'];DATASETS=['vidstg','hc2'];LR=.01
def prefix(c):return f'{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}'
def key(c):return '/'.join(str(c[k]) for k in ['dataset','split','condition','order','arrival'])
def st(s,**kw):status(BASE/'STATUS.json',dict(status=s,pid=os.getpid(),time=time.time(),**kw))
def load(p):return torch.load(p,map_location='cpu',weights_only=False,mmap=True)
def checked(p):assert sha(p)==read(p.with_suffix('.json'))['sha256'];return load(p)
def dh(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify(labels=False):
    z=read(BASE/'RUNTIME_LOCK.json')
    for p,h in {**z['code'],**z['inputs'],**(z['oracle_label_inputs'] if labels else {})}.items():assert sha(ROOT/p)==h,p
    return z

def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists(),'Do not restart the completed/prepared study'
    BASE.mkdir(parents=True,exist_ok=True);PUB.mkdir(parents=True,exist_ok=True)
    assert read(R1/'FINAL_COMPLETION.json')['GitHub_commit']=='885762286dd75b806fae34b0bb712583d5088d84'
    cfg1=read(R1PUB/'CONFIG.json');source=read(R1/'SOURCE_SELECTION_BARRIER.json');assert source['choices']=={'vidstg':LR,'hc2':LR}
    assert cfg1['K']==STEPS==3 and cfg1['beta']==BETA==1 and cfg1['head_parameters']==514
    oldlock=read(R1/'RUNTIME_LOCK.json');cells=read(R1/'COHORT.json')['cells'];assert len(cells)==1152 and sum(c['scheduled'] for c in cells)==288
    inputs={};labels={};supports={};selection=[];cache_seen=set();counts=[]
    def pin(p):inputs[str(p.relative_to(ROOT))]=sha(p)
    for p in [R1/'COHORT.json',R1/'RUNTIME_LOCK.json',R1/'SOURCE_SELECTION_BARRIER.json',R1PUB/'CONFIG.json',ROOT/'methods/CURRENT_METHOD.json']:pin(p)
    for ds in DATASETS:
        pin(R1/ds/'HEAD.pt');pin(VIEW/ds/'PLAN.json')
        for sp in ['search','confirm']:
            p=POOL/ds/f'GT_LABELS_{sp}.json';labels[str(p.relative_to(ROOT))]=oldlock['inputs'][str(p.relative_to(ROOT))]
        for c in [c for c in cells if c['dataset']==ds]:
            pin(PRE/ds/'predictions'/f'{prefix(c)}.json')
            if not c['scheduled']:continue
            f=LAT/ds/'target_features'/f'{prefix(c)}.pt';z=checked(f);assert z['GT_read'] is False and z['A_spatial_bitwise_parity']
            assert z['A_state_pre_sha256']==c['pre_sha'] and z['A_state_post_sha256']==c['post_sha'];pin(f);pin(f.with_suffix('.json'))
            rf=POOL/ds/'capture'/c['condition']/f'{c["parent"]:05}.json';r=read(rf);assert r['pixel_sha256']==c['pixel_sha256'];pin(rf);pin(POOL/ds/r['cache'])
            assert inputs[str((POOL/ds/r['cache']).relative_to(ROOT))]==r['sha256']
            ef=POOL/ds/'experts/temporal'/c['condition']/f'{c["parent"]:05}.json';e=read(ef);ep=POOL/ds/'experts'/e['cache'];pin(ef);pin(ep)
            assert inputs[str(ep.relative_to(ROOT))]==e['cache_sha256'] and e['pixel_sha256']==c['pixel_sha256'] and not e['GT_read']
            expert=load(ep);assert expert['GT_read'] is False
            p,conf=validate_support(expert['proposals'],expert['proposal_confidence']);ids=z['frame_ids']
            assert p.min()>=ids[0]-1e-4 and p.max()<=ids[-1]+1+1e-4
            choice=deploy_choice(p,conf);k=key(c);supports[k]=dict(proposals=p.tolist(),confidence=conf.tolist(),
                expert_cache_sha256=e['cache_sha256'],input_sha256=expert['input_sha256'],pixel_sha256=c['pixel_sha256'],duration=expert['duration'],deploy=choice)
            cache_seen.add(str(ep));counts.append(len(p));selection.append(dict(cell_key=k,dataset=ds,source_id=c['parent'],split=c['split'],
                proposal_count=len(p),selected_index=choice['index'],selected_confidence=choice['confidence'],
                teacher_interval_sha256=dh(choice['interval']),expert_input_sha256=expert['input_sha256'],GT_used=False))
    write(BASE/'COHORT.json',dict(cells=cells));pin(BASE/'COHORT.json')
    write(BASE/'EXPERT_SUPPORT.json',supports);pin(BASE/'EXPERT_SUPPORT.json')
    write(PUB/'DEPLOY_SELECTION.json',selection);pin(PUB/'DEPLOY_SELECTION.json')
    cfg=dict(version='tastvg_dta_expert_r2_v1',predecessor_commit='885762286dd75b806fae34b0bb712583d5088d84',
        checkpoints=cfg1['checkpoints'],head_parameters=514,K=3,beta=1.,lr={ds:LR for ds in DATASETS},
        source_lr_inherited=True,source_selection_barrier_sha256=sha(R1/'SOURCE_SELECTION_BARRIER.json'),
        sigma=cfg1['sigma'],probabilities=cfg1['probabilities'],readout=cfg1['readout'],head_arithmetic=cfg1['head_arithmetic'],
        shared_R1_fit_code_sha256=sha(ROOT/'vg_tta/tastvg_dta_oracle_v1.py'),temporal_state=cfg1['temporal_state'],
        spatial_A_fixed=True,sampling=cfg1['sampling'],conditions=cfg1['conditions'],schedule_fraction=.25,
        target_design_sources=cfg1['target_design_sources'],target_expert_sources=cfg1['target_expert_sources'],target_design_arrivals=1152,
        query_arm_adaptations=576,scheduled_cells=288,unchanged_nonexpert_cells=864,history_exposure=True,
        teacher_shapes='single Gaussian; only center changes',EDeploy='raw UVTG proposal confidence first-argmax',
        EOracle='GT continuous physical tIoU first-argmax on identical raw UVTG support',
        empty_support_policy='preparation error; no fallback',raw_proposals_preserved=True,unique_expert_cache_files=len(cache_seen),
        proposal_count_range=[min(counts),max(counts)],Gaussian_centers_fractional_unclipped=True,
        direct_teacher_metric_endpoints='official dense integer truncation, separately from continuous support tIoU',
        target_tuning=False,source_retuning=False,mixture=False,PoE=False,confidence_weighting=False,gate=False,
        bootstrap_draws=10000,bootstrap_seed=20261004,production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        new_GPU_calls=0,new_backbone_calls=0,new_expert_calls=0,new_suffix_replays=0,R2b_started=False,R3_started=False,time=time.time())
    write(BASE/'CONFIG.json',cfg);write(PUB/'CONFIG.json',cfg);pin(BASE/'CONFIG.json')
    write(BASE/'RUNTIME_LOCK.json',dict(code={p:sha(ROOT/p) for p in OWN},inputs=inputs,oracle_label_inputs=labels,time=time.time()))
    write(BASE/'PREPARATION.json',dict(status='pass',GT_centers_read=False,support_cells=288,unique_cache_files=len(cache_seen),
        deploy_selection_sha256=sha(PUB/'DEPLOY_SELECTION.json'),time=time.time()))
    st('prepared_pending_deploy',target_GT_read=False)

def adapt(arm):
    assert arm in ARMS
    if arm=='EDeploy':sys.addaudithook(read_guard)
    verify(labels=arm=='EOracle');tick=time.monotonic();cohort=read(BASE/'COHORT.json')['cells'];support=read(BASE/'EXPERT_SUPPORT.json')
    if arm=='EOracle':
        sealed=read(BASE/'DEPLOY_PREDICTION_BARRIER.json');assert sealed['status']=='sealed' and sealed['GT_read'] is False
        labels={ds:{sp:read(POOL/ds/f'GT_LABELS_{sp}.json') for sp in ['search','confirm']} for ds in DATASETS}
        choices={key(c):oracle_choice(support[key(c)]['proposals'],support[key(c)]['confidence'],labels[c['dataset']][c['split']][str(c['parent'])]['span']) for c in cohort if c['scheduled']}
        write(BASE/'ORACLE_SELECTION.json',choices)
        write(BASE/'ORACLE_SELECTION_BARRIER.json',dict(status='sealed',time=time.time(),GT_used=True,
            deploy_prediction_barrier_sha256=sha(BASE/'DEPLOY_PREDICTION_BARRIER.json'),selection_sha256=sha(BASE/'ORACLE_SELECTION.json')))
    else:choices={k:v['deploy'] for k,v in support.items()}
    files={};traces=[];count=0
    for ds in DATASETS:
        head=load(R1/ds/'HEAD.pt')
        for c in [c for c in cohort if c['dataset']==ds and c['scheduled']]:
            k=key(c);z=checked(LAT/ds/'target_features'/f'{prefix(c)}.pt');rf=read(POOL/ds/'capture'/c['condition']/f'{c["parent"]:05}.json');data=load(POOL/ds/rf['cache'])
            assert z['frame_ids']==data['frame_ids'] and data['pixel_sha256']==c['pixel_sha256']
            a=fit_query(z['hidden'],z['frame_ids'],head,choices[k]['interval'],LR,data['records'])
            a['GT_supervised']=arm=='EOracle';a.update(arm=arm,teacher_center=choices[k]['interval'],teacher_proposal_index=choices[k]['index'],
                cell_key=k,dataset=ds,split=c['split'],source_id=c['parent'],frame_ids=z['frame_ids'],
                A_state_pre_sha256=c['pre_sha'],A_state_post_sha256=c['post_sha'],pixel_sha256=c['pixel_sha256'])
            assert a['before']['indices']==data['prediction']['indices']
            p=BASE/ds/('deploy_runs' if arm=='EDeploy' else 'oracle_runs')/f'{prefix(c)}.pt';save(p,a);files[str(p.relative_to(BASE))]=sha(p)
            trace=[]
            for t in a['trace']:
                u={('teacher_KL' if j=='GT_KL' else 'after_teacher_KL' if j=='after_GT_KL' else j):v for j,v in t.items() if j!='prediction'}
                u['interval_normalized']=[(v-z['frame_ids'][0])/(z['frame_ids'][-1]+1-z['frame_ids'][0]) for v in t['prediction']['physical_interval']];trace.append(u)
            traces.append(dict(cell_key=k,arm=arm,dataset=ds,split=c['split'],source_id=c['parent'],condition=c['condition'],order=c['order'],arrival=c['arrival'],
                GT_supervised=arm=='EOracle',episodic_reset=True,lr=LR,trace=trace,teacher_proposal_index=choices[k]['index'],
                teacher_interval_sha256=dh(choices[k]['interval']),proposal_count=len(support[k]['proposals']),
                head_initial_sha256=a['head_initial_sha256'],head_final_sha256=a['head_final_sha256'],
                A_state_pre_sha256=c['pre_sha'],A_state_post_sha256=c['post_sha'],pixel_sha256=c['pixel_sha256']))
            count+=1
            if count%24==0:st(arm+'_running',done=count,total=288,target_GT_read=arm=='EOracle');print('R2',arm,count,288,flush=True)
    assert count==288
    write(PUB/(arm+'_TRACES.json'),traces)
    barrier=dict(status='sealed',arm=arm,time=time.time(),pid=os.getpid(),GT_read=arm=='EOracle',cells=288,files=files,
        trace_sha256=sha(PUB/(arm+'_TRACES.json')),runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        CPU_wall_seconds=time.monotonic()-tick,backward_calls=864,GT_guard_active=arm=='EDeploy',new_expert_calls=0,
        new_backbone_calls=0,new_suffix_calls=0,spatial_unchanged=True,CUDA_initialized=torch.cuda.is_initialized())
    name='DEPLOY' if arm=='EDeploy' else 'ORACLE';write(BASE/(name+'_PREDICTION_BARRIER.json'),barrier);write(PUB/(name+'_PREDICTION_BARRIER.json'),{k:v for k,v in barrier.items() if k!='files'})
    st(arm+'_sealed_pending_next',done=288)
    if arm=='EOracle':
        d=read(BASE/'DEPLOY_PREDICTION_BARRIER.json');assert d['time']<read(BASE/'ORACLE_SELECTION_BARRIER.json')['time']<barrier['time']
        gb=dict(status='sealed',time=time.time(),query_arm_adaptations=576,
            deploy_barrier_sha256=sha(BASE/'DEPLOY_PREDICTION_BARRIER.json'),oracle_barrier_sha256=sha(BASE/'ORACLE_PREDICTION_BARRIER.json'),
            deploy_GT_read=False,oracle_GT_read=True,all_predictions_sealed_before_dense=True)
        write(BASE/'GLOBAL_PREDICTION_BARRIER.json',gb);write(PUB/'GLOBAL_PREDICTION_BARRIER.json',gb)

def score():
    verify(labels=True);tick=time.monotonic();assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['status']=='sealed'
    from scripts.tastvg_correction_views_common_v1 import oldcell
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    cells=read(BASE/'COHORT.json')['cells'];support=read(BASE/'EXPERT_SUPPORT.json');choices=read(BASE/'ORACLE_SELECTION.json')
    r1={r['cell_key']:r for r in read(R1PUB/'ROWS.json')};rows=[];selection=[];maxerror=0.;traces={arm:{r['cell_key']:r for r in read(PUB/(arm+'_TRACES.json'))} for arm in ARMS}
    for ds in DATASETS:
        plan=read(VIEW/ds/'PLAN.json');labels={sp:read(POOL/ds/f'GT_LABELS_{sp}.json') for sp in ['search','confirm']}
        for c in [c for c in cells if c['dataset']==ds]:
            k=key(c);r=dict(r1[k]);old=oldcell(ds,c['split'],c['condition'],c['order'],c['arrival']);assert old['pre_sha']==c['pre_sha'] and old['post_sha']==c['post_sha']
            row=plan['rows'][c['parent']];g=labels[c['split']][str(c['parent'])];truth={int(j):v for j,v in g['truth'].items()};boxes=old['slow']['boxes'];dense=DenseTube(boxes,row,truth,g['span'],ds=='hc2')
            for arm in ARMS:
                r[arm+'_GT_supervised']=bool(c['scheduled'] and arm=='EOracle')
                if c['scheduled']:
                    name='DEPLOY' if arm=='EDeploy' else 'ORACLE';p=BASE/ds/('deploy_runs' if arm=='EDeploy' else 'oracle_runs')/f'{prefix(c)}.pt';bar=read(BASE/(name+'_PREDICTION_BARRIER.json'));assert sha(p)==bar['files'][str(p.relative_to(BASE))]
                    a=load(p);choice=support[k]['deploy'] if arm=='EDeploy' else choices[k]
                    assert a['teacher_center']==choice['interval'] and a['A_state_pre_sha256']==c['pre_sha'] and a['A_state_post_sha256']==c['post_sha']
                    intervals={arm:a['after']['physical_interval'],arm+'_direct':choice['interval'],arm+'_map':a['teacher_MAP']['physical_interval']}
                    tr=traces[arm][k]['trace']
                    for t,pt in zip(tr,a['trace']):m=dense.score(pt['prediction']['physical_interval']);t['v']=m['v'];t['t']=m['t']
                    r.update({arm+'_loss_before':a['trace'][0]['loss'],arm+'_loss_after':a['trace'][-1]['after_loss'],arm+'_displacement':a['trace'][-1]['arrival_displacement'],arm+'_interval_changed':a['after']['indices']!=a['before']['indices']})
                    selection.append(dict(cell_key=k,dataset=ds,split=c['split'],source_id=c['parent'],condition=c['condition'],order=c['order'],arm=arm,
                        proposal_count=len(support[k]['proposals']),selected_index=choice['index'],confidence=choice['confidence'],
                        continuous_teacher_tIoU=tiou(choice['interval'],g['span']),GT_used_in_selection=arm=='EOracle'))
                else:intervals={arm:[row['frame_ids'][read(PRE/ds/'predictions'/f'{prefix(c)}.json')['A_indices'][0]],row['frame_ids'][read(PRE/ds/'predictions'/f'{prefix(c)}.json')['A_indices'][1]]+1]}
                for name,interval in intervals.items():
                    m=dense.score(interval);ref=official(boxes,row,truth,g['span'],interval,ds)
                    for metric in ['v','t','s']:err=abs(m[metric]-ref[metric]);assert err<1e-10;maxerror=max(maxerror,err)
                    r.update({name+'_'+metric:m[metric] for metric in ['v','t']})
                if not c['scheduled']:
                    for name in [arm+'_direct',arm+'_map']:
                        for metric in ['v','t']:r[name+'_'+metric]=r['A8_'+metric]
                for b in ['N','A8','R1']:
                    for metric in ['v','t']:r[arm+'_minus_'+b+'_'+metric]=r[arm+'_'+metric]-r[b+'_'+metric]
                    delta=r[arm+'_minus_'+b+'_v'];r[arm+'_gross_gain_'+b]=max(delta,0.);r[arm+'_gross_loss_'+b]=max(-delta,0.)
            for metric in ['v','t']:r['EOracle_minus_EDeploy_'+metric]=r['EOracle_'+metric]-r['EDeploy_'+metric]
            rows.append(r)
    assert len(rows)==1152 and len(selection)==576
    fields=[a+'_'+m for a in ['N','A8','R1','EDeploy','EOracle','EDeploy_direct','EOracle_direct','EDeploy_map','EOracle_map'] for m in ['v','t']]
    fields += [a+'_minus_'+b+'_'+m for a in ARMS for b in ['N','A8','R1'] for m in ['v','t']]
    fields += ['EOracle_minus_EDeploy_v','EOracle_minus_EDeploy_t']+[a+'_gross_'+x+'_'+b for a in ARMS for b in ['N','A8'] for x in ['gain','loss']]
    summary={}
    for ds in DATASETS:
        summary[ds]={}
        for sp in ['search','confirm']:
            rr=[r for r in rows if r['dataset']==ds and r['split']==sp];panels={}
            predicates=dict(expert_corrupt=lambda r:r['expert_scheduled'] and r['condition']!='clean',expert_clean=lambda r:r['expert_scheduled'] and r['condition']=='clean',
                expert_all=lambda r:r['expert_scheduled'],flow_corrupt=lambda r:r['condition']!='clean',flow_clean=lambda r:r['condition']=='clean',nonexpert_corrupt=lambda r:not r['expert_scheduled'] and r['condition']!='clean')
            for name,test in predicates.items():
                q=[r for r in rr if test(r)];z=paired_summary(q,fields)
                z['negative_tails']={a:{b:dict(severe_harm=sum(r[a+'_minus_'+b+'_v']<-.05 for r in q),harm=sum(r[a+'_minus_'+b+'_v']<-1e-12 for r in q),gain=sum(r[a+'_minus_'+b+'_v']>1e-12 for r in q),
                    baseline_good_destroyed_at_03=sum(r[b+'_v']>=.3 and r[a+'_v']<.3 for r in q),baseline_bad_rescued_at_03=sum(r[b+'_v']<.3 and r[a+'_v']>=.3 for r in q)) for b in ['N','A8']} for a in ARMS}
                z['orders']={o:paired_summary([r for r in q if r['order']==o],fields) for o in ['order1','order2']}
                z['loss_down_task_down']={a:sum(r.get(a+'_loss_after',0)<r.get(a+'_loss_before',0) and r[a+'_minus_N_v']<-1e-12 for r in q) for a in ARMS}
                panels[name]=z
            summary[ds][sp]=panels
    write(PUB/'ROWS.json',rows);write(PUB/'TEACHER_SELECTION_SCORED.json',selection);write(PUB/'SUMMARY.json',summary)
    for arm in ARMS:write(PUB/(arm+'_TRACES_SCORED.json'),list(traces[arm].values()))
    resources=dict(deploy_CPU_wall_seconds=read(BASE/'DEPLOY_PREDICTION_BARRIER.json')['CPU_wall_seconds'],oracle_CPU_wall_seconds=read(BASE/'ORACLE_PREDICTION_BARRIER.json')['CPU_wall_seconds'],
        dense_score_CPU_wall_seconds=time.monotonic()-tick,query_arm_adaptations=576,backward_calls=1728,
        new_GPU_calls=0,GPU_initialized=torch.cuda.is_initialized(),new_backbone_calls=0,new_suffix_replays=0,new_expert_calls=0,spatial_updates=0,temporal_persistence_writes=0,
        private_run_bytes=sum(p.stat().st_size for p in BASE.glob('*/*_runs/*.pt')),independent_dense_max_error=maxerror,
        stage_wall_times_exclude_prephase_hashes_loading_development_audits=True,time=time.time())
    write(PUB/'RESOURCES.json',resources);write(BASE/'SCORE_COMPLETION.json',dict(status='completed_pending_root_audit_publication',rows=1152,query_arm_adaptations=576,time=time.time(),rows_sha256=sha(PUB/'ROWS.json'),summary_sha256=sha(PUB/'SUMMARY.json')))
    st('completed_pending_root_audit_publication',rows=1152,adaptations=576)

def main():
    torch.set_num_threads(2);torch.manual_seed(20261004)
    if len(sys.argv)>1:return {'prepare':prepare,'deploy':lambda:adapt('EDeploy'),'oracle':lambda:adapt('EOracle'),'score':score}[sys.argv[1]]()
    BASE.mkdir(parents=True,exist_ok=True)
    for phase in ['prepare','deploy','oracle','score']:
        with (BASE/(phase.upper()+'.log')).open('x') as out:
            p=subprocess.run([sys.executable,'-B',__file__,phase],stdout=out,stderr=subprocess.STDOUT)
        if p.returncode:st('failed',phase=phase,returncode=p.returncode);raise RuntimeError('R2 phase failed: '+phase)
        print('R2_STAGE_COMPLETED',phase,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise
