"""Finite R2b: cached raw equal-mixture, GT-guarded fit, then dense scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,time,json,traceback,subprocess,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,save,status
from vg_tta.tastvg_dta_oracle_v1 import paired_summary,STEPS,BETA
from vg_tta.tastvg_dta_mixture_r2b_v1 import fit_mixture,validate_proposals,read_guard
BASE=ROOT/'artifacts/tastvg_dta_mixture_r2b_v1';PUB=ROOT/'results/tastvg_dta_mixture_r2b/2026-10-04'
R2=ROOT/'artifacts/tastvg_dta_expert_r2_v1';R2PUB=ROOT/'results/tastvg_dta_expert_r2/2026-10-04'
R1=ROOT/'artifacts/tastvg_dta_oracle_r1_v1';LAT=ROOT/'artifacts/tastvg_temporal_latent_quality_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3';VIEW=ROOT/'artifacts/tastvg_current_correction_views_v1'
PRE=ROOT/'artifacts/tastvg_temporal_boundary_support_v1'
OWN=['vg_tta/tastvg_dta_mixture_r2b_v1.py','vg_tta/tastvg_dta_oracle_v1.py','vg_tta/tastvg_dta_expert_r2_v1.py',
     'scripts/run_tastvg_dta_mixture_r2b_v1.py','scripts/test_tastvg_dta_mixture_r2b_v1.py',
     'protocols/tastvg_dta_mixture_r2b_v1.md','docs/tastvg_dta_mixture_r2b_v1/EXECUTION.md']
DATASETS=['vidstg','hc2'];LR=.01;BASELINES=['N','A8','R1','EDeploy','EOracle']
def prefix(c):return f'{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}'
def key(c):return '/'.join(str(c[k]) for k in ['dataset','split','condition','order','arrival'])
def st(s,**kw):status(BASE/'STATUS.json',dict(status=s,pid=os.getpid(),time=time.time(),**kw))
def load(p):return torch.load(p,map_location='cpu',weights_only=False,mmap=True)
def dh(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify(labels=False,baselines=False):
    z=read(BASE/'RUNTIME_LOCK.json')
    pins={**z['code'],**z['inputs'],**(z['post_seal_label_inputs'] if labels else {}),**(z['post_seal_baselines'] if baselines else {})}
    for p,h in pins.items():assert sha(ROOT/p)==h,p
    return z

def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists(),'Never restart a prepared/completed study'
    BASE.mkdir(parents=True,exist_ok=True);PUB.mkdir(parents=True,exist_ok=True)
    final=read(R2/'FINAL_COMPLETION.json');assert final['status']=='verified_complete' and final['GitHub_commit']=='324b3357dbd67b435d1fb507ae516a44ecf53961'
    old=read(R2/'RUNTIME_LOCK.json');cfg2=read(R2PUB/'CONFIG.json');cells=read(R2/'COHORT.json')['cells'];supports=read(R2/'EXPERT_SUPPORT.json')
    assert len(cells)==1152 and sum(c['scheduled'] for c in cells)==len(supports)==288
    assert cfg2['K']==STEPS==3 and cfg2['beta']==BETA==1 and cfg2['head_parameters']==514 and cfg2['lr']=={d:LR for d in DATASETS}
    inputs=dict(old['inputs']);inputs.update(old['code'])
    def pin(p):inputs[str(p.relative_to(ROOT))]=sha(p)
    for p in [R2/'RUNTIME_LOCK.json',R2/'COHORT.json',R2/'EXPERT_SUPPORT.json',R2/'FINAL_COMPLETION.json',R2PUB/'CONFIG.json']:pin(p)
    counts=[]
    for c in cells:
        if not c['scheduled']:continue
        z=supports[key(c)];p=validate_proposals(z['proposals']);assert z['pixel_sha256']==c['pixel_sha256'];counts.append(len(p))
    assert [min(counts),max(counts)]==cfg2['proposal_count_range']
    # Pin old public control hashes from the already verified export manifest,
    # without reading any metric content before the new prediction seal.
    manifest=read(R2/'PUBLIC_EXPORT_MANIFEST.json');post={f['path']:f['sha256'] for f in manifest['files']
        if f['path'].startswith('results/') and f['path'].endswith(('ROWS.json','EDeploy_TRACES_SCORED.json','EOracle_TRACES_SCORED.json'))}
    assert len(post)==3
    write(BASE/'COHORT.json',dict(cells=cells));pin(BASE/'COHORT.json')
    cfg={k:cfg2[k] for k in ['checkpoints','head_parameters','K','beta','lr','source_lr_inherited','source_selection_barrier_sha256','sigma',
        'probabilities','readout','head_arithmetic','temporal_state','spatial_A_fixed','sampling','conditions','schedule_fraction','target_design_sources',
        'target_expert_sources','history_exposure','empty_support_policy','raw_proposals_preserved','unique_expert_cache_files','proposal_count_range',
        'Gaussian_centers_fractional_unclipped','production_method_sha256']}
    cfg.update(version='tastvg_dta_mixture_r2b_v1',predecessor_commit=final['GitHub_commit'],teacher='all valid raw UVTG proposals equal mixture; duplicates/order preserved',
        component_normalization='each R1 Gaussian independently normalized on original offset strict i<j BEFORE arithmetic averaging',
        weights='1/M per raw row, including duplicates',teacher_joint_preserved=True,product_of_mixed_marginals=False,
        teacher_shapes='joint Gaussian mixture; unchanged component sigma and centers',target_design_arrivals=1152,scheduled_cells=288,
        new_query_arm_adaptations=288,reused_control_arms=['Native','A8','R1','EDeploy','EOracle'],unchanged_nonexpert_cells=864,
        source_retuning=False,target_tuning=False,mixture=True,PoE=False,confidence_weighting=False,top_K=False,deduplication=False,gate=False,new_view=False,
        bootstrap_draws=10000,bootstrap_seed=20261004,shared_R1_helpers_sha256=sha(ROOT/'vg_tta/tastvg_dta_oracle_v1.py'),
        teacher_MAP='FP64 joint MAP per original offset, then original envelope; diagnostic only',new_GPU_calls=0,new_backbone_calls=0,new_suffix_replays=0,
        new_expert_calls=0,temporal_persistence_writes=0,spatial_updates=0,R3_started=False,time=time.time())
    write(BASE/'CONFIG.json',cfg);write(PUB/'CONFIG.json',cfg);pin(BASE/'CONFIG.json')
    write(BASE/'RUNTIME_LOCK.json',dict(code={p:sha(ROOT/p) for p in OWN},inputs=inputs,
        post_seal_label_inputs=old['oracle_label_inputs'],post_seal_baselines=post,time=time.time()))
    write(BASE/'PREPARATION.json',dict(status='pass',GT_read=False,scored_control_content_read=False,scheduled_cells=288,
        unique_expert_cache_files=cfg['unique_expert_cache_files'],proposal_count_range=cfg['proposal_count_range'],time=time.time()))
    st('prepared_pending_mix',target_GT_read=False)

def adapt():
    sys.addaudithook(read_guard);verify();tick=time.monotonic();cells=read(BASE/'COHORT.json')['cells'];support=read(R2/'EXPERT_SUPPORT.json')
    files={};traces=[];count=0
    for ds in DATASETS:
        head=load(R1/ds/'HEAD.pt')
        for c in [c for c in cells if c['dataset']==ds and c['scheduled']]:
            k=key(c);f=LAT/ds/'target_features'/f'{prefix(c)}.pt';assert sha(f)==read(f.with_suffix('.json'))['sha256'];z=load(f)
            rf=read(POOL/ds/'capture'/c['condition']/f'{c["parent"]:05}.json');data=load(POOL/ds/rf['cache'])
            assert not z['GT_read'] and z['A_spatial_bitwise_parity'] and z['frame_ids']==data['frame_ids'] and data['pixel_sha256']==c['pixel_sha256']
            a=fit_mixture(z['hidden'],z['frame_ids'],head,support[k]['proposals'],LR,data['records'])
            a.update(arm='EMix',cell_key=k,dataset=ds,split=c['split'],source_id=c['parent'],frame_ids=z['frame_ids'],
                raw_proposals_sha256=dh(support[k]['proposals']),A_state_pre_sha256=c['pre_sha'],A_state_post_sha256=c['post_sha'],pixel_sha256=c['pixel_sha256'])
            assert a['before']['indices']==data['prediction']['indices']
            p=BASE/ds/'mix_runs'/f'{prefix(c)}.pt';save(p,a);files[str(p.relative_to(BASE))]=sha(p)
            trace=[]
            for s in a['trace']:
                t={j:v for j,v in s.items() if j!='prediction'}
                t['interval_normalized']=[(v-z['frame_ids'][0])/(z['frame_ids'][-1]+1-z['frame_ids'][0]) for v in s['prediction']['physical_interval']];trace.append(t)
            traces.append(dict(cell_key=k,arm='EMix',dataset=ds,split=c['split'],source_id=c['parent'],condition=c['condition'],order=c['order'],arrival=c['arrival'],
                GT_supervised=False,episodic_reset=True,lr=LR,trace=trace,proposal_count=a['proposal_count'],raw_proposals_sha256=a['raw_proposals_sha256'],
                teacher_logq_sha256=[dh(q.tolist()) for q in a['logq']],teacher_joint_entropy=[float(-(q.exp()*q).sum()) for q in a['logq']],
                teacher_mass=[float(q.exp().sum()) for q in a['logq']],head_initial_sha256=a['head_initial_sha256'],head_final_sha256=a['head_final_sha256'],
                A_state_pre_sha256=c['pre_sha'],A_state_post_sha256=c['post_sha'],pixel_sha256=c['pixel_sha256']))
            count+=1
            if count%24==0:st('EMix_running',done=count,total=288,target_GT_read=False);print('R2B_MIX',count,288,flush=True)
    assert count==288
    write(PUB/'MIX_TRACES.json',traces)
    bar=dict(status='sealed',arm='EMix',time=time.time(),pid=os.getpid(),GT_read=False,cells=288,files=files,
        trace_sha256=sha(PUB/'MIX_TRACES.json'),runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),CPU_wall_seconds=time.monotonic()-tick,
        backward_calls=864,GT_guard_active=True,new_expert_calls=0,new_backbone_calls=0,new_suffix_calls=0,spatial_unchanged=True,CUDA_initialized=torch.cuda.is_initialized())
    write(BASE/'MIX_PREDICTION_BARRIER.json',bar);write(PUB/'MIX_PREDICTION_BARRIER.json',{k:v for k,v in bar.items() if k!='files'})
    glob=dict(status='sealed',time=time.time(),new_query_arm_adaptations=288,mix_barrier_sha256=sha(BASE/'MIX_PREDICTION_BARRIER.json'),
        reused_R2_commit=read(PUB/'CONFIG.json')['predecessor_commit'],mix_GT_read=False,all_new_predictions_sealed_before_GT_scoring=True)
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',glob);write(PUB/'GLOBAL_PREDICTION_BARRIER.json',glob)
    st('mix_sealed_pending_score',done=288)

def teacher_diagnostic(a,raw,span):
    raw=np.asarray(raw,float);s,e=map(float,span);ov=np.maximum(0,np.minimum(raw[:,1],e)-np.maximum(raw[:,0],s));ts=ov/(raw[:,1]-raw[:,0]+e-s-ov)
    ds=[]
    for off,q in enumerate(a['logq']):
        ids=np.asarray(a['frame_ids'][off::2],float);i,j=np.triu_indices(len(ids),1);ss=ids[i];ee=ids[j]+1
        ov=np.maximum(0,np.minimum(ee,e)-np.maximum(ss,s));t=ov/(ee-ss+e-s-ov);mass=q.exp().numpy()
        ds.append(dict(joint_entropy=float(-(q.exp()*q).sum()),mass=float(mass.sum()),expected_joint_tIoU=float(mass@t),legal_spans=len(t)))
    return dict(raw_mean_tIoU=float(ts.mean()),raw_max_tIoU=float(ts.max()),raw_fraction_tIoU_ge_05=float(np.mean(ts>=.5)),
        raw_rows=len(raw),raw_unique_intervals=len(np.unique(raw,axis=0)),offsets=ds,
        note='GT diagnostics AFTER prediction seal; offset expectation is not an envelope prediction metric')

def score():
    assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['status']=='sealed';verify(labels=True,baselines=True);tick=time.monotonic()
    from scripts.tastvg_correction_views_common_v1 import oldcell
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    cells=read(BASE/'COHORT.json')['cells'];support=read(R2/'EXPERT_SUPPORT.json');oldrows={r['cell_key']:r for r in read(R2PUB/'ROWS.json')}
    traces={r['cell_key']:r for r in read(PUB/'MIX_TRACES.json')};rows=[];teachers=[];maxerror=0.;bar=read(BASE/'MIX_PREDICTION_BARRIER.json')
    meta=['cell_key','dataset','split','source_id','condition','order','arrival','expert_scheduled','A_spatial_unchanged','A_state_pre_sha256','A_state_post_sha256']
    for ds in DATASETS:
        plan=read(VIEW/ds/'PLAN.json');labels={sp:read(POOL/ds/f'GT_LABELS_{sp}.json') for sp in ['search','confirm']}
        for c in [c for c in cells if c['dataset']==ds]:
            k=key(c);oldr=oldrows[k];r={j:oldr[j] for j in meta};r['pixel_sha256']=c['pixel_sha256'];r['EMix_GT_supervised']=False
            for name in BASELINES+['teacher_MAP','GT_time','EDeploy_map','EOracle_map']:
                for m in ['v','t']:r[name+'_'+m]=oldr[name+'_'+m]
            old=oldcell(ds,c['split'],c['condition'],c['order'],c['arrival']);assert old['pre_sha']==c['pre_sha'] and old['post_sha']==c['post_sha']
            row=plan['rows'][c['parent']];g=labels[c['split']][str(c['parent'])];truth={int(j):v for j,v in g['truth'].items()};boxes=old['slow']['boxes'];dense=DenseTube(boxes,row,truth,g['span'],ds=='hc2')
            if c['scheduled']:
                p=BASE/ds/'mix_runs'/f'{prefix(c)}.pt';assert sha(p)==bar['files'][str(p.relative_to(BASE))];a=load(p)
                assert a['A_state_pre_sha256']==c['pre_sha'] and a['A_state_post_sha256']==c['post_sha'] and not a['GT_supervised']
                intervals={'EMix':a['after']['physical_interval'],'EMix_map':a['teacher_MAP']['physical_interval']}
                for t,pt in zip(traces[k]['trace'],a['trace']):
                    m=dense.score(pt['prediction']['physical_interval']);t['v']=m['v'];t['t']=m['t']
                r.update(EMix_loss_before=a['trace'][0]['loss'],EMix_loss_after=a['trace'][-1]['after_loss'],EMix_displacement=a['trace'][-1]['arrival_displacement'],
                    EMix_interval_changed=a['after']['indices']!=a['before']['indices'],proposal_count=a['proposal_count'])
                teachers.append(dict(cell_key=k,dataset=ds,split=c['split'],source_id=c['parent'],condition=c['condition'],order=c['order'],
                    **teacher_diagnostic(a,support[k]['proposals'],g['span'])))
            else:
                pr=read(PRE/ds/'predictions'/f'{prefix(c)}.json');iv=[row['frame_ids'][pr['A_indices'][0]],row['frame_ids'][pr['A_indices'][1]]+1]
                intervals={'EMix':iv,'EMix_map':iv}
            for name,iv in intervals.items():
                m=dense.score(iv);ref=official(boxes,row,truth,g['span'],iv,ds)
                for metric in ['v','t','s']:err=abs(m[metric]-ref[metric]);assert err<1e-10;maxerror=max(maxerror,err)
                r.update({name+'_'+metric:m[metric] for metric in ['v','t']})
            for b in BASELINES:
                for m in ['v','t']:r['EMix_minus_'+b+'_'+m]=r['EMix_'+m]-r[b+'_'+m]
                if b in ['N','A8','EDeploy']:
                    d=r['EMix_minus_'+b+'_v'];r['EMix_gross_gain_'+b]=max(d,0);r['EMix_gross_loss_'+b]=max(-d,0)
            rows.append(r)
    assert len(rows)==1152 and len(teachers)==288
    fields=[a+'_'+m for a in BASELINES+['EMix','EMix_map'] for m in ['v','t']]
    fields += ['EMix_minus_'+b+'_'+m for b in BASELINES for m in ['v','t']]
    fields += ['EMix_gross_'+x+'_'+b for b in ['N','A8','EDeploy'] for x in ['gain','loss']]
    summary={}
    predicates=dict(expert_corrupt=lambda r:r['expert_scheduled'] and r['condition']!='clean',expert_clean=lambda r:r['expert_scheduled'] and r['condition']=='clean',
        expert_all=lambda r:r['expert_scheduled'],flow_corrupt=lambda r:r['condition']!='clean',flow_clean=lambda r:r['condition']=='clean',nonexpert_corrupt=lambda r:not r['expert_scheduled'] and r['condition']!='clean')
    for ds in DATASETS:
        summary[ds]={}
        for sp in ['search','confirm']:
            rr=[r for r in rows if r['dataset']==ds and r['split']==sp];panels={}
            for name,test in predicates.items():
                q=[r for r in rr if test(r)];z=paired_summary(q,fields)
                z['negative_tails']={b:dict(severe_harm=sum(r['EMix_minus_'+b+'_v']<-.05 for r in q),harm=sum(r['EMix_minus_'+b+'_v']<-1e-12 for r in q),
                    gain=sum(r['EMix_minus_'+b+'_v']>1e-12 for r in q),baseline_good_destroyed_at_03=sum(r[b+'_v']>=.3 and r['EMix_v']<.3 for r in q),
                    baseline_bad_rescued_at_03=sum(r[b+'_v']<.3 and r['EMix_v']>=.3 for r in q)) for b in ['N','A8','EDeploy']}
                z['orders']={o:paired_summary([r for r in q if r['order']==o],fields) for o in ['order1','order2']}
                z['loss_down_task_down']=sum(r.get('EMix_loss_after',0)<r.get('EMix_loss_before',0) and r['EMix_minus_N_v']<-1e-12 for r in q)
                panels[name]=z
            summary[ds][sp]=panels
    write(PUB/'ROWS.json',rows);write(PUB/'TEACHER_DIAGNOSTICS.json',teachers);write(PUB/'SUMMARY.json',summary);write(PUB/'MIX_TRACES_SCORED.json',list(traces.values()))
    resources=dict(mix_CPU_wall_seconds=bar['CPU_wall_seconds'],dense_score_CPU_wall_seconds=time.monotonic()-tick,new_query_arm_adaptations=288,backward_calls=864,
        new_GPU_calls=0,CUDA_initialized=torch.cuda.is_initialized(),new_backbone_calls=0,new_suffix_replays=0,new_expert_calls=0,spatial_updates=0,temporal_persistence_writes=0,
        reused_controls=['Native','A8','R1','EDeploy','EOracle'],private_run_bytes=sum(p.stat().st_size for p in BASE.glob('*/mix_runs/*.pt')),
        independent_dense_max_error=maxerror,stage_wall_times_exclude_prephase_hashes_loading_and_root_audits=True,time=time.time())
    write(PUB/'RESOURCES.json',resources);write(BASE/'SCORE_COMPLETION.json',dict(status='completed_pending_root_audit_publication',rows=1152,new_query_arm_adaptations=288,
        time=time.time(),score_pid=os.getpid(),rows_sha256=sha(PUB/'ROWS.json'),summary_sha256=sha(PUB/'SUMMARY.json')))
    st('completed_pending_root_audit_publication',rows=1152,adaptations=288)

def main():
    torch.set_num_threads(2);torch.manual_seed(20261004)
    if len(sys.argv)>1:return {'prepare':prepare,'mix':adapt,'score':score}[sys.argv[1]]()
    BASE.mkdir(parents=True,exist_ok=True)
    for phase in ['prepare','mix','score']:
        with (BASE/(phase.upper()+'.log')).open('x') as out:
            p=subprocess.run([sys.executable,'-B',__file__,phase],stdout=out,stderr=subprocess.STDOUT)
        if p.returncode:st('failed',phase=phase,returncode=p.returncode);raise RuntimeError('R2b phase failed: '+phase)
        print('R2B_STAGE_COMPLETED',phase,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
