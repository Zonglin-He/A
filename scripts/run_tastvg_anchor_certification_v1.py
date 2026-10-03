"""Bounded CPU-only source calibration, sealed decisions, then cached GT audit."""
import os,sys,time,json,hashlib,collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tastvg_anchor_certification_math_v1 import *
from scripts.tastvg_large_evidence_math_v1 import geometry
BASE=ROOT/'artifacts/tastvg_anchor_certification_v1'
OUT=ROOT/'results/tastvg_anchor_certification/2026-10-03'
OLD=ROOT/'results/tastvg_temporal_latent_quality/2026-10-03'
DATASETS=['vidstg','hc2']
OWN=['scripts/run_tastvg_anchor_certification_v1.py','scripts/tastvg_anchor_certification_math_v1.py',
 'scripts/test_tastvg_anchor_certification_v1.py','protocols/tastvg_anchor_certification_v1.md',
 'docs/tastvg_anchor_certification_v1/EXECUTION.md','scripts/tastvg_large_evidence_math_v1.py']

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),str(p)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def status(s,**d):
    BASE.mkdir(parents=True,exist_ok=True);p=BASE/'STATUS.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(status=s,time=time.time(),pid=os.getpid(),**d),indent=2)+'\n');t.replace(p)
def key(r):return '/'.join(str(r[k]) for k in ['dataset','split','condition','order','arrival'])
def verify():
    r=read(BASE/'RUNTIME_LOCK.json')
    for f,h in {**r['code'],**r['inputs']}.items():assert sha(ROOT/f)==h,f
    assert sha(OUT/'CONFIG.json')==r['config_sha256']
    assert 'torch' not in sys.modules
    return r
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
        p=str(args[0])
        if p.endswith('/ROWS.json') or any(x in p for x in ['/annos/','/annotations/','SOURCE_GT.json']):
            raise PermissionError('Calibration/decision phase cannot open target labelled rows or annotations')

def prepare():
    assert read(ROOT/'artifacts/tastvg_controlled_query_v1/FINAL_COMPLETION.json')['commit']=='5c3a3c93c1b6b23d93ee2146b2ec20f5ec35663e'
    assert read(OLD/'ROOT_READBACK.json')['status']=='pass'
    names=['SOURCE_ROWS.json','SCORE_ROWS.json','GLOBAL_SCORE_SEAL.json','SOURCE_FIT_SEAL.json','CONFIG.json','LABEL_JOIN.json','ROOT_READBACK.json']
    inputs={str((OLD/n).relative_to(ROOT)):sha(OLD/n) for n in names}
    inputs['methods/CURRENT_METHOD.json']=sha(ROOT/'methods/CURRENT_METHOD.json')
    # Hash-only provenance of already-exposed cached labels; do not parse them.
    for ds in DATASETS:
        for sp in ['search','confirm']:
            p=OLD/sp/ds/'ROWS.json';inputs[str(p.relative_to(ROOT))]=sha(p)
    for ds in DATASETS:
        p=ROOT/'artifacts/tastvg_temporal_latent_quality_v1'/ds/'FROZEN_PROBE.json';inputs[str(p.relative_to(ROOT))]=sha(p)
    config=dict(version='tastvg_anchor_certification_v1',predecessor_commit='5c3a3c93c1b6b23d93ee2146b2ec20f5ec35663e',
        source_validation_counts=dict(vidstg=31,hc2=16),source_anchor='frozen native candidate0',target_anchor='unchanged A8',
        user_explicit_native_to_A8_CPU_authorization=True,source_validation_reused_after_ridge_alpha_selection=True,
        calibration='increasing isotonic on eligible source-validation top1 winners, one/source',
        bootstrap_draws=DRAWS,bootstrap_seed=SEED,primary_lower_quantile=.05,curve_lower_quantiles=QUANTILES,
        curve_margin_quantiles=np.linspace(0,1,21).tolist(),source_margin_extrapolation=False,
        target_counts=dict(arrivals=1152,expert=288,nonexpert=864,corrupt_expert=240,clean_expert=48),
        source_fit_counts=dict(vidstg=[95,31],hc2=[48,16]),target_design='32 development+16 confirmation/dataset;two orders;clean+five5%;25% expert',
        source_supervised=True,target_GT_fit=False,target_GT_threshold_selection=False,target_historical_exposure=True,
        intake_target_label_access='hash-only existing cached file binding; contents parsed after global new-decision seal',
        arms=['A','L32','Selective'],ranking_unchanged=True,checkpoint_sampling_A_state_unchanged=True,
        new_GPU_model_expert_replay_backward_parameter_update_counts=0,
        lower_bound_scope='approximate pointwise bootstrap lower mean-delta curve; not individual or target safety guarantee',
        GO='both confirm full-corruption mean_v>0;>=3 accepting sources/dataset;positive LOO;gross loss and severe count lower than L32',
        production_method_sha256=inputs['methods/CURRENT_METHOD.json'])
    write(OUT/'CONFIG.json',config)
    write(BASE/'RUNTIME_LOCK.json',dict(time=time.time(),code={f:sha(ROOT/f) for f in OWN},inputs=inputs,config_sha256=sha(OUT/'CONFIG.json')))
    status('prepared_pending_CPU_source_calibration',target_GT_read=False)

def calibrate():
    verify();sys.addaudithook(guard);tick=time.time();src=read(OLD/'SOURCE_ROWS.json');stats={}
    for ds in DATASETS:
        rows=[]
        rr=[r for r in src if r['dataset']==ds and r['split']=='validation']
        assert len(rr)=={'vidstg':31,'hc2':16}[ds] and len({r['source_id'] for r in rr})==len(rr)
        for r in rr:
            a=r['native_index'];assert a==0
            scores=r['frozen_scores']['L'];k=top1(scores,a);gt=r['candidate_t']
            rows.append(dict(dataset=ds,source_id=r['source_id'],source_id_sha256=r['source_id_sha256'],
                native_index=a,top_index=k,eligible=k!=a,margin=float(scores[k]-scores[a]),
                true_delta_t=float(gt[k]-gt[a]),source_native_t=float(gt[a]),source_top_t=float(gt[k]),
                scores=scores,candidate_t=gt,split='source_validation'))
        model=fit_calibration(rows);write(OUT/ds/'SOURCE_CALIBRATION_ROWS.json',rows);write(OUT/ds/'CALIBRATION.json',model)
        stats[ds]=dict(validation_sources=len(rr),eligible_sources=model['eligible_sources'],
            source_native_t=float(np.mean([r['source_native_t'] for r in rows])),
            source_L32_t=float(np.mean([r['source_top_t'] for r in rows])))
        print('SOURCE_CALIBRATED',ds,len(rr),model['eligible_sources'],flush=True)
    write(OUT/'CALIBRATION_SEAL.json',dict(time=time.time(),models={ds:sha(OUT/ds/'CALIBRATION.json') for ds in DATASETS},
        rows={ds:sha(OUT/ds/'SOURCE_CALIBRATION_ROWS.json') for ds in DATASETS},source_native_anchor=True,target_GT_read=False,
        target_scores_opened=False,stats=stats,CPU_wall_seconds=time.time()-tick))
    status('source_calibration_frozen_pending_target_decisions',target_GT_read=False)

def seal():
    verify();sys.addaudithook(guard);tick=time.time();cal=read(OUT/'CALIBRATION_SEAL.json');scores=read(OLD/'SCORE_ROWS.json')
    assert len(scores)==288
    models={ds:read(OUT/ds/'CALIBRATION.json') for ds in DATASETS};rows=[];counts=collections.Counter()
    for r in scores:
        ds=r['dataset'];assert sha(OUT/ds/'CALIBRATION.json')==cal['models'][ds]
        d=decide(r['scores']['L'],r['anchor_index'],models[ds]);assert d['choices']['L32']==r['choices']['L32']
        d.update({k:r[k] for k in ['cell_key','dataset','split','source_id','condition','order','arrival','anchor_index',
            'candidate_indices','intervals','A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256']})
        d['scores']=r['scores']['L'];d['top_geometry']=geometry(r['intervals'][r['anchor_index']],r['intervals'][d['top_index']])
        rows.append(d);counts[ds+'/'+d['reason']]+=1
    write(OUT/'DECISIONS.json',rows)
    write(OUT/'GLOBAL_DECISION_SEAL.json',dict(time=time.time(),decisions_sha256=sha(OUT/'DECISIONS.json'),
        calibration_seal_sha256=sha(OUT/'CALIBRATION_SEAL.json'),target_GT_read=False,cells=288,
        nonexpert_fallback_fixed=864,counts=dict(counts),CPU_wall_seconds=time.time()-tick))
    status('all_target_decisions_sealed_pending_cached_GT_join',target_GT_read=False,decisions=288)

def metric_row(r,d,arm):
    if r['expert_scheduled']:
        a=d['anchor_index'];i=d['choices'][arm];v=float(r['candidate_v'][i]);t=float(r['candidate_t'][i]);accepted=i!=a
        physical=accepted and np.max(np.abs(np.array(d['intervals'][i])-d['intervals'][a]))>EPS
    else:i=None;v=r['A8_v'];t=r['A8_t'];accepted=physical=False
    dv=v-r['A8_v'];dt=t-r['A8_t']
    return dict(v=v,t=t,dv=dv,dt=dt,gross_gain=max(dv,0.),gross_loss=max(-dv,0.),
        accepted=float(accepted),physical_replacement=float(physical),benefit_t=float(accepted and dt>EPS),
        accepted_dt=dt if accepted else 0.,severe=float(dv<-.05),accepted_severe=float(accepted and dv<-.05),
        benefit_v=float(dv>EPS),harm_v=float(dv< -EPS),neutral_v=float(abs(dv)<=EPS))

def summarize_arm(rows,arm,group='all'):
    rr=[]
    for r in rows:
        if group!='all' and (not r['expert_scheduled'] or r['top_geometry']['large']!=(group=='large')):continue
        e={k:r[k] for k in ['source_id','order','condition']};e.update(r['arms'][arm]);rr.append(e)
    fields=['v','t','dv','dt','gross_gain','gross_loss','accepted','physical_replacement','benefit_t','accepted_dt',
        'severe','accepted_severe','benefit_v','harm_v','neutral_v']
    z=summarize(rr,fields,dict(beneficial_precision=('benefit_t','accepted'),accepted_mean_delta_t=('accepted_dt','accepted'),
        accepted_severe_rate=('accepted_severe','accepted')))
    z['raw_counts']={k:int(sum(r[k] for r in rr)) for k in ['accepted','physical_replacement','benefit_t','severe','benefit_v','harm_v','neutral_v']}
    z['accepting_sources']=len({r['source_id'] for r in rr if r['accepted']})
    return z

def diagnose():
    verify();tick=time.time();seal=read(OUT/'GLOBAL_DECISION_SEAL.json');assert sha(OUT/'DECISIONS.json')==seal['decisions_sha256']
    ev={r['cell_key']:r for r in read(OUT/'DECISIONS.json')};coverage=collections.Counter();labelpins={};decision={}
    for ds in DATASETS:
        for split in ['search','confirm']:
            f=OLD/split/ds/'ROWS.json';labelpins[str(f.relative_to(ROOT))]=sha(f);prior=read(f);rows=[]
            for r in prior:
                coverage['arrivals']+=1;e=ev[key(r)] if r['expert_scheduled'] else None
                out={k:r[k] for k in ['dataset','split','source_id','condition','order','arrival','expert_scheduled',
                    'A_state_pre_sha256','A_state_post_sha256','A8_v','A8_t']}
                if e:
                    assert r['anchor_index']==e['anchor_index'];a=e['anchor_index']
                    assert abs(r['candidate_v'][a]-r['A8_v'])<EPS and abs(r['candidate_t'][a]-r['A8_t'])<EPS
                    out.update(anchor_index=a,candidate_v=r['candidate_v'],candidate_t=r['candidate_t'],
                        top_geometry=e['top_geometry'],choices=e['choices'],margin=e['margin'],reason=e['reason'])
                    arms=list(e['choices']);coverage['expert']+=1
                else:arms=['A','L32','Selective'];coverage['nonexpert']+=1
                out['arms']={arm:metric_row(r,e,arm) for arm in arms}
                assert out['arms']['L32']['v']==r['L32_v'] and out['arms']['L32']['t']==r['L32_t']
                if not e:
                    model=read(OUT/ds/'CALIBRATION.json');allarms=['LCB_'+str(q) for q in QUANTILES]+['Margin_'+str(j) for j in range(len(model['thresholds']))]
                    for arm in allarms:out['arms'][arm]=dict(out['arms']['A'])
                rows.append(out)
            write(OUT/split/ds/'ROWS.json',rows);summary={};curves={}
            for cat in ['corruption','clean']:
                catrows=[r for r in rows if (r['condition']=='clean')==(cat=='clean')]
                summary[cat]={}
                for subset in ['all','expert','nonexpert']:
                    rr=[r for r in catrows if subset=='all' or r['expert_scheduled']==(subset=='expert')]
                    summary[cat][subset]={arm:summarize_arm(rr,arm) for arm in ['A','L32','Selective']}
                    if subset=='all':
                        paired=[]
                        for r in rr:
                            a=r['arms'];paired.append(dict(source_id=r['source_id'],condition=r['condition'],order=r['order'],
                                Selective_vs_L32_v=a['Selective']['v']-a['L32']['v'],Selective_vs_L32_t=a['Selective']['t']-a['L32']['t'],
                                Selective_vs_L32_gross_loss=a['Selective']['gross_loss']-a['L32']['gross_loss']))
                        summary[cat]['paired_vs_L32']=summarize(paired,['Selective_vs_L32_v','Selective_vs_L32_t','Selective_vs_L32_gross_loss'])
                experts=[r for r in catrows if r['expert_scheduled']]
                curves[cat]={g:{arm:summarize_arm(experts,arm,g) for arm in rows[0]['arms']} for g in ['all','small','large']}
            write(OUT/split/ds/'SUMMARY.json',summary);write(OUT/split/ds/'CURVES.json',curves)
            experts=[r for r in rows if r['expert_scheduled'] and r['condition']!='clean']
            write(OUT/split/ds/'CASES.json',dict(positive=sorted(experts,key=lambda r:-r['arms']['Selective']['dv'])[:3],negative=sorted(experts,key=lambda r:r['arms']['Selective']['dv'])[:3]))
            print('CACHED_GT_PANEL',ds,split,len(rows),flush=True)
            if split=='confirm':
                z=summary['corruption']['all'];s=z['Selective'];l=z['L32'];m=s['metrics']
                tests=dict(positive_mean=m['dv']['mean']>EPS,accepting_sources=s['accepting_sources']>=3,
                    positive_leave_one_out=m['dv']['leave_one_out_range'][0]>EPS,
                    less_gross_loss=m['gross_loss']['mean']<l['metrics']['gross_loss']['mean']-EPS,
                    fewer_severe_harms=s['raw_counts']['severe']<l['raw_counts']['severe'])
                decision[ds]=dict(tests=tests,pass_all=all(tests.values()),mean_v_delta=m['dv']['mean'],accepting_sources=s['accepting_sources'])
    assert dict(coverage)==dict(arrivals=1152,expert=288,nonexpert=864)
    write(OUT/'LABEL_JOIN.json',dict(time=time.time(),global_seal_time=seal['time'],label_files=labelpins,
        raw_target_annotations_opened=False,target_GT_fit=False,target_GT_threshold_selection=False))
    write(OUT/'DECISION.json',dict(status='GO_development_only' if all(d['pass_all'] for d in decision.values()) else 'NO_GO_locked_calibration',
        datasets=decision,method_promoted=False,followup_started=False,scope='this source-native to target-A8 isotonic bootstrap heuristic only'))
    write(OUT/'RESOURCES.json',dict(time=time.time(),coverage=dict(coverage),CPU_diagnose_wall_seconds=time.time()-tick,
        new_GPU_forwards=0,new_expert_calls=0,new_replays=0,new_backward=0,new_parameter_updates=0,
        source_supervised_one_dimensional_calibration=True,ranking_refit=False,production_method_unchanged=True))
    status('completed_pending_root_audit_report_publication',coverage=dict(coverage),target_GT_joined=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        import traceback
        status('engineering_failed',stage=sys.argv[1],traceback=traceback.format_exc());raise
