"""Bounded source-only CPU fits, immutable choices, then cached target labels."""
import os, sys, time, json, hashlib, collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tastvg_pairwise_certification_math_v1 import *
from scripts.tastvg_large_evidence_math_v1 import geometry
from scripts.run_tastvg_anchor_certification_v1 import metric_row, summarize_arm

BASE=ROOT/'artifacts/tastvg_pairwise_certification_v1'
OUT=ROOT/'results/tastvg_pairwise_certification/2026-10-03'
OLD=ROOT/'results/tastvg_temporal_latent_quality/2026-10-03'
DATASETS=['vidstg','hc2']
OWN=['scripts/run_tastvg_pairwise_certification_v1.py','scripts/tastvg_pairwise_certification_math_v1.py',
     'scripts/test_tastvg_pairwise_certification_v1.py','protocols/tastvg_pairwise_certification_v1.md',
     'docs/tastvg_pairwise_certification_v1/EXECUTION.md','scripts/tastvg_large_evidence_math_v1.py',
     'scripts/tastvg_anchor_certification_math_v1.py','scripts/run_tastvg_anchor_certification_v1.py']

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),str(p)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def status(s,**kw):
    BASE.mkdir(parents=True,exist_ok=True);p=BASE/'STATUS.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(status=s,time=time.time(),pid=os.getpid(),**kw),indent=2)+'\n');tmp.replace(p)
def key(r):return '/'.join(str(r[k]) for k in ['dataset','split','condition','order','arrival'])
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
        p=str(args[0])
        if p.endswith('/ROWS.json') or any(x in p for x in ['/annos/','/annotations/','SOURCE_GT.json']):
            raise PermissionError('No target labelled rows or raw annotations before all decisions seal')
def verify():
    lock=read(BASE/'RUNTIME_LOCK.json')
    for f,h in {**lock['code'],**lock['inputs']}.items():assert sha(ROOT/f)==h,f
    assert sha(OUT/'CONFIG.json')==lock['config_sha256'];assert 'torch' not in sys.modules
    return lock

def prepare():
    assert read(ROOT/'artifacts/tastvg_anchor_certification_v1/FINAL_COMPLETION.json')['commit']=='b7a8be7a0952a70fce7c231f7fef38e394a6e299'
    assert read(OLD/'ROOT_READBACK.json')['status']=='pass'
    paths=[OLD/n for n in ['SOURCE_ROWS.json','SCORE_ROWS.json','SOURCE_FIT_SEAL.json','GLOBAL_SCORE_SEAL.json','CONFIG.json','ROOT_READBACK.json']]
    paths += [OLD/sp/ds/'ROWS.json' for ds in DATASETS for sp in ['search','confirm']]
    public_inputs={str(p.relative_to(ROOT)):sha(p) for p in paths}
    private_paths=[ROOT/'methods/CURRENT_METHOD.json']+[ROOT/'artifacts/tastvg_temporal_latent_quality_v1'/ds/'FROZEN_PROBE.json' for ds in DATASETS]
    inputs={**public_inputs,**{str(p.relative_to(ROOT)):sha(p) for p in private_paths}}
    config=dict(version='tastvg_pairwise_certification_v1',predecessor_commit='b7a8be7a0952a70fce7c231f7fef38e394a6e299',
        source_validation_counts=dict(vidstg=31,hc2=16),source_ridge_train_counts=dict(vidstg=95,hc2=48),
        source_validation_reused_after_ridge_alpha_and_calibration=True,source_supervised=True,
        source_pairs='all unordered L-score-strict pairs oriented higher-score minus lower-score; GT ties/negative delta retained',
        source_weight='total one/source; equal within-source pairs; source bootstrap',
        calibration='increasing isotonic fitted mean tIoU difference',IQR_epsilon=IQR_EPS,IQR_percentile_method='linear',
        pair_score_tie_epsilon=EPS,bootstrap_draws=DRAWS,bootstrap_seed=SEED,primary='Pair-Norm',
        matched_diagnostic='Pair-Raw',fixed_lower_quantile=Q,threshold_family_searched=False,
        primary_rule='unique L32> A8 and within source pair-margin domain and lower mean>1e-12',
        lower_curve_scope='pointwise fitted-mean bootstrap heuristic; not individual/conformal/simultaneous/target safety',
        target_anchor='unchanged A8',ranking_unchanged=True,target_GT_fit=False,target_GT_threshold_selection=False,
        target_historical_exposure=True,target_design='32 development+16 confirmation/dataset;one query/source;two orders;clean+five5%;25% expert',
        target_counts=dict(arrivals=1152,expert=288,nonexpert=864,corrupt_expert=240,clean_expert=48),arms=ARMS,
        production_method_sha256=inputs['methods/CURRENT_METHOD.json'],checkpoint_sampling_A_state_unchanged=True,
        GO='both confirm full-corruption mean_v>0;>=3 accepting sources/dataset;positive LOO;gross loss and severe count lower than L32',
        scientific_limits='source arbitrary-pair to target selected-top1/A8 and domain shift remain; no unique causal identification',
        new_GPU_expert_replay_backward_update_counts=0)
    write(OUT/'CONFIG.json',config)
    lock=dict(time=time.time(),code={f:sha(ROOT/f) for f in OWN},inputs=inputs,config_sha256=sha(OUT/'CONFIG.json'))
    write(BASE/'RUNTIME_LOCK.json',lock)
    write(OUT/'RUNTIME_BINDING.json',dict(code=lock['code'],public_required_inputs=public_inputs,config_sha256=lock['config_sha256']))
    status('prepared_pending_source_calibration',target_GT_read=False)

def calibrate():
    verify();sys.addaudithook(guard);start=time.time();source=read(OLD/'SOURCE_ROWS.json');seal={}
    for ds in DATASETS:
        rows=[dict(source_id=r['source_id'],source_id_sha256=r['source_id_sha256'],scores=r['frozen_scores']['L'],candidate_t=r['candidate_t'])
              for r in source if r['dataset']==ds and r['split']=='validation']
        assert len(rows)=={'vidstg':31,'hc2':16}[ds] and len({r['source_id'] for r in rows})==len(rows)
        pr,meta=pairs(rows);assert all(r['strict_pairs']>0 for r in meta)
        write(OUT/ds/'SOURCE_CANDIDATES.json',rows);write(OUT/ds/'SOURCE_PAIRS.json',pr);write(OUT/ds/'PAIR_METADATA.json',meta)
        models={}
        for arm in ARMS[2:]:
            tick=time.time();status('calibrating_source_only',dataset=ds,arm=arm,target_GT_read=False)
            print('SOURCE_FIT_BEGIN',ds,arm,len(pr),len(rows),flush=True)
            model=calibrate_model(pr,len(rows),arm);model['CPU_wall_seconds']=time.time()-tick
            p=OUT/ds/(arm+'.json');write(p,model);models[arm]=sha(p)
            print('SOURCE_FIT_SEALED',ds,arm,'wall_seconds',model['CPU_wall_seconds'],flush=True)
        seal[ds]=dict(models=models,source_candidates=sha(OUT/ds/'SOURCE_CANDIDATES.json'),source_pairs=sha(OUT/ds/'SOURCE_PAIRS.json'),metadata=sha(OUT/ds/'PAIR_METADATA.json'))
    write(OUT/'CALIBRATION_SEAL.json',dict(time=time.time(),datasets=seal,target_GT_read=False,target_score_rows_opened=False,
                                        CPU_wall_seconds=time.time()-start))
    status('source_calibration_sealed_pending_target_decisions',target_GT_read=False)

# Separate alias prevents the stage function from shadowing the mathematical fit.
from scripts.tastvg_pairwise_certification_math_v1 import calibrate as calibrate_model

def seal():
    verify();sys.addaudithook(guard);start=time.time();cal=read(OUT/'CALIBRATION_SEAL.json')
    models={ds:{arm:read(OUT/ds/(arm+'.json')) for arm in ARMS[2:]} for ds in DATASETS}
    for ds in DATASETS:
        for arm in ARMS[2:]:assert sha(OUT/ds/(arm+'.json'))==cal['datasets'][ds]['models'][arm]
    rows=[];counts=collections.Counter()
    for r in read(OLD/'SCORE_ROWS.json'):
        d=decide(r['scores']['L'],r['anchor_index'],models[r['dataset']]);assert d['top_index']==r['choices']['L32']
        d.update({k:r[k] for k in ['cell_key','dataset','split','source_id','condition','order','arrival','anchor_index',
                                  'candidate_indices','intervals','A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256']})
        d['scores']=r['scores']['L'];d['top_geometry']=geometry(r['intervals'][r['anchor_index']],r['intervals'][d['top_index']])
        for arm in ARMS[2:]:counts[r['dataset']+'/'+arm+'/'+d['evidence'][arm]['reason']]+=1
        rows.append(d)
    assert len(rows)==288;write(OUT/'DECISIONS.json',rows)
    write(OUT/'GLOBAL_DECISION_SEAL.json',dict(time=time.time(),decisions_sha256=sha(OUT/'DECISIONS.json'),
        calibration_seal_sha256=sha(OUT/'CALIBRATION_SEAL.json'),target_GT_read=False,cells=288,counts=dict(counts),CPU_wall_seconds=time.time()-start))
    status('all_target_decisions_sealed_pending_cached_labels',target_GT_read=False)

def diagnose():
    verify();start=time.time();barrier=read(OUT/'GLOBAL_DECISION_SEAL.json');assert sha(OUT/'DECISIONS.json')==barrier['decisions_sha256']
    ev={r['cell_key']:r for r in read(OUT/'DECISIONS.json')};total=collections.Counter();pins={};decisions={arm:{} for arm in ARMS[2:]}
    for ds in DATASETS:
        for split in ['search','confirm']:
            p=OLD/split/ds/'ROWS.json';pins[str(p.relative_to(ROOT))]=sha(p);rows=[]
            for r in read(p):
                d=ev[key(r)] if r['expert_scheduled'] else None;total['arrivals']+=1;total['expert' if d else 'nonexpert']+=1
                out={k:r[k] for k in ['dataset','split','source_id','condition','order','arrival','expert_scheduled','A_state_pre_sha256','A_state_post_sha256','A8_v','A8_t']}
                if d:
                    assert r['anchor_index']==d['anchor_index'];a=d['anchor_index']
                    assert abs(r['candidate_v'][a]-r['A8_v'])<EPS and abs(r['candidate_t'][a]-r['A8_t'])<EPS
                    out.update(anchor_index=a,candidate_v=r['candidate_v'],candidate_t=r['candidate_t'],choices=d['choices'],
                               top_geometry=d['top_geometry'],evidence=d['evidence'],raw_margin=d['raw_margin'],IQR=d['IQR'])
                out['arms']={arm:metric_row(r,d,arm) for arm in ARMS}
                assert out['arms']['L32']['v']==r['L32_v'] and out['arms']['L32']['t']==r['L32_t'];rows.append(out)
            write(OUT/split/ds/'ROWS.json',rows);summary={};strata={}
            for cat in ['corruption','clean']:
                rr=[r for r in rows if (r['condition']=='clean')==(cat=='clean')];summary[cat]={}
                for sub in ['all','expert','nonexpert']:
                    pp=[r for r in rr if sub=='all' or r['expert_scheduled']==(sub=='expert')]
                    summary[cat][sub]={arm:summarize_arm(pp,arm) for arm in ARMS}
                comparisons=[]
                for r in rr:
                    z=dict(source_id=r['source_id'],order=r['order'],condition=r['condition'])
                    for arm in ARMS[2:]:
                        for f in ['v','t','gross_loss']:z[arm+'_vs_L32_'+f]=r['arms'][arm][f]-r['arms']['L32'][f]
                    z['Norm_vs_Raw_v']=r['arms']['Pair-Norm']['v']-r['arms']['Pair-Raw']['v'];comparisons.append(z)
                summary[cat]['paired']=summarize(comparisons,list(comparisons[0].keys())[3:])
                expert=[r for r in rr if r['expert_scheduled']]
                strata[cat]={g:{arm:summarize_arm(expert,arm,g) for arm in ARMS} for g in ['all','small','large']}
            write(OUT/split/ds/'SUMMARY.json',summary);write(OUT/split/ds/'DIAGNOSTIC_STRATA.json',strata)
            expert=[r for r in rows if r['expert_scheduled'] and r['condition']!='clean']
            cases={arm:dict(positive=sorted(expert,key=lambda r:-r['arms'][arm]['dv'])[:3],negative=sorted(expert,key=lambda r:r['arms'][arm]['dv'])[:3],
                            useful_rejected=sorted([r for r in expert if not r['arms'][arm]['accepted'] and r['arms']['L32']['dv']>EPS],key=lambda r:-r['arms']['L32']['dv'])[:3]) for arm in ARMS[2:]}
            write(OUT/split/ds/'CASES.json',cases)
            if split=='confirm':
                z=summary['corruption']['all'];l=z['L32']
                for arm in ARMS[2:]:
                    s=z[arm];m=s['metrics'];tests=dict(positive_mean=m['dv']['mean']>EPS,accepting_sources=s['accepting_sources']>=3,
                        positive_leave_one_out=m['dv']['leave_one_out_range'][0]>EPS,
                        less_gross_loss=m['gross_loss']['mean']<l['metrics']['gross_loss']['mean']-EPS,
                        fewer_severe_harms=s['raw_counts']['severe']<l['raw_counts']['severe'])
                    decisions[arm][ds]=dict(tests=tests,pass_all=all(tests.values()),mean_v_delta=m['dv']['mean'],accepting_sources=s['accepting_sources'])
            print('CACHED_LABEL_PANEL',ds,split,len(rows),flush=True)
    assert dict(total)==dict(arrivals=1152,expert=288,nonexpert=864)
    write(OUT/'LABEL_JOIN.json',dict(time=time.time(),global_seal_time=barrier['time'],label_files=pins,target_GT_fit=False,
                                    target_threshold_selection=False,raw_target_annotations_opened=False))
    primary_go=all(d['pass_all'] for d in decisions['Pair-Norm'].values())
    write(OUT/'DECISION.json',dict(status='GO_development_only' if primary_go else 'NO_GO_locked_pairwise_certification',primary='Pair-Norm',
        arm_decisions=decisions,raw_joint_go=all(d['pass_all'] for d in decisions['Pair-Raw'].values()),method_promoted=False,
        followup_started=False,causal_identification=False,scope='these frozen raw/IQR pairwise isotonic mean-delta rules only'))
    write(OUT/'RESOURCES.json',dict(coverage=dict(total),CPU_diagnose_wall_seconds=time.time()-start,new_GPU_forwards=0,new_expert_calls=0,
        new_replays=0,new_backward=0,new_parameter_updates=0,source_supervised=True,ranking_refit=False,production_method_unchanged=True))
    status('completed_pending_root_audit_report_publication',coverage=dict(total),target_GT_joined=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        import traceback
        status('engineering_failed',stage=sys.argv[1],traceback=traceback.format_exc());raise
