"""Finite CPU baseline diagnosis -> source LOSO -> seal -> cached evaluation."""
import os,sys,time,json,hashlib,collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tastvg_anchor_quality_math_v1 import EPS,ARMS,feature,pseudo_pairs,fit as fit_model,predict,choose,quartiles,correlation
from scripts.tastvg_anchor_certification_math_v1 import summarize
from scripts.run_tastvg_anchor_certification_v1 import metric_row,summarize_arm
from scripts.tastvg_large_evidence_math_v1 import geometry
BASE=ROOT/'artifacts/tastvg_anchor_quality_v1'
OUT=ROOT/'results/tastvg_anchor_quality/2026-10-03'
OLD=ROOT/'results/tastvg_temporal_latent_quality/2026-10-03'
PAIR=ROOT/'results/tastvg_pairwise_certification/2026-10-03'
DS=['vidstg','hc2']
OWN=['scripts/run_tastvg_anchor_quality_v1.py','scripts/tastvg_anchor_quality_math_v1.py','scripts/test_tastvg_anchor_quality_v1.py',
     'protocols/tastvg_anchor_quality_v1.md','docs/tastvg_anchor_quality_v1/EXECUTION.md',
     'scripts/tastvg_anchor_certification_math_v1.py','scripts/run_tastvg_anchor_certification_v1.py','scripts/tastvg_large_evidence_math_v1.py']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),str(p)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def status(s,**kw):
    BASE.mkdir(parents=True,exist_ok=True);p=BASE/'STATUS.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(status=s,time=time.time(),pid=os.getpid(),**kw),indent=2)+'\n');tmp.replace(p)
def key(r):return '/'.join(str(r[k]) for k in ['dataset','split','condition','order','arrival'])
def verify():
    lock=read(BASE/'RUNTIME_LOCK.json')
    for p,h in {**lock['code'],**lock['inputs']}.items():assert sha(ROOT/p)==h,p
    assert sha(OUT/'CONFIG.json')==lock['config_sha256'] and 'torch' not in sys.modules
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
        p=str(args[0])
        if p.endswith('/ROWS.json') or any(x in p for x in ['BASELINE_DIAGNOSIS','/annotations/','/annos/','GT_EXPOSURE']):
            raise PermissionError('Source fitting/sealing cannot parse target labels or baseline-case diagnostics')

def prepare():
    assert read(ROOT/'artifacts/tastvg_pairwise_certification_v1/FINAL_COMPLETION.json')['commit']=='d5c67eb178b973d68cbd410c9ccff3a6c092c645'
    assert read(OLD/'ROOT_READBACK.json')['status']=='pass'
    paths=[OLD/n for n in ['SOURCE_ROWS.json','SCORE_ROWS.json','CONFIG.json','GLOBAL_SCORE_SEAL.json','ROOT_READBACK.json']]
    paths += [OLD/sp/ds/'ROWS.json' for sp in ['search','confirm'] for ds in DS]
    paths += [PAIR/n for n in ['DECISIONS.json','GLOBAL_DECISION_SEAL.json','CONFIG.json','ROOT_AUDIT.json']]
    pub={str(p.relative_to(ROOT)):sha(p) for p in paths}
    private=[ROOT/'methods/CURRENT_METHOD.json']
    config=dict(version='tastvg_anchor_quality_v1',predecessor_commit='d5c67eb178b973d68cbd410c9ccff3a6c092c645',
        source_validation_counts=dict(vidstg=31,hc2=16),source_validation_reused_after_alpha_and_calibration=True,
        pseudo_anchor_rule='all32 including unique top self; ambiguous top source excluded; retain GT-negative-neutral and duplicate intervals',
        weighting='total1/source, equal pseudo-anchors',models=dict(M0=['intercept','margin'],M1=['intercept','margin','anchor_score']),
        solver='unregularized weighted NumPy SVD least squares',rcond=1e-12,IQR_epsilon=1e-8,score_epsilon=EPS,
        source_LOSO=True,source_LOSO_selects_nothing=True,bootstrap_draws=10000,seed=20261003,
        target_rule='unique score top>A8 and predicted mean delta>1e-12; same rule M0/M1; no fitted threshold/LCB/domain gate',
        marginal_extrapolation_recorded_only=True,arms=ARMS,ranking_unchanged=True,
        target_design='32search+16confirm/dataset;onequery;twoorders;clean+five5%;25%expert',
        coverage=dict(arrivals=1152,expert=288,nonexpert=864,corrupt_expert=240,clean_expert=48),
        target_historical_GT_exposure=True,baseline_case_GT_diagnosis_before_fit=True,new_choices_seal_before_new_arm_label_join=True,
        target_GT_fit=False,target_threshold_selection=False,quartile_rule='A8_t then cellkey; equal count within panel corrupt experts, no-op retained',
        feature_identity='margin+anchor_score=top_context within source/pool; not independent true anchor-quality measurement',
        correlation_limit='delta_t subtracts anchor_t; negative association does not identify a cause',
        source_supervised=True,production_method_sha256=sha(private[0]),new_GPU_expert_replay_backward_updates=0,
        GO='both datasets: LOSO M0-M1 MSE>0;confirm v delta>0;M1-M0 v>0;>=3 accepting sources;positive LOO;less loss and fewer severe than L32',
        method_promoted=False,structured_model_started=False)
    write(OUT/'CONFIG.json',config)
    lock=dict(time=time.time(),code={p:sha(ROOT/p) for p in OWN},inputs={**pub,**{str(p.relative_to(ROOT)):sha(p) for p in private}},config_sha256=sha(OUT/'CONFIG.json'))
    write(BASE/'RUNTIME_LOCK.json',lock)
    write(OUT/'RUNTIME_BINDING.json',dict(code=lock['code'],public_required_inputs=pub,config_sha256=lock['config_sha256']))
    status('prepared_pending_baseline_case_audit')

def quartile_pack(rows,arms):
    packs=[]
    for qi,group in enumerate(quartiles(rows)):
        p=dict(quartile=qi+1,cells=len(group),cell_keys=[r['cell_key'] for r in group],sources=len({r['source_id'] for r in group}),
               cell_mean_A8_t=float(np.mean([r['A8_t'] for r in group])),A8_t_range=[group[0]['A8_t'],group[-1]['A8_t']],arms={})
        for arm in arms:
            r0=[dict(source_id=r['source_id'],order=r['order'],condition=r['condition'],A8_t=r['A8_t'],
                     **r['arms'][arm]) for r in group]
            p['arms'][arm]=dict(cell_means={f:float(np.mean([r['arms'][arm][f] for r in group])) for f in ['dt','dv','accepted']},
                counts={f:sum(int(r['arms'][arm][f]) for r in group) for f in ['accepted','benefit_t','benefit_v','harm_v','severe']},
                harmful_t=sum(r['arms'][arm]['dt']< -EPS for r in group),neutral_t=sum(abs(r['arms'][arm]['dt'])<=EPS for r in group),
                source_balanced=summarize(r0,['A8_t','dt','dv','accepted','gross_gain','gross_loss','severe']))
        if 'M1' in arms:
            helpful=[r for r in group if r['arms']['L32']['dt']>EPS]
            harmful=[r for r in group if r['arms']['L32']['dt']< -EPS]
            p['proposal_retention']={a:dict(helpful_total=len(helpful),helpful_accepted=sum(r['arms'][a]['accepted']>0 for r in helpful),
                harmful_total=len(harmful),harmful_rejected=sum(r['arms'][a]['accepted']==0 for r in harmful)) for a in ['Pair-Norm','M0','M1']}
        packs.append(p)
    return packs

def cases():
    verify();tick=time.time();all_panels={};pins={}
    prior={r['cell_key']:r for r in read(PAIR/'DECISIONS.json')}
    score={r['cell_key']:r for r in read(OLD/'SCORE_ROWS.json')}
    for ds in DS:
        for sp in ['search','confirm']:
            path=OLD/sp/ds/'ROWS.json';pins[str(path.relative_to(ROOT))]=sha(path);rows=[]
            for r in read(path):
                if not r['expert_scheduled'] or r['condition']=='clean':continue
                ck=key(r);d=prior[ck];f=feature(d['scores'],d['anchor_index'])
                out={k:r[k] for k in ['dataset','split','source_id','condition','order','arrival','A8_t','A8_v']}
                out.update(cell_key=ck,margin=f['margin'],anchor_score=f['anchor_score'],top_context=f['top_context'],
                    radius=d['top_geometry']['radius'],Pair_Norm_evidence=d['evidence']['Pair-Norm'],
                    arms={a:metric_row(r,d,a) for a in ['A','L32','Pair-Norm']})
                rows.append(out)
            source_average_delta=[np.mean(score[r['cell_key']]['scores']['L']) for r in rows] # scores only, not a quality label
            label={key(r):r for r in read(path) if r['expert_scheduled']}
            random_delta=[float(np.mean(label[r['cell_key']]['candidate_t'])-r['A8_t']) for r in rows]
            z=dict(rows=rows,quartiles=quartile_pack(rows,['L32','Pair-Norm']),
                   correlations=dict(A8_t_vs_L32_dt=correlation([r['A8_t'] for r in rows],[r['arms']['L32']['dt'] for r in rows]),
                                     A8_t_vs_mean_candidate_delta=correlation([r['A8_t'] for r in rows],random_delta),
                                     anchor_score_vs_A8_t=correlation([r['anchor_score'] for r in rows],[r['A8_t'] for r in rows])),
                   mean_candidate_delta=random_delta,unlabelled_mean_score=source_average_delta,
                   examples=dict(helpful=sorted(rows,key=lambda r:-r['arms']['L32']['dt'])[:3],
                                 harmful=sorted(rows,key=lambda r:r['arms']['L32']['dt'])[:3],
                                 cited_source37_exposure=[r for r in rows if r['source_id']==37 and r['condition']=='exposure'],
                                 cited_source34=[r for r in rows if r['source_id']==34]))
            all_panels[ds+'/'+sp]=z
            print('BASELINE_CASES',ds,sp,len(rows),flush=True)
    write(OUT/'BASELINE_DIAGNOSIS.json',dict(time=time.time(),panels=all_panels,posthoc_GT_not_a_gate=True))
    write(OUT/'GT_EXPOSURE.json',dict(time=time.time(),label_files=pins,prior_and_current_baseline_exposure=True,
        purpose='verify attachment strata/good/failure cases; no GT input to new predictors/rule',CPU_wall_seconds=time.time()-tick))
    status('baseline_cases_completed_pending_source_only_fit')

def source_fit():
    verify();sys.addaudithook(guard);tick=time.time();source=read(OLD/'SOURCE_ROWS.json');seal={}
    for ds in DS:
        rows=[dict(source_id=r['source_id'],source_id_sha256=r['source_id_sha256'],scores=r['frozen_scores']['L'],candidate_t=r['candidate_t'])
              for r in source if r['dataset']==ds and r['split']=='validation']
        assert len(rows)=={'vidstg':31,'hc2':16}[ds]
        pairs,meta=pseudo_pairs(rows);assert len(pairs)==32*len(rows)
        write(OUT/ds/'SOURCE_CANDIDATES.json',rows);write(OUT/ds/'SOURCE_PAIRS.json',pairs);write(OUT/ds/'PAIR_METADATA.json',meta)
        models={};folds=[];oof=[]
        for arm in ['M0','M1']:
            model=fit_model(pairs,arm);write(OUT/ds/(arm+'.json'),model);models[arm]=sha(OUT/ds/(arm+'.json'))
        for sid in sorted({r['source_id'] for r in pairs}):
            train=[r for r in pairs if r['source_id']!=sid];test=[r for r in pairs if r['source_id']==sid]
            fold_models={a:fit_model(train,a) for a in ['M0','M1']};folds.append(dict(held_out_source=sid,models=fold_models))
            for r in test:
                pred={a:predict(fold_models[a],r) for a in fold_models};e={a:pred[a]-r['y'] for a in pred}
                oof.append(dict(**r,predictions=pred,M0_MSE=e['M0']**2,M1_MSE=e['M1']**2,zero_MSE=r['y']**2,
                    M0_MAE=abs(e['M0']),M1_MAE=abs(e['M1']),MSE_improvement=e['M0']**2-e['M1']**2))
        def pack(pp):
            rr=[dict(r,order='LOSO',condition='source') for r in pp]
            z=summarize(rr,['M0_MSE','M1_MSE','zero_MSE','M0_MAE','M1_MAE','MSE_improvement'])
            z['positive_sources']=sum(v>EPS for v in z['metrics']['MSE_improvement']['source_values'].values());return z
        summary=dict(all_pseudo_anchors=pack(oof),eligible_nonself=pack([r for r in oof if r['eligible']]),
            source_counts=len(rows),pseudo_anchors=len(pairs),GT_positive=sum(r['y']>EPS for r in pairs),
            GT_negative=sum(r['y']< -EPS for r in pairs),GT_neutral=sum(abs(r['y'])<=EPS for r in pairs),
            max_identity_residual=float(max(abs(r['margin']+r['anchor_score']-r['top_context']) for r in pairs)),
            within_source_anchor_proxy_GT_correlations={str(r['source_id']):correlation(r['scores'],r['candidate_t']) for r in rows})
        for name,obj in [('LOSO_FOLDS.json',folds),('LOSO_ROWS.json',oof),('LOSO_SUMMARY.json',summary)]:write(OUT/ds/name,obj)
        seal[ds]=dict(models=models,source_candidates=sha(OUT/ds/'SOURCE_CANDIDATES.json'),pairs=sha(OUT/ds/'SOURCE_PAIRS.json'),
                      metadata=sha(OUT/ds/'PAIR_METADATA.json'),folds=sha(OUT/ds/'LOSO_FOLDS.json'),oof=sha(OUT/ds/'LOSO_ROWS.json'),summary=sha(OUT/ds/'LOSO_SUMMARY.json'))
        print('SOURCE_LOSO_SEALED',ds,len(rows),len(pairs),flush=True)
    write(OUT/'CALIBRATION_SEAL.json',dict(time=time.time(),datasets=seal,no_target_label_or_case_reads_in_fit_process=True,
        prior_baseline_GT_exposure_disclosed=True,CPU_wall_seconds=time.time()-tick))
    status('source_models_sealed_pending_target_choices')

def seal():
    verify();sys.addaudithook(guard);tick=time.time();cal=read(OUT/'CALIBRATION_SEAL.json')
    models={ds:{a:read(OUT/ds/(a+'.json')) for a in ['M0','M1']} for ds in DS}
    for ds in DS:
        for a in ['M0','M1']:assert sha(OUT/ds/(a+'.json'))==cal['datasets'][ds]['models'][a]
    prior={r['cell_key']:r for r in read(PAIR/'DECISIONS.json')};rows=[]
    for r in read(OLD/'SCORE_ROWS.json'):
        d=choose(r['scores']['L'],r['anchor_index'],models[r['dataset']]);assert d['choices']['L32']==r['choices']['L32']
        d['choices']['Pair-Norm']=prior[r['cell_key']]['choices']['Pair-Norm']
        d.update({k:r[k] for k in ['cell_key','dataset','split','source_id','condition','order','arrival','anchor_index',
            'candidate_indices','intervals','A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256']})
        d['scores']=r['scores']['L'];d['top_geometry']=geometry(r['intervals'][d['anchor_index']],r['intervals'][d['top_index']]);rows.append(d)
    assert len(rows)==288;write(OUT/'DECISIONS.json',rows)
    write(OUT/'GLOBAL_DECISION_SEAL.json',dict(time=time.time(),decisions_sha256=sha(OUT/'DECISIONS.json'),calibration_sha256=sha(OUT/'CALIBRATION_SEAL.json'),
        decisions=288,nonexpert_A=864,no_target_label_or_case_reads_in_decision_process=True,prior_baseline_GT_exposure_disclosed=True,CPU_wall_seconds=time.time()-tick))
    status('new_choices_sealed_pending_new_arm_metric_join')

def evaluate():
    verify();tick=time.time();barrier=read(OUT/'GLOBAL_DECISION_SEAL.json');assert sha(OUT/'DECISIONS.json')==barrier['decisions_sha256']
    ev={r['cell_key']:r for r in read(OUT/'DECISIONS.json')};total=collections.Counter();pins={};dec={}
    for ds in DS:
        loso=read(OUT/ds/'LOSO_SUMMARY.json')
        for sp in ['search','confirm']:
            p=OLD/sp/ds/'ROWS.json';pins[str(p.relative_to(ROOT))]=sha(p);rows=[]
            for r in read(p):
                ck=key(r);d=ev[ck] if r['expert_scheduled'] else None;total['arrivals']+=1;total['expert' if d else 'nonexpert']+=1
                out={k:r[k] for k in ['dataset','split','source_id','condition','order','arrival','expert_scheduled','A_state_pre_sha256','A_state_post_sha256','A8_t','A8_v']}
                out['cell_key']=ck
                if d:
                    assert r['anchor_index']==d['anchor_index']
                    assert abs(r['candidate_v'][d['anchor_index']]-r['A8_v'])<EPS and abs(r['candidate_t'][d['anchor_index']]-r['A8_t'])<EPS
                    out.update(candidate_v=r['candidate_v'],candidate_t=r['candidate_t'],choices=d['choices'],top_geometry=d['top_geometry'],
                        evidence=d['evidence'],margin=d['margin'],anchor_score=d['anchor_score'],top_context=d['top_context'])
                out['arms']={a:metric_row(r,d,a) for a in ARMS};assert out['arms']['L32']['v']==r['L32_v'];rows.append(out)
            write(OUT/sp/ds/'ROWS.json',rows);summary={}
            for cat in ['corruption','clean']:
                rr=[r for r in rows if (r['condition']=='clean')==(cat=='clean')];summary[cat]={}
                for sub in ['all','expert','nonexpert']:
                    pp=[r for r in rr if sub=='all' or r['expert_scheduled']==(sub=='expert')]
                    summary[cat][sub]={a:summarize_arm(pp,a) for a in ARMS}
                pp=[]
                for r in rr:
                    p0=dict(source_id=r['source_id'],order=r['order'],condition=r['condition'])
                    for a,b in [('M1','M0'),('M1','L32'),('M0','L32'),('M1','Pair-Norm')]:
                        for f in ['v','t','gross_loss']:p0[a+'_vs_'+b+'_'+f]=r['arms'][a][f]-r['arms'][b][f]
                    pp.append(p0)
                summary[cat]['paired']=summarize(pp,list(pp[0].keys())[3:])
            write(OUT/sp/ds/'SUMMARY.json',summary)
            expert=[r for r in rows if r['expert_scheduled'] and r['condition']!='clean']
            write(OUT/sp/ds/'ANCHOR_STRATA.json',dict(quartiles=quartile_pack(expert,ARMS),
                anchor_score_vs_A8_t=correlation([r['anchor_score'] for r in expert],[r['A8_t'] for r in expert])))
            cases={a:dict(helpful=sorted(expert,key=lambda r:-r['arms'][a]['dv'])[:3],harmful=sorted(expert,key=lambda r:r['arms'][a]['dv'])[:3],
                useful_rejected=sorted([r for r in expert if not r['arms'][a]['accepted'] and r['arms']['L32']['dt']>EPS],key=lambda r:-r['arms']['L32']['dt'])[:3]) for a in ['M0','M1']}
            write(OUT/sp/ds/'CASES.json',cases)
            if sp=='confirm':
                m=summary['corruption']['all']['M1'];l=summary['corruption']['all']['L32'];pair=summary['corruption']['paired']['metrics']['M1_vs_M0_v']
                tests=dict(source_LOSO_MSE_improves=loso['all_pseudo_anchors']['metrics']['MSE_improvement']['mean']>EPS,
                    positive_mean=m['metrics']['dv']['mean']>EPS,better_than_M0=pair['mean']>EPS,accepting_sources=m['accepting_sources']>=3,
                    positive_leave_one_out=m['metrics']['dv']['leave_one_out_range'][0]>EPS,
                    less_gross_loss=m['metrics']['gross_loss']['mean']<l['metrics']['gross_loss']['mean']-EPS,
                    fewer_severe_harms=m['raw_counts']['severe']<l['raw_counts']['severe'])
                dec[ds]=dict(tests=tests,pass_all=all(tests.values()),M1_delta_v=m['metrics']['dv'],M1_vs_M0_v=pair)
            print('NEW_ARM_METRICS',ds,sp,len(rows),flush=True)
    assert dict(total)==dict(arrivals=1152,expert=288,nonexpert=864)
    write(OUT/'LABEL_JOIN.json',dict(time=time.time(),global_seal_time=barrier['time'],label_files=pins,target_GT_fit=False,
        target_threshold_selection=False,baseline_exposure_precedes_fit=True,raw_target_annotations_opened=False))
    write(OUT/'DECISION.json',dict(status='GO_development_only' if all(x['pass_all'] for x in dec.values()) else 'NO_GO_locked_anchor_score_linear_models',
        datasets=dec,causal_anchor_blindness_identified=False,method_promoted=False,followup_started=False,
        scope='locked weighted-linear source-top-vs-pseudo-anchor construction and anchor-score proxy; not all latent representations'))
    write(OUT/'RESOURCES.json',dict(coverage=dict(total),CPU_evaluation_wall_seconds=time.time()-tick,new_GPU_forwards=0,new_expert_calls=0,
        new_replays=0,new_backward=0,new_parameter_updates=0,production_method_unchanged=True))
    status('completed_pending_independent_audit_report_publication',coverage=dict(total))

if __name__=='__main__':
    stages=dict(prepare=prepare,cases=cases,fit=source_fit,seal=seal,evaluate=evaluate)
    try:stages[sys.argv[1]]()
    except Exception:
        import traceback
        status('engineering_failed',stage=sys.argv[1],traceback=traceback.format_exc());raise
