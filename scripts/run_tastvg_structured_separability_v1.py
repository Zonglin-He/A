"""Finite extraction and diagnostic join of previously frozen CPU readouts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import sys,time,json,hashlib,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_structured_separability_math_v1 import *
from scripts.tastvg_anchor_quality_math_v1 import feature
BASE=ROOT/'artifacts/tastvg_structured_separability_v1'
OUT=ROOT/'results/tastvg_structured_separability/2026-10-03'
ATLAS=ROOT/'artifacts/tastvg_temporal_information_atlas_v1'
OLD=ROOT/'results/tastvg_temporal_latent_quality/2026-10-03'
PREV=ROOT/'results/tastvg_anchor_quality/2026-10-03'
DATASETS=['vidstg','hc2']
OWN=['protocols/tastvg_structured_separability_v1.md','docs/tastvg_structured_separability_v1/EXECUTION.md',
     'scripts/run_tastvg_structured_separability_v1.py','scripts/tastvg_structured_separability_math_v1.py',
     'scripts/test_tastvg_structured_separability_v1.py']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),str(p)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def status(name,**kw):
    BASE.mkdir(parents=True,exist_ok=True);p=BASE/'STATUS.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(status=name,time=time.time(),pid=os.getpid(),**kw),indent=2)+'\n');t.replace(p)
def key(r):return '/'.join(str(r[k]) for k in ['dataset','split','condition','order','arrival'])
def verify():
    z=read(BASE/'RUNTIME_LOCK.json')
    for p,h in {**z['code'],**z['inputs']}.items():assert sha(ROOT/p)==h,p
    assert sha(OUT/'CONFIG.json')==z['config_sha256']
    return z
def payload(ds):
    import torch
    torch.set_num_threads(1)
    receipt=read(ATLAS/ds/'SEALED_READOUT.json');p=ATLAS/ds/'SEALED_READOUT.pt'
    assert sha(p)==receipt['sha256']==read(ATLAS/'GLOBAL_READOUT_SEAL.json')['files'][ds]
    z=torch.load(p,map_location='cpu',weights_only=False)
    assert not torch.cuda.is_initialized()
    return {r['cell']:r['predictions'] for r in z}
def prepare():
    assert read(ROOT/'artifacts/tastvg_anchor_quality_v1/FINAL_COMPLETION.json')['commit']=='a60dcbbabdda76303d67cce888f3e221d8f1e92a'
    paths=[ROOT/'methods/CURRENT_METHOD.json',ATLAS/'FINAL_COMPLETION.json',ATLAS/'FIT_SEAL.json',ATLAS/'GLOBAL_READOUT_SEAL.json',
           PREV/'DECISIONS.json',PREV/'GLOBAL_DECISION_SEAL.json',PREV/'ROOT_AUDIT.json',OLD/'SOURCE_ROWS.json',OLD/'SCORE_ROWS.json',OLD/'CONFIG.json']
    paths += [OLD/sp/ds/'ROWS.json' for sp in ['search','confirm'] for ds in DATASETS]
    for ds in DATASETS:
        paths += [ATLAS/ds/'SEALED_READOUT.pt',ATLAS/ds/'SEALED_READOUT.json',ATLAS/ds/'FIT_SUMMARY.json',
                  ROOT/f'artifacts/tastvg_current_correction_views_v1/{ds}/PLAN.json']
        paths += [ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_{sp}.json' for sp in ['search','confirm']]
    config=dict(version='tastvg_structured_separability_v1',predecessor_commit='a60dcbbabdda76303d67cce888f3e221d8f1e92a',
        atlas_commit='fd57375d91933d1509ec914b5ff5bf6a040ad5dc',recipes=RECIPES,primary='Inside_Endpoint',fields=FIELDS,orientations=SIGNS,
        labels='primary tIoU help/harm; secondary fixed-spatial vIoU; neutrals/noops counted, excluded from binary discrimination',
        epsilon=EPS,bootstrap_draws=DRAWS,seed=SEED,source_weighting='equal source inside each class; whole-source paired resampling',
        source_validation_counts=dict(vidstg=31,hc2=16),source_anchor='native0',target_anchor='actual A8',
        source_validation_previously_used=True,target_historical_exposure=True,first_ever_GT_blind=False,
        target_coverage=dict(expert=288,corrupt=240,clean=48,nonexpert_readouts=0,expert_sources=45),
        planned_sources_per_dataset=dict(search=32,confirm=16),actual_target_expert_sources=dict(vidstg=dict(search=16,confirm=8),hc2=dict(search=14,confirm=7)),
        checkpoints=read(OLD/'CONFIG.json')['checkpoint_state_sha256'],sampling='original Paper48 two-offset',
        persistent_A='Uniform Rank-RKL 1792 parameters; VidK1 HCK8; no current output/state changed',
        LOSO='delete-one-source fixed-statistic influence; no training or out-of-fold predictor',
        no_probe_fitting=True,no_new_scores_or_choices=True,raw_ridge_no_clipping=True,source_supervised=True,
        production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),new_GPU_model_expert_replay_candidate_backward_updates=0,
        scalar_gate=False,structured_model_started=False)
    write(OUT/'CONFIG.json',config)
    pins={str(p.relative_to(ROOT)):sha(p) for p in paths}
    lock=dict(time=time.time(),code={p:sha(ROOT/p) for p in OWN},inputs=pins,config_sha256=sha(OUT/'CONFIG.json'))
    write(BASE/'RUNTIME_LOCK.json',lock)
    write(OUT/'RUNTIME_BINDING.json',dict(code=lock['code'],config_sha256=lock['config_sha256'],
        input_hashes=pins,private_assets_excluded=True,hash_only_label_file_integrity_reads=True))
    status('prepared_pending_frozen_feature_extraction')
def extract_stage():
    verify();tick=time.time()
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
            p=str(args[0])
            if any(q in p for q in ['/GT_LABELS_', '/SOURCE_ROWS.json','/BASELINE_DIAGNOSIS','/annotations/']) or p.endswith('/ROWS.json'):
                raise PermissionError('Target feature extraction cannot parse diagnostic labels')
    # Runtime integrity verification did hash-only byte reads before this guard.
    sys.addaudithook(guard)
    decisions={r['cell_key']:r for r in read(PREV/'DECISIONS.json')};scores=read(OLD/'SCORE_ROWS.json');rows=[]
    for ds in DATASETS:
        atlas=payload(ds)
        for r in scores:
            if r['dataset']!=ds:continue
            d=decisions[r['cell_key']];a=r['anchor_index'];f=feature(r['scores']['L'],a)
            assert f['top_index']==d['top_index'] and f['eligible']==d['eligible']
            w=d['choices']['L32'];assert w==(f['top_index'] if f['eligible'] else a)
            out={k:r[k] for k in ['cell_key','dataset','split','source_id','condition','order','arrival','anchor_index',
                 'A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256']}
            out.update(domain='target',winner_index=w,eligible=f['eligible'],
                readouts=extract(atlas[r['cell_key']],a,w),previous_L_margin=float(r['scores']['L'][w]-r['scores']['L'][a]),
                frozen_target_vectors_sha256=sha(ATLAS/ds/'SEALED_READOUT.pt'))
            rows.append(out)
    assert len(rows)==288 and len({r['cell_key'] for r in rows})==288
    write(OUT/'FEATURE_ROWS.json',rows)
    write(OUT/'FEATURE_SEAL.json',dict(time=time.time(),target_feature_rows=288,feature_sha256=sha(OUT/'FEATURE_ROWS.json'),
        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),labels_parsed_in_extraction=False,
        inherited_atlas_readout_seal_sha256=sha(ATLAS/'GLOBAL_READOUT_SEAL.json'),CPU_wall_seconds=time.time()-tick))
    status('all_target_features_sealed_pending_diagnostic_join')

def temporal_labels(interval,gt):
    s,e=map(float,interval);gs,ge=map(float,gt);assert e>s and ge>gs
    inter=max(0.,min(e,ge)-max(s,gs))
    return dict(precision=inter/(e-s),recall=inter/(ge-gs),tiou=inter/(e-s+ge-gs-inter))
def panel_summary(rows,domain):
    ans={}
    for recipe in RECIPES:
        packs={}
        for outcome in (['t','v'] if domain=='target' else ['t']):
            fields={f:binary(rows,recipe,f,outcome) for f in FIELDS}
            for f in FIELDS:
                fields[f]['orders']={o:binary([r for r in rows if r['order']==o],recipe,f,outcome,draws=0) for o in sorted({r['order'] for r in rows})}
            packs[outcome]=fields
        pairs=quadrants(rows,recipe)
        corr={}
        if domain=='target':
            pack=[dict(source_id=r['source_id'],P_A=r['readouts'][recipe]['P_A'],R_A=r['readouts'][recipe]['R_A'],
                delta_P=r['readouts'][recipe]['delta_P'],delta_R=r['readouts'][recipe]['delta_R'],
                true_P_A=r['true_P_A'],true_R_A=r['true_R_A'],A8_t=r['anchor_t'],dt=r['delta_t']) for r in rows]
            corr={f'{x}/{y}':correlation(pack,x,y) for x,y in [('P_A','true_P_A'),('R_A','true_R_A'),
                ('P_A','A8_t'),('R_A','A8_t'),('delta_P','dt'),('delta_R','dt')]}
        ans[recipe]=dict(discrimination=packs,quadrants=pairs,correlations=corr)
    coverage=dict(cells=len(rows),sources=len({r['source_id'] for r in rows}),eligible=sum(r['eligible'] for r in rows),
        noop=sum(not r['eligible'] for r in rows),helpful_t=sum(r['eligible'] and r['label_t']=='helpful' for r in rows),
        harmful_t=sum(r['eligible'] and r['label_t']=='harmful' for r in rows),neutral_t=sum(r['eligible'] and r['label_t']=='neutral' for r in rows),
        severe_v=sum(r['delta_v'] is not None and r['delta_v']< -.05 for r in rows))
    return dict(coverage=coverage,recipes=ans)

def diagnose():
    verify();tick=time.time();seal=read(OUT/'FEATURE_SEAL.json')
    assert sha(OUT/'FEATURE_ROWS.json')==seal['feature_sha256']
    original={r['cell_key']:r for r in read(OLD/'SCORE_ROWS.json')};rows=[];coverage=0
    for ds in DATASETS:
        plans=read(ROOT/f'artifacts/tastvg_current_correction_views_v1/{ds}/PLAN.json')['rows']
        gt={sp:read(ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_{sp}.json') for sp in ['search','confirm']}
        metrics={key(r):r for sp in ['search','confirm'] for r in read(OLD/sp/ds/'ROWS.json') if r['expert_scheduled']}
        coverage += sum(len(read(OLD/sp/ds/'ROWS.json')) for sp in ['search','confirm'])
        for r0 in read(OUT/'FEATURE_ROWS.json'):
            if r0['dataset']!=ds:continue
            r=dict(r0);m=metrics[r['cell_key']];a=r['anchor_index'];w=r['winner_index']
            assert m['anchor_index']==a
            r.update(anchor_t=float(m['candidate_t'][a]),winner_t=float(m['candidate_t'][w]),
                anchor_v=float(m['candidate_v'][a]),winner_v=float(m['candidate_v'][w]))
            assert abs(r['anchor_t']-m['A8_t'])<3e-12 and abs(r['anchor_v']-m['A8_v'])<3e-12
            ids=plans[r['source_id']]['frame_ids'];span=gt[r['split']][str(r['source_id'])]['span']
            scores=original[r['cell_key']];actual=[]
            for index in [a,w]:
                i,j=scores['candidate_indices'][index]
                actual.append(temporal_labels([ids[i],ids[j]+1],span))
            assert abs(actual[0]['tiou']-r['anchor_t'])<3e-12 and abs(actual[1]['tiou']-r['winner_t'])<3e-12
            r.update(true_P_A=actual[0]['precision'],true_R_A=actual[0]['recall'],true_P_W=actual[1]['precision'],true_R_W=actual[1]['recall'],
                true_delta_P=actual[1]['precision']-actual[0]['precision'],true_delta_R=actual[1]['recall']-actual[0]['recall'])
            r['delta_t']=r['winner_t']-r['anchor_t'];r['delta_v']=r['winner_v']-r['anchor_v']
            r['label_t']=label(r['delta_t']);r['label_v']=label(r['delta_v']);rows.append(r)
    assert coverage==1152 and len(rows)==288
    write(OUT/'ROWS.json',rows)
    source=[]
    for ds in DATASETS:
        atlas=payload(ds)
        for r in read(OLD/'SOURCE_ROWS.json'):
            if r['dataset']!=ds or r['split']!='validation':continue
            a=r['native_index'];f=feature(r['frozen_scores']['L'],a);w=f['top_index'] if f['eligible'] else a
            cell=f'source/{ds}/{r["source_id"]}'
            t0=r['candidate_t'][a];tw=r['candidate_t'][w]
            source.append(dict(dataset=ds,domain='source',split='validation',source_id=r['source_id'],cell_key=cell,
                condition='clean',order='source',anchor_index=a,winner_index=w,eligible=f['eligible'],anchor_t=t0,winner_t=tw,
                delta_t=tw-t0,delta_v=None,label_t=label(tw-t0),readouts=extract(atlas[cell],a,w),
                validation_previously_used_for_alpha=True,source_anchor='native'))
    assert len(source)==47;write(OUT/'SOURCE_ROWS.json',source)
    write(OUT/'LABEL_JOIN.json',dict(time=time.time(),feature_seal_time=seal['time'],target_GT_historical_exposure=True,
        target_GT_used_only_for_offline_diagnosis=True,feature_or_choice_fit_to_current_labels=False,
        raw_target_annotations_read=False,cached_span_labels_for_P_R=True,source_labels_reused_descriptive=True))
    for ds in DATASETS:
        groups={'source_validation':[r for r in source if r['dataset']==ds]}
        for sp in ['search','confirm']:
            for mode in ['corrupt','clean']:
                groups[sp+'_'+mode]=[r for r in rows if r['dataset']==ds and r['split']==sp and (r['condition']=='clean')==(mode=='clean')]
        summaries={}
        for name,rr in groups.items():
            summaries[name]=panel_summary(rr,'source' if name=='source_validation' else 'target')
            print('STRUCTURED_PANEL',ds,name,len(rr),flush=True)
        write(OUT/ds/'SUMMARY.json',summaries)
    examples={}
    for ds in DATASETS:
        for sp in ['search','confirm']:
            rr=[r for r in rows if r['dataset']==ds and r['split']==sp and r['condition']!='clean' and r['eligible']]
            helpful=sorted([r for r in rr if r['label_t']=='helpful'],key=lambda r:(-r['delta_t'],r['cell_key']))
            harmful=sorted([r for r in rr if r['label_t']=='harmful'],key=lambda r:(r['delta_v'],r['cell_key']))
            chosen={r['cell_key']:r for r in helpful[:3]+harmful[:3]}
            matched=[]
            for sid in sorted({r['source_id'] for r in helpful}&{r['source_id'] for r in harmful}):
                h=next(r for r in helpful if r['source_id']==sid);f=next(r for r in harmful if r['source_id']==sid)
                matched.append(dict(source_id=sid,helpful_cell=h['cell_key'],harmful_cell=f['cell_key']))
            examples[ds+'/'+sp]=dict(helpful=[r['cell_key'] for r in helpful[:3]],harmful=[r['cell_key'] for r in harmful[:3]],
                same_source_pairs=matched,cases=list(chosen.values()))
    write(OUT/'CASES.json',examples)
    write(OUT/'DECISION.json',dict(status='diagnostic_only_no_promotion',A_preserved=True,
        new_structured_model=False,new_gate=False,new_threshold=False,new_GPU_phase=False,
        scope='frozen separate univariate quantities and sign quadrants; not full multivariate separability or causal identification',
        target_historical_exposure=True,source_validation_reused=True,LOSO_is_influence_not_refit=True))
    write(OUT/'RESOURCES.json',dict(diagnosis_CPU_wall_seconds=time.time()-tick,
        extraction_CPU_wall_seconds=seal['CPU_wall_seconds'],new_GPU_calls=0,new_model_calls=0,new_expert_calls=0,
        new_probe_fits=0,new_backbone_calls=0,new_candidate_calls=0,new_replay_calls=0,new_backward_calls=0,
        target_readout_rows=288,source_validation_rows=47,nonexpert_new_readouts=0,
        timing_includes_bootstrap_and_serialization=True))
    status('completed_pending_root_audit_report_publication',target_expert_rows=288,source_validation_rows=47)
def main():
    action=sys.argv[1]
    {'prepare':prepare,'extract':extract_stage,'diagnose':diagnose}[action]()
    if 'torch' in sys.modules:assert not sys.modules['torch'].cuda.is_initialized()
if __name__=='__main__':
    try:main()
    except Exception:
        if BASE.exists():write(BASE/f'FAILURE_{sys.argv[1]}.json',dict(time=time.time(),traceback=traceback.format_exc()))
        status('failed',action=sys.argv[1]);raise
