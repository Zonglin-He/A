"""Finite target-supervised CPU readout diagnostic. No inference/providers."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import sys,json,time,hashlib,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_pr_accessibility_math_v1 import *
BASE=ROOT/'artifacts/tastvg_pr_accessibility_v1'
OUT=ROOT/'results/tastvg_pr_accessibility/2026-10-03'
CACHE=ROOT/'artifacts/tastvg_temporal_latent_quality_v1'
ATLAS=ROOT/'artifacts/tastvg_temporal_information_atlas_v1'
OLD=ROOT/'results/tastvg_temporal_latent_quality/2026-10-03'
PRIOR=ROOT/'results/tastvg_structured_separability/2026-10-03'
DATASETS=['vidstg','hc2']
PARTS={'P':slice(512,768),'R':slice(0,512)}
MODEL_NAMES={'P':'candidate/Inside/precision/real','R':'candidate/Endpoint/recall/real'}
OWN=['protocols/tastvg_pr_accessibility_v1.md','docs/tastvg_pr_accessibility_v1/EXECUTION.md',
     'scripts/run_tastvg_pr_accessibility_v1.py','scripts/tastvg_pr_accessibility_math_v1.py',
     'scripts/test_tastvg_pr_accessibility_v1.py']

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,z):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),str(p)
    p.write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def status(state,**kw):
    BASE.mkdir(parents=True,exist_ok=True);p=BASE/'STATUS.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(status=state,time=time.time(),pid=os.getpid(),**kw),indent=2)+'\n');t.replace(p)
def filename(r):return f'{r["split"]}_{r["condition"]}_{r["order"]}_{r["arrival"]:05}.pt'
def feature_path(r):return CACHE/r['dataset']/'target_features'/filename(r)
def verify():
    r=read(BASE/'RUNTIME_LOCK.json')
    code=dict(r['code']);revision=BASE/'RUNTIME_REVISION_001.json'
    if revision.exists():
        rev=read(revision);assert rev['prior_runtime_sha256']==sha(BASE/'RUNTIME_LOCK.json')
        code.update(rev['effective_code'])
    for name,h in {**code,**r['inputs'],**r['payloads']}.items():assert sha(ROOT/name)==h,name
    assert sha(OUT/'CONFIG.json')==r['config_sha256']
    return r
def torch_init():
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    return torch
def load_bound(path):
    torch=torch_init();assert sha(path)==read(path.with_suffix('.json'))['sha256']
    z=torch.load(path,map_location='cpu',weights_only=False)
    assert not torch.cuda.is_initialized()
    return z
def guarded(stage):
    def guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=str(args[0])
        forbidden=['/annotations/','/SOURCE_GT.json','/videos/','/media/']
        if stage=='fit':forbidden+=['GT_LABELS_confirm','/target_features/confirm_', '/ROWS.json']
        else:forbidden+=['GT_LABELS_','/ROWS.json']
        if any(s in p for s in forbidden):raise PermissionError(f'{stage} label/payload isolation: '+p)
    sys.addaudithook(guard)
def candidate_truth(z,gt):
    ids=z['frame_ids'];intervals=[[ids[i],ids[j]+1] for i,j in z['candidate_indices']]
    return interval_pr(intervals,gt)
def model_save(ds,models):
    p=BASE/ds/'SEARCH_PR.npz';p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
    data={name+'_'+field:np.asarray(value) for name,m in models.items()
          for field,value in m.items() if field not in ['model','intercept']}
    np.savez(p,**data)
    write(p.with_suffix('.json'),dict(sha256=sha(p),time=time.time(),runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        explicitly_target_search_GT_supervised=True))
    return p
def model_load(ds):
    p=BASE/ds/'SEARCH_PR.npz';assert sha(p)==read(p.with_suffix('.json'))['sha256']
    with np.load(p,allow_pickle=False) as z:
        return {name:{**{k:z[name+'_'+k].copy() for k in ['mean','std','weight','bias','alpha']},
            'model':'ridge','intercept':True} for name in PARTS}

def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    previous=ROOT/'artifacts/tastvg_structured_separability_v1/FINAL_COMPLETION.json'
    assert read(previous)['status']=='completed_verified_publication'
    assert read(previous)['commit']=='f5e3f2d4758f50379ac94baa2b3e16c293d652dd'
    scores=read(OLD/'SCORE_ROWS.json');prior={r['cell_key']:r for r in read(PRIOR/'FEATURE_ROWS.json')}
    rows=[];paths=[ROOT/'methods/CURRENT_METHOD.json',previous,OLD/'SCORE_ROWS.json',OLD/'CONFIG.json',
        PRIOR/'FEATURE_ROWS.json',PRIOR/'FEATURE_SEAL.json',PRIOR/'ROWS.json',PRIOR/'ROOT_AUDIT.json',
        ATLAS/'FIT_SEAL.json',ATLAS/'GLOBAL_READOUT_SEAL.json']
    paths += [OLD/sp/ds/'ROWS.json' for sp in ['search','confirm'] for ds in DATASETS]
    payloads={};alphas={}
    for ds in DATASETS:
        fs=read(ATLAS/ds/'FIT_SUMMARY.json')
        alphas[ds]={k:fs[v]['selected_alpha'] for k,v in MODEL_NAMES.items()}
        paths += [ATLAS/ds/n for n in ['FIT_SUMMARY.json','FROZEN_ATLAS.json','FROZEN_ATLAS.pt','SEALED_READOUT.json','SEALED_READOUT.pt']]
        paths += [ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_{sp}.json' for sp in ['search','confirm']]
        source_sets={sp:set() for sp in ['search','confirm']}
        for r in scores:
            if r['dataset']!=ds:continue
            a=prior[r['cell_key']];f=feature_path(r)
            receipt=read(f.with_suffix('.json'));assert sha(f)==receipt['sha256']
            payloads[str(f.relative_to(ROOT))]=receipt['sha256'];paths.append(f.with_suffix('.json'))
            keep={k:r[k] for k in ['dataset','split','cell_key','source_id','condition','order','arrival','anchor_index',
                'A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256']}
            keep.update(winner_index=a['winner_index'],eligible=a['eligible'],feature_file=str(f.relative_to(ROOT)))
            rows.append(keep);source_sets[r['split']].add(r['source_id'])
        assert not source_sets['search']&source_sets['confirm']
        assert sum(r['dataset']==ds and r['split']=='search' for r in rows)==96
        assert sum(r['dataset']==ds and r['split']=='confirm' for r in rows)==48
    assert len(rows)==288 and alphas=={'vidstg':{'P':1.,'R':1.},'hc2':{'P':1.,'R':.1}}
    write(OUT/'COHORT.json',rows)
    config=dict(version='tastvg_pr_accessibility_v1',predecessor_commit=read(previous)['commit'],
        source_fit_alpha=alphas,latent='same cached final sixth temporal decoder hidden',
        views={'P':'Inside[512:768]','R':'Endpoint[0:512]'},training='all 32 candidates of search expert cells, clean and corruption',
        target_training_rows_per_dataset=3072,target_training_cells_per_dataset=96,
        planned_sources={'search':32,'confirm':16},actual_expert_sources={'vidstg':{'search':16,'confirm':8},'hc2':{'search':14,'confirm':7}},
        source_probe_train_sources={'vidstg':95,'hc2':48},source_alpha_validation_sources={'vidstg':31,'hc2':16},
        source_equal_weight=True,normalization='same weighted-train mean/std procedure, re-estimated on target-search only',
        ridge_objective='source-weighted MSE + fixed alpha*coefficient squared norm; unpenalized intercept',
        target_supervision='diagnostic search GT fitting only; confirm joins after model/readout seals',
        historically_exposed=True,confirmation_fresh=False,readout_regression_raw=True,
        analytic_t='clip P/R to [0,1]; PR/(P+R-PR); zero denominator ->0',
        decision='fixed W accepted iff eligible and predicted analytic delta_T>0',
        winner_frozen=True,alpha_tuning=False,ladders=LADDERS,fits=FITS,
        epsilon_true_labels=EPS,bootstrap_draws=DRAWS,seed=SEED,
        source_weighting='condition -> order -> equal source; binary AUC equal source within each class',
        checkpoint_state_sha256=read(OLD/'CONFIG.json')['checkpoint_state_sha256'],sampling='original Paper48 two-offset',
        fixed_persistent_A='Uniform Rank-RKL 1792 parameters; VidK1 HCK8',
        new_GPU_backbone_expert_candidate_replay_backward_method_updates=0,new_diagnostic_ridge_fits=4,
        production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'))
    write(OUT/'CONFIG.json',config)
    paths.append(OUT/'COHORT.json');pins={str(p.relative_to(ROOT)):sha(p) for p in paths}
    lock=dict(time=time.time(),code={p:sha(ROOT/p) for p in OWN},inputs=pins,payloads=payloads,config_sha256=sha(OUT/'CONFIG.json'))
    write(BASE/'RUNTIME_LOCK.json',lock)
    write(OUT/'RUNTIME_BINDING.json',lock)
    status('prepared_pending_search_supervised_fit',cells=288,diagnostic_GPU_calls=0)
    print('PR_ACCESSIBILITY_LOCKED',len(payloads),'cached payloads',flush=True)

def fit():
    verify();torch_init();tick=time.time();guarded('fit');cohort=read(OUT/'COHORT.json');cfg=read(OUT/'CONFIG.json')
    summaries={};models_sha={}
    for ds in DATASETS:
        rr=[r for r in cohort if r['dataset']==ds and r['split']=='search']
        spans=read(ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_search.json')
        xx=[];yy={'P':[],'R':[]};groups=[]
        for r in rr:
            z=load_bound(ROOT/r['feature_file']);assert z['split']=='search' and z['cell_key']==r['cell_key']
            assert z['source_id']==r['source_id'] and z['anchor_index']==r['anchor_index']
            assert z['source_A_temporal_bitwise_parity'] and z['A_spatial_bitwise_parity'] and not z['GT_read']
            p,rec,t=candidate_truth(z,spans[str(r['source_id'])]['span'])
            xx.append(np.asarray(z['x'],float));yy['P'].append(p);yy['R'].append(rec);groups.extend([r['source_id']]*32)
        x=np.concatenate(xx);assert x.shape==(3072,1792)
        models={};details={}
        for name,part in PARTS.items():
            y=np.concatenate(yy[name]);alpha=cfg['source_fit_alpha'][ds][name]
            m=fit_ridge(x[:,part],y,groups,alpha);models[name]=m
            w=source_weights(groups);a=(x[:,part]-m['mean'])/m['std'];pr=predict(m,x[:,part])
            residual=a.T@(w*(pr-y))+alpha*m['weight']
            assert np.max(abs(residual))<1e-10 and abs(w@(pr-y))<1e-10
            details[name]=dict(alpha=alpha,dimensions=part.stop-part.start,training_rows=len(y),
                training_sources=len(set(groups)),training_cells=len(rr),
                weighted_training_MSE=float(w@((y-pr)**2)),weighted_training_MAE=float(w@abs(y-pr)),
                normal_equation_error=float(np.max(abs(residual))),intercept_error=float(abs(w@(pr-y))),
                confirmation_GT_used=False,confirmation_features_used=False,
                fitted_normalizer_hash=hashlib.sha256(m['mean'].tobytes()+m['std'].tobytes()).hexdigest())
        path=model_save(ds,models);models_sha[ds]=sha(path);summaries[ds]=details
        print('PR_SEARCH_FIT',ds,'models2 sources',len(set(groups)),'rows3072',flush=True)
    write(OUT/'FIT_SUMMARY.json',summaries)
    write(BASE/'FIT_SEAL.json',dict(status='four_search_supervised_ridge_fits_frozen',time=time.time(),
        models=models_sha,alphas_fixed=cfg['source_fit_alpha'],confirmation_GT_read=False,confirmation_features_read=False,
        target_search_GT_for_fit=True,CPU_wall_seconds=time.time()-tick,runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json')))
    status('fit_sealed_pending_GT_free_readout',models=4)

def readout():
    verify();torch_init();tick=time.time();fitseal=read(BASE/'FIT_SEAL.json');guarded('readout')
    cohort=read(OUT/'COHORT.json');rows=[];parity=0.;count=0
    for ds in DATASETS:
        assert sha(BASE/ds/'SEARCH_PR.npz')==fitseal['models'][ds]
        m=model_load(ds);oldm=load_bound(ATLAS/ds/'FROZEN_ATLAS.pt')['models']
        oldscores={r['cell']:r['predictions'] for r in load_bound(ATLAS/ds/'SEALED_READOUT.pt')}
        for r in cohort:
            if r['dataset']!=ds:continue
            z=load_bound(ROOT/r['feature_file']);assert z['cell_key']==r['cell_key'] and z['split']==r['split']
            x=np.asarray(z['x'],float);pred={}
            for fitname in FITS:
                pred[fitname]={}
                for name,part in PARTS.items():
                    model=oldm[MODEL_NAMES[name]] if fitname=='source_fit' else m[name]
                    values=predict(model,x[:,part]);assert values.shape==(32,) and np.isfinite(values).all()
                    pred[fitname][name]=values.tolist()
                    if fitname=='source_fit':
                        error=float(np.max(abs(values-oldscores[r['cell_key']][MODEL_NAMES[name]])))
                        assert error<3e-12;parity=max(parity,error);count+=32
            out={k:v for k,v in r.items() if k!='feature_file'};out['predictions']=pred;rows.append(out)
        print('PR_READOUT',ds,'144 cells; source parity verified',flush=True)
    assert len(rows)==288
    write(OUT/'PREDICTIONS.json',rows)
    write(OUT/'GLOBAL_READOUT_SEAL.json',dict(status='all_scalar_readouts_sealed',time=time.time(),cells=288,
        prediction_sha256=sha(OUT/'PREDICTIONS.json'),fit_seal_sha256=sha(BASE/'FIT_SEAL.json'),fit_models=fitseal['models'],
        confirmation_GT_read=False,source_prediction_parity_max=parity,source_parity_scalars=count,
        CPU_wall_seconds=time.time()-tick))
    status('readouts_sealed_pending_confirm_GT_join',cells=288)

def roles(p,idx):
    a,w=idx
    return dict(P_A=p['P'][a],R_A=p['R'][a],P_W=p['P'][w],R_W=p['R'][w])

def summarize(rr,draws=DRAWS):
    regs={};regboots={};dec={};decboots={};delta={}
    for fn in FITS:
        regs[fn]={};regboots[fn]={};dec[fn]={};decboots[fn]={}
        for field in ROLES+['P_pool','R_pool','T_A','T_W','delta_T']:
            if field in ['P_pool','R_pool']:
                name=field[0];yy=[r['true_candidates'][name] for r in rr];pp=[r['predictions'][fn][name] for r in rr]
            elif field in ROLES:
                yy=[[r['truth'][field]] for r in rr];pp=[[r['readouts'][fn][field]] for r in rr]
            else:
                yy=[[r['anchor_t'] if field=='T_A' else r['winner_t'] if field=='T_W' else r['delta_t']] for r in rr]
                pp=[[r['ladder'][fn]['predicted'][field]] for r in rr]
            regs[fn][field],regboots[fn][field]=regression(rr,yy,pp,draws)
        for arm in LADDERS:
            sc=[r['ladder'][fn][arm]['delta_T'] for r in rr]
            dec[fn][arm],decboots[fn][arm]=decision(rr,sc,draws)
    paired_reg={};paired_dec={};ladder_effect={}
    for field in regs['source_fit']:
        paired_reg[field]={}
        for k in regs['source_fit'][field]['metrics']:
            p0=regs['source_fit'][field]['metrics'][k]['mean'];p1=regs['target_search_fit'][field]['metrics'][k]['mean']
            samples=regboots['target_search_fit'][field][k]-regboots['source_fit'][field][k]
            paired_reg[field][k]=pack(p1-p0 if p0 is not None and p1 is not None else np.nan,samples)
    for arm in LADDERS:
        paired_dec[arm]={}
        for k in dec['source_fit'][arm]['metrics']:
            p0=dec['source_fit'][arm]['metrics'][k]['mean'];p1=dec['target_search_fit'][arm]['metrics'][k]['mean']
            paired_dec[arm][k]=pack(p1-p0 if p0 is not None and p1 is not None else np.nan,
                decboots['target_search_fit'][arm][k]-decboots['source_fit'][arm][k])
    for fn in FITS:
        ladder_effect[fn]={}
        for arm in LADDERS[1:]:
            ladder_effect[fn][arm]={}
            for k in dec[fn][arm]['metrics']:
                a=dec[fn][arm]['metrics'][k]['mean'];b=dec[fn]['predicted']['metrics'][k]['mean']
                ladder_effect[fn][arm][k]=pack(a-b if a is not None and b is not None else np.nan,
                    decboots[fn][arm][k]-decboots[fn]['predicted'][k])
    orders={}
    if draws:
        for order in sorted({r['order'] for r in rr}):
            orders[order]=summarize([r for r in rr if r['order']==order],draws=0)
    return dict(cells=len(rr),sources=len({r['source_id'] for r in rr}),regression=regs,decision=dec,
        paired_target_minus_source_regression=paired_reg,paired_target_minus_source_decision=paired_dec,
        ladder_minus_predicted=ladder_effect,orders=orders,draws=draws,seed=SEED)

def diagnose():
    verify();tick=time.time();seal=read(OUT/'GLOBAL_READOUT_SEAL.json');fit=read(BASE/'FIT_SEAL.json')
    assert fit['time']<seal['time'] and sha(OUT/'PREDICTIONS.json')==seal['prediction_sha256']
    prior={r['cell_key']:r for r in read(PRIOR/'ROWS.json')};predictions=read(OUT/'PREDICTIONS.json');rows=[]
    formula_error=0.
    for ds in DATASETS:
        labels={sp:read(ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_{sp}.json') for sp in ['search','confirm']}
        metric={'/'.join(str(r[k]) for k in ['dataset','split','condition','order','arrival']):r
                for sp in ['search','confirm'] for r in read(OLD/sp/ds/'ROWS.json') if r['expert_scheduled']}
        cohort={r['cell_key']:r for r in read(OUT/'COHORT.json')}
        for r0 in predictions:
            if r0['dataset']!=ds:continue
            r=dict(r0);z=load_bound(feature_path(cohort[r['cell_key']]))
            p,rec,t=candidate_truth(z,labels[r['split']][str(r['source_id'])]['span'])
            a,w=r['anchor_index'],r['winner_index'];m=metric[r['cell_key']]
            err=float(np.max(abs(t-m['candidate_t'])));assert err<3e-12;formula_error=max(formula_error,err)
            r['true_candidates']={'P':p.tolist(),'R':rec.tolist(),'T':t.tolist()}
            truth=roles(r['true_candidates'],(a,w));r['truth']=truth
            r.update(anchor_t=float(m['candidate_t'][a]),winner_t=float(m['candidate_t'][w]),
                anchor_v=float(m['candidate_v'][a]),winner_v=float(m['candidate_v'][w]))
            r['delta_t']=r['winner_t']-r['anchor_t'];r['delta_v']=r['winner_v']-r['anchor_v']
            r['readouts']={fn:roles(r['predictions'][fn],(a,w)) for fn in FITS}
            r['ladder']={fn:{arm:ladder(r['readouts'][fn],truth,arm) for arm in LADDERS} for fn in FITS}
            old=prior[r['cell_key']]
            assert a==old['anchor_index'] and w==old['winner_index'] and r['eligible']==old['eligible']
            for k in ROLES:
                assert abs(truth[k]-old['true_'+k])<3e-12
                assert abs(r['readouts']['source_fit'][k]-old['readouts']['Inside_Endpoint'][k])<3e-12
            assert abs(r['ladder']['source_fit']['GT_all']['T_A']-r['anchor_t'])<3e-12
            assert abs(r['ladder']['source_fit']['GT_all']['T_W']-r['winner_t'])<3e-12
            for fn in FITS:
                if a==w:assert r['ladder'][fn]['predicted']['delta_T']==0.
            rows.append(r)
    assert len(rows)==288
    write(OUT/'ROWS.json',rows)
    write(OUT/'LABEL_JOIN.json',dict(time=time.time(),fit_seal_time=fit['time'],readout_seal_time=seal['time'],
        confirmation_GT_for_fit=False,search_GT_for_fit=True,GT_raw_annotations_read=False,
        all_GT_formula_max_error=formula_error,confirmation_historical_exposure=True,
        no_first_ever_blindness_claim=True,readout_sha256=seal['prediction_sha256']))
    for ds in DATASETS:
        ss={}
        for sp in ['search','confirm']:
            for mode in ['all','corrupt','clean']:
                rr=[r for r in rows if r['dataset']==ds and r['split']==sp and
                    (mode=='all' or (r['condition']=='clean')==(mode=='clean'))]
                ss[sp+'_'+mode]=summarize(rr)
                print('PR_PANEL',ds,sp,mode,len(rr),flush=True)
        write(OUT/ds/'SUMMARY.json',ss)
    cases={}
    for ds in DATASETS:
        rr=[r for r in rows if r['dataset']==ds and r['split']=='confirm' and r['condition']!='clean' and r['eligible']]
        good=sorted([r for r in rr if r['delta_t']>EPS],key=lambda r:(-r['delta_t'],r['cell_key']))[:3]
        bad=sorted([r for r in rr if r['delta_t']< -EPS],key=lambda r:(r['delta_v'],r['cell_key']))[:3]
        selected={r['cell_key']:r for r in good+bad}
        for r in rr:
            if ds=='vidstg' and (r['source_id'],r['condition'],r['order']) in [(37,'exposure_5','order2'),(34,'frame_drop_5','order1')]:selected[r['cell_key']]=r
        cases[ds]=list(selected.values())
    write(OUT/'CASES.json',cases)
    write(OUT/'DECISION.json',dict(status='supervised_diagnostic_only_no_promotion',A_preserved=True,CURRENT_preserved=True,
        target_fit_oracle_ladder_not_unsupervised_TTA=True,trained_gate=False,alpha_search=False,new_GPU_phase=False,
        causal_scope='fixed family/alpha and small train sources; mapping recovery/failure cannot uniquely assign root cause',
        next_experiment_started=False))
    write(OUT/'RESOURCES.json',dict(search_fit_CPU_wall_seconds=fit['CPU_wall_seconds'],
        readout_CPU_wall_seconds=seal['CPU_wall_seconds'],diagnosis_CPU_wall_seconds=time.time()-tick,
        target_supervised_ridge_fits=4,new_GPU_calls=0,new_backbone_calls=0,new_expert_calls=0,new_candidate_calls=0,
        new_replay_calls=0,new_backward_calls=0,production_updates=0,new_full_online_streams=0,
        cached_payloads=288,candidate_readout_rows=9216,target_training_rows=6144,confirmation_candidate_rows=3072))
    status('completed_pending_root_audit_report_publication',cells=288,trained_readouts=4)

def main():
    action=sys.argv[1]
    {'prepare':prepare,'fit':fit,'readout':readout,'diagnose':diagnose}[action]()
    if 'torch' in sys.modules:assert not sys.modules['torch'].cuda.is_initialized()
if __name__=='__main__':
    try:main()
    except Exception:
        if BASE.exists():
            p=BASE/'FAILURE.json'
            if not p.exists():write(p,dict(time=time.time(),action=sys.argv[1],traceback=traceback.format_exc()))
            status('failed',action=sys.argv[1])
        raise
