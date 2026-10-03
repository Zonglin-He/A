"""Finite nested CPU diagnostic; no model/provider/inference imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_pr_nested_alpha_math_v1 import *
BASE=ROOT/'artifacts/tastvg_pr_nested_alpha_v1';OUT=ROOT/'results/tastvg_pr_nested_alpha/2026-10-03'
PRE=ROOT/'artifacts/tastvg_pr_loso_v1';OLD=ROOT/'results/tastvg_pr_loso/2026-10-03'
DATASETS=['vidstg','hc2']
OWN=['protocols/tastvg_pr_nested_alpha_v1.md','docs/tastvg_pr_nested_alpha_v1/EXECUTION.md',
    'scripts/run_tastvg_pr_nested_alpha_v1.py','scripts/tastvg_pr_nested_alpha_math_v1.py',
    'scripts/test_tastvg_pr_nested_alpha_v1.py','scripts/tastvg_pr_loso_math_v1.py',
    'scripts/tastvg_pr_role_math_v1.py','scripts/tastvg_pr_accessibility_math_v1.py']

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,z):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),str(p)
    p.write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def status(state,**kw):
    BASE.mkdir(parents=True,exist_ok=True);p=BASE/'STATUS.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(status=state,time=time.time(),pid=os.getpid(),**kw),indent=2)+'\n');t.replace(p)
def pack_path(ds,s):return PRE/'source_packs'/ds/f'{s:03d}.npz'
def model_path(f,pop):return BASE/'models'/f"{f['dataset']}_{f['held_source']:03d}_{pop}.npz"
def model_load(p):
    with np.load(p,allow_pickle=False) as z:
        return {n:{k:z[n+'_'+k].copy() for k in ['mean','std','weight','bias','alpha']} for n in PARTS}
def verify():
    z=read(BASE/'RUNTIME_LOCK.json')
    for p,h in {**z['code'],**z['inputs'],**z['payloads']}.items():assert sha(ROOT/p)==h,p
    assert sha(OUT/'CONFIG.json')==z['config_sha256'];return z
def guard(stage,allowed=None):
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=str(Path(args[0]).resolve())
        if any(s in p for s in ['/annotations/','/videos/','/media/','/ROWS.json','GT_LABELS_',
                '/target_features/confirm_']):raise PermissionError(stage+' access: '+p)
        if '/source_packs/' in p and (stage!='fit' or p not in allowed['paths']):
            raise PermissionError(stage+' outer labelled pack: '+p)
        if stage=='fit' and '/target_features/' in p:raise PermissionError(stage+' feature read: '+p)
    sys.addaudithook(hook)

def prepare():
    tick=time.time();assert not (BASE/'RUNTIME_LOCK.json').exists()
    done=read(PRE/'FINAL_COMPLETION.json')
    assert done['status']=='completed_and_publicly_verified' and done['commit']=='bf45294c2de6cce96ce808da080ecb79dfe4298f'
    oldcfg=read(OLD/'CONFIG.json');cohort=read(OLD/'COHORT.json');ff=[]
    assert len(cohort)==192 and all(r['split']=='search' for r in cohort)
    for ds in DATASETS:
        ids=sorted({r['source_id'] for r in cohort if r['dataset']==ds});ff+=nested_folds(ds,ids)
    controls=[{**{k:v for k,v in r.items() if k!='readouts'},
        'readouts':{p+'/fixed':r['readouts'][p+'/nmax'] for p in POPS}} for r in read(OLD/'PREDICTIONS.json')]
    write(OUT/'COHORT.json',cohort);write(OUT/'FOLDS.json',ff);write(BASE/'FIXED_READOUTS.json',controls)
    cfg={k:oldcfg[k] for k in ['views','latent','checkpoint_state_sha256','sampling','fixed_persistent_A',
        'production_method_sha256','weight_sum','normalizer','historically_exposed','held_source_unseen_by_readout_fit_only',
        'analytic_t','decision','bootstrap_draws','seed','epsilon','bootstrap_refits','bootstrap_scope']}
    cfg.update(version='tastvg_pr_nested_alpha_v1',predecessor_commit=done['commit'],old_alpha=oldcfg['alpha'],alpha_grid=GRID,
        outer_source_count={'vidstg':16,'hc2':14},outer_train_source_count={'vidstg':15,'hc2':13},
        inner_train_source_count={'vidstg':14,'hc2':12},inner_LOSO=True,populations=POPS,raw_regression=True,
        selection='separate P/R alpha; inner equal-source A/W MAE over all six conditions, exact ties smallest alpha',
        ridge_objective='equal-source weighted MSE + alpha*coefficient squared norm; unpenalized bias',
        ordered_inner_folds=422,inner_candidate_head_fits=11816,selected_outer_head_fits=120,
        new_scientific_CPU_fits=11936,eigendecompositions=1808,
        reused_fixed_LOSO_heads=120,source_symmetry_cache_reuse=False,confirmation_read_or_refitted=False,
        outer_labels_used_for_alpha_selection=False,target_search_GT_supervised_diagnostic_only=True,
        primary_panel='corruption raw A/W regression; nested minus same-cell fixed-alpha nmax LOSO',
        GPU_backbone_expert_candidate_replay_backward_online_production_calls=0)
    # 422 inner folds *2 populations *2 heads =1688 eigensystems, plus120 outer.
    write(OUT/'CONFIG.json',cfg)
    paths=[PRE/'FINAL_COMPLETION.json',PRE/'PACK_SEAL.json',OLD/'CONFIG.json',OLD/'COHORT.json',OLD/'FOLDS.json',
        OLD/'PREDICTIONS.json',OLD/'GLOBAL_READOUT_SEAL.json',OLD/'ROWS.json',OLD/'ROOT_AUDIT.json',ROOT/'methods/CURRENT_METHOD.json',
        OUT/'COHORT.json',OUT/'FOLDS.json',BASE/'FIXED_READOUTS.json']
    payloads={};packseal=read(PRE/'PACK_SEAL.json')
    for p,h in packseal['packs'].items():assert sha(ROOT/p)==h;payloads[p]=h
    for r in cohort:
        p=ROOT/r['feature_file'];h=read(p.with_suffix('.json'))['sha256'];assert sha(p)==h
        payloads[r['feature_file']]=h;paths.append(p.with_suffix('.json'))
    lock=dict(time=time.time(),code={p:sha(ROOT/p) for p in OWN},inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},
        payloads=payloads,config_sha256=sha(OUT/'CONFIG.json'),CPU_wall_seconds=time.time()-tick)
    write(BASE/'RUNTIME_LOCK.json',lock);write(OUT/'RUNTIME_BINDING.json',lock)
    status('prepared_pending_nested_CPU_fits',outer_folds=30,inner_folds=422)

def fit():
    verify();tick=time.time();allowed={'paths':set()};guard('fit',allowed)
    cohort=read(OUT/'COHORT.json');ledger=[];summaries=[];hashes={};fits=0;eigen=0
    for f in read(OUT/'FOLDS.json'):
        ds=f['dataset'];outer=f['held_source'];rr=[r for r in cohort if r['dataset']==ds]
        allowed['paths']={str(pack_path(ds,s).resolve()) for s in f['train_sources']};packs={}
        for s in f['train_sources']:
            with np.load(pack_path(ds,s),allow_pickle=False) as z:packs[s]={k:z[k].copy() for k in z.files}
        assert outer not in packs
        for pop in POPS:
            cv={n:[] for n in PARTS}
            for inner in f['inner_sources']:
                train={s:z for s,z in packs.items() if s!=inner};x,y,g=subset_arrays(train,pop)
                assert set(g)==set(f['train_sources'])-{inner} and outer not in g
                v=packs[inner];sr=[r for r in rr if r['source_id']==inner]
                assert v['keys'].tolist()==[r['cell_key'] for r in sr]
                ix=np.column_stack([v['anchors'],v['winners']]);vx=v['x'][np.arange(len(ix))[:,None],ix]
                for n,part in PARTS.items():
                    models=ridge_grid(x[:,part],y[n],g);eigen+=1;yy=v[n][np.arange(len(ix))[:,None],ix]
                    evaluations=[]
                    for m in models:
                        pp=predict(m,vx[:,:,part].reshape(-1,x[:,part].shape[1])).reshape(len(ix),2)
                        evaluations.append(dict(alpha=m['alpha'],**validation_errors(sr,yy,pp)))
                    cv[n].append(dict(inner_source=inner,train_sources=sorted(train),validation_cells=len(sr),
                        training_rows=len(g),training_source_count=len(train),scores=evaluations))
                    fits+=len(GRID)
            x,y,g=subset_arrays(packs,pop);models={};meta={};selection_at=time.time()
            for n,part in PARTS.items():
                curve=[float(np.mean([z['scores'][j]['mean'] for z in cv[n]])) for j in range(len(GRID))]
                alpha=select_alpha(curve);m=ridge_grid(x[:,part],y[n],g,[alpha])[0];eigen+=1;fits+=1
                models[n]=m;ww=source_weights(g);pp=predict(m,x[:,part])
                meta[n]=dict(alpha=alpha,inner_source_MAE=curve,selected_at=selection_at,
                    training_rows=len(g),training_sources=len(packs),training_MAE=float(ww@abs(y[n]-pp)),
                    training_MSE=float(ww@((y[n]-pp)**2)),coefficient_norm=float(np.linalg.norm(m['weight'])))
            path=model_path(f,pop);path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
            np.savez(path,**{n+'_'+k:m[k] for n,m in models.items() for k in ['mean','std','weight','bias','alpha']})
            hashes[str(path.relative_to(ROOT))]=sha(path)
            ledger.append(dict(**f,population=pop,heads=cv))
            summaries.append(dict(**f,population=pop,heads=meta,outer_labels_used=False,outer_refit_completed_at=time.time()))
        allowed['paths']=set();status('nested_CPU_fitting',fold=f['fold_key'],completed_head_fits=fits)
        print('NESTED_FIT',f['fold_key'],'candidate+selected heads',fits,flush=True)
    assert fits==11936 and eigen==1808
    write(OUT/'INNER_CV.json',ledger);write(OUT/'FIT_SUMMARY.json',summaries)
    write(BASE/'FIT_SEAL.json',dict(time=time.time(),models=hashes,inner_cv_sha256=sha(OUT/'INNER_CV.json'),
        selection_sha256=sha(OUT/'FIT_SUMMARY.json'),scientific_CPU_fits=fits,eigendecompositions=eigen,
        outer_source_label_packs_opened_in_fits=False,outer_labels_used_for_selection=False,confirmation_read=False,
        CPU_wall_seconds=time.time()-tick))
    status('selected_models_sealed_pending_GT_free_outer_readout',scientific_CPU_fits=fits)

def readout():
    verify();tick=time.time();fs=read(BASE/'FIT_SEAL.json')
    import torch
    torch.set_num_threads(1);assert not torch.cuda.is_initialized();guard('readout')
    fixed={r['cell_key']:r for r in read(BASE/'FIXED_READOUTS.json')};rows=[]
    cohort=read(OUT/'COHORT.json');ff=read(OUT/'FOLDS.json');models={}
    for f in ff:
        for pop in POPS:
            path=model_path(f,pop);assert sha(path)==fs['models'][str(path.relative_to(ROOT))]
            models[f['dataset'],f['held_source'],pop]=model_load(path)
    for r in cohort:
        z=torch.load(ROOT/r['feature_file'],map_location='cpu',weights_only=False)
        assert not z['GT_read'] and z['cell_key']==r['cell_key'] and z['source_id']==r['source_id']
        x=np.asarray(z['x'],float)[[r['anchor_index'],r['winner_index']]]
        rd=dict(fixed[r['cell_key']]['readouts'])
        for pop in POPS:
            mm=models[r['dataset'],r['source_id'],pop];vals={n:predict(mm[n],x[:,part]) for n,part in PARTS.items()}
            rd[pop+'/nested']={n+'_'+role:float(vals[n][i]) for n in PARTS for i,role in enumerate(['A','W'])}
            if r['anchor_index']==r['winner_index']:
                assert rd[pop+'/nested']['P_A']==rd[pop+'/nested']['P_W'] and rd[pop+'/nested']['R_A']==rd[pop+'/nested']['R_W']
        rows.append({**{k:v for k,v in r.items() if k!='feature_file'},'readouts':rd})
    assert not torch.cuda.is_initialized();write(OUT/'PREDICTIONS.json',rows)
    write(OUT/'GLOBAL_READOUT_SEAL.json',dict(time=time.time(),prediction_sha256=sha(OUT/'PREDICTIONS.json'),
        fit_seal_sha256=sha(BASE/'FIT_SEAL.json'),outer_GT_read=False,confirmation_read=False,CPU_wall_seconds=time.time()-tick))
    status('192_outer_readouts_sealed_pending_cached_GT_join',cells=192)

def paired_utility(a,b,draws):
    ids=list(a['utility_source_values']);assert ids==list(b['utility_source_values'])
    d=np.array([b['utility_source_values'][s] for s in ids])-np.array([a['utility_source_values'][s] for s in ids]);n=len(ids)
    bs=np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=draws)/n@d
    return {k:pack(d[:,j].mean(),bs[:,j]) for j,k in enumerate(['delta_t','delta_v','gross_v_gain','gross_v_loss','accept_rate'])}

def summarize(rr,draws=DRAWS):
    regs={};rb={};decs={};db={}
    for v in VARIANTS:
        regs[v]={};rb[v]={}
        for role in ROLES:
            regs[v][role],rb[v][role]=regression(rr,[[r['truth'][role]] for r in rr],[[r['readouts'][v][role]] for r in rr],draws)
            m=np.mean(list(regs[v][role]['source_moments'].values()),0);regs[v][role]['truth_variance']=float(max(0.,m[2]-m[0]**2))
        decs[v],db[v]=decision(rr,[r['analytic'][v]['delta_T'] for r in rr],draws)
    pairs=[(p+'/fixed',p+'/nested') for p in POPS]+[('all/nested','role/nested')];comparison={}
    for a,b in pairs:
        comparison[b+' minus '+a]=dict(regression={r:paired(regs[a][r],regs[b][r],rb[a][r],rb[b][r]) for r in ROLES},
            decision=paired(decs[a],decs[b],db[a],db[b]),utility=paired_utility(decs[a],decs[b],draws))
    orders={o:summarize([r for r in rr if r['order']==o],0) for o in sorted({r['order'] for r in rr})} if draws else {}
    return dict(cells=len(rr),sources=len({r['source_id'] for r in rr}),regression=regs,decision=decs,paired=comparison,
        orders=orders,draws=draws,seed=SEED)

def diagnose():
    verify();tick=time.time();fs=read(BASE/'FIT_SEAL.json');seal=read(OUT/'GLOBAL_READOUT_SEAL.json')
    assert fs['time']<seal['time'] and sha(OUT/'PREDICTIONS.json')==seal['prediction_sha256']
    rows=read(OUT/'PREDICTIONS.json');old={r['cell_key']:r for r in read(OLD/'ROWS.json')}
    for r in rows:
        prev=old[r['cell_key']]
        for k in ['anchor_index','winner_index','eligible','A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256']:assert r[k]==prev[k]
        r.update({k:prev[k] for k in ['truth','anchor_t','winner_t','anchor_v','winner_v','delta_t','delta_v']})
        for pop in POPS:assert r['readouts'][pop+'/fixed']==prev['readouts'][pop+'/nmax']
        r['analytic']={v:analytic_readout(r['readouts'][v]) for v in VARIANTS}
        for v in VARIANTS:r['analytic'][v]['accepted']=bool(r['eligible'] and r['analytic'][v]['delta_T']>0)
    write(OUT/'ROWS.json',rows);write(OUT/'LABEL_JOIN.json',dict(time=time.time(),readout_seal_time=seal['time'],
        cached_search_truth_joined=True,outer_labels_used_for_fit_or_selection=False,confirmation_read=False,raw_media_read=False))
    fits=read(OUT/'FIT_SUMMARY.json');distribution={};cases={}
    for ds in DATASETS:
        rr=[r for r in rows if r['dataset']==ds];ss={}
        for mode in ['all','corrupt','clean']:
            pr=[r for r in rr if mode=='all' or (r['condition']=='clean')==(mode=='clean')];ss[mode]=summarize(pr)
        ss['conditions']={c:summarize([r for r in rr if r['condition']==c],0) for c in sorted({r['condition'] for r in rr})}
        write(OUT/ds/'SUMMARY.json',ss);distribution[ds]={};selected={};cr=[r for r in rr if r['condition']!='clean']
        for pop in POPS:
            distribution[ds][pop]={n:{str(a):sum(f['heads'][n]['alpha']==a for f in fits if f['dataset']==ds and f['population']==pop)
                for a in GRID} for n in PARTS}
            def error(r,v):return np.mean([abs(r['readouts'][pop+'/'+v][k]-r['truth'][k]) for k in ROLES])
            for r in sorted(cr,key=lambda r:(error(r,'nested')-error(r,'fixed'),r['cell_key']))[:3]+sorted(cr,key=lambda r:(error(r,'fixed')-error(r,'nested'),r['cell_key']))[:3]:selected[r['cell_key']]=r
            changed=[r for r in cr if r['analytic'][pop+'/nested']['accepted']!=r['analytic'][pop+'/fixed']['accepted']]
            for r in sorted(changed,key=lambda r:(r['delta_v'],r['cell_key']))[:2]+sorted(changed,key=lambda r:(-r['delta_v'],r['cell_key']))[:2]:selected[r['cell_key']]=r
        cases[ds]=list(selected.values());print('NESTED_DIAGNOSIS',ds,'all/clean/corrupt/orders/conditions complete',flush=True)
    write(OUT/'ALPHA_DISTRIBUTION.json',distribution);write(OUT/'CASES.json',cases)
    write(OUT/'RESOURCES.json',dict(inner_candidate_head_fits=11816,selected_outer_head_fits=120,new_scientific_CPU_fits=11936,
        eigendecompositions=1808,private_outer_model_files=60,reused_fixed_head_count=120,
        fit_CPU_wall_seconds=fs['CPU_wall_seconds'],readout_CPU_wall_seconds=seal['CPU_wall_seconds'],diagnosis_CPU_wall_seconds=time.time()-tick,
        search_expert_cells=192,independent_outer_sources=30,GPU_backbone_expert_candidate_replay_backward_online_production_calls=0))
    write(OUT/'DECISION.json',dict(status='nested-alpha-supervised-diagnostic-no-promotion',A_preserved=True,CURRENT_preserved=True,
        new_representation_or_dimensionality_or_layer=False,new_threshold=False,confirmation_evaluated=False,next_experiment_started=False,
        scope='same fixed linear family; alpha chosen only by inner source A/W MAE; no unique causal inference'))
    status('completed_pending_independent_root_audit_review_publication',cells=192,scientific_CPU_fits=11936)

if __name__=='__main__':
    action=sys.argv[1];assert action in ['prepare','fit','readout','diagnose']
    try:globals()[action]()
    except Exception:status('failed',stage=action);raise
