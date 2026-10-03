"""Finite CPU-only population audit, isolated from inference and providers."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import sys,json,time,hashlib,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_pr_role_math_v1 import *
BASE=ROOT/'artifacts/tastvg_pr_role_v1'
OUT=ROOT/'results/tastvg_pr_role/2026-10-03'
PREV=ROOT/'artifacts/tastvg_pr_accessibility_v1'
OLD=ROOT/'results/tastvg_pr_accessibility/2026-10-03'
DATASETS=['vidstg','hc2'];PARTS={'P':slice(512,768),'R':slice(0,512)}
OWN=['protocols/tastvg_pr_role_v1.md','docs/tastvg_pr_role_v1/EXECUTION.md',
     'scripts/run_tastvg_pr_role_v1.py','scripts/tastvg_pr_role_math_v1.py',
     'scripts/test_tastvg_pr_role_v1.py','scripts/tastvg_pr_accessibility_math_v1.py']

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
def verify():
    z=read(BASE/'RUNTIME_LOCK.json')
    for p,h in {**z['code'],**z['inputs'],**z['payloads']}.items():assert sha(ROOT/p)==h,p
    assert sha(OUT/'CONFIG.json')==z['config_sha256']
    return z
def torch_init():
    import torch
    torch.set_num_threads(1);assert not torch.cuda.is_initialized()
    return torch
def load_bound(p):
    torch=torch_init();assert sha(p)==read(p.with_suffix('.json'))['sha256']
    z=torch.load(p,map_location='cpu',weights_only=False);assert not torch.cuda.is_initialized();return z
def guard(stage):
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=str(args[0]);forbidden=['/annotations/','/videos/','/media/','/SOURCE_GT.json','/ROWS.json']
        if stage=='fit':forbidden+=['GT_LABELS_confirm','/target_features/confirm_']
        else:forbidden+=['GT_LABELS_']
        if any(s in p for s in forbidden):raise PermissionError(stage+' isolation: '+p)
    sys.addaudithook(hook)
def truths(z,span):
    ids=z['frame_ids'];return interval_pr([[ids[i],ids[j]+1] for i,j in z['candidate_indices']],span)
def save_model(ds,models):
    p=BASE/ds/'ROLE_PR.npz';p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
    np.savez(p,**{n+'_'+k:np.asarray(v) for n,m in models.items() for k,v in m.items() if k not in ['model','intercept']})
    write(p.with_suffix('.json'),dict(time=time.time(),sha256=sha(p),diagnostic_target_search_GT_supervised=True))
    return p
def load_model(path):
    assert sha(path)==read(path.with_suffix('.json'))['sha256']
    with np.load(path,allow_pickle=False) as z:
        return {n:{**{k:z[n+'_'+k].copy() for k in ['mean','std','weight','bias','alpha']},
                   'model':'ridge','intercept':True} for n in PARTS}

def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    completed=read(PREV/'FINAL_COMPLETION.json')
    assert completed['status']=='completed_and_publicly_verified' and completed['commit']=='d408f154fb753764fef4aac2ca510f9b3b8fd36a'
    cohort=read(OLD/'COHORT.json');previous=read(OLD/'CONFIG.json')
    assert len(cohort)==288
    pop=[];payloads={};paths=[PREV/'FINAL_COMPLETION.json',PREV/'FIT_SEAL.json',OLD/'CONFIG.json',
        OLD/'COHORT.json',OLD/'FIT_SUMMARY.json',OLD/'PREDICTIONS.json',OLD/'GLOBAL_READOUT_SEAL.json',
        OLD/'ROWS.json',OLD/'ROOT_AUDIT.json',ROOT/'methods/CURRENT_METHOD.json']
    for ds in DATASETS:
        search=[r for r in cohort if r['dataset']==ds and r['split']=='search']
        confirm=[r for r in cohort if r['dataset']==ds and r['split']=='confirm']
        assert len(search)==96 and len(confirm)==48
        assert not {r['source_id'] for r in search}&{r['source_id'] for r in confirm}
        paths += [PREV/ds/'SEARCH_PR.npz',PREV/ds/'SEARCH_PR.json']
        paths += [ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_{sp}.json' for sp in ['search','confirm']]
        for r in search+confirm:
            p=ROOT/r['feature_file'];h=read(p.with_suffix('.json'))['sha256'];assert sha(p)==h
            payloads[r['feature_file']]=h;paths.append(p.with_suffix('.json'))
        for r in search:
            ix=training_indices(r['anchor_index'],r['winner_index'])
            pop.append(dict(dataset=ds,cell_key=r['cell_key'],source_id=r['source_id'],indices=ix.tolist(),
                roles=['A','W'],duplicate_roles=bool(ix[0]==ix[1])))
    write(OUT/'COHORT.json',cohort);write(OUT/'TRAINING_POPULATION.json',pop)
    cfg=dict(version='tastvg_pr_role_v1',predecessor_commit=completed['commit'],fits=FITS,
        alpha=previous['source_fit_alpha'],views=previous['views'],latent=previous['latent'],
        planned_sources=previous['planned_sources'],actual_expert_sources=previous['actual_expert_sources'],
        checkpoint_state_sha256=previous['checkpoint_state_sha256'],sampling=previous['sampling'],
        fixed_persistent_A=previous['fixed_persistent_A'],production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        populations={'all':'reuse target-search 32-candidate model','role':'ordered A,W pair per cell, keep duplicates'},
        training_rows_per_dataset={'all':3072,'role':192},training_cells_per_dataset=96,
        new_scientific_fits=4,reused_all_fits=4,
        normalized_source_weight_sum=1.,ridge_objective=previous['ridge_objective'],
        normalizer='identical weighted search-only procedure; re-estimate statistics on role population',
        target_search_GT_supervised_diagnostic_only=True,confirmation_fresh=False,historically_exposed=True,
        confirm_used_for_fit=False,role_input_feature=False,separate_A_W_heads=False,
        alpha_search=False,threshold_search=False,winner_reselected=False,oracle_ladder_added=False,
        analytic_t=previous['analytic_t'],decision=previous['decision'],raw_regression=True,
        bootstrap_draws=DRAWS,seed=SEED,epsilon=EPS,
        primary_panel='confirmation corruption; clean/search/orders/influence separately',
        new_GPU_backbone_expert_candidate_replay_backward_online_production_calls=0)
    write(OUT/'CONFIG.json',cfg);paths += [OUT/'COHORT.json',OUT/'TRAINING_POPULATION.json']
    lock=dict(time=time.time(),code={p:sha(ROOT/p) for p in OWN},
        inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},payloads=payloads,config_sha256=sha(OUT/'CONFIG.json'))
    write(BASE/'RUNTIME_LOCK.json',lock);write(OUT/'RUNTIME_BINDING.json',lock)
    status('prepared_pending_search_role_fit',cells=288,new_GPU_calls=0)
    print('ROLE_LOCKED 288 cells; 192 training rows/dataset',flush=True)

def fit():
    verify();torch_init();tick=time.time();guard('fit')
    cohort=read(OUT/'COHORT.json');cfg=read(OUT/'CONFIG.json');summaries={};modelhash={}
    for ds in DATASETS:
        rr=[r for r in cohort if r['dataset']==ds and r['split']=='search']
        gt=read(ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_search.json')
        xx=[];yy={'P':[],'R':[]};groups=[];noops=0
        for r in rr:
            z=load_bound(ROOT/r['feature_file']);assert z['split']=='search' and z['cell_key']==r['cell_key']
            assert z['source_id']==r['source_id'] and z['anchor_index']==r['anchor_index']
            assert z['source_A_temporal_bitwise_parity'] and z['A_spatial_bitwise_parity'] and not z['GT_read']
            ix=training_indices(r['anchor_index'],r['winner_index']);p,rec,t=truths(z,gt[str(r['source_id'])]['span'])
            xx.append(np.asarray(z['x'],float)[ix]);yy['P'].append(p[ix]);yy['R'].append(rec[ix])
            groups.extend([r['source_id']]*2);noops+=int(ix[0]==ix[1])
        x=np.vstack(xx);assert x.shape==(192,1792);models={};details={}
        for name,part in PARTS.items():
            y=np.concatenate(yy[name]);alpha=cfg['alpha'][ds][name]
            m=fit_ridge(x[:,part],y,groups,alpha);models[name]=m
            w=source_weights(groups);a=(x[:,part]-m['mean'])/m['std'];pr=predict(m,x[:,part])
            residual=a.T@(w*(pr-y))+alpha*m['weight']
            assert max(abs(residual))<1e-10 and abs(w@(pr-y))<1e-10
            details[name]=dict(alpha=alpha,training_rows=192,training_cells=96,training_sources=len(set(groups)),
                repeated_A_W_cells=noops,weight_sum=float(w.sum()),dimensions=part.stop-part.start,
                weighted_training_MSE=float(w@((y-pr)**2)),weighted_training_MAE=float(w@abs(y-pr)),
                normal_equation_error=float(max(abs(residual))),intercept_error=float(abs(w@(pr-y))),
                confirmation_features_used=False,confirmation_labels_used=False)
        path=save_model(ds,models);modelhash[ds]=sha(path);summaries[ds]=details
        print('ROLE_FIT',ds,'192 rows, 2 heads; sources',len(set(groups)),flush=True)
    write(OUT/'FIT_SUMMARY.json',summaries)
    write(BASE/'FIT_SEAL.json',dict(time=time.time(),status='four_role_models_frozen',models=modelhash,
        reused_all_models={ds:sha(PREV/ds/'SEARCH_PR.npz') for ds in DATASETS},
        target_search_GT_supervised=True,confirmation_features_read=False,confirmation_GT_read=False,
        CPU_wall_seconds=time.time()-tick,runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json')))
    status('fit_sealed_pending_GT_free_readout',models=4)

def readout():
    verify();torch_init();tick=time.time();seal=read(BASE/'FIT_SEAL.json');guard('readout')
    cohort=read(OUT/'COHORT.json');old={r['cell_key']:r for r in read(OLD/'PREDICTIONS.json')}
    rows=[];maxerror=0.;count=0
    for ds in DATASETS:
        assert sha(BASE/ds/'ROLE_PR.npz')==seal['models'][ds]
        assert sha(PREV/ds/'SEARCH_PR.npz')==seal['reused_all_models'][ds]
        models={'all':load_model(PREV/ds/'SEARCH_PR.npz'),'role':load_model(BASE/ds/'ROLE_PR.npz')}
        for r in cohort:
            if r['dataset']!=ds:continue
            z=load_bound(ROOT/r['feature_file']);assert z['cell_key']==r['cell_key'] and z['split']==r['split']
            x=np.asarray(z['x'],float);pp={}
            for fn in FITS:
                pp[fn]={}
                for name,part in PARTS.items():
                    values=predict(models[fn][name],x[:,part]);assert values.shape==(32,) and np.isfinite(values).all()
                    pp[fn][name]=values.tolist()
                    if fn=='all':
                        err=float(max(abs(values-np.array(old[r['cell_key']]['predictions']['target_search_fit'][name]))))
                        assert err<3e-12;maxerror=max(maxerror,err);count+=32
            out={k:v for k,v in r.items() if k!='feature_file'};out['predictions']=pp;rows.append(out)
        print('ROLE_READOUT',ds,'144 cells, M-all parity passed',flush=True)
    write(OUT/'PREDICTIONS.json',rows)
    write(OUT/'GLOBAL_READOUT_SEAL.json',dict(time=time.time(),status='288_cell_readouts_frozen',
        prediction_sha256=sha(OUT/'PREDICTIONS.json'),fit_seal_sha256=sha(BASE/'FIT_SEAL.json'),
        all_prediction_parity_max=maxerror,all_prediction_parity_scalars=count,
        confirmation_GT_read=False,CPU_wall_seconds=time.time()-tick))
    status('readout_sealed_pending_confirm_GT_join',cells=288)

def paired_utility(a,b,draws):
    ids=list(a['utility_source_values']);assert ids==list(b['utility_source_values'])
    x=np.array([a['utility_source_values'][s] for s in ids]);y=np.array([b['utility_source_values'][s] for s in ids])
    n=len(ids);w=np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=draws)/n
    d=y-x;bs=w@d
    return {k:pack(d[:,j].mean(),bs[:,j]) for j,k in enumerate(['delta_t','delta_v','gross_v_gain','gross_v_loss','accept_rate'])}

def summarize(rr,draws=DRAWS):
    regs={};regboots={};decs={};decboots={}
    for fn in FITS:
        regs[fn]={};regboots[fn]={}
        for field in ROLES+['P_pool','R_pool','T_A','T_W','delta_T']:
            if field.endswith('_pool'):
                n=field[0];ys=[r['true_candidates'][n] for r in rr];ps=[r['predictions'][fn][n] for r in rr]
            elif field in ROLES:ys=[[r['truth'][field]] for r in rr];ps=[[r['readouts'][fn][field]] for r in rr]
            else:
                key={'T_A':'anchor_t','T_W':'winner_t','delta_T':'delta_t'}[field]
                ys=[[r[key]] for r in rr];ps=[[r['analytic'][fn][field]] for r in rr]
            regs[fn][field],regboots[fn][field]=regression(rr,ys,ps,draws)
            moments=np.mean(list(regs[fn][field]['source_moments'].values()),axis=0)
            regs[fn][field]['truth_variance']=float(max(0.,moments[2]-moments[0]**2))
        decs[fn],decboots[fn]=decision(rr,[r['analytic'][fn]['delta_T'] for r in rr],draws)
    preg={};pdec={}
    for field in regs['all']:
        preg[field]={}
        for k in regs['all'][field]['metrics']:
            a=regs['all'][field]['metrics'][k]['mean'];b=regs['role'][field]['metrics'][k]['mean']
            preg[field][k]=pack(b-a if a is not None and b is not None else np.nan,regboots['role'][field][k]-regboots['all'][field][k])
    for k in decs['all']['metrics']:
        a=decs['all']['metrics'][k]['mean'];b=decs['role']['metrics'][k]['mean']
        pdec[k]=pack(b-a if a is not None and b is not None else np.nan,decboots['role'][k]-decboots['all'][k])
    orders={o:summarize([r for r in rr if r['order']==o],0) for o in sorted({r['order'] for r in rr})} if draws else {}
    return dict(cells=len(rr),sources=len({r['source_id'] for r in rr}),regression=regs,decision=decs,
        paired_role_minus_all_regression=preg,paired_role_minus_all_decision=pdec,
        paired_role_minus_all_utility=paired_utility(decs['all'],decs['role'],draws),orders=orders,draws=draws,seed=SEED)

def diagnose():
    verify();tick=time.time();fitseal=read(BASE/'FIT_SEAL.json');seal=read(OUT/'GLOBAL_READOUT_SEAL.json')
    assert fitseal['time']<seal['time'] and seal['prediction_sha256']==sha(OUT/'PREDICTIONS.json')
    old={r['cell_key']:r for r in read(OLD/'ROWS.json')};cohort={r['cell_key']:r for r in read(OUT/'COHORT.json')}
    rows=read(OUT/'PREDICTIONS.json');formulaerr=0.
    for ds in DATASETS:
        gt={sp:read(ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_{sp}.json') for sp in ['search','confirm']}
        for r in rows:
            if r['dataset']!=ds:continue
            z=load_bound(ROOT/cohort[r['cell_key']]['feature_file']);p,rec,t=truths(z,gt[r['split']][str(r['source_id'])]['span'])
            previous=old[r['cell_key']];a,w=r['anchor_index'],r['winner_index']
            for k in ['anchor_index','winner_index','eligible','A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256']:
                assert r[k]==previous[k]
            for n,v in [('P',p),('R',rec),('T',t)]:assert max(abs(v-np.array(previous['true_candidates'][n])))<3e-12
            r['true_candidates']={'P':p.tolist(),'R':rec.tolist(),'T':t.tolist()}
            r['truth']=role_values(r['true_candidates'],a,w)
            r.update({k:previous[k] for k in ['anchor_t','winner_t','anchor_v','winner_v','delta_t','delta_v']})
            r['readouts']={fn:role_values(r['predictions'][fn],a,w) for fn in FITS}
            r['analytic']={fn:analytic_readout(r['readouts'][fn]) for fn in FITS}
            check=analytic_readout(r['truth']);formulaerr=max(formulaerr,abs(check['delta_T']-r['delta_t']))
            assert abs(check['T_A']-r['anchor_t'])<3e-12 and abs(check['T_W']-r['winner_t'])<3e-12
            for fn in FITS:
                r['analytic'][fn]['accepted']=bool(r['eligible'] and r['analytic'][fn]['delta_T']>0)
                if a==w:assert r['analytic'][fn]['delta_T']==0
    write(OUT/'ROWS.json',rows)
    write(OUT/'LABEL_JOIN.json',dict(time=time.time(),fit_seal_time=fitseal['time'],readout_seal_time=seal['time'],
        search_GT_for_fit=True,confirmation_GT_for_fit=False,raw_annotations_read=False,
        all_GT_formula_max_error=formulaerr,historically_exposed=True,prediction_sha256=seal['prediction_sha256']))
    for ds in DATASETS:
        ss={}
        for sp in ['search','confirm']:
            for mode in ['all','corrupt','clean']:
                rr=[r for r in rows if r['dataset']==ds and r['split']==sp and (mode=='all' or (r['condition']=='clean')==(mode=='clean'))]
                ss[sp+'_'+mode]=summarize(rr)
                print('ROLE_PANEL',ds,sp,mode,len(rr),flush=True)
        write(OUT/ds/'SUMMARY.json',ss)
    cases={}
    for ds in DATASETS:
        rr=[r for r in rows if r['dataset']==ds and r['split']=='confirm' and r['condition']!='clean' and r['eligible']]
        picked={r['cell_key']:r for r in sorted(rr,key=lambda r:(r['delta_v'],r['cell_key']))[:3]}
        for r in sorted([r for r in rr if r['delta_t']>EPS],key=lambda r:(-r['delta_v'],r['cell_key']))[:3]:picked[r['cell_key']]=r
        changed=[r for r in rr if r['analytic']['all']['accepted']!=r['analytic']['role']['accepted']]
        for r in sorted(changed,key=lambda r:(r['delta_v'],r['cell_key']))[:3]+sorted(changed,key=lambda r:(-r['delta_v'],r['cell_key']))[:3]:picked[r['cell_key']]=r
        cases[ds]=list(picked.values())
    write(OUT/'CASES.json',cases)
    write(OUT/'DECISION.json',dict(status='supervised_population_diagnostic_no_promotion',A_preserved=True,CURRENT_preserved=True,
        fixed_alpha=True,threshold_search=False,new_gate=False,new_oracle_ladder=False,next_experiment_started=False,
        claim_scope='population-conditioned fit and generalization in fixed small-source linear family; no unique root cause or latent-information absence proof'))
    write(OUT/'RESOURCES.json',dict(new_scientific_CPU_ridge_fits=4,reused_M_all_models=4,
        fit_CPU_wall_seconds=fitseal['CPU_wall_seconds'],readout_CPU_wall_seconds=seal['CPU_wall_seconds'],diagnose_CPU_wall_seconds=time.time()-tick,
        new_GPU_calls=0,new_backbone_calls=0,new_expert_calls=0,new_candidate_calls=0,new_replay_calls=0,new_backward_calls=0,
        new_online_streams=0,production_updates=0,expert_cells=288,role_training_rows=384))
    status('completed_pending_root_audit_report_publication',cells=288,new_scientific_fits=4)

if __name__=='__main__':
    try:
        {'prepare':prepare,'fit':fit,'readout':readout,'diagnose':diagnose}[sys.argv[1]]()
        if 'torch' in sys.modules:assert not sys.modules['torch'].cuda.is_initialized()
    except Exception:
        if BASE.exists():
            p=BASE/'FAILURE.json'
            if not p.exists():write(p,dict(time=time.time(),action=sys.argv[1],traceback=traceback.format_exc()))
            status('failed',action=sys.argv[1])
        raise
