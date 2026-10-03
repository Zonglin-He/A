"""Finite source-held-out CPU diagnostic. Never imports inference/providers."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import sys,json,time,hashlib,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_pr_loso_math_v1 import *
BASE=ROOT/'artifacts/tastvg_pr_loso_v1'
OUT=ROOT/'results/tastvg_pr_loso/2026-10-03'
PRIOR=ROOT/'artifacts/tastvg_pr_role_v1'
OLD=ROOT/'results/tastvg_pr_role/2026-10-03'
ACCESS=ROOT/'artifacts/tastvg_pr_accessibility_v1'
DATASETS=['vidstg','hc2'];PARTS={'P':slice(512,768),'R':slice(0,512)}
OWN=['protocols/tastvg_pr_loso_v1.md','docs/tastvg_pr_loso_v1/EXECUTION.md',
    'scripts/run_tastvg_pr_loso_v1.py','scripts/tastvg_pr_loso_math_v1.py',
    'scripts/test_tastvg_pr_loso_v1.py','scripts/tastvg_pr_role_math_v1.py',
    'scripts/tastvg_pr_accessibility_math_v1.py']

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
def verify():
    z=read(BASE/'RUNTIME_LOCK.json')
    for p,h in {**z['code'],**z['inputs'],**z['payloads']}.items():assert sha(ROOT/p)==h,p
    assert sha(OUT/'CONFIG.json')==z['config_sha256'];return z
def torch_init():
    import torch
    torch.set_num_threads(1);assert not torch.cuda.is_initialized();return torch
def load_bound(p):
    assert sha(p)==read(p.with_suffix('.json'))['sha256']
    z=torch_init().load(p,map_location='cpu',weights_only=False)
    assert not torch_init().cuda.is_initialized();return z
def guard(stage,allowed=None):
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=str(args[0])
        if any(s in p for s in ['/annotations/','/videos/','/media/','/ROWS.json',
                'GT_LABELS_confirm','/target_features/confirm_']):raise PermissionError(stage+' access: '+p)
        if stage=='fit' and ('/GT_LABELS_' in p or '/target_features/' in p or '/IN_SAMPLE' in p):
            raise PermissionError(stage+' access: '+p)
        if '/source_packs/' in p and not source_pack_allowed(stage,str(Path(p).resolve()),
                args[1] if len(args)>1 else None,allowed['paths'] if allowed else set()):
            raise PermissionError(stage+' held/source-pack access: '+p)
        if stage=='readout' and '/GT_LABELS_' in p:raise PermissionError(stage+' label access: '+p)
    sys.addaudithook(hook)
def model_path(f,pop):return BASE/'models'/f"{f['dataset']}_{f['held_source']:03d}_{f['level']}_{pop}.npz"
def model_load(p):
    with np.load(p,allow_pickle=False) as z:
        return {n:{k:z[n+'_'+k].copy() for k in ['mean','std','weight','bias','alpha']} for n in PARTS}
def pack_path(ds,s):return BASE/'source_packs'/ds/f'{s:03d}.npz'

def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    done=read(PRIOR/'FINAL_COMPLETION.json')
    assert done['status']=='completed_and_publicly_verified' and done['commit']=='c1558266eaefee13b745202f332f7f52bffa1748'
    oldcfg=read(OLD/'CONFIG.json');cohort=[r for r in read(OLD/'COHORT.json') if r['split']=='search']
    assert len(cohort)==192
    ff=[];payloads={};inputs=[PRIOR/'FINAL_COMPLETION.json',OLD/'CONFIG.json',OLD/'COHORT.json',
        OLD/'PREDICTIONS.json',OLD/'GLOBAL_READOUT_SEAL.json',OLD/'ROOT_AUDIT.json',
        OLD/'ROWS.json',ROOT/'methods/CURRENT_METHOD.json']
    for ds in DATASETS:
        rr=[r for r in cohort if r['dataset']==ds];ids=sorted({r['source_id'] for r in rr})
        assert len(rr)==96 and len(ids)==(16 if ds=='vidstg' else 14)
        ff.extend(folds(ds,ids))
        inputs += [ACCESS/ds/'SEARCH_PR.npz',ACCESS/ds/'SEARCH_PR.json',PRIOR/ds/'ROLE_PR.npz',PRIOR/ds/'ROLE_PR.json',
            ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_search.json']
        for r in rr:
            p=ROOT/r['feature_file'];h=read(p.with_suffix('.json'))['sha256'];assert sha(p)==h
            payloads[r['feature_file']]=h;inputs.append(p.with_suffix('.json'))
    write(OUT/'COHORT.json',cohort);write(OUT/'FOLDS.json',ff)
    priorpred={r['cell_key']:r for r in read(OLD/'PREDICTIONS.json')}
    reused=[]
    from scripts.tastvg_pr_role_math_v1 import role_values
    for r in cohort:
        old=priorpred[r['cell_key']]
        reused.append(dict(cell_key=r['cell_key'],readouts={p:role_values(old['predictions'][p],r['anchor_index'],r['winner_index']) for p in POPS}))
    write(BASE/'IN_SAMPLE_READOUTS.json',reused)
    cfg={k:oldcfg[k] for k in ['alpha','views','latent','checkpoint_state_sha256','sampling','fixed_persistent_A','production_method_sha256']}
    cfg.update(version='tastvg_pr_loso_v1',predecessor_commit=done['commit'],search_expert_cells_per_dataset=96,
        actual_search_sources={'vidstg':16,'hc2':14},held_source_count=30,levels=LEVELS,max_train_sources={'vidstg':15,'hc2':13},
        populations=POPS,normalizer='same source-weighted training-only mean/std procedure, re-estimated each fold/population',
        weight_sum=1.,ridge_objective=oldcfg['ridge_objective'],new_scientific_CPU_fits=480,reused_in_sample_heads=8,
        source_subsets='nested SHA256(tastvg_pr_loso_v1|dataset|held_id|train_id), one prelocked list/fold',
        target_search_GT_supervised_diagnostic_only=True,confirmation_read_or_refitted=False,
        historically_exposed=True,held_source_unseen_by_readout_fit_only=True,
        primary_panel='search corruption source-LOSO maximum vs same-cell in-sample',
        raw_regression=True,analytic_t=oldcfg['analytic_t'],decision=oldcfg['decision'],
        bootstrap_draws=DRAWS,seed=SEED,epsilon=EPS,bootstrap_refits=False,
        bootstrap_scope='conditional fixed OOF predictions; folds overlap; one nested sequence; no multiplicity correction',
        GPU_backbone_expert_candidate_replay_backward_online_production_calls=0)
    write(OUT/'CONFIG.json',cfg)
    inputs += [OUT/'COHORT.json',OUT/'FOLDS.json',BASE/'IN_SAMPLE_READOUTS.json']
    lock=dict(time=time.time(),code={p:sha(ROOT/p) for p in OWN},inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        payloads=payloads,config_sha256=sha(OUT/'CONFIG.json'))
    write(BASE/'RUNTIME_LOCK.json',lock);write(OUT/'RUNTIME_BINDING.json',lock)
    status('prepared_pending_search_pack_extraction',folds=120,new_fits=480)

def extract():
    verify();torch_init();tick=time.time();allowed={'paths':set()};guard('extract',allowed);cohort=read(OUT/'COHORT.json');hashes={}
    for ds in DATASETS:
        rr=[r for r in cohort if r['dataset']==ds]
        gt=read(ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_search.json')
        for source in sorted({r['source_id'] for r in rr}):
            ss=[r for r in rr if r['source_id']==source];xx=[];pp=[];rec=[]
            for r in ss:
                z=load_bound(ROOT/r['feature_file'])
                assert z['split']=='search' and z['cell_key']==r['cell_key'] and z['source_id']==source and not z['GT_read']
                assert z['source_A_temporal_bitwise_parity'] and z['A_spatial_bitwise_parity']
                ids=z['frame_ids'];p,q,t=interval_pr([[ids[i],ids[j]+1] for i,j in z['candidate_indices']],gt[str(source)]['span'])
                xx.append(np.asarray(z['x'],float)[:,:768]);pp.append(p);rec.append(q)
            path=pack_path(ds,source);path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
            np.savez(path,x=np.array(xx),P=np.array(pp),R=np.array(rec),
                anchors=np.array([r['anchor_index'] for r in ss]),winners=np.array([r['winner_index'] for r in ss]),
                keys=np.array([r['cell_key'] for r in ss]),source=np.array(source))
            allowed['paths']={str(path.resolve())}
            hashes[str(path.relative_to(ROOT))]=sha(path)
            allowed['paths']=set()
        print('LOSO_EXTRACT',ds,len(rr),'cells',flush=True)
    write(BASE/'PACK_SEAL.json',dict(time=time.time(),packs=hashes,search_only=True,CPU_wall_seconds=time.time()-tick))
    status('search_packs_sealed_pending_held_source_fits',sources=30)

def fit():
    verify();torch_init();tick=time.time();ps=read(BASE/'PACK_SEAL.json');cfg=read(OUT/'CONFIG.json')
    allowed={'paths':set()};guard('fit',allowed);hashes={};summary=[]
    for f in read(OUT/'FOLDS.json'):
        ds=f['dataset'];assert f['held_source'] not in f['train_sources']
        allowed['paths']={str(pack_path(ds,s).resolve()) for s in f['train_sources']}
        packs={}
        for s in f['train_sources']:
            p=pack_path(ds,s);assert sha(p)==ps['packs'][str(p.relative_to(ROOT))]
            with np.load(p,allow_pickle=False) as z:packs[s]={k:z[k].copy() for k in z.files}
            assert int(packs[s]['source'])==s
        for pop in POPS:
            x,ys,groups=subset_arrays(packs,pop);assert set(groups)==set(f['train_sources'])
            models={};dd={};w=source_weights(groups)
            for name,part in PARTS.items():
                y=ys[name];m=fit_ridge(x[:,part],y,groups,cfg['alpha'][ds][name]);models[name]=m
                a=(x[:,part]-m['mean'])/m['std'];pred=predict(m,x[:,part])
                residual=a.T@(w*(pred-y))+m['alpha']*m['weight']
                assert max(abs(residual))<1e-9 and abs(w@(pred-y))<1e-10
                dd[name]=dict(dimensions=part.stop-part.start,alpha=m['alpha'],training_rows=len(y),
                    training_sources=len(set(groups)),training_cells=sum(len(z['x']) for z in packs.values()),
                    training_MAE=float(w@abs(y-pred)),training_MSE=float(w@((y-pred)**2)),
                    normal_equation_error=float(max(abs(residual))),weight_sum=float(w.sum()),held_labels_or_normalizer_used=False)
            path=model_path(f,pop);path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
            np.savez(path,**{n+'_'+k:np.asarray(v) for n,m in models.items() for k,v in m.items() if k not in ['model','intercept']})
            hashes[str(path.relative_to(ROOT))]=sha(path);summary.append(dict(**f,population=pop,heads=dd,model_sha256=sha(path)))
        del packs
        print('LOSO_FIT',f['fold_key'],len(hashes)*2,'/480',flush=True)
    assert len(hashes)==240
    write(OUT/'FIT_SUMMARY.json',summary)
    write(BASE/'FIT_SEAL.json',dict(time=time.time(),models=hashes,new_scientific_CPU_fits=480,
        pack_seal_sha256=sha(BASE/'PACK_SEAL.json'),held_source_labels_opened_during_fold_fit=False,
        confirmation_features_or_labels_read=False,CPU_wall_seconds=time.time()-tick))
    status('480_models_sealed_pending_GT_free_OOF_readout',fits=480)

def readout():
    verify();torch_init();tick=time.time();fitseal=read(BASE/'FIT_SEAL.json');guard('readout')
    cohort=read(OUT/'COHORT.json');ff=read(OUT/'FOLDS.json');prior={r['cell_key']:r for r in read(BASE/'IN_SAMPLE_READOUTS.json')}
    rows=[];paritymax=0.;checked=0
    for ds in DATASETS:
        rr=[r for r in cohort if r['dataset']==ds];cache={r['cell_key']:load_bound(ROOT/r['feature_file']) for r in rr}
        reused={'all':model_load(ACCESS/ds/'SEARCH_PR.npz'),'role':model_load(PRIOR/ds/'ROLE_PR.npz')}
        modelcache={}
        for f in (f for f in ff if f['dataset']==ds):
            for pop in POPS:
                path=model_path(f,pop);assert sha(path)==fitseal['models'][str(path.relative_to(ROOT))]
                modelcache[f['held_source'],f['level'],pop]=model_load(path)
        for r in rr:
            z=cache[r['cell_key']];assert z['cell_key']==r['cell_key'] and z['source_id']==r['source_id']
            x=np.array(z['x'],float)[[r['anchor_index'],r['winner_index']]];rd={}
            for pop in POPS:
                for regime in REGIMES:
                    models=reused[pop] if regime=='in_sample' else modelcache[r['source_id'],regime,pop]
                    vals={n:predict(models[n],x[:,part]) for n,part in PARTS.items()}
                    rd[pop+'/'+regime]={n+'_'+role:float(vals[n][i]) for n in PARTS for i,role in enumerate(['A','W'])}
                    if regime=='in_sample':
                        p=prior[r['cell_key']]['readouts'][pop]
                        err=max(abs(rd[pop+'/'+regime][k]-p[k]) for k in ROLES)
                        assert err<3e-12;paritymax=max(paritymax,err);checked+=4
                    if r['anchor_index']==r['winner_index']:
                        assert rd[pop+'/'+regime]['P_A']==rd[pop+'/'+regime]['P_W']
                        assert rd[pop+'/'+regime]['R_A']==rd[pop+'/'+regime]['R_W']
            rows.append({**{k:v for k,v in r.items() if k!='feature_file'},'readouts':rd})
        print('LOSO_READOUT',ds,96,'cells x10 regimes; in-sample parity',flush=True)
    write(OUT/'PREDICTIONS.json',rows)
    write(OUT/'GLOBAL_READOUT_SEAL.json',dict(time=time.time(),prediction_sha256=sha(OUT/'PREDICTIONS.json'),
        fit_seal_sha256=sha(BASE/'FIT_SEAL.json'),test_GT_read=False,confirmation_read=False,
        in_sample_parity_max=paritymax,in_sample_parity_scalars=checked,CPU_wall_seconds=time.time()-tick))
    status('192_search_OOF_readouts_sealed_pending_label_join',cells=192)

def paired_utility(a,b,draws):
    ids=list(a['utility_source_values']);assert ids==list(b['utility_source_values'])
    x=np.array([a['utility_source_values'][s] for s in ids]);y=np.array([b['utility_source_values'][s] for s in ids]);d=y-x;n=len(ids)
    weights=np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=draws)/n;bs=weights@d
    return {k:pack(d[:,j].mean(),bs[:,j]) for j,k in enumerate(['delta_t','delta_v','gross_v_gain','gross_v_loss','accept_rate'])}

def summarize(rr,draws=DRAWS):
    regs={};rb={};decs={};db={}
    for variant in VARIANTS:
        regs[variant]={};rb[variant]={}
        for role in ROLES:
            regs[variant][role],rb[variant][role]=regression(rr,[[r['truth'][role]] for r in rr],[[r['readouts'][variant][role]] for r in rr],draws)
            m=np.mean(list(regs[variant][role]['source_moments'].values()),0)
            regs[variant][role]['truth_variance']=float(max(0.,m[2]-m[0]**2))
        decs[variant],db[variant]=decision(rr,[r['analytic'][variant]['delta_T'] for r in rr],draws)
    pairs=[(p+'/in_sample',p+'/nmax') for p in POPS]+[('all/nmax','role/nmax')]
    pairs += [(p+'/'+a,p+'/'+b) for p in POPS for a,b in zip(LEVELS,LEVELS[1:])]
    comparison={}
    for a,b in pairs:
        comparison[b+' minus '+a]=dict(
            regression={r:paired(regs[a][r],regs[b][r],rb[a][r],rb[b][r]) for r in ROLES},
            decision=paired(decs[a],decs[b],db[a],db[b]),utility=paired_utility(decs[a],decs[b],draws))
    orders={o:summarize([r for r in rr if r['order']==o],0) for o in sorted({r['order'] for r in rr})} if draws else {}
    return dict(cells=len(rr),sources=len({r['source_id'] for r in rr}),regression=regs,decision=decs,
        paired=comparison,orders=orders,draws=draws,seed=SEED)

def diagnose():
    verify();tick=time.time();fitseal=read(BASE/'FIT_SEAL.json');seal=read(OUT/'GLOBAL_READOUT_SEAL.json')
    assert fitseal['time']<seal['time'] and seal['prediction_sha256']==sha(OUT/'PREDICTIONS.json')
    rows=read(OUT/'PREDICTIONS.json');old={r['cell_key']:r for r in read(OLD/'ROWS.json') if r['split']=='search'}
    for r in rows:
        previous=old[r['cell_key']]
        for k in ['anchor_index','winner_index','eligible','A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256']:assert r[k]==previous[k]
        r.update({k:previous[k] for k in ['truth','anchor_t','winner_t','anchor_v','winner_v','delta_t','delta_v']})
        r['analytic']={v:analytic_readout(r['readouts'][v]) for v in VARIANTS}
        for v in VARIANTS:r['analytic'][v]['accepted']=bool(r['eligible'] and r['analytic'][v]['delta_T']>0)
    write(OUT/'ROWS.json',rows)
    write(OUT/'LABEL_JOIN.json',dict(time=time.time(),fit_seal_time=fitseal['time'],readout_seal_time=seal['time'],
        held_labels_used_for_fitting=False,confirmation_read=False,cached_search_labels_joined=True,
        search_label_packs_preextracted=True,raw_annotations_read=False,historically_exposed=True))
    cases={}
    for ds in DATASETS:
        rr=[r for r in rows if r['dataset']==ds];ss={}
        for mode in ['all','corrupt','clean']:
            panel=[r for r in rr if mode=='all' or (r['condition']=='clean')==(mode=='clean')]
            ss[mode]=summarize(panel);print('LOSO_PANEL',ds,mode,len(panel),flush=True)
        ss['conditions']={c:summarize([r for r in rr if r['condition']==c],0) for c in sorted({r['condition'] for r in rr})}
        write(OUT/ds/'SUMMARY.json',ss)
        cr=[r for r in rr if r['condition']!='clean'];selected={}
        for pop in POPS:
            def err(r):return np.mean([abs(r['readouts'][pop+'/nmax'][k]-r['truth'][k]) for k in ROLES])
            for r in sorted(cr,key=lambda r:(err(r),r['cell_key']))[:3]+sorted(cr,key=lambda r:(-err(r),r['cell_key']))[:3]:selected[r['cell_key']]=r
            changed=[r for r in cr if r['analytic'][pop+'/nmax']['accepted']!=r['analytic'][pop+'/in_sample']['accepted']]
            for r in sorted(changed,key=lambda r:(r['delta_v'],r['cell_key']))[:2]+sorted(changed,key=lambda r:(-r['delta_v'],r['cell_key']))[:2]:selected[r['cell_key']]=r
        cases[ds]=list(selected.values())
    write(OUT/'CASES.json',cases)
    ps=read(BASE/'PACK_SEAL.json')
    write(OUT/'RESOURCES.json',dict(new_scientific_CPU_fits=480,reused_in_sample_heads=8,
        extraction_CPU_wall_seconds=ps['CPU_wall_seconds'],fit_CPU_wall_seconds=fitseal['CPU_wall_seconds'],
        readout_CPU_wall_seconds=seal['CPU_wall_seconds'],diagnosis_CPU_wall_seconds=time.time()-tick,
        GPU_calls=0,backbone_expert_candidate_replay_backward_online_production_calls=0,
        folds=120,private_model_files=240,search_expert_cells=192,independent_test_sources=30))
    write(OUT/'DECISION.json',dict(status='source-held-out-supervised-diagnostic-no-promotion',A_preserved=True,CURRENT_preserved=True,
        threshold_or_alpha_search=False,next_experiment_started=False,
        scope='fixed linear family, original alpha, 14/16 search sources; not unique overfit or representation-causality proof'))
    status('completed_pending_independent_audit_review_publication',cells=192,new_fits=480)

if __name__=='__main__':
    action=sys.argv[1];assert action in ['prepare','extract','fit','readout','diagnose']
    try:globals()[action]()
    except Exception:
        status('failed',stage=action);raise
