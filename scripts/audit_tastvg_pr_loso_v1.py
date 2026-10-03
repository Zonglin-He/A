"""Independent SciPy fit/OOF/source-metric verification, no producer imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import solve
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import audit_tastvg_pr_accessibility_v1 as ref
read=ref.read;sha=ref.sha;close=ref.close;pack=ref.pack;checks=ref.checks
BASE=ROOT/'artifacts/tastvg_pr_loso_v1'
OUT=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'results/tastvg_pr_loso/2026-10-03'
OLD=ROOT/'results/tastvg_pr_role/2026-10-03'
ROLES=['P_A','R_A','P_W','R_W'];POPS=['all','role'];LEVELS=['n4','n8','n12','nmax']
VARIANTS=[p+'/'+r for p in POPS for r in ['in_sample']+LEVELS];SEED=20261003

def fold_checks():
    cfg=read(OUT/'CONFIG.json');cohort=read(OUT/'COHORT.json');ff=read(OUT/'FOLDS.json')
    assert len(cohort)==192 and len(ff)==120 and cfg['new_scientific_CPU_fits']==480
    assert cfg['alpha']=={'vidstg':{'P':1.,'R':1.},'hc2':{'P':1.,'R':.1}}
    for ds in ['vidstg','hc2']:
        rr=[r for r in cohort if r['dataset']==ds];ids=sorted({r['source_id'] for r in rr})
        assert len(rr)==96 and len(ids)==(16 if ds=='vidstg' else 14) and all(r['split']=='search' for r in rr)
        assert sum(r['condition']!='clean' for r in rr)==80
        for held in ids:
            order=sorted((s for s in ids if s!=held),key=lambda s:(hashlib.sha256(
                f'tastvg_pr_loso_v1|{ds}|{held}|{s}'.encode()).hexdigest(),s))
            for level,n in zip(LEVELS,[4,8,12,len(ids)-1]):
                matches=[f for f in ff if f['dataset']==ds and f['held_source']==held and f['level']==level]
                assert len(matches)==1;f=matches[0]
                assert f['train_sources']==order[:n] and held not in f['train_sources'] and f['train_source_count']==n
                assert f['fold_key']==f'{ds}/{held:03d}/{level}';checks['fold_source_holdout_nesting']+=1
    assert cfg['bootstrap_refits'] is False and cfg['confirmation_read_or_refitted'] is False

def audit_summary(rr,got,draws):
    assert got['cells']==len(rr) and got['sources']==len({r['source_id'] for r in rr})
    rb={};db={}
    for v in VARIANTS:
        rb[v]={}
        for k in ROLES:
            rb[v][k]=ref.audit_regression(rr,[[r['truth'][k]] for r in rr],[[r['readouts'][v][k]] for r in rr],got['regression'][v][k],draws)
            m=np.mean(list(got['regression'][v][k]['source_moments'].values()),0)
            close(got['regression'][v][k]['truth_variance'],max(0.,m[2]-m[0]**2),'truth_variance')
        db[v]=ref.decision_reference(rr,[r['analytic'][v]['delta_T'] for r in rr],got['decision'][v],draws)
    pairs=[(p+'/in_sample',p+'/nmax') for p in POPS]+[('all/nmax','role/nmax')]
    pairs += [(p+'/'+a,p+'/'+b) for p in POPS for a,b in zip(LEVELS,LEVELS[1:])]
    assert set(got['paired'])=={b+' minus '+a for a,b in pairs}
    for a,b in pairs:
        diff=got['paired'][b+' minus '+a]
        for k in ROLES:
            for m,z in diff['regression'][k].items():
                av=got['regression'][a][k]['metrics'][m]['mean'];bv=got['regression'][b][k]['metrics'][m]['mean']
                ref.verify_pack(z,pack(bv-av if av is not None and bv is not None else np.nan,rb[b][k][m]-rb[a][k][m]),'paired_reg/'+m)
        for m,z in diff['decision'].items():
            av=got['decision'][a]['metrics'][m]['mean'];bv=got['decision'][b]['metrics'][m]['mean']
            ref.verify_pack(z,pack(bv-av if av is not None and bv is not None else np.nan,db[b][m]-db[a][m]),'paired_decision/'+m)
        values=[]
        for r in rr:
            aa=bool(r['eligible'] and r['analytic'][a]['delta_T']>0);bb=bool(r['eligible'] and r['analytic'][b]['delta_T']>0);sign=int(bb)-int(aa)
            values.append([sign*r['delta_t'],sign*r['delta_v'],sign*max(r['delta_v'],0),sign*min(r['delta_v'],0),sign])
        src=ref.source_moments(rr,values);m=np.array(list(src.values()));n=len(m)
        weights=np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=draws)/n;bs=weights@m
        for j,k in enumerate(['delta_t','delta_v','gross_v_gain','gross_v_loss','accept_rate']):
            ref.verify_pack(diff['utility'][k],pack(m[:,j].mean(),bs[:,j]),'paired_utility/'+k)
    if draws:
        for o in sorted({r['order'] for r in rr}):audit_summary([r for r in rr if r['order']==o],got['orders'][o],0)
    else:assert got['orders']=={}

def root_checks():
    import torch
    torch.set_num_threads(1);assert not torch.cuda.is_initialized()
    cfg=read(OUT/'CONFIG.json');lock=read(BASE/'RUNTIME_LOCK.json');ps=read(BASE/'PACK_SEAL.json');fs=read(BASE/'FIT_SEAL.json')
    seal=read(OUT/'GLOBAL_READOUT_SEAL.json');join=read(OUT/'LABEL_JOIN.json')
    assert lock['time']<ps['time']<fs['time']<seal['time']<join['time']
    assert not fs['held_source_labels_opened_during_fold_fit'] and not fs['confirmation_features_or_labels_read']
    assert not seal['test_GT_read'] and not seal['confirmation_read'] and not join['confirmation_read']
    assert sha(BASE/'FIT_SEAL.json')==seal['fit_seal_sha256'] and sha(BASE/'PACK_SEAL.json')==fs['pack_seal_sha256']
    for p,h in {**lock['code'],**lock['inputs'],**lock['payloads']}.items():assert sha(ROOT/p)==h,p;checks['immutable_SHA']+=1
    assert sha(OUT/'CONFIG.json')==lock['config_sha256']
    cohort=read(OUT/'COHORT.json');pred={r['cell_key']:r for r in read(OUT/'PREDICTIONS.json')}
    rows={r['cell_key']:r for r in read(OUT/'ROWS.json')};prior={r['cell_key']:r for r in read(OLD/'ROWS.json') if r['split']=='search'}
    metadata={(f['fold_key'],f['population']):f for f in read(OUT/'FIT_SUMMARY.json')}
    for ds in ['vidstg','hc2']:
        rr=[r for r in cohort if r['dataset']==ds];packs={};cache={};index={}
        gt=read(ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_search.json')
        for source in sorted({r['source_id'] for r in rr}):
            path=BASE/'source_packs'/ds/f'{source:03d}.npz';assert sha(path)==ps['packs'][str(path.relative_to(ROOT))]
            with np.load(path,allow_pickle=False) as z:packs[source]={k:z[k].copy() for k in z.files}
            ss=[r for r in rr if r['source_id']==source];assert packs[source]['keys'].tolist()==[r['cell_key'] for r in ss]
            s,e=gt[str(source)]['span']
            for j,r in enumerate(ss):
                path=ROOT/r['feature_file'];assert sha(path)==read(path.with_suffix('.json'))['sha256']
                z=torch.load(path,map_location='cpu',weights_only=False);cache[r['cell_key']]=z;index[r['cell_key']]=j
                assert z['split']=='search' and z['source_id']==source and not z['GT_read']
                assert z['source_A_temporal_bitwise_parity'] and z['A_spatial_bitwise_parity']
                close(packs[source]['x'][j],np.array(z['x'],float)[:,:768],'private_pack_feature',0)
                labs={'P':[],'R':[],'T':[]};ids=z['frame_ids']
                for i,k in z['candidate_indices']:
                    a,b=ids[i],ids[k]+1;v=max(0.,min(e,b)-max(s,a))
                    labs['P'].append(v/(b-a));labs['R'].append(v/(e-s));labs['T'].append(v/(b-a+e-s-v))
                for n in ['P','R']:close(packs[source][n][j],labs[n],'physical_label_pack',3e-12)
                old=prior[r['cell_key']]
                for k in ['anchor_index','winner_index','eligible','A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256']:assert r[k]==old[k]==rows[r['cell_key']][k]
                ai,wi=r['anchor_index'],r['winner_index']
                close([rows[r['cell_key']]['truth'][k] for k in ROLES],[labs['P'][ai],labs['R'][ai],labs['P'][wi],labs['R'][wi]],'physical_role_truth',3e-12)
                close([rows[r['cell_key']]['anchor_t'],rows[r['cell_key']]['winner_t']],[labs['T'][ai],labs['T'][wi]],'cached_t',3e-12)
                close([rows[r['cell_key']]['anchor_v'],rows[r['cell_key']]['winner_v']],[old['anchor_v'],old['winner_v']],'cached_v',0)
                for pop in POPS:
                    p=old['readouts'][pop];got=pred[r['cell_key']]['readouts'][pop+'/in_sample']
                    close([got[k] for k in ROLES],[p[k] for k in ROLES],'in_sample_exact_parity',3e-12)
        for f in (f for f in read(OUT/'FOLDS.json') if f['dataset']==ds):
            for pop in POPS:
                xx=[];yy={'P':[],'R':[]};g=[]
                for source in sorted(f['train_sources']):
                    z=packs[source]
                    for j in range(len(z['x'])):
                        ix=list(range(32)) if pop=='all' else [int(z['anchors'][j]),int(z['winners'][j])]
                        xx.append(z['x'][j][ix])
                        for n in yy:yy[n].append(z[n][j][ix])
                        g.extend([source]*len(ix))
                x=np.vstack(xx);groups=np.array(g);u,nn=np.unique(groups,return_counts=True);counts=dict(zip(u,nn))
                assert set(u)==set(f['train_sources']) and f['held_source'] not in u
                weights=np.array([1/counts[s]/len(u) for s in g]);close(weights.sum(),1.,'equal_source_sum')
                for s in u:close(weights[groups==s].sum(),1/len(u),'equal_source_group')
                path=BASE/'models'/f"{ds}_{f['held_source']:03d}_{f['level']}_{pop}.npz"
                assert sha(path)==fs['models'][str(path.relative_to(ROOT))]
                with np.load(path,allow_pickle=False) as z:stored={n:{k:z[n+'_'+k].copy() for k in ['mean','std','weight','bias','alpha']} for n in ['P','R']}
                meta=metadata[f['fold_key'],pop];assert meta['train_sources']==f['train_sources']
                for n,part in [('P',slice(512,768)),('R',slice(0,512))]:
                    a=x[:,part];y=np.concatenate(yy[n]);mean=np.average(a,axis=0,weights=weights)
                    std=np.sqrt(np.average((a-mean)**2,axis=0,weights=weights));std[std<1e-8]=1.;bias=float(np.average(y,weights=weights))
                    norm=(a-mean)/std;alpha=cfg['alpha'][ds][n]
                    coef=solve(norm.T@(weights[:,None]*norm)+alpha*np.eye(len(mean)),norm.T@(weights*(y-bias)),assume_a='pos')
                    m=stored[n]
                    for k,v in [('mean',mean),('std',std),('bias',bias),('weight',coef)]:close(m[k],v,'independent_ridge/'+k,1e-10)
                    close(m['alpha'],alpha,'fixed_alpha',0);checks['independent_verification_refits']+=1
                    fm=meta['heads'][n];pp=norm@coef+bias
                    close(fm['training_MAE'],weights@abs(y-pp),'fit_MAE');close(fm['training_MSE'],weights@((y-pp)**2),'fit_MSE')
                    assert fm['training_rows']==len(y) and fm['training_sources']==len(u) and not fm['held_labels_or_normalizer_used']
                    for r in (r for r in rr if r['source_id']==f['held_source']):
                        z=cache[r['cell_key']];a=np.array(z['x'],float)[[r['anchor_index'],r['winner_index']],part]
                        values=(a-mean)/std@coef+bias
                        close([pred[r['cell_key']]['readouts'][pop+'/'+f['level']][n+'_'+role] for role in ['A','W']],values,'OOF_prediction',3e-11)
            print('LOSO_ROOT_FIT',f['fold_key'],flush=True)
    assert checks['independent_verification_refits']==480 and not torch.cuda.is_initialized()

def main():
    mode=sys.argv[1];assert mode in ['root','public'];tick=time.time();fold_checks()
    if mode=='root':root_checks()
    rows=read(OUT/'ROWS.json');pred={r['cell_key']:r for r in read(OUT/'PREDICTIONS.json')};seal=read(OUT/'GLOBAL_READOUT_SEAL.json')
    assert len(rows)==192 and len(pred)==192 and sha(OUT/'PREDICTIONS.json')==seal['prediction_sha256']
    for r in rows:
        assert r['readouts']==pred[r['cell_key']]['readouts'] and set(r['readouts'])==set(VARIANTS)
        close(r['delta_t'],r['winner_t']-r['anchor_t'],'delta_t',3e-12);close(r['delta_v'],r['winner_v']-r['anchor_v'],'delta_v',3e-12)
        for v in VARIANTS:
            z=r['readouts'][v];a=ref.reference_formula(z['P_A'],z['R_A']);w=ref.reference_formula(z['P_W'],z['R_W'])
            close([r['analytic'][v][k] for k in ['T_A','T_W','delta_T']],[a,w,w-a],'analytic_F',3e-12)
            assert r['analytic'][v]['accepted']==bool(r['eligible'] and w-a>0);checks['fixed_decision']+=1
            if r['anchor_index']==r['winner_index']:assert w==a and not r['analytic'][v]['accepted']
    for ds in ['vidstg','hc2']:
        ss=read(OUT/ds/'SUMMARY.json');rr=[r for r in rows if r['dataset']==ds]
        for panel in ['all','corrupt','clean']:
            pr=[r for r in rr if panel=='all' or (r['condition']=='clean')==(panel=='clean')]
            audit_summary(pr,ss[panel],10000);print('LOSO_AUDIT',mode,ds,panel,flush=True)
        for c,z in ss['conditions'].items():audit_summary([r for r in rr if r['condition']==c],z,0)
    receipt=dict(status='pass',mode=mode,time=time.time(),CPU_wall_seconds=time.time()-tick,
        checks=dict(checks),total_scalar_checks=sum(checks.values()),max_error=dict(ref.error),
        new_GPU_calls=0,private_latents_models_labels_read=mode=='root',
        independent_solver='SciPy SPD solve vs producer NumPy eigen,480 verification refits' if mode=='root' else None,
        scope='source exclusions/nested folds/training normalizers/source weights/private fits/OOF predictions/physical truth/scalar metrics/bootstrap' if mode=='root' else 'anonymous held-source folds/raw roles/analytic decisions/source metrics/paired intervals')
    path=OUT/('ROOT_AUDIT.json' if mode=='root' else 'PUBLIC_AUDIT.json')
    if path.exists():
        assert mode=='public', 'root receipts are immutable'
        existing=read(path)
        for key in ['status','mode','checks','total_scalar_checks','max_error','new_GPU_calls','private_latents_models_labels_read','scope']:
            assert existing[key]==receipt[key],('existing public receipt differs',key)
        checks['existing_receipt_verified_without_overwrite']+=1
    else:path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['checks','max_error']}))

if __name__=='__main__':main()
