"""Independent nested source audit: SciPy SPD solves, no producer imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import solve
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import audit_tastvg_pr_accessibility_v1 as ref
read=ref.read;sha=ref.sha;close=ref.close;pack=ref.pack;checks=ref.checks
BASE=ROOT/'artifacts/tastvg_pr_nested_alpha_v1';PRE=ROOT/'artifacts/tastvg_pr_loso_v1'
OUT=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'results/tastvg_pr_nested_alpha/2026-10-03'
OLD=ROOT/'results/tastvg_pr_loso/2026-10-03'
GRID=[.001,.01,.1,1.,10.,100.,1000.];POPS=['all','role'];ROLES=['P_A','R_A','P_W','R_W']
VARIANTS=[p+'/'+r for p in POPS for r in ['fixed','nested']];SEED=20261003

def nested_checks():
    cfg=read(OUT/'CONFIG.json');cohort=read(OUT/'COHORT.json');ff=read(OUT/'FOLDS.json')
    cv=read(OUT/'INNER_CV.json');fits=read(OUT/'FIT_SUMMARY.json');dist=read(OUT/'ALPHA_DISTRIBUTION.json')
    assert len(cohort)==192 and len(ff)==30 and len(cv)==len(fits)==60
    assert cfg['alpha_grid']==GRID and cfg['new_scientific_CPU_fits']==11936 and cfg['eigendecompositions']==1808
    assert not cfg['outer_labels_used_for_alpha_selection'] and not cfg['confirmation_read_or_refitted']
    assert 'all six conditions' in cfg['selection'] and 'A/W MAE' in cfg['selection']
    cn=0;headmodels=0
    for ds in ['vidstg','hc2']:
        rr=[r for r in cohort if r['dataset']==ds];ids=sorted({r['source_id'] for r in rr})
        assert len(ids)==(16 if ds=='vidstg' else 14) and len(rr)==96 and sum(r['condition']!='clean' for r in rr)==80
        assert all(r['split']=='search' for r in rr)
        for held in ids:
            f=[f for f in ff if f['dataset']==ds and f['held_source']==held];assert len(f)==1;f=f[0]
            assert f['train_sources']==f['inner_sources']==[s for s in ids if s!=held]
            assert f['fold_key']==f'{ds}/{held:03d}'
            for pop in POPS:
                c=[c for c in cv if c['dataset']==ds and c['held_source']==held and c['population']==pop];assert len(c)==1;c=c[0]
                m=[m for m in fits if m['dataset']==ds and m['held_source']==held and m['population']==pop];assert len(m)==1;m=m[0]
                for z in [c,m]:
                    assert z['train_sources']==f['train_sources'] and held not in z['train_sources']
                assert not m['outer_labels_used']
                for n in ['P','R']:
                    assert [z['inner_source'] for z in c['heads'][n]]==f['inner_sources']
                    for z in c['heads'][n]:
                        inner=z['inner_source'];expected=[s for s in ids if s not in [held,inner]]
                        assert z['train_sources']==expected and held not in expected and inner not in expected
                        assert z['training_source_count']==len(expected)
                        assert z['training_rows']==sum(r['source_id'] in expected for r in rr)*(32 if pop=='all' else 2)
                        assert z['validation_cells']==sum(r['source_id']==inner for r in rr)
                        assert [e['alpha'] for e in z['scores']]==GRID
                        for e in z['scores']:
                            close(e['mean'],(e['A']+e['W'])/2,'inner_mean_roles')
                            close(e['mean'],(e['clean']+5*e['corrupt'])/6,'inner_mean_conditions')
                            assert min(e[k] for k in ['mean','A','W','clean','corrupt'])>=0
                        cn+=1;headmodels+=7
                    curve=[np.mean([z['scores'][j]['mean'] for z in c['heads'][n]]) for j in range(7)]
                    close(m['heads'][n]['inner_source_MAE'],curve,'alpha_curve_public')
                    assert m['heads'][n]['alpha']==GRID[int(np.argmin(curve))]
                    assert m['heads'][n]['selected_at']<m['outer_refit_completed_at']
                    assert m['heads'][n]['training_rows']==sum(r['source_id']!=held for r in rr)*(32 if pop=='all' else 2)
                    assert m['heads'][n]['training_sources']==len(ids)-1
                    checks['alpha_selection_excludes_outer_source']+=1
        for pop in POPS:
            for n in ['P','R']:
                expected={str(a):sum(m['heads'][n]['alpha']==a for m in fits if m['dataset']==ds and m['population']==pop) for a in GRID}
                assert dist[ds][pop][n]==expected;checks['alpha_frequency']+=1
    assert cn==1688 and headmodels==11816

def data_arrays(packs,sources,pop):
    xx=[];ys={'P':[],'R':[]};groups=[]
    for s in sorted(sources):
        p=packs[s]
        for i in range(len(p['x'])):
            ix=list(range(32)) if pop=='all' else [int(p['anchors'][i]),int(p['winners'][i])]
            xx.append(p['x'][i][ix]);groups.extend([s]*len(ix))
            for n in ys:ys[n].extend(p[n][i][ix])
    x=np.vstack(xx);g=np.array(groups);u,nn=np.unique(g,return_counts=True);count=dict(zip(u,nn))
    w=np.array([1/count[s]/len(u) for s in g]);close(sum(w),1.,'source_weight_total')
    for s in u:close(sum(w[g==s]),1/len(u),'source_weight_equal')
    return x,{n:np.array(y) for n,y in ys.items()},w

def reference_ridge(x,y,w,alpha):
    mean=np.average(x,axis=0,weights=w);std=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));std[std<1e-8]=1
    a=(x-mean)/std;bias=float(np.average(y,weights=w));cov=a.T@(w[:,None]*a);rhs=a.T@(w*(y-bias))
    coef=solve(cov+alpha*np.eye(x.shape[1]),rhs,assume_a='pos')
    return mean,std,coef,bias

def inner_reference(x,y,w,vx):
    mean=np.average(x,axis=0,weights=w);std=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));std[std<1e-8]=1
    a=(x-mean)/std;bias=float(np.average(y,weights=w));cov=a.T@(w[:,None]*a);rhs=a.T@(w*(y-bias))
    preds=[]
    for alpha in GRID:
        coef=solve(cov+alpha*np.eye(x.shape[1]),rhs,assume_a='pos')
        preds.append((vx-mean)/std@coef+bias);checks['independent_inner_SPD_refits']+=1
    return preds

def root_checks():
    import torch
    torch.set_num_threads(1);assert not torch.cuda.is_initialized()
    lock=read(BASE/'RUNTIME_LOCK.json');fs=read(BASE/'FIT_SEAL.json');seal=read(OUT/'GLOBAL_READOUT_SEAL.json');join=read(OUT/'LABEL_JOIN.json')
    assert lock['time']<fs['time']<seal['time']<join['time']
    for p,h in {**lock['code'],**lock['inputs'],**lock['payloads']}.items():assert sha(ROOT/p)==h,p;checks['immutable_SHA']+=1
    assert sha(OUT/'CONFIG.json')==lock['config_sha256'] and sha(BASE/'FIT_SEAL.json')==seal['fit_seal_sha256']
    assert sha(OUT/'INNER_CV.json')==fs['inner_cv_sha256'] and sha(OUT/'FIT_SUMMARY.json')==fs['selection_sha256']
    assert not fs['outer_source_label_packs_opened_in_fits'] and not fs['outer_labels_used_for_selection']
    assert not seal['outer_GT_read'] and not join['outer_labels_used_for_fit_or_selection']
    assert not fs['confirmation_read'] and not seal['confirmation_read'] and not join['confirmation_read']
    cohort=read(OUT/'COHORT.json');pred={r['cell_key']:r for r in read(OUT/'PREDICTIONS.json')};rows={r['cell_key']:r for r in read(OUT/'ROWS.json')}
    prior={r['cell_key']:r for r in read(OLD/'ROWS.json')};cv=read(OUT/'INNER_CV.json');fit=read(OUT/'FIT_SUMMARY.json')
    for ds in ['vidstg','hc2']:
        rr=[r for r in cohort if r['dataset']==ds];ids=sorted({r['source_id'] for r in rr});packs={}
        gt=read(ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_search.json')
        for s in ids:
            path=PRE/'source_packs'/ds/f'{s:03d}.npz';assert sha(path)==lock['payloads'][str(path.relative_to(ROOT))]
            with np.load(path,allow_pickle=False) as z:packs[s]={k:z[k].copy() for k in z.files}
            ss=[r for r in rr if r['source_id']==s];assert packs[s]['keys'].tolist()==[r['cell_key'] for r in ss]
            gs,ge=gt[str(s)]['span']
            for j,r in enumerate(ss):
                z=torch.load(ROOT/r['feature_file'],map_location='cpu',weights_only=False);assert not z['GT_read']
                close(packs[s]['x'][j],np.asarray(z['x'],float)[:,:768],'feature_pack',0)
                labs={'P':[],'R':[]};times=[];grid=z['frame_ids']
                for i,k in z['candidate_indices']:
                    a,b=grid[i],grid[k]+1;over=max(0.,min(b,ge)-max(a,gs))
                    labs['P'].append(over/(b-a));labs['R'].append(over/(ge-gs));times.append(over/(b-a+ge-gs-over))
                for n in labs:close(packs[s][n][j],labs[n],'physical_labels')
                ai,wi=r['anchor_index'],r['winner_index'];zrow=rows[r['cell_key']]
                close([zrow['truth'][k] for k in ROLES],[labs['P'][ai],labs['R'][ai],labs['P'][wi],labs['R'][wi]],'physical_role_truth')
                close([zrow['anchor_t'],zrow['winner_t']],[times[ai],times[wi]],'physical_tIoU')
                for key in ['anchor_index','winner_index','eligible','A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256']:
                    assert r[key]==prior[r['cell_key']][key]==zrow[key];checks['A_candidate_input_unchanged']+=1
                close([zrow[k] for k in ['anchor_v','winner_v']],[prior[r['cell_key']][k] for k in ['anchor_v','winner_v']],'cached_v',0)
                for pop in POPS:assert pred[r['cell_key']]['readouts'][pop+'/fixed']==prior[r['cell_key']]['readouts'][pop+'/nmax']
        for f in (f for f in read(OUT/'FOLDS.json') if f['dataset']==ds):
            outer=f['held_source']
            for pop in POPS:
                c=next(c for c in cv if c['dataset']==ds and c['held_source']==outer and c['population']==pop)
                m=next(m for m in fit if m['dataset']==ds and m['held_source']==outer and m['population']==pop)
                for inner in f['inner_sources']:
                    sources=[s for s in ids if s not in [outer,inner]];x,y,w=data_arrays(packs,sources,pop)
                    p=packs[inner];sr=[r for r in rr if r['source_id']==inner];ix=np.column_stack([p['anchors'],p['winners']])
                    vx=p['x'][np.arange(len(ix))[:,None],ix]
                    for n,part in [('P',slice(512,768)),('R',slice(0,512))]:
                        expected=next(z for z in c['heads'][n] if z['inner_source']==inner)
                        yy=p[n][np.arange(len(ix))[:,None],ix]
                        pp=inner_reference(x[:,part],y[n],w,vx[:,:,part].reshape(-1,x[:,part].shape[1]))
                        for j,q in enumerate(pp):
                            err=abs(q.reshape(len(ix),2)-yy);mom=ref.source_moments(sr,err);z=np.array(list(mom.values()))
                            e=expected['scores'][j]
                            close([e['A'],e['W'],e['mean']],[z[0,0],z[0,1],z.mean()],'independent_inner_MAE',2e-8)
                            for label in ['clean','corrupt']:
                                ind=[i for i,r in enumerate(sr) if (r['condition']=='clean')==(label=='clean')]
                                a=ref.source_moments([sr[i] for i in ind],err[ind]);close(e[label],np.mean(list(a.values())),'inner_clean_corrupt',2e-8)
                x,y,w=data_arrays(packs,f['train_sources'],pop);path=BASE/'models'/f'{ds}_{outer:03d}_{pop}.npz'
                assert sha(path)==fs['models'][str(path.relative_to(ROOT))]
                with np.load(path,allow_pickle=False) as z:stored={k:z[k].copy() for k in z.files}
                for n,part in [('P',slice(512,768)),('R',slice(0,512))]:
                    alpha=m['heads'][n]['alpha'];mean,std,coef,bias=reference_ridge(x[:,part],y[n],w,alpha)
                    for k,v in [('mean',mean),('std',std),('weight',coef),('bias',bias),('alpha',alpha)]:close(stored[n+'_'+k],v,'outer_model/'+k,2e-8)
                    pp=(x[:,part]-mean)/std@coef+bias
                    close(m['heads'][n]['training_MAE'],w@abs(pp-y[n]),'outer_training_MAE',2e-8)
                    close(m['heads'][n]['training_MSE'],w@((pp-y[n])**2),'outer_training_MSE',2e-8)
                    close(m['heads'][n]['coefficient_norm'],np.linalg.norm(coef),'outer_coef_norm',2e-8)
                    p=packs[outer]
                    for j,key in enumerate(p['keys']):
                        vx=p['x'][j,[int(p['anchors'][j]),int(p['winners'][j])],part];pp=(vx-mean)/std@coef+bias
                        close([pred[str(key)]['readouts'][pop+'/nested'][n+'_'+r] for r in ['A','W']],pp,'outer_OOF_pred',2e-8)
                    checks['independent_selected_outer_SPD_refits']+=1
            print('NESTED_ROOT_REFIT',f['fold_key'],flush=True)
    assert checks['independent_inner_SPD_refits']==11816 and checks['independent_selected_outer_SPD_refits']==120
    assert not torch.cuda.is_initialized()

def audit_summary(rr,got,draws):
    assert got['cells']==len(rr) and got['sources']==len({r['source_id'] for r in rr});rb={};db={}
    for v in VARIANTS:
        rb[v]={}
        for k in ROLES:
            rb[v][k]=ref.audit_regression(rr,[[r['truth'][k]] for r in rr],[[r['readouts'][v][k]] for r in rr],got['regression'][v][k],draws)
            m=np.mean(list(got['regression'][v][k]['source_moments'].values()),0);close(got['regression'][v][k]['truth_variance'],max(0,m[2]-m[0]**2),'truth_variance')
        db[v]=ref.decision_reference(rr,[r['analytic'][v]['delta_T'] for r in rr],got['decision'][v],draws)
    pairs=[(p+'/fixed',p+'/nested') for p in POPS]+[('all/nested','role/nested')]
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
        for j,k in enumerate(['delta_t','delta_v','gross_v_gain','gross_v_loss','accept_rate']):ref.verify_pack(diff['utility'][k],pack(m[:,j].mean(),bs[:,j]),'paired_utility/'+k)
    if draws:
        for o in sorted({r['order'] for r in rr}):audit_summary([r for r in rr if r['order']==o],got['orders'][o],0)
    else:assert got['orders']=={}

def main():
    mode=sys.argv[1];assert mode in ['root','public'];tick=time.time();nested_checks()
    if mode=='root':root_checks()
    rows=read(OUT/'ROWS.json');pred={r['cell_key']:r for r in read(OUT/'PREDICTIONS.json')};seal=read(OUT/'GLOBAL_READOUT_SEAL.json')
    assert len(rows)==len(pred)==192 and sha(OUT/'PREDICTIONS.json')==seal['prediction_sha256']
    for r in rows:
        assert r['readouts']==pred[r['cell_key']]['readouts'] and set(r['readouts'])==set(VARIANTS)
        close(r['delta_t'],r['winner_t']-r['anchor_t'],'delta_t');close(r['delta_v'],r['winner_v']-r['anchor_v'],'delta_v')
        for v in VARIANTS:
            z=r['readouts'][v];a=ref.reference_formula(z['P_A'],z['R_A']);w=ref.reference_formula(z['P_W'],z['R_W'])
            close([r['analytic'][v][k] for k in ['T_A','T_W','delta_T']],[a,w,w-a],'analytic_F')
            assert r['analytic'][v]['accepted']==bool(r['eligible'] and w-a>0);checks['fixed_decision']+=1
            if r['anchor_index']==r['winner_index']:assert w==a and not r['analytic'][v]['accepted']
    for ds in ['vidstg','hc2']:
        ss=read(OUT/ds/'SUMMARY.json');rr=[r for r in rows if r['dataset']==ds]
        for panel in ['all','corrupt','clean']:
            pr=[r for r in rr if panel=='all' or (r['condition']=='clean')==(panel=='clean')]
            audit_summary(pr,ss[panel],10000);print('NESTED_AUDIT',mode,ds,panel,flush=True)
        for c,z in ss['conditions'].items():audit_summary([r for r in rr if r['condition']==c],z,0)
    receipt=dict(status='pass',mode=mode,time=time.time(),CPU_wall_seconds=time.time()-tick,checks=dict(checks),
        total_scalar_checks=sum(checks.values()),max_error=dict(ref.error),GPU_calls=0,private_data_read=mode=='root',
        independent_solver='SciPy SPD,11816 inner plus120 outer independent refits' if mode=='root' else None,
        scope='all nested exclusions/alpha curves/selection/normalization/physical truth/OOF/fixed control/metrics/bootstrap' if mode=='root' else 'public nested folds/inner source curves/selection/raw OOF/metrics/paired intervals')
    path=OUT/('ROOT_AUDIT.json' if mode=='root' else 'PUBLIC_AUDIT.json')
    if path.exists():
        assert mode=='public','root receipt immutable'
        existing=read(path)
        for k in ['status','mode','checks','total_scalar_checks','max_error','GPU_calls','private_data_read','scope']:assert existing[k]==receipt[k],k
    else:path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['checks','max_error']}))

if __name__=='__main__':main()
