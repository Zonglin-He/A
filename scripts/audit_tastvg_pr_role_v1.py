"""Independent CPU solve, interval arithmetic and scalar-statistic verification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import solve
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
# Established independent scalar auditor; never import this experiment's producer.
from scripts import audit_tastvg_pr_accessibility_v1 as ref
read=ref.read;sha=ref.sha;close=ref.close;pack=ref.pack;checks=ref.checks
BASE=ROOT/'artifacts/tastvg_pr_role_v1'
OUT=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'results/tastvg_pr_role/2026-10-03'
PREV=ROOT/'artifacts/tastvg_pr_accessibility_v1'
OLD=ROOT/'results/tastvg_pr_accessibility/2026-10-03'
FITS=['all','role'];ROLES=['P_A','R_A','P_W','R_W'];SEED=20261003;EPS=1e-12

def audit_summary(rr,got,draws):
    assert got['cells']==len(rr) and got['sources']==len({r['source_id'] for r in rr})
    rb={};db={}
    for fn in FITS:
        rb[fn]={}
        for field in ROLES+['P_pool','R_pool','T_A','T_W','delta_T']:
            if field.endswith('_pool'):
                n=field[0];yy=[r['true_candidates'][n] for r in rr];pp=[r['predictions'][fn][n] for r in rr]
            elif field in ROLES:yy=[[r['truth'][field]] for r in rr];pp=[[r['readouts'][fn][field]] for r in rr]
            else:
                key={'T_A':'anchor_t','T_W':'winner_t','delta_T':'delta_t'}[field]
                yy=[[r[key]] for r in rr];pp=[[r['analytic'][fn][field]] for r in rr]
            rb[fn][field]=ref.audit_regression(rr,yy,pp,got['regression'][fn][field],draws)
            m=np.mean(list(got['regression'][fn][field]['source_moments'].values()),axis=0)
            close(got['regression'][fn][field]['truth_variance'],max(0.,m[2]-m[0]**2),'truth_variance')
        db[fn]=ref.decision_reference(rr,[r['analytic'][fn]['delta_T'] for r in rr],got['decision'][fn],draws)
    for field,metrics in got['paired_role_minus_all_regression'].items():
        for k,z in metrics.items():
            a=got['regression']['all'][field]['metrics'][k]['mean'];b=got['regression']['role'][field]['metrics'][k]['mean']
            ref.verify_pack(z,pack(b-a if a is not None and b is not None else np.nan,rb['role'][field][k]-rb['all'][field][k]),'paired/regression/'+k)
    for k,z in got['paired_role_minus_all_decision'].items():
        a=got['decision']['all']['metrics'][k]['mean'];b=got['decision']['role']['metrics'][k]['mean']
        ref.verify_pack(z,pack(b-a if a is not None and b is not None else np.nan,db['role'][k]-db['all'][k]),'paired/decision/'+k)
    values=[]
    for r in rr:
        aa=bool(r['eligible'] and r['analytic']['all']['delta_T']>0);bb=bool(r['eligible'] and r['analytic']['role']['delta_T']>0)
        sign=int(bb)-int(aa)
        values.append([sign*r['delta_t'],sign*r['delta_v'],sign*max(r['delta_v'],0),sign*min(r['delta_v'],0),sign])
    source=ref.source_moments(rr,values);m=np.array(list(source.values()));n=len(m)
    bs=np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=draws)/n
    samples=bs@m
    for j,k in enumerate(['delta_t','delta_v','gross_v_gain','gross_v_loss','accept_rate']):
        ref.verify_pack(got['paired_role_minus_all_utility'][k],pack(m[:,j].mean(),samples[:,j]),'paired/utility/'+k)
    if draws:
        for o in sorted({r['order'] for r in rr}):audit_summary([r for r in rr if r['order']==o],got['orders'][o],0)
    else:assert got['orders']=={}

def root_checks():
    import torch
    torch.set_num_threads(1);assert not torch.cuda.is_initialized()
    def load(p):
        assert sha(p)==read(p.with_suffix('.json'))['sha256'];checks['payload_SHA']+=1
        return torch.load(p,map_location='cpu',weights_only=False)
    lock=read(BASE/'RUNTIME_LOCK.json');cfg=read(OUT/'CONFIG.json');fit=read(BASE/'FIT_SEAL.json')
    seal=read(OUT/'GLOBAL_READOUT_SEAL.json');join=read(OUT/'LABEL_JOIN.json')
    for p,h in {**lock['code'],**lock['inputs'],**lock['payloads']}.items():assert sha(ROOT/p)==h,p;checks['immutable_SHA']+=1
    assert sha(OUT/'CONFIG.json')==lock['config_sha256']
    assert lock['time']<fit['time']<seal['time']<join['time']
    assert not fit['confirmation_features_read'] and not fit['confirmation_GT_read'] and fit['target_search_GT_supervised']
    assert not seal['confirmation_GT_read'] and not join['confirmation_GT_for_fit']
    assert seal['fit_seal_sha256']==sha(BASE/'FIT_SEAL.json')
    assert cfg['alpha']=={'vidstg':{'P':1.,'R':1.},'hc2':{'P':1.,'R':.1}}
    cohort=read(OUT/'COHORT.json');pop=read(OUT/'TRAINING_POPULATION.json')
    pred={r['cell_key']:r for r in read(OUT/'PREDICTIONS.json')};rows={r['cell_key']:r for r in read(OUT/'ROWS.json')}
    oldpred={r['cell_key']:r for r in read(OLD/'PREDICTIONS.json')};oldrows={r['cell_key']:r for r in read(OLD/'ROWS.json')}
    ps={r['cell_key']:r for r in pop};assert len(ps)==192
    for ds in ['vidstg','hc2']:
        search=[r for r in cohort if r['dataset']==ds and r['split']=='search'];confirm=[r for r in cohort if r['dataset']==ds and r['split']=='confirm']
        assert len(search)==96 and len(confirm)==48
        assert not {r['source_id'] for r in search}&{r['source_id'] for r in confirm}
        assert len({r['source_id'] for r in search})==cfg['actual_expert_sources'][ds]['search']
        assert len({r['source_id'] for r in confirm})==cfg['actual_expert_sources'][ds]['confirm']
        cache={r['cell_key']:load(ROOT/r['feature_file']) for r in search+confirm}
        gt={sp:read(ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_{sp}.json') for sp in ['search','confirm']}
        def labels(z):
            s,e=gt[z['split']][str(z['source_id'])]['span'];ids=z['frame_ids'];p=[];rec=[];tt=[]
            for i,j in z['candidate_indices']:
                a,b=ids[i],ids[j]+1;v=max(0.,min(e,b)-max(s,a))
                p.append(v/(b-a));rec.append(v/(e-s));tt.append(v/(b-a+e-s-v))
            return np.array(p),np.array(rec),np.array(tt)
        fitted={}
        for fn,path in [('all',PREV/ds/'SEARCH_PR.npz'),('role',BASE/ds/'ROLE_PR.npz')]:
            assert sha(path)==(fit['reused_all_models'] if fn=='all' else fit['models'])[ds]
            with np.load(path,allow_pickle=False) as z:
                fitted[fn]={n:{k:z[n+'_'+k].copy() for k in ['mean','std','weight','bias','alpha']} for n in ['P','R']}
            xx=[];ys=[[],[]];groups=[]
            for r in search:
                z=cache[r['cell_key']];ix=list(range(32)) if fn=='all' else [r['anchor_index'],r['winner_index']]
                if fn=='role':
                    row=ps[r['cell_key']];assert row['indices']==ix and row['roles']==['A','W'] and row['duplicate_roles']==(ix[0]==ix[1])
                    checks['training_population']+=2
                xx.append(np.array(z['x'],float)[ix]);lab=labels(z)
                for i in [0,1]:ys[i].append(lab[i][ix])
                groups.extend([r['source_id']]*len(ix))
            x=np.vstack(xx);assert x.shape==(3072 if fn=='all' else 192,1792)
            groups=np.array(groups);u,counts=np.unique(groups,return_counts=True);look=dict(zip(u,counts));w=np.array([1/look[g] for g in groups]);w/=w.sum()
            close(w.sum(),1.,'weight_sum')
            for g in u:close(w[groups==g].sum(),1/len(u),'source_total_weight')
            for name,i,part in [('P',0,slice(512,768)),('R',1,slice(0,512))]:
                a=x[:,part];y=np.concatenate(ys[i]);m=fitted[fn][name];mean=np.average(a,axis=0,weights=w)
                std=np.sqrt(np.average((a-mean)**2,axis=0,weights=w));std[std<1e-8]=1
                bias=float(np.average(y,weights=w));norm=(a-mean)/std;alpha=cfg['alpha'][ds][name]
                assert float(m['alpha'])==alpha
                coef=solve(norm.T@(w[:,None]*norm)+alpha*np.eye(norm.shape[1]),norm.T@(w*(y-bias)),assume_a='pos')
                for k,v in [('mean',mean),('std',std),('bias',bias),('weight',coef)]:close(m[k],v,'independent_ridge/'+k,1e-10)
                checks['independent_verification_refits']+=1
                if fn=='role':
                    got=read(OUT/'FIT_SUMMARY.json')[ds][name];pr=norm@coef+bias
                    close(got['weighted_training_MSE'],w@((pr-y)**2),'training_MSE')
                    close(got['weighted_training_MAE'],w@abs(pr-y),'training_MAE')
                    assert got['training_rows']==192 and got['training_sources']==len(u) and not got['confirmation_features_used'] and not got['confirmation_labels_used']
        for r in search+confirm:
            key=r['cell_key'];z=cache[key];rr=rows[key];pr=pred[key];old=oldrows[key];a,widx=r['anchor_index'],r['winner_index']
            assert z['source_A_temporal_bitwise_parity'] and z['A_spatial_bitwise_parity'] and not z['GT_read']
            for k in ['anchor_index','winner_index','eligible','A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256']:
                assert r[k]==rr[k]==pr[k]==old[k]
            assert z['anchor_index']==a and rr['predictions']==pr['predictions'];checks['frozen_choices_state']+=1
            p,rec,t=labels(z)
            for n,v in [('P',p),('R',rec),('T',t)]:close(rr['true_candidates'][n],v,'physical_labels',3e-12)
            close([rr['truth'][k] for k in ROLES],[p[a],rec[a],p[widx],rec[widx]],'role_labels',3e-12)
            close([rr['anchor_t'],rr['winner_t']],[t[a],t[widx]],'cached_t',3e-12)
            close([rr['anchor_v'],rr['winner_v']],[old['anchor_v'],old['winner_v']],'cached_v',3e-12)
            for fn in FITS:
                for name,part in [('P',slice(512,768)),('R',slice(0,512))]:
                    m=fitted[fn][name];values=(np.array(z['x'],float)[:,part]-m['mean'])/m['std']@m['weight']+m['bias']
                    close(pr['predictions'][fn][name],values,'predictions',3e-12)
                    if fn=='all':close(values,oldpred[key]['predictions']['target_search_fit'][name],'M_all_parity',3e-12)
    assert not torch.cuda.is_initialized()

def main():
    mode=sys.argv[1];assert mode in ['root','public'];tick=time.time()
    if mode=='root':root_checks()
    cfg=read(OUT/'CONFIG.json');rows=read(OUT/'ROWS.json');pred={r['cell_key']:r for r in read(OUT/'PREDICTIONS.json')}
    seal=read(OUT/'GLOBAL_READOUT_SEAL.json');assert sha(OUT/'PREDICTIONS.json')==seal['prediction_sha256']
    assert len(rows)==288 and len(pred)==288
    for r in rows:
        assert r['predictions']==pred[r['cell_key']]['predictions']
        a,w=r['anchor_index'],r['winner_index']
        for fn in FITS:
            pp=r['predictions'][fn];pr=dict(P_A=pp['P'][a],R_A=pp['R'][a],P_W=pp['P'][w],R_W=pp['R'][w])
            for k in ROLES:close(r['readouts'][fn][k],pr[k],'roles',3e-12)
            ta=ref.reference_formula(pr['P_A'],pr['R_A']);tw=ref.reference_formula(pr['P_W'],pr['R_W'])
            close([r['analytic'][fn][k] for k in ['T_A','T_W','delta_T']],[ta,tw,tw-ta],'analytic_map',3e-12)
            assert r['analytic'][fn]['accepted']==bool(r['eligible'] and tw-ta>0);checks['fixed_zero_decision']+=1
            if a==w:assert not r['analytic'][fn]['accepted'] and r['analytic'][fn]['delta_T']==0
        close(r['delta_t'],r['winner_t']-r['anchor_t'],'delta_t',3e-12)
        close(r['delta_v'],r['winner_v']-r['anchor_v'],'delta_v',3e-12)
        close(ref.reference_formula(r['truth']['P_A'],r['truth']['R_A']),r['anchor_t'],'true_t_identity',3e-12)
        close(ref.reference_formula(r['truth']['P_W'],r['truth']['R_W']),r['winner_t'],'true_t_identity',3e-12)
    for ds in ['vidstg','hc2']:
        summary=read(OUT/ds/'SUMMARY.json')
        for key,got in summary.items():
            sp,mode0=key.split('_',1)
            rr=[r for r in rows if r['dataset']==ds and r['split']==sp and (mode0=='all' or (r['condition']=='clean')==(mode0=='clean'))]
            audit_summary(rr,got,10000);print('ROLE_AUDIT',mode,ds,key,len(rr),flush=True)
    receipt=dict(status='pass',mode=mode,time=time.time(),CPU_wall_seconds=time.time()-tick,
        checks=dict(checks),total_scalar_checks=sum(checks.values()),max_error=dict(ref.error),
        private_features_weights_labels_read=mode=='root',new_GPU_calls=0,
        independent_solver='SciPy SPD solve vs producer eigen solver; 4 M-all and 4 M-role verification refits' if mode=='root' else None,
        scope='all population memberships/fits/immutable inputs/predictions/physical labels/states/scalar metrics/bootstrap' if mode=='root' else 'all anonymous role readouts/accept rules/raw regression/decision metrics/paired bootstrap')
    path=OUT/('ROOT_AUDIT.json' if mode=='root' else 'PUBLIC_AUDIT.json');assert not path.exists()
    path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['checks','max_error']},ensure_ascii=False))

if __name__=='__main__':main()
