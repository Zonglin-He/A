"""Independent solver and scalar-statistic audit; no producer/math imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import sys,json,time,hashlib,collections
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from sklearn.metrics import roc_auc_score,r2_score,mean_absolute_error
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/tastvg_pr_accessibility_v1'
OUT=ROOT/'results/tastvg_pr_accessibility/2026-10-03'
EPS=1e-12;SEED=20261003
FITS=['source_fit','target_search_fit']
ARMS=['predicted','GT_anchor','GT_winner','GT_precision','GT_recall','GT_all']
ROLES=['P_A','R_A','P_W','R_W']
checks=collections.Counter();error=collections.defaultdict(float)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,key,tol=1e-9):
    if a is None or b is None:
        assert a is None and b is None,(key,a,b);checks[key]+=1;return
    a=np.asarray(a,float);b=np.asarray(b,float);assert a.shape==b.shape,(key,a.shape,b.shape)
    delta=float(np.max(abs(a-b))) if a.size else 0.
    assert np.isfinite(delta) and delta<=tol,(key,delta)
    checks[key]+=max(a.size,1);error[key]=max(error[key],delta)
def scalar(v):return float(v) if np.isfinite(v) else None
def reference_formula(p,r):
    p=min(1.,max(0.,float(p)));r=min(1.,max(0.,float(r)))
    return p*r/(p+r-p*r) if p+r-p*r else 0.
def column_roles(values,a,w):return dict(P_A=values['P'][a],R_A=values['R'][a],P_W=values['P'][w],R_W=values['R'][w])
def source_moments(rr,values):
    result={}
    for sid in sorted({r['source_id'] for r in rr}):
        per_order=[]
        for order in sorted({r['order'] for r in rr if r['source_id']==sid}):
            per_condition=[]
            for cond in sorted({r['condition'] for r in rr if r['source_id']==sid and r['order']==order}):
                idx=[i for i,r in enumerate(rr) if r['source_id']==sid and r['order']==order and r['condition']==cond]
                per_condition.append(np.mean([values[i] for i in idx],axis=0))
            per_order.append(np.mean(per_condition,axis=0))
        result[str(sid)]=np.mean(per_order,axis=0)
    return result
def pack(v,bs):
    bs=np.asarray(bs);ok=np.isfinite(bs)
    return dict(mean=scalar(v),ci95=np.quantile(bs[ok],[.025,.975]).tolist() if ok.any() else None,
        bootstrap_defined=int(ok.sum()),bootstrap_undefined=int((~ok).sum()))
def verify_pack(got,expected,key):
    assert set(got)==set(expected),(key,set(got),set(expected))
    for k in expected:close(got[k],expected[k],key+'/'+k)
def stats_from(a):
    a=np.asarray(a,float);vy=np.maximum(0.,a[...,2]-a[...,0]**2);vp=np.maximum(0.,a[...,3]-a[...,1]**2)
    div=lambda num,den,ok:np.divide(num,den,out=np.full(np.broadcast_shapes(np.shape(num),np.shape(den)),np.nan),where=ok)
    return dict(r2=1-div(a[...,5],vy,vy>EPS),mae=a[...,6],mse=a[...,5],
        rho=div(a[...,4]-a[...,0]*a[...,1],np.sqrt(vy*vp),(vy>EPS)&(vp>EPS)),
        truth_mean=a[...,0],prediction_mean=a[...,1],below_zero=a[...,7],above_one=a[...,8])
def audit_regression(rr,ys,ps,got,draws):
    values=[]
    for y,p in zip(ys,ps):
        y=np.array(y);p=np.array(p);d=y-p
        values.append(np.array([sum(y)/len(y),sum(p)/len(p),sum(y*y)/len(y),sum(p*p)/len(p),
            sum(y*p)/len(p),sum(d*d)/len(y),sum(abs(d))/len(y),sum(p<0)/len(p),sum(p>1)/len(p)]))
    source=source_moments(rr,values);ids=list(source);m=np.array(list(source.values()));n=len(ids)
    assert got['cells']==len(rr) and got['sources']==n
    for s in ids:close(got['source_moments'][s],source[s],'regression/source_moments')
    bs=np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=draws)/n
    one=stats_from(m.mean(0));many=stats_from(bs@m)
    for k in one:verify_pack(got['metrics'][k],pack(one[k],many[k]),'regression/'+k)
    for i,s in enumerate(ids):
        vals=stats_from(np.delete(m,i,axis=0).mean(0)) if n>1 else {}
        assert set(got['delete_one_source'][s])==set(vals)
        for k,v in vals.items():close(got['delete_one_source'][s][k],scalar(v),'regression/delete/'+k)
    # Direct sklearn point cross-check using actual hierarchy weights per cell/candidate.
    flat_y=[];flat_p=[];weights=[]
    for i,(r,y,p) in enumerate(zip(rr,ys,ps)):
        s=r['source_id'];orders={a['order'] for a in rr if a['source_id']==s}
        conditions={a['condition'] for a in rr if a['source_id']==s and a['order']==r['order']}
        count=sum(a['source_id']==s and a['order']==r['order'] and a['condition']==r['condition'] for a in rr)
        w=1/n/len(orders)/len(conditions)/count/len(y)
        flat_y.extend(y);flat_p.extend(p);weights.extend([w]*len(y))
    close(got['metrics']['mae']['mean'],mean_absolute_error(flat_y,flat_p,sample_weight=weights),'sklearn/MAE')
    if got['metrics']['r2']['mean'] is not None:
        close(got['metrics']['r2']['mean'],r2_score(flat_y,flat_p,sample_weight=weights),'sklearn/R2')
    return many
def decision_reference(rr,scores,got,draws):
    eligible=[(r,x) for r,x in zip(rr,scores) if r['eligible'] and abs(r['delta_t'])>EPS]
    ids=sorted({r['source_id'] for r,x in eligible});n=len(ids)
    counts=dict(cells=len(rr),sources=len({r['source_id'] for r in rr}),eligible=sum(r['eligible'] for r in rr),
        noop=sum(not r['eligible'] for r in rr),neutral=sum(r['eligible'] and abs(r['delta_t'])<=EPS for r in rr),
        helpful=sum(r['delta_t']>EPS for r,x in eligible),harmful=sum(r['delta_t']< -EPS for r,x in eligible),informative_sources=n,
        accepted=sum(r['eligible'] and x>0 for r,x in zip(rr,scores)),accepted_helpful=sum(r['delta_t']>EPS and x>0 for r,x in eligible),
        accepted_harmful=sum(r['delta_t']< -EPS and x>0 for r,x in eligible),
        accepted_severe_v_harm=sum(r['eligible'] and x>0 and r['delta_v']< -.05 for r,x in zip(rr,scores)))
    samples={}
    if n:
        h=[[x for r,x in eligible if r['source_id']==s and r['delta_t']>EPS] for s in ids]
        f=[[x for r,x in eligible if r['source_id']==s and r['delta_t']< -EPS] for s in ids]
        ph=np.array([int(bool(z)) for z in h]);nf=np.array([int(bool(z)) for z in f])
        counts.update(helpful_sources=int(ph.sum()),harmful_sources=int(nf.sum()),mixed_sources=int((ph*nf).sum()))
        matrix=np.zeros((n,n));tp=[];fp=[];accuracy=[];ap=[];accept=[]
        for i,s in enumerate(ids):
            tp.append(sum(x>0 for x in h[i])/len(h[i]) if h[i] else 0.)
            fp.append(sum(x>0 for x in f[i])/len(f[i]) if f[i] else 0.)
            pairs=[(r,x) for r,x in eligible if r['source_id']==s]
            accuracy.append(sum((x>0)==(r['delta_t']>EPS) for r,x in pairs)/len(pairs))
            ap.append(sum(x>0 and r['delta_t']>EPS for r,x in pairs)/len(pairs));accept.append(sum(x>0 for r,x in pairs)/len(pairs))
            for j in range(n):
                if h[i] and f[j]:
                    matrix[i,j]=sum(1. if a>b else .5 if a==b else 0. for a in h[i] for b in f[j])/len(h[i])/len(f[j])
        def metrics(w):
            w=np.asarray(w);positive=w@ph;negative=w@nf
            div=lambda a,b:np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=b>0)
            true_positive=div(w@tp,positive);false_positive=div(w@fp,negative)
            return dict(auc=div(np.einsum('...i,ij,...j->...',w,matrix,w),positive*negative),
                within_source_auc=div(w@np.diag(matrix),w@(ph*nf)),tpr=true_positive,fpr=false_positive,
                balanced_accuracy=(true_positive+1-false_positive)/2,
                accuracy=div(w@accuracy,w.sum(-1)),accepted_precision=div(w@ap,w@accept))
        bs=np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=draws)
        one=metrics(np.ones(n));samples=metrics(bs)
        for k in one:verify_pack(got['metrics'][k],pack(one[k],samples[k]),'decision/'+k)
        for i,s in enumerate(ids):
            w=np.ones(n);w[i]=0;values=metrics(w)
            for k in values:close(got['delete_one_source'][str(s)][k],scalar(values[k]),'decision/delete/'+k)
        if ph.sum() and nf.sum():
            yy=[];xx=[];ww=[]
            for i,s in enumerate(ids):
                yy += [1]*len(h[i])+[0]*len(f[i]);xx+=h[i]+f[i]
                ww += ([1/len(h[i])]*len(h[i]) if h[i] else [])+([1/len(f[i])]*len(f[i]) if f[i] else [])
            close(got['metrics']['auc']['mean'],roc_auc_score(yy,xx,sample_weight=ww),'sklearn/AUC')
    else:assert got['metrics']=={} and got['delete_one_source']=={}
    assert got['counts']==counts,('counts',got['counts'],counts);checks['decision/counts']+=len(counts)
    values=[]
    for r,x in zip(rr,scores):
        take=r['eligible'] and x>0
        values.append([r['delta_t'] if take else 0.,r['delta_v'] if take else 0.,
            max(r['delta_v'],0.) if take else 0.,min(r['delta_v'],0.) if take else 0.,float(take)])
    source=source_moments(rr,values);uid=list(source);u=np.array(list(source.values()));un=len(uid)
    for s in uid:close(got['utility_source_values'][s],source[s],'decision/utility_source')
    w=np.random.default_rng(SEED).multinomial(un,np.full(un,1/un),size=draws)/un;b=w@u;p=u.mean(0)
    for j,k in enumerate(['delta_t','delta_v','gross_v_gain','gross_v_loss','accept_rate']):
        verify_pack(got['utility'][k],pack(p[j],b[:,j]),'decision/utility/'+k)
    return samples
def audit_summary(rr,got,draws):
    assert got['cells']==len(rr) and got['sources']==len({r['source_id'] for r in rr})
    regs={};decs={}
    for fit in FITS:
        regs[fit]={};decs[fit]={}
        for field in ROLES+['P_pool','R_pool','T_A','T_W','delta_T']:
            if field in ['P_pool','R_pool']:
                k=field[0];yy=[r['true_candidates'][k] for r in rr];pp=[r['predictions'][fit][k] for r in rr]
            elif field in ROLES:yy=[[r['truth'][field]] for r in rr];pp=[[r['readouts'][fit][field]] for r in rr]
            else:
                yy=[[r['anchor_t'] if field=='T_A' else r['winner_t'] if field=='T_W' else r['delta_t']] for r in rr]
                pp=[[r['ladder'][fit]['predicted'][field]] for r in rr]
            regs[fit][field]=audit_regression(rr,yy,pp,got['regression'][fit][field],draws)
        for arm in ARMS:
            decs[fit][arm]=decision_reference(rr,[r['ladder'][fit][arm]['delta_T'] for r in rr],got['decision'][fit][arm],draws)
    for field in got['paired_target_minus_source_regression']:
        for k,v in got['paired_target_minus_source_regression'][field].items():
            a=got['regression']['target_search_fit'][field]['metrics'][k]['mean'];b=got['regression']['source_fit'][field]['metrics'][k]['mean']
            verify_pack(v,pack(a-b if a is not None and b is not None else np.nan,
                regs['target_search_fit'][field][k]-regs['source_fit'][field][k]),'paired/regression/'+k)
    for arm in ARMS:
        for k,v in got['paired_target_minus_source_decision'][arm].items():
            a=got['decision']['target_search_fit'][arm]['metrics'][k]['mean'];b=got['decision']['source_fit'][arm]['metrics'][k]['mean']
            verify_pack(v,pack(a-b if a is not None and b is not None else np.nan,
                decs['target_search_fit'][arm][k]-decs['source_fit'][arm][k]),'paired/decision/'+k)
    for fit in FITS:
        for arm in ARMS[1:]:
            for k,v in got['ladder_minus_predicted'][fit][arm].items():
                a=got['decision'][fit][arm]['metrics'][k]['mean'];b=got['decision'][fit]['predicted']['metrics'][k]['mean']
                verify_pack(v,pack(a-b if a is not None and b is not None else np.nan,
                    decs[fit][arm][k]-decs[fit]['predicted'][k]),'paired/ladder/'+k)
    if draws:
        for order in sorted({r['order'] for r in rr}):audit_summary([r for r in rr if r['order']==order],got['orders'][order],0)
    else:assert not got['orders']
def root_checks():
    import torch
    torch.set_num_threads(1)
    def load(p):
        assert sha(p)==read(p.with_suffix('.json'))['sha256'];checks['payload_sha']+=1
        return torch.load(p,map_location='cpu',weights_only=False)
    lock=read(BASE/'RUNTIME_LOCK.json')
    code=dict(lock['code']);revision=BASE/'RUNTIME_REVISION_001.json'
    if revision.exists():
        rev=read(revision);assert rev['prior_runtime_sha256']==sha(BASE/'RUNTIME_LOCK.json')
        original=BASE/'recovery/metric_key_001/run_tastvg_pr_accessibility_v1.py'
        assert sha(original)==lock['code']['scripts/run_tastvg_pr_accessibility_v1.py']
        code.update(rev['effective_code'])
        for f,h in rev['preserved_outputs'].items():assert sha(ROOT/f)==h,f
    for f,h in {**code,**lock['inputs'],**lock['payloads']}.items():assert sha(ROOT/f)==h,f;checks['immutable_SHA']+=1
    cohort=read(OUT/'COHORT.json');cfg=read(OUT/'CONFIG.json');fit=read(BASE/'FIT_SEAL.json')
    seal=read(OUT/'GLOBAL_READOUT_SEAL.json');join=read(OUT/'LABEL_JOIN.json')
    assert lock['time']<fit['time']<seal['time']<join['time']
    assert not fit['confirmation_GT_read'] and not fit['confirmation_features_read'] and fit['target_search_GT_for_fit']
    assert not seal['confirmation_GT_read'] and not join['confirmation_GT_for_fit']
    assert seal['prediction_sha256']==sha(OUT/'PREDICTIONS.json') and seal['fit_seal_sha256']==sha(BASE/'FIT_SEAL.json')
    predicted={r['cell_key']:r for r in read(OUT/'PREDICTIONS.json')};rows={r['cell_key']:r for r in read(OUT/'ROWS.json')}
    predecessor={r['cell_key']:r for r in read(ROOT/'results/tastvg_structured_separability/2026-10-03/ROWS.json')}
    for ds in ['vidstg','hc2']:
        search=[r for r in cohort if r['dataset']==ds and r['split']=='search'];confirm=[r for r in cohort if r['dataset']==ds and r['split']=='confirm']
        assert len(search)==96 and len(confirm)==48
        assert not {r['source_id'] for r in search}&{r['source_id'] for r in confirm}
        cache={r['cell_key']:load(ROOT/r['feature_file']) for r in search+confirm}
        gt={sp:read(ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_{sp}.json') for sp in ['search','confirm']}
        def labels(z):
            s,e=gt[z['split']][str(z['source_id'])]['span'];ids=z['frame_ids'];pp=[];rr=[];tt=[]
            for i,j in z['candidate_indices']:
                a,b=ids[i],ids[j]+1;n=max(0.,min(e,b)-max(s,a))
                pp.append(n/(b-a));rr.append(n/(e-s));tt.append(n/(e-s+b-a-n))
            return np.array(pp),np.array(rr),np.array(tt)
        x=np.vstack([cache[r['cell_key']]['x'] for r in search]);groups=np.repeat([r['source_id'] for r in search],32)
        unique,counts=np.unique(groups,return_counts=True);w=np.array([1/counts[list(unique).index(s)] for s in groups]);w/=w.sum()
        source=load(ROOT/f'artifacts/tastvg_temporal_information_atlas_v1/{ds}/FROZEN_ATLAS.pt')['models']
        oldarrays={r['cell']:r['predictions'] for r in load(ROOT/f'artifacts/tastvg_temporal_information_atlas_v1/{ds}/SEALED_READOUT.pt')}
        model_path=BASE/ds/'SEARCH_PR.npz';assert sha(model_path)==fit['models'][ds]
        fitted={}
        with np.load(model_path,allow_pickle=False) as z:
            for name in ['P','R']:fitted[name]={k:z[name+'_'+k].copy() for k in ['mean','std','weight','bias','alpha']}
        for name,index in [('P',0),('R',1)]:
            part=slice(512,768) if name=='P' else slice(0,512);a=x[:,part];mean=np.average(a,axis=0,weights=w)
            std=np.sqrt(np.average((a-mean)**2,axis=0,weights=w));std[std<1e-8]=1.
            y=np.hstack([labels(cache[r['cell_key']])[index] for r in search]);bias=np.average(y,weights=w)
            normalized=(a-mean)/std;alpha=cfg['source_fit_alpha'][ds][name]
            assert float(fitted[name]['alpha'])==alpha
            coefficient=solve(normalized.T@(w[:,None]*normalized)+alpha*np.eye(normalized.shape[1]),
                normalized.T@(w*(y-bias)),assume_a='pos')
            for k,value in [('mean',mean),('std',std),('bias',bias),('weight',coefficient)]:close(fitted[name][k],value,'independent_fit/'+k,1e-10)
            checks['independent_ridge_refits']+=1
        for r in search+confirm:
            z=cache[r['cell_key']];pr=predicted[r['cell_key']];out=rows[r['cell_key']];old=predecessor[r['cell_key']]
            a,widx=r['anchor_index'],r['winner_index']
            assert a==old['anchor_index']==z['anchor_index'] and widx==old['winner_index']
            assert r['eligible']==old['eligible'] and z['source_A_temporal_bitwise_parity'] and z['A_spatial_bitwise_parity']
            assert out['predictions']==pr['predictions'];checks['fixed_choices']+=1
            p,rec,t=labels(z)
            for name,y in [('P',p),('R',rec),('T',t)]:close(out['true_candidates'][name],y,'physical_labels')
            for fitname in FITS:
                for name in ['P','R']:
                    part=slice(512,768) if name=='P' else slice(0,512)
                    model=source['candidate/Inside/precision/real' if name=='P' else 'candidate/Endpoint/recall/real'] if fitname=='source_fit' else fitted[name]
                    values=(np.array(z['x'])[:,part]-model['mean'])/model['std']@model['weight']+model['bias']
                    close(pr['predictions'][fitname][name],values,'frozen_predictions',3e-12)
                    if fitname=='source_fit':close(values,oldarrays[r['cell_key']]['candidate/Inside/precision/real' if name=='P' else 'candidate/Endpoint/recall/real'],'source_readout_parity',3e-12)
                close([out['readouts'][fitname][k] for k in ROLES],[column_roles(pr['predictions'][fitname],a,widx)[k] for k in ROLES],'role_extraction',3e-12)
            close([out['truth'][k] for k in ROLES],[column_roles({'P':p,'R':rec},a,widx)[k] for k in ROLES],'role_truth',3e-12)
            close([out['anchor_t'],out['winner_t']],[t[a],t[widx]],'cached_tIoU',3e-12)
            close([out['anchor_v'],out['winner_v']],[old['anchor_v'],old['winner_v']],'cached_vIoU',3e-12)
    assert not torch.cuda.is_initialized()
def main():
    mode=sys.argv[1];assert mode in ['root','public'];tick=time.time()
    if mode=='root':root_checks()
    cfg=read(OUT/'CONFIG.json');rows=read(OUT/'ROWS.json');pred={r['cell_key']:r for r in read(OUT/'PREDICTIONS.json')}
    assert len(rows)==288 and len(pred)==288
    seal=read(OUT/'GLOBAL_READOUT_SEAL.json');assert sha(OUT/'PREDICTIONS.json')==seal['prediction_sha256']
    for r in rows:
        assert pred[r['cell_key']]['predictions']==r['predictions']
        for fit in FITS:
            p=r['readouts'][fit]
            for arm in ARMS:
                z=dict(p)
                for k in {'predicted':[],'GT_anchor':['P_A','R_A'],'GT_winner':['P_W','R_W'],
                    'GT_precision':['P_A','P_W'],'GT_recall':['R_A','R_W'],'GT_all':ROLES}[arm]:z[k]=r['truth'][k]
                a=reference_formula(z['P_A'],z['R_A']);w=reference_formula(z['P_W'],z['R_W'])
                close([r['ladder'][fit][arm][k] for k in ['T_A','T_W','delta_T']],[a,w,w-a],'oracle_ladder',3e-12)
            if abs(r['delta_t'])>EPS:
                assert (r['ladder'][fit]['GT_all']['delta_T']>0)==(r['delta_t']>0);checks['GT_positive_control']+=1
        close(r['delta_t'],r['winner_t']-r['anchor_t'],'delta_t',3e-12)
        close(r['delta_v'],r['winner_v']-r['anchor_v'],'delta_v',3e-12)
    for ds in ['vidstg','hc2']:
        summary=read(OUT/ds/'SUMMARY.json')
        for name,got in summary.items():
            sp,mode0=name.split('_',1)
            rr=[r for r in rows if r['dataset']==ds and r['split']==sp and
                (mode0=='all' or (r['condition']=='clean')==(mode0=='clean'))]
            audit_summary(rr,got,10000)
            print('PR_AUDIT_PANEL',mode,ds,name,len(rr),flush=True)
    receipt=dict(status='pass',mode=mode,time=time.time(),CPU_wall_seconds=time.time()-tick,
        checks=dict(checks),total_scalar_checks=sum(checks.values()),max_error=dict(error),
        new_GPU_calls=0,source_fit_predictions_unchanged=True,target_fit_diagnostic_not_method=True,
        independent_solver='scipy positive-definite solve versus producer eigh' if mode=='root' else None,
        public_does_not_read_private_features_weights_or_GT=mode=='public',
        scope='all fits/predictions/labels/choices/oracle algebra/statistics/bootstrap' if mode=='root' else 'all anonymous scalar statistics/choices/algebra/bootstrap')
    path=OUT/('ROOT_AUDIT.json' if mode=='root' else 'PUBLIC_AUDIT.json')
    assert not path.exists();path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['checks','max_error']},ensure_ascii=False))
if __name__=='__main__':main()
