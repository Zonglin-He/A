"""Independent scalar/source resampling audit; sklearn is used only for AUC."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import sys,json,time,hashlib,collections
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
EPS=1e-12;SEED=20261003;DRAWS=10000
FIELDS=['P_A','R_A','delta_P','delta_R']
SIGN=dict(P_A=-1,R_A=-1,delta_P=1,delta_R=1)
RECIPES={'Inside_Endpoint':('Inside','Endpoint','real'),'Inside_Context':('Inside','Context','real'),
    'Full':('Full','Full','real'),'Geometry':('Geometry','Geometry','real'),
    'Shuffle_Inside_Endpoint':('Inside','Endpoint','shuffle'),'Shuffle_Inside_Context':('Inside','Context','shuffle')}
checks=0;maxerr=0.
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b,tolerance=3e-12):
    global checks,maxerr
    if isinstance(b,dict):
        assert isinstance(a,dict) and set(a)==set(b),(set(a),set(b))
        for k in b:eq(a[k],b[k],tolerance)
    elif isinstance(b,list):
        assert isinstance(a,list) and len(a)==len(b)
        for x,y in zip(a,b):eq(x,y,tolerance)
    elif isinstance(b,(int,float,np.number)) and not isinstance(b,bool):
        assert a is not None and np.isfinite(a) and np.isfinite(b),(a,b)
        err=abs(float(a)-float(b));maxerr=max(maxerr,err);assert err<=tolerance*max(1.,abs(float(b))),(a,b,err)
        checks+=1
    else:assert a==b,(a,b);checks+=1
def defined_mean(v):return float(v) if np.isfinite(v) else None
def interval(v):
    a=np.asarray(v,float);ok=np.isfinite(a);z=a[ok]
    return dict(ci95=np.quantile(z,[.025,.975]).tolist() if len(z) else None,bootstrap_defined=int(ok.sum()),bootstrap_undefined=int((~ok).sum()))
def safe(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b>0)
def draw(n,draws):return np.random.default_rng(SEED).multinomial(n,np.repeat(1/n,n),size=draws).astype(float) if n else np.empty((draws,0))
def batch(rows,recipe,field,outcome,draws):
    rr=[r for r in rows if r['eligible'] and r['label_'+outcome] in ['helpful','harmful']]
    ids=sorted({r['source_id'] for r in rr});n=len(ids);index={s:i for i,s in enumerate(ids)}
    counts=collections.Counter((r['source_id'],r['label_'+outcome]) for r in rr)
    p=np.array([bool(counts[s,'helpful']) for s in ids],float);q=np.array([bool(counts[s,'harmful']) for s in ids],float)
    x=np.array([r['readouts'][recipe][field] for r in rr]);yy=np.array([r['label_'+outcome]=='helpful' for r in rr],int)
    w0=np.array([1/counts[r['source_id'],r['label_'+outcome]] for r in rr],float)
    at=np.array([index[r['source_id']] for r in rr]);matrix=np.zeros((n,n));h=np.zeros(n);f=np.zeros(n);local=np.zeros(n)
    for i in range(n):
        ph=(at==i)&(yy==1);pf=(at==i)&(yy==0)
        if ph.any():h[i]=x[ph].mean()
        if pf.any():f[i]=x[pf].mean()
        if ph.any() and pf.any():
            m=at==i;local[i]=roc_auc_score(yy[m],SIGN[field]*x[m])
    # Independently accumulate row-pair probabilities into source-pair bins.
    if (yy==1).any() and (yy==0).any():
        ph=np.flatnonzero(yy==1);pf=np.flatnonzero(yy==0)
        z=SIGN[field]*x;kernel=(z[ph,None]>z[None,pf]).astype(float)+.5*(z[ph,None]==z[None,pf])
        m=(kernel*w0[ph,None]*w0[None,pf]).ravel()
        i=(at[ph,None]*n+at[None,pf]).ravel();matrix=np.bincount(i,weights=m,minlength=n*n).reshape(n,n)
        point=float(roc_auc_score(yy,z,sample_weight=w0))
    else:point=None
    bs=draw(n,draws);den=(bs@p)*(bs@q)
    boot=safe(np.sum((bs@matrix)*bs,axis=1),den)
    local_boot=safe(bs@local,bs@(p*q))
    hboot=safe(bs@h,bs@p);fboot=safe(bs@f,bs@q)
    within=defined_mean(safe(local.sum(),(p*q).sum()));hm=defined_mean(safe(h.sum(),p.sum()));fm=defined_mean(safe(f.sum(),q.sum()))
    dist_diff=hm-fm if hm is not None and fm is not None else None
    auc=dict(mean=point,**interval(boot),leave_one_source_out={});within0=dict(mean=within,**interval(local_boot),leave_one_source_out={})
    difference=dict(mean=dist_diff,**interval(hboot-fboot),leave_one_source_out={})
    for sid in ids:
        mask=np.array([r['source_id']!=sid for r in rr]);vals=yy[mask]
        auc['leave_one_source_out'][str(sid)]=float(roc_auc_score(vals,SIGN[field]*x[mask],sample_weight=w0[mask])) if len(np.unique(vals))==2 else None
        j=index[sid];keep=np.arange(n)!=j
        within0['leave_one_source_out'][str(sid)]=defined_mean(safe(local[keep].sum(),(p*q)[keep].sum()))
        difference['leave_one_source_out'][str(sid)]=defined_mean(safe(h[keep].sum(),p[keep].sum())-safe(f[keep].sum(),q[keep].sum()))
    def quant(c):
        # Match the locked FP64 empirical inverse CDF in canonical source
        # order. At a jump, changing the summation order can pick a neighbor.
        cells=[np.flatnonzero((at==i)&(yy==c)) for i in range(n)]
        cells=[v for v in cells if len(v)]
        if not cells:return None
        chosen=np.concatenate(cells);xx=x[chosen];ww=w0[chosen]
        ix=np.argsort(xx,kind='stable');cw=np.cumsum(ww[ix])/ww[ix].sum()
        return [float(xx[ix][min(np.argmax(cw>=v),len(ix)-1)]) for v in [.25,.5,.75]]
    counts0=dict(cells=len(rows),sources=len({r['source_id'] for r in rows}),eligible=sum(r['eligible'] for r in rows),
        noop=sum(not r['eligible'] for r in rows),neutral_replacements=sum(r['eligible'] and r['label_'+outcome]=='neutral' for r in rows),
        helpful=int(yy.sum()),harmful=int((yy==0).sum()),helpful_sources=int(p.sum()),harmful_sources=int(q.sum()),informative_sources=n,mixed_sources=int((p*q).sum()))
    per=[dict(source_id=s,helpful=counts[s,'helpful'],harmful=counts[s,'harmful'],helpful_mean=float(h[i]) if p[i] else None,
        harmful_mean=float(f[i]) if q[i] else None,within_auc=float(local[i]) if p[i]*q[i] else None) for i,s in enumerate(ids)]
    if not n:
        expected=dict(counts=counts0,orientation=SIGN[field],auc=None,within_source_auc=None,distributions=None,per_source=[],orders={},draws=draws,seed=SEED)
    else:
        expected=dict(counts=counts0,orientation=SIGN[field],auc=auc,within_source_auc=within0,
            distributions=dict(helpful=dict(mean=hm,**interval(hboot)),harmful=dict(mean=fm,**interval(fboot)),
                helpful_quantiles=quant(1),harmful_quantiles=quant(0),difference=difference),per_source=per,orders={},draws=draws,seed=SEED)
    return expected,(ids,matrix,p,q,local,bs)

def source_stats(rows,fields):
    if not rows:return dict(cells=0,sources=0,metrics={})
    ids=sorted({r['source_id'] for r in rows});a=[]
    for sid in ids:
        src=[r for r in rows if r['source_id']==sid];orders=[]
        for order in sorted({r['order'] for r in src}):
            cells=[r for r in src if r['order']==order];v=[]
            for c in sorted({r['condition'] for r in cells}):v.append(np.array([[r[f] for f in fields] for r in cells if r['condition']==c]).mean(0))
            orders.append(np.array(v).mean(0))
        a.append(np.array(orders).mean(0))
    a=np.array(a);n=len(ids);b=draw(n,DRAWS)@a/n
    return dict(cells=len(rows),sources=n,metrics={f:dict(mean=float(a[:,j].mean()),**interval(b[:,j]),
        source_values={str(s):float(a[i,j]) for i,s in enumerate(ids)},
        leave_one_source_out={str(s):float((a[:,j].sum()-a[i,j])/(n-1)) if n>1 else None for i,s in enumerate(ids)}) for j,f in enumerate(fields)})
def sign(x):return 'up' if x>EPS else 'down' if x< -EPS else 'tie'
def correlate(rows,x,y):
    if not rows:return dict(mean=None,ci95=None,sources=0)
    ids=sorted({r['source_id'] for r in rows});a=[]
    for s in ids:
        rr=[r for r in rows if r['source_id']==s];xx=np.array([r[x] for r in rr]);yy=np.array([r[y] for r in rr])
        a.append([xx.mean(),yy.mean(),(xx**2).mean(),(yy**2).mean(),(xx*yy).mean()])
    a=np.array(a);n=len(ids)
    def value(m):return safe(m[...,4]-m[...,0]*m[...,1],np.sqrt(np.maximum(0,m[...,2]-m[...,0]**2)*np.maximum(0,m[...,3]-m[...,1]**2)))
    return dict(mean=defined_mean(value(a.mean(0))),**interval(value(draw(n,DRAWS)@a/n)),sources=n,
        leave_one_source_out={str(s):defined_mean(value(a[np.arange(n)!=i].mean(0))) if n>1 else None for i,s in enumerate(ids)})

def check_summary(saved,rows,domain):
    cov=dict(cells=len(rows),sources=len({r['source_id'] for r in rows}),eligible=sum(r['eligible'] for r in rows),noop=sum(not r['eligible'] for r in rows),
        helpful_t=sum(r['eligible'] and r['label_t']=='helpful' for r in rows),harmful_t=sum(r['eligible'] and r['label_t']=='harmful' for r in rows),
        neutral_t=sum(r['eligible'] and r['label_t']=='neutral' for r in rows),severe_v=sum(r['delta_v'] is not None and r['delta_v']< -.05 for r in rows))
    eq(saved['coverage'],cov)
    for recipe in RECIPES:
        z=saved['recipes'][recipe]
        for outcome in (['t','v'] if domain=='target' else ['t']):
            for field in FIELDS:
                expected,_=batch(rows,recipe,field,outcome,DRAWS)
                expected['orders']={o:batch([r for r in rows if r['order']==o],recipe,field,outcome,0)[0] for o in sorted({r['order'] for r in rows})}
                eq(z['discrimination'][outcome][field],expected)
        for q,v in z['quadrants'].items():
            rr=[r for r in rows if r['eligible'] and sign(r['readouts'][recipe]['delta_P'])+'/'+sign(r['readouts'][recipe]['delta_R'])==q]
            eq(v['counts'],dict(cells=len(rr),helpful=sum(r['label_t']=='helpful' for r in rr),harmful=sum(r['label_t']=='harmful' for r in rr),
                neutral=sum(r['label_t']=='neutral' for r in rr),sources=len({r['source_id'] for r in rr})))
            pack=[dict(source_id=r['source_id'],order=r['order'],condition=r['condition'],dt=r['delta_t'],dv=r['delta_v'] or 0.,
                helpful=float(r['label_t']=='helpful'),harmful=float(r['label_t']=='harmful'),neutral=float(r['label_t']=='neutral'),
                true_delta_P=r.get('true_delta_P',0.),true_delta_R=r.get('true_delta_R',0.)) for r in rr]
            eq(v['summary'],source_stats(pack,['dt','dv','helpful','harmful','neutral']+(['true_delta_P','true_delta_R'] if domain=='target' else [])))
        if domain=='target':
            pack=[dict(source_id=r['source_id'],**{f:r['readouts'][recipe][f] for f in FIELDS},true_P_A=r['true_P_A'],true_R_A=r['true_R_A'],A8_t=r['anchor_t'],dt=r['delta_t']) for r in rows]
            for label,val in z['correlations'].items():
                x,y=label.split('/')
                # Pearson moments subtract close numbers for the almost
                # constant shuffled recall control. This bounded rho-only
                # tolerance does not apply to AUC, task labels or choices.
                eq(val,correlate(pack,x,y),tolerance=1e-8)
        else:eq(z['correlations'],{})
def group_rows(rows,source,ds):
    rr={'source_validation':[r for r in source if r['dataset']==ds]}
    for sp in ['search','confirm']:
        for mode in ['corrupt','clean']:rr[sp+'_'+mode]=[r for r in rows if r['dataset']==ds and r['split']==sp and (r['condition']=='clean')==(mode=='clean')]
    return rr

def run(mode,folder):
    tick=time.time();out=Path(folder);cfg=read(out/'CONFIG.json');eq(cfg['recipes'],{k:list(v) for k,v in RECIPES.items()})
    feature_rows=read(out/'FEATURE_ROWS.json');rows=read(out/'ROWS.json');src=read(out/'SOURCE_ROWS.json')
    assert len(feature_rows)==len(rows)==288 and len(src)==47
    eq(sha(out/'FEATURE_ROWS.json'),read(out/'FEATURE_SEAL.json')['feature_sha256'])
    assert read(out/'FEATURE_SEAL.json')['time']<read(out/'LABEL_JOIN.json')['time']
    features={r['cell_key']:r for r in feature_rows};assert len(features)==288
    for r in rows:
        eq({k:r[k] for k in features[r['cell_key']]},features[r['cell_key']])
        eq(r['delta_t'],r['winner_t']-r['anchor_t']);eq(r['delta_v'],r['winner_v']-r['anchor_v'])
        for kind in ['t','v']:eq(r['label_'+kind],'helpful' if r['delta_'+kind]>EPS else 'harmful' if r['delta_'+kind]< -EPS else 'neutral')
        eq(r['true_delta_P'],r['true_P_W']-r['true_P_A']);eq(r['true_delta_R'],r['true_R_W']-r['true_R_A'])
        for p,r0,t in [(r['true_P_A'],r['true_R_A'],r['anchor_t']),(r['true_P_W'],r['true_R_W'],r['winner_t'])]:
            eq(t,1/(1/p+1/r0-1) if p and r0 else 0.)
        for v in r['readouts'].values():eq(v['delta_P'],v['P_W']-v['P_A']);eq(v['delta_R'],v['R_W']-v['R_A'])
        if not r['eligible']:eq(r['winner_index'],r['anchor_index']);eq(r['delta_t'],0.);eq(r['delta_v'],0.)
    for r in src:
        eq(r['delta_t'],r['winner_t']-r['anchor_t']);eq(r['anchor_index'],0)
        for v in r['readouts'].values():eq(v['delta_P'],v['P_W']-v['P_A']);eq(v['delta_R'],v['R_W']-v['R_A'])
    old=ROOT/'results/tastvg_temporal_latent_quality/2026-10-03'
    # Public auditor also verifies unchanged choices/labels from already public predecessors.
    oldscore={r['cell_key']:r for r in read(old/'SCORE_ROWS.json')}
    oldsource={(r['dataset'],r['source_id']):r for r in read(old/'SOURCE_ROWS.json')}
    for ds in ['vidstg','hc2']:
        metrics={'/'.join(str(r[k]) for k in ['dataset','split','condition','order','arrival']):r for sp in ['search','confirm'] for r in read(old/sp/ds/'ROWS.json') if r['expert_scheduled']}
        for r in rows:
            if r['dataset']!=ds:continue
            o=oldscore[r['cell_key']];a=o['anchor_index'];z=np.array(o['scores']['L']);top=np.flatnonzero(z.max()-z<=EPS)
            w=int(top[0]) if len(top)==1 and z.max()>z[a]+EPS else a
            eq(r['winner_index'],w);eq(r['eligible'],w!=a);eq(r['anchor_index'],a)
            m=metrics[r['cell_key']]
            eq(r['anchor_t'],m['candidate_t'][a]);eq(r['winner_t'],m['candidate_t'][w]);eq(r['anchor_v'],m['candidate_v'][a]);eq(r['winner_v'],m['candidate_v'][w])
            for k in ['A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256']:eq(r[k],o[k])
        for r in src:
            if r['dataset']!=ds:continue
            o=oldsource[ds,r['source_id']];assert o['split']=='validation';z=np.array(o['frozen_scores']['L']);top=np.flatnonzero(z.max()-z<=EPS)
            w=int(top[0]) if len(top)==1 and z.max()>z[0]+EPS else 0
            eq(r['winner_index'],w);eq(r['anchor_t'],o['candidate_t'][0]);eq(r['winner_t'],o['candidate_t'][w]);eq(r['eligible'],w!=0)
    if mode=='root':
        lock=read(ROOT/'artifacts/tastvg_structured_separability_v1/RUNTIME_LOCK.json')
        for p,h in {**lock['code'],**lock['inputs']}.items():eq(sha(ROOT/p),h)
        import torch
        for ds in ['vidstg','hc2']:
            atlas=ROOT/'artifacts/tastvg_temporal_information_atlas_v1'/ds/'SEALED_READOUT.pt'
            prior={r['cell']:r['predictions'] for r in torch.load(atlas,map_location='cpu',weights_only=False)}
            for r in [r for r in rows+src if r['dataset']==ds]:
                a,w=r['anchor_index'],r['winner_index'];scores=prior[r['cell_key']]
                for recipe,(pv,rv,c) in RECIPES.items():
                    pp=np.asarray(scores[f'candidate/{pv}/precision/{c}']);rr=np.asarray(scores[f'candidate/{rv}/recall/{c}'])
                    eq(r['readouts'][recipe],dict(P_A=float(pp[a]),R_A=float(rr[a]),P_W=float(pp[w]),R_W=float(rr[w]),delta_P=float(pp[w]-pp[a]),delta_R=float(rr[w]-rr[a])))
            plans=read(ROOT/f'artifacts/tastvg_current_correction_views_v1/{ds}/PLAN.json')['rows']
            labels={sp:read(ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_{sp}.json') for sp in ['search','confirm']}
            for r in [r for r in rows if r['dataset']==ds]:
                ids=plans[r['source_id']]['frame_ids'];gs,ge=labels[r['split']][str(r['source_id'])]['span']
                for tag,index in [('A',r['anchor_index']),('W',r['winner_index'])]:
                    i,j=oldscore[r['cell_key']]['candidate_indices'][index];s,e=ids[i],ids[j]+1
                    overlap=max(0,min(e,ge)-max(s,gs))
                    eq(r['true_P_'+tag],overlap/(e-s));eq(r['true_R_'+tag],overlap/(ge-gs))
        assert not torch.cuda.is_initialized();eq(sha(ROOT/'methods/CURRENT_METHOD.json'),cfg['production_method_sha256'])
    controls=read(out/'PAIRED_CONTROLS.json')
    for ds in ['vidstg','hc2']:
        summaries=read(out/ds/'SUMMARY.json')
        for name,rr in group_rows(rows,src,ds).items():
            check_summary(summaries[name],rr,'source' if name=='source_validation' else 'target')
            for item in controls[ds][name]:
                _,(ids,a,p,q,d,w)=batch(rr,item['left'],item['field'],item['outcome'],DRAWS)
                _,(ids0,b,p0,q0,d0,w0)=batch(rr,item['right'],item['field'],item['outcome'],DRAWS)
                assert ids==ids0
                eq(item['sources'],len(ids))
                for kind in ['class_source_auc','within_source_auc']:
                    if kind=='class_source_auc':
                        den=(w@p)*(w@q);num=np.sum((w@(a-b))*w,axis=1);one=safe((a-b).sum(),p.sum()*q.sum())
                    else:den=w@(p*q);num=w@(d-d0);one=safe((d-d0).sum(),(p*q).sum())
                    eq(item['metrics'][kind],dict(mean=defined_mean(one),**interval(safe(num,den))))
    receipt=dict(status='pass',mode=mode,time=time.time(),checks=checks,max_absolute_error=maxerr,
        CPU_wall_seconds=time.time()-tick,independent_sklearn_AUC=True,source_bootstrap_and_deletion=True,
        frozen_readout_vector_reconstruction=mode=='root',public_predecessor_choice_label_checks=True,
        no_new_fits=True,no_new_GPU=True,scalar_tolerance=3e-12,Pearson_only_tolerance=1e-8)
    path=out/('ROOT_AUDIT.json' if mode=='root' else 'PUBLIC_AUDIT.json');assert not path.exists();path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else ROOT/'results/tastvg_structured_separability/2026-10-03')
