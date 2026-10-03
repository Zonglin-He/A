"""Independent private CPU readback; no atlas learner/label/metric imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import sys,json,time,hashlib,collections
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/tastvg_temporal_information_atlas_v1'
PREV=ROOT/'artifacts/tastvg_temporal_latent_quality_v1'
PUB=ROOT/'results/tastvg_temporal_information_atlas/2026-10-03'
PARTS={'Endpoint':slice(0,512),'Inside':slice(512,768),'Context':slice(768,1280),
       'Contrast':slice(1280,1792),'Full':slice(0,1792)}
ALPHAS=[.001,.01,.1,1.,10.,100.,1000.]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    torch.set_num_threads(2);tick=time.time();checks=collections.Counter();errors=collections.defaultdict(float)
    def close(a,b,key,tol=1e-8):
        a=np.asarray(a,float);b=np.asarray(b,float);assert a.shape==b.shape,(key,a.shape,b.shape)
        d=float(np.max(abs(a-b))) if a.size else 0.;assert d<tol,(key,d)
        errors[key]=max(errors[key],d);checks[key]+=a.size
    def loaded(p):
        assert sha(p)==read(p.with_suffix('.json'))['sha256'];checks['payload_SHA']+=1
        return torch.load(p,map_location='cpu',weights_only=False)
    lock=read(BASE/'RUNTIME_LOCK.json');fit=read(BASE/'FIT_SEAL.json');seal=read(BASE/'GLOBAL_READOUT_SEAL.json')
    assert fit['source_only_fit'] and not fit['target_GT_for_fit']
    assert fit['time']<seal['time']<read(PUB/'LABEL_JOIN.json')['time']
    assert sha(BASE/'FIT_SEAL.json')==seal['source_fit_seal_sha256']
    for f,h in {**lock['code'],**lock['metadata'],**lock['payloads']}.items():
        assert sha(ROOT/f)==h,f;checks['immutable_hashes']+=1
    inp=read(PREV/'SOURCE_INPUTS.json');sourcegt=read(PREV/'SOURCE_GT.json');cfg=read(PUB/'CONFIG.json')
    def labels(ids,intervals,gt,anchor):
        s,e=gt;width=ids[-1]+1-ids[0];pos=(ids-ids[0])/width;event=(ids>=s)&(ids<e)
        frame={'position':pos,'event':event.astype(float),'start_distance':(ids-s)/width,
               'end_distance':(e-ids)/width,'phase':np.where(event,(ids-s)/(e-s),np.nan)}
        pp=[];rr=[];tt=[]
        for a,b in intervals:
            overlap=max(0.,min(e,b)-max(s,a));pp.append(overlap/(b-a));rr.append(overlap/(e-s))
            tt.append(overlap/(b-a+e-s-overlap))
        candidate={'precision':np.array(pp),'recall':np.array(rr),'tiou':np.array(tt)}
        candidate['delta']=candidate['tiou']-candidate['tiou'][anchor]
        return frame,candidate
    def shuffle(y,source,family,mask,anchor=None):
        seed=int(hashlib.sha256(f'20261003|{source}|{family}'.encode()).hexdigest()[:16],16)
        at=np.flatnonzero(mask);at=at[at!=anchor] if anchor is not None else at
        z=y.copy();z[at]=y[np.random.default_rng(seed).permutation(at)]
        close(np.sort(z[mask]),np.sort(y[mask]),'shuffle_histogram',1e-12);return z
    def packet(z,gt,source,panel,domain):
        ids=np.asarray(z['frame_ids'],float);h=z['hidden'].double().numpy();anchor=0 if domain=='source' else z['anchor_index']
        pairs=[r['indices'] for r in z['candidates']] if domain=='source' else z['candidate_indices']
        x=[];geo=[];ints=[]
        sourceinput=inp[z['dataset']][source]['input'] if domain=='source' else plans[source]['input']
        fps=sourceinput['fps'];width=ids[-1]+1-ids[0]
        for i,j in pairs:
            s,e=ids[i],ids[j]+1;l=np.flatnonzero((ids>=s-fps)&(ids<s));r=np.flatnonzero((ids>=e)&(ids<e+fps))
            ml=np.mean(h[l],axis=0) if len(l) else np.zeros(256);mr=np.mean(h[r],axis=0) if len(r) else np.zeros(256)
            inside=np.sum(h[i:j+1],axis=0)/(j-i+1)
            x.append(np.concatenate([h[i],h[j],inside,ml,mr,h[i]-ml,h[j]-mr]))
            geo.append([(s-ids[0])/width,(e-ids[0])/width,(e-s)/width]);ints.append([s,e])
        close(x,z['x'],'independent_candidate_features');close(geo,z['geometry'],'independent_geometry')
        frame,candidate=labels(ids,ints,gt,anchor)
        return dict(h=h,x=np.asarray(x),geometry=np.asarray(geo),frame=frame,candidate=candidate,
            anchor=anchor,source=source,domain=domain,panel=panel,frames=len(ids))
    def x_for(p,family,view,task):
        if family=='frame':return p['h'] if view=='Hidden' else p['frame']['position'][:,None]
        x=p['geometry'] if view=='Geometry' else p['x'][:,PARTS[view]]
        return x-x[p['anchor']] if task=='delta' else x
    def raw_moments(y,p,event=False,logit=None):
        if not len(y):return None
        d=p-y;var=np.var(y)
        z=dict(y=float(np.mean(y)),y2=float(np.mean(y*y)),prediction=float(np.mean(p)),
            mse=float(np.mean(d*d)),mae=float(np.mean(abs(d))),within_r2=float(1-np.mean(d*d)/var) if var>1e-12 else None,
            auc=None,ap=None,prevalence=None,logloss=None,n=len(y))
        if event:
            z['prevalence']=float(np.mean(y))
            if logit is not None:z['logloss']=float(np.mean(np.logaddexp(0,logit)-y*logit))
            else:
                pr=np.clip(p,1e-15,1-1e-15);z['logloss']=float(-np.mean(y*np.log(pr)+(1-y)*np.log1p(-pr)))
            positives=int(y.sum());negatives=len(y)-positives
            if positives and negatives:
                groups=[]
                for value in np.unique(p):
                    at=p==value;groups.append((int(y[at].sum()),int(at.sum()-y[at].sum())))
                below=0;numer=0.
                for pos,neg in groups:numer+=pos*(below+neg*.5);below+=neg
                z['auc']=numer/(positives*negatives)
                tp=0;total=0;ap=0.
                for pos,neg in reversed(groups):
                    tp+=pos;total+=pos+neg;ap+=(pos/positives)*(tp/total)
                z['ap']=ap
        return z
    models_total=0;packets_total=0
    for ds in ['vidstg','hc2']:
        plans=read(ROOT/f'artifacts/tastvg_current_correction_views_v1/{ds}/PLAN.json')['rows']
        assert not {r['source'] for r in inp[ds]}&{r['source'] for r in plans}
        assert not {r['input']['video_sha256'] for r in inp[ds]}&{r['input']['video_sha256'] for r in plans}
        src={}
        for r in inp[ds]:
            z=loaded(PREV/ds/'source_features'/f'{r["index"]:04}.pt')
            src[r['index']]=packet(z,sourcegt[ds][str(r['index'])],r['index'],r['split'],'source')
        tr=[p for p in src.values() if p['panel']=='train'];va=[p for p in src.values() if p['panel']=='validation']
        assert [len(tr),len(va)]==cfg['source_splits'][ds]
        assert sha(BASE/ds/'FROZEN_ATLAS.pt')==fit['probe_hashes'][ds]
        frozen=loaded(BASE/ds/'FROZEN_ATLAS.pt');models=frozen['models'];stats=read(BASE/ds/'FIT_SUMMARY.json')
        designs={};solves={}
        def design(family,view,task):
            kind='phase' if task=='phase' else ('delta' if task=='delta' else 'ordinary');key=(family,view,kind)
            if key not in designs:
                def join(pp):
                    xs=[];group=[];maps=[]
                    for p in pp:
                        mask=np.isfinite(p[family][task]);n=int(mask.sum())
                        if not n:continue
                        xs.append(x_for(p,family,view,task)[mask]);group.extend([p['source']]*n);maps.append((p,mask))
                    x=np.concatenate(xs);group=np.array(group);u,c=np.unique(group,return_counts=True);lookup=dict(zip(u,c))
                    w=np.array([1/lookup[g] for g in group],float);w/=w.sum();return x,w,maps
                x,w,maps=join(tr);vx,vw,vmaps=join(va);center=w@x;std=np.sqrt(w@((x-center)**2));std[std<1e-8]=1
                mean=np.zeros_like(center) if kind=='delta' else center;a=(x-mean)/std;v=(vx-mean)/std
                designs[key]=(x,w,maps,vx,vw,vmaps,mean,std,a,v,a.T@(w[:,None]*a))
            return key,designs[key]
        for name,m in models.items():
            family,view,task,control=name.split('/');key,d=design(family,view,task)
            x,w,maps,vx,vw,vmaps,mean,std,a,v,cov=d
            close(m['mean'],mean,'source_only_mean');close(m['std'],std,'source_only_std')
            yy=[];vyy=[]
            for p,mask in maps:
                y=p[family][task]
                if control=='shuffle':
                    y=shuffle(y,f'{ds}/{p["source"]}','candidate' if family=='candidate' else ('phase' if task=='phase' else 'frame'),mask,p['anchor'] if task=='delta' else None)
                yy.extend(y[mask])
            for p,mask in vmaps:vyy.extend(p[family][task][mask])
            y=np.array(yy);vy=np.array(vyy)
            if m['model']=='ridge':
                bias=0. if task=='delta' else float(w@y);close(m['bias'],bias,'ridge_intercept')
                rhs=a.T@(w*(y-bias));path=[]
                for alpha in ALPHAS:
                    skey=(key,alpha)
                    if skey not in solves:solves[skey]=np.linalg.cholesky(cov+alpha*np.eye(cov.shape[0]))
                    from scipy.linalg import cho_solve
                    beta=cho_solve((solves[skey],True),rhs,check_finite=False)
                    pred=v@beta+bias;loss=float(vw@((pred-vy)**2));path.append(loss)
                    old=stats[name]['path'][ALPHAS.index(alpha)]
                    close(loss,old['validation_MSE'],'independent_ridge_validation_path')
                    close(float(w@((a@beta+bias-y)**2)),old['training_MSE'],'independent_ridge_training_path')
                    if alpha==m['alpha']:close(beta,m['weight'],'independent_ridge_coefficients',1e-7)
                chosen=min(range(7),key=lambda i:(stats[name]['path'][i]['validation_MSE'],ALPHAS[i]))
            else:
                z=a@m['weight']+m['bias'];grad=a.T@(w*(expit(z)-y))+m['alpha']*m['weight']
                close(grad,np.zeros_like(grad),'independent_logistic_KKT',1e-5)
                close(float(w@(expit(z)-y)),0.,'independent_logistic_intercept',1e-5)
                vz=v@m['weight']+m['bias'];loss=float(vw@(np.logaddexp(0,vz)-vy*vz))
                close(loss,stats[name]['path'][stats[name]['selected_index']]['validation_logloss'],'logistic_selected_validation')
                chosen=min(range(7),key=lambda i:(stats[name]['path'][i]['validation_logloss'],ALPHAS[i]))
            assert chosen==stats[name]['selected_index'] and ALPHAS[chosen]==m['alpha'];models_total+=1
        assert sha(BASE/ds/'SEALED_READOUT.pt')==seal['files'][ds]
        predictions={r['cell']:r['predictions'] for r in loaded(BASE/ds/'SEALED_READOUT.pt')}
        targetlabels={p:read(ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_{p}.json') for p in ['search','confirm']}
        packets={f'source/{ds}/{p["source"]}':p for p in va}
        for f in sorted((PREV/ds/'target_features').glob('*.pt')):
            z=loaded(f);p=packet(z,targetlabels[z['split']][str(z['source_id'])]['span'],z['source_id'],z['split'],'target')
            assert z['A_state_pre_sha256'] and z['source_A_temporal_bitwise_parity'] and z['A_spatial_bitwise_parity']
            packets[z['cell_key']]=p
        exports={r['cell']:r for r in read(PUB/ds/'ROWS.json')};assert set(exports)==set(packets)==set(predictions)
        for cell,p in packets.items():
            scores=predictions[cell];r=exports[cell];assert r['anchor_index']==p['anchor'] and r['frames']==p['frames']
            assert r['event_frames']==int(p['frame']['event'].sum());packets_total+=1
            for name,m in models.items():
                family,view,task,control=name.split('/');x=x_for(p,family,view,task)
                z=(x-m['mean'])/m['std']@m['weight']+m['bias'];out=expit(z) if m['model']=='logistic' else z
                close(out,scores[name],'independent_frozen_readout')
                if task=='delta':close(out[p['anchor']],0.,'delta_anchor_zero')
            for name,out in scores.items():
                if name.endswith('/logit'):continue
                family,view,task,control=name.split('/');lab='delta' if task=='delta_from_absolute' else task
                if task=='delta_from_absolute':
                    raw=scores[f'candidate/{view}/tiou/{control}'];close(out,raw-raw[p['anchor']],'absolute_delta_arithmetic')
                y=p[family][lab];mask=np.isfinite(y)
                expected=raw_moments(y[mask],out[mask],task=='event',scores[name+'/logit'][mask] if name+'/logit' in scores else None)
                actual=r['metrics'][name]
                assert (expected is None)==(actual is None)
                if expected:
                    for k,v in expected.items():
                        if v is None:assert actual[k] is None
                        else:close(v,actual[k],'independent_point_metrics',1e-8)
                if family=='frame' and task in ['start_distance','end_distance']:
                    pos=p['frame']['position'];ep=pos-out if task=='start_distance' else pos+out
                    truth=pos-y if task=='start_distance' else pos+y
                    ex=r['position_removed_endpoints'][name]
                    close(abs(ep.mean()-truth.mean()),ex['endpoint_abs_error'],'position_removed_endpoints')
            for family,tasks in [('frame',['position','event','start_distance','end_distance','phase']),('candidate',['precision','recall','tiou','delta'])]:
                for task in tasks:
                    valid=[q[family][task][np.isfinite(q[family][task])] for q in tr];valid=[v for v in valid if len(v)]
                    null=float(np.mean([v.mean() for v in valid]));close(null,frozen['null'][family+'/'+task],'source_only_null')
                    y=p[family][task];mask=np.isfinite(y);ex=raw_moments(y[mask],np.full(mask.sum(),null),task=='event')
                    actual=r['metrics'][f'{family}/Null/{task}/real']
                    assert (ex is None)==(actual is None)
                    if ex:
                        for k,v in ex.items():
                            if v is None:assert actual[k] is None
                            else:close(v,actual[k],'independent_null_metrics')
        print('ATLAS_ROOT_PASS',ds,'models',len(models),'packets',len(packets),flush=True)
    assert models_total==136 and packets_total==335 and not torch.cuda.is_initialized()
    receipt=dict(status='pass',checks=dict(checks),maximum_errors=dict(errors),frozen_models=136,
        source_queries=190,source_validation_cells=47,target_cells=288,evaluation_cells=335,
        source_fit_validation_only=True,target_GT_not_used_for_fit=True,source_native_target_A8_anchors_disclosed=True,
        CPU_only=True,CUDA_initialized=False,new_model_calls=0,new_expert_calls=0,
        verification_scope='Independent labels/features, every selected ridge solve and ridge path, selected logistic KKT/validation, every frozen point readout and anonymous moment; aggregation checked by separate public audit.',
        CPU_wall_seconds=time.time()-tick)
    write(BASE/'FINAL_ROOT_AUDIT.json',receipt);write(PUB/'ROOT_READBACK.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)
if __name__=='__main__':main()
