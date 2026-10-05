"""Independent coordinate/IoU/dense audit and portable anonymous result audit."""
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.decota_public_result_io_v1 import read as public_read
import numpy as np


def independent_stats(rr,fields):
    ids=sorted({r['source_id'] for r in rr})
    if not ids:return {}
    x=np.array([[np.mean([r[f] for r in rr if r['source_id']==i]) for f in fields] for i in ids])
    rng=np.random.default_rng(20261004);b=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return {f:(float(x[:,j].mean()),np.quantile(b[:,j],[.025,.975])) for j,f in enumerate(fields)}


def independent_association(rr,score):
    ids=sorted({r['source_id'] for r in rr});n=len(ids)
    if not n:return None
    moments=[];hp=[];hn=[];kernel=np.zeros((n,n));blocks=[]
    for i in ids:
        q=[r for r in rr if r['source_id']==i];x=np.array([r[score] for r in q]);y=np.array([r['correction_v'] for r in q]);w=1/len(q)
        dx=x-x.mean();dy=y-y.mean()
        moments.append([x.mean(),y.mean(),(dx*dx).mean(),(dy*dy).mean(),(dx*dy).mean()])
        hp.append(float((y>0).sum())*w);hn.append(float((y<0).sum())*w);blocks.append((x,y,w))
    for a,(x,y,wa) in enumerate(blocks):
        pos=x[y>0]
        for b,(u,v,wb) in enumerate(blocks):
            neg=u[v<0];diff=pos[:,None]-neg[None,:]
            kernel[a,b]=float(((diff>0)+.5*(diff==0)).sum())*wa*wb
    m=np.array(moments);hp=np.array(hp);hn=np.array(hn)
    rng=np.random.default_rng(20261005);counts=rng.multinomial(n,np.full(n,1/n),size=10000)
    counts=np.vstack([np.ones(n),counts]);total=counts.sum(1);mom=counts@m/total[:,None]
    dx=m[None,:,0]-mom[:,None,0];dy=m[None,:,1]-mom[:,None,1]
    covariance=(counts*(m[None,:,4]+dx*dy)).sum(1)/total
    vx=(counts*(m[None,:,2]+dx*dx)).sum(1)/total;vy=(counts*(m[None,:,3]+dy*dy)).sum(1)/total
    corr=np.divide(covariance,np.sqrt(np.maximum(vx*vy,0)),out=np.full(len(counts),np.nan),where=(vx>1e-24)&(vy>1e-24))
    numerator=np.einsum('bi,ij,bj->b',counts,kernel,counts);den=(counts@hp)*(counts@hn)
    auc=np.divide(numerator,den,out=np.full(len(counts),np.nan),where=den>0)
    out={}
    for name,val in [('correlation',corr),('AUC',auc)]:
        valid=val[1:][np.isfinite(val[1:])]
        out[name]=dict(mean=float(val[0]) if np.isfinite(val[0]) else None,ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,valid_draws=len(valid))
    return out


def close(a,b):
    if a is None or b is None:assert a is b,(a,b);return
    assert np.allclose(a,b,rtol=0,atol=2e-10),(a,b)


def public(folder):
    tick=time.time();tr=public_read(folder/'temporal/ROWS.json');sr=public_read(folder/'spatial/ROWS.json')
    ts=public_read(folder/'temporal/SUMMARY.json');ss=public_read(folder/'spatial/SUMMARY.json');ass=public_read(folder/'spatial/ASSOCIATIONS.json');checks=0
    assert len(tr)==8064 and len(sr)==2304
    def subset(rr,ds,sp,arm,group):
        q=[r for r in rr if r['dataset']==ds and r['split']==sp and r['arm']==arm]
        if group=='corruption':return [r for r in q if r['condition']!='clean']
        if group=='non_directional_corruption':return [r for r in q if r['condition']!='clean' and not r['directional']]
        if group in ['order1','order2']:return [r for r in q if r['condition']!='clean' and r['order']==group]
        return [r for r in q if r['condition']==group]
    for rows,sums in [(tr,ts),(sr,ss)]:
        for ds,dsx in sums.items():
            for sp,spx in dsx.items():
                for arm,ax in spx.items():
                    for group,z in ax.items():
                        rr=subset(rows,ds,sp,arm,group);assert len(rr)==z['cells']
                        fields=list(z['metrics']);expected=independent_stats(rr,fields)
                        for f,(mean,ci) in expected.items():close(mean,z['metrics'][f]['mean']);close(ci,z['metrics'][f]['ci95']);checks+=3
                        d='correction_v' if rows is sr else 'delta_v'
                        assert z['tails']['harm_gt5pp']==sum(r[d]<-.05 for r in rr)
                        assert z['tails']['harm_gt20pp']==sum(r[d]<-.2 for r in rr);checks+=2
    for ds,dx in ass.items():
        for sp,sx in dx.items():
            for arm,ax in sx.items():
                for group,gx in ax.items():
                    rr=subset(sr,ds,sp,arm,group)
                    for score,z in gx.items():
                        e=independent_association(rr,score)
                        if not rr:continue
                        for k in ['correlation','AUC']:
                            close(z[k]['mean'],e[k]['mean']);close(z[k]['ci95'],e[k]['ci95']);assert z[k]['valid_draws']==e[k]['valid_draws'];checks+=4
    # Fixed gates independently evaluated from saved public numeric results.
    dec=public_read(folder/'DECISION.json');tg={};sg={}
    for ds in ['vidstg','hc2']:
        a=ts[ds]['confirm']['consensus_episodic']['corruption']['metrics'];b=ts[ds]['search']['consensus_episodic']['corruption']['metrics']
        tg[ds]=all(a[k]['ci95'][0]>0 and b[k]['mean']>0 for k in ['delta_t','delta_v'])
        passed=True
        for arm in ['episodic','online100']:
            a=ass[ds]['confirm'][arm]['corruption']['delta_event'];b=ass[ds]['search'][arm]['corruption']['delta_event']
            for metric,threshold in [('correlation',0),('AUC',.5)]:
                passed=passed and a[metric]['ci95'] is not None and a[metric]['ci95'][0]>threshold and b[metric]['mean'] is not None and b[metric]['mean']>threshold
        sg[ds]=bool(passed)
    assert dec['temporal']['dataset_gates']==tg and dec['spatial']['dataset_gates']==sg
    assert dec['temporal']['qualified']==all(tg.values()) and dec['spatial']['qualified']==all(sg.values())
    out=dict(status='pass',checks=checks,rows=len(tr)+len(sr),independent_source_bootstrap=True,independent_source_pair_AUC_kernel=True,
             independent_decisions=True,private_gt_not_available=True,seconds=time.time()-tick)
    return out


def root():
    import torch
    from scripts.decota_transform_common_v1 import BASE,PUB,verify,checked,old,c1,DATASETS
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.tastvg_oracle_event5_v1 import official
    verify();bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');ex=read(BASE/'GT_EXPOSURE.json');assert ex['time']>bar['time'] and not bar['GT_read']
    cpu=read(BASE/'CPU_LOCK.json')
    for f,h in {**cpu['pins'],**cpu['labels']}.items():assert sha(ROOT/f)==h
    cases={};tr=public_read(PUB/'temporal/ROWS.json');sr=public_read(PUB/'spatial/ROWS.json');checks=0;tick=time.time()
    for ds in DATASETS:
        plan=read(BASE/ds/'PLAN.json')
        for row in plan['rows']:
            for cond in plan['conditions']:
                f=BASE/'predictions'/ds/cond/f"{row['ordinal']:05}.pt";assert sha(f)==bar['files'][str(f.relative_to(BASE))];x=checked(f)
                ids=row['frame_ids'];native=x['native_interval'];assert ids==x['frame_ids']
                n=len(ids);gap=max(1,int(np.median(np.diff(ids))));pad=max(2,int(np.ceil(n/8)))
                a=ids.index(native[0])//2;end=ids.index(native[1]-1)+1;b=end+(n-end+1)//2
                if b-a<4:a,b=0,n
                assert x['temporal_views']['crop']['spec']['a']==a and x['temporal_views']['crop']['spec']['b']==b
                assert ids[a]<=native[0] and ids[b-1]+1>=native[1]
                assert x['temporal_views']['shift']['spec']['delta']==pad*gap;checks+=4
                intervals=[native]
                for name in ['shift','crop']:
                    v=x['temporal_views'][name];raw=v['interval'];s=v['spec']
                    if v['identity_reused']:expected=native
                    else:
                        off=s['origin']-(s['delta'] if name=='shift' else 0);rr=[raw[0]+off,raw[1]+off]
                        cc=[max(ids[0],rr[0]),min(ids[-1]+1,rr[1])];expected=native if cc[1]<=cc[0] else cc
                        assert rr==v['mapped']['raw'];checks+=2
                    assert expected==v['mapped']['interval'];intervals.append(expected);checks+=2
                med=[sorted([iv[j] for iv in intervals])[1] for j in range(2)];assert med==x['consensus_interval'];checks+=2
                for key,ref in x['references'].items():
                    path=ROOT/ref['path'];assert sha(path)==ref['sha256'];z=old.checked(path)
                    for suffix,state in [('before',z['initial']),('after',z['fit']['state'])]:
                        for name in ['flip','dim95']:
                            q=x['spatial_views'][name]['results'][key+'_'+suffix];assert q['state_sha256']==state_hash(state)
                            raw=q['raw'].numpy().astype(float);mapped=q['mapped'];expected=raw.copy()
                            if name=='flip':expected[:,0]=1-expected[:,0]
                            assert np.max(np.abs(expected-mapped))==0;checks+=raw.size+1
                    cv=[]
                    for suffix in ['before','after']:
                        aa=x['spatial_views']['flip']['results'][key+'_'+suffix]['mapped'];bb=x['spatial_views']['dim95']['results'][key+'_'+suffix]['mapped']
                        vals=[]
                        for aa0,bb0 in zip(aa,bb):
                            l1,t1,r1,b1=aa0[0]-aa0[2]/2,aa0[1]-aa0[3]/2,aa0[0]+aa0[2]/2,aa0[1]+aa0[3]/2
                            l2,t2,r2,b2=bb0[0]-bb0[2]/2,bb0[1]-bb0[3]/2,bb0[0]+bb0[2]/2,bb0[1]+bb0[3]/2
                            inter=max(0,min(r1,r2)-max(l1,l2))*max(0,min(b1,b2)-max(t1,t2));union=max(0,r1-l1)*max(0,b1-t1)+max(0,r2-l2)*max(0,b2-t2)-inter
                            vals.append(inter/union if union>0 else 0.)
                        active=[v for f,v in zip(ids,vals) if native[0]<=f<native[1]]
                        q=x['consistency'][key][suffix];close(np.mean(vals),q['full']);close(np.mean(active),q['event']);cv.append(q['event']);checks+=len(vals)+2
                    close(cv[1]-cv[0],x['consistency'][key]['delta_event']);checks+=1
                cases[ds,row['ordinal'],cond]=x
    # Recompute every scalar official dense score, not only the vectorized scorer.
    labels={(ds,sp):read(c1.POOL/ds/f'GT_LABELS_{sp}.json') for ds in DATASETS for sp in ['search','confirm']}
    plans={ds:read(BASE/ds/'PLAN.json') for ds in DATASETS};refs={}
    for r in tr+sr:
        ds,sp,parent,cond,order=r['dataset'],r['split'],r['source_id'],r['condition'],r['order'];x=cases[ds,parent,cond]
        if r['arm']=='consensus_frozen':stream='episodic';which='native'
        elif r['arm'] in ['episodic','online100']:stream=r['arm'];which='after'
        else:stream=r['arm'].rsplit('_',1)[1];which='after'
        key=(ds,parent,cond,order,stream)
        if key not in refs:refs[key]=old.checked(ROOT/x['references'][stream+'_'+order]['path'])
        ref=refs[key];row=plans[ds]['rows'][parent];lab=labels[ds,sp][str(parent)];truth={int(k):v for k,v in lab['truth'].items()};span=lab['span']
        boxes=ref['native']['boxes'] if which=='native' else ref['after'];iv=r.get('predicted_interval',x['native_interval'])
        off=official(boxes.numpy(),row,truth,span,iv,ds)
        for f in ['v','t','s']:close(off[f],r[f]);checks+=1
        if 'before_v' in r:
            before=official(ref['before'].numpy(),row,truth,span,x['native_interval'],ds);close(before['v'],r['before_v']);close(off['v']-before['v'],r['correction_v']);checks+=2
    out=dict(status='pass',private_scalar_checks=checks,unique_inputs=len(cases),official_dense_rows=len(tr)+len(sr),
        independent_coordinate_inverse=True,independent_scalar_box_IoU=True,old_state_hash_binding=True,GT_after_global_seal=True,
        seconds=time.time()-tick,new_model_calls=0)
    write(PUB/'ROOT_AUDIT.json',out);write(PUB/'PUBLIC_AUDIT.json',public(PUB));print('TRANSFORM_ROOT_AUDIT',out,flush=True)

if __name__=='__main__':
    if len(sys.argv)>1:print(public(Path(sys.argv[1])),flush=True)
    else:root()
