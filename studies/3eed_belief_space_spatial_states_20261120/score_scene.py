"""Join annotations to sealed hypotheses; retain errors, abstentions and individual direction metrics."""
import argparse
import csv
from collections import Counter,defaultdict
import gzip
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from io_scene import ROOT,PRIVATE,OUT,SPEC,PLATFORMS,digest,read,write
from belief_scene import ARMS
from spatial_simulator import NAMES,outcomes
from audit_math import verify
sys.path.insert(0,str(ROOT/'research/round6/source_validation_scoring'))
from batch_yaw_iou import many_iou

def owners(boxes, all_boxes):
    hit=np.stack([many_iou(boxes,b)>.25 for b in all_boxes],axis=1)
    return np.where(hit.sum(1)==1,hit.argmax(1),-1)

def correct(probs,target):
    return np.where(np.abs(probs-.5)<=1.e-14,.5,((probs>.5)==target).astype(float))

def labels(boxes,s,r,cam,names):
    # Independent basis construction and analytic squared-distance difference.
    d=boxes[s,:2]-boxes[r,:2]
    theta=np.arctan2(cam[0,1],cam[0,0])
    axis=lambda angle:np.array([np.cos(angle),np.sin(angle)])
    x=d@axis(theta); f=d@axis(theta+np.pi/2)
    values=[x,f,d@axis(theta-np.pi/4),4*x+4]
    result=np.array([values[i]>0 for i in range(len(names))])
    # Cross check the gold by forward-transforming literal GT boxes.
    checked=outcomes(boxes[[s]],boxes[[r]],cam)[:len(names)]
    np.testing.assert_array_equal(result,[e.item() for e in checked])
    return result

def risk_at(hits,conf,cover):
    idx=np.argsort(-conf,kind='stable');levels,starts=np.unique(-conf[idx],return_index=True)
    n=len(idx); count=np.diff(np.r_[starts,n]);errors=np.add.reduceat(1-hits[idx],starts)
    remaining=cover*n-np.r_[0,np.cumsum(count)[:-1]]
    take=np.clip(remaining,0,count)/count
    return float(errors@take/(cover*n))

def calibration(block):
    p=block.prob.to_numpy();y=block.truth.to_numpy()
    bins=np.minimum(9,(10*p).astype(int))
    ece=sum(abs((p[ids]-y[ids]).sum()) for k in range(10) if (ids:=bins==k).any())/len(p)
    return dict(acc=block.hit.mean(),brier=np.mean((p-y)**2),ece=ece,neutral=np.mean(abs(p-.5)<1.e-14),
                no_mass=block.no_mass.mean(),**{'risk'+str(k):risk_at(block.hit.to_numpy(),block.confidence.to_numpy(),k/100) for k in range(10,101,10)})

def summarize(rows,cohorts,stage):
    records=[]
    for scope,b in cohorts:
        if b.empty:continue
        for method,g in b.groupby('arm',sort=False):
            dirs=[]
            for name,c in g.groupby('direction'):
                a=calibration(c);dirs.append(a)
                records.append(dict(scope=scope,direction=name,n=len(c),arm=method,**a))
            avg={k:float(np.mean([m[k] for m in dirs])) for k in dirs[0]}
            records.append(dict(scope=scope,direction='macro',n=len(g),arm=method,**avg))
    pd.DataFrame(records).to_csv(OUT/(stage+'_metrics.csv'),index=False,float_format='%.12g')
    return {r['arm']:r for r in records if r['scope']=='all' and r['direction']=='macro'}

def bootstrap(frame, n=2000):
    """Paired recording draws with shared multiplicities between two sources, ties pooled proportionally."""
    rng=np.random.default_rng(1947)
    out={a:np.zeros((n,3)) for a in ARMS}
    directions=0
    for tgt,t in frame.groupby('target'):
        recs=sorted(set(t.recording)); codes={v:i for i,v in enumerate(recs)}
        draws=rng.multinomial(len(recs),np.ones(len(recs))/len(recs),size=n)
        for _,d in t.groupby('direction'):
            directions+=1
            for a in ARMS:
                v=d[d.arm==a]; idx=np.array([codes[r] for r in v.recording]); w=draws[:,idx]
                total=w.sum(1); hits=v.hit.to_numpy();conf=v.confidence.to_numpy()
                assert np.all(total>0)
                brier=(v.prob.to_numpy()-v.truth.to_numpy())**2
                order=np.argsort(-conf,kind='stable')
                starts=np.flatnonzero(np.r_[True,np.diff(conf[order])!=0])
                ww=w[:,order]
                mass=np.add.reduceat(ww,starts,axis=1)
                err=np.add.reduceat(ww*(1-hits[order]),starts,axis=1)
                available=total[:,None]/2-(np.cumsum(mass,axis=1)-mass)
                take=np.divide(np.clip(available,0,mass),mass,out=np.zeros_like(mass,dtype=float),where=mass>0)
                out[a]+=np.stack([w@hits/total,w@brier/total,(take*err).sum(1)/(total/2)],axis=1)
    result={}
    for a in ARMS:
        if a=='tree_joint':continue
        delta=(out['tree_joint']-out[a])/directions
        result[a]={name:list(np.quantile(delta[:,i],[.025,.975])) for i,name in enumerate(('acc','brier','risk50'))}
    return result

def run(expanded):
    stage='intervention' if expanded else 'static'
    if expanded:assert read(OUT/'static_decision.json')['advance']
    seal=read(OUT/'prediction_seal.json');panel=read(OUT/'panel_seal.json')
    assert digest(SPEC)==seal['protocol']==panel['protocol'] and digest(OUT/'panel_seal.json')==seal['panel']
    assert digest(PRIVATE/'questions.json')==panel['questions'] and digest(PRIVATE/'answer_key.json')==panel['truth']
    rows=read(PRIVATE/'questions.json');gold=read(PRIVATE/'answer_key.json')
    assert set(gold)=={r['uid'] for r in rows}
    end,qa=[],[]
    tasks=NAMES if expanded else NAMES[:2]
    for q in rows:
        uid=q['uid'];target=q['frame']['target']; e=gold[uid]; s,r=e['s'],e['r'];gb=np.array(e['boxes'])
        for src in PLATFORMS:
            if src==target:continue
            pred=PRIVATE/'predictions'/src/(uid+'.npz')
            assert digest(pred)==seal['files'][src+'/'+uid]
            with np.load(pred) as d:
                ys=labels(gb,s,r,d['camera'],tasks)
                os=owners(d['sb'],gb);ore=owners(d['rb'],gb)
                dual=(os[:,None]==s)&(ore==r);associated=(os[:,None]>=0)&(ore>=0);duplicate=associated&(os[:,None]==ore)
                support=float(np.median(d['size'])) if len(os) else 0.
                visible=float(np.median(d['visible_count']/d['size'])) if len(os) else 0.
                # Minimum distance to another annotated box measures adjacency without model filtering.
                dist=np.linalg.norm(gb[:,:2]-gb[s,:2],axis=1);dist[s]=np.inf
                info=dict(frame=uid,recording=q['recording'],direction=src+'_to_'+target,target=target,
                          cohort=q['cohort'],support=support,visible_fraction=visible,adjacent=int(dist.min()<=5.))
                for a,method in enumerate(ARMS):
                    p=d['p'][a];no=int(d['no_mass'][a])
                    end.append(dict(**info,arm=method,dual_mass=p[dual].sum(),peak_dual=int(bool(p.size and not no and dual.flat[int(p.argmax())])),
                             association_mass=p[associated].sum(),duplicate_mass=p[duplicate].sum(),impossible_mass=d['invalid'][a],
                             both_in_inventory=int(dual.any()),no_mass=no))
                    for idx,name in enumerate(tasks):
                        if expanded and idx<2:continue
                        prob=float(d['answers'][a,idx]);y=int(ys[idx])
                        qa.append(dict(**info,arm=method,task=name,truth=y,prob=prob,
                                   hit=float(correct(np.array([prob]),y)[0]),confidence=float(d['confidence'][a,idx]),
                                   no_mass=no,**(dict(unchanged=int(ys[0]) if idx==2 else 0,
                                   stable_rule=int(ys[0])) if expanded else {})))
    df=pd.DataFrame(qa);ep=pd.DataFrame(end)
    assert len(df)==len(rows)*2*2*len(ARMS)
    unique=ep[ep.arm=='tree_joint']
    med_s=unique.support.median();med_v=unique.visible_fraction.median()
    scopes=[('all',df),('unique_nouns',df[df.cohort=='unique']),('height_designated',df[df.cohort=='height']),
            ('sparse',df[df.support<=med_s]),('low_visible_fraction',df[df.visible_fraction<=med_v]),
            ('adjacent_annotated',df[df.adjacent==1])]
    scopes += [(name,g) for name,g in df.groupby('task')]
    if expanded:scopes.append(('changed_vs_no_op',df[df.truth!=df.unchanged]))
    summary=summarize(df,scopes,stage)
    df.to_csv(OUT/(stage+'_all_rows.csv.gz'),index=False,compression={'method':'gzip','mtime':0},float_format='%.16g')
    ci=bootstrap(df)
    if not expanded:
        ep.to_csv(OUT/'endpoint_all_rows.csv.gz',index=False,compression={'method':'gzip','mtime':0},float_format='%.16g')
        denom=['dual_mass','peak_dual','association_mass','duplicate_mass','impossible_mass','both_in_inventory','no_mass']
        by=ep.groupby(['direction','arm'])[denom].mean()
        cc=by.groupby('arm').mean().reset_index();cc['direction']='macro'
        pd.concat([by.reset_index(),cc]).to_csv(OUT/'endpoint_metrics.csv',index=False,float_format='%.12g')
    p=summary['tree_joint']; controls=('map_pair','map_world_pair','masked_product')
    better_acc=all(p['acc']>summary[a]['acc']+1.e-12 for a in controls)
    better_risk=all(p['risk50']<summary[a]['risk50']-1.e-12 for a in controls)
    advance=(better_acc or better_risk) and p['brier']<=summary['masked_product']['brier']+1.e-12
    # Compare changed decisions including any half-credit ties, paired by physical frame, source and query.
    hits=df.pivot(index=['frame','direction','task'],columns='arm',values='hit')
    pairs={}
    for a in ARMS:
        diff=hits['tree_joint']-hits[a]
        pairs[a]=dict(better=int((diff>0).sum()),worse=int((diff<0).sum()),equal=int((diff==0).sum()))
    groups={t:int(g.recording.nunique()) for t,g in df.groupby('target')}
    result=dict(physical_frames=len(rows),source_frames=int(len(unique)),questions_per_arm=int(len(hits)),
                groups=groups,math=verify(),gt_oracle_accuracy=1.,pairing=pairs,
                observed_strata_medians=dict(point_count=float(med_s),visible_fraction=float(med_v)),
                delta_tree_minus_control_ci95=ci,paired_recording_bootstrap=2000)
    if not expanded:result.update(advance=bool(advance),primary_accuracy_higher=bool(better_acc),primary_risk_lower=bool(better_risk))
    if expanded:
        direct=df[df.arm=='tree_joint']
        for k in ('unchanged','stable_rule'):
            result[k+'_accuracy']=float(direct.assign(ok=direct[k]==direct.truth).groupby('direction').ok.mean().mean())
        result['changed_questions']=int((direct.truth!=direct.unchanged).sum())
    write(OUT/(stage+'_decision.json'),result)
    print(stage,'primary (acc / risk50 / Brier)',{k:tuple(round(v[m],5) for m in ('acc','risk50','brier')) for k,v in summary.items()},'advance',advance,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--expanded',action='store_true')
    run(parser.parse_args().expanded)
