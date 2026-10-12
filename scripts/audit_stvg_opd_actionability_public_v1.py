"""Independent anonymous parent ROC/AP audit and private official-frame GT labels."""
import collections,gzip,json,math,sys
from pathlib import Path
import numpy as np
EPS=1e-12;BOOT=10000;SEED=20261012
SIGNALS=['holdout_agreement_gain','detector_score','before_training_agreement','training_agreement_gain','teacher_geometric_consistency','negative_output_displacement']

def near(a,b):
    if isinstance(a,dict):assert set(a)==set(b),(set(a),set(b));return sum(near(v,b[k]) for k,v in a.items())
    if isinstance(a,list):assert len(a)==len(b);return sum(near(x,y) for x,y in zip(a,b))
    if a is None or isinstance(a,(str,bool,int)):assert a==b,(a,b)
    else:assert abs(a-b)<2e-10*(1+abs(a)),(a,b)
    return 1
def private_labels(z,j,raw,truth,span,independent=False):
    if independent:
        from vg_tta.tastvg_paper48_metrics_v1 import xyxy as convert
        from vg_tta.tastvg_oracle_event5_v1 import box_iou as overlap
    else:
        from scripts.stvg_opd_evidence_actionability_audit_v1 import xyxy as convert,iou as overlap
    ids=z['frame_ids'];clip=z['dataset']=='hc2';W=raw['input']['width'];H=raw['input']['height']
    b=convert(z['before'],W,H);a=convert(z['after'][j],W,H)
    if clip:b=np.maximum(b,0);a=np.maximum(a,0)
    own=set(map(int,z['visited'][j]));union=set(map(int,z['visited'][0]+z['visited'][1]))
    available=[k for k,f in enumerate(ids) if f in truth and span[0]<=f<span[1]]
    masks={'own_U':[k for k in available if k not in own],'common_U':[k for k in available if k not in union]}
    out={'GT_event_real_frames':len(available)}
    for key,positions in masks.items():
        delta=[float(overlap(a[k],truth[ids[k]])-overlap(b[k],truth[ids[k]])) for k in positions]
        out[key+'_frames']=len(positions);out[key+'_delta']=math.fsum(delta)/len(delta) if delta else None
    quality=[]
    for anchor in z['anchors'][j]:
        p=int(anchor['position']);frame=ids[p]
        if frame not in truth or not span[0]<=frame<span[1]:continue
        box=convert([anchor['box']],W,H)[0]
        if clip:box=np.maximum(box,0)
        quality.append(float(overlap(box,truth[frame])))
    out.update(teacher_quality=math.fsum(quality)/len(quality) if quality else None,teacher_labeled_anchors=len(quality),teacher_unknown_anchors=len(z['anchors'][j])-len(quality),
        teacher_fully_labeled_correct=bool(len(quality)==len(z['anchors'][j]) and all(v>=.5 for v in quality)) if quality else None)
    return out

def boot_weights(n):
    idx=np.random.default_rng(SEED).integers(n,size=(BOOT,n));w=np.zeros((BOOT,n),np.int16)
    for k in range(n):w[:,k]=(idx==k).sum(1)
    return w
def distribution(s,y,w):
    """Independent pair-comparison ROC and precision-at-distinct-score AP."""
    s=np.asarray(s,float);y=np.asarray(y,bool);w=np.atleast_2d(w).astype(float)
    pos=np.flatnonzero(y);neg=np.flatnonzero(~y);P=w[:,pos].sum(1);N=w[:,neg].sum(1);valid=(P>0)&(N>0)
    cmp=(s[pos,None]>s[None,neg]).astype(float)+.5*(s[pos,None]==s[None,neg]);roc=np.full(len(w),np.nan);ap=roc.copy()
    roc[valid]=np.sum((w[:,pos]@cmp)*w[:,neg],axis=1)[valid]/(P[valid]*N[valid])
    numerator=np.zeros(len(w))
    for threshold in sorted(set(s),reverse=True):
        at=(s==threshold)&y;above=s>=threshold;tp=w[:,above&y].sum(1);allp=w[:,above].sum(1)
        precision=np.divide(tp,allp,out=np.zeros(len(w)),where=allp>0)
        numerator+=w[:,at].sum(1)*precision
    ap[valid]=numerator[valid]/P[valid];return roc,ap
def ci(x):
    valid=np.asarray(x)[np.isfinite(x)];return dict(ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,valid_resamples=len(valid),undefined_resamples=len(x)-len(valid))
def rank(records,score,outcome):
    eligible=[r for r in records if r.get(score) is not None and r.get(outcome) is not None and abs(r[outcome])>EPS];n=len(eligible);p=sum(r[outcome]>EPS for r in eligible)
    z=dict(total_parents=len(records),eligible_parents=n,positive_parents=p,negative_parents=n-p,positive_prevalence=p/n if n else None,signal_undefined_parents=sum(r.get(score) is None for r in records),neutral_or_undefined_outcome_parents=sum(r.get(outcome) is None or abs(r[outcome])<=EPS for r in records))
    if p==0 or p==n:return dict(**z,AUROC=None,AP=None,AUROC_bootstrap=None,AP_bootstrap=None)
    s=[r[score] for r in eligible];y=[r[outcome]>EPS for r in eligible];a,b=distribution(s,y,np.ones((1,n)));aa,bb=distribution(s,y,boot_weights(n))
    return dict(**z,AUROC=float(a[0]),AP=float(b[0]),AUROC_bootstrap=ci(aa),AP_bootstrap=ci(bb))
def matched(records,reference,outcome):
    rr=[r for r in records if r.get('holdout_agreement_gain') is not None and r.get(reference) is not None and r.get(outcome) is not None and abs(r[outcome])>EPS];n=len(rr);p=sum(r[outcome]>EPS for r in rr);z=dict(eligible_parents=n,positive_parents=p,negative_parents=n-p)
    if p==0 or p==n:return dict(**z,AUROC_difference=None,ci95=None,valid_resamples=0,undefined_resamples=BOOT)
    y=[r[outcome]>EPS for r in rr];s=[r['holdout_agreement_gain'] for r in rr];t=[r[reference] for r in rr];w=boot_weights(n);one=np.ones((1,n))
    a=distribution(s,y,w)[0];b=distribution(t,y,w)[0];point=distribution(s,y,one)[0][0]-distribution(t,y,one)[0][0]
    return dict(**z,AUROC_difference=float(point),**ci(a-b))
def parents(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['parent_id']].append(r)
    out=[]
    for parent,group in groups.items():
        d=dict(parent_id=parent,contributing_orders=len(group))
        for field in group[0]:
            if field in ['public_cell','parent_id','dataset','order','previous_P6','scope']:continue
            vals=[r[field] for r in group if r[field] is not None]
            d[field]=math.fsum(vals)/len(vals) if vals and all(isinstance(v,(int,float,bool)) for v in vals) else None
        out.append(d)
    return out
def matrix(rr,outcome,strict):
    counts=collections.Counter();unknown=0
    for r in rr:
        q=r['teacher_quality']
        if q is None:unknown+=1;continue
        positive=(r['teacher_fully_labeled_correct']==1) if strict else q>=.5
        a=r[outcome];effect='undefined' if a is None else 'positive' if a>EPS else 'negative' if a< -EPS else 'neutral'
        counts[('geometrically_correct' if positive else 'not_geometrically_correct')+'_'+effect]+=1
    return dict(total_parents=len(rr),teacher_unknown_parents=unknown,teacher_labeled_parents=len(rr)-unknown,cells=dict(counts))
def independent_summary(rr,summary):
    checks=0
    for ds,data in summary.items():
        for scope,subsets in data.items():
            for subset,orders in subsets.items():
                for order,block in orders.items():
                    selected=[r for r in rr if r['dataset']==ds and r['scope']==scope and (subset=='all128' or r['previous_P6']==(subset=='previous_P6_32')) and (order=='both' or r['order']==order)]
                    pp=parents(selected);assert len(pp)==block['parents']=={'all128':128,'previous_P6_32':32,'added96':96}[subset]
                    assert len(selected)==block['update_cells'];checks+=2
                    for outcome,metrics in block['rank'].items():
                        for score,expected in metrics.items():checks+=near(rank(pp,score,outcome),expected)
                        for ref,expected in block['matched'][outcome].items():checks+=near(matched(pp,ref,outcome),expected)
                        v=[r[outcome] for r in pp if r[outcome] is not None]
                        expected=dict(defined_parents=len(v),undefined_parents=len(pp)-len(v),positive=sum(x>EPS for x in v),negative=sum(x< -EPS for x in v),neutral=sum(abs(x)<=EPS for x in v),harm_5pp=sum(x<-.05 for x in v),harm_20pp=sum(x<-.2 for x in v))
                        checks+=near(expected,block['outcomes'][outcome]);checks+=near(matrix(pp,outcome,False),block['cross'][outcome]['mean_quality']);checks+=near(matrix(pp,outcome,True),block['cross'][outcome]['fully_labeled_all_anchors'])
                    for field,expected in block['mean'].items():
                        values=[r[field] for r in pp if r.get(field) is not None]
                        if not values:checks+=near(dict(mean=None,ci95=None,parents=0,undefined_parents=len(pp)),expected);continue
                        x=np.array(values);idx=np.random.default_rng(SEED).integers(len(x),size=(BOOT,len(x)));b=x[idx].sum(1)/len(x)
                        checks+=near(dict(mean=math.fsum(x)/len(x),ci95=np.quantile(b,[.025,.975]).tolist(),parents=len(x),undefined_parents=len(pp)-len(x)),expected)
    return checks
def run(root):
    p=Path(root)/'results/stvg_opd_evidence_actionability_audit/2026-10-12'
    with gzip.open(p/'ROWS.jsonl.gz','rt') as f:rr=[json.loads(v) for v in f]
    path=p/'SUMMARY.json'
    if path.exists():summary=json.loads(path.read_text())
    else:
        with gzip.open(p/'SUMMARY.json.gz','rt') as f:summary=json.load(f)
    assert len(rr)==1024 and len({r['parent_id'] for r in rr})==256
    for r in rr:
        assert (r['holdout_agreement_gain'] is None)==(r['holdout_anchors']==0)
        if r['holdout_agreement_gain'] is not None:near(r['holdout_agreement_gain'],r['holdout_after_agreement']-r['holdout_before_agreement'])
        assert r['teacher_labeled_anchors']+r['teacher_unknown_anchors']==r['training_anchors']
        assert r['common_U_frames']<=r['own_U_frames']<=r['GT_event_real_frames']
    checks=independent_summary(rr,summary)
    return dict(status='pass',scope='all1024 anonymous positive-negative-neutral-undefined rows and all36 target/scope/subset/order parent ROC/AP, matched comparisons, cross tables and bootstrap',saved_updates=1024,parents=256,scalar_checks=checks,CPU_only=True,GT_read=False)
if __name__=='__main__':print(json.dumps(run(sys.argv[1]),allow_nan=False))
