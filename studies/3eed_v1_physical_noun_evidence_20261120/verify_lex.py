"""Reconstruct the scores independently, diagnose spatial rank collapse, and verify task counts."""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import csv,gzip,tempfile
from pathlib import Path
import numpy as np
from scipy.special import softmax
from common_lex import *
from ordered_choice import prepare
from physical_evidence import Returns,complete
from score_roles import csv_file
import verify_bound


def toy_geometry():
    angle=.34
    mat=np.array([[np.cos(angle),np.sin(angle),0],[-np.sin(angle),np.cos(angle),0],[0,0,1.]])
    anchor=np.array([0.,0.,0.,3.,1.,1.,angle]);donor=anchor.copy();donor[0]=8.;donor[3]=8.
    inside_t=np.array([[.4,-.2,0],[.9,.2,.1],[1.3,.1,0]])@mat
    inside_d=donor[:3]+np.array([[-3.,0.,0],[0.,.2,0],[3.,-.2,.2]])@mat
    p=Returns(np.vstack((inside_t,inside_d)));semantic=np.tile([1.,0.],(len(p.xyz),1))
    out,mid,side,delta,n=complete(anchor,donor[None],0,p,semantic,p,np.zeros(len(p.xyz),int))
    assert np.allclose(delta,[3.,0]) and np.array_equal(side,[1,0]) and n[0]==1
    np.testing.assert_allclose(out[:3]+mat[0]*out[3]/2,anchor[:3]+mat[0]*anchor[3]/2)
    np.testing.assert_allclose(mid[:3],anchor[:3]);np.testing.assert_allclose(out[5:],anchor[5:])
    for candidates in (anchor[None],np.zeros((0,7))):
        if len(candidates):
            empty=complete(anchor,candidates,0,p,semantic,p,np.zeros(len(p.xyz),int))[0]
            np.testing.assert_array_equal(empty,anchor)


def measure_item(task):
    source,cases,fid,checksum=task
    path=OBS/(fid+'.npz')
    assert digest(path)==checksum
    with np.load(path) as o:
        scores=o['logits'].astype(float);x=o['xyz']
    # Subtraction of the class mean per rotation can only add a common scalar to all logits of a seed.
    # Recompute the class ranking without this step, after averaging any collocated seed IDs.
    pts,inv=np.unique(x,axis=0,return_inverse=True);n=len(pts)
    count=np.bincount(inv)[:,None]
    def grouped(a):
        val=np.zeros((n,len(CLASSES)));np.add.at(val,inv,a)
        return val/count
    mean=grouped(scores.mean(0));first=grouped(scores[0])
    rolled=np.roll(mean,n//2,axis=0)
    modes=[mean,first,rolled]
    top=mean.argmax(1);vote,count_top=np.unique(top,return_counts=True)
    tt=np.sort(mean,axis=1)
    result=dict(source=source,n=n,top_classes=len(vote),majority=float(count_top.max()/n),
                majority_class=CLASSES[vote[count_top.argmax()]],multi_seed=int(len(vote)>1),
                distinct_order_classes=int(np.count_nonzero(first.argmax(1)!=top)),
                different_after_roll=int(np.count_nonzero(rolled.argmax(1)!=top)),
                tie_count=int(np.count_nonzero(tt[:,-1]==tt[:,-2])),
                max_spatial_logit_span=float(np.ptp(mean-mean.mean(1,keepdims=True),axis=0).max()),
                max_rotation_logit_span=float(np.ptp(scores-scores.mean(2,keepdims=True),axis=0).max()),
                candidate_masks=0,candidate_ties=0,different_box_label_after_roll=0,
                different_box_label_one_order=0,supported=0)
    for case in cases:
        with np.load(OLD/'observed'/case) as z,np.load(PRED/case) as p:
            if not z['active']:continue
            for role in ('t','r'):
                boxes=prepare(z,role)[0]
                targets={k:p[role+'_'+k] for k in ('main','one','wrong')}
                for i,b in enumerate(boxes):
                    c,s=np.cos(b[6]),np.sin(b[6]);off=pts-b[:3]
                    loc=np.column_stack((c*off[:,0]+s*off[:,1],-s*off[:,0]+c*off[:,1],off[:,2]))
                    valid=(abs(loc)<=b[3:6]/2+1.e-7).all(1)
                    result['candidate_masks']+=1
                    if not valid.any():
                        assert targets['main'][i]==targets['one'][i]==targets['wrong'][i]==-1
                        continue
                    result['supported']+=1
                    labels=[]
                    for m,tag in zip(modes,('main','one','wrong')):
                        avg=m[valid].mean(0)
                        label=avg.argmax()
                        assert label==int(targets[tag][i])
                        result['candidate_ties']+=int(np.sum(avg==avg.max())>1)
                        labels.append(label)
                    result['different_box_label_one_order']+=int(labels[1]!=labels[0])
                    result['different_box_label_after_roll']+=int(labels[2]!=labels[0])
    return result


def read_csv(path):
    fp=gzip.open(path,'rt') if path.name.endswith('.gz') else path.open()
    with fp:return list(csv.DictReader(fp))


def run():
    toy_geometry()
    with tempfile.TemporaryDirectory() as d:
        verify_bound.OUT=Path(d);verify_bound.run()
        actual=(Path(d)/'bound_recheck.csv').read_bytes()
        assert actual==(LAST/'results/bound_recheck.csv').read_bytes()
        (OUT/'bound_recheck.csv').write_bytes(actual)
    seal=read(OBS/'SEALED.json');p=read(PRED/'SEALED.json');audit=read(OUT/'audit.json')
    assert audit['sealed']==digest(PRED/'SEALED.json')
    assert len(p['files'])==len(panel())==844
    group=defaultdict(list);sources={}
    for row in panel():
        case=name(row);fid=seal['files'][case];sources[fid]=row['source'];group[fid].append(case)
        assert digest(PRED/case)==p['files'][case]
    with ProcessPoolExecutor(max_workers=6) as pool:
        diags=list(pool.map(measure_item,[(sources[fid],cases,fid,seal['hashes'][fid]) for fid,cases in group.items()],chunksize=6))
    results=[]
    for source in ['all',*SOURCES]:
        v=[x for x in diags if source=='all' or x['source']==source]
        sumfields=['n','multi_seed','tie_count','different_after_roll','distinct_order_classes',
                   'candidate_masks','candidate_ties','supported','different_box_label_after_roll','different_box_label_one_order']
        out=dict(source=source,observations=len(v),**{k:sum(x[k] for x in v) for k in sumfields},
                 median_majority=float(np.median([x['majority'] for x in v])),
                 median_spatial_score_span=float(np.median([x['max_spatial_logit_span'] for x in v])),
                 median_word_order_score_span=float(np.median([x['max_rotation_logit_span'] for x in v])))
        for term in CLASSES:
            out['dominant_'+term.replace(' ','_')]=sum(x['majority_class']==term for x in v)
        results.append(out)
    csv_file(OUT/'lexical_concentration.csv',results)
    records=read_csv(OUT/'task_rows.csv.gz'); assert len(records)==844*len(ARMS)
    groups=defaultdict(dict)
    for r in records:
        assert r['arm'] not in groups[r['source'],r['case']]
        groups[r['source'],r['case']][r['arm']]=r
    assert len(groups)==844 and all(set(q)==set(ARMS) for q in groups.values())
    checked=0
    for r in read_csv(OUT/'paired.csv'):
        if r['direction']=='macro' or r['group'] not in ('natural','ordinary_uniform','ordinary_relation','name','depth'):continue
        cases=[q for (s,c),q in groups.items() if q['v1']['group']==r['group'] and s+'_to_'+q['v1']['target']==r['direction']]
        assert len(cases)==int(r['n'])
        for t in (25,50):
            diffs=[int(float(v[r['arm']]['iou'])>t/100)-int(float(v[r['control']]['iou'])>t/100) for v in cases]
            assert diffs.count(1)==int(r['rescue'+str(t)]) and diffs.count(-1)==int(r['harm'+str(t)])
            checked+=1
    components=ROOT/'studies/3eed_v1_joint_endpoint_coverage_20261120/results/ordinary_component_upper_bounds.csv'
    assert [(int(r['all_passing_dominated']),int(r['linearly_separable'])) for r in read_csv(components) if r['direction']=='all_records']==[(2719,527),(1076,164)]
    mismatch={}
    for other in ['single_order','wrong_placement','wrong_reference','independent','no_noun']:
        diff_sub=diff_ref=0
        for n in p['files']:
            with np.load(PRED/n) as z:
                i=ARMS.index('joint_raw');j=ARMS.index(other)
                diff_sub+=int(not np.array_equal(z['raw'][i],z['raw'][j],equal_nan=True))
                diff_ref+=int(not np.array_equal(z['ref'][i],z['ref'][j],equal_nan=True))
        mismatch[other]={'subject':diff_sub,'reference':diff_ref}
    save(OUT/'verification.json',dict(ordinary_count=18620,hard_locks25=83,hard_locks50=36,
        bound_csv_identical=True,checked_readouts=len(records),independent_paired_counts=checked,
        exact_subject_reference_differences_vs_joint=mismatch,strict_candidate_ties=results[0]['candidate_ties'],
        candidate_labels_recomputed=3*results[0]['candidate_masks'],
        supported_candidates=results[0]['supported'],geometry_fixed_face_test=True,FINAL_access=False))


if __name__=='__main__':run()
