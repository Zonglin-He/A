"""Retain every outcome when combining four constructed scales and reference-prior removal."""
import csv
import gzip
from collections import defaultdict
import numpy as np
from common import HERE, OUT, WORK, catalog, digest, row_id, save, read
from compression import code
from scope183 import ordinary
from shared_paths import anonymous
from score_roles import score_boxes, sensor_frame_box, sensor_truth, csv_file
from endpoint_score import write_gz,arrays_to_owners
from score_components import summarize,recording_interval
from common import prior


def score():
    cat=catalog();seal=read(WORK/'COMPACT_SEAL.json')
    assert seal['code']==code() and not seal['labels_used'] and seal['base']==digest(WORK/'SEALED.json')
    assert digest(WORK/'construction_time.csv')==seal['time']
    table=ordinary()
    updates={};maxerr=0.;different=0;checked=0;halves=defaultdict(list)
    labels=read(prior.EXTRA/'labels.json')
    for r in cat['rows']:
        if r['scope'] not in ('ordinary','noun'):continue
        k=row_id(r);p=WORK/'compact'/(k+'.npz')
        assert digest(p)==seal['files'][k]
        with np.load(p) as z:
            dx=float(z['four_replay_error']);maxerr=max(maxerr,dx);different+=int(dx>1e-7);checked+=1
            if r['scope']=='ordinary':
                target=r['target'];ix=r['ordinal']
                row=table[target][ix]
                truth=sensor_truth(target,row['metapath'],row['caption'],row['suffix'])
                b=np.stack([sensor_frame_box(v,target,row['suffix']) for v in z['subject']])
                v,_=score_boxes(b,truth)
                updates[r['source'],anonymous((target,ix))]=v
            else:
                e=labels[r['uid']]
                with np.load(r['prior']) as prev:
                    own=arrays_to_owners([z['subject'],z['reference']],np.array(e['truth']),prev['seeds'])
                for rule,(s,ref) in own.items():
                    halves[r['source'],e['pair'],rule].append(np.stack([s==e['subject'],ref==e['reference']]))
    names=['full','four_scales','no_reference_proposals','direct_four_scales','four_without_reference_prior']
    cases=[];rec=defaultdict(list);maps=defaultdict(list)
    with gzip.open(OUT/'ordinary_rows.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            loc=(r['source'],r['case']);v=[float(r[n]) for n in names[:3]]
            sub=updates.get(loc,[v[0],v[0]])
            assert (loc in updates)==bool(int(r['active']))
            vals=np.r_[v,sub];key=(r['source'],r['target'])
            rec[key].append(r['recording']);maps[key].append(vals)
            cases.append({**{n:r[n] for n in ('source','target','recording','case','active')},'direct_four_scales':vals[-2],'four_without_reference_prior':vals[-1]})
    metrics=[];pairs=[];values=[]
    def emit(tag,n,m,c,ci=None):
        for i,arm in enumerate(names):
            metrics.append(dict(direction=tag,n=n,arm=arm,acc25=m[0,i],acc50=m[1,i],miou=m[2,i],
                delta50=m[1,i]-m[1,0],rescue25=c[0,i],harm25=c[1,i],rescue50=c[2,i],harm50=c[3,i],
                ci50_low=ci[0,i] if ci is not None else '',ci50_high=ci[1,i] if ci is not None else ''))
    for k,x in maps.items():
        a=np.array(x);n,c=summarize(a,np.zeros_like(a));values.append(n);pairs.append(c);emit('_to_'.join(k),len(a),n,c)
    ci=recording_interval({k:(np.array(x),np.array(rec[k])) for k,x in maps.items()})
    emit('macro',len(cases),np.mean(values,0),np.sum(pairs,0),ci)
    csv_file(OUT/'compression_metrics.csv',metrics);write_gz(OUT/'compression_rows.csv.gz',cases)
    t=list(csv.DictReader((WORK/'construction_time.csv').open()))
    def dist(field):
        v=np.array([float(r[field]) for r in t])
        return dict(mean=float(v.mean()),median=float(np.median(v)),sum=float(v.sum()))
    check={}
    for rule in ('unique_iou','seed_ownership'):
        a=np.stack([np.logical_and(*v) for k,v in halves.items() if k[2]==rule])
        assert len(a)==64
        check[rule]={m:np.all(a,axis=1)[:,i].sum().item() for i,m in enumerate(names[-2:])}
    save(OUT/'compression_check.json',dict(checked=checked,four_materialization_disagreements=different,
         max_endpoint_error=maxerr,distinct_frame_source=len(t),construction_cost={f:dist(f) for f in t[0] if f!='full_replay'},
         full_reference_replay_error=max(float(v['full_replay']) for v in t),noun_all_four=check,
         paired_scope=len(cases),prediction_seal=digest(WORK/'COMPACT_SEAL.json'),used_for_primary=False,FINAL_access=False))


if __name__=='__main__':score()
