"""Only called after prediction seal. Includes every chosen row and both endpoint ownership definitions."""
import argparse
from hashlib import sha256
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import csv
import gzip
import numpy as np

from paths_witness import WORK,OBS,PRED,OUT,HERE,ROOT,LAST,SPEC,SOURCES,tag,name,read,save,digest
from predict_witness import ARMS
from registry_swap import PANELS,ordinary
from score_swap import best
from endpoint_score import arrays_to_owners,write_gz
from score_roles import anon,csv_file,inventory,normalize,state,score_boxes,sensor_frame_box,sensor_truth
from diagnose_roles import historical_endpoints,historical_records,COHORTS
from ordered_choice import prepare
from tables_witness import tables

def metadata(row,entry,panels,hist,cache):
    target=row['frame']['target'];g=entry['group']
    if 'pair' in entry:
        from scope183 import VIEWS
        b=np.asarray(entry['truth'])
        view=VIEWS.index(row['frame']['waymo_view']) if target=='waymo' else 0
        return b,entry['subject'],entry['reference'],entry['frame'],entry['recording'],view,None
    ix=entry['case']
    item=panels[g][ix] if g in COHORTS else None
    data=item['row'] if item is not None else ordinary_cached()[target][ix]
    view=int(data['suffix'] or 0)
    b,text=inventory(data['metapath'],target,view,data['object_index'])
    assert normalize(row['frame']['text'])==text
    key=(g,ix)
    if item is not None:
        if key not in cache:cache[key]=historical_endpoints(COHORTS[g],item,b,hist)[0]
        s,r=cache[key]
    else:s,r=0,None
    gt=sensor_truth(target,str(Path(data['metapath']).resolve()),data['caption'],data['suffix'])
    return b,s,r,tag(str(Path(data['pointpath']).resolve())),tag(str(Path(data['metapath']).resolve().parent.parent)),view,gt

ordinary_cached=lru_cache(maxsize=1)(ordinary)

def check_seals():
    emitted,obs=read(PRED/'SEALED.json'),read(OBS/'SEALED.json')
    assert emitted['n']==obs['n']==len(emitted['files']) and emitted['files'].keys()==obs['files'].keys()
    assert emitted['obs']==digest(OBS/'SEALED.json') and emitted['protocol']==obs['pin']['protocol']==digest(SPEC)
    assert not any(x['labels_used'] or x['FINAL_access'] for x in (emitted,obs))
    for code in (emitted['code'],obs['pin']['code']):
        assert all(digest(HERE/f)==h for f,h in code.items())
    panel=read(OUT/'panel.json')
    assert panel['observations']==obs['pin']['roster']==digest(WORK/'roster.json') and panel['labels']==digest(WORK/'labels.json')
    return emitted,obs

def run(repo):
    emitted,obs=check_seals()
    labels=read(WORK/'labels.json');observations=read(WORK/'roster.json')
    panels={g:read(PANELS[g]) for g in COHORTS};history=historical_records(repo);cache={}
    with gzip.open(LAST/'results/all_rows.csv.gz','rt') as stream:
        previous={(r['source'],r['case']):float(r['iou']) for r in csv.DictReader(stream) if r['arm']=='v1'}
    data=[];endpoints={};err=0.;replayed=0;directions=set();pool_digest={}
    for row in observations:
        uid=row['uid'];entry=labels[uid];filename=name(row);g=entry['group'];target=row['frame']['target'];src=row['source']
        assert digest(PRED/filename)==emitted['files'][filename] and digest(OBS/filename)==obs['files'][filename]
        truth,s,r,fr,rec,view,gt_sensor=metadata(row,entry,panels,history,cache)
        target_id=entry['subject'] if 'pair' in entry else 0
        with np.load(OBS/filename) as f,np.load(PRED/filename) as p:
            raw,refs,boxes=p['raw'],p['ref'],p['boxes']
            both,inv=np.unique(np.vstack((raw,boxes)),axis=0,return_inverse=True)
            scores,e=score_boxes(both,truth[target_id]);err=max(err,e)
            ts=gt_sensor if gt_sensor is not None else sensor_frame_box(truth[target_id],target,view)
            sensor,extra=score_boxes(np.stack([sensor_frame_box(b,target,view) for b in both]),ts)
            np.testing.assert_allclose(scores,sensor,atol=1e-8,rtol=0)
            for th in (.25,.5):assert np.array_equal(scores>th,sensor>th)
            err=max(err,extra,float(max(abs(scores-sensor))))
            ri,fi=np.split(scores[inv],2)
            if 'pair' not in entry:
                expected=previous[src,anon(f"{g}|{target}|{entry['case']}")]
                assert abs(fi[0]-expected)<1e-8
                err=max(err,abs(fi[0]-expected));replayed+=1
            else:
                key=entry['pair'],src
                h=sha256(f['t_box'].tobytes()+f['t_p'].tobytes()).hexdigest()
                assert key not in pool_digest or pool_digest[key]==h
                pool_digest[key]=h
            original=best(f['a'],truth[target_id]);whole=original
            seeds=f['seeds'].astype(float)
            if g in COHORTS:
                case=panels[g][entry['case']]['case']
                location=f"{COHORTS[g]}/{case:02d}_{src}.npz"
                path=ROOT/'crossplatform/round171/predictions'/location
                assert digest(path)==history[0][location]
                with np.load(path) as historical:
                    seeds=historical['seeds'].copy()
            cover={}
            if f['active']:
                T=prepare(f,'t')[0];R=prepare(f,'r')[0]
                whole=max(whole,best(f['t_box'],truth[target_id]),best(f['r_box'],truth[target_id]))
                if r is not None and len(T) and len(R):
                    owner_pool=arrays_to_owners([T,R],truth,seeds)
                    for k,(tslots,rslots) in owner_pool.items():
                        cover[k]=int(s in tslots and r in rslots)
            owners=arrays_to_owners([raw,refs,boxes],truth,seeds)
            for k,arm in enumerate(ARMS):
                sampled=bool(p['applied'][k])
                recrow=dict(group='natural' if g in COHORTS else g,case=uid,frame=fr,recording=rec,source=src,target=target,
                    arm=arm,parsed=int(f['active']),applied=int(sampled),raw_iou=float(ri[k]),iou=float(fi[k]),
                    base_pool=original,all_pool=whole,edit=int(np.max(abs(raw[k]-boxes[k]))>1e-8),
                    p_joint=float(p['evidence'][k]),known_ends=int(s is not None and r is not None))
                for field,tagrule in [('unique_iou','iou'),('seed_ownership','seed')]:
                    u,v,w=(int(chunk[k]) for chunk in owners[field])
                    recrow.update({tagrule+'_subject':state(u,s),tagrule+'_ref':state(v,r),tagrule+'_final':state(w,s),
                        tagrule+'_official':state(u,target_id),tagrule+'_official_final':state(w,target_id),
                        tagrule+'_covered':cover.get(field,0)})
                    if 'pair' in entry:
                        endpoints[entry['pair'],src,entry['half'],arm,tagrule]=(u,v,w,s,r)
                data.append(recrow)
        directions.add((src,target))
    assert len(data)==emitted['n']*len(ARMS) and len(directions)==6
    write_gz(OUT/'all_rows.csv.gz',data)
    tables(data,labels,endpoints)
    save(OUT/'audit.json',dict(sealed_rows=emitted['n'],arms=list(ARMS),task_scores=len(data),
        matched_historical_v1=replayed,identical_subject_query_pool_for_both_captions=len(pool_digest),
        max_iou_difference=err,observations=digest(OBS/'SEALED.json'),predictions=digest(PRED/'SEALED.json'),
        protocol=digest(SPEC),training_steps=0,test_updates=0,FINAL_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path);run(p.parse_args().archive)
