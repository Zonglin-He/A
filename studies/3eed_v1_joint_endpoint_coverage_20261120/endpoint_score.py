"""After the prediction seal, audit localization, role ownership, and conditional box generation."""
import argparse
import csv
import gzip
from pathlib import Path
import numpy as np

from collect import CONFIG,HERE,OUT,PARENT,PRIVATE,ROOT,SOURCES,digest,read,save
from joint_coverage import ARMS
from predict_all import DECISIONS,SEALED
from score_roles import PANELS,anon,inventory,many_iou,normalize,score_boxes,sensor_frame_box,sensor_truth,state
from diagnose_roles import COHORTS,historical_records,historical_endpoints,old_seed_owners


def verify_inputs():
    s=read(SEALED);o=read(PRIVATE/'OBSERVED.json')
    assert s['n']==len(s['files'])==o['n']==501 and s['files'].keys()==o['files'].keys()
    assert not s['labels'] and not s['FINAL_access'] and s['target_updates']==s['source_training']==0
    assert s['protocol']==digest(CONFIG) and s['observations']==digest(PRIVATE/'OBSERVED.json')
    assert all(digest(HERE/f)==h for f,h in s['code'].items())
    for name,value in s['files'].items():
        assert digest(DECISIONS/name)==value and digest(PRIVATE/name)==o['files'][name]
    return s


def unique_iou(boxes,labels):
    hit=np.stack([many_iou(boxes,b)>.25 for b in labels],axis=1)
    return np.where(hit.sum(1)==1,hit.argmax(1),-1)


def arrays_to_owners(chunks,truth,seeds):
    stack=np.vstack([v.reshape(-1,7) for v in chunks])
    ok=np.isfinite(stack).all(1)
    boxes,inv=np.unique(stack[ok],axis=0,return_inverse=True)
    rules={'unique_iou':unique_iou(boxes,truth)}
    if seeds is not None:
        rules['seed_ownership']=old_seed_owners(boxes,seeds,truth)
    out={}
    for rule,ids in rules.items():
        vals=np.full(len(stack),-1,int); vals[ok]=ids[inv]
        breaks=np.cumsum([c.reshape(-1,7).shape[0] for c in chunks])[:-1]
        out[rule]=[v.reshape(c.shape[:-1]) for v,c in zip(np.split(vals,breaks),chunks)]
    return out


def fields(known,owners_s,owners_r,chosen_s,chosen_r,final_id,prior,qualified):
    subj,ref=known
    return dict(prior=state(prior,subj),raw=state(chosen_s,subj),final=state(final_id,subj),ref=state(chosen_r,ref),
        cover_s=('NA' if subj is None else int(subj in owners_s)),cover_r=('NA' if ref is None else int(ref in owners_r)),
        different_resolved=int(prior>=0 and chosen_s>=0 and prior!=chosen_s),
        any_raw_qualified=int(np.any(qualified)))


def evaluate_case(panel,index,entry,source,truth,truth_sensor,endpoints,oldpins,original_scores):
    row=entry['row'];target=row['platform'];view=int(row['suffix'] or 0)
    name=f'{panel}_{index:03d}_{source}.npz'
    seed=None
    if panel in COHORTS:
        tag=f"{COHORTS[panel]}/{entry['case']:02d}_{source}.npz"
        historical=ROOT/'crossplatform/round171/predictions'/tag
        assert digest(historical)==oldpins[tag]
        with np.load(historical,allow_pickle=False) as prior: seed=prior['seeds']
    case=anon(f'{panel}|{index}')
    group=('own_proxy' if source==target else 'cross_proxy') if panel=='source_proxy' else panel
    with np.load(PRIVATE/name,allow_pickle=False) as z,np.load(DECISIONS/name,allow_pickle=False) as dec:
        a,d=z['a'],z['d']
        rs=d if len(d) else a
        raw,ref,box,ss,rr=[dec[k] for k in ('raw','refs','boxes','subjects','references')]
        joined=np.vstack([raw,box])
        ious,e1=score_boxes(joined,truth[0])
        sensor=np.stack([sensor_frame_box(b,target,view) for b in joined])
        sim,e2=score_boxes(sensor,truth_sensor)
        discrepancy=float(np.max(np.abs(ious-sim)))
        np.testing.assert_allclose(ious,sim,atol=1.e-8,rtol=0.)
        for t in (.25,.50):
            np.testing.assert_array_equal(ious>t,sim>t)
        raw_iou,end_iou=np.split(ious,2)
        assert abs(end_iou[0]-original_scores[group,source,case])<1.e-10
        native=a.copy(); native[:,6]=0
        full_phase=np.vstack([a,rs])
        flat=full_phase.copy();flat[:,6]=0
        prior_best=float(max(many_iou(a,truth[0]).max(),many_iou(native,truth[0]).max()))
        all_best=float(max(many_iou(full_phase,truth[0]).max(),many_iou(flat,truth[0]).max()))
        assert np.all(raw_iou<=all_best+1.e-8)
        base=dict(panel=panel,group=group,source=source,target=target,case=case,
            recording=anon(str(Path(row['metapath']).resolve().parent.parent)),active=int(z['active']),applied=int(dec['applied']),
            v1_iou=float(end_iou[0]),best_original=prior_best,best_all=all_best,
            native_dual_known=int(None not in endpoints),native_s_known=int(endpoints[0] is not None),native_r_known=int(endpoints[1] is not None),
            mutual_information=float(dec['mutual_information']),sub_visible=float(dec['visible_fraction'][0]),ref_visible=float(dec['visible_fraction'][1]))
        owners=arrays_to_owners([raw,ref,box,ss,rr,a,rs],truth,seed)
        records=[];associations=[]
        for k,arm in enumerate(ARMS):
            r=dict(**base,arm=arm,raw_iou=float(raw_iou[k]),iou=float(end_iou[k]),
                expected_coverage=float(dec['coverage'][k,0]),selected_mass=float(dec['coverage'][k,1]),
                full_pool=int(dec['full_pool'][k]),raw_changed=int(not np.array_equal(raw[k],raw[0])),
                same_raw_as_main=int(np.array_equal(raw[k],raw[-1])),
                same_ref_as_main=int(np.array_equal(ref[k],ref[-1],equal_nan=True)),
                same_box_as_main=int(np.allclose(box[k],box[-1],atol=1.e-9,rtol=0.)),
                same_subject_slate_as_main=int({tuple(b) for b in ss[k]}=={tuple(b) for b in ss[-1]}),
                same_ref_slate_as_main=int(np.array_equal(rr[k],rr[-1],equal_nan=True) or {tuple(b) for b in rr[k]}=={tuple(b) for b in rr[-1]}),
                shape_changed=int(1-many_iou(raw[k:k+1],box[k])[0]>1.e-8))
            if dec['full_pool'][k]:
                slate_best=float(many_iou(a,truth[0]).max())
            else:
                slate_best=float(many_iou(ss[k],truth[0]).max())
            r['slate_best']=slate_best
            for rule,(sid,rid,fid,sids,rids,aids,dids) in owners.items():
                us,ur=(aids,dids) if dec['full_pool'][k] else (sids[k],rids[k])
                vals=fields(endpoints,us,ur,sid[k],rid[k],fid[k],sid[0],slate_best>.50)
                associations.append(dict(group=group,source=source,target=target,case=case,arm=arm,rule=rule,
                    applied=base['applied'],both_known=base['native_dual_known'],**vals,
                    pool_s=('NA' if endpoints[0] is None else int(endpoints[0] in aids)),
                    pool_r=('NA' if endpoints[1] is None else int(endpoints[1] in dids)),
                    raw_iou=float(raw_iou[k]),iou=float(end_iou[k]),best_original=prior_best,best_all=all_best,
                    hit_new_outside_all=int(end_iou[k]>.5>=all_best)))
                if rule=='unique_iou':
                    r.update(official_raw=state(sid[k],0),official_final=state(fid[k],0),official_prior=state(sid[0],0),
                             human_raw=vals['raw'],human_ref=vals['ref'],human_final=vals['final'])
            records.append(r)
    return records,associations,max(e1,e2,discrepancy)


def write_gz(path,data):
    with gzip.open(path,'wt',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=data[0]);writer.writeheader();writer.writerows(data)


def run(repo):
    seal=verify_inputs()
    people=historical_records(repo)
    with gzip.open(PARENT/'results/all_rows.csv.gz','rt') as f:
        baseline={(r['group'],r['source'],r['case']):float(r['iou']) for r in csv.DictReader(f) if r['arm']=='v1'}
    assert len(baseline)==501
    rows=[];identities=[];err=0.
    for panel,path in PANELS.items():
        for case,entry in enumerate(read(path)):
            row=entry['row'];target=row['platform'];view=int(row['suffix'] or 0)
            truth,cap=inventory(row['metapath'],target,view,int(row['object_index']))
            assert cap==normalize(row['caption'])
            truth_sensor=sensor_truth(target,str(Path(row['metapath']).resolve()),row['caption'],row['suffix'])
            if panel in COHORTS:
                endpoints,_=historical_endpoints(COHORTS[panel],entry,truth,people)
            else:
                endpoints=(0,int(entry['ref_id']))
            for source in SOURCES:
                if source==target and panel!='source_proxy': continue
                rr,ii,dis=evaluate_case(panel,case,entry,source,truth,truth_sensor,endpoints,people[0],baseline)
                rows.extend(rr);identities.extend(ii);err=max(dis,err)
    assert len(rows)==501*len(ARMS) and len(identities)==(501+168)*len(ARMS)
    write_gz(OUT/'all_rows.csv.gz',rows);write_gz(OUT/'identity_rows.csv.gz',identities)
    save(OUT/'scoring.json',dict(records=501,arm_rows=len(rows),identity_rows=len(identities),
         eligible=seal['active'],applied=seal['applied'],old_v1_predictions_matched=501,
         max_iou_disagreement=err,decision_hash=digest(SEALED),observed_hash=digest(PRIVATE/'OBSERVED.json'),
         protocol=digest(CONFIG),source_updates=0,target_updates=0,FINAL_access=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path)
    run(p.parse_args().archive)

