"""Task and membership scoring, only after the predictions on every input are sealed."""
import argparse
from collections import defaultdict
from pathlib import Path
import csv
import gzip
import numpy as np
from common_lex import *
from evaluate_witness import metadata, PANELS, COHORTS, historical_records, tag
from endpoint_score import arrays_to_owners, write_gz
from score_roles import score_boxes, state, sensor_frame_box, csv_file, measure, record_intervals
from score_swap import best
from ordered_choice import prepare,relations
from physical_evidence import choose
from panel_witness import inventory
from scope183 import VIEWS
from batch_yaw_iou import many_iou
from crossplatform.candidate_frontend import orient_box


def lexical_has_pass(raw_boxes, sample, truth, cut=.5):
    # For any yaw the intersection is bounded by the smaller cross-section times the z overlap.
    boxes=raw_boxes.reshape(-1,6).copy()
    boxes[:,3:6]=boxes[:,3:6].clip(1.e-6)
    area=np.prod(boxes[:,3:5].astype(float),axis=1)
    area_gt=float(np.prod(truth[3:5]))
    vol=area*boxes[:,5];vol_gt=area_gt*truth[5]
    height=np.maximum(0,np.minimum(boxes[:,2].astype(float)+boxes[:,5]/2,truth[2]+truth[5]/2)-
                         np.maximum(boxes[:,2].astype(float)-boxes[:,5]/2,truth[2]-truth[5]/2))
    threshold=(vol+vol_gt)*cut/(1+cut)
    close=np.linalg.norm(boxes[:,:2]-truth[:2],axis=1)<=(
        np.linalg.norm(boxes[:,3:5],axis=1)+np.linalg.norm(truth[3:5]))/2+1.e-5
    possible=boxes[close & (np.minimum(area,area_gt)*height+1.e-5>=threshold)]
    if len(possible):
        flat=np.column_stack((possible,np.zeros(len(possible))))
        if many_iou(flat,truth).max()>cut:return True
        for box in possible:
            if many_iou(orient_box(sample,box)[None],truth)[0]>cut:return True
    return False


def members(row,entry,z,p,truth,owner,ts,rs,cc):
    classifications=[];upper=[]
    view=VIEWS.index(row['frame']['waymo_view'])
    classes,checked,conflict=inventory(dict(metapath=row['frame']['calibration'],platform=row['frame']['target'],suffix=view))
    assert not conflict
    np.testing.assert_array_equal(checked,truth)
    T,_,mt,tp=prepare(z,'t');R,_,mr,rp=prepare(z,'r')
    for mode,(st,rf) in owner.items():
        truth_masks=[]
        for prefix,ids,target,noun in [('t',st,ts,cc[0]),('r',rf,rs,cc[1])]:
            assert noun>=0 and classes[target]==CLASSES[noun]
            correct=np.zeros(len(ids),bool)
            assigned=ids>=0
            correct[assigned]=classes[ids[assigned]]==CLASSES[noun]
            truth_masks.append(correct)
            for arm in ('main','one','wrong'):
                chosen=p[prefix+'_'+arm]==noun
                others=[ix for ix in np.unique(ids[assigned]) if classes[ix]!=CLASSES[noun]]
                right=[ix for ix in np.unique(ids[assigned]) if classes[ix]==CLASSES[noun]]
                classifications.append(dict(case=row['uid'],source=row['source'],pair=entry['pair'],half=entry['half'],
                  rule=mode,role=prefix,arm=arm,available=int(target in ids),retained=int(np.any(chosen & (ids==target))),
                  correct_classes=len(right),retained_classes=sum(np.any(chosen & (ids==ix)) for ix in right),
                  wrong_classes=len(others),accepted_wrong=sum(np.any(chosen & (ids==ix)) for ix in others),
                  unknown_selected=int(np.sum(chosen & ~assigned))))
        # The only intervention is supplying membership for the noun; individual endpoint IDs stay hidden.
        result=choose(T,R,tp,rp,mt,mr,relations(tp,rp,int(z['side'])),*truth_masks,
                      int(z['depth']),bool(z['immediate']),z['raw'])
        raw,ref=(z['raw'],z['ref']) if result is None else result
        upper.append((mode,raw,ref))
    return classifications,upper


def run(repo):
    roster=panel()
    pred=read(PRED/'SEALED.json');obs=read(OBS/'SEALED.json')
    assert pred['observations']==digest(OBS/'SEALED.json') and obs['protocol']==pred['protocol']==digest(SPEC)
    assert not pred['labels_used'] and not obs['labels_used'] and not pred['FINAL_access']
    assert all(digest(HERE/f)==h for f,h in pred['code'].items())
    assert obs['code']==digest(HERE/'observe_lex.py') and obs['common']==digest(HERE/'common_lex.py')
    old_obs=read(OLD/'observed/SEALED.json')['files']
    with gzip.open(LAST/'results/all_rows.csv.gz','rt') as f:
        historical={(r['case'],r['source'],r['arm']):r for r in csv.DictReader(f) if r['arm'] in ('v1','witness_raw')}
    labels=read(OLD/'labels.json')
    assert digest(OLD/'labels.json')==read(LAST/'results/panel.json')['labels']
    panels={g:read(PANELS[g]) for g in COHORTS};history=historical_records(repo);cache={}
    rows=[];cl=[];oracle=[];maxerr=0.;lookups=0;outcomes={}
    for row in roster:
        uid=row['uid'];entry=labels[uid];target=row['frame']['target'];src=row['source']
        g=entry['group'];n=name(row);fid=obs['files'][n]
        assert digest(PRED/n)==pred['files'][n] and digest(OLD/'observed'/n)==old_obs[n]
        assert digest(OBS/(fid+'.npz'))==obs['hashes'][fid]
        truth,ts,rs,frame,rec,view,gt_sensor=metadata(row,entry,panels,history,cache)
        tid=entry.get('subject',0);gt=truth[tid]
        with np.load(PRED/n) as p,np.load(OLD/'observed'/n) as z,np.load(OBS/(fid+'.npz')) as lex:
            raw,refs,boxes=p['raw'],p['ref'],p['boxes']
            uu,iv=np.unique(np.vstack((raw,boxes)),axis=0,return_inverse=True)
            task,error=score_boxes(uu,gt);maxerr=max(maxerr,error)
            sensor,serr=score_boxes(np.stack([sensor_frame_box(b,target,view) for b in uu]),
                gt_sensor if gt_sensor is not None else sensor_frame_box(gt,target,view))
            assert np.allclose(sensor,task,rtol=0,atol=1.e-8)
            assert all(np.array_equal(sensor>t,task>t) for t in (.25,.5))
            maxerr=max(maxerr,serr,float(np.max(abs(sensor-task))))
            direct,final=np.split(task[iv],2)
            for i,key in enumerate(('v1','witness_raw')):
                assert abs(final[i]-float(historical[uid,src,key]['iou']))<1.e-8
            base=best(z['a'],gt)
            union=max(base,best(z['t_box'],gt),best(z['r_box'],gt)) if z['active'] else base
            external=False
            if union<=.5 and np.any(final>.5):
                lookups+=1
                external=lexical_has_pass(lex['native_boxes'],lex['sample'],gt)
            seeds=z['seeds'].copy()
            if g in COHORTS:
                case=panels[g][entry['case']]['case']
                loc=f'{COHORTS[g]}/{case:02d}_{src}.npz'
                path=ROOT/'crossplatform/round171/predictions'/loc
                assert digest(path)==history[0][loc]
                with np.load(path) as inside:seeds=inside['seeds']
            both_known=int(ts is not None and rs is not None)
            owners=arrays_to_owners([raw,refs,boxes],truth,seeds)
            if 'pair' in entry:
                source_owner=arrays_to_owners([prepare(z,'t')[0],prepare(z,'r')[0]],truth,seeds)
                diag,upp=members(row,entry,z,p,truth,source_owner,ts,rs,obs['nouns'][n])
                cl+=diag
                for mode,sbox,rbox in upp:
                    sown,rown=arrays_to_owners([sbox[None],rbox[None]],truth,seeds)[mode]
                    oracle.append(dict(pair=entry['pair'],source=src,case=uid,half=entry['half'],rule=mode,
                        available=int(ts in source_owner[mode][0] and rs in source_owner[mode][1]),
                        s=int(sown[0]==ts),r=int(rown[0]==rs),both=int(sown[0]==ts and rown[0]==rs)))
            for i,arm in enumerate(ARMS):
                x=dict(group='natural' if g in COHORTS else g,case=uid,source=src,target=target,frame=frame,
                    recording=rec,arm=arm,known_both=both_known,active=int(z['active']),applied=int(p['applied'][i]),
                    subject_noun=int(obs['nouns'][n][0]>=0),ref_noun=int(obs['nouns'][n][1]>=0),
                    raw_iou=float(direct[i]),iou=float(final[i]),raw_pool=base,role_pool=union,
                    changed_identity_box=int(not np.array_equal(raw[i],raw[0])),
                    geometry_edit=int(np.any(np.abs(boxes[i]-raw[i])>1.e-8)),
                    donors=int(p['witnesses'][i].max()),beyond_original=int(final[i]>.5>=base),
                    beyond_every_pool=int(final[i]>.5>=union and not external))
                for mode,tagrule in [('unique_iou','iou'),('seed_ownership','seed')]:
                    so,ro,fo=(int(chunk[i]) for chunk in owners[mode])
                    x.update({tagrule+'_s':state(so,ts),tagrule+'_r':state(ro,rs),
                          tagrule+'_official':state(so,tid),tagrule+'_final':state(fo,tid)})
                    if 'pair' in entry:
                        outcomes[entry['pair'],src,arm,mode,entry['half']]=(so,ro,ts,rs)
                rows.append(x)
    assert len(rows)==844*len(ARMS)
    write_gz(OUT/'task_rows.csv.gz',rows);write_gz(OUT/'class_rows.csv.gz',cl);write_gz(OUT/'oracle_rows.csv.gz',oracle)
    pair_rows=[]
    for pair,src,arm,mode,h in outcomes:
        if h!=0:continue
        x,y=(outcomes[pair,src,arm,mode,k] for k in (0,1))
        assert x[2]!=y[2] and x[3]!=y[3]
        pair_rows.append(dict(pair=pair,source=src,arm=arm,rule=mode,
             both_subjects=int(x[0]==x[2] and y[0]==y[2]),both_refs=int(x[1]==x[3] and y[1]==y[3]),
             four=int(x[:2]==x[2:] and y[:2]==y[2:]),resolved_switch=int(x[0]>=0 and y[0]>=0 and x[0]!=y[0])))
    write_gz(OUT/'pair_rows.csv.gz',pair_rows)
    from tables_lex import tables
    tables(rows,pair_rows,cl,oracle)
    save(OUT/'audit.json',dict(records=len(roster),source_frames=len(obs['hashes']),predictions=len(rows),
       observed=digest(OBS/'SEALED.json'),sealed=digest(PRED/'SEALED.json'),protocol=digest(SPEC),
       parent_observed=CFG['previous_observations'],v1_replay=len(roster),previous_joint_replay=len(roster),
       native_logit_box_errors=obs['native'],max_two_scorer_error=maxerr,lexical_pool_certificate_rows=lookups,
       source_training=0,target_updates=0,FINAL_access=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    run(p.parse_args().archive)
