"""Independently check thresholds, hard caption geometry and the interpretability of size evidence."""
from collections import defaultdict
import csv
import gzip
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from paths_shape import *
from predict_shape import observations
from projection_witness import projection
from physical_evidence import Returns, axes
from scene_faces import split_returns
from shape_evidence import observed_extent, predicted_extent, compatibility
from score_roles import csv_file, score_boxes
from score_swap import best
from score_lex import lexical_has_pass


def load_gz(p):
    with gzip.open(p,'rt') as f:
        return list(csv.DictReader(f))


def object_audit():
    label=read(WORK/'labels.json')
    seal=read(OBS/'SEALED.json')
    seen=set();rows=[];cert=[];specials=[]
    byword=defaultdict(list)
    for row,_,_,lp,lh,fid,words,*_ in observations():
        uid=row['uid']
        if uid not in label:continue
        entry=label[uid];pair=entry['pair'];source=row['source']
        kind=entry['group'];f=row['frame']
        key=(source,pair); n=name(row)
        with np.load(PRED/n) as result,np.load(WORK/'role'/n) as role:
            target=entry['truth'][entry['subject']]
            k=ARMS.index('local_shape')
            metric=lambda B:float(score_boxes(np.array([B]),np.asarray(target))[0][0])
            original=role['a'];zero=original.copy();zero[:,6]=0
            if metric(result['final'][k])>.5>=metric(result['raw'][k]):
                bl=float(best(original,np.asarray(target)));bz=float(best(zero,np.asarray(target)))
                kr=ARMS.index('subject_only');kn=ARMS.index('wrong_reference')
                with np.load(lp) as lex:
                    lexical=lexical_has_pass(lex['native_boxes'],lex['sample'],np.asarray(target))
                cert.append(dict(case=uid,source=source,raw=metric(result['raw'][k]),final=metric(result['final'][k]),
                     subject_only_raw=metric(result['raw'][kr]),subject_only_final=metric(result['final'][kr]),
                     wrong_ref_raw=metric(result['raw'][kn]),a256_pca=bl,a256_zero_yaw=bz,
                     best_subject_phrase=best(role['t_box'],np.asarray(target)),best_ref_phrase=best(role['r_box'],np.asarray(target)),
                     multilist_lexical_already_passes=int(lexical)))
        if (key,uid) in seen:continue
        seen.add((key,uid))
        if digest(lp)!=lh:raise ValueError('Unsealed returns')
        wordp=OBS/(fid+'.npz');assert digest(wordp)==seal['hashes'][fid]
        with np.load(lp) as cloud,np.load(wordp) as w:
            x=Returns(cloud['sample']);parity=split_returns(x.xyz)
            np.testing.assert_equal(cloud['xyz'],w['xyz'])
            seeds,inv=np.unique(w['xyz'],axis=0,return_inverse=True)
            dims=np.maximum(w['extent'].astype(float),1e-6)
            sizes=np.zeros((len(seeds),)+dims.shape[1:]);np.add.at(sizes,inv,dims)
            sizes=predicted_extent(sizes/np.bincount(inv)[:,None,None])
            nearest=cKDTree(seeds).query(x.xyz)[1]
            for label_role,tid,c in [('subject',entry['subject'],words[0]),('reference',entry['reference'],words[1])]:
                dedup=(key,tid)
                if dedup in byword:
                    assert byword[dedup]==c;continue
                byword[dedup]=c
                box=np.asarray(entry['truth'][tid]);inside,_=x.local(box)
                # Remove points also owned by another annotated box; no task predictor receives these boundaries.
                unique=np.ones(len(inside),bool)
                for j,other in enumerate(entry['truth']):
                    if j==tid:continue
                    b=np.asarray(other);off=(x.xyz[inside]-b[:3])@axes(b).T
                    unique&=~np.all(abs(off)<=b[3:6]/2+1e-7,axis=1)
                ids=inside[unique];width=observed_extent(x.xyz[ids],parity[ids])
                full=predicted_extent(box[3:6]);near=np.unique(nearest[ids])
                local=np.median(sizes[near],axis=0) if len(near) else np.median(sizes,axis=0)
                for measure,lengths in [('measured_lower_bound',width),('measured_two_sided',width),
                                        ('oracle_full_size_lower_bound',full),('oracle_full_size_two_sided',full)]:
                    for mode,templates in [('local',local),('scene',np.median(sizes,axis=0))]:
                        scores=compatibility(lengths,templates)
                        if measure.endswith('two_sided'):
                            ratios=np.ones_like(templates)
                            valid=lengths>0
                            ratios[:,valid]=np.minimum(templates[:,valid]/lengths[valid],lengths[valid]/templates[:,valid])
                            scores=ratios.prod(axis=-1)
                        alts=np.delete(scores,c)
                        auc=np.mean((scores[c]>alts)+.5*(scores[c]==alts))
                        top=scores==scores.max()
                        rows.append(dict(pair=pair,source=source,frame=entry['frame'],recording=entry['recording'],
                            physical_role=label_role,ordinal=len(byword),measure=measure,model=mode,points=len(ids),
                            usable=int(np.min(width)>0),all_tied=int(np.ptp(scores)<=1.e-12),
                            auc=float(auc),top1=float(top[c]/top.sum())))
    csv_file(OUT/'known_object_scalar_rows.csv',rows)
    if cert:csv_file(OUT/'challenge_geometry.csv',cert)
    totals=[]
    for (role,measure,model),items in grouped_local(rows,['physical_role','measure','model']).items():
        totals.append(dict(role=role,measure=measure,model=model,objects_x_source=len(items),
            frames=len({v['frame'] for v in items}),all_tied=sum(v['all_tied'] for v in items),usable=sum(v['usable'] for v in items),
            auc=np.mean([v['auc'] for v in items]),fractional_top1=np.mean([v['top1'] for v in items])))
    csv_file(OUT/'known_object_scores.csv',totals)
    return len({k[0] for k in seen}),len(rows),len(cert)


def grouped_local(rows,keys):
    out=defaultdict(list)
    for r in rows:out[tuple(r[k] for k in keys)].append(r)
    return out


def run():
    a=read(OUT/'audit.json');p=read(PRED/'SEALED.json')
    assert digest(PRED/'SEALED.json')==a['prediction_seal']
    assert all(digest(PRED/n)==v for n,v in p['files'].items())
    rows=load_gz(OUT/'task_rows.csv.gz');pairs=load_gz(OUT/'endpoint_rows.csv.gz')
    groups=grouped_local(rows,['panel','source','target','arm'])
    with (OUT/'metrics.csv').open() as f:metrics=list(csv.DictReader(f))
    for r in metrics:
        panel=r['panel'];aa=r['arm']
        selected=['ordinary_uniform','ordinary_relation'] if panel=='ordinary_combined' else [panel]
        results=[]
        for s in SOURCES:
            for t in SOURCES:
                d=s+'_to_'+t
                if d!=r['direction'] and r['direction']!='macro':continue
                sub=[q for g in selected for q in groups.get((g,s,t,aa),[])]
                if sub:
                    values=np.asarray([float(q['raw_iou']) for q in sub])
                    results.append(100*np.array([(values>.25).mean(),(values>.5).mean(),values.mean()]))
        np.testing.assert_allclose(np.mean(results,axis=0),[float(r[k]) for k in ('acc25','acc50','miou')],atol=1.e-10)
    assert a['n']*len(ARMS)==len(rows) and not a['FINAL_access']
    assert len(set((z['case'],z['source'],z['arm']) for z in rows))==len(rows)
    old=WORK/'signed_size_audit'
    diff={}
    if (old/'task_rows.csv.gz').is_file():
        early={(r['case'],r['source'],r['arm']):r for r in load_gz(old/'task_rows.csv.gz')}
        for (arm,),sub in grouped_local(rows,['arm']).items():
            delta=[]
            for r in sub:
                e=early[r['case'],r['source'],r['arm']]
                delta.append([float(r['raw_iou'])-float(e['raw_iou']),float(r['final_iou'])-float(e['final_iou'])])
            diff[arm]=(np.abs(delta)>1.e-10).sum(0).tolist()
    w=read(OBS/'SEALED.json')
    bad=np.zeros(3,int);total=0;bad_sets=0
    for fid in w['hashes']:
        with np.load(OBS/(fid+'.npz')) as f:
            bad+=(f['extent']<=0).sum((0,1));total+=f['extent'].size;bad_sets+=int((f['extent']<=0).any())
    q=object_audit()
    save(OUT/'verification.json',dict(metric_rows_checked=len(metrics),all_predictions_checked=len(rows),
        endpoint_pair_rows=len(pairs),hard_source_frame_pairs=q[0],labeled_object_comparisons=q[1],
        hard_geometry_crossings=q[2],metric_source_frames=len(w['hashes']),native_size_replay_errors=w['native'],
        nonpositive_component_counts=bad.tolist(),raw_extent_components=total,metric_frames_with_invalid_component=bad_sets,
        pointwise_production_floor=1e-6,corrected_vs_first_pass_changed_ious=diff,labels_only_after_sealing=True,
        protocol=digest(SPEC),FINAL_access=False))

if __name__=='__main__':run()
