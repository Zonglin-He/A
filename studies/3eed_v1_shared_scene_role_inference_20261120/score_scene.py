"""Unlock ordinary and verified-endpoint labels only after sealing the entire prediction table."""
import argparse
from collections import defaultdict
import csv
import gzip
from pathlib import Path

import numpy as np

from shared_paths import HERE, ROOT, LEX, LAST, OLD, NEW, SEG, PRED, OUT, SPEC, ARMS, digest, read, save, inputs, name
from predict_scene import IMPLEMENTATION
from evaluate_witness import metadata, PANELS, COHORTS, historical_records
from endpoint_score import arrays_to_owners,write_gz
from score_roles import score_boxes,state,sensor_frame_box
from score_swap import best
from score_lex import lexical_has_pass


def load_labels(rows):
    assert digest(NEW/'labels.json')==read(OUT/'panel.json')['labels']
    assert digest(OLD/'labels.json')==read(LAST/'results/panel.json')['labels']
    old=read(OLD/'labels.json');new=read(NEW/'labels.json')
    assert not (old.keys()&new.keys())
    # The original sampling roster located 313/397 old paths in ordinary metadata.
    # Check recording disjointness independently for *every* old acquisition by its canonical recording path.
    rec=lambda r:(r['frame']['target'],Path(r['frame']['calibration']).resolve().parent.parent.name)
    previous={rec(row[0]) for row in rows if row[0]['uid'] in old}
    unseen=set()
    for row,*_ in rows:
        if row['uid'] in new:
            already=rec(row) in previous
            assert new[row['uid']]['exposed_recording']==already
            if not already:unseen.add(rec(row))
    return {**old,**new},len(previous),len(unseen)


def evaluate(repo):
    rows=list(inputs())
    seal=read(PRED/'SEALED.json')
    assert seal['protocol']==digest(SPEC) and set(seal['files'])=={name(row[0]) for row in rows}
    assert seal['n']==len(rows) and not seal['labels_used'] and not seal['FINAL_access']
    assert seal['new']==digest(NEW/'SEALED.json') and seal['old']==digest(LEX/'private/observations/SEALED.json')
    for k in IMPLEMENTATION:assert seal['code'][k]==digest(HERE/k)
    for s in ('old','new'):assert seal['partition'][s]==digest(SEG/(s+'.json'))
    labels,hist_rec,new_rec=load_labels(rows)
    cache={};history=historical_records(repo);panels={g:read(PANELS[g]) for g in COHORTS}
    with gzip.open(LEX/'results/task_rows.csv.gz','rt') as f:
        replay={(r['case'],r['source'],r['arm']):float(r['iou']) for r in csv.DictReader(f) if r['arm'] in ('v1','joint_raw')}
    audits=defaultdict(int);max_error=0.
    result=[];pairs={};ranks={};slates={}
    paired_modes=('independent','separate_worlds','separate_scales','wrong_reference','image_center','joint_raw')
    for index,(row,src,zsha,lex,lsha,fid,words,old,_) in enumerate(rows,1):
        uid=row['uid'];n=name(row);entry=labels[uid]
        target=row['frame']['target'];source=row['source'];g=entry['group']
        truth,subject,reference,frame,recording,view,sensor=metadata(row,entry,panels,history,cache)
        tid=entry.get('subject',0);gt=truth[tid]
        assert digest(src)==zsha and digest(PRED/n)==seal['files'][n]
        with np.load(src) as z,np.load(PRED/n) as p:
            raw=p['raw'];ref=p['ref'];boxes=p['boxes']
            u,inverse=np.unique(np.vstack((raw,boxes)),axis=0,return_inverse=True)
            task,err=score_boxes(u,gt)
            target_scores,se=score_boxes(np.stack([sensor_frame_box(b,target,view) for b in u]),
                    sensor if sensor is not None else sensor_frame_box(gt,target,view))
            max_error=max(max_error,err,se,float(abs(target_scores-task).max()))
            assert max_error<1.e-8
            np.testing.assert_array_equal(target_scores[:,None]>[.25,.5],task[:,None]>[.25,.5])
            before,after=np.split(task[inverse],2)
            if old:
                for i,arm in enumerate(('v1','joint_raw')):
                    assert abs(after[i]-replay[uid,source,arm])<1.e-8
                audits['previous_cases_replayed']+=1
            whole=original=best(z['a'],gt)
            if z['active']:whole=max(original,best(z['t_box'],gt),best(z['r_box'],gt))
            lex_pass=False
            if whole<=.5 and np.any(after>.5):
                assert digest(lex)==lsha
                with np.load(lex) as saved:lex_pass=lexical_has_pass(saved['native_boxes'],saved['sample'],gt)
                audits['lexical_certificates_checked']+=1
            seeds=z['seeds']
            if g in COHORTS:
                case=panels[g][entry['case']]['case']
                rel=f'{COHORTS[g]}/{case:02d}_{source}.npz'
                path=ROOT/'crossplatform/round171/predictions'/rel
                assert digest(path)==history[0][rel]
                with np.load(path) as f:seeds=f['seeds']
            owners=arrays_to_owners([raw,ref,boxes],truth,seeds)
            if 'pair' in entry:
                pair=entry['pair'];half=entry['half'];slate=p['candidates']
                full_bytes=z['t_box'].tobytes()+z['t_p'].tobytes()
                from hashlib import sha256
                h=sha256(full_bytes+slate.tobytes()).digest()
                assert h==slates.setdefault((pair,source),h)
                if len(slate):
                    classified=arrays_to_owners([slate],truth,seeds)
                else:classified={mode:[np.empty(0,int)] for mode in owners}
                for mode,(ids,) in classified.items():
                    for arm in paired_modes:
                        scores=p['subject_scores'][ARMS.index(arm)]
                        ranks[pair,source,mode,arm,half]=({int(k):float(scores[ids==k].max()) for k in np.unique(ids) if k>=0},subject)
            for i,arm in enumerate(ARMS):
                d=dict(group='natural' if g in COHORTS else g,case=uid,frame=frame,recording=recording,
                  exposure=('seen' if entry['exposed_recording'] else 'new') if 'exposed_recording' in entry else 'NA',
                  source=source,target=target,arm=arm,known=int(subject is not None and reference is not None),
                  parsed=int(z['active']),applied=int(p['applied'][i]),evidence=float(p['activation_mass'][i]),
                  raw_iou=float(before[i]),iou=float(after[i]),original_pool=original,role_pool=whole,
                  edited=int(np.max(abs(boxes[i]-raw[i]))>1.e-8),changed_raw=int(not np.array_equal(raw[i],raw[0])),
                  beyond_a256=int(after[i]>.5>=original),beyond_all=int(after[i]>.5>=whole and not lex_pass))
                for mode,short in [('unique_iou','iou'),('seed_ownership','seed')]:
                    s,r,f=(int(chunk[i]) for chunk in owners[mode])
                    d.update({short+'_s':state(s,subject),short+'_r':state(r,reference),
                        short+'_raw':state(s,tid),short+'_final':state(f,tid)})
                    if 'pair' in entry:
                        pairs[pair,source,mode,arm,half]=(g,entry.get('exposed_recording'),target,s,r,subject,reference,frame,recording)
                result.append(d)
        if index%250==0:print('scored',index,len(rows),flush=True)
    endpoints=[];contrast=[]
    for (pair,src,mode,arm,half),value in pairs.items():
        if half!=0:continue
        x,y=value,pairs[pair,src,mode,arm,1]
        assert x[5]!=y[5] and (x[6]==y[6])==(x[0]=='new_side')
        info=dict(panel='new_side' if x[0]=='new_side' else 'old_switch',pair=pair,frame=x[7],recording=x[8],
            exposure=('seen' if x[1] else 'new') if x[1] is not None else 'NA',target=x[2],source=src,arm=arm,rule=mode)
        endpoints.append(dict(**info,both_subjects=int(x[3]==x[5] and y[3]==y[5]),
          both_refs=int(x[4]==x[6] and y[4]==y[6]),four=int(x[3:5]==x[5:7] and y[3:5]==y[5:7]),
          distinct=int(min(x[3],y[3])>=0 and x[3]!=y[3])))
        if arm not in paired_modes:continue
        (one,s0),(two,s1)=(ranks[pair,src,mode,arm,i] for i in range(2))
        assert set(one)==set(two) and s0!=s1
        available=s0 in one and s1 in one
        m0=one[s0]-one[s1] if available else 0.
        m1=two[s1]-two[s0] if available else 0.
        contrast.append(dict(**info,available=int(available),margin0=m0,margin1=m1,
             either_tie=int(available and min(abs(m0),abs(m1))<=1.e-12),
             both_correct=int(available and m0>1.e-12 and m1>1.e-12),
             both_reversed=int(available and m0< -1.e-12 and m1< -1.e-12)))
    assert len(result)==len(rows)*len(ARMS) and len({(r['source'],r['target']) for r in result})==6
    write_gz(OUT/'task_rows.csv.gz',result);write_gz(OUT/'endpoint_rows.csv.gz',endpoints)
    write_gz(OUT/'contrast_rows.csv.gz',contrast)
    from tables_scene import tabulate
    tabulate(result,endpoints,contrast)
    save(OUT/'audit.json',dict(n=len(rows),predictions=len(result),arms=len(ARMS),old_recordings=hist_rec,
        new_recordings=new_rec,side_and_old_pair_directions=len(slates),same_subject_pool_hash_checks=len(slates),
        prior_replays=dict(audits),max_sensor_and_scorer_error=max_error,prediction_seal=digest(PRED/'SEALED.json'),
        source_training=0,target_updates=0,FINAL_access=False,protocol=digest(SPEC)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    evaluate(parser.parse_args().archive)
