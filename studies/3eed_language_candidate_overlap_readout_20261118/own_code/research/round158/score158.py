"""Post-seal evaluation; labels and argmax IoU are strictly evaluator-only."""
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research/round156'))
from score156 import META,aggregates,digest,ground_truth,many_iou,oriented_iou,read,read_csv,require,rotation,save,table
from reader158 import ARMS

INPUT=ROOT/'research/round145/INPUT_REGISTRY_PRIVATE.json'
OUT=HERE/'results'
DATA=ROOT/'crossplatform/round158'
MAIN='role_risk'


def summarize_pair(rr,arm,control):
    x=np.array([r[arm]-r[control] for r in rr])
    w=np.array([r['sampling_weight'] for r in rr]); wdir=np.zeros_like(w)
    directions=sorted({(r['source'],r['target']) for r in rr})
    for src,tgt in directions:
        keep=np.array([(r['source'],r['target'])==(src,tgt) for r in rr])
        wdir[keep]=w[keep]/(len(directions)*w[keep].sum())
    positive=x>1e-12; negative=x< -1e-12
    return dict(n=len(x),better=int(positive.sum()),worse=int(negative.sum()),same=int((~(positive|negative)).sum()),
                positive_pp=float(100*wdir@np.maximum(x,0)),negative_pp=float(100*wdir@np.minimum(x,0)),
                net_pp=float(100*wdir@x),rescued25=sum(r[control]<=.25<r[arm] for r in rr),
                harmed25=sum(r[arm]<=.25<r[control] for r in rr),
                rescued50=sum(r[control]<=.5<r[arm] for r in rr),harmed50=sum(r[arm]<=.5<r[control] for r in rr))


def main(revision=False,working=False):
    if working:
        from context158 import NAMES
        names_order=(*ARMS,*NAMES,'overlap_reader','overlap_A_only','overlap_wrong_B')
        primary='overlap_reader'
        destination=HERE/'working_results'
        seal=read(DATA/'WORKING_SEAL.json');pred=DATA/'WORKING_PRIVATE.npz'
    elif revision:
        from context158 import NAMES
        names_order=(*ARMS,*NAMES)
        primary='context_overlap'
        destination=HERE/'revised_results'
        seal=read(DATA/'REVISION_SEAL.json');pred=DATA/'REVISION_PRIVATE.npz'
    else:
        names_order=ARMS
        primary=MAIN
        destination=OUT
        seal=read(DATA/'PREDICTION_SEAL.json');pred=DATA/'PREDICTIONS_PRIVATE.npz' 
    require(digest(pred)==seal['sha256'] and seal['n']==1798 and not seal['target_GT_read'] and not seal['FINAL_access'],'prediction seal')
    require(digest(INPUT)==read(HERE/'experiment/DESIGN_LOCK.json')['registry_sha256'],'cohort pin')
    old=read_csv(ROOT/'research/round156/results/pair_rows.csv')
    reg=read(INPUT)
    with np.load(pred,allow_pickle=False) as f:
        p=dict(f)
    require(len(old)==len(reg)==len(p['boxes'])==1798 and list(p['arms'])==list(names_order) and np.array_equal(p['ordinal'],np.arange(1,1799)),'schema')
    rows=[]; cross_err=hist_err=0.;checks=0; n=len(names_order);names=[*names_order,*[k+'_axis' for k in names_order]]
    for i,(original,e) in enumerate(zip(old,reg)):
        rec=e['row']
        require(int(original['ordinal'])==i+1 and (original['source'],original['target'])==(e['source'],e['target']) and
                (original['has_B']=='True')==e['has_B'] and e['cohort']=='EXT' and not e['protected_alias_fallback'],'row join')
        gt=ground_truth(e['target'],rec['metapath'],rec['object_index'],rec['caption'])
        boxes=p['boxes'][i].astype(np.float64);boxes[:,3:6]=np.maximum(boxes[:,3:6],1e-6)
        axis=boxes.copy();axis[:,-1]=0
        unique,ix=np.unique(np.concatenate([boxes,axis]),return_inverse=True,axis=0)
        y=many_iou(unique,gt)
        independent=np.array([oriented_iou(b[:3],b[3:6],rotation(b[6]),gt[:3],gt[3:6],rotation(gt[6])) for b in unique])
        cross_err=max(cross_err,float(abs(y-independent).max()));checks+=len(unique)
        require(np.array_equal(y>.25,independent>.25) and np.array_equal(y>.5,independent>.5),'threshold mismatch')
        row={k:original[k] for k in META}
        for k in ('ordinal','scene_cluster','frame_cluster','point_content_cluster'):
            row[k]=int(row[k])
        row.update(has_B=e['has_B'],sampling_weight=float(original['sampling_weight']))
        row.update(zip(names,map(float,y[ix])))
        hist_err=max(hist_err,*(abs(row[key]-float(original[parent])) for key,parent in
            [('joint_zero','A_pca'),('joint_zero_axis','A_axis'),('frozen_A','A_single_pca'),('frozen_A_axis','frozen_A'),('R145','R145_pca')]))
        row['new_top1']=bool(p['indices'][i,names_order.index(primary)]!=p['indices'][i,names_order.index('joint_zero')])
        for k in ('subject_phrase','reference_edges','layout_axes','visible_boxes','covered_B_boxes','current_old_new_overlap'):
            row[k]=p[k][i].item()
        rows.append(row)
    require(hist_err<1e-7 and cross_err<1e-7,'historical or independent scorer mismatch')
    destination.mkdir(exist_ok=True)
    macro,directions,scenes,subsets=aggregates(rows,names)
    for name,records in [('all_rows',rows),('macro',macro),('direction',directions),('scene',scenes),('subset',subsets)]:
        table(destination/(name+'.csv'),records)
    scopes={'all':rows,'has_B':[r for r in rows if r['has_B']],'no_B':[r for r in rows if not r['has_B']],
            'with_reference':[r for r in rows if r['reference_edges']],
            'with_layout':[r for r in rows if r['layout_axes']]}
    scopes.update({s+'_to_'+t:[r for r in rows if (r['source'],r['target'])==(s,t)] for s,t in {(r['source'],r['target']) for r in rows}})
    pairs=[]
    for scope,rr in scopes.items():
        if rr:
            for control in names_order:
                if control!=primary:
                    pairs.append(dict(scope=scope,arm=primary,control=control,**summarize_pair(rr,primary,control)))
    table(destination/'paired.csv',pairs)
    by_overlap={}
    for label,test in [('disjoint',lambda v:v<=1e-12),('overlap_below_0.5',lambda v:1e-12<v<.5),('overlap_at_least_0.5',lambda v:v>=.5)]:
        rr=[r for r in rows if r['new_top1'] and test(r['current_old_new_overlap'])]
        delta=np.array([r[primary]-r['joint_zero'] for r in rr])
        by_overlap[label]={'n':len(rr),'better':int((delta>1e-12).sum()),'worse':int((delta< -1e-12).sum())}
    coverage=dict(roles=sum(r['subject_phrase'] for r in rows),references=sum(r['reference_edges']>0 for r in rows),
                  layout=sum(r['layout_axes']>0 for r in rows),queries_changed=sum(r['new_top1'] for r in rows),
                  positive_B_coverage=sum(r['covered_B_boxes']>0 for r in rows),overlap_on_changed=by_overlap)
    audit=dict(status='PASS',independent_geometry_comparisons=checks,max_independent_iou_error=cross_err,
               max_parent_iou_error=hist_err,ordinary_records=len(rows),has_B=sum(r['has_B'] for r in rows),
               no_B=sum(not r['has_B'] for r in rows),prediction_hash=seal['sha256'],coverage=coverage,FINAL_access=False)
    save(destination/'AUDIT.json',audit)
    print(json.dumps(audit),flush=True)

if __name__=='__main__': main('--revision' in sys.argv,'--working' in sys.argv)
