"""Postsealed paired-response scalars: changed content vs admitted feedback."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_dino_content_match32_v1 import D,P,verify
from scripts.score_stvg_joint_robust_correction32_v1 import paired


def stats(v):
    return dict(count=len(v),mean=float(np.mean(v))if v else None,quantiles=None if not v else np.quantile(v,[0,.5,.95,.99,1]).tolist())


def main():
    p=verify();r=read(P/'RESULTS_PUBLIC.json');rows=[];groups={}
    for ds in ['hc2','vidstg']:
        c=collections.Counter();scores={m:[]for m in ['nearest','target_high']};dscore=[];nativehigh=[];topslots=collections.Counter();maxacceptedbox=0.;maxacceptedscore=0.
        for rec in p['records']:
            if rec['dataset']!=ds:continue
            z=load(D/'predictions'/(rec['alias']+'.pt'))
            for x in z['controls']:nativehigh.append(x['native_top_score']);topslots[x['native_top_slot']]+=1
            for a,b in zip(z['matched']['nearest']['details'],z['matched']['target_high']['details']):
                for pos,x in a['anchors'].items():
                    y=b['anchors'][pos];c['references']+=1;c['content_slot_changed']+=x['slot']!=y['slot'];c['same_content_slot']+=x['slot']==y['slot']
                    c['newly_admitted']+=(not x['accepted'])and y['accepted'];c['newly_rejected']+=x['accepted']and not y['accepted']
                    c['box_changed_above_1e6']+=float(abs(x['box']-y['box']).max())>1e-6
                    c['score_increase_above_1e6']+=y['score']>x['score']+1e-6;c['score_decrease_above_1e6']+=y['score']<x['score']-1e-6
                    c['high_rejected_near_gate030_035']+=.30<=y['score']<.35
                    c['accepted_same_content']+=x['accepted']and x['slot']==y['slot']
                    scores['nearest'].append(x['score']);scores['target_high'].append(y['score']);dscore.append(y['score']-x['score'])
                    if x['accepted']or y['accepted']:
                        maxacceptedbox=max(maxacceptedbox,float(abs(x['box']-y['box']).max()));maxacceptedscore=max(maxacceptedscore,abs(x['score']-y['score']))
        rr=[x for x in r['rows']if x['dataset']==ds]
        groups[ds]=dict(counts=dict(c),native_top_slot_histogram=dict(topslots),native_high_score=stats(nativehigh),
            scores={m:stats(v)for m,v in scores.items()},score_difference=stats(dscore),
            accepted_pair_max_box_difference=maxacceptedbox,accepted_pair_max_score_difference=maxacceptedscore,
            raw_quality_high_minus_nearest_pp=paired([x['values']['target_high']['raw_delta_pp']-x['values']['nearest']['raw_delta_pp']for x in rr]))
    put(P/'PAIRED_RESPONSE_DIAGNOSTICS.json',dict(status='complete',groups=groups,
        changed_low_score_outputs_not_new_admitted_evidence=True,no_threshold_sweep=True,no_new_model_calls=True,not_identity_accuracy=True))
    print(groups)


if __name__=='__main__':main()
