"""Seal all matched32 before offline GT; candidate pairs are not independent."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from scripts.run_stvg_dino_content_match32_v1 import D,P,verify
from scripts.score_stvg_joint_robust_correction32_v1 import paired
from vg_tta.stvg_error_state_correction_v1 import encode,decode
MODES=['nearest','target_high']


def mean(v):
    v=[float(x)for x in v if x is not None];return float(np.mean(v))if v else None


def quality_iou(b,truth,row,ds):
    from vg_tta.tastvg_oracle_event5_v1 import box_iou
    b=np.asarray(b,float);xy=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
    if ds=='hc2':xy=np.maximum(xy,0)
    return float(box_iou(xy[None],np.array(truth)[None])[0])


def score():
    from scripts.score_stvg_opd_p1_v1 import truths
    import torch
    p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all_outputs_before_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    gtall={d:truths(d,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==d]))for d in ['hc2','vidstg']};torch.set_num_threads(4)
    plans={d:read(ROOT/f'artifacts/decota_paper_experiments_v1/{d}/PLAN.json')['rows']for d in gtall}
    rows=[];private=[];started=time.perf_counter()
    for r in p['records']:
        z=load(D/'predictions'/(r['alias']+'.pt'));e=load(ROOT/r['prior_prediction']);b=np.asarray(e['outputs']['Source_no_correction'],float)
        ds=r['dataset'];row=plans[ds][r['query_ordinal']];gt=gtall[ds][0][r['query_ordinal']];scale=np.array(p['scale'][ds]);counts={m:collections.Counter()for m in MODES};records=[]
        for k,action in enumerate(z['actions']):
            refs=decode(encode(b)+action*scale)
            for pos in z['planned_positions']:
                fid=int(z['frame_ids'][pos]);known=fid in gt;initial=quality_iou(refs[pos],gt[fid],row,ds)if known else None
                rec=dict(sample=k,position=pos,known_GT=known,input_iou=initial,arms={})
                for m in MODES:
                    d=z['matched'][m]['details'][k];v=d['anchors'].get(pos);c=counts[m];c['planned_anchor_refs']+=1;c['known_GT_refs']+=known;c['returned_refs']+=v is not None;c['accepted_refs']+=bool(v and v['accepted']);c['known_accepted_refs']+=bool(known and v and v['accepted'])
                    raw=quality_iou(v['box'],gt[fid],row,ds)if known and v else initial
                    gated=raw if v and v['accepted']else initial
                    rec['arms'][m]=dict(returned=v is not None,accepted=bool(v and v['accepted']),raw_iou=raw,gated_iou=gated,
                        score=None if v is None else v['score'],slot=None if v is None else v['slot'])
                    if known and v and v['accepted']:
                        c['accepted_improve']+=raw>initial+1e-10;c['accepted_harm']+=raw<initial-1e-10;c['accepted_neutral']+=abs(raw-initial)<=1e-10
                records.append(rec)
        values={}
        for m in MODES:
            dd=z['matched'][m]['details'];known=[x for x in records if x['known_GT']];accepted=[x for x in known if x['arms'][m]['accepted']]
            counts[m].update(candidate_count=len(dd),legal_candidates=sum(x['legal']for x in dd),candidates_with_feedback=sum(x['accepted_anchors']>0 for x in dd))
            values[m]=dict(coverage=dict(counts[m]),query_with_feedback=any(x['accepted_anchors']for x in dd),
                gated_delta_pp=100*mean([x['arms'][m]['gated_iou']-x['input_iou']for x in known])if known else 0.,
                raw_delta_pp=100*mean([x['arms'][m]['raw_iou']-x['input_iou']for x in known])if known else 0.,
                accepted_input_iou=mean([x['input_iou']for x in accepted]),accepted_output_iou=mean([x['arms'][m]['raw_iou']for x in accepted]),
                unknown_if_no_known_GT=not bool(known),score_mean=mean([x['arms'][m]['score']for x in records if x['arms'][m]['score']is not None]))
        rowout=dict(alias=r['alias'],dataset=ds,active=z['active'],values=values,compute=z['compute'],nearest_prior_checks=z['nearest_prior_checks'],
            known_anchor_positions=sum(int(z['frame_ids'][pos])in gt for pos in z['planned_positions']))
        rows.append(rowout);private.append(dict(alias=r['alias'],dataset=ds,records=records))
    groups={};passed=True
    for ds in gtall:
        rr=[r for r in rows if r['dataset']==ds];g={}
        for m in MODES:
            c=collections.Counter()
            for r in rr:c.update(r['values'][m]['coverage'])
            g[m]=dict(coverage=dict(c),queries_with_feedback=sum(r['values'][m]['query_with_feedback']for r in rr),
                gated_delta=paired([r['values'][m]['gated_delta_pp']for r in rr]),raw_delta=paired([r['values'][m]['raw_delta_pp']for r in rr]))
        contrast=paired([r['values']['target_high']['gated_delta_pp']-r['values']['nearest']['gated_delta_pp']for r in rr])
        gate=(g['target_high']['coverage'].get('accepted_refs',0)>g['nearest']['coverage'].get('accepted_refs',0)and g['target_high']['gated_delta']['mean']>0 and contrast['mean']>0)
        g.update(high_minus_nearest_gated_delta=contrast,gate_pass=bool(gate),queries=16,active=sum(r['active']for r in rr));passed &=gate;groups[ds]=g
    decision=dict(status='qualified_for_bounded_repaired_OPD'if passed else 'stop_content_variant_before_OPD',pass_for_OPD=bool(passed),
        gate_prelocked=True,automatic_expansion=False,no_new_threshold_or_content_search=True,independent_confirmation=False)
    save(D/'SCORED_ANCHOR_ROWS_PRIVATE.pt',private)
    put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,decision=decision,
        query_level_aggregation=True,unknown_GT_not_failure=True,unsupported_queries_zero_correction=True,all_outputs_before_new_GT=True,
        multiple_references_not_independent_samples=True,OPD_executed=False))
    put(P/'DECISION.json',decision);put(D/'CPU_COMPLETION.json',dict(status='complete',count=32,seconds=time.perf_counter()-started,result_sha256=sha(P/'RESULTS_PUBLIC.json')))
    print(dict(groups=groups,decision=decision))


if __name__=='__main__':score()
