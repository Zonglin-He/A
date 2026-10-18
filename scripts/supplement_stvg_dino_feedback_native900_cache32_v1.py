"""Matching original32 cached native900 records only; no model or GT access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.diagnose_stvg_dino_feedback_cache32_v1 import D,P,PRE,verify,read_guard,export
from vg_tta.stvg_dino_feedback_cache32_v1 import iou,quant,slotbin,public_value


def main():
    p=verify();read_guard();other=read(ROOT/'artifacts/stvg_spatial_readout32_v1/PLAN_LOCK.json');wanted={(r['dataset'],r['query_ordinal']):r for r in p['records']}
    source=[]
    for r in other['records']:
        if (r['dataset'],r['query_ordinal'])in wanted:
            source.append((r,wanted[(r['dataset'],r['query_ordinal'])]))
    lock=dict(status='metadata_match_locked_before_extended_comparison',count=len(source),count_not_added_to_cohort=True,
        code_sha256=sha(Path(__file__)),native_files=[dict(path=str((ROOT/'artifacts/stvg_spatial_readout32_v1/predictions'/(r['alias']+'.pt')).relative_to(ROOT)),sha256=sha(ROOT/'artifacts/stvg_spatial_readout32_v1/predictions'/(r['alias']+'.pt')))for r,_ in source],
        original_diagnosis_sha256=sha(P/'DIAGNOSIS.json'),no_model_or_GT_or_media=True)
    export(D/'SUPPLEMENTAL_INPUT_LOCK.json',lock)
    rawrows=load(D/'CACHED_SCALAR_ROWS.pt')['anchors'];rows=[];qual=[]
    for older,current in source:
        raw=load(ROOT/'artifacts/stvg_spatial_readout32_v1/predictions'/(older['alias']+'.pt'));s=load(ROOT/current['prior_Source'])
        assert list(raw['frame_ids'])==list(s['frame_ids'])
        assert sorted(raw['raw'])==s['expert']['positions4']
        for pos,record in raw['raw'].items():
            a=raw['old_expert']['observations'][('original',pos)];b=s['expert']['observations'][('original',pos)]
            assert all(a['receipt'][k]==b['receipt'][k]for k in ['text','rgb_sha256','inputs','target_span'])
            assert np.array_equal(a['probe']['boxes'],b['probe']['boxes'])and np.array_equal(a['probe']['target_scores'],b['probe']['target_scores'])and a['probe']['candidate_ids']==b['probe']['candidate_ids']
            det=record['detection'];assert record['context']
            scores=np.asarray(det['target_all_scores'],float);boxes=np.asarray(det['all_boxes'],float);assert scores.shape==(900,)and boxes.shape==(900,4)
            kval=np.asarray(b['probe']['candidate_ids'],int);assert np.array_equal(scores[kval],np.asarray(b['probe']['target_scores'],float))
            qual.append(dict(alias=current['alias'],dataset=current['dataset'],position=pos,RGB_text_input_tensors_target_span_and_probe_exact=True,native_queries=900,native_score_ge035_count=int((scores>=.35).sum()),native_max_score=float(scores.max())))
            z=load(PRE/'predictions'/(current['alias']+'.pt'));base=np.asarray(load(ROOT/current['prior_prediction'])['outputs']['Source_no_correction'],float)
            scale=np.asarray(read(PRE/'PLAN_LOCK.json')['scale'][current['dataset']]);chart=np.c_[base[:,:2],np.log(base[:,2:])]
            for arm,name in [('One_at_E','One_correction'),('Fixed24','Fixed_sample_distill'),('Dynamic3','On_policy_distill')]:
                for st in z['fits'][name]['rounds']:
                    detail=st['details'][:1]if arm=='One_at_E'else st['details']
                    for k,d in enumerate(detail):
                        if not d['legal']:continue
                        v=d['anchors'][pos];lat=chart[pos]+d['action']*scale;ref=np.r_[lat[:2],np.exp(lat[2:])];overlap=iou(ref,boxes)[0]
                        slot=int(v['slot']);near=int(overlap.argmax());mask=scores>=.35
                        rows.append(dict(alias=current['alias'],dataset=current['dataset'],arm=arm,round=int(st['round']),sample=k,position=pos,
                            selected_slot=slot,original_selected_slot_score=float(scores[slot]),conditional_score=float(v['score']),conditional_accepted=bool(v['accepted']),
                            input_selected_native_final_iou=float(overlap[slot]),nearest_native_final_slot=near,nearest_native_final_score=float(scores[near]),nearest_native_final_iou=float(overlap[near]),
                            input_high_score_native_max_iou=float(overlap[mask].max())if mask.any()else None,
                            original_selected_slot_ge035=bool(scores[slot]>=.35),conditional_minus_original_selected_slot_score=float(v['score']-scores[slot])))
    groups={}
    for arm in ['One_at_E','Fixed24','Dynamic3']:
        rr=[x for x in rows if x['arm']==arm];conf=[x for x in rr if x['original_selected_slot_ge035']];weak=[x for x in rr if not x['original_selected_slot_ge035']]
        highnear=[x for x in rr if x['input_high_score_native_max_iou']is not None and x['input_high_score_native_max_iou']>=.75]
        groups[arm]=dict(anchor_references=len(rr),selected_slot_original_native_score=quant([x['original_selected_slot_score']for x in rr]),
            conditional_score=quant([x['conditional_score']for x in rr]),selected_slot_original_ge035=len(conf),
            originally_high_score_slot_conditional_pass=sum(x['conditional_accepted']for x in conf),
            originally_low_score_slot_conditional_pass=sum(x['conditional_accepted']for x in weak),
            conditional_minus_original_selected_score=quant([x['conditional_minus_original_selected_slot_score']for x in rr]),
            input_overlap_confident_native_final_ge075=len(highnear),
            among_those_selected_slot_original_low=sum(not x['original_selected_slot_ge035']for x in highnear),
            among_those_conditional_pass=sum(x['conditional_accepted']for x in highnear))
    export(P/'MATCHED_NATIVE900_CASE.json',dict(status='cached_metadata_and_input_exact',matched_queries=len(source),matched_frames=len(qual),qualifications=qual,
        groups=groups,rows=rows,no_extra_cohort_queries_read=True,
        initial_proposals_and_content_vectors_not_available=True,same_reference_content_swap_not_available=True,
        interpretation='One matching current-cohort case only; native final outputs recover selected-slot target support before reanchoring. Not a causal content intervention, not an identity label, not evidence that swapping to a high-score slot would work.'))
    print(dict(qualifications=qual,groups=groups))


if __name__=='__main__':main()
