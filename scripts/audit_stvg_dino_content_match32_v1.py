"""Independent scalar reconstruction of matched control and quality evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_content_match32_v1 import D,P,verify,PRE
from scripts.score_stvg_joint_robust_correction32_v1 import paired


def main():
    p=verify();result=read(P/'RESULTS_PUBLIC.json');private=load(D/'SCORED_ANCHOR_ROWS_PRIVATE.pt');checks=collections.Counter();rows={r['alias']:r for r in result['rows']}
    for r,annot in zip(p['records'],private):
        z=load(D/'predictions'/(r['alias']+'.pt'));old=load(PRE/'predictions'/(r['alias']+'.pt'));assert z['GT_read']is False and z['pretrained_weights_frozen_exact']
        if z['active']:
            assert np.array_equal(z['actions'],old['fits']['Fixed_sample_distill']['rounds'][0]['actions']);checks['exact_action_arrays']+=1
        assert z['active']==old['active']
        for control in z['controls']:assert len(control['matched_controls']['mask_sha256'])==64;checks['matched_frame_controls']+=1
        for m in ['nearest','target_high']:
            c=collections.Counter();rr=annot['records'];d=z['matched'][m]['details'];v=rows[r['alias']]['values'][m]
            for row in rr:
                q=row['arms'][m];pos=row['position'];k=row['sample'];raw=d[k]['anchors'].get(pos)
                assert q['accepted']==bool(raw and raw['score']>=.35);checks['admission_checks']+=1
                if raw:
                    other=z['matched']['target_high'if m=='nearest'else'nearest']['details'][k]['anchors'][pos]
                    assert raw['mask_slot']==other['mask_slot'];checks['fixed_mask_slots']+=1
                    if m=='nearest':assert raw['slot']==raw['mask_slot']
                    else:assert raw['slot']==next(x['native_top_slot']for x in z['controls']if x['position']==pos)
                    if m=='nearest':
                        prev=old['fits']['Fixed_sample_distill']['rounds'][0]['details'][k]['anchors'][pos]
                        assert abs(raw['score']-prev['score'])<1e-5 and abs(raw['box']-prev['box']).max()<1e-5;checks['prior_output_matches']+=1
                if not row['known_GT']:assert q['raw_iou']is None and q['gated_iou']is None
                else:assert q['gated_iou']==(q['raw_iou']if q['accepted']else row['input_iou'])
            known=[x for x in rr if x['known_GT']]
            delta=np.mean([x['arms'][m]['gated_iou']-x['input_iou']for x in known])*100 if known else 0.
            assert abs(delta-v['gated_delta_pp'])<1e-12;checks['query_quality_means']+=1
            assert v['coverage'].get('accepted_refs',0)==sum(x['arms'][m]['accepted']for x in rr)
            assert v['coverage'].get('returned_refs',0)==sum(x['arms'][m]['returned']for x in rr)
    for ds,g in result['groups'].items():
        rr=[r for r in result['rows']if r['dataset']==ds]
        for m in ['nearest','target_high']:
            for field,key in [('gated_delta','gated_delta_pp'),('raw_delta','raw_delta_pp')]:
                assert paired([r['values'][m][key]for r in rr])==g[m][field];checks['independent_bootstrap_summaries']+=1
        assert paired([r['values']['target_high']['gated_delta_pp']-r['values']['nearest']['gated_delta_pp']for r in rr])==g['high_minus_nearest_gated_delta']
        expected=g['target_high']['coverage'].get('accepted_refs',0)>g['nearest']['coverage'].get('accepted_refs',0)and g['target_high']['gated_delta']['mean']>0 and g['high_minus_nearest_gated_delta']['mean']>0
        assert expected==g['gate_pass'];checks['prelocked_gate']+=1
    assert result['decision']['pass_for_OPD']==all(g['gate_pass']for g in result['groups'].values())
    put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),exact_actions_and_fixed_attention=True,
        prior_nearest_replication_tolerance=1e-5,score_threshold=.35,all_signed_zero_unknown_retained=True,private_scored_rows_sha256=sha(D/'SCORED_ANCHOR_ROWS_PRIVATE.pt')))
    print(dict(status='pass',checks=dict(checks)))


if __name__=='__main__':main()
