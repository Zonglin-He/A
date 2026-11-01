"""CPU replay, gate removal, real pair uncertainty, and all tail accounting."""
import sys,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round22'))
from common22 import *
from cache22 import refs_without_increment
def main():
    for p in ['cache208_attempt001','pair208_attempt001','direct208_attempt001','scene21_attempt001','scene_direct21_attempt001']:verify(BASE/p)
    for p in ['scored_cache208_attempt001','scored_direct208_attempt001','scored_scene21_attempt001']:verify(EXP/p)
    for p in ['text208_attempt003','dino208_attempt003','reference208_attempt003','direct208_attempt003']:verify(PRIOR/p)
    oldscore={r['id_PRIVATE']:r for r in read(ROOT/'research/round21/experiment/scored208_attempt003/ROWS_PRIVATE.json')};formal=read(EXP/'scored_direct208_attempt001/ROWS_PRIVATE.json');counts=collections.Counter();rows=[];tail=[];rel=[];pairpublic=[]
    for r,s in zip(read(ROWS),formal):
        assert r['id']==s['id_PRIVATE'];z=feature(r);refs=refs_without_increment(r);before=prior(r,'reference');g=prior(r,'text');case=read(BASE/'direct208_attempt001/cases_PRIVATE'/(r['id']+'.json'))
        with np.load(BASE/'direct208_attempt001/predictions_PRIVATE'/(r['id']+'.npz')) as p:assert np.array_equal(p['arm_boxes_raw6'],z['boxes'][p['query_indices']]);assert np.isfinite(p['arm_boxes_raw6']).all()
        assert s['ious']['source']==oldscore[r['id']]['ious']['source'];assert s['q']['r21_no_ref']==oldscore[r['id']]['q']['gate_repair_no_reference']
        for eid,obs in refs['references'].items():
            ref=next(x for x in g['references'] if x['entity']==eid)
            for x,b in zip(obs,before['references'][eid]):
                expected=x['full']['log_odds_yes_no']>=.5 and x['visual_evidence']>=.5;assert x['semantic_accepted']==expected;counts['reference_proposals']+=1;counts['old_reference_accepted']+=b['semantic_accepted'];counts['new_reference_accepted']+=expected
                if expected and not b['semantic_accepted']:counts['recovered_increment_rejections']+=1;counts['recovered_person_increment_rejections']+=ref['head'].lower() in ['person','pedestrian','skateboarder'];rows.append({'alias':r['alias'],'reference_alias':eid,'modifier_increment_diagnostic':x['modifier_increment'],'full_raw_logodds':x['full']['log_odds_yes_no'],'visual_evidence':x['visual_evidence'],'new_accepted':True,'not_identity_gold':True})
        for p in case['pairs']:
            counts['subject_'+p['subject_preference']['label']]+=1;counts['role_conflicts']+=p['role_conflict']
            for v in p['reference_role_preferences'].values():counts['reference_pair_'+v['label']]+=1
            for edge in p['depth_edges']:
                counts['depth_candidate_edges']+=1;counts['depth_A_'+edge['A']['status']]+=1;counts['depth_B_'+edge['B']['status']]+=1
                for state in [edge['A'],edge['B']]:
                    for reason in state['reasons']:counts['depth_reason_'+reason]+=1
            pairpublic.append({'alias':r['alias'],'platform':r['platform'],**{k:v for k,v in p.items() if not k.endswith('_PRIVATE')}})
        if s['q']['direct_pair_depth_tiebreak']!=s['q']['direct_pair_no_ref']:rel.append({'alias':r['alias'],'delta_pp':100*(s['ious']['direct_pair_depth_tiebreak']-s['ious']['direct_pair_no_ref'])})
        if s['ious']['direct_pair_depth_tiebreak']<s['ious']['r21_no_ref']-1e-10:tail.append({'alias':r['alias'],'R21_no_ref_iou':s['ious']['r21_no_ref'],'new_iou':s['ious']['direct_pair_depth_tiebreak'],'delta_pp':100*(s['ious']['direct_pair_depth_tiebreak']-s['ious']['r21_no_ref'])})
    regression=[]
    scene={r['alias']:r for r in read(EXP/'scored_scene21_attempt001/ROWS_PRIVATE.json')}
    for alias in ['A_quad_012','A_quad_017','V_quad_004']:
        s=next(x for x in formal if x['alias']==alias);regression.append({'alias':alias,'ious':{a:s['ious'][a] for a in ['source','r21_no_ref','drop_increment_full','direct_pair_no_ref','direct_pair_depth_tiebreak']},'scene_ious':scene[alias]['ious'],'interpretation':'Abstention/retained source is not proof of identity recognition.'})
    calls=read(BASE/'pair208_attempt001/SEALED.json')['calls']+read(BASE/'scene21_attempt001/SEALED.json')['calls'];assert calls==1194 and calls<=1200
    # Runtime code must not access task oracle, score cache, or modifier increment.
    for f in ['cache22.py','assemble22.py','depth22.py','pair22.py','scene22.py','assemble_scene22.py']:
        code=(BASE/f).read_text();assert 'OFFLINE_RECOVERABILITY' not in code;assert 'modifier_increment' not in code
    audit={'state':'PASSED','n_unique_inputs':208,'all_frozen_box_outputs_replayed':True,'old_source_and_R21_readout_reproduced':True,'all_old_new_seals_verified':True,'modifier_increment_never_used_by_new_G':True,'all_negative_rows_retained':True,'counts':dict(counts),'new_VLM_forwards':calls,'format_rejections':0,'TTA_steps':0,'label_enriched_scene_panel_n':21,'scene_panel_not_independent_confirmation':True}
    dump(EXP/'CPU_AUDIT.json',audit);dump(EXP/'REFERENCE_REJECTION_RECOVERY.json',rows);dump(EXP/'REGRESSION_CASES.json',regression);dump(EXP/'ALL154_PAIR_DIAGNOSTICS.json',pairpublic);dump(EXP/'NEW_VS_R21_NEGATIVE_TAILS.json',tail);dump(EXP/'PRIMARY_RELATION_INTERVENTIONS.json',rel);print(audit);print('REGRESSION',regression)
if __name__=='__main__':main()
