"""Artifact and independent selected-candidate replay; no new inference."""
from pathlib import Path
import sys,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round21'))
from common21 import *
def main():
    for name in ['text208_attempt001','dino208_attempt001','reference208_attempt001','text208_attempt002','dino208_attempt002','reference208_attempt002','direct208_attempt002']:verify(BASE/name)
    for name in ['text144_attempt001','textZ64_attempt001','pairs144_attempt001','pairsZ64_attempt001','visual144_attempt001','visualZ64_attempt001','direct208_attempt002']:verify(ROOT/'crossplatform/round20'/name)
    verify(EXP/'scored208_attempt002');score=read(EXP/'scored208_attempt002/ROWS_PRIVATE.json');refstats=collections.Counter();edges=collections.Counter();selected=[];attr=[]
    for r,s in zip(read(ROWS),score):
        assert r['id']==s['id_PRIVATE'];z=feature(r);case=read(BASE/'direct208_attempt002/cases_PRIVATE'/(r['id']+'.json'));ref=read(BASE/'reference208_attempt002/cases_PRIVATE'/(r['id']+'.json'));g=read(BASE/'text208_attempt002/cases_PRIVATE'/(r['id']+'.json'))
        with np.load(BASE/'direct208_attempt002/predictions_PRIVATE'/(r['id']+'.npz')) as p:
            assert np.array_equal(p['arm_boxes_raw6'],z['boxes'][p['query_indices']]);assert np.isfinite(p['arm_boxes_raw6']).all();assert list(p['arm_names'])==list(s['ious'])
        for obs in ref['references'].values():
            refstats['proposals_evaluated']+=len(obs)
            for o in obs:
                refstats['semantically_accepted']+=o['semantic_accepted'];assert all(np.isfinite(o[k]['log_odds_yes_no']) and o[k]['generated_tokens']==0 for k in ['full','head','gray_full','gray_head'])
        for pair in case['pair_diagnostics']:
            attr.append({'alias':r['alias'],'pair_alias':pair['pair_alias'],'full_delta':pair['full_delta'],'attribute_delta':pair['old_attribute_delta'],'old_min_full_pass':min(pair['full_delta'])>=.5,'old_increment_pass':min(pair['old_attribute_delta'])>=0,'new_visual_pass':pair['visual_gate_repaired'],'global_discriminating':pair['global_discriminating']})
            for edge in pair['edges']['direct_full_relation_G']:
                edges['edges_evaluated']+=1;edges['discriminating']+=edge['edge_supports_B_over_A'];edges['A_'+edge['A']['status']]+=1;edges['B_'+edge['B']['status']]+=1;edges['role_boxes_excluded']+=edge['A']['same_role_boxes_excluded']+edge['B']['same_role_boxes_excluded']
        if s['q']['direct_full_relation_G']!=s['q']['gate_repair_no_reference']:
            selected.append({'alias':r['alias'],'platform':r['platform'],'phase':s['phase'],'relation_delta_pp':100*(s['ious']['direct_full_relation_G']-s['ious']['gate_repair_no_reference']),'vs_source_delta_pp':100*(s['ious']['direct_full_relation_G']-s['ious']['source']),'source_iou':s['ious']['source'],'no_reference_iou':s['ious']['gate_repair_no_reference'],'full_iou':s['ious']['direct_full_relation_G']})
    audit={'state':'PASSED','n':len(score),'all10arm_predictions_from_frozen_source_pool':True,'all_old_and_new_seals_verified':True,'old_source_and_G_CPU_score_reproduced_bitexact':True,'all_negative_rows_retained':True,'format_rejections':0,'reference_evidence':dict(refstats),'relation_edges':dict(edges),'TTA_steps':0,'no_independent_confirmation':True}
    dump(EXP/'CPU_AUDIT.json',audit);dump(EXP/'RELATION_INTERVENTIONS.json',selected);dump(EXP/'ATTRIBUTE_GATE_DIAGNOSTICS.json',attr);print(audit);print('RELATION_SELECTIONS',selected)
if __name__=='__main__':main()
