"""Post-seal task/coverage mechanism diagnostics, never runtime supervision."""
from pathlib import Path
import sys,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round21'),str(ROOT/'research/round5/audit')]
from common21 import *
from independent_core_v2 import transform_gt,candidate_ious
def main():
    rows=read(EXP/'scored208_attempt003/ROWS_PRIVATE.json');index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};stats=collections.Counter();oracle=[];tails=[];gate={}
    for r,s in zip(read(ROWS),rows):
        g=read(BASE/'text208_attempt003/cases_PRIVATE'/(r['id']+'.json'));d=read(BASE/'dino208_attempt003/cases_PRIVATE'/(r['id']+'.json'));ref=read(BASE/'reference208_attempt003/cases_PRIVATE'/(r['id']+'.json'));prep=old(r,'pairs');case=read(BASE/'direct208_attempt003/cases_PRIVATE'/(r['id']+'.json'))
        if prep['pairs']:
            stats['real_mapped_pair_cases']+=1
            for rr in g['references']:
                stats['reference_queries']+=1;stats['reference_own_modifiers']+=rr['has_role_attributes'];stats['wall_graffiti_compositions']+=rr['head'].lower() in ['wall','walls'] and 'graffiti' in rr['phrase'].lower();stats['isolated_graffiti_heads']+=rr['head'].lower()=='graffiti';stats['role_described_person_references']+=rr['head'].lower() in ['person','pedestrian','skateboarder'] and rr['has_role_attributes'];stats['reference_has_stable_proposal']+=bool(d['objects'][rr['entity']]['stable_indices']);stats['reference_has_semantically_accepted_proposal']+=any(x['semantic_accepted'] for x in ref['references'].get(rr['entity'],[]))
            for edge in g['reference_relations']:stats['edge_frame_'+edge['frame_provenance']]+=1
        if s['q']['direct_full_relation_G']!=s['q']['gate_repair_no_reference']:
            ix=index[r['id']];gt=transform_gt(read(ix['meta_path'])['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));z=feature(r);b=z['boxes'].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);ious=candidate_ious(b,gt)[0];offered=[prep['q0_PRIVATE']]+[v['qB_PRIVATE'] for v in prep['pairs']];oracle.append({'alias':r['alias'],'all256_oracle_iou':float(ious.max()),'offered_pair_oracle_iou':float(ious[offered].max()),'all256_any25':bool((ious>.25).any()),'GT_only_postseal_diagnostic':True})
        if s['ious']['gate_repair_no_reference']<s['ious']['r20_fullG']-1e-10:tails.append({'alias':r['alias'],'source_iou':s['ious']['source'],'R20_iou':s['ious']['r20_fullG'],'no_reference_iou':s['ious']['gate_repair_no_reference'],'full_relation_iou':s['ious']['direct_full_relation_G']})
    for platform in ['drone','quad']:
        rr=[s for s in rows if s['platform']==platform];delta=np.array([s['ious']['gate_repair_no_reference']-s['ious']['r20_fullG'] for s in rr]);gate[platform]={'changed':sum(s['q']['gate_repair_no_reference']!=s['q']['r20_fullG'] for s in rr),'gains':int((delta>1e-10).sum()),'harms':int((delta< -1e-10).sum()),'mean_gain_pp':float(delta.mean()*100)}
    dump(EXP/'MECHANISM_DIAGNOSIS.json',{'counts':dict(stats),'attribute_gate_repair_vs_R20':gate,'same208_development':True,'no_front_back_depth_operator':True,'qualitative_review_not_identity_gold':True});dump(EXP/'ZERO_IOU_RELATION_CANDIDATE_ORACLE.json',oracle);dump(EXP/'ATTRIBUTE_REPAIR_NEGATIVE_TAILS.json',tails);print('MECHANISM21',dict(stats),gate,oracle)
if __name__=='__main__':main()
