"""Synthetic Drone contracts only; no model/GT/realimages/taskIoU."""
from pathlib import Path
import json,hashlib
import numpy as np
from drone_association import *
from camera_projection import CameraSpec
D=B/'research/round14/drone'
def main():
    caption='The person on the skateboard, next to another person.'
    raw=json.dumps({'status':'RESOLVED','target_phrase':'The person','target_head':'person','references':[{'phrase':'the skateboard','relation_phrase':'on'},{'phrase':'another person','relation_phrase':'next to'}]});roles=parse_text_roles(raw,caption);assert roles['status']=='RESOLVED_text_role_hypothesis'
    fenced=parse_text_roles('```json\n'+raw+'\n```',caption);assert fenced['status']=='RESOLVED_text_role_hypothesis' and fenced['format_wrapper_removed'] is True
    plainfence=parse_text_roles('```\n'+raw+'\n```',caption);assert plainfence['status']=='RESOLVED_text_role_hypothesis' and plainfence['format_wrapper_removed'] is True
    assert parse_text_roles('```json\n{"status":"UNKNOWN","status":"RESOLVED","target_phrase":"The person","target_head":"person","references":[]}\n```',caption)['status']=='NA_text_role_contract'
    for bad in ('prose\n```json\n'+raw+'\n```','```json\n'+raw+'\n```\nprose','```json\n'+raw+'\n```\n```json\n'+raw+'\n```','```json\n'+raw,'```python\n'+raw+'\n```'):
        assert parse_text_roles(bad,caption)['status']=='NA_text_role_contract'
    ca=[{'candidate_id':'C001','mask_index':0},{'candidate_id':'C002','mask_index':1},{'candidate_id':'C003','mask_index':2}];cb=[{'candidate_id':'C001','mask_index':2},{'candidate_id':'C002','mask_index':1},{'candidate_id':'C003','mask_index':0}]
    a=parse_visual_roles('{"status":"RESOLVED","target":"C001","references":["C002","C003"]}',ca,roles);b=parse_visual_roles('{"status":"RESOLVED","target":"C003","references":["C002","C001"]}',cb,roles);assert role_consensus([a,b],roles,np.asarray([1,41,1]))['selected_mask_index']==0
    vf=parse_visual_roles('```json\n{"status":"RESOLVED","target":"C001","references":["C002","C003"]}\n```',ca,roles);assert vf['status']=='RESOLVED_visual_role_hypothesis' and vf['format_wrapper_removed']
    assert parse_visual_roles('{"status":"RESOLVED","target":"C002","references":["C002","C003"]}',ca,roles)['status']=='NA_visual_role_contract'
    partial=parse_visual_roles('{"status":"RESOLVED","target":"C001","references":["C999","C003"]}',ca,roles);assert partial['status']=='RESOLVED_visual_role_hypothesis' and partial['reference_mask_indices']==[None,2] and partial['reference_unlisted_ID_answers_UNKNOWN_n']==1
    short=parse_visual_roles('{"status":"RESOLVED","target":"C001","references":["C002"]}',ca,roles);assert short['status']=='RESOLVED_visual_role_hypothesis' and short['reference_mask_indices']==[None,None] and short['reference_missing_slots_UNKNOWN_n']==1
    extra=parse_visual_roles('{"status":"RESOLVED","target":"C001","references":["C002","C003","C001"]}',ca,roles);assert extra['status']=='RESOLVED_visual_role_hypothesis' and extra['reference_mask_indices']==[None,None] and extra['reference_extra_answers_ignored_n']==1
    empty=parse_visual_roles('{"status":"RESOLVED","target":"C001","references":[]}',ca,roles);assert empty['status']=='RESOLVED_visual_role_hypothesis' and empty['reference_mask_indices']==[None,None]
    assert role_consensus([short,empty],roles,np.asarray([1,41,1]))['selected_mask_index']==0
    assert parse_visual_roles('{"status":"RESOLVED","target":"C999","references":["C002","C003"]}',ca,roles)['status']=='NA_visual_role_contract'
    assert parse_visual_roles('{"status":"RESOLVED","target":"C001","references":"C002"}',ca,roles)['status']=='NA_visual_role_contract'
    for nonstring in ({'phrase':'C001','relation_phrase':'next to','target':'C001'},None,2,True,[],{'candidate_id':'C003'}):
        invalid_slot=parse_visual_roles(json.dumps({'status':'RESOLVED','target':'C001','references':[nonstring,'C003']}),ca,roles)
        assert invalid_slot['status']=='RESOLVED_visual_role_hypothesis' and invalid_slot['reference_mask_indices']==[None,2] and invalid_slot['ignored_nonstring_reference_n']==1 and invalid_slot['nonstring_reference_contents_never_inspected']
        assert role_consensus([invalid_slot,invalid_slot],roles,np.asarray([1,41,1]))['selected_mask_index']==0
    short_nonstring=parse_visual_roles('{"status":"RESOLVED","target":"C001","references":[{}]}',ca,roles);assert short_nonstring['reference_mask_indices']==[None,None] and short_nonstring['ignored_nonstring_reference_n']==1
    assert parse_visual_roles('{"status":"RESOLVED","target":{"candidate_id":"C001"},"references":[null,null]}',ca,roles)['status']=='NA_visual_role_contract'
    assert parse_visual_roles('{"status":"RESOLVED","target":"C001","references":{"candidate_id":"C002"}}',ca,roles)['status']=='NA_visual_role_contract'

    collision=parse_visual_roles('{"status":"RESOLVED","target":"C001","references":["C999","C001"]}',ca,roles);assert collision['status']=='NA_visual_role_contract'
    conflicting=parse_visual_roles('{"status":"RESOLVED","target":"C001","references":["C003","C002"]}',ca,roles);assert role_consensus([a,conflicting],roles,np.asarray([1,41,1]))['status']=='NA_two_numberings_reference_conflict'
    for bad in ('prose\n```json\n{"status":"RESOLVED","target":"C001","references":[]}\n```','```json\n{"status":"RESOLVED","target":"C001","references":[]}\n```\n```json\n{}\n```','{"status":"RESOLVED","target":"C001","target":"C002","references":[]}','{"status":"RESOLVED","target":"C001","references":[],"extra":1}'):
        assert parse_visual_roles(bad,ca,roles)['status']=='NA_visual_role_contract'
    assert DRAFT_CONFIG.visual_role_max_new_tokens==256 and DRAFT_CONFIG.text_role_max_new_tokens==128
    assert parse_text_roles('{"status":"RESOLVED","target_phrase":"a dog","target_head":"dog","references":[]}',caption)['status']=='NA_text_role_contract'
    assert parse_text_roles('{"status":"RESOLVED","target_phrase":"The person","target_head":"car","references":[]}',caption)['status']=='NA_text_role_contract'
    assert parse_text_roles('{"status":"UNKNOWN","target_phrase":"","target_head":"","references":[]}',caption)['status']=='UNKNOWN_text_roles'
    assert parse_text_roles('{"status":"UNKNOWN","status":"RESOLVED","target_phrase":"x","target_head":"x","references":[]}',caption)['status']=='NA_text_role_contract'
    un=parse_visual_roles('{"status":"RESOLVED","target":"C001","references":["UNKNOWN","UNKNOWN"]}',ca,roles);assert un['reference_undefined_n']==2 and role_consensus([a,un],roles,np.asarray([1,41,1]))['selected_mask_index']==0
    assert role_consensus([a,b],roles,np.asarray([41,41,1]))['status']=='NA_predicted_family_conflict_not_GT'
    assert predicted_head_family('skateboard')==(41,)
    unknownref=parse_text_roles('{"status":"RESOLVED","target_phrase":"The person","target_head":"person","references":[{"phrase":"UNKNOWN","relation_phrase":"UNKNOWN"}]}',caption);assert unknownref['status']=='RESOLVED_text_role_hypothesis' and unknownref['references'][0]['phrase'] is None
    ignored=parse_visual_roles('{"status":"RESOLVED","target":"C001","references":["C002"]}',ca,unknownref);assert ignored['status']=='RESOLVED_visual_role_hypothesis' and ignored['reference_mask_indices']==[None] and ignored['model_answer_for_unknown_reference_ignored_n']==1
    repeated=parse_text_roles('{"status":"RESOLVED","target_phrase":"The man","target_head":"man","references":[{"phrase":"a car","relation_phrase":"behind"}]}','The man is behind a car and next to a car.');assert repeated['status']=='RESOLVED_text_role_hypothesis' and repeated['references'][0]['phrase'] is None
    repeated_relation=parse_text_roles('{"status":"RESOLVED","target_phrase":"The man","target_head":"man","references":[{"phrase":"a red car","relation_phrase":"next to"}]}','The man is next to a red car and next to a blue car.');assert repeated_relation['status']=='RESOLVED_text_role_hypothesis' and repeated_relation['references'][0]['phrase'] is None
    missing_ref=parse_text_roles('{"status":"RESOLVED","target_phrase":"The man","target_head":"man","references":[{"phrase":"a green car","relation_phrase":"behind"}]}','The man is behind a red car.');assert missing_ref['status']=='RESOLVED_text_role_hypothesis' and missing_ref['references'][0]['phrase'] is None
    cam=CameraSpec('drone',np.asarray([[100,0,50],[0,100,50],[0,0,1]],float),np.eye(4),np.zeros(5));masks=np.zeros((1,100,100),bool);masks[0,35:66,35:66]=True
    pts=np.asarray([[x,y,11.8] for x in (-.5,0,.5) for y in (-.5,0,.5)],np.float32);boxes=np.tile([0,0,11.8,2,2,2],(256,1)).astype(np.float32);scores=np.linspace(1,0,256,dtype=np.float32)
    def run(bb,pp=pts):return map_selected_mask_positive_support(boxes_raw6=bb,text_scores=scores,q0=0,points_model=pp,camera_spec=cam,masks_bool=masks,selected_mask_index=0)
    out=run(boxes);assert out['q_selected']==0 and out['mapping_status']=='q0_preserved_positive_support_already_compatible' and np.array_equal(out['box_raw6'],boxes[0])
    moved=boxes.copy();moved[0,:3]=[3,0,11.8];out=run(moved);assert out['q_selected']==1 and np.array_equal(out['box_raw6'],moved[1])
    ambiguous=moved.copy();ambiguous[2,3:]=[4,4,2];out=run(ambiguous);assert out['mapping_status']=='NA_multiple_proposal_groups_positive_feasible' and out['q_selected']==0
    assert run(boxes,np.repeat(pts[:1],30,axis=0))['mapping_status']=='NA_insufficient_visible_unique_mask_support'
    out=run(moved,np.repeat(pts,5,axis=0));assert out['q_selected']==1 and out['visible_mask_unique_XYZ_n']==9
    giant=np.tile([4,0,11.8,1,1,1],(256,1)).astype(np.float32);giant[1]=[0,0,11.8,8,8,8];out=run(giant);assert out['q_selected']==1 and out['larger_boxes_and_mask_contamination_can_still_pass']
    receipt={'state':'PASS_R14_DRONE_SYNTHETIC_CPU_CONTRACTS_DRAFT_NO_REALDATA','config':asdict(DRAFT_CONFIG),'program_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),B/'crossplatform/round14/drone_association.py']},'checks':['literal_main_reference_roles','duplicate_missing_invented_JSON_roles_NA','no_target_reference_mask_collision','two_numberings_physical_target_agreement','reference_UNKNOWN_not_wrong_label','q0_compatible_protection','single_positive_group_switch_only','multi_group_NA','no_box_refit','duplicateXYZ_not_independent_support','giant_box_counterexample_explicit','visual256_text128_fixed_budget','invalid_reference_ID_UNKNOWN_not_target_rejection','reference_length_mismatch_all_references_UNKNOWN_no_slot_guess','strict_target_schema_and_defined_reference_collision_conflict_unchanged','nonstring_reference_slots_UNKNOWN_no_dict_interpretation','nonlist_reference_and_target_dict_strict_reject'],'no_model_GPU_GT_pixels_or_taskIoU_called':True,'method_effectiveness_untested':True};D.mkdir(parents=True,exist_ok=True);(D/'CPU_FIXTURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['state'])
if __name__=='__main__':main()
