"""R15 synthetic-only contracts, no real scene/GT/model/image/taskIoU."""
from pathlib import Path
import copy,hashlib,json
import numpy as np
from semantic_graph import *
D=Path(__file__).resolve().parents[2]/'research/round15/association'
def obj(mid,x,cls=3,depth=10.,border=False):return {'mask_id':mid,'predicted_class_id':cls,'mask_centroid_norm2':[x,.5],'mask_bbox_norm4':[x-.04,.4,x+.04,.6],'mask_extent_norm2':[.08,.2],'mask_centroid_defined':True,'mask_bbox_defined':True,'touches_image_border':border,'depth_camera':{'defined':True,'n':12,'median':depth,'q10':depth-.1,'q90':depth+.1}}
def raw_graph(caption,target='the black car',refs=None,attrs=None,rels=None):
 return json.dumps({'status':'RESOLVED','target':{'quote':target,'head':'car'},'references':refs or [],'attributes':attrs or [],'relations':rels or []})
def main():
 caption='the black car on the left of the image'
 raw=raw_graph(caption,rels=[{'id':'L1','subject':'T','object':'SCENE','quote':'left','predicate':'GLOBAL_LEFT','frame':'camera_image','viewpoint_quote':'the image'}]);g=parse_caption_graph(raw,caption);assert g['status']=='PARSED_text_graph'
 scene={'objects':[obj('M00000',.2),obj('M00001',.8)]};v={'defined':True,'role_proposals':{'T':['M00000']},'attribute_claims':{}}
 out=match_scene_graph(g,scene,v);assert out['status']=='unique_supported' and out['selected_target_mask_id']=='M00000' and out['search_complete'];json.dumps(out)
 # Without the independent relation, sparse VLM proposals still contain full
 # relation info. The control must say exactly that, not no-language-relation.
 off=match_scene_graph(g,scene,v,relation_check=False);assert off['selected_target_mask_id'] is None and off['relation_mode']=='disabled' and off['search_complete']
 assert proposal_consensus(v)['status']=='proposal_control_unique' and not proposal_consensus(v)['semantic_allmask_uniqueness_claimed']
 flipped=match_relation_flip_diagnostic(g,scene,v);assert flipped['relation_mode']=='flip_diagnostic' and flipped['selected_target_mask_id'] is None and flipped['flipped_relation_ids']==['L1']
 only_left={'objects':[obj('M00000',.2)]};one_off=match_scene_graph(g,only_left,v,relation_check=False);assert one_off['status']=='unique_supported' and not one_off['literal_relations_satisfied']
 unsupported=copy.deepcopy(g);unsupported['relations'][0]['predicate']='NEAR';uf=match_relation_flip_diagnostic(unsupported,scene,v);original=match_scene_graph(unsupported,scene,v);assert not uf['flip_applied'] and uf['status']==original['status'] and uf['selected_target_mask_id']==original['selected_target_mask_id']
 ambiguous={'objects':[obj('M00000',.2),obj('M00001',.25)]};a=match_scene_graph(g,ambiguous,v);assert a['status']=='ambiguous_possible_targets' and a['selected_target_mask_id'] is None and a['possible_target_mask_ids']==['M00000','M00001']
 partial={'objects':[obj('M00000',.2),obj('M00001',.8,border=True)]};assert match_scene_graph(g,partial,v)['selected_target_mask_id'] is None
 badview=copy.deepcopy(g);badview['relations'][0]['viewpoint_quote']=None;assert match_scene_graph(badview,scene,v)['selected_target_mask_id'] is None
 badsign=copy.deepcopy(g);badsign['relations'][0]['predicate']='GLOBAL_RIGHT';assert relation_evidence(badsign['relations'][0],scene['objects'][0],None)['status']==UNKNOWN
 # Nested target/reference spans are legal; references are distinct roles, not
 # auto-invalidated merely because the main descriptive phrase contains them.
 nested='the car beside the white car in the image'
 ng=parse_caption_graph(raw_graph(nested,target='the car beside the white car',refs=[{'entity':'R1','quote':'the white car','head':'car','distinct_from_target':True}]),nested);assert ng['status']=='PARSED_text_graph' and ng['entities'][1]['status']=='PARSED'
 assert parse_caption_graph(raw_graph('the car and the car',target='the car'),'the car and the car')['status']=='NA_text_graph_contract'
 # A full quoted attribute actually vetoes a proposed mask if both visual
 # orders contradict; one missing order is UNKNOWN and cannot become support.
 ag=copy.deepcopy(g);ag['attributes']=[{'id':'A1','entity':'T','kind':'color','quote':{'text':'black','span':[4,9]},'status':'PARSED'}]
 av=copy.deepcopy(v);av['attribute_claims'][('A1','M00000')]=CONTRADICT;assert match_scene_graph(ag,scene,av)['selected_target_mask_id'] is None
 av['attribute_claims'][('A1','M00000')]=SUPPORT;assert match_scene_graph(ag,scene,av)['selected_target_mask_id']=='M00000'
 av['attribute_claims'][('A1','M00000')]=UNKNOWN;assert match_scene_graph(ag,scene,av)['selected_target_mask_id'] is None
 ca=[{'candidate_id':'C001','mask_id':'M00000'},{'candidate_id':'C002','mask_id':'M00001'}];cb=list(reversed([{'candidate_id':'C001','mask_id':'M00001'},{'candidate_id':'C002','mask_id':'M00000'}]))
 vr1=parse_visual_graph('{"status":"RESOLVED","roles":[{"entity":"T","candidates":["C001"]}],"attribute_checks":[{"attribute":"A1","candidate":"C001","evidence":"SUPPORT"}]}',ca,ag)
 vr2=parse_visual_graph('{"status":"RESOLVED","roles":[{"entity":"T","candidates":["C002"]}],"attribute_checks":[]}',cb,ag);cv=combine_visual_graphs([vr1,vr2]);assert cv['role_proposals']['T']==['M00000'] and cv['attribute_claims'][('A1','M00000')]==UNKNOWN
 # Camera image pair direction and camera depth are separately observed.
 rel={'status':'PARSED','quote':{'text':'left of'},'viewpoint_quote':{'text':'in the image'},'frame':'camera_image','predicate':'LEFT_OF'};assert relation_evidence(rel,obj('M00000',.2),obj('M00001',.8))['status']==SUPPORT
 dr={'status':'PARSED','quote':{'text':'closer to the viewer'},'viewpoint_quote':{'text':'the viewer'},'frame':'observer_depth','predicate':'NEARER_OBSERVER'};assert relation_evidence(dr,obj('M00000',.5,depth=5),obj('M00001',.5,depth=10))['status']==SUPPORT
 dr['predicate']='FRONT_OF';assert relation_evidence(dr,obj('M00000',.5,depth=5),obj('M00001',.5,depth=10))['status']==UNKNOWN
 negated=copy.deepcopy(rel);negated['quote']={'text':'not left of'};assert relation_evidence(negated,obj('M00000',.2),obj('M00001',.8))['status']==UNKNOWN
 undefined=obj('M00000',.2);undefined['mask_centroid_defined']=False;assert relation_evidence(rel,undefined,obj('M00001',.8))['status']==UNKNOWN
 # Exceeding the fully known finite search budget cannot yield uniqueness.
 cap=match_scene_graph(g,scene,v,config=SemanticConfig(max_assignments=0));assert cap['status']=='NA_assignment_search_cap' and not cap['search_complete'] and cap['selected_target_mask_id'] is None
 for bad in ('prefix '+raw,'```json\n'+raw+'\n```\nprose','{"status":"UNKNOWN","status":"RESOLVED","target":{},"references":[],"attributes":[],"relations":[]}'):
  assert parse_caption_graph(bad,caption)['status']=='NA_text_graph_contract'
 receipt={'state':'PASS_R15_SEMANTIC_GRAPH_SYNTHETIC_CPU_ONLY_UNTESTED_METHOD','checks':['nested_role_quotes_allowed','literal_distinctive_main_required','global_position_interval_acts','pair_image_direction_acts','explicit_observer_depth_only','ambiguous_front_behind_UNKNOWN','sameclass_ambiguity_retained','VLM_omitted_mask_not_pruned','UNKNOWN_reference_attribute_not_pass','full_attribute_contradiction_veto','two_order_missing_claim_unknown','truncated_mask_centroid_unknown','finite_search_cap_NA','relationoff_allmask_search_preserved_no_required_relation_support','proposal_control_does_not_claim_allmask_unique','flip_whitelist_diagnostic_only','JSON_no_raw_tuple_keys','strict_JSON_fence_duplicate_prose_reject','bbox4_separate_from_widthheight2','undefined_position_and_negated_relation_unknown'],'config':asdict(DEFAULT_CONFIG),'program_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('semantic_graph.py'))},'actual_model_GPU_GT_scores_media_or_new_scene_read':False,'method_effectiveness':'untested'};D.mkdir(parents=True,exist_ok=True);(D/'CPU_FIXTURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['state'])
if __name__=='__main__':main()
