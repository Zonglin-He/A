"""R51 v2: frozen G with proposal-support groups, not semantic identities.

Source-only pure construction and readout. Whole support bags can contain
multiple components; no .6 dominance or global-sharing acceptance gate.
Original v1 files and all original G outputs remain unchanged.
"""
import numpy as np
from instances51 import DEFAULT_CONFIG,_local_components,array_sha256
from proposal_groups51 import build_proposal_groups,compatible_bags,pair_geometry


def exact(a,b,label):
 a=np.asarray(a);b=np.asarray(b)
 assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),label
def bags(offsets,indices):
 offsets=np.asarray(offsets,np.int64);indices=np.asarray(indices,np.int64)
 assert offsets.shape==(257,) and offsets[0]==0 and offsets[-1]==len(indices) and np.all(np.diff(offsets)>=0)
 return [indices[offsets[q]:offsets[q+1]] for q in range(256)]
def packed(parts):
 offsets=np.r_[0,np.cumsum([len(x) for x in parts])].astype(np.int64)
 return offsets,np.concatenate(parts).astype(np.int64) if parts else np.empty(0,np.int64)


def make_structure(*,raw_points,platform,prepared_A,prepared_receipt,cache_A,cache_B,fixed_g,fixed_components,
                   prepared_A_group_info=None,prepared_A_group_arrays=None):
 raw=np.asarray(raw_points);cfg=DEFAULT_CONFIG
 assert platform in ('drone','quad') and raw.ndim==2 and raw.shape[1]==3 and np.isfinite(raw).all()
 assert prepared_receipt['raw_points_sha256']==array_sha256(raw) and prepared_receipt['platform']==platform
 A=np.asarray(cache_A['boxes']);B=np.asarray(cache_B['boxes'])
 exact(A,prepared_A['source_boxes'],'source_A_box_bytes')
 assert A.shape==B.shape==(256,6) and np.isfinite(B).all()
 old_offsets=np.asarray(prepared_A['query_support_point_offsets']);old_indices=np.asarray(prepared_A['query_support_point_indices'])
 abags=bags(old_offsets,old_indices);componentsA=np.asarray(prepared_A['query_component_counts'],np.int64)
 if prepared_A_group_info is None:
  assert prepared_A_group_arrays is None
  groups,group_arrays=build_proposal_groups(source_boxes=A,point_offsets=old_offsets,point_indices=old_indices,component_counts=componentsA)
 else:
  assert prepared_A_group_arrays is not None
  groups=prepared_A_group_info;group_arrays=prepared_A_group_arrays
  assert groups['schema']=='R51_WHOLE_PROPOSAL_SUPPORT_EQUIVALENCE_V1'
  exact(group_arrays['source_boxes'],A,'cached_A_group_source_boxes')
  exact(group_arrays['support_offsets'],old_offsets,'cached_A_support_offsets')
  exact(group_arrays['support_indices'],old_indices,'cached_A_support_raw_ids')
  exact(group_arrays['component_counts'],componentsA,'cached_A_component_counts')
  assert group_arrays['query_group_ids'].tolist()==groups['query_group_ids']
 aid=np.asarray(groups['query_group_ids'],np.int64)
 assert aid.shape==(256,) and groups['semantic_identity_certified'] is False
 # Build B support independently in its own slot order, using the same full raw
 # coordinates. Convert unique-point indices to the original representative IDs.
 unique,first=np.unique(raw.astype(np.float64),axis=0,return_index=True);model=unique.astype(np.float32)
 if platform=='drone':model[:,2]+=np.float32(1.8)
 bbags=[];bcount=np.zeros(256,np.int64);bradius=np.full(256,np.nan);bdominant=np.zeros((256,2))
 bid=np.full(256,-1,np.int64);bstatus=[];matchcounts=np.zeros(256,np.int64)
 for q,box in enumerate(B):
  ids=(np.flatnonzero(np.all(abs(model.astype(np.float64)-box[:3])<=box[3:]/2,axis=1))
   if (box[3:]>0).all() else np.empty(0,np.int64))
  support=first[ids].astype(np.int64);bbags.append(support)
  local,radius=_local_components(model,ids,cfg);bcount[q]=len(local)
  if radius is not None:bradius[q]=radius
  if local:bdominant[q]=[len(local[0])/len(ids),len(local[1])/len(ids) if len(local)>1 else 0.]
  if len(ids)<6 or not local:bstatus.append('UNKNOWN_B_INSUFFICIENT_WHOLE_SUPPORT_OR_NO_COMPONENT');continue
  matches=[]
  for group in groups['groups']:
   if all(compatible_bags(box,support,A[j],abags[j]) for j in group['members']):matches.append(group['group_id'])
  matchcounts[q]=len(matches)
  if len(matches)==1:bid[q]=matches[0];bstatus.append('UNIQUE_SAME_POINT_SUPPORTED_PROPOSAL_GROUP_NOT_SEMANTIC_GOLD')
  elif len(matches)>1:bstatus.append('UNKNOWN_B_COMPATIBLE_WITH_MULTIPLE_A_GROUPS')
  else:bstatus.append('UNKNOWN_B_NO_COMPATIBLE_A_GROUP')
 cues=np.asarray(fixed_components,np.float64);p=np.asarray(cache_A['scores']);g=np.asarray(fixed_g,np.float64)
 assert p.shape==g.shape==(256,) and cues.shape==(3,256) and np.isfinite(g).all() and np.isfinite(cues).all()
 F=p.astype(np.float64)+2*cues[0]+2*cues[1]+2*cues[2];exact(g,F-p.astype(np.float64),'diagnostic_G_increment')
 q0=int(p.argmax());winner=int(F.argmax());positive=int(aid[winner]);negative=int(aid[q0]);reasons=[]
 if winner==q0:reasons.append('G_keeps_source_top1')
 if positive<0 or negative<0:reasons.append('UNKNOWN_A_proposal_group')
 if positive==negative:reasons.append('A_same_proposal_group')
 if not g[winner]>g[q0]:reasons.append('G_increment_does_not_prefer_replacement')
 b_offsets,b_indices=packed(bbags)
 groupsB={'groups':[{'group_id':int(group),'members':np.flatnonzero(bid==group).tolist()}
  for group in np.unique(bid[bid>=0])]}
 pairs={};overlaps={};pair_details={}
 for view,gj,offsets,indices in [('A',groups,old_offsets,old_indices),('B',groupsB,b_offsets,b_indices)]:
  diagnostic=pair_geometry(gj,positive,negative,offsets,indices)
  pairs[view]={'P':diagnostic['positive_queries'],'N':diagnostic['negative_queries']}
  pair_details[view]=diagnostic
  overlaps[view]=[diagnostic['max_positive_direction_overlap'],diagnostic['max_negative_direction_overlap']]
  reasons.extend(view+'_'+r for r in diagnostic['rejection_reasons'])
 representatives=[]
 for group in groups['groups']:
  members=group['members'];representatives.append(min(members,key=lambda j:(-float(F[j]),j)))
 arrays={'A_proposal_group_ids':aid,'B_A_proposal_group_ids':bid,
  'fixed_G_increment_A':g.copy(),'fixed_G_components_A':cues.copy(),'literal_G_scores_A':F,
  'A_support_point_offsets':old_offsets.copy(),'A_support_point_indices':old_indices.copy(),
  'B_support_point_offsets':b_offsets,'B_support_point_indices':b_indices,
  'A_component_counts':componentsA.copy(),'B_component_counts':bcount,'B_component_radius_m':bradius,
  'B_dominant_runner_fractions_diagnostic_only':bdominant,'B_A_compatible_group_counts':matchcounts,
  'A_old_dominant_runner_shared_diagnostic_only':np.asarray(prepared_A['query_dominant_runner_shared_fractions']).copy(),
  'A_fixed_G_representative_query_DIAGNOSTIC_ONLY':np.asarray(representatives,np.int64)}
 # Keep helper's own diagnostic arrays under a prefix; do not conflate them
 # with active propagation weights or use them to remove final A queries.
 for key,value in group_arrays.items():arrays['A_group_diagnostic__'+key]=value
 info={'schema':'R51_FIXED_PROPOSAL_BAG_EPISODIC_STRUCTURE_V2','accepted':not reasons,'pairs':pairs,
  'source_A_query':q0,'literal_G_A_query':winner,'positive_A_proposal_group':positive,'negative_A_proposal_group':negative,
  'rejection_reasons':reasons,'A_proposal_groups':groups,'B_query_status':bstatus,
  'A_B_PN_max_directional_shared':overlaps,'A_B_PN_pair_geometry':pair_details,'G_and_groups_frozen_before_update':True,
  'representation_repair':'PROPOSAL_BAGS_NO_G_CHANGE','semantic_identity_gold':False,
  'supervision':'LOCAL_CANDIDATE_HYPOTHESIS_GROUP_PREFERENCE_NOT_GLOBAL_OR_SEMANTIC_LABEL',
  'all_256_A_queries_remain_selectable':True,'fixed_G_representatives_are_DIAGNOSTIC_ONLY':True,
  'dominance_and_global_shared_are_not_acceptance_gates':True,'UNKNOWN_is_not_negative_evidence':True,
  'readout':'PER_A_QUERY_MEAN_OWN_NATIVE_P_WITH_SAME_GROUP_MAX_B_P_THEN_ORIGINAL_2V_2P_2R_ALL256_ARGMAX'}
 validate_structure(info,arrays,{'A':cache_A,'B':cache_B})
 return info,arrays


def validate_structure(info,arrays,caches):
 assert info['schema']=='R51_FIXED_PROPOSAL_BAG_EPISODIC_STRUCTURE_V2' and type(info['accepted']) is bool
 assert info['G_and_groups_frozen_before_update'] is True and info['semantic_identity_gold'] is False
 assert info['all_256_A_queries_remain_selectable'] is True and info['fixed_G_representatives_are_DIAGNOSTIC_ONLY'] is True
 aid=np.asarray(arrays['A_proposal_group_ids']);bid=np.asarray(arrays['B_A_proposal_group_ids'])
 assert aid.shape==bid.shape==(256,) and aid.dtype.kind==bid.dtype.kind=='i'
 cues=np.asarray(arrays['fixed_G_components_A']);p=np.asarray(caches['A']['scores'])
 F=p.astype(np.float64)+2*cues[0]+2*cues[1]+2*cues[2]
 exact(F,arrays['literal_G_scores_A'],'sealed_G_original_addition_order')
 exact(F-p.astype(np.float64),arrays['fixed_G_increment_A'],'sealed_G_diagnostic_increment')
 assert int(p.argmax())==info['source_A_query'] and int(F.argmax())==info['literal_G_A_query']
 for view,ids in [('A',aid),('B',bid)]:
  for key,group in [('P',info['positive_A_proposal_group']),('N',info['negative_A_proposal_group'])]:
   assert info['pairs'][view][key]==(np.flatnonzero(ids==group).tolist() if group>=0 else [])
 if info['accepted']:
  assert not info['rejection_reasons'] and info['positive_A_proposal_group']!=info['negative_A_proposal_group']
  assert info['positive_A_proposal_group']>=0 and info['negative_A_proposal_group']>=0
  for view in ('A','B'):
   assert info['pairs'][view]['P'] and info['pairs'][view]['N']
   assert max(info['A_B_PN_max_directional_shared'][view])<=.25
 else:assert info['rejection_reasons']
 return {'accepted':info['accepted'],'A_grouped_queries':int((aid>=0).sum()),'B_matched_queries':int((bid>=0).sum())}


def readout51(view_outputs,info,arrays):
 assert set(view_outputs)=={'A','B'}
 a=np.asarray(view_outputs['A']['probability'],np.float64);b=np.asarray(view_outputs['B']['probability'],np.float64)
 assert a.shape==b.shape==(256,) and np.isfinite(a).all() and np.isfinite(b).all()
 aid=np.asarray(arrays['A_proposal_group_ids']);bid=np.asarray(arrays['B_A_proposal_group_ids'])
 consensus=a.copy();mapped=np.zeros(256,bool)
 for group in np.unique(aid[aid>=0]):
  aq=np.flatnonzero(aid==group);bq=np.flatnonzero(bid==group)
  if len(bq):consensus[aq]=(a[aq]+b[bq].max())/2;mapped[aq]=True
 cues=np.asarray(arrays['fixed_G_components_A'],np.float64)
 scores=consensus+2*cues[0]+2*cues[1]+2*cues[2]
 return {'canonical_A_query':int(scores.argmax()),'scores':scores,'consensus_probability':consensus,
  'A_queries_receiving_B_information':int(mapped.sum()),'all_selected_boxes_from_original_A':True,
  'fixed_group_representatives_do_not_restrict_final_selection':True}
