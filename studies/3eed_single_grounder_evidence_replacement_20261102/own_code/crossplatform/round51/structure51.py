"""R51 fixed point-carrier A/B structure and same-information readout.

Pure functions: no files, media, labels, model, external evidence or GPU I/O.
Root must authorize and seal the producer before fitting. Point-supported
hypotheses are not independently verified semantic identities.
"""
import numpy as np
from instances51 import DEFAULT_CONFIG, _local_components, _compatible, array_sha256


def exact(a, b):
 a=np.asarray(a);b=np.asarray(b)
 assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def make_structure(*,raw_points,platform,prepared_A,prepared_receipt,cache_A,cache_B,fixed_g,fixed_components):
 """Bind B components to existing canonical A carriers; never add an instance.

 raw_points: original full raw XYZ, not sampled A/B or a visible-point mean.
 prepared_A: sealed original instances51 arrays (including strict IDs and
 canonical instance_point_* raw representative-index sets).
 fixed_g: sealed G F-source difference, for feedback acceptance diagnostics.
 fixed_components: original sealed V/P/R [3,256]; preserve original addition
 order source+2V+2P+2R rather than silently regroup floating-point arithmetic.
 """
 cfg=DEFAULT_CONFIG;raw=np.asarray(raw_points)
 assert platform in ('drone','quad') and raw.ndim==2 and raw.shape[1]==3 and np.isfinite(raw).all()
 assert prepared_receipt['raw_points_sha256']==array_sha256(raw) and prepared_receipt['platform']==platform
 boxesA=np.asarray(cache_A['boxes']);boxesB=np.asarray(cache_B['boxes'])
 exact(boxesA,prepared_A['source_boxes'])
 assert boxesA.shape==boxesB.shape==(256,6) and np.isfinite(boxesB).all()
 aid=np.asarray(prepared_A['query_instance_ids'],np.int64)
 offsets=np.asarray(prepared_A['instance_point_offsets'],np.int64)
 points=np.asarray(prepared_A['instance_point_indices'],np.int64)
 assert aid.shape==(256,) and offsets.ndim==1 and offsets[0]==0 and offsets[-1]==len(points)
 assert np.all(np.diff(offsets)>=cfg.minimum_component_points)
 groups=[np.sort(points[offsets[j]:offsets[j+1]]) for j in range(len(offsets)-1)]
 assert ((aid>=-1)&(aid<len(groups))).all()
 assert all(len(x)==len(np.unique(x)) and ((x>=0)&(x<len(raw))).all() for x in groups)
 unique,first=np.unique(raw.astype(np.float64),axis=0,return_index=True)
 model=unique.astype(np.float32)
 if platform=='drone':model[:,2]+=np.float32(1.8)
 membership=np.zeros(len(raw),np.int32)
 for group in groups:membership[group]+=1
 bid=np.full(256,-1,np.int64);fractions=np.zeros((256,3),np.float64)
 reasons=[];radii=np.full(256,np.nan);matched_counts=np.zeros(256,np.int64)
 for q,box in enumerate(boxesB):
  reason='UNKNOWN_insufficient_connected_support'
  if not (box[3:]>0).all():reasons.append('UNKNOWN_nonpositive_B_size');continue
  support=np.flatnonzero(np.all(np.abs(model.astype(np.float64)-box[:3])<=box[3:]/2+cfg.support_box_tolerance_m,axis=1))
  comps,radius=_local_components(model,support,cfg)
  if radius is not None:radii[q]=radius
  if comps:
   component=np.sort(first[comps[0]])
   dominant=len(comps[0])/len(support);runner=len(comps[1])/len(support) if len(comps)>1 else 0.
   shared=float(np.mean(membership[component]>1))
   fractions[q]=[dominant,runner,shared]
   if ((len(comps)>1 and len(comps[0])==len(comps[1])) or dominant<cfg.dominant_component_fraction or runner>cfg.maximum_runner_fraction):
    reason='UNKNOWN_multiple_B_components'
   elif shared>cfg.maximum_shared_fraction:reason='UNKNOWN_shared_canonical_support'
   else:
    compatible=[i for i,g in enumerate(groups) if _compatible(component,g,cfg)]
    matched_counts[q]=len(compatible)
    if len(compatible)==1:bid[q]=compatible[0];reason='UNIQUE_POINT_SUPPORTED_CANONICAL_MATCH_not_semantic_identity'
    else:reason='UNKNOWN_no_unique_canonical_match'
  reasons.append(reason)
 g=np.asarray(fixed_g,np.float64);cues=np.asarray(fixed_components,np.float64);p=np.asarray(cache_A['scores'])
 assert g.shape==p.shape==(256,) and np.isfinite(g).all() and np.isfinite(p).all()
 assert cues.shape==(3,256) and np.isfinite(cues).all()
 original_G=p.astype(np.float64)+2*cues[0]+2*cues[1]+2*cues[2]
 exact(g,original_G-p.astype(np.float64))
 q0=int(p.argmax());winner=int(original_G.argmax())
 positive=int(aid[winner]);negative=int(aid[q0]);reject=[]
 if winner==q0:reject.append('G_keeps_source_top1')
 if positive<0 or negative<0:reject.append('UNKNOWN_A_point_identity')
 if positive==negative:reject.append('A_same_point_hypothesis')
 if not g[winner]>g[q0]:reject.append('G_increment_does_not_prefer_replacement')
 pairs={}
 for view,ids in [('A',aid),('B',bid)]:
  P=np.flatnonzero(ids==positive).tolist() if positive>=0 else []
  N=np.flatnonzero(ids==negative).tolist() if negative>=0 else []
  if not P or not N:reject.append(view+'_missing_P_or_N_strict_group')
  pairs[view]={'P':P,'N':N}
 accepted=not reject
 arrays={'A_strict_instance_ids':aid.copy(),'B_canonical_instance_ids':bid,'fixed_G_increment_A':g.copy(),
  'fixed_G_components_A':cues.copy(),'literal_G_scores_A':original_G,
  'B_dominant_runner_shared_fractions':fractions,'B_component_radius_m':radii,
  'B_canonical_compatible_count':matched_counts,
  'canonical_instance_point_offsets':offsets.copy(),'canonical_instance_point_indices':points.copy()}
 info={'schema':'R51_FIXED_POINT_CARRIER_EPISODIC_STRUCTURE_V1','accepted':accepted,'pairs':pairs,
  'source_A_query':q0,'literal_G_A_query':winner,'positive_canonical_instance':positive,
  'negative_canonical_instance':negative,'rejection_reasons':reject,'B_query_status':reasons,
  'canonical_instance_count':len(groups),'A_B_query_number_identity_assumed':False,
  'point_supported_hypothesis_is_not_semantic_identity_gold':True,'G_and_groups_frozen_before_update':True,
  'canonical_A_boxes_only':True,'UNKNOWN_is_not_negative_evidence':True,
  'B_rule':{'minimum_component_points':6,'dominant_at_least':.6,'runner_at_most':.25,
   'shared_at_most':.25,'jaccard_at_least':.7,'each_coverage_at_least':.75},
  'readout':'MEAN_A_NATIVE_P_AND_SAME_CARRIER_MAX_B_NATIVE_P_IF_AVAILABLE_THEN_ORIGINAL_2V_2P_2R_ADDITION_ORDER'}
 validate_structure(info,arrays,{'A':cache_A,'B':cache_B})
 return info,arrays


def validate_structure(info,arrays,caches):
 assert info['schema']=='R51_FIXED_POINT_CARRIER_EPISODIC_STRUCTURE_V1'
 assert type(info['accepted']) is bool and info['G_and_groups_frozen_before_update'] is True
 aid=np.asarray(arrays['A_strict_instance_ids']);bid=np.asarray(arrays['B_canonical_instance_ids'])
 g=np.asarray(arrays['fixed_G_increment_A'])
 cues=np.asarray(arrays['fixed_G_components_A'])
 assert aid.shape==bid.shape==g.shape==(256,) and np.isfinite(g).all()
 assert cues.shape==(3,256) and np.isfinite(cues).all()
 assert aid.dtype.kind==bid.dtype.kind=='i' and np.all(aid>=-1) and np.all(bid>=-1)
 assert info['source_A_query']==int(np.asarray(caches['A']['scores']).argmax())
 original_G=np.asarray(caches['A']['scores']).astype(np.float64)+2*cues[0]+2*cues[1]+2*cues[2]
 exact(original_G,arrays['literal_G_scores_A']);exact(g,original_G-np.asarray(caches['A']['scores']).astype(np.float64))
 assert info['literal_G_A_query']==int(original_G.argmax())
 for view,ids in [('A',aid),('B',bid)]:
  for key,entity in [('P',info['positive_canonical_instance']),('N',info['negative_canonical_instance'])]:
   expected=np.flatnonzero(ids==entity).tolist() if entity>=0 else []
   assert info['pairs'][view][key]==expected
 if info['accepted']:
  assert not info['rejection_reasons'] and info['positive_canonical_instance']>=0 and info['negative_canonical_instance']>=0
  assert info['positive_canonical_instance']!=info['negative_canonical_instance']
  assert all(info['pairs'][v]['P'] and info['pairs'][v]['N'] for v in ('A','B'))
 else:assert info['rejection_reasons']
 return {'accepted':info['accepted'],'A_strict_count':int((aid>=0).sum()),'B_mapped_count':int((bid>=0).sum())}


def readout51(view_outputs,info,arrays):
 """Identical 0/1/3 rule; repeated B queries count once through group max."""
 assert set(view_outputs)=={'A','B'}
 a=np.asarray(view_outputs['A']['probability'],np.float64);b=np.asarray(view_outputs['B']['probability'],np.float64)
 assert a.shape==b.shape==(256,) and np.isfinite(a).all() and np.isfinite(b).all()
 aid=np.asarray(arrays['A_strict_instance_ids']);bid=np.asarray(arrays['B_canonical_instance_ids'])
 consensus=a.copy();mapped=np.zeros(256,bool)
 for group in np.unique(aid[aid>=0]):
  aq=np.flatnonzero(aid==group);bq=np.flatnonzero(bid==group)
  if len(bq):consensus[aq]=(a[aq]+b[bq].max())/2;mapped[aq]=True
 cues=np.asarray(arrays['fixed_G_components_A'],np.float64)
 scores=consensus+2*cues[0]+2*cues[1]+2*cues[2]
 query=int(scores.argmax())
 return {'canonical_A_query':query,'scores':scores,'consensus_probability':consensus,
  'A_queries_receiving_B_information':int(mapped.sum()),'all_selected_boxes_from_original_A':True}
