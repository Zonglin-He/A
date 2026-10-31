"""R14 PUBLIC-projection helpers only. No filesystem copy/build/upload API.

Never pass raw manifests/GT/caches/calls to a generic copier. Score rows must go
through the exact alias/metric whitelist below. Source code is a separate bounded
byteexact allowlist, not recursively rewritten by this JSON privacy filter.
"""
import re
import numpy as np

STUDY_PREFIX='studies/3eed_quad_regression_crossinput_transfer_and_drone_role_mapping_20261031'
INDEX_SUFFIXES=('maskindex','maskindices','maskidx','qidx','qindex','qindices',
                'queryindex','queryindices','queryidx','queryidxs','qsource','qassociation')
PRIVATE_EXACT={'id','sampleid','aid','bid','pairid','idprivate','frame','frameid','framenumber','framename',
 'scene','sceneprivate','scenename','caption','utterance','text','rawanswer','rawanswerprivate',
 'rawcall','rawcalls','source6','target6','pseudo6','asource6','atarget6','basesxyz','basexyz',
 'points','pointcloud','rawpoints','sampledpointsmodel','features','centerfeatures','sizefeatures',
 'boxesraw6','boxeseval6','armboxesraw6','armboxeseval6','sourceboxesraw6','sourceboxeseval6',
 'rawindices','physicalrawids','arawids','brawids','q0','qa','qb','qselected','selectedquery',
 'sourceq0','sourcequery','nativeindices','nativetop10','sourcenativetop10'}
DROP=object()


def normalized_key(key):return re.sub(r'[^a-z0-9]','',str(key).casefold())
def private_scalar_key(key):
 n=normalized_key(key)
 return (n in PRIVATE_EXACT or n.endswith(INDEX_SUFFIXES)
         or any(token in n for token in ('maskindex','maskindices','queryindex','queryindices','qidx','rawindices','physicalrawids'))
         or n.endswith('private') or n.endswith('path') or n.endswith('paths'))

def sanitize_scalar_tree(value):
 """Defense in depth after projection, not authorization to export arbitrary JSON.

 Case-insensitive suffix removal covers e.g nested Target_Mask_Index,
 CandidateMASK_INDICES, auxQIDX, proposalQueryIndex, arm_query_index. Absolute
 host paths are never retained. Raw arbitrary text/coordinates need upstream
 whitelist rejection; no regex can certify arbitrary free-form payloads safe.
 """
 if isinstance(value,dict):
  out={}
  for key,item in value.items():
   if private_scalar_key(key):continue
   safe=sanitize_scalar_tree(item)
   if safe is not DROP:out[key]=safe
  return out
 if isinstance(value,(list,tuple)):
  return [x for x in (sanitize_scalar_tree(v) for v in value) if x is not DROP]
 if isinstance(value,np.ndarray):return sanitize_scalar_tree(value.tolist())
 if isinstance(value,(np.integer,np.floating,np.bool_)):return sanitize_scalar_tree(value.item())
 if isinstance(value,float):
  if not np.isfinite(value):raise ValueError('Nonfinite public scalar')
  return value
 if isinstance(value,(bool,int)) or value is None:return value
 if isinstance(value,str):
  if '/home/' in value or ('/'+'Users'+'/') in value or value.startswith((('file'+':'+'//'),'/tmp/','/mnt/')):return DROP
  return value
 raise ValueError('Unsupported public scalar type')


def alias_metric_rows(rows,*,platform,expected_n,prefix,preserve_ordinal_alias=False):
 """Only task/error scalars; never emits raw IDs/queries/masks/coordinates/calls.

 Input order must be a pre-frozen source/pair order. Root binds the private join
 externally; aliases here are ordinal and disclose no original content ID.
 """
 if len(rows)!=expected_n:raise ValueError('All registered rows required; no winner filtering')
 aliases=None
 if preserve_ordinal_alias:
  aliases=[row['pair_id'] for row in rows]
  if len(set(aliases))!=expected_n or any(not isinstance(a,str) or not re.fullmatch(re.escape(prefix)+r'_[0-9]{5}',a) for a in aliases):
   raise ValueError('Preserved alias must be unique five-digit public ordinal; raw A/B IDs forbidden')
 scenes=sorted({str(r.get('scene',r.get('scene_PRIVATE',''))) for r in rows})
 scene_alias={scene:'scene_'+str(i+1).zfill(2) for i,scene in enumerate(scenes)}
 output=[]
 for i,row in enumerate(rows):
  out={'case_alias':aliases[i] if aliases is not None else prefix+'_'+str(i+1).zfill(3),'platform':platform,
       'scene_alias':scene_alias[str(row.get('scene',row.get('scene_PRIVATE','')))],'arms':{}}
  for key in ('A_nonzero_signal','A_gate','case_role','cohort'):
   if key in row:out[key]=row[key]
  for name,arm in row['arms'].items():
   selected={key:arm[key] for key in ('iou','Acc25','Acc50','query_changed','actual_changed','actual_changed_axis6','actual_changed_axes',
       'raw_nonpositive_size_count','raw_nonpositive_sizes','status','center_shift_L2_m','size_shift_L2_m') if key in arm}
   errors=arm.get('coordinate_errors',{})
   selected['coordinate_errors']={key:errors[key] for key in ('center_L2_m','size_L2_m','abs_error6_m') if key in errors}
   out['arms'][name]=selected
  output.append(sanitize_scalar_tree(out))
 return output


def project_protocol(plan):
 """Explicit reusable numerical/scientific settings, never private path bindings."""
 keys=('stage','expected_pairs','expected_samples','strict_FP32_flags','fractions',
 'parameter_kernel_atol_m','A_fit_coordinate_atol_m','fit_full6D','same_object_or_known_ego_transform_claimed',
 'allow_fixed_prefix2_engineering','seed','torch_num_threads','sentinel','noun_fallback',
 'source_checkpoint_sha256','source_training_plan_sha256','source_model_state_fingerprint',
 'source_no_G_or_GT','no_source_retraining_or_teacher_EMA','no_source_retraining',
 'formal_preflight_policy','expected_Drone_n','expected_A_nonzero_n','drone_config','legacy_association_config',
 'expected_cases','formal_expected_new_calls','formal_expected_reused_calls','model_dtype','attn_implementation',
 'Quad_A_count','Quad_B_count','Quad_B_pool_count','Quad_pairing_salt',
 'Quad_B_image_and_point_each_distinct_from_every_A',
 'Quad_same_target_or_different_instance_correspondence_not_established',
 'Quad_B_R12_historical_legacy_G_and_scoring_exposure_explicit_current_R13G_not_constructed_for_B',
 'Quad_B_no_G_or_RGB_decode_or_training_or_gate_in_R14','Drone_old_count','Drone_recent_count',
 'Drone_recent_excluded_content_groups','Drone_recent_available',
 'all5912_historically_model_and_GT_scored_in_R7_R8',
 'new_does_not_mean_pristine_input_label_or_scene_holdout')
 out={k:plan[k] for k in keys if k in plan}
 config_keys={
  'drone_config':('minimum_visible_mask_unique_points','positive_box_distance_tolerance_m',
   'minimum_positive_support_fraction','semantic_predicted_family_conflict_reject',
   'max_distinct_reference_entities','text_role_max_new_tokens','visual_role_max_new_tokens'),
  'legacy_association_config':('dedup_aabb_iou','mapping_2D_iou_min','mapping_visible_point_min',
   'mapping_iou_runnerup_gap_min','max_independent_masks','candidate_order_salt',
   'min_pixels','max_pixels','max_new_tokens','seed')}
 for name,fields in config_keys.items():
  if name in out:out[name]={k:plan[name][k] for k in fields if k in plan[name]}
 if 'Quad_B_no_G_or_RGB_decode_or_training_or_gate_in_R14' in out:
  out['B_input_isolation_scope']='Algorithm runner only: B RGB does not enter G, fit or gates; an independent authorized data audit decoded172 images and point contents to verify distinctness.'
 return sanitize_scalar_tree(out)
