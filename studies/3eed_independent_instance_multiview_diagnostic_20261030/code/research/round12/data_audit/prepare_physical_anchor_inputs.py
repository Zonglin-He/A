"""R12 146 content-disjoint anchors and counter-neighbor inputs. No GT/media decode."""
from pathlib import Path
import collections,hashlib,json
B=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
SALT='R12_independent_instance_temporal_v1';WRONG='R12_wrongscene_reference_v1';KEYS={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'}
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();h=lambda s:hashlib.sha256(s.encode()).hexdigest()
def save(name,value):
 p=D/name;assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n');return p

def main():
 population=B/'research/round8/experiment/full5912_gpu_inputs.json';full=read(population);fm={r['id']:r for r in full};assert len(fm)==5912 and all(set(r)==KEYS for r in full)
 hi=D/'PHYSICAL_INPUT_HASH_INDEX_PRIVATE.json';hashed=read(hi)['rows_PRIVATE'];hm={r['id_PRIVATE']:r for r in hashed};assert set(hm)==set(fm)
 mi=D/'TARGET5912_INPUT_FIELDS_PRIVATE.json';metas={r['meta_path_PRIVATE']:r['input_fields_PRIVATE'] for r in read(mi)['rows_PRIVATE']}
 prior_sources={'R9_160':'research/round9/experiment/diagnostic160_gpu_inputs.json','R10_160':'research/round10/input_prepare/diagnostic160_gpu_inputs.json','R11_320':'research/round11/input_prepare/diagnostic320_gpu_inputs.json','R8_40':'research/round8/experiment/smoke40_gpu_inputs.json','R7_20':'research/round7/experiment/smoke_gpu_inputs.json'};old=[r for p in prior_sources.values() for r in read(B/p)];assert len(old)==len({r['id'] for r in old})==700;excluded={hm[r['id']]['physical_group'] for r in old};assert len(excluded)==588
 groups=collections.defaultdict(list)
 for x in hashed:groups[x['physical_group']].append(x)
 canonical={g:min(xs,key=lambda r:r['id_PRIVATE']) for g,xs in groups.items()}
 scene_keys=sorted({(x['platform'],x['scene_PRIVATE']) for x in hashed});assert len(scene_keys)==5
 quota={key:(18 if key[0]=='drone' and sum((x['platform'],x['scene_PRIVATE'])==key and g not in excluded for g,x in canonical.items())==18 else 32) for key in scene_keys}
 assert sorted(quota.values())==[18,32,32,32,32]
 save('history/PATH_ONLY_V1_PROPOSAL.json',{'state':'initial_path_only160_proposal_not_executed_or_GPUfrozen','recorded_after_inputduplicate_discovery':True,'requested32_per_scene':True,'initial_exclusion':'700unique path/ID cases','why_superseded':'Path uniqueness does not imply unique RGB+point observation; exactbyte duplicates confirmed before new source/mask inference.','not_a_retrospective_immutable_pre_discovery_registration':True})
 selected=[];availability=[]
 for key in scene_keys:
  candidates=[x for g,x in canonical.items() if (x['platform'],x['scene_PRIVATE'])==key and g not in excluded];ordered=sorted(candidates,key=lambda x:(h(SALT+'|'+x['id_PRIVATE']),x['id_PRIVATE']));take=ordered[:quota[key]];assert len(take)==quota[key];selected+=take;availability.append({'platform':key[0],'scene_PRIVATE':key[1],'available_content_groups':len(candidates),'actual_quota':quota[key]})
 assert len(selected)==len({r['physical_group'] for r in selected})==146 and not {r['physical_group'] for r in selected}&excluded
 anchors=[fm[x['id_PRIVATE']] for x in selected];assert all(set(r)==KEYS for r in anchors)
 # Counter->content is one-to-one in each released targetscene. This is an
 # observed mapping, not certification of absolute time or sensor sync.
 byscene=collections.defaultdict(dict)
 for g,x in canonical.items():
  key=(x['platform'],x['scene_PRIVATE']);t=x['input_timestamp'];assert t not in byscene[key] or byscene[key][t]['physical_group']==g;byscene[key][t]=x
 group_choices={};used=set(x['physical_group'] for x in selected)
 for x in selected:
  key=(x['platform'],x['scene_PRIVATE']);timeline=byscene[key];t=x['input_timestamp'];before=[v for k,v in timeline.items() if k<t];after=[v for k,v in timeline.items() if k>t];prev=max(before,key=lambda v:v['input_timestamp']) if before else None;nxt=min(after,key=lambda v:v['input_timestamp']) if after else None
  wrongs=[v for v in canonical.values() if v['platform']==x['platform'] and v['scene_PRIVATE']!=x['scene_PRIVATE']];wrong=min(wrongs,key=lambda v:(h(WRONG+'|'+x['id_PRIVATE']+'|'+v['physical_group']),v['physical_group'])) if wrongs else None
  group_choices[x['physical_group']]={'previous':prev,'next':nxt,'wrongview':wrong}
  for v in (prev,nxt,wrong):
   if v:assert v['physical_group']!=x['physical_group'];used.add(v['physical_group'])
 ordered_used=sorted([canonical[g] for g in used],key=lambda x:(x['platform'],x['scene_PRIVATE'],x['input_timestamp'],x['physical_group']));aliases={x['physical_group']:f'input_frame_{i+1:05d}' for i,x in enumerate(ordered_used)}
 frames={}
 for x in ordered_used:
  rr=fm[x['id_PRIVATE']];fields=metas[rr['metapath']];alias=aliases[x['physical_group']]
  frames[alias]={'platform':rr['platform'],'scene_PRIVATE':rr['scene'],'pointpath':rr['pointpath'],'metapath':rr['metapath'],'imagepath':rr['imagepath'],'image_sha256':x['image_sha256'],'point_sha256':x['point_sha256'],'metadata_sha256':sha(rr['metapath']),'physical_group_sha256':x['physical_group'],'timestamp':fields['timestamp'],'folder_index':x['folder_index'],'pose':fields['pose'],'image_intrinsic':fields['image_intrinsic'],'image_extrinsic':fields['image_extrinsic'],'camera_distortion':fields.get('camera_distortion'),'distortion_missing':fields.get('camera_distortion') is None,'sequence_name':fields.get('sequence_name'),'original_input_image_path':fields.get('image_path'),'original_input_lidar_path':fields.get('lidar_path_proj'),'pose_semantics':'released_input_pose_unverified_sensor_world_direction_provenance_not_objectboxGT','model_to_raw_note':'undo_source_z_plus1.8_before_raw_pose_or_camera' if rr['platform']=='drone' else 'modelXYZ_equals_rawXYZ'}
 mask=[{'frame_alias':a,'imagepath':f['imagepath']} for a,f in frames.items()];assert all(set(x)=={'frame_alias','imagepath'} for x in mask)
 pairs=[];gaps=collections.Counter();na=collections.Counter()
 for x in selected:
  rr=fm[x['id_PRIVATE']];choices=group_choices[x['physical_group']];rec={'anchor_id_PRIVATE':rr['id'],'anchor_frame_alias':aliases[x['physical_group']],'platform':rr['platform'],'scene_PRIVATE':rr['scene'],'anchor_timestamp_counter':x['input_timestamp']}
  for role,v in choices.items():
   if v is None:rec[role]=None;na[role]+=1;continue
   dt=v['input_timestamp']-x['input_timestamp'];rec[role]={'frame_alias':aliases[v['physical_group']],'counter_delta':dt,'absolute_counter_gap':abs(dt),'within5_counter_steps':bool(abs(dt)<=5) if role!='wrongview' else None,'different_content_proved':True,'same_scene':v['scene_PRIVATE']==x['scene_PRIVATE'],'physical_time_seconds_or_sync_verified':False}
   if role!='wrongview':gaps[str(abs(dt))]+=1
  pairs.append(rec)
 plan=save('INPUT_PLAN_LOCK.json',{'state':'R12_content_disjoint146_anchor_and_input_neighbor_plan_before_model_execution','selection_salt':SALT,'wrongview_salt':WRONG,'original_requested_n':160,'actual_n':146,'actual_platform_n':{'drone':50,'quad':96},'actual_scene_quotas_PRIVATE':availability,'input_validity_revision':'Deduplicate exact(imageJPEG SHA,pointfile SHA) tuples, exclude all588 contentgroups of prior700 pathcases. Droneplaza has18remaining so no32 fabrication; other4scenes retain32.','canonical_row':'min original opaque R8 input ID within each exact-content group; one group one hash ticket, no caption/color/GT scoring selection.','rank':'SHA256(salt|canonical_id), then canonical_id, within scene','source_population_sha256':sha(population),'physical_input_index_sha256':sha(hi),'input_metadata_table_sha256':sha(mi),'prior_manifest_sha256':{k:sha(B/p) for k,p in prior_sources.items()},'grouping_and_neighbor_scope':'All5912 targetinput paths may provide content-distinct companions, including historically exposed frames; onlyanchors exclude prior700 content. Nearestprev/next among unique provided timestamp counters in same scene; no numericdirectory adjacency or synchronization claim.','counter_gap_policy':'Nearest available counter candidate retained with actualgap+within5 flag; missingboundary NA, no arbitrary replacement. Any Gvalidity/temporalwindow rejection requires separately preregistered validation.','wrongview_rule':'Same platform another scene minSHA256(wrongsalt|anchor_canonicalID|candidate_contentgroup), no taskbenefit/color/GT matching.','statistical_units':'146distinct byte input pairs, D50/Q96 withDscene32/18. Primary allanchor micro plus sceneequal summary; not146independentdomains, nor unseen research targetVAL.','program_sha256':sha(__file__),'no_GT_annotation_score_pixel_pointarray_model_GPU_or_external_rawtrajectory_access':True})
 anchorpath=save('ANCHOR146_GPU_INPUTS.json',anchors);framepath=save('FRAME_INDEX_PRIVATE.json',frames);maskpath=save('MASK_FRAME_INPUTS.json',mask)
 pairpath=save('PAIRS_PRIVATE.json',{'state':'R12_inputonly_timestamp_counter_companion_candidates_not_verified_temporal_correspondence','anchor_count':146,'unique_frame_count':len(frames),'input_plan_sha256':sha(plan),'anchor_manifest_sha256':sha(anchorpath),'frame_index_sha256':sha(framepath),'rows_PRIVATE':pairs,'gap_histogram':dict(sorted(gaps.items(),key=lambda x:float(x[0]))),'missing_roles':dict(na),'no_task_GT_or_imagecontent_matching_used':True})
 receipt=save('INPUT_RECEIPT.json',{'state':'R12_actual146_content_disjoint_anchors_and_candidate_reference_inputs_sealed','anchor_n':146,'by_platform':{'drone':50,'quad':96},'unique_frontend_frames':len(frames),'input_plan_sha256':sha(plan),'anchor_manifest_sha256':sha(anchorpath),'pairs_sha256':sha(pairpath),'frame_index_sha256':sha(framepath),'mask_frame_inputs_sha256':sha(maskpath),'all_anchor_9keys_values_original':all(fm[r['id']]==r for r in anchors),'prior700_ID_overlap':len({r['id'] for r in anchors}&{r['id'] for r in old}),'prior700_exact_content_overlap':len({r['physical_group'] for r in selected}&excluded),'anchor_unique_content_groups':146,'companion_counter_gap_histogram':dict(gaps),'companion_missing_roles':dict(na),'companion_frames_may_overlap_prior_data_but_never_samecontent_as_ownanchor':True,'no_model_or_mask_extraction_run':True})
 print(json.dumps({'anchor_n':146,'frame_n':len(frames),'plan_sha256':sha(plan),'anchor_sha256':sha(anchorpath),'pairs_sha256':sha(pairpath),'frame_index_sha256':sha(framepath),'mask_inputs_sha256':sha(maskpath),'receipt_sha256':sha(receipt),'gaps':dict(gaps),'missing':dict(na)}))
if __name__=='__main__':main()
