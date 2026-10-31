"""Post-both-branches-seal Drone64 CPU scoring. Never called before root release.
Old32/recent32/all64 descriptive,7original-query arms. No model/GPU/optimizer.
"""
from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,math,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('R14_Drone_reused_R13_scoring_helpers',ROOT/'research/round13/evaluation/score_direct.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
candidate_ious=prior.candidate_ious;transform_gt=prior.transform_gt;coordinate_errors=prior.coordinate_errors;eval_boxes=prior.eval_boxes;mean=prior.mean;signed_stats=prior.signed_stats;paired_scene=prior.paired_scene
GEOMETRY_DEPENDENCIES=prior.DEPENDENCIES
ARMS=('source','old_full_order0','old_full_consensus','old_noun_consensus','semantic_only','mapping_only','combined')
COHORTS=('old_R13_Drone32_development','recent_content_disjoint32_historically_exposed','all64_descriptive_union')
sha=prior.sha;read=prior.read;dump=prior.dump;member=prior.member;bind_map=prior.bind_map
REQUIRED_NPZ=('sample_id','case_alias','case_role','arm_names','arm_query_index','arm_boxes_raw6','arm_status','source_boxes_raw6','source_text_scores256','q_source','actual_changed_axes')

def verify_artifact(artifact,programs):
 if set(artifact)!={'directory','seal_sha256'}:raise ValueError('sealed_artifacts require exact directory/seal_sha256')
 base=Path(artifact['directory']);base=base if base.is_absolute() else ROOT/base
 if sha(base/'SEALED.json')!=artifact['seal_sha256']:raise ValueError('Artifact seal hash mismatch')
 seal=read(base/'SEALED.json')
 if not isinstance(seal.get('files'),dict) or not seal['files']:raise ValueError('Empty file allowlist')
 for rel,h in seal['files'].items():
  if sha(member(base,rel))!=h:raise ValueError('Artifact member mismatch')
 producer=seal.get('program_sha256',{})
 if isinstance(producer,dict):
  for rel,h in producer.items():
   if sha(member(ROOT,rel))!=h:raise ValueError('Producer map mismatch')
 elif isinstance(producer,str):
  if producer not in programs.values():raise ValueError('String producer lacks explicit program binding')
 else:raise ValueError('Invalid producer proof')
 return base,seal

def validate_release_header(release):
 if not release.get('CPU_SCORE_RELEASE') or not release.get('both_branches_complete_and_sealed') or release.get('stage')!='R14_both_branches_complete_before_target_GT':raise ValueError('Both-branch complete root CPU release required')
 if release.get('expected_Drone_n')!=64:raise ValueError('Drone64 scope required')
 if release.get('geometry_dependency_sha256')!=GEOMETRY_DEPENDENCIES:raise ValueError('Exact frozen metric/transform/OBB dependencies required')
 for k in ('quad_prediction_dir','quad_prediction_seal_sha256','drone_prediction_dir','drone_prediction_seal_sha256','drone_manifest_sha256','drone_cases_sha256','scoring_index_sha256','drone_scorer_sha256','source_checkpoint_path','source_checkpoint_sha256','program_bindings','sealed_artifacts'):
  if k not in release:raise ValueError('Missing root release field '+k)

QUAD_ARMS=('source','direct_additive_Aresidual','direct_size_normalized_Aresidual','parameter_half','kernel_half','parameter_full','kernel_full')
def validate_quad_final(seal):
 if seal.get('state')!='sealed_R14_Quad76_Aonly_transfer7arms_before_B_GT' or seal.get('n_pairs')!=76 or seal.get('n_arms')!=7 or tuple(seal.get('arm_names',()))!=QUAD_ARMS:raise ValueError('Formal Quad76sevenarm sibling seal required; engineering is ineligible')
 if not seal.get('source_state_restored') or not seal.get('full_B_forward_check_done') or not seal.get('all_B_fixed_own_source_q') or seal.get('B_in_loss_gate_LR_or_G') is not False:raise ValueError('Quad independent-B/state restoration proof required')

def behavioral_scalars(cj,visuals):
 if len(visuals)!=2:raise ValueError('Two role call receipts required')
 pp=[v['parsed'] for v in visuals];valid=all(p['status']=='RESOLVED_visual_role_hypothesis' for p in pp)
 agree=pp[0]['target_mask_index']==pp[1]['target_mask_index'] if valid else None
 conflict=any(p.get('reason')=='target shares a mask with a distinct reference' for p in pp)
 sem=cj['semantic_consensus'];old=cj['old_full_consensus'];newmask=sem.get('selected_mask_index');oldmask=old.get('selected_mask_index') if old.get('two_numberings_same_physical_mask') else None
 out={'text_role_status':cj['text_roles']['status'],'text_target_quote_resolved':cj['text_roles']['status']=='RESOLVED_text_role_hypothesis','two_numbering_target_agreement_defined':valid,'two_numbering_target_agreement':agree,'distinct_reference_collision_observed':conflict,'defined_reference_order_conflict':sem['status']=='NA_two_numberings_reference_conflict','semantic_status':sem['status'],'semantic_mask_defined':newmask is not None,'predicted_family_conflict_rejected':sem['status']=='NA_predicted_family_conflict_not_GT','subject_family_unsupported_unknown':sem['status']=='NA_subject_head_family_not_supported','mask_choice_comparison_defined':newmask is not None and oldmask is not None,'mask_choice_changed_vs_old':newmask!=oldmask if newmask is not None and oldmask is not None else None,'old_controls_strictly_reused':bool(cj['old_control_reused']),'role_counts_are_model_behaviors_not_subject_accuracy':True}
 out['ignored_unresolved_reference_ID_answers']=sum(int(p.get('model_answer_for_unknown_reference_ignored_n',0)) for p in pp)
 out['unknown_reference_slots_defined_call_n']=sum(p['status']=='RESOLVED_visual_role_hypothesis' for p in pp)
 out['unknown_reference_slots_sum']=sum(int(p.get('reference_undefined_n',0)) for p in pp)
 for label,key in (('mapping_only','mapping_only'),('combined','combined')):
  d=cj[key];out[label+'_mapping_status']=d['mapping_status'];out[label+'_visible_support_n']=d.get('visible_mask_unique_XYZ_n');out[label+'_feasible_group_n']=d.get('feasible_group_n');out[label+'_q0_positive_feasible']=d.get('source_q0_positive_feasible');out[label+'_q0_preserved_by_positive_support']=d['mapping_status']=='q0_preserved_positive_support_already_compatible';out[label+'_unique_group_readout']=d['mapping_status']=='single_positive_feasible_proposal_group_source_prior_readout';out[label+'_mapping_NA']=d['mapping_status'].startswith('NA_')
 return out

def unlabelled_gate(a):
 if sha(a.release)!=a.release_sha256:raise ValueError('Root release byte hash mismatch')
 r=read(a.release);validate_release_header(r)
 for key,path in [('drone_prediction_seal_sha256',a.predictions/'SEALED.json'),('drone_manifest_sha256',a.manifest),('drone_cases_sha256',a.cases),('scoring_index_sha256',a.scoring_index),('drone_scorer_sha256',Path(__file__))]:
  if sha(path)!=r[key]:raise ValueError('Release binding mismatch '+key)
 if Path(r['drone_prediction_dir']).resolve()!=a.predictions.resolve():raise ValueError('Drone directory identity mismatch')
 for path,h in GEOMETRY_DEPENDENCIES.items():
  if sha(ROOT/path)!=h:raise ValueError('Frozen geometry dependency changed')
 if sha(ROOT/'research/round13/evaluation/score_direct.py')!=r['program_bindings'].get('research/round13/evaluation/score_direct.py'):raise ValueError('Reused R13 helper source must be explicit')
 bind_map(r['program_bindings'],'program_bindings')
 for label in ('input_bindings','prompt_bindings'):
  if label in r:bind_map(r[label],label)
 if sha(Path(r['source_checkpoint_path']))!=r['source_checkpoint_sha256']:raise ValueError('Original source checkpoint changed')
 artifacts={}
 for item in r['sealed_artifacts']:
  base,s=verify_artifact(item,r['program_bindings']);key=base.resolve()
  if key in artifacts:raise ValueError('Duplicate sealed directory')
  artifacts[key]=s
 for prefix in ('quad','drone'):
  path=Path(r[prefix+'_prediction_dir']).resolve()
  if path not in artifacts or sha(path/'SEALED.json')!=r[prefix+'_prediction_seal_sha256']:raise ValueError('Both final prediction artifacts must be byte-verified')
 validate_quad_final(artifacts[Path(r['quad_prediction_dir']).resolve()])
 se=artifacts[a.predictions.resolve()]
 if se['state']!='sealed_R14_Drone64_seven_direct_arms_before_GT' or se['n_cases']!=64 or tuple(se['arm_names'])!=ARMS or se['new_model_calls']!=320 or se['reused_old_R13_calls']!=128 or not se['model_state_unchanged'] or not se['no_parameter_update_GT_or_newbox']:raise ValueError('Drone complete formal64 contract mismatch')
 manifest=read(a.manifest);cr=read(a.cases);cases={c['id']:c for c in cr}
 if len(manifest)!=64 or len(cases)!=64 or len({x['id'] for x in manifest})!=64 or {x['id'] for x in manifest}!=set(cases):raise ValueError('Complete64 inputs required')
 arrays={};behaviors={};cohorts=collections.Counter()
 for row in manifest:
  sid=row['id'];case=cases[sid];alias=case['case_alias'];cohort=COHORTS[0] if case.get('legacy_case_alias') is not None else COHORTS[1]
  if row['platform']!='drone' or row['scene']!=case['scene_PRIVATE']:raise ValueError('Drone input-only case join mismatch')
  row=dict(row,case_role=cohort,case_alias=alias);cohorts[cohort]+=1
  p=a.predictions/'cases_PRIVATE'/(alias+'.npz');j=a.predictions/'cases_PRIVATE'/(alias+'.json')
  for f in (p,j):
   rel=str(f.relative_to(a.predictions))
   if rel not in se['files'] or sha(f)!=se['files'][rel]:raise ValueError('Unsealed Drone member')
  with np.load(p,allow_pickle=False) as z:
   if not set(REQUIRED_NPZ)<=set(z.files):raise ValueError('Actual Drone NPZ schema mismatch')
   d={k:z[k].copy() for k in REQUIRED_NPZ}
  if str(d['sample_id'])!=sid or str(d['case_alias'])!=alias or str(d['case_role'])!=cohort or tuple(d['arm_names'])!=ARMS:raise ValueError('Sample/role/arms mismatch')
  raw=d['arm_boxes_raw6'];pool=d['source_boxes_raw6'];scores=d['source_text_scores256'];qs=d['arm_query_index'];q=int(d['q_source'])
  if raw.shape!=(7,6) or raw.dtype!=np.float32 or pool.shape!=(256,6) or pool.dtype!=np.float32 or scores.shape!=(256,) or qs.shape!=(7,) or not np.issubdtype(qs.dtype,np.integer) or not np.isfinite(raw).all() or not np.isfinite(pool).all() or not np.isfinite(scores).all() or np.any((qs<0)|(qs>=256)) or q!=int(np.argmax(scores)):raise ValueError('Native source/querypool numerical contract')
  if not np.array_equal(raw,pool[qs]) or int(qs[0])!=q or not np.array_equal(d['actual_changed_axes'],raw!=raw[0]):raise ValueError('All7 outputs must be exact original sourcepool boxes')
  sd=Path(case['source_dir']);sd=sd if sd.is_absolute() else ROOT/sd;md=Path(case['mask_dir']);md=md if md.is_absolute() else ROOT/md
  if sd.resolve() not in artifacts or md.resolve() not in artifacts:raise ValueError('Every old/new source/mask must be verified sealed artifact')
  srcseal=artifacts[sd.resolve()];cp=sd/'source_cache'/(sid+'.npz')
  if not srcseal['source_unchanged'] or not srcseal['source_repeat_bitexact'] or srcseal['source_checkpoint_sha256']!=r['source_checkpoint_sha256'] or sha(cp)!=srcseal['files'][str(cp.relative_to(sd))]:raise ValueError('Immutable source cache proof missing')
  with np.load(cp,allow_pickle=False) as z:
   if not np.array_equal(pool,z['boxes_raw6'][0]) or not np.array_equal(scores,z['text_scores'][0]) or q!=int(z['query_index']):raise ValueError('Drone pool not exact immutable sourceforward')
  cj=read(j);vc=[]
  for order in range(2):
   f=a.predictions/'calls_PRIVATE'/(alias+'__new_visual_roles_order'+str(order)+'.json')
   if sha(f)!=se['files'][str(f.relative_to(a.predictions))]:raise ValueError('Rolecall receipt unsealed')
   vc.append(read(f))
  behaviors[sid]=behavioral_scalars(cj,vc);arrays[sid]=(row,d)
 if dict(cohorts)!={COHORTS[0]:32,COHORTS[1]:32}:raise ValueError('All old32 and recent32 retained')
 if a.output.exists():raise ValueError('New scoring output required')
 a.output.mkdir(parents=True)
 dump(a.output/'SCORER_LOCK.json',{'state':'all_Drone64_sevenarm_source_helper_and_bothbranch_seals_checked_before_target_GT','n':64,'cohort_counts':dict(cohorts),'release_sha256':sha(a.release),'drone_prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'quad_prediction_seal_sha256':r['quad_prediction_seal_sha256'],'scorer_sha256':sha(Path(__file__)),'geometry_dependency_sha256':GEOMETRY_DEPENDENCIES,'unlabelled_behavior_not_subject_accuracy':True})
 return r,manifest,cases,arrays,behaviors

def evaluate_row(row,d,behavior,target):
 raw=d['arm_boxes_raw6'];values,cost=candidate_ious(eval_boxes(raw),target);out={'id':row['id'],'case_alias':row['case_alias'],'case_role':row['case_role'],'platform':'drone','scene':row['scene'],'behavior':behavior,'arms':{}}
 for j,arm in enumerate(ARMS):
  change=raw[j]!=raw[0];qchanged=int(d['arm_query_index'][j])!=int(d['q_source']);iou=float(values[j]);out['arms'][arm]={'iou':iou,'Acc25':iou>.25,'Acc50':iou>.5,'coordinate_errors':coordinate_errors(raw[j],target),'query_changed':qchanged,'actual_changed':bool(change.any()),'actual_changed_axes':change.tolist(),'center_shift_L2_m':float(np.linalg.norm(raw[j,:3].astype(float)-raw[0,:3].astype(float))),'size_shift_L2_m':float(np.linalg.norm(raw[j,3:].astype(float)-raw[0,3:].astype(float))),'raw_nonpositive_sizes':int((raw[j,3:]<=0).sum()),'status':str(d['arm_status'][j])}
 return out,cost

def behavioral_summary(rows):
 n=len(rows);out={'n':n,'scope':'Model role/candidate behaviors, never main-subject semantic accuracy or identity truth','status_counts':{},'boolean_rates':{},'scalar_means':{}}
 for field in ('text_role_status','semantic_status','mapping_only_mapping_status','combined_mapping_status'):out['status_counts'][field]=dict(collections.Counter(r['behavior'][field] for r in rows))
 bools=('text_target_quote_resolved','two_numbering_target_agreement','distinct_reference_collision_observed','defined_reference_order_conflict','semantic_mask_defined','predicted_family_conflict_rejected','subject_family_unsupported_unknown','mask_choice_changed_vs_old','old_controls_strictly_reused','mapping_only_q0_positive_feasible','combined_q0_positive_feasible','mapping_only_q0_preserved_by_positive_support','combined_q0_preserved_by_positive_support','mapping_only_unique_group_readout','combined_unique_group_readout','mapping_only_mapping_NA','combined_mapping_NA')
 for f in bools:
  vals=[r['behavior'].get(f) for r in rows];defined=[x for x in vals if x is not None];yes=sum(x is True for x in defined);out['boolean_rates'][f]={'all_cases_n':n,'defined_n':len(defined),'NA_n':n-len(defined),'true_n':yes,'false_defined_n':len(defined)-yes,'true_fraction_all_cases':yes/n if n else None,'true_fraction_defined':yes/len(defined) if defined else None}
 for f in ('mapping_only_visible_support_n','combined_visible_support_n','mapping_only_feasible_group_n','combined_feasible_group_n','ignored_unresolved_reference_ID_answers'):
  vals=[r['behavior'].get(f) for r in rows];v=[x for x in vals if x is not None];out['scalar_means'][f]={'all_cases_n':n,'defined_n':len(v),'NA_n':n-len(v),'mean_defined':mean(v),'min':min(v) if v else None,'max':max(v) if v else None}
 return out

def summarize(rows):
 tables=[];pairs=[];tails=[];beh=[]
 contrasts=[(a,'source') for a in ARMS[1:]]+[('combined','semantic_only'),('combined','mapping_only'),('combined','old_full_consensus'),('semantic_only','old_full_consensus'),('mapping_only','old_full_consensus'),('old_full_consensus','old_noun_consensus')]
 for cohort in COHORTS:
  rr=[r for r in rows if cohort==COHORTS[2] or r['case_role']==cohort];beh.append({'cohort':cohort,**behavioral_summary(rr)})
  for arm in ARMS:
   vals=[r['arms'][arm] for r in rr];iou=[v['iou'] for v in vals];ce=[v['coordinate_errors']['center_L2_m'] for v in vals];se=[v['coordinate_errors']['size_L2_m'] for v in vals]
   tables.append({'cohort':cohort,'arm':arm,'n':len(rr),'scene_clusters':len({r['scene'] for r in rr}),'mean_iou':mean(iou),'equal_scene_mean_iou':paired_scene(rr,iou)['equal_scene_mean'],'Acc25':mean([float(x>.25) for x in iou]),'Acc50':mean([float(x>.5) for x in iou]),'center_L2_m_mean':mean(ce),'size_L2_m_mean':mean(se),'query_changed_n':sum(v['query_changed'] for v in vals),'actual_changed_n':sum(v['actual_changed'] for v in vals),'status_counts':dict(collections.Counter(v['status'] for v in vals)),'all_fallbacks_retained':True,'all64_union_is_descriptive_not_pristine_confirmation':cohort==COHORTS[2]})
  for arm,ref in contrasts:
   delta=[r['arms'][arm]['iou']-r['arms'][ref]['iou'] for r in rr];ce=[r['arms'][ref]['coordinate_errors']['center_L2_m']-r['arms'][arm]['coordinate_errors']['center_L2_m'] for r in rr];se=[r['arms'][ref]['coordinate_errors']['size_L2_m']-r['arms'][arm]['coordinate_errors']['size_L2_m'] for r in rr];pairs.append({'cohort':cohort,'arm':arm,'reference':ref,'n':len(rr),'IoU_gain':signed_stats(delta,1e-8),'center_error_reduction_m':signed_stats(ce,1e-6),'size_error_reduction_m':signed_stats(se,1e-6),'IoU_scene':paired_scene(rr,delta),'few_scenes_descriptive_only':True})
   positive=sorted([(r,d) for r,d in zip(rr,delta) if d>1e-8],key=lambda x:(-x[1],x[0]['case_alias']))[:5];negative=sorted([(r,d) for r,d in zip(rr,delta) if d< -1e-8],key=lambda x:(x[1],x[0]['case_alias']))[:5]
   tails.append({'cohort':cohort,'arm':arm,'reference':ref,'positive':[{'case_alias':r['case_alias'],'delta_iou':d} for r,d in positive],'negative':[{'case_alias':r['case_alias'],'delta_iou':d} for r,d in negative],'ties_never_mislabeled_positive_or_negative':True,'postscore_effect_conditioned_not_teacher':True})
 return tables,pairs,beh,tails

def main(a):
 start=time.perf_counter();release,manifest,cases,arrays,behaviors=unlabelled_gate(a)
 # First target-bearing payload read occurs only after durable SCORER_LOCK.
 index={x['id']:x for x in read(a.scoring_index)};metadata={};rows=[];cost=collections.Counter()
 for item in manifest:
  sid=item['id'];row,d=arrays[sid];ix=index[sid]
  if ix['platform']!='drone' or ix['scene']!=row['scene']:raise ValueError('Target index platform/scene join mismatch')
  p=Path(ix['meta_path'])
  if sha(p)!=ix['metadata_container_sha256']:raise ValueError('Original target container changed')
  if str(p) not in metadata:metadata[str(p)]=read(p)
  target=transform_gt(metadata[str(p)]['ground_info'][ix['object_index']]['bbox_3d'],'drone',ix.get('waymo_view_suffix'));r,c=evaluate_row(row,d,behaviors[sid],target);rows.append(r);cost.update(c)
 summary,pairs,beh,tails=summarize(rows);result={'state':'R14_Drone64_seven_direct_arms_scored_after_bothbranch_complete_seal','n':64,'cohort_counts':{COHORTS[0]:32,COHORTS[1]:32},'summary':summary,'paired_contrasts':pairs,'behavioral_diagnostics':beh,'signed_alias_tails':tails,'rows_PRIVATE':rows,'geometry_cost_counts':dict(cost),'seconds':time.perf_counter()-start,'scope':'Old32development/recent32relative-content/all64descriptive; historical5912exposure explicit. Behaviors not subjectaccuracy. Originalqueryboxes/yaw0/evalsizeclamp only; no literal identity labels, no oracle selection or adaptation.'};dump(a.output/'RESULTS_PRIVATE.json',result);dump(a.output/'SCALAR_SUMMARY.json',{k:v for k,v in result.items() if not k.endswith('_PRIVATE')});dump(a.output/'SEALED.json',{'state':'sealed_R14_Drone_postrelease_CPU_scoring','n':64,'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'drone_prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'quad_prediction_seal_sha256':release['quad_prediction_seal_sha256'],'release_sha256':sha(a.release),'scorer_sha256':sha(Path(__file__))});print(json.dumps({'state':result['state'],'n':64}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('predictions','manifest','cases','scoring-index','release','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--release-sha256',required=True);main(p.parse_args())
