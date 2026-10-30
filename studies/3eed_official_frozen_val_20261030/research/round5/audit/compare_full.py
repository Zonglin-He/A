"""Read-only comparisons of root fullVAL scores against sealed independent audits.
Does not open metadata or compute target IoUs. Scalar-only public receipt; no raw
candidate arrays, originalscene names, captions or annotation values are emitted.
"""
from pathlib import Path
import json,hashlib,argparse,collections
import numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def verify(folder):
 seal=read(folder/'SEALED.json')
 for n,h in seal['files'].items():assert sha(folder/n)==h,n
 return seal

def main(a):
 for folder in [a.root_score,a.independent,a.literal]:verify(folder)
 release=read(a.release);assert release['all_predictions_sealed_before_numeric_GT'];d=read(a.root_score/'RESULTS_PRIVATE.json');m=read(a.independent/'RESULTS_PRIVATE.json');l=read(a.literal/'RESULTS_PRIVATE.json');assert d['n']==m['n_raw_targets']==l['n_parents']==release['n_samples'];assert d['prediction_seal_sha256']==release['prediction_seal_sha256']
 for folder in [a.independent,a.literal]:assert read(folder/'SCORER_LOCK.json')['release_sha256']==sha(a.release)
 by={x['id']:x for x in d['rows']};errors=[];bools=0;oracle_errors=[];oracle_bools=0;legacy_errors=[];literal_audit_errors=[];vectors=[];summarychecks=0
 for x in m['rows']:
  if x['cohort']=='firstnoun_officialeligible':continue
  assert x['cohort'] in ['GTclass_span_available','firstnoun_allraw'];arm=('official_full_' if x['cohort']=='GTclass_span_available' else 'text_'+x['condition']+'_')+x['head'].replace('_','')+'_'+x['mode'];y=by[x['id']]['arms'][arm];assert x['native_top1_query']==y['query']
  for k in [1,5,10]:
   v=x['metrics']['top1_iou' if k==1 else 'top'+str(k)+'_best_iou_GTassisted'];errors.append(abs(v-y['topk'][str(k)]['max_iou_GT_assisted_if_k_gt_1']))
   for t in [25,50]:assert x['metrics']['top'+str(k)+'_acc'+str(t)]==y['topk'][str(k)]['acc'+str(t)];bools+=1
 # Dense candidate audit is a hash-selected subset. Do not rename this fulloraclecoverage.
 dense={(x['id'],x['condition'],x['head']):np.asarray(x['all256_ious_PRIVATE'],float) for x in m['dense_oracle_audit']};sids=sorted({k[0] for k in dense})
 for sid in sids:
  y=by[sid]
  with np.load(a.root_score/(sid+'_IOU_PRIVATE.npz'),allow_pickle=False) as z:
   for cond in ['full','noun']:vectors.extend(abs(dense[sid,cond,'last_']-z[cond+'_last']).tolist())
  for arm,z in y['arms'].items():
   cond='noun' if arm.startswith('text_noun') else 'full';head='proposal_' if 'proposal' in arm else 'last_';v=float(dense[sid,cond,head].max());oracle_errors.append(abs(v-z['oracle_iou']))
   for t in [25,50]:assert (v>t/100)==z['topk']['256']['acc'+str(t)];oracle_bools+=1
 events={};warnings=[];primary_literal=[]
 for x in l['rows']:
  y=by[x['id']]
  if x['condition']=='full' and x['stream']=='official_bbf':
   hi=['proposal_','0head_','1head_','2head_','3head_','4head_','last_'].index(x['head']);legacy_errors.append(abs(x['audited_top1_iou']-y['legacy_7head_top1_iou'][hi]))
  if x['head'] in ['last_','proposal_']:
   mode=x['stream'].split('_')[-1];arm=('official_full_' if x['stream'].startswith('official') else 'text_'+x['condition']+'_')+x['head'].replace('_','')+'_'+mode;z=y['arms'][arm]
   for k in [1,5,10]:
    key='audited_top1_iou' if k==1 else 'audited_best_top'+str(k)+'_iou';literal_audit_errors.append(abs(x[key]-z['topk'][str(k)]['max_iou_GT_assisted_if_k_gt_1']))
   if x['condition']=='full' and x['head']=='last_' and x['stream'].startswith('official'):primary_literal.append(x)
  for e in x['literal_pair_engineering']['swallowed_exceptions']:events[x['id'],x['condition'],x['head'],e['candidate_call_offset']]=e['exception_type']
  warnings.extend(x['literal_pair_engineering']['warnings'])
 assert max(errors+vectors+oracle_errors+legacy_errors+literal_audit_errors)<1e-10
 for x in d['summaries']:
  condition='noun' if x['arm'].startswith('text_noun') else 'full';cohort='GTclass_span_available' if x['arm'].startswith('official') else ('firstnoun_officialeligible' if x['population']=='official_eligible' else 'firstnoun_allraw');head='proposal_' if 'proposal' in x['arm'] else 'last_';mode=x['arm'].split('_')[-1];mine=next(t for t in m['summaries'] if (t['platform'],t['condition'],t['cohort'],t['head'],t['mode'])==(x['platform'],condition,cohort,head,mode));assert mine['n']==x['n'] and mine['scene_clusters']==x['scene_clusters'];assert abs(mine['metrics']['top1_iou']['micro_mean']-x['mean_iou'])<1e-10;assert abs(mine['metrics']['top1_iou']['scene_macro_mean']-x['equal_scene_mean_iou'])<1e-10;summarychecks+=1
 for platform,c in m['counts'].items():
  assert c['raw_forwarded_targets']==d['eligibility'][platform]['raw'];assert c['full_official_eligible']==d['eligibility'][platform]['eligible'];assert c['full_eligible_empty_map']==d['eligibility'][platform]['eligible_empty_map']
 result={'state':'passed_fullVAL_independent_selected_and_stratified_candidate_audit','n_raw_targets':d['n'],'release_sha256':sha(a.release),'root_result_sha256':sha(a.root_score/'RESULTS_PRIVATE.json'),'independent_result_sha256':sha(a.independent/'RESULTS_PRIVATE.json'),'literal_result_sha256':sha(a.literal/'RESULTS_PRIVATE.json'),'full_selected_Top1_5_10_numeric_cells':len(errors),'full_selected_max_abs_error':max(errors),'full_selected_strict_boolean_cells_exact':bools,'dense_candidate_parent_subset_n':len(sids),'dense_candidate_IoU_cells':len(vectors),'dense_candidate_max_abs_error':max(vectors),'dense_oracle_scalar_cells':len(oracle_errors),'dense_oracle_max_abs_error':max(oracle_errors),'dense_oracle_strict_boolean_cells_exact':oracle_bools,'dense_oracle_scope':'fixedhash64/platform only; root coverage isfull, independentaudit isstratifiedsubset','legacy7head_top1_cells':len(legacy_errors),'legacy_max_abs_error':max(legacy_errors),'literal_auditor_float64_selected_crosscheck_cells':len(literal_audit_errors),'literal_auditor_float64_max_error':max(literal_audit_errors),'platform_population_headmode_summary_cells_checked':summarychecks,'all_eligibility_and_empty_map_counts_match':True,'literal_official_float32_vs_independent_float64':{'max_abs_finite_difference':l['max_abs_finite_difference'],'threshold_disagreement_rows':l['n_rows_with_strict_threshold_disagreement'],'nonfinite_literal_rows':l['n_rows_with_nonfinite_literal'],'deduplicated_swallowed_exception_pairs':len(events),'exception_type_counts':dict(collections.Counter(events.values())),'warning_records_including_streamduplicates':len(warnings),'literal_is_distinct_measurement_not_silently_replaced':True},'no_model_GPU_newGTscoring_in_this_comparison':True,'public_projection':'Only explicitscalar/count/bool/hash/scope fields here; no annotationcoords/fields8or9 interpretations, rawscenes/captions or perqueryIoUs','protocol_clarification':'ActualofficialGTfirst7 only; remainingrawfields unnamedextras; neverinterpretgenericutilitycommentsasdatasetgeneration schema'}
 assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root-score',type=Path,required=True);ap.add_argument('--independent',type=Path,required=True);ap.add_argument('--literal',type=Path,required=True);ap.add_argument('--release',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);main(ap.parse_args())
