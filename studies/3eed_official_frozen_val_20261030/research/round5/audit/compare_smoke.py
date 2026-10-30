"""Read-only smoke audit comparison from independently sealed CPU results."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parent;B=R.parents[2];ROOT=R.parent;P=B/'crossplatform/round5';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 release=ROOT/'SMOKE_SCORE_RELEASE.json';assert sha(release)=='ff55079bbae82cbacaf5753807fd10a654645a0c5080374aa8c916337707ef79';assert read(release)['all_predictions_sealed_before_numeric_GT'];assert sha(P/'smoke48_source_v2/sealed.json')==read(release)['prediction_seal_sha256'];d=read(ROOT/'smoke48_scored/RESULTS_PRIVATE.json');m=read(R/'smoke48_independent_v2/RESULTS_PRIVATE.json');l=read(R/'smoke48_literal_bridge_v2/RESULTS_PRIVATE.json');by={x['id']:x for x in d['rows']};err=[];bc=0;orerr=[];orb=0
 for x in m['rows']:
  if x['cohort']=='firstnoun_officialeligible' or x['cohort']=='GTclass_span_available' and x['condition']!='full':continue
  arm=('official_full_' if x['cohort']=='GTclass_span_available' else 'text_'+x['condition']+'_')+x['head'].replace('_','')+'_'+x['mode'];y=by[x['id']]['arms'][arm]
  assert x['native_top1_query']==y['query']
  for k in [1,5,10]:
   err.append(abs(x['metrics']['top1_iou' if k==1 else 'top'+str(k)+'_best_iou_GTassisted']-y['topk'][str(k)]['max_iou_GT_assisted_if_k_gt_1']))
   for t in [25,50]:assert x['metrics']['top'+str(k)+'_acc'+str(t)]==y['topk'][str(k)]['acc'+str(t)];bc+=1
 dense={(x['id'],x['condition'],x['head']):np.asarray(x['all256_ious_PRIVATE'],float) for x in m['dense_oracle_audit']};vectors=[]
 for sid,y in by.items():
  with np.load(ROOT/'smoke48_scored'/(sid+'_IOU_PRIVATE.npz'),allow_pickle=False) as z:
   for cond in ['full','noun']:vectors.extend(abs(dense[sid,cond,'last_']-z[cond+'_last']).tolist())
  for arm,z in y['arms'].items():
   cond='noun' if arm.startswith('text_noun') else 'full';head='proposal_' if 'proposal' in arm else 'last_';v=float(dense[sid,cond,head].max());orerr.append(abs(v-z['oracle_iou']))
   for t in [25,50]:assert (v>t/100)==z['topk']['256']['acc'+str(t)];orb+=1
 le=[];events={};warnings=[]
 for x in l['rows']:
  if x['condition']=='full' and x['stream']=='official_bbf':
   hi=['proposal_','0head_','1head_','2head_','3head_','4head_','last_'].index(x['head']);le.append(abs(x['audited_top1_iou']-by[x['id']]['legacy_7head_top1_iou'][hi]))
  for e in x['literal_pair_engineering']['swallowed_exceptions']:events[x['id'],x['condition'],x['head'],e['candidate_call_offset']]=e['exception_type']
  warnings.extend(x['literal_pair_engineering']['warnings'])
 assert max(err+le+vectors+orerr)<1e-12
 # Check all primary per-platform summary denominators and scene counts explicitly.
 summarychecks=0
 for x in d['summaries']:
  if x['population']!='official_eligible' and x['arm'].startswith('official'):continue
  condition='noun' if x['arm'].startswith('text_noun') else 'full';cohort='GTclass_span_available' if x['arm'].startswith('official') else ('firstnoun_officialeligible' if x['population']=='official_eligible' else 'firstnoun_allraw');head='proposal_' if 'proposal' in x['arm'] else 'last_';mode=x['arm'].split('_')[-1];mine=next(t for t in m['summaries'] if (t['platform'],t['condition'],t['cohort'],t['head'],t['mode'])==(x['platform'],condition,cohort,head,mode));assert mine['n']==x['n'] and mine['scene_clusters']==x['scene_clusters'];assert abs(mine['metrics']['top1_iou']['micro_mean']-x['mean_iou'])<1e-12;assert abs(mine['metrics']['top1_iou']['scene_macro_mean']-x['equal_scene_mean_iou'])<1e-12;summarychecks+=1
 result={'state':'passed_smoke_engineering_unblocks_frozen_fullVAL_no_method_tuning','source_seal_sha256':read(release)['prediction_seal_sha256'],'postseal_GT_release_sha256':sha(release),'root_scored_sha256':sha(ROOT/'smoke48_scored/RESULTS_PRIVATE.json'),'independent_v2_sha256':sha(R/'smoke48_independent_v2/RESULTS_PRIVATE.json'),'literal_bridge_v2_sha256':sha(R/'smoke48_literal_bridge_v2/RESULTS_PRIVATE.json'),'top1_5_10_numeric_cells':len(err),'max_abs_selected_metric_difference_root2D_vs_independent3D':max(err),'strict25_50_boolean_cells_exact':bc,'all256_candidate_IoU_cells':len(vectors),'max_abs_candidate_difference':max(vectors),'all256_oracle_scalar_cells':len(orerr),'max_abs_oracle_difference':max(orerr),'oracle25_50_booleans_exact':orb,'legacy7head_top1_cells':len(le),'max_abs_legacy_difference':max(le),'platform_population_headmode_summaries_checked':summarychecks,'literal_official_float32_vs_audited3D_float64':{'numeric_comparison_scope':'All7heads,2conditions,4scorestreams whereGTclassspanexists, nativeTop10','rows':l['n_stream_head_condition_rows'],'max_abs_finite_difference':l['max_abs_finite_difference'],'strict_threshold_disagreement_rows':l['n_rows_with_strict_threshold_disagreement'],'nonfinite_literal_rows':l['n_rows_with_nonfinite_literal'],'deduplicated_swallowed_exceptions':len(events),'warnings':len(warnings),'do_not_claim_exactnumericalparity':True},'retained_v1_engineering':{'status':'notaccepted_as_float64geometryaudit','cause':'independentwrapperpassedfloat32extenttolibrarywhosevolumes use np.prod originalinputdtype','selected_root_maxdiff':8.566708498136677e-08,'fix':'newindependent_core_v2 castsallgeometryinputfloat64; v1codeandoutputs preserved','predictions_models_methods_changed':False},'scope':['Frozenunioncheckpoint, noadaptation/fulltraining','smoke48engineeringnotnewheldoutmethodconfirmation','OfficialGTclasscolumnchosenonlypostseal;all6classcandidatesprecomputed','Last/proposal separate;legacy7headmIoUexplicit','FullVALfutureGTstillrequiresitsowncompletepredictionseal/release','No rawPRIVATEdataforpublicexport'],'source_code_sha256':{f:sha(R/f) for f in ['independent_core_v2.py','independent_score_v2.py','literal_iou_bridge_v2.py','compare_smoke.py']},'no_model_GPU_calls':True};out=R/'SMOKE_INDEPENDENT_RECEIPT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
