"""Round5 independent CPU audit: full primary Top1/5/10 and7headBBFlegacy.
No model or GPU. General3D convex geometry only; fixed hash-stratified dense
candidate/oracle checks supplement, not replace, selected-box full coverage.
"""
from pathlib import Path
import json,hashlib,argparse,time
from collections import defaultdict
import numpy as np
from independent_core_v2 import candidate_ious,transform_gt,validate_native_top10
from literal_iou_bridge import validate_cache
R=Path(__file__).resolve().parent;B=R.parents[2];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
HEADS=['proposal_','0head_','1head_','2head_','3head_','4head_','last_'];CLASSES=['car','pedestrian','bus','othervehicle','truck','cyclist'];SALT='round5-independent-dense-audit-v1|20261030|'
def arr(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def native_metrics(value_lookup,order):
 v=np.array([value_lookup[int(q)] for q in order]);return {'top1_iou':float(v[0]),'top5_best_iou_GTassisted':float(v[:5].max()),'top10_best_iou_GTassisted':float(v.max()),'top1_acc25':bool(v[0]>.25),'top1_acc50':bool(v[0]>.5),'top5_acc25':bool(v[:5].max()>.25),'top5_acc50':bool(v[:5].max()>.5),'top10_acc25':bool(v.max()>.25),'top10_acc50':bool(v.max()>.5)}
def aggregate(rows):
 grouped=defaultdict(list)
 for r in rows:grouped[r['platform'],r['condition'],r['cohort'],r['head'],r['mode']].append(r)
 result=[]
 for (platform,condition,cohort,head,mode),rr in grouped.items():
  scenes=sorted({r['scene'] for r in rr});keys=list(rr[0]['metrics']);out={'platform':platform,'condition':condition,'cohort':cohort,'head':head,'mode':mode,'n':len(rr),'scene_clusters':len(scenes),'metrics':{}}
  for key in keys:
   v=np.array([r['metrics'][key] for r in rr]);sv=np.array([np.mean([r['metrics'][key] for r in rr if r['scene']==s]) for s in scenes]);item={'micro_mean':float(v.mean()),'scene_macro_mean':float(sv.mean())}
   if 'acc' in key:item['success_count']=int(v.sum())
   out['metrics'][key]=item
  out['scene_groups']=[{'scene':s,'n':sum(r['scene']==s for r in rr),'top1_mean_iou':float(np.mean([r['metrics']['top1_iou'] for r in rr if r['scene']==s]))} for s in scenes];result.append(out)
 return result

def main(a):
 assert a.authorized_postseal,'No targetclass/GT withoutparentrelease';source=a.predictions;sealpath=source/'sealed.json';se=read(sealpath);release=read(a.release);assert sha(a.release)==a.parent_grant and release['all_predictions_sealed_before_numeric_GT'];assert release['prediction_seal_sha256']==sha(sealpath) and release['manifest_sha256']==sha(a.manifest);assert se['state']=='sealed_frozen_baseline_no_labels_selected' and se['no_actual_target_class_or_numeric_GT_read'];assert sha(a.manifest)==se['manifest_sha256'];assert sha(a.scoring_index)==a.scoring_index_sha
 for n,h in se['files'].items():assert sha(source/n)==h
 for filename,key in [('trace.json','trace_sha256'),('repeat_control.json','repeat_control_sha256'),('point_provenance.json','point_provenance_sha256')]:assert sha(source/filename)==se[key]
 samples=read(a.manifest);assert len(samples)==se['n'];assert len(se['files'])==len(samples)*2;assert se['head_names']==HEADS and se['class_names']==CLASSES;index={x['id']:x for x in read(a.scoring_index)};assert len({s['id'] for s in samples})==len(samples)
 for s in samples:assert sha(index[s['id']]['meta_path'])==index[s['id']]['metadata_container_sha256']
 # Dense audit subset fixed solelybyplatform+samplehash before opening any targetvalues.
 dense=set()
 for p in ['waymo','drone','quad']:
  ordered=sorted([s['id'] for s in samples if s['platform']==p],key=lambda sid:hashlib.sha256((SALT+sid).encode()).hexdigest());dense.update(ordered[:a.dense_per_platform])
 out=a.output;out.mkdir(exist_ok=False);lock={'source_sha256':sha(__file__),'core_sha256':sha(R/'independent_core_v2.py'),'convex3D_sha256':sha(B/'tools/obb_iou.py'),'source_seal_sha256':sha(sealpath),'index_sha256':sha(a.scoring_index),'all_source_and_metadata_hashes_checked_before_targetvalueaccess':True,'parent_authorization':a.parent_grant,'release_sha256':sha(a.release),'coverage':'alllast/proposalnativeTop10BBF/BBS+firstnoun; all7headsBBFtop1legacy; dense256allheads onfixedsubset','dense_sampling_salt':SALT,'dense_sample_ids':sorted(dense),'dense_per_platform_cap':a.dense_per_platform,'noGPU_model_training_calls':True};(out/'SCORER_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n');rows=[];legacy=[];dense_records=[];counters=defaultdict(lambda:defaultdict(int));start=time.perf_counter()
 for sample in samples:
  sid=sample['id'];s=index[sid];obj=read(s['meta_path'])['ground_info'][s['object_index']];cls=str(obj['class']).lower();assert cls in CLASSES,'Unknownofficialclass iserror, notsilentskip';ci=CLASSES.index(cls);gt=transform_gt(obj['bbox_3d'],s['platform'],s['waymo_view_suffix']);counts=counters[s['platform']];counts['raw_forwarded_targets']+=1
  for cond in ['full','noun']:
   z=arr(source/(sid+'_'+cond+'.npz'));clamp=validate_cache(z);counts[cond+'_clamped_dimensions']+=clamp['clamped_dimension_count'];eligible=bool(z['class_synonym_found'][ci]);counts[cond+'_official_eligible']+=eligible;
   if cond=='full':official_full_eligible=eligible
   counts[cond+'_official_no_synonym']+=not eligible;counts[cond+'_eligible_empty_map']+=bool(eligible and not z['class_positive_map_nonzero'][ci])
   for hi,head in enumerate(HEADS):
    full_metrics=head in ['last_','proposal_']
    if cond=='noun' and not full_metrics:continue
    streams=[]
    for mode in ['bbf','bbs']:
     if cond=='full' and eligible and (full_metrics or mode=='bbf'):streams.append(('GTclass_span_available',mode,z[mode+'_scores_by_class'][hi,ci],z[mode+'_native_top10_by_class'][hi,ci]))
     if full_metrics:streams.append(('firstnoun_allraw',mode,z['firstnoun_'+mode][hi],z['firstnoun_'+mode+'_native_top10'][hi]))
    validated=[(cohort,mode,sc,validate_native_top10(sc,order)) for cohort,mode,sc,order in streams]
    if sid in dense:queries=np.arange(256)
    else:queries=np.unique(np.concatenate([order if full_metrics else order[:1] for _,_,_,order in validated])) if validated else np.array([],int)
    if not len(queries):continue
    values,gd=candidate_ious(z['boxes'][hi,queries],gt);lookup={int(q):float(v) for q,v in zip(queries,values)};counts['independent_convex_calls']+=gd['convex_intersection_calls'];counts['proven_disjoint_zero_pairs']+=gd['broadphase_proven_zero_count']
    if sid in dense:dense_records.append({'id':sid,'platform':s['platform'],'condition':cond,'head':head,'all256_ious_PRIVATE':values.tolist(),'best256_iou_GTassisted':float(values.max()),'candidate_coverage_failure25':bool(values.max()<=.25),'candidate_coverage_failure50':bool(values.max()<=.5)})
    for cohort,mode,sc,order in validated:
     if cohort=='GTclass_span_available' and mode=='bbf':legacy.append({'id':sid,'platform':s['platform'],'condition':cond,'head':head,'top1_iou':lookup[int(order[0])]})
     if not full_metrics:continue
     metrics=native_metrics(lookup,order);record={'id':sid,'platform':s['platform'],'scene':s['scene'],'condition':cond,'cohort':cohort,'head':head,'mode':mode,'metrics':metrics,'condition_trueclass_map_nonzero':bool(z['class_positive_map_nonzero'][ci]),'native_top1_query':int(order[0])};rows.append(record)
     if cohort=='firstnoun_allraw' and official_full_eligible:rows.append(dict(record,cohort='firstnoun_officialeligible'))
  print('INDEPENDENT_SELECTED',sid,flush=True)
 summaries=aggregate(rows);legacy_summary=[]
 for p in ['waymo','drone','quad']:
  for cond in ['full','noun']:
   rr=[x for x in legacy if x['platform']==p and x['condition']==cond];assert len(rr)%7==0;legacy_summary.append({'platform':p,'condition':cond,'n_object_head_observations':len(rr),'n_eligible_targets':len(rr)//7,'sevenhead_BBF_legacy_mean_iou':float(np.mean([x['top1_iou'] for x in rr])) if rr else None,'not_seven_independent_samples':True})
 result={'scope':'Independentfloat64selectedboxscores, NOTliteralofficialfloat32numbers; officialeligibleGTclassspan distinctfromallrawtext','n_raw_targets':len(samples),'rows':rows,'summaries':summaries,'counts':dict(counters),'legacy7head_BBF_miou':legacy_summary,'dense_oracle_audit':dense_records,'dense_oracle_scope':'fixedhashsubsetonly; doesnot independentlycertify fullVALoracle extrema','all_fullVAL_oracle_is_root_vector2D_separate_measurement':True,'seconds':time.perf_counter()-start,'limits':['PrimaryofficialreadoutusesGTclass offline, notunconditionaldeployment','Last/proposal separate;GTclasseligiblecohort conditionfullvsnouncanchange','Firstnounallraw andofficialeligible textcontrol bothretained; matchedeligible usesoriginalfullcaption GTclassspan, alsofornouncondition','MeanbestTop5/10isGTassistedretrievaldiagnostic','Sceneclusters notframe/heads/views independent','No labelsusedtosourceforward oradaptation (thereisnoadaptation)']};(out/'RESULTS_PRIVATE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(out/'SEALED.json').write_text(json.dumps({'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}},indent=2)+'\n');print('AUDIT_SELECTED_COMPLETE',len(samples),sha(out/'RESULTS_PRIVATE.json'))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--release',type=Path,required=True);ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--scoring-index',type=Path,required=True);ap.add_argument('--scoring-index-sha',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--dense-per-platform',type=int,default=64);ap.add_argument('--authorized-postseal',action='store_true');ap.add_argument('--parent-grant',required=True);main(ap.parse_args())
