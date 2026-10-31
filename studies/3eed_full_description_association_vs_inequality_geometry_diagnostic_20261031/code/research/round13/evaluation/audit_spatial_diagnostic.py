"""R13 independent sealed spatial-diagnostic scalar audit; no GT/IoU recompute."""
from pathlib import Path
import collections,hashlib,json,math,re
import numpy as np
B=Path(__file__).resolve().parents[3];R=B/'research/round13';V=R/'evaluation';D=R/'association_diagnostic';O=D/'actual116_attempt001';E=R/'experiment'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
seal=read(O/'SEALED.json');assert sha(O/'SEALED.json')=='d8dec2e0a675baf4b5b81ef8fc8fbc3a5e822d0b2a8670b288a0e62f83839630'
for rel,h in seal['files'].items():assert sha(O/rel)==h
release=read(D/'COMPUTE_RELEASE.json');assert sha(D/'COMPUTE_RELEASE.json')==seal['release_sha256']=='562acfad17c0b927cb3a788befef331285350364cc920e76b43dad7b52e1f3ec'
assert release['POSTSCORE_DIAGNOSTIC_RELEASE'] and release['all_task_predictions_and_main_scores_sealed'] and not release['method_or_primary_metric_modification_permitted'];assert seal['program_sha256']==sha(D/'diagnose_association.py')=='5977b58d68e897b7855a53e410135b2c1980a8bbc8651c396bf66d7f8a076b48'
for rel,h in release['program_bindings'].items():p=Path(rel);assert sha(p if p.is_absolute() else B/p)==h
for key in ('program','plan_v1','plan_v2','manifest','cases'):assert sha(release[key+'_path'])==release[key+'_sha256']
plan=read(D/'PLAN_V2_LOCK.json');assert sha(D/'PLAN_V2_LOCK.json')=='ab64503b001ed1311187f82f07205ccee0582be45de433e3c8df729ce3064d0b'
for rel,h in {**plan['sha256'],**plan['geometry_dependency_sha256']}.items():assert sha(B/rel)==h
lock=read(O/'DIAGNOSTIC_LOCK.json');assert lock['release_sha256']==sha(D/'COMPUTE_RELEASE.json') and lock['plan_v2_sha256']==sha(D/'PLAN_V2_LOCK.json') and not lock['query_selection_performed']
countschema=read(D/'count_schema_attempt001/SUMMARY.json');assert countschema['counts']=={'ground_info_entries':116,'count_field_int_and_list':116,'declared_equals_list_length':116,'other_entries_total':425,'dict_with_bbox_3d_other_shape7':425};assert plan['actual_schema']=={'cases':116,'target_per_container':1,'target_raw_dimension':9,'other_total':425,'other_raw_dimension':7,'others_num_alignment':116,'explicit_ID_keys_found':0}
rows=read(O/'CASES_PRIVATE.json');summary=read(O/'SUMMARY.json');assert len(rows)==len({r['id_PRIVATE'] for r in rows})==summary['n']==116 and len(summary['summary'])==30
primary=read(E/'scored116_attempt001/RESULTS_PRIVATE.json');pr={r['id']:r for r in primary['rows_PRIVATE']};cases=read(release['cases_path']);ci={c['id']:c for c in cases};manifest={r['id']:r for r in read(release['manifest_path'])};art={x['role']:Path(x['directory']) for x in release['sealed_artifacts']}
for artitem in release['sealed_artifacts']:assert sha(Path(artitem['directory'])/'SEALED.json')==artitem['seal_sha256']
checks=0;maxdiff=0.;bridge=0.;failures=[]
def compare(x,y,path):
 global checks,maxdiff
 checks+=1
 if isinstance(y,dict):
  if not isinstance(x,dict) or set(x)!=set(y):failures.append(path+' keys');return
  for k,v in y.items():compare(x[k],v,path+'/'+k)
 elif isinstance(y,list):
  if not isinstance(x,list) or len(x)!=len(y):failures.append(path+' length');return
  for i,v in enumerate(y):compare(x[i],v,path+'/'+str(i))
 elif y is None or isinstance(y,(str,bool,int)):
  if x!=y:failures.append(path+' exact')
 else:
  if not isinstance(x,(int,float)) or not math.isfinite(x) or abs(x-y)>1e-11:failures.append(path+' numeric')
  else:maxdiff=max(maxdiff,abs(x-y))
def stats(v):
 x=[float(t) for t in v if t is not None]
 return {'defined_n':len(x),'NA_n':len(v)-len(x),'mean':math.fsum(x)/len(x) if x else None,'median':float(np.median(x)) if x else None,'min':min(x,default=None),'max':max(x,default=None)}
def category(t,o,s):
 if s.startswith('NA_'):return s
 if o is None:return 'NA_no_distinct_annotated_other_geometry'
 if t>.25 and t-o>.1:return 'target_spatially_dominant'
 if o>.25 and o-t>.1:return 'annotated_other_spatially_dominant'
 return 'unconfirmed_spatial_relation'
lex=('car','vehicle','sedan','van','truck','bus','pedestrian','person','man','woman','people','child','boy','girl','cyclist','biker','bicycle','bike','taxi','suv','jeep','trailer','walker','worker');motor={'vehicle','car','sedan','van','taxi','suv','jeep','truck','bus'};person={'person','pedestrian','man','woman','people','child','boy','girl','walker','worker'}
input_other_total=0;targetbridge_n=0
for r in rows:
 sid=r['id_PRIVATE'];c=ci[sid];p=pr[sid];assert r['case_alias']==c['case_alias'] and r['platform']==p['platform'] and r['case_role']==p['case_role'];a=read(art['A']/'cases'/(c['case_alias']+'.json'))
 with np.load(art['B']/(sid+'.npz'),allow_pickle=False) as z:bm=int(z['selected_mask_index'])
 bm=None if bm<0 else bm;am=a['consensus']['full']['selected_mask_index'];nm=a['consensus']['noun']['selected_mask_index'];relation='both_undefined' if am is None and bm is None else 'A_undefined' if am is None else 'B_undefined' if bm is None else 'same_defined_mask' if am==bm else 'different_defined_masks'
 compare(r['A_B_mask_relation'],relation,'mask_relation');compare(r['A_full_mask_index'],am,'A_mask');compare(r['B_support_mask_index'],bm,'B_mask');compare(r['fullA_query_changed'],p['arms']['A_full_consensus']['query_changed'],'changed_q');compare(r['fullA_box_changed'],p['arms']['A_full_consensus']['actual_changed'],'changed_box')
 s=r['annotation_schema'];input_other_total+=s['input_other_n'];assert 0<=s['deduplicated_other_n']<=s['input_other_n'] and s['malformed_other_n']==0 and s['schema']=='defined';assert len(s['group_member_counts'])==s['deduplicated_other_n']+1 and sum(s['group_member_counts'])==s['input_other_n']+1 and s['group_member_counts'][0]-1==s['target_exact_duplicate_annotation_n'];assert s['target_source_schema']=='target9' and s['other_source_schema']=='other7_no_fabricated_tails'
 for label,arm in [('q0','source'),('fullA','A_full_consensus'),('nounA','A_noun_consensus')]:
  x=r['readouts'][label];old=p['arms'][arm];err=abs(x['target_IoU']-old['iou']);bridge=max(bridge,err);targetbridge_n+=1;compare(x['target_IoU'],old['iou'],'primarytargetbridge');compare(x['query_index'],old['query_index'],'primaryquerybridge');t=x['target_IoU'];o=x['annotated_other_max_IoU'];assert 0<=t<=1 and (o is None or 0<=o<=1);assert (o is None)==(s['deduplicated_other_n']==0)
  compare(x['target_minus_other_margin'],t-o if o is not None else None,'margin');compare(x['spatial_category'],category(t,o,s['schema']),'category');compare(x['literal_identity_GT_claimed'],False,'not_literalGT')
 m=re.search(r'\b(?:'+'|'.join(lex)+r')\b',manifest[sid]['caption'],re.I);noun=m.group(0).lower() if m else None;family=(3,6,8) if noun in motor else (1,) if noun in person else (2,) if noun in ('bicycle','bike') else ()
 maskdir=Path(c['mask_dir']);maskdir=maskdir if maskdir.is_absolute() else B/maskdir;mp=maskdir/(c['frame_alias']+'.npz');ms=read(maskdir/'SEALED.json');assert sha(mp)==ms['files'][mp.name]
 with np.load(mp,allow_pickle=False) as z:classes=z['class_ids'].copy()
 for key,selected in [('A_full',am),('A_noun',nm),('B_support',bm)]:
  x=r['predicted_category_addendum'][key]
  if selected is None:expected={'defined':False,'COCO_predicted_class_id':None,'firstnoun_family_compatible':None}
  else:cls=int(classes[selected]);expected={'defined':True,'COCO_predicted_class_id':cls,'firstnoun_family_compatible':cls in family if family else None,'scope':'frozen category prediction versus input firstnoun family; not GT identity/category'}
  compare(x,expected,'predictedcategory')
assert input_other_total==425 and targetbridge_n==348;compare(summary['target_IoU_existing_primary_max_abs_error'],bridge,'primary_maxerror');assert bridge==0.
expected=[]
for role in ('new_content108','regression8'):
 for platform in ('drone','quad','all_platforms'):
  base=[r for r in rows if r['case_role']==role and (platform=='all_platforms' or r['platform']==platform)]
  strata={'all':base,'same_defined_mask_and_fullA_query_changed':[r for r in base if r['A_B_mask_relation']=='same_defined_mask' and r['fullA_query_changed']],'same_defined_mask':[r for r in base if r['A_B_mask_relation']=='same_defined_mask'],'different_defined_masks':[r for r in base if r['A_B_mask_relation']=='different_defined_masks'],'one_or_both_masks_undefined':[r for r in base if r['A_B_mask_relation'] not in ('same_defined_mask','different_defined_masks')]}
  for name,rr in strata.items():
   d={'case_role':role,'platform':platform,'stratum':name,'n':len(rr),'scene_clusters':len({r['scene_pseudonym'] for r in rr}),'mask_relation_counts':dict(collections.Counter(r['A_B_mask_relation'] for r in rr)),'query_changed_n':sum(r['fullA_query_changed'] for r in rr),'box_changed_n':sum(r['fullA_box_changed'] for r in rr),'readouts':{},'annotation_other_count':stats([r['annotation_schema']['deduplicated_other_n'] for r in rr]),'spatial_category_transition_q0_to_fullA':dict(collections.Counter(r['readouts']['q0']['spatial_category']+' -> '+r['readouts']['fullA']['spatial_category'] for r in rr))}
   for label in ('q0','fullA','nounA'):
    vals=[r['readouts'][label] for r in rr];d['readouts'][label]={'target_IoU':stats([v['target_IoU'] for v in vals]),'annotated_other_max_IoU':stats([v['annotated_other_max_IoU'] for v in vals]),'target_minus_other_margin':stats([v['target_minus_other_margin'] for v in vals]),'spatial_category_counts':dict(collections.Counter(v['spatial_category'] for v in vals))}
   d['fullA_minus_q0_target_IoU']=stats([r['readouts']['fullA']['target_IoU']-r['readouts']['q0']['target_IoU'] for r in rr]);d['fullA_minus_q0_margin']=stats([r['readouts']['fullA']['target_minus_other_margin']-r['readouts']['q0']['target_minus_other_margin'] if r['readouts']['fullA']['target_minus_other_margin'] is not None and r['readouts']['q0']['target_minus_other_margin'] is not None else None for r in rr]);d['predicted_COCO_category_not_GT']={}
   for key in ('A_full','A_noun','B_support'):
    vals=[r['predicted_category_addendum'][key] for r in rr];d['predicted_COCO_category_not_GT'][key]={'defined_n':sum(v['defined'] for v in vals),'compatible_n':sum(v['firstnoun_family_compatible'] is True for v in vals),'incompatible_n':sum(v['firstnoun_family_compatible'] is False for v in vals),'compatibility_NA_n':sum(v['firstnoun_family_compatible'] is None for v in vals)}
   expected.append(d)
compare(summary['summary'],expected,'summary30')
caseevidence={}
for alias in ('new_00067','regression_00006'):
 r=next(r for r in rows if r['case_alias']==alias);caseevidence[alias]={'mask_relation':r['A_B_mask_relation'],'q0_target_IoU':r['readouts']['q0']['target_IoU'],'fullA_target_IoU':r['readouts']['fullA']['target_IoU'],'q0_annotated_other_max':r['readouts']['q0']['annotated_other_max_IoU'],'fullA_annotated_other_max':r['readouts']['fullA']['annotated_other_max_IoU'],'predicted_COCO_classes_not_GT':{k:x['COCO_predicted_class_id'] for k,x in r['predicted_category_addendum'].items()},'predicted_class_firstnoun_compatibility_not_GT':{k:x['firstnoun_family_compatible'] for k,x in r['predicted_category_addendum'].items()}}
by={}
for p in ('drone','quad'):
 s=next(x for x in expected if x['case_role']=='new_content108' and x['platform']==p and x['stratum']=='all');by[p]={k:s['readouts'][k]['spatial_category_counts'] for k in ('q0','fullA','nounA')}
out={'state':'PASS_R13_POSTSCORE_SPATIAL_DIAGNOSTIC_INDEPENDENT_REVIEW','program_sha256':sha(D/'diagnose_association.py'),'plan_v2_sha256':sha(D/'PLAN_V2_LOCK.json'),'diagnostic_seal_sha256':sha(O/'SEALED.json'),'diagnostic_release_sha256':sha(D/'COMPUTE_RELEASE.json'),'diagnostic_summary_sha256':sha(O/'SUMMARY.json'),'diagnostic_cases_sha256':sha(O/'CASES_PRIVATE.json'),'primary_results_sha256':sha(E/'scored116_attempt001/RESULTS_PRIVATE.json'),'auditor_sha256':sha(Path(__file__)),'synthetic_adapter_fixture_sha256':sha(V/'SPATIAL_ADAPTER_INDEPENDENT_CPU_FIXTURE.json'),'checks':checks,'max_numeric_difference':maxdiff,'target_primary_scalar_bridge_n':348,'target_primary_max_abs_difference':bridge,'annotation_schema_cases':116,'raw_others_records_not_instance_count':425,'stratified_summary_rows_recomputed':30,'new108_spatial_categories':by,'case_alias_evidence':caseevidence,'code_scope_review':['Separate postscore release/hash/input/helper/score/schema/program closure verified before numericGT in producer, durable DIAGNOSTIC_LOCK first. Primarymetrics and selectedqueries never changed.','Targetraw9 through immutable R5helper; otherraw7 explicitfloat64 oneDrone z+1.8 thenf32, no fakepadding. Independent128synthetic arithmetic/R5bridges pass; realGT geometry not reread/recomputed by reviewer.','Exacttransformedfloat32 7D grouping removes exactnumeric duplicates only, not near/yaw-equivalent objects or uniqueinstance identification; malformed schema NA retained.','Strict targetIoU>.25 and signedmargin>.10 labels spatial dominance only relative to givenannotations. NA/unconfirmed not wrong, no complete scene inventory or literalidentitytruth.','PredictedCOCO versus firstnounfamily from input is a separate helper prediction diagnostic, never GTclass/identity.','Same2Dmask with less targetoverlap and othermax0 is compatiblewith proposal/mapping/extent error; doesnotprove literal targetmask or ruleout unannotatedobjects.','Allcohort/strata empty NA denominators preserved; output-conditioned/postscore diagnostic, not newconfirmation or methodselection.'],'no_new_GT_annotations_scoringindex_payload_realGTvectors_sensor_or_images_task_IoU_model_GPU_or_methodchanges':True,'no_independent_geometric_recompute_of_othermax_IoU_claimed':True,'failures':failures}
p=V/'SPATIAL_DIAGNOSTIC_INDEPENDENT_REVIEW.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'state':out['state'],'checks':checks,'maxdiff':maxdiff,'bridge':bridge,'failures':failures[:8],'sha256':sha(p)}))
if failures:raise SystemExit(1)
