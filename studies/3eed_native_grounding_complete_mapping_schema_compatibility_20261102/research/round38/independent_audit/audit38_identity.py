"""Independent cached-parser/readout/task replay + immutable identity24 join. CPU."""
import sys,json,re,math,time,hashlib,collections,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'crossplatform/round30'),str(ROOT/'research/round5/audit')]
from common30 import rows,read,dump,sha,verify,load_npz
from independent_core_v2 import candidate_ious,transform_gt
B37=ROOT/'crossplatform/round37';B38=ROOT/'crossplatform/round38';P37=B37/'inference208_attempt003';P38=B38/'predictions208_attempt001';E38=ROOT/'research/round38/experiment';IDENT=ROOT/'research/round37/independent_identity'
ARMS=['source','R29_direct','legacy_G','point_strict_fullmap','point_strict_oldsubset','som_strict_fullmap','som_strict_oldsubset','point_compatible_fullmap','point_compatible_oldsubset','som_compatible_fullmap','som_compatible_oldsubset']
ANNOTATION_SHA='6e382dc7cd7cdc4257a8d4811c1b640ca0d466eaab6b2b72fb1dde916c099097'
def close(a,b,t=1e-10):assert abs(float(a)-float(b))<t,(a,b,t)
def decoded(raw):
 t=str(raw).strip()
 for pattern in [r'```(?:json)?\s*(.*?)\s*```',r'<answer>\s*(.*?)\s*</answer>']:
  m=re.fullmatch(pattern,t,re.I|re.S)
  if m:t=m.group(1).strip()
 def pairs(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError('duplicate key')
   d[k]=v
  return d
 return json.loads(t,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def parse(raw,interface,compatible):
 try:o=decoded(raw)
 except Exception:return {'status':'parse_failure','value':None,'kind':'malformed_JSON'}
 singleton=False
 if compatible and isinstance(o,list):
  if len(o)!=1 or not isinstance(o[0],dict):return {'status':'parse_failure','value':None,'kind':'not_unique_singleton_object'}
  o=o[0];singleton=True
 if not isinstance(o,dict):return {'status':'parse_failure','value':None,'kind':'not_object'}
 location={'point':{'point','point2d','point_2d'},'som':{'target_id','id','candidate_id'}}[interface];found=set(o)&location
 if len(found)!=1:return {'status':'parse_failure','value':None,'kind':'not_exactly_one_location'}
 k=next(iter(found));extras=set(o)-{k}
 if extras and (not compatible or extras!={'label'} or not isinstance(o['label'],str)):return {'status':'parse_failure','value':None,'kind':'extras_or_invalid_label'}
 v=o[k]
 if v is not None:
  valid=isinstance(v,list) and len(v)==2 and all(isinstance(n,(int,float)) and not isinstance(n,bool) and math.isfinite(n) for n in v) if interface=='point' else isinstance(v,int) and not isinstance(v,bool)
  if not valid:return {'status':'parse_failure','value':None,'kind':'invalid_location_value'}
 kind=('singleton_' if singleton else 'object_')+k+('_label' if 'label' in o else '_only')
 return {'status':'explicit_null' if v is None else 'parsed','value':v,'kind':kind,'raw_location_key':k,'raw_unique_value':v,'label_ignored':'label' in o,'singleton':singleton}

def readout(interface,parsed,raw,case,z,subset):
 scores=z['source_scores'];q0=int(scores.argmax());m=None;point=None;status=parsed['status'];hits=[]
 if status=='parsed':
  if interface=='som':
   c=next((c for c in case['candidates'] if c['target_id']==parsed['value']),None)
   if c is None:status='unlisted_target_id'
   else:m=int(c['mask_index']);status='listed_independent_mask'
  else:
   grid=raw['image_grid_thw'][0];assert len(grid)==3 and grid[0]==1;W,H=case['image_size_wh'];pw,ph=grid[2]*14,grid[1]*14;x,y=parsed['value']
   if 0<=x<pw and 0<=y<ph:point=[float(x*W/pw),float(y*H/ph)]
   if point is None or not (0<=point[0]<z['masks_bool'].shape[2] and 0<=point[1]<z['masks_bool'].shape[1]):status='coordinate_outside_image'
   else:
    hits=np.flatnonzero(z['masks_bool'][:,int(np.floor(point[1])),int(np.floor(point[0]))]).tolist()
    if len(hits)==1:m=int(hits[0]);status='single_independent_mask'
    else:status='no_mask' if not hits else 'overlapping_multiple_masks'
 table=case['old_subset_groups'] if subset else case['full_groups'];g=table.get(str(m),[]) if m is not None else []
 q=max(g,key=lambda i:(float(scores[i]),-i)) if g else q0
 if not g:status=status if m is None else 'absent_from_old_subset' if subset and str(m) not in table else 'mask_without_3D_group'
 else:status='mapped_original_query'
 record={'query':int(q),'mapped':bool(g),'status':status,'mask_index':m,'original_point':point,'group_size':len(g),'query_changed':q!=q0,'old_subset_contains_selected_mask':m is not None and str(m) in case['old_subset_groups'],'full_group_nonempty':m is not None and bool(case['full_groups'].get(str(m),[]))}
 return record,hits

def main():
 started=time.perf_counter();assert (P38/'SEALED.json').exists(),'Wait for compatibility predictions to seal before any join';verify(P37);verify(P38);verify(B37/'prepared208_attempt001')
 assert sha(IDENT/'IDENTITY_RECORDS_PRIVATE.json')==ANNOTATION_SHA;annseal=read(IDENT/'ANNOTATIONS_SEALED.json');assert annseal['identity_records_sha256']==ANNOTATION_SHA;anns={r['review_alias']:r for r in read(IDENT/'IDENTITY_RECORDS_PRIVATE.json')};co=read(IDENT/'COHORT_PRIVATE.json');idann={c['row_PRIVATE']['id']:(c,anns[c['review_alias']]) for c in co};ib={r['review_alias']:r for r in read(IDENT/'RENDER_BINDINGS_PRIVATE.json')};assert len(idann)==24 and sum(a['status']=='clear_single_mask' for a in anns.values())==18
 seal=read(P38/'SEALED.json');assert seal['original_R37_seal_sha256']==sha(P37/'SEALED.json') and seal['new_VLM_calls']==0 and seal['new_GPU_or_TTA_steps']==0
 rr=rows();assert len(rr)==208;idx={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};original={r['id_PRIVATE']:r for r in read(E38/'ALL_ROWS_PRIVATE.json')};schema=[];tasks=[];identity=[];groupdiagnostics=[];counters=collections.Counter();kinds=collections.Counter();malformed=[];maxpoint=0.
 for r in rr:
  case=read(B37/'prepared208_attempt001/cases_PRIVATE'/(r['id']+'.json'));z=load_npz(B37/'prepared208_attempt001/arrays_PRIVATE'/(r['id']+'.npz'));c38=read(P38/'cases_PRIVATE'/(r['id']+'.json'));p38=load_npz(P38/'predictions_PRIVATE'/(r['id']+'.npz'));p37=load_npz(P37/'predictions_PRIVATE'/(r['id']+'.npz'));v37=dict(zip(p37['arm_names'].tolist(),map(int,p37['queries'])));assert p38['arm_names'].tolist()==ARMS
  S=z['support_count'];C=z['support_coverage'];I=z['projection_iou'];valid=(S>=3)&(C>=.15)&(I>=.1)&(z['source_boxes'][:,3:]>0).all(1)[:,None];unique=valid.sum(1)==1;assert np.array_equal(valid,z['valid_query_mask']) and np.array_equal(unique,z['unique_query'])
  for m in range(z['masks_bool'].shape[0]):assert case['full_groups'][str(m)]==np.flatnonzero(valid[:,m]&unique).tolist()
  for m,g in case['old_subset_groups'].items():assert g==case['full_groups'][m]
  q={n:v37[n] for n in ['source','R29_direct','legacy_G']};reconstructed={};rawcalls={};parsemap={};hitmaps={}
  for interface in ['point','som']:
   ref=c38['calls'][interface];path=ROOT/ref['raw_file'];assert path.parent==P37/'calls_PRIVATE' and sha(path)==ref['raw_sha256'];raw=read(path)['result'];rawcalls[interface]=raw
   for version,compat in [('strict',False),('compatible',True)]:
    pa=parse(raw['raw_answer_PRIVATE'],interface,compat);parsemap[interface,version]=pa;expected=ref[version+'_parse'];assert pa['status']==expected['status'] and pa['value']==expected['value'],(r['alias'],interface,version,pa,expected)
    if compat and pa['status']!='parse_failure':assert pa['value']==expected['raw_unique_position'] and expected['value_equal_raw_unique_position']
    for mapping in ['fullmap','oldsubset']:
     arm=interface+'_'+version+'_'+mapping;a,hits=readout(interface,pa,raw,case,z,mapping=='oldsubset');b=c38['readouts'][arm];assert a.keys()==b.keys()
     for n in a:
      if n=='original_point' and a[n] is not None:maxpoint=max(maxpoint,float(abs(np.array(a[n])-b[n]).max()));assert maxpoint<1e-10
      else:assert a[n]==b[n],(r['alias'],arm,n,a[n],b[n])
     reconstructed[arm]=a;hitmaps[interface,version]=hits;q[arm]=a['query'];counters['readout_rows']+=1
     if version=='strict':assert q[arm]==v37[interface+'_'+mapping]
   sp=parsemap[interface,'strict'];cp=parsemap[interface,'compatible'];recover=sp['status']=='parse_failure' and cp['status']!='parse_failure';counters['raw_calls']+=1
   if sp['status']=='parse_failure':counters[interface+'_strict_fail']+=1
   if cp['status']=='parse_failure':counters[interface+'_compatible_fail']+=1
   if recover:
    assert interface=='point' and cp['singleton'] and cp['value']==cp['raw_unique_value'];kinds[cp['kind']]+=1;counters['recovered_unique_position']+=1
   if sp['status']!='parse_failure':assert cp['status']==sp['status'] and cp['value']==sp['value'];counters['strict_valid_unchanged']+=1
   if cp['status']=='parse_failure':malformed.append({'alias':r['alias'],'interface':interface,'raw_file':ref['raw_file'],'raw_sha256':ref['raw_sha256'],'independent_reason':cp['kind'],'no_trailing_character_repair':True})
   schema.append({'alias':r['alias'],'platform':r['platform'],'interface':interface,'strict_status':sp['status'],'compatible_status':cp['status'],'recovered':recover,'kind':cp['kind'],'raw_coordinate_value_unchanged':cp.get('value')==cp.get('raw_unique_value') if cp['status']!='parse_failure' else None})
  queries=np.array([q[n] for n in ARMS]);assert np.array_equal(queries,p38['queries']) and np.array_equal(z['source_boxes'][queries],p38['boxes']);assert q['som_strict_fullmap']==q['som_compatible_fullmap'] and q['som_strict_oldsubset']==q['som_compatible_oldsubset']
  ix=idx[r['id']];path=ix['meta_path']
  if path not in containers:assert sha(path)==ix['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'))
  def task_for(qs):
   b=z['source_boxes'][list(qs)].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);return candidate_ious(b,gt)[0]
  vv=task_for(queries);values=dict(zip(ARMS,map(float,vv)));reference=original[r['id']]
  for arm in ARMS:assert q[arm]==reference['queries'][arm];close(values[arm],reference['IoU'][arm],1e-12)
  tasks.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':q,'IoU':values,'readouts':reconstructed});counters['selected_box_task_values']+=len(ARMS)
  if r['id'] in idann:
   cohort,a=idann[r['id']];ra=cohort['review_alias'];binding=ib[ra];assert case['candidates']==binding['candidates_PRIVATE'];assert case['image_sha256']==binding['image_file_sha256'];clear=a['status']=='clear_single_mask';reviewm=next((c['mask_index'] for c in case['candidates'] if clear and c['target_id']==a['target_display_ids'][0]),None);group=case['full_groups'].get(str(reviewm),[]) if reviewm is not None else [];rep=max(group,key=lambda j:(float(z['source_scores'][j]),-j)) if group else None;gvals=task_for(group) if group else np.array([])
   gd={'review_alias':ra,'original_alias_PRIVATE':r['alias'],'platform':r['platform'],'annotation_status':a['status'],'review_mask_index_PRIVATE':reviewm,'target_group_size':len(group),'source_representative_query_PRIVATE':rep,'source_representative_IoU':float(task_for([rep])[0]) if rep is not None else None,'target_group_oracle_IoU_postseal_only':float(gvals.max()) if len(gvals) else None,'source_q0_in_review_group':q['source'] in group if group else None,'source_IoU':values['source'],'not_identity_gold_or_runtime_gate':True};groupdiagnostics.append(gd)
   for interface in ['point','som']:
    for version in ['strict','compatible']:
     pa=parsemap[interface,version];hits=hitmaps[interface,version];claim=reconstructed[interface+'_'+version+'_fullmap']['mask_index'];m_id=next((c['target_id'] for c in case['candidates'] if c['mask_index']==claim),None)
     for mapping in ['fullmap','oldsubset']:
      arm=interface+'_'+version+'_'+mapping;rd=reconstructed[arm];point_target=reviewm in hits if interface=='point' and clear and pa['status']=='parsed' else None
      identity.append({'review_alias':ra,'original_alias_PRIVATE':r['alias'],'platform':r['platform'],'arm':arm,'interface':interface,'version':version,'mapping':mapping,'annotation_status':a['status'],'identity_determinate':clear,'review_target_display_ids':a['target_display_ids'],'claimed_display_id':m_id,'claimed_mask_index_PRIVATE':claim,'claimed_mask_matches_review':bool(claim==reviewm) if clear and claim is not None else None,'point_hits_review_mask_even_if_overlap':point_target,'point_hit_mask_count':len(hits) if interface=='point' and pa['status']=='parsed' else None,'parse_status':pa['status'],'mapped':rd['mapped'],'mapping_status':rd['status'],'old_subset_contains_selected_mask':rd['old_subset_contains_selected_mask'],'full_group_nonempty':rd['full_group_nonempty'],'group_rep_selected_exactly_as_source':True,'selected_query_PRIVATE':rd['query'],'selected_query_in_review_target_group':rd['query'] in group if group else None,'final_IoU':values[arm],'source_IoU':values['source'],'gain_vs_source_pp':100*(values[arm]-values['source']),'query_changed':rd['query_changed'],'prior_seen_artifact_access':a['prior_seen_artifact_access'],'case_specific_prior_seen':a['case_specific_prior_seen'],'ambiguous_annotation_never_scored_correct':not clear,'nonexpert_hypothesis_not_gold':True})
 assert counters['raw_calls']==416 and counters['recovered_unique_position']==62 and counters['point_strict_fail']==63 and counters['point_compatible_fail']==1 and counters['selected_box_task_values']==2288
 assert kinds=={'singleton_point_only':22,'singleton_point_label':28,'singleton_point_2d_label':12},kinds
 assert len(malformed)==1 and malformed[0]['interface']=='point' and malformed[0]['independent_reason']=='malformed_JSON'
 # Recompute declared task aggregations and retain every negative comparison.
 summary=read(E38/'TASK_SUMMARY.json');aggs=[]
 for pl in ['drone','quad']:
  pr=[r for r in tasks if r['platform']==pl]
  for arm in ARMS:
   v=np.array([r['IoU'][arm] for r in pr]);d=v-np.array([r['IoU']['source'] for r in pr]);s={'n':len(pr),'mIoU_percent':100*float(v.mean()),'Acc25_percent':100*float((v>.25).mean()),'Acc50_percent':100*float((v>.5).mean()),'gain_vs_source_pp':100*float(d.mean()),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries']['source'] for r in pr)}
   for k,v in s.items():close(v,summary[pl][arm][k])
   aggs.append({'platform':pl,'arm':arm,**s})
 for arm in ARMS:close(summary['balanced_mIoU_percent'][arm],np.mean([r['mIoU_percent'] for r in aggs if r['arm']==arm]))
 nraw=read(E38/'ALL_NEGATIVES.json');tby={r['alias']:r for r in tasks}
 for n in nraw:
  r=tby[n['alias']];close(n['delta_pp'],100*(r['IoU'][n['arm']]-r['IoU'][n['reference']]),1e-10);assert n['delta_pp']<0
 # Identity aggregation distinguishes no claimed mask from a contradictory mask.
 idsumm=[]
 for arm in ARMS[3:]:
  for pl in ['drone','quad','both']:
   rs=[r for r in identity if r['arm']==arm and (pl=='both' or r['platform']==pl)];cc=[r for r in rs if r['identity_determinate']];claims=[r for r in cc if r['claimed_mask_index_PRIVATE'] is not None];match=[r for r in claims if r['claimed_mask_matches_review']];wrong=[r for r in claims if not r['claimed_mask_matches_review']];mapped=[r for r in match if r['mapped']]
   idsumm.append({'arm':arm,'platform':pl,'n':len(rs),'clear_identity_n':len(cc),'ambiguous_n':len(rs)-len(cc),'clear_claimed_mask_n':len(claims),'clear_claim_matches':len(match),'clear_claim_contradicts':len(wrong),'clear_no_mask_claim':len(cc)-len(claims),'point_target_mask_hits_including_overlap':sum(r['point_hits_review_mask_even_if_overlap'] is True for r in cc),'correct_2D_claim_mapped3D_n':len(mapped),'correct_2D_claim_but_empty3D_or_subset':len(match)-len(mapped),'mapped_wrong_identity_n':sum(r['mapped'] for r in wrong),'all_review_mapped':sum(r['mapped'] for r in rs),'all_review_query_changes':sum(r['query_changed'] for r in rs),'all_review_mean_IoU_percent':100*float(np.mean([r['final_IoU'] for r in rs])),'all_review_mean_gain_vs_source_pp':float(np.mean([r['gain_vs_source_pp'] for r in rs])),'all_review_improved':sum(r['gain_vs_source_pp']>1e-8 for r in rs),'all_review_harmed':sum(r['gain_vs_source_pp']< -1e-8 for r in rs),'all_review_wrong_identity_yet_IoU_gain':sum(r['gain_vs_source_pp']>1e-8 for r in wrong),'not_accuracy_gold':True})
 assert len(identity)==192 and sha(IDENT/'IDENTITY_RECORDS_PRIVATE.json')==ANNOTATION_SHA
 dump(OUT/'ALL416_SCHEMA_AUDIT_ROWS_PRIVATE.json',schema);dump(OUT/'MALFORMED_REMAINING_PRIVATE.json',malformed);dump(OUT/'ALL208_IDENTITY_TASK_ROWS_PRIVATE.json',tasks);dump(OUT/'TASK_AGGREGATES.json',aggs);dump(IDENT/'UNBLINDED_STRICT_COMPATIBLE_JOIN_PRIVATE.json',identity);dump(IDENT/'UNBLINDED_IDENTITY_SUMMARY.json',idsumm);dump(IDENT/'TARGET_GROUP_REP_DIAGNOSTICS_PRIVATE.json',groupdiagnostics)
 result={'state':'PASS','raw_answers_independently_parsed':416,'strict_point_parse_failures':63,'compatible_point_parse_failures':1,'recovered_unique_raw_points':62,'recovery_schema_categories':dict(kinds),'no_raw_coordinate_edits_or_model_calls':True,'SoM_strict_compatible_values_and_queries_identical':True,'strict_predictions_reproduce_R37':True,'readout_rows_recomputed':counters['readout_rows'],'task_values_independently_CPU_scored':counters['selected_box_task_values'],'negative_comparisons_values_checked':len(nraw),'identity24_seal_unchanged':ANNOTATION_SHA,'identity_clear_ambiguous':{'clear':18,'ambiguous':6},'identity_join_rows':len(identity),'identity_current_teacher_unblinded_only_after_R38seal':True,'identity_labels_not_used_in_parser_or_feedback':True,'recovered_point_original_pixel_maxerror':maxpoint,'remaining_bad_JSON_not_repaired':malformed,'balanced_mIoU_percent':summary['balanced_mIoU_percent'],'no_GPU_optimizer_or_VLM':True,'FINAL_not_read':True,'limitations':['R38 is a uniform structural repair afterR37 metrics exposure on already-developedA208, not new heldoutconfirmation.','Identity18clear/6ambiguous is a nonexpert assistant hypothesis; historical source/taskartifact exposure disclosed, and ambiguity is never scored ascorrect.','Point-hit comparisons use noisy existingmask support and cannot certify true physicalsubject identity fromIoU alone.','Group oracle values are postseal geometry diagnosis only, never runtime selection or newnegative labels.','This audit reuses recorded support matrices to validate threshold/unique groups; it does not rerun detector or geometricpoint preprocessing.'],'seconds':time.perf_counter()-started};dump(OUT/'IDENTITY_AUDIT_RESULT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
