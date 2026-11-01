"""CPU-only independent bbox-center replay against immutable prior24 identities."""
import sys,json,math,re,hashlib,collections,datetime,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'crossplatform/round30'),str(ROOT/'research/round5/audit')]
from common30 import read,dump,sha,verify,load_npz
from independent_core_v2 import candidate_ious,transform_gt
IDENT=ROOT/'research/round37/independent_identity';B37=ROOT/'crossplatform/round37/prepared208_attempt001';B39=ROOT/'crossplatform/round39/inference208_attempt002'
ANN='6e382dc7cd7cdc4257a8d4811c1b640ca0d466eaab6b2b72fb1dde916c099097'
def parse(s,wh):
 t=s.strip()
 for pat in [r'```(?:json)?\s*(.*?)\s*```',r'<answer>\s*(.*?)\s*</answer>']:
  m=re.fullmatch(pat,t,re.I|re.S)
  if m:t=m.group(1).strip()
 def pairs(ps):
  d={}
  for k,v in ps:
   if k in d:raise ValueError('duplicate_key')
   d[k]=v
  return d
 try:o=json.loads(t,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite')))
 except Exception:return 'parse_failure',None
 if isinstance(o,list):
  if len(o)!=1 or not isinstance(o[0],dict):return 'parse_failure',None
  o=o[0]
 if not isinstance(o,dict):return 'parse_failure',None
 k=set(o)&{'bbox_2d','bbox2d','bbox'}
 if len(k)!=1 or set(o)-k-{'label'} or ('label' in o and not isinstance(o['label'],str)):return 'parse_failure',None
 v=o[next(iter(k))]
 if v is None:return 'explicit_null',None
 if not isinstance(v,list) or len(v)!=4 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in v):return 'parse_failure',None
 x1,y1,x2,y2=v;w,h=wh
 return ('parsed' if x1<x2 and y1<y2 and 0<=x1<=w and 0<=x2<=w and 0<=y1<=h and 0<=y2<=h else 'invalid_box'),v
def main():
 verify(B39);verify(B37);assert sha(IDENT/'IDENTITY_RECORDS_PRIVATE.json')==ANN
 ann={r['review_alias']:r for r in read(IDENT/'IDENTITY_RECORDS_PRIVATE.json')};co=read(IDENT/'COHORT_PRIVATE.json');bindings={r['review_alias']:r for r in read(IDENT/'RENDER_BINDINGS_PRIVATE.json')};old=read(IDENT/'UNBLINDED_STRICT_COMPATIBLE_JOIN_PRIVATE.json');task={r['id_PRIVATE']:r for r in read(ROOT/'research/round39/experiment/ALL_ROWS_PRIVATE.json')};idx={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};joined=[]
 for c in co:
  r=c['row_PRIVATE'];a=ann[c['review_alias']];clear=a['status']=='clear_single_mask';case=read(B37/'cases_PRIVATE'/(r['id']+'.json'));z=load_npz(B37/'arrays_PRIVATE'/(r['id']+'.npz'));p=read(B39/'predictions_PRIVATE'/(r['id']+'.json'));ref=p['call_ref'];path=B39/ref['file'];assert sha(path)==ref['sha256'];raw=read(path)['result'];grid=raw['image_grid_thw'][0];wh=[grid[2]*14,grid[1]*14];assert wh==raw['model_resized_wh'];status,v=parse(raw['raw_answer_PRIVATE'],wh);assert status==raw['parsed']['status'] and v==raw['parsed']['value'];assert case['candidates']==bindings[c['review_alias']]['candidates_PRIVATE'];assert case['image_sha256']==bindings[c['review_alias']]['image_file_sha256']
  center=point=None;hits=[];m=None;maskstatus=status
  if status=='parsed':
   center=[(v[0]+v[2])/2,(v[1]+v[3])/2];point=[center[0]*case['image_size_wh'][0]/wh[0],center[1]*case['image_size_wh'][1]/wh[1]];hits=np.flatnonzero(z['masks_bool'][:,math.floor(point[1]),math.floor(point[0])]).tolist();m=hits[0] if len(hits)==1 else None;maskstatus='single_independent_mask' if m is not None else 'no_mask' if not hits else 'overlapping_multiple_masks'
  reviewm=next((x['mask_index'] for x in case['candidates'] if clear and x['target_id']==a['target_display_ids'][0]),None);targetg=case['full_groups'].get(str(reviewm),[]) if clear else [];claimid=next((x['target_id'] for x in case['candidates'] if x['mask_index']==m),None);q0=int(z['source_scores'].argmax());qs={}
  for mapping in ['fullmap','oldsubset']:
   arm='bbox_'+mapping;table=case['full_groups'] if mapping=='fullmap' else case['old_subset_groups'];g=table.get(str(m),[]) if m is not None else [];q=max(g,key=lambda j:(float(z['source_scores'][j]),-j)) if g else q0;qs[arm]=q;expected=p['readouts'][arm];st='mapped_original_query' if g else maskstatus if m is None else 'absent_from_old_subset' if mapping=='oldsubset' and str(m) not in table else 'mask_without_3D_group';assert expected['query']==q and expected['mask_index']==m and expected['mapped']==bool(g) and expected['group_size']==len(g) and expected['status']==st and expected['original_point']==point and expected['bbox_center_resized']==center
   joined.append({'review_alias':c['review_alias'],'original_alias_PRIVATE':r['alias'],'platform':r['platform'],'arm':arm,'interface':'bbox','mapping':mapping,'annotation_status':a['status'],'identity_determinate':clear,'review_target_display_ids':a['target_display_ids'],'claimed_display_id':claimid,'claimed_mask_index_PRIVATE':m,'claimed_mask_matches_review':bool(m==reviewm) if clear and m is not None else None,'point_hits_review_mask_even_if_overlap':reviewm in hits if clear and status=='parsed' else None,'point_hit_mask_count':len(hits) if status=='parsed' else None,'parse_status':status,'mapped':bool(g),'mapping_status':st,'old_subset_contains_selected_mask':m is not None and str(m) in case['old_subset_groups'],'full_group_nonempty':m is not None and bool(case['full_groups'].get(str(m),[])),'selected_query_PRIVATE':int(q),'selected_query_in_review_target_group':q in targetg if targetg else None,'final_IoU':task[r['id']]['IoU'][arm],'source_IoU':task[r['id']]['IoU']['source'],'gain_vs_source_pp':100*(task[r['id']]['IoU'][arm]-task[r['id']]['IoU']['source']),'query_changed':q!=q0,'prior_seen_artifact_access':a['prior_seen_artifact_access'],'case_specific_prior_seen':a['case_specific_prior_seen'],'ambiguous_annotation_never_scored_correct':not clear,'nonexpert_hypothesis_not_gold':True,'center_probes_mask_not_3D_center':True,'center_match_does_not_certify_bbox_semantics':True})
  ix=idx[r['id']];path=ix['meta_path']
  if path not in containers:assert sha(path)==ix['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));b=z['source_boxes'][list(qs.values())].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);vv=candidate_ious(b,gt)[0]
  for name,value in zip(qs,vv):assert abs(float(value)-task[r['id']]['IoU'][name])<1e-12
  for o in old:
   if o['review_alias']==c['review_alias']:assert abs(o['final_IoU']-task[r['id']]['IoU'][o['arm']])<1e-12
 assert len(joined)==48 and sha(IDENT/'IDENTITY_RECORDS_PRIVATE.json')==ANN
 summaries=[]
 for arm in ['bbox_fullmap','bbox_oldsubset']:
  for platform in ['drone','quad','both']:
   rr=[x for x in joined if x['arm']==arm and (platform=='both' or x['platform']==platform)];cc=[x for x in rr if x['identity_determinate']];claims=[x for x in cc if x['claimed_mask_index_PRIVATE'] is not None];match=[x for x in claims if x['claimed_mask_matches_review']];wrong=[x for x in claims if not x['claimed_mask_matches_review']]
   summaries.append({'arm':arm,'platform':platform,'n':len(rr),'clear_identity_n':len(cc),'ambiguous_n':len(rr)-len(cc),'clear_claimed_mask_n':len(claims),'clear_claim_matches':len(match),'clear_claim_contradicts':len(wrong),'clear_no_mask_claim':len(cc)-len(claims),'center_target_mask_hits_including_overlap':sum(x['point_hits_review_mask_even_if_overlap'] is True for x in cc),'correct_2D_claim_mapped3D_n':sum(x['mapped'] for x in match),'correct_2D_claim_but_empty3D_or_subset':sum(not x['mapped'] for x in match),'mapped_wrong_identity_n':sum(x['mapped'] for x in wrong),'all_review_mapped':sum(x['mapped'] for x in rr),'all_review_query_changes':sum(x['query_changed'] for x in rr),'all_review_mean_gain_vs_source_pp':float(np.mean([x['gain_vs_source_pp'] for x in rr])),'all_review_improved':sum(x['gain_vs_source_pp']>1e-8 for x in rr),'all_review_harmed':sum(x['gain_vs_source_pp']< -1e-8 for x in rr),'not_expert_or_new_blind_gold':True})
 dump(OUT/'BBOX_IDENTITY_JOIN_PRIVATE.json',joined);dump(OUT/'BBOX_IDENTITY_SUMMARY.json',summaries);dump(OUT/'BBOX_WITH_PRIOR_POINT_SOM_JOIN_PRIVATE.json',old+joined)
 result={'state':'PASS_IMMUTABLE24_IDENTITY_BBOX_JOIN','annotation_sha256':ANN,'prediction_seal_sha256':sha(B39/'SEALED.json'),'n':24,'clear':18,'ambiguous':6,'independently_reparsed_raw_calls':24,'independently_reconstructed_readouts':48,'independently_scored_task_values':48,'all_prior_point_som_identity_task_values_preserved':True,'no_new_image_review_or_annotation_change':True,'no_inference_GPU_optimizer_or_FINAL':True,'limits':['Nonexpert identity hypotheses with historical artifact exposure, not gold.','Center-to-mask matches do not disentangle bbox semantics, coordinate error, occlusion and mask-support faults.','No identities or task labels may filter supervision or change parsers.','Small exposedA208 subset, not independent confirmation.']};dump(OUT/'AUDIT_RESULT.json',result)
 seal={'state':'R39identity24_join_sealed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'annotation_sha256':ANN,'files':{f.name:sha(f) for f in OUT.iterdir() if f.is_file() and f.suffix in ['.json','.py'] and f.name!='SEALED.json'}};dump(OUT/'SEALED.json',seal);print(json.dumps({'audit':result,'summaries':summaries},indent=2))
if __name__=='__main__':main()
