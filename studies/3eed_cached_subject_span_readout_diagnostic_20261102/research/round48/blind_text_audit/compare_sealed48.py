"""Fixed blind24 TEXT ROLE join only; no scores, boxes, metadata, NLP or models."""
from pathlib import Path
import json,hashlib,datetime,collections
import numpy as np
R=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent;B=R/'crossplatform/round48/readout336_attempt001';P=R/'crossplatform/round48/parse336_attempt001'
PRED_SHA='f40374b6f47b013d865748e503f763a4a412fbc24a77a2dd587ca6562802d67d'
ANNO_SHA='5fa0512998f1372e89cd3dc4b32ee21129cf53ad0195f7e6aa1aa87c04e92266'
COHORT_SHA='7b8e65ce84274b61fa80412c199f9a9c042f3be8556b1a2c58a9d645b9fbe8ea'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def union(spans):
 out=[]
 for a,b in sorted(spans):
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out

def main():
 assert sha(B/'SEALED.json')==PRED_SHA and sha(D/'ANNOTATIONS_PRIVATE.json')==ANNO_SHA and sha(D/'COHORT_PRIVATE.json')==COHORT_SHA
 sealed=read(B/'SEALED.json');ps=read(P/'SEALED.json');annseal=read(D/'ANNOTATIONS_SEALED.json');assert sealed['parser_seal_sha256']==sha(P/'SEALED.json') and annseal['annotation_sha256']==ANNO_SHA
 cohort=read(D/'COHORT_PRIVATE.json');anns={r['id_PRIVATE']:r for r in read(D/'ANNOTATIONS_PRIVATE.json')};assert len(cohort)==len(anns)==24;out=[];bridge=[]
 for r in cohort:
  sid=r['id'];ann=anns[sid];caption=r['caption'];actual=caption+' . not mentioned';cf=B/'cases_PRIVATE'/(sid+'.json');zf=B/'predictions_PRIVATE'/(sid+'.npz');pf=P/'cases_PRIVATE'/(sid+'.json')
  for f,base,seal in [(cf,B,sealed),(zf,B,sealed),(pf,P,ps)]:assert sha(f)==seal['files'][f.relative_to(base).as_posix()]
  c=read(cf);parser=read(pf);assert c['id_PRIVATE']==parser['id_PRIVATE']==sid and c['parser']==parser and c['GTbbox_read'] is False and c['actual_text_sha256']==hashlib.sha256(actual.encode()).hexdigest();assert parser['caption_sha256']==hashlib.sha256(caption.encode()).hexdigest()
  # NPZ is lazy: only these three allowed arrays are materialized, never scores/queries/boxes.
  with np.load(zf,allow_pickle=False) as f:offsets=f['actual_offsets'].copy();masks=f['positive_masks'].copy();arms=f['arm_names'].tolist()
  assert masks.shape==(4,len(offsets)) and offsets.shape==(len(offsets),2) and masks.dtype==np.bool_ and len(arms)==5
  assert all(0<=a<=b<=len(actual) for a,b in offsets)
  clear=ann['status']=='CLEAR_INTENDED_SUBJECT_HEAD_IN_TEXT';span=ann['subject_head_charspan'];lo,hi=span;assert caption[lo:hi]==ann['subject_head_text_PRIVATE'];expected=(offsets[:,1]>lo)&(offsets[:,0]<hi)&(offsets[:,1]>offsets[:,0])
  specs=[('CURRENT',0,c['proxy_span_actual'],False),('official_LITERAL_actual_offsets',1,None,c['official_literal_fallback_independent']),('text_only_subject_parser',2,parser['head_span'] if parser['status']=='FOUND' else None,c['parser_fallback_to_current']),('official_charspan_REMAP',3,c['official'].get('span_actual'),c['official_remap_fallback_independent'])]
  for name,j,nominal,fallback in specs:
   selected=masks[j];spans=union([list(map(int,x)) for x in offsets[selected] if x[1]>x[0]]);coverage=sum(max(0,min(b,hi)-max(a,lo)) for a,b in spans)/(hi-lo) if clear else None
   nominal_exact=nominal==span if nominal is not None and clear else None;nominaltext=actual[nominal[0]:nominal[1]] if nominal is not None else None;sameword_other=bool(clear and nominal is not None and nominal!=span and nominaltext.casefold()==ann['subject_head_text_PRIVATE'].casefold());actual_other=any([a,b]!=span and actual[a:b].casefold()==ann['subject_head_text_PRIVATE'].casefold() for a,b in spans)
   out.append({'review_alias':r['review_alias'],'id_PRIVATE':sid,'platform':r['platform'],'stratum':r['stratum'],'arm':name,'blind_status':ann['status'],'blind_head_charspan_PRIVATE':span,'nominal_charspan_PRIVATE':nominal,'nominal_span_exact_blind':nominal_exact,'effective_token_charspans_PRIVATE':spans,'effective_head_character_coverage':coverage,'effective_contains_full_blind_head':bool(coverage==1) if clear else None,'effective_exact_blind_head_token_mask':bool(np.array_equal(selected,expected)) if clear else None,'effective_exact_blind_head_character_union':spans==[span] if clear else None,'same_head_word_at_different_nominal_position_not_credit':sameword_other,'same_head_word_at_other_effective_position_not_credit':actual_other,'selected_noncaption_tokens':any(b>len(caption) for a,b in spans),'declared_UNKNOWN_or_fallback':bool(fallback or (name=='text_only_subject_parser' and parser['status']!='FOUND') or (name.startswith('official') and c['official']['status']!='matched')),'effective_fallback_to_current':bool(fallback),'mask_positive_tokens':int(selected.sum()),'literal_uses_actual_offsets_not_official_raw_span_as_if_aligned':name=='official_LITERAL_actual_offsets','nonexpert_text_roles_not_visual_or_GTidentity_gold':True,'no_TASK_score_or_GTbox_loaded':True})
  bridge.append({'review_alias':r['review_alias'],'platform':r['platform'],'arm':arms[4],'text_span_metric':'NOT_APPLICABLE_NO_INDEPENDENT_TEXT_MASK','not_counted_as_fifth_span_prediction':True})
 summary=[]
 for pl,st in [('all','all')]+[(p,'all') for p in ['drone','quad']]+[(p,s) for p in ['drone','quad'] for s in ['explicitObjecttype','freeform']]:
  for name in [x[0] for x in specs]:
   rr=[x for x in out if x['arm']==name and (pl=='all' or x['platform']==pl) and (st=='all' or x['stratum']==st)];clear=[x for x in rr if x['blind_status']=='CLEAR_INTENDED_SUBJECT_HEAD_IN_TEXT'];summary.append({'platform':pl,'stratum':st,'arm':name,'n':len(rr),'clear_blind_n':len(clear),'ambiguous_or_unknown_blind_n':len(rr)-len(clear),'nominal_exact_blind_head':sum(x['nominal_span_exact_blind'] is True for x in clear) if name!='official_LITERAL_actual_offsets' else None,'nominal_metric_for_literal':'NOT_APPLICABLE_actualmaskoffsets_are_measurement' if name=='official_LITERAL_actual_offsets' else 'exactunmodifiedcaptioncharspan','effective_exact_head_token_mask':sum(x['effective_exact_blind_head_token_mask'] is True for x in clear),'effective_full_head_coverage':sum(x['effective_contains_full_blind_head'] is True for x in clear),'effective_exact_head_character_union':sum(x['effective_exact_blind_head_character_union'] is True for x in clear),'declared_UNKNOWN_or_fallback':sum(x['declared_UNKNOWN_or_fallback'] for x in rr),'same_head_word_wrong_nominal_location':sum(x['same_head_word_at_different_nominal_position_not_credit'] for x in rr),'same_head_word_wrong_effective_location':sum(x['same_head_word_at_other_effective_position_not_credit'] for x in rr),'selected_noncaption_tokens':sum(x['selected_noncaption_tokens'] for x in rr)})
 assert sha(D/'ANNOTATIONS_PRIVATE.json')==ANNO_SHA and sha(D/'COHORT_PRIVATE.json')==COHORT_SHA and sha(B/'SEALED.json')==PRED_SHA
 dump(D/'TEXT_ROLE_COMPARISON_PRIVATE.json',out);dump(D/'TEXT_ROLE_SUMMARY.json',summary);dump(D/'NATIVE_BRIDGE_NOT_A_SPAN_PREDICTION.json',bridge)
 receipt={'state':'COMPLETE_FIXED24_SEALED_TEXT_ROLE_COMPARISON_ONLY','n':24,'text_span_arms':4,'text_rows':96,'native_bridge_not_applicable_rows':24,'prediction_seal_sha256':PRED_SHA,'original_blind_annotation_sha256_unchanged':ANNO_SHA,'cohort_sha256_unchanged':COHORT_SHA,'parse_seal_sha256':sha(P/'SEALED.json'),'clear_blind_n':sum(x['status']=='CLEAR_INTENDED_SUBJECT_HEAD_IN_TEXT' for x in anns.values()),'no_GTclass_bbox_IoU_task_results_metadata_FINAL_media_XYZ_or_new_NLP_model_access':True,'only_materialized_NPZ_arrays':['actual_offsets','positive_masks','arm_names'],'not_expertgold_and_historical_text_exposure_disclosed':True,'not_feedback_for_parser_repair_or_selection':True,'same_word_wrong_position_never_credited':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};dump(D/'COMPARISON_RECEIPT.json',receipt);print(json.dumps({'receipt':receipt,'summary_all':[x for x in summary if x['platform']=='all']},indent=2))
if __name__=='__main__':main()
