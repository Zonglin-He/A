"""Descriptive fixed-prediction mask/priority audit. No new predictions or GT boxes."""
from audit48_independent import *
from official_span_independent48 import choose_official_charspan
from collections import Counter

def main():
 rr,member=cohort();idx=old_index(rr);classes=caption_classes(rr,idx);plan=read(EXP/'PLAN_LOCK.json');P=ROOT/plan['prediction_output'];verify(P);syn=dictionary_from_source((ROOT/'crossplatform/round3/p0/upstream/src/joint_det_dataset.py').read_text(),'M3ED_SYNONYMS');tasks={r['id_PRIVATE']:r for r in read(HERE/'ALL336_TASK_ROWS_PRIVATE.json')};out=[]
 for r in rr:
  p=load(P/'predictions_PRIVATE'/(r['id']+'.npz'));c=read(P/'cases_PRIVATE'/(r['id']+'.json'));span=c['proxy_span_actual'];official=c['official'];op=official['span_actual'];names=syn.get(classes[r['id']],[classes[r['id']]]);proxyword=r['caption'][span[0]:span[1]].casefold() if span else None;word=official.get('matched_word');same=bool(np.array_equal(p['positive_masks'][0],p['positive_masks'][1]));available=proxyword in [w.casefold() for w in names] if proxyword else False
  if same:reason='same_positive_mask'
  elif op is None:reason='official_no_match_fallback'
  elif span is None:reason='current_all_content_fallback_vs_official_span'
  elif not available:reason='current_proxy_term_outside_annotation_class_synonym_list'
  elif op[0]>span[0]:reason='official_synonym_list_priority_prefers_later_available_synonym'
  elif op[0]<span[0]:reason='official_synonym_precedes_current_lexicon_term'
  else:reason='same_start_different_positive_span'
  t=tasks[r['id']];d=t['IoU'][ARMS[1]]-t['IoU'][ARMS[0]];out.append({'id_PRIVATE':r['id'],'alias':r['alias'],'split':member[r['id']],'platform':r['platform'],'official_current_mask_equal':same,'current_parser_mask_equal':bool(np.array_equal(p['positive_masks'][0],p['positive_masks'][2])),'literal_remap_mask_equal':bool(np.array_equal(p['positive_masks'][1],p['positive_masks'][3])),'official_current_query_changed':int(p['queries'][0])!=int(p['queries'][1]),'reason':reason,'official_synonym_priority_index':names.index(word) if word in names else None,'proxy_term_available_in_same_official_list':available,'task_delta_pp_offline_only':float(d*100),'task_direction':'improved' if d>1e-10 else 'worsened' if d< -1e-10 else 'equal','no_new_prediction_score_or_rule_change':True})
 summary={}
 for split in ['ALL336','D208','V64','M64']:
  summary[split]={}
  for pl in ['all','drone','quad']:
   part=[x for x in out if (split=='ALL336' or x['split']==split) and (pl=='all' or x['platform']==pl)];summary[split][pl]={'n':len(part),'official_current_mask_differences':sum(not x['official_current_mask_equal'] for x in part),'current_parser_mask_differences':sum(not x['current_parser_mask_equal'] for x in part),'literal_remap_mask_differences':sum(not x['literal_remap_mask_equal'] for x in part),'official_current_query_changes':sum(x['official_current_query_changed'] for x in part),'reason_counts':dict(Counter(x['reason'] for x in part)),'change_outcomes':dict(Counter(x['task_direction'] for x in part if x['official_current_query_changed']))}
 reasons=[]
 for reason in sorted({x['reason'] for x in out}):
  part=[x for x in out if x['reason']==reason];reasons.append({'reason':reason,'n':len(part),'query_changes':sum(x['official_current_query_changed'] for x in part),'outcomes':dict(Counter(x['task_direction'] for x in part)),'sum_task_delta_pp_offline':sum(x['task_delta_pp_offline_only'] for x in part),'GTclass_annotation_informed_not_subject_identity_gold':True})
 dump(HERE/'ALL336_FIXED_MASK_DIAGNOSTIC_PRIVATE.json',out);dump(HERE/'FIXED_MASK_PRIORITY_SUMMARY.json',{'state':'DESCRIPTIVE_FIXED_PREDICTION_REUSE_NO_NEW_TASK_SCORE','split_platform':summary,'reason_strata':reasons,'parser_rule_unchanged':True,'FINAL_access':False});print(json.dumps(summary['ALL336']))
if __name__=='__main__':main()
