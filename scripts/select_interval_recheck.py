"""Metadata-only confirmation availability audit; never loads GT or models."""
from pathlib import Path
import hashlib,json,os,subprocess
from collections import Counter,defaultdict
import numpy as np,torch
ROOT=Path.cwd();OUT=ROOT/'outputs/interval_pool';OLD='7a0a8131733d124c9262db0678260c46f095ead4'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
def historical(p):
 b=subprocess.check_output(['git','show',OLD+':'+p],cwd=ROOT);return json.loads(b),hashlib.sha256(b).hexdigest()
def main():
 assert not(OUT/'confirmation_selection.json').exists(),'Preserve any existing frozen roster; use a new output directory for reproduction'
 torch.set_num_threads(1);formal=ROOT/'outputs/formal/state.json';state_before=sha(formal);state=read(formal)
 assert state['status']=='paused_by_user'and state['jobs']['tastvg-hc2']['status']=='scored'and state['jobs']['tastvg-vidstg']['status']=='paused_by_user'
 devpath=ROOT/'data/evaluation/development.jsonl';dev=rows(devpath);devby=defaultdict(list)
 for r in dev:devby[r['dataset']].append(r)
 history=[]
 for ds in ['hc2','vidstg']:
  p=f'results/tastvg_best_full/2026-10-01/registration/{ds}/COHORT.json';d,h=historical(p)
  assert d['historically_exposed']is True and d['full_official_cohort']is True
  history.append(dict(commit=OLD,path=p,sha256=h,dataset=ds,sources=d['sources'],queries=d['queries'],historically_exposed=True,GT_used_for_selection=d['GT_used_for_selection'],interpretation='Historical full-domain task-output exposure; this registration is not itself a complete per-query completion log'))
 p='results/stvg_error_state/2026-10-20/spatial_strong_bidirectional256/EXPOSURE_PUBLIC.json';d,h=historical(p)
 history.append(dict(commit=OLD,path=p,sha256=h,scope=d['scope'],roster_hashes=d['roster_hashes']))
 p='results/stvg_error_state/2026-10-19/temporal_increment_confirm109/EXPOSURE_PUBLIC.json';d,h=historical(p)
 history.append(dict(commit=OLD,path=p,sha256=h,parents=d['parents'],not_untouched_method_test=d['not_untouched_method_test'],exposure=d['exposure']))
 # All post-cleanup locks expose metadata/ids only; no labels or scores read.
 panels=[];dev_ids={r['id']for r in dev}
 for p in sorted((ROOT/'outputs').glob('*/lock.json')):
  d=read(p);ids=d.get('ids')or [x.get('id')for x in d.get('inputs',[])if isinstance(x,dict)and x.get('id')]
  if not ids:continue
  panels.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),planned_ids=len(ids),known_development_aliases=sum(x in dev_ids for x in ids),other_aliases=[x for x in ids if x not in dev_ids],note='Lock is planning metadata; actual-completed scopes are separately documented; not all planned entries imply execution'))
 chosen_by={};summary={};selected_signatures=defaultdict(list)
 for ds in ['hc2','vidstg']:
  manifest=ROOT/f'data/evaluation/{ds}.jsonl';full=rows(manifest);devparents={r['metadata']['source']for r in devby[ds]};byparent=defaultdict(list)
  for line,r in enumerate(full,1):byparent[r['metadata']['source']].append((line,r))
  available=[p for p in sorted(byparent,key=str)if p not in devparents]
  selected=[]
  for parent in available[:32]:
   line,r=min(byparent[parent],key=lambda x:str(x[1]['id']));sig=(ds,r['metadata']['caption'],tuple(r['metadata']['frame_ids']))
   row=dict(alias=f'{ds.upper()}_RECHECK{len(selected)+1:03d}',dataset=ds,parent=parent,query_id=r['id'],manifest_path=str(manifest.relative_to(ROOT)),manifest_line=line,input_record_sha256=hashlib.sha256(canonical(r)).hexdigest(),record=r,
    current_development_parent=False,historical_full_domain_exposure=True,strict_independent_confirmation_eligible=False,all_historical_small_panel_membership_fully_reconstructible=False,
    original_video_id=r['metadata']['original_video_id'],video_sha256=r['metadata']['video_sha256'],GT_read=False)
   pred=ROOT/f'outputs/formal/tastvg/{ds}/predictions/{r["id"]}.npz';receipt=pred.with_suffix('.json')
   cache=dict(Source_E_prediction_only=None,Source_spatial_inputs=None,native_mask_logits=None,six_layer_hidden_or_logits=None,complete_reusable_capture=False,formal_teacher_candidates=[])
   if pred.exists()and receipt.exists():
    pr=read(receipt);assert pr['labels_read']is False and sha(pred)==pr['prediction_sha256']
    assert pr['request_sha256']==row['input_record_sha256']
    with np.load(pred,allow_pickle=False)as z:
     assert set(['Source','E_Ref','frame_ids','positions8'])<=set(z.files)
     assert np.array_equal(z['frame_ids'],r['metadata']['frame_ids']);row['cached_positions8']=z['positions8'].tolist()
     assert not any('hidden'in k or 'logit'in k or 'mask'in k or 'feature'in k for k in z.files)
    cache['Source_E_prediction_only']=dict(path=str(pred.relative_to(ROOT)),sha256=pr['prediction_sha256'],receipt_path=str(receipt.relative_to(ROOT)),receipt_sha256=sha(receipt),not_sufficient_for_BoxFixed_or_six_layer_pool=True)
   row['cache']=cache;row['prior_exposure']=dict(current_formal_prediction_exists=cache['Source_E_prediction_only']is not None,current_formal_scored=ds=='hc2',historical_full_domain_results_exposed=True,current_formal_inference_is_not_itself_GT_exposure=True)
   selected.append(row);selected_signatures[sig].append(row)
  chosen_by[ds]=selected
  summary[ds]=dict(direction='VidSTG_to_HC2'if ds=='hc2'else'HC2_to_VidSTG',official_queries=len(full),official_parents=len(byparent),current_development_queries=len(devby[ds]),current_development_parents=len(devparents),official_parents_overlapping_development=len(set(byparent)&devparents),outside_current_development_parent_pool=len(available),selected_development_recheck_queries=len(selected),selected_parents=len(selected),strict_verified_unexposed_confirmation_parents=0,
   current_formal_status=state['jobs']['tastvg-'+ds]['status'],current_formal_GT_scored=(ds=='hc2'),current_formal_predictions_observed=len(list((ROOT/f'outputs/formal/tastvg/{ds}/predictions').glob('*.json'))),input_manifest_sha256=sha(manifest))
 # Indexed only by existing context, never decode RGB or instantiate a teacher.
 teacher_context_count=0;matched_count=0;loaded={}
 for p in (ROOT/'outputs/formal/teacher_cache').glob('*.json'):
  d=read(p);ctx=d.get('context',{});sig=(ctx.get('dataset'),ctx.get('caption'),tuple(ctx.get('frame_ids',[])))
  teacher_context_count+=1
  if sig not in selected_signatures:continue
  for row in selected_signatures[sig]:
   if 'cached_positions8'in row and list(ctx['positions'])!=row['cached_positions8']:continue
   pt=p.with_suffix('.pt');assert pt.is_file()and sha(pt)==d['sha256']
   if str(pt)not in loaded:
    z=torch.load(pt,map_location='cpu',weights_only=False);loaded[str(pt)]={k:bool(z.get(k)is not None)for k in ['boxes','native_mask_logits','raw_out','pred_masks']}
   flags=loaded[str(pt)]
   row['cache']['formal_teacher_candidates'].append(dict(path=str(pt.relative_to(ROOT)),sha256=d['sha256'],receipt_path=str(p.relative_to(ROOT)),receipt_sha256=sha(p),fields_present=flags,positions_match_existing_prediction=('cached_positions8'in row),runtime_context_fingerprint=ctx['fingerprint'],RGB_context_sha256=ctx['RGB_sha256'],require_decode_binding_before_reuse=True))
   matched_count+=1
 for ds,selected in chosen_by.items():
  s=summary[ds];s['existing_formal_prediction_queries']=sum(r['cache']['Source_E_prediction_only']is not None for r in selected)
  s['Source_spatial_inputs_queries']=0;s['six_layer_hidden_or_logits_queries']=0;s['native_masks_available_queries']=sum(any(x['fields_present']['native_mask_logits']for x in r['cache']['formal_teacher_candidates'])for r in selected)
  s['boxes_only_teacher_cache_queries']=sum(bool(r['cache']['formal_teacher_candidates'])and not any(x['fields_present']['native_mask_logits']for x in r['cache']['formal_teacher_candidates'])for r in selected)
  s['complete_reusable_capture_queries']=0;s['shared_native224_capture_required_queries']=len(selected);s['eight_frame_teacher_with_masks_required_queries']=len(selected)-s['native_masks_available_queries']
 assert sha(formal)==state_before
 result=dict(schema='interval_pool_confirmation_selection_audit_v1',status='no_verified_pristine_confirmation_pool; deterministic_development_recheck_roster_prepared',GT_read=False,new_model_calls=0,new_GPU_calls=0,new_RGB_reads=0,new_scores=0,
  instruction_sha256='51738741a3fc5a95544d7b966098a367610dfeaee3e51f582d3383ad66e0eb3f',development_manifest_sha256=sha(devpath),formal_state_sha256=state_before,formal_state_unchanged=True,
  selection_rule='Use current official label-free manifests only; exclude all current HC128/Vid128 parent sources; lexicographically sort parent IDs; choose lexicographically first query ID within each parent; take first32 parents. No scores, GT, cache presence or model output influence membership.',
  strict_confirmation=dict(eligible_queries=0,eligible_parents=0,reason='Both full official domains have documented historical task-output exposure; HC current full1902 also scored. Complete identities for every earlier method-development panel were retired in cleanup, so exclusion from all actual historical panels cannot be proven. Outside current128 is not independent confirmation.'),
  recommended_label='Frozen parent-separated development recheck, not historically untouched independent confirmation',
  recommendation='Freeze the new interval-mixture rule and its config before executing these32+32; retain all outcomes. Do not claim these rosters exclude all historical development/exposure panels.',
  historical_provenance=history,post_cleanup_lock_audit=panels,summary=summary,development_recheck_selection=chosen_by,
  teacher_cache_contexts_indexed=teacher_context_count,teacher_cache_matching_records=matched_count,
  existing_cache_scope=dict(student_cache='outputs/layout_reward_dev/student_cache:128 HC development .pt only',native_masks='outputs/layout_reward_dev/research/native_masks:128 HC development .pt only',temporal_capture='outputs/tube_temporal_hc64/captures:HC64 development subset only',reference='outputs/reference/predictions:inherited exposed256 result NPZ only',formal='outputs/formal/tastvg/{dataset}/predictions:final predictions/intervals, not rawSource fusion grid or all6 temporal hidden'),
  execution_boundary='All selected records need one new shared224 Source capture; boxes-only old teacher cache cannot provide native masks. At most one eight-frame expert call per missing-mask query, shared by Source/oldhull/newmix arms, no448/conditionaldecoder/training/formalresume. Existing prediction cost remains historical.',
  limitations=['Historical COHORT registration describes exposure and scope, not complete per-query run completion. Independent confirmation is conservatively not asserted.','No new label file was opened; formal scoring status is read from savedstate, and historical exposure from archived metadata/records.','Training/pretraining identity-level overlap is not fully reconstructible; these records cannot certify absence of all upstream exposure.'])
 OUT.mkdir(exist_ok=True);p=OUT/'confirmation_selection.json';p.write_text(json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),summary=summary,teacher_contexts=teacher_context_count,matching_records=matched_count,status=result['status']),indent=2))
if __name__=='__main__':main()
