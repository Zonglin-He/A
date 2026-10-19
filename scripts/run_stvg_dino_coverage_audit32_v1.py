"""Read-only historical observation provenance and target-text gate audit."""
import os,sys,time,hashlib,json,collections,types
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
D=ROOT/'artifacts/stvg_dino_coverage_audit32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/dino_coverage_audit32';VER=None
PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1';LEGACY=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2';PAPER=ROOT/'artifacts/decota_paper_experiments_v1'
FOCUS=['NV003','NV006','NV009','NV013','NV015']

def verify(full=False):
 global VER
 from main_method.runtime import PinVerifier,resource_guard
 p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
 if VER is None:VER=PinVerifier(l['pins'])
 VER.verify(full=full);resource_guard();return p

def prepare():
 assert not D.exists();D.mkdir();P.mkdir();rs=read(PRE/'PLAN_LOCK.json')['records'];assert len(rs)==len({r['parent']for r in rs})==32
 files=['scripts/run_stvg_dino_coverage_audit32_v1.py','protocols/stvg_dino_coverage_audit32_v1.md','methods/decota_final_simplified_v1/observations.py','scripts/run_decota_paper_main_v1.py','scripts/run_stvg_dino_supported_expanded256_recovery002.py','scripts/run_stvg_dino_endpoint_confirm32_v1.py','scripts/stvg_opd_legacy_input_bridge003.py','scripts/decota_paper_common_v1.py','artifacts/decota_paper_experiments_v1/RUNTIME_LOCK.json','artifacts/stvg_opd_paper_hc2_revision_v2/stages/P1_hc2/PREDICTION_BARRIER.json','artifacts/stvg_opd_paper_hc2_revision_v2/stages/P1_vidstg/PREDICTION_BARRIER.json','results/stvg_error_state/2026-10-19/reference_capacity32/RESULTS_PUBLIC.json','results/stvg_error_state/2026-10-19/reference_capacity32/SUPPORT_AND_CAPACITY_PUBLIC.json','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','configs/stvg_fine_center_dataset_candidate_v1.json','results/stvg_error_state/2026-10-19/fine_dataset_formal64/RESULTS_PUBLIC.json']
 for ds in ['hc2','vidstg']:files +=[str((PRE/('SOURCE_BARRIER_'+ds+'.json')).relative_to(ROOT)),str((PRE/('capture_'+ds+'.log')).relative_to(ROOT))]
 files +=[str((PRE/'GLOBAL_PREDICTION_BARRIER.json').relative_to(ROOT)),str((PRE/'measure_None.log').relative_to(ROOT))]
 for r in rs:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];job='t1_ours_hc2'if ds=='hc2'else't1_ours_vid';files +=[r['prior_input'],f'artifacts/decota_paper_experiments_v1/{ds}/subjects/{q:05}.json']
  for sub in ['source','E','predictions']:files +=[str((PRE/sub/(a+'.pt')).relative_to(ROOT)),str((PRE/sub/(a+'.receipt.json')).relative_to(ROOT))]
  files +=[str((PAPER/job/'inputs/clean'/f'{q:05}{ext}').relative_to(ROOT))for ext in ['.npz','.json']]
 p=dict(status='locked_read_only_cache_audit',records=rs,count=32,primary='NV16 with five specified missing original observations',focus=FOCUS,new_models_GPU_DINO_experts_training_candidates_adaptation_GT_access_scoring=0,CPU_cap_seconds=900,created_epoch=time.time(),formal64_excluded=True,no_automatic_followup=True)
 put(D/'PLAN_LOCK.json',p);put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in sorted(set(files))}));print('locked read-only coverage audit32',flush=True)

def canonical(ex):
 import torch,numpy as np
 def js(v):
  if torch.is_tensor(v):return v.cpu().tolist()
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,dict):return {str(k):js(x)for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [js(x)for x in v]
  if isinstance(v,np.generic):return v.item()
  return v
 out={k:js(v)for k,v in ex.items()if k!='observations'}
 if isinstance(ex['observations'],dict):out['observations']=[dict(view=v,position=po,probe=js(a['probe']),receipt=js(a['receipt']))for(v,po),a in sorted(ex['observations'].items())]
 else:out['observations']=js(ex['observations'])
 return out

def digest(ex):return hashlib.sha256(json.dumps(canonical(ex),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def audit():
 import torch,numpy as np
 from methods.decota_final_simplified_v1.observations import observations,parse_visual_words,uniform_positions
 torch.set_num_threads(4);assert not torch.cuda.is_initialized();p=verify(True);start=time.perf_counter();rows=[];private=[];checks=collections.Counter();originrt=read(PAPER/'RUNTIME_LOCK.json');srcfile='methods/decota_final_simplified_v1/observations.py';assert originrt['pins'][srcfile]==sha(ROOT/srcfile);prev=read(ROOT/'results/stvg_error_state/2026-10-19/reference_capacity32/RESULTS_PUBLIC.json');barriers={ds:read(LEGACY/'stages'/('P1_'+ds)/'PREDICTION_BARRIER.json')for ds in ['hc2','vidstg']};sb={ds:{f['path']:f['sha256']for f in read(PRE/('SOURCE_BARRIER_'+ds+'.json'))['files']}for ds in ['hc2','vidstg']};pb={f['path']:f['sha256']for f in read(PRE/'GLOBAL_PREDICTION_BARRIER.json')['files']}
 class ForbiddenExpert:
  touches=0
  def __getattr__(self,key):self.touches+=1;raise AssertionError('No expert/processor access permitted in skipped branch')
 for r in p['records']:
  verify();assert time.perf_counter()-start<900;a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];job='t1_ours_hc2'if ds=='hc2'else't1_ours_vid';f=PAPER/job/'inputs/clean'/f'{q:05}.npz';rc=read(f.with_suffix('.json'));assert sha(f)==rc['sha256']and rc['runtime_lock_sha256']==sha(PAPER/'RUNTIME_LOCK.json')
  with np.load(f,allow_pickle=False)as archive:md=json.loads(archive['metadata'].tobytes());native=archive['native_boxes'].copy()
  subpath=PAPER/ds/'subjects'/f'{q:05}.json';sub=read(subpath);assert md['subject_sha256']==sha(subpath)and not md['GT_read']and not sub['GT_read'];priorpath=ROOT/r['prior_input'];assert sha(priorpath)==barriers[ds]['inputs'][str(priorpath.relative_to(LEGACY))];old=torch.load(priorpath,map_location='cpu',weights_only=False);assert sha(f)==old['old_cache_sha256']and np.array_equal(native,old['native_boxes'].numpy());sourcepath=PRE/'source'/(a+'.pt');source=load(sourcepath);e=load(PRE/'E'/(a+'.pt'));conditionalpath=PRE/'predictions'/(a+'.pt');conditional=load(conditionalpath);assert sha(sourcepath)==sb[ds][str(sourcepath.relative_to(ROOT))];assert sha(conditionalpath)==pb[str(conditionalpath.relative_to(ROOT))]
  dd=[digest(v)for v in [md['expert'],old['expert'],source['expert']]];assert len(set(dd))==1;ex=source['expert'];assert np.array_equal(source['Source'].numpy(),native)and source['Native_interval']==md['interval']==old['interval'];assert source['frame_ids']==md['frame_ids']==old['frame_ids'];par=sub['parses'];ctx=par['context'];s1=par['old'];plan=uniform_positions(md['frame_ids'],md['indices'],4);assert plan==ex['positions4']and len(plan)==4;called=len(ex['observations']);accepted=sum(v['probe']['accepted']for v in ex['observations'].values());assert called==ex['new_DINO']==len(ex['actual_observation_positions'])and accepted==len(ex['anchors']['single4']);reason=collections.Counter(v['probe']['reason']for v in ex['observations'].values());receipts=[v['receipt']for v in ex['observations'].values()];assert all(v['rgb_sha256']and v['inputs']and v['forward_seconds']>=0 for v in receipts);checks['original_to_P1_to_Source_expert_exact']+=1;checks['origin_subject_and_runtime_hash_chain']+=1
  skip=(not ctx['eligible'])and not s1['phrase'];replay=False;question_word=ctx['tokens'][0]['lemma']if ctx['tokens']else None;wh_role=ctx['tokens'][0]['deprel']if ctx['tokens']else None;fallback_match=None
  if skip:
   assert called==0;fake=ForbiddenExpert();xx=observations(fake,par,np.zeros((len(md['frame_ids']),1,1,3),np.uint8),md['frame_ids'],md['indices'],audit=True);assert fake.touches==0 and digest(xx)==digest(ex);words=[types.SimpleNamespace(**w)for w in ctx['tokens']];parsed=parse_visual_words(words,leading_question=True);assert parsed==s1;fallback_match=True;replay=True;checks['no_model_exact_skip_replays']+=1
  complete_line=any(line.startswith(ds+' '+a+' ')for line in (PRE/('capture_'+ds+'.log')).read_text().splitlines());conditional_line=any(line.startswith(a+' ')for line in (PRE/'measure_None.log').read_text().splitlines());assert complete_line and conditional_line
  if a in FOCUS:
   assert skip and ctx['reason']=='unresolved_wh_category'and s1['rule']=='unknown_queried_entity'and question_word=='what';assert not conditional['active']and conditional['compute']['encoding_frames']==0 and not conditional['center_feedback']['anchors']and not conditional['sample_feedback']['details'];assert e['estimate']['support_frames']==0 and np.array_equal(e['E'],source['Source'].double().numpy());checks['focus_0_of4_before_DINO_not_rejection']+=1
  if called:category='executed_with_some_rejected_feedback'if called>accepted else'executed_all_feedback_accepted'
  elif skip:category='pre_call_target_text_gate_skip'
  else:category='unresolved_missing_evidence'
  prevrow=next(v for v in prev['rows']if v['alias']==a);row=dict(alias=a,dataset=ds,focus=a in FOCUS,classification=category,planned_frames=len(plan),original_DINO_forwards=called,accepted_frames=accepted,rejected_frames=called-accepted,rejection_reasons=dict(reason),cached_context_eligible=ctx['eligible'],context_reason=ctx['reason'],fallback_phrase_empty=not bool(s1['phrase']),fallback_rule=s1['rule'],question_word=question_word if a in FOCUS else None,question_role=wh_role if a in FOCUS else None,canonical_expert_three_layer_equal=True,origin_receipt_and_subject_hashes=True,original_input_present=True,capture_completed_log=True,conditional_completed_log=True,pre_call_CPU_replay_exact=replay,expert_model_processor_touched_by_replay=False,source_fresh_original_DINO_calls=source['compute']['new_original_DINO_calls'],later_conditional_encoding_frames=conditional['compute']['encoding_frames'],later_conditional_active=conditional['active'],later_center_logical_queries=conditional['compute']['center_logical_queries'],later_sample_logical_queries=conditional['compute']['sample_logical_queries'],new_DINO_calls_in_P1_cache=old['new_DINO_calls'],saved_GT_capacity_readback_only=dict(known_GT_anchor_positions=prevrow['known_planned_GT'],GT_E_v_gain_pp=100*(prevrow['values']['GT_E_known4']['v']-prevrow['values']['E_MBR']['v']),E_v=prevrow['values']['E_MBR']['v'],E_t=prevrow['values']['E_MBR']['t']))
  rows.append(row);private.append(dict(alias=a,origin_npz=str(f.relative_to(ROOT)),subject=str(subpath.relative_to(ROOT)),P1_input=r['prior_input'],expert_canonical_sha256=dd[0],source_sha256=sha(sourcepath),origin_npz_sha256=sha(f)));checks['complete_capture_and_conditional_log_records']+=1;print(a,category,'planned/called/accepted',len(plan),called,accepted,flush=True)
 groups={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(parents=len(rr),planned_frames=sum(r['planned_frames']for r in rr),actual_original_DINO_frames=sum(r['original_DINO_forwards']for r in rr),accepted_frames=sum(r['accepted_frames']for r in rr),rejected_frames=sum(r['rejected_frames']for r in rr),unspent_frames=sum(r['planned_frames']-r['original_DINO_forwards']for r in rr),pre_call_skipped_queries=sum(r['classification']=='pre_call_target_text_gate_skip'for r in rr),all_four_called_queries=sum(r['original_DINO_forwards']==4 for r in rr),all_four_called_no_accepted_queries=sum(r['original_DINO_forwards']==4 and r['accepted_frames']==0 for r in rr),later_conditional_encoding_frames=sum(r['later_conditional_encoding_frames']for r in rr),full_Vid16_GT_E_capacity_contribution_from_no_original_DINO_pp=sum(r['saved_GT_capacity_readback_only']['GT_E_v_gain_pp']for r in rr if r['original_DINO_forwards']==0)/len(rr))
 assert {r['alias']for r in rows if r['dataset']=='vidstg'and r['original_DINO_forwards']==0}==set(FOCUS);assert groups['vidstg']['actual_original_DINO_frames']==44 and groups['vidstg']['accepted_frames']==42;assert not torch.cuda.is_initialized();verify(True)
 put(D/'PRIVATE_LINEAGE.json',dict(rows=private));put(P/'RESULTS_PUBLIC.json',dict(status='completed_read_only_cache_audit',count=32,focus=FOCUS,rows=rows,groups=groups,checks=dict(checks),origin_observation_code_matches_original_runtime=True,no_new_GT_read_or_task_scoring=True,no_new_model_expert_DINO_GPU_training_or_adaptation=True,interpretation='five Vid parents skipped before expert invocation because unresolved what referent plus empty phrase, not rejected detection or cache omission'))
 put(P/'COST_BREAKDOWN.json',dict(CPU_audit_seconds=time.perf_counter()-start,new_GPU_seconds=0,total_Source_captures_completed=0,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,new_expert_forwards_training_scoring_GT_access=0,previous_acquisition_and_GT_oracle_cost_excluded=True));put(D/'AUDIT_COMPLETION.json',dict(status='complete',count=32,model_calls=0,CUDA_initialized=False));print('coverage audit32 complete',groups,dict(checks),flush=True)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','audit']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  D.mkdir(exist_ok=True,parents=True);put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=a.stage,error_type=type(e).__name__,error=str(e),no_models_started=True,partial_preserved=True));raise
