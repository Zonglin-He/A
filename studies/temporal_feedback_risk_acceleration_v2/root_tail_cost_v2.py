"""Original scalar/tail report with actual parallel-worker cost lineage."""
import runtime_v2 as v
from pathlib import Path
import sys,hashlib,json
def accelerated_costs(ds,summary):
 assert ds=='vidstg';d=v.c.D/ds;ad=v.D;state=v.c.read(ad/'state_v3.json');assert state['status']=='Vid_whole_official_scored_and_allarray_audited_pending_root_semantic_cost_image_publication'
 old=v.c.read(ad/'preserved_original_serial_score/preservation.json');assert v.c.read(ad/'original_owned_exit.json')['actual_owned_exit_verified']
 pred=v.c.read(d/'sealed.json');assert v.c.read(v.c.D/'vidstg_predict_exit.json')['code']==0
 costs=dict(predict=dict(worker=pred['worker'],elapsed_wall_seconds=pred['elapsed_wall_seconds'],process_CPU_time_independently_measured=False))
 for stage in ['score','audit']:
  workers=[]
  for i in [0,1]:
   folder=ad/stage/str(i);receipt=v.c.read(folder/'receipt.json');exitfile=ad/(f'score_{i}_exit_v3.json'if stage=='score'else f'audit_{i}_exit.json');exit=v.c.read(exitfile);assert exit['code']==0;workers.append(dict(worker=receipt['worker'],queries=receipt['queries'],elapsed_wall_seconds=receipt['elapsed_wall_seconds'],actual_exit_code=0,exit_evidence_sha256=v.c.sha(exitfile)))
  mergeexit=ad/f'{stage}_merge_exit_v3.json';assert v.c.read(mergeexit)['code']==0;canonical=v.c.read(d/('score_receipt.json'if stage=='score'else'reporting_receipt.json'))
  costs[stage]=dict(actual_parallel_shards=workers,merge_worker=canonical['worker'],whole_phase_elapsed_wall_seconds=canonical['elapsed_wall_seconds'],parallel_workers_not_additive=True,merge_exit_code=0,process_CPU_time_independently_measured=False,original_full_report_suffix_unchanged=True)
 costs['prior_uncheckpointed_serial_score']=dict(original_processes=old['before'],completed_query_lower_bound=old['completed_lower_bound'],official_metric_checks_lower_bound=8*old['completed_lower_bound'],no_perquery_scalar_checkpoint_available=True,not_claimed_reused=True,actual_intentional_exit_after_new_full_audit=True,OS_wait_code_unavailable_not_fabricated=True)
 costs['engineering_profile_and_equivalence']=dict(Vid_official_checks=32,HC_official_checks=32,new_inference_calls=0,no_full_HC_rerun=True)
 return costs
def run():
 src=(v.OLD/'root_tail_cost_v1.py').read_text();start=src.index(' costs={}\n');end=src.index(" atomic(p/'actual_cost_segments_v1.json'",start);old=src[start:end];new=' costs=accelerated_costs(ds,summary)\n';changed=src[:start]+new+src[end:];assert changed.replace(new,old,1)==src
 # The frozen old reporter's metric/tail/interpretation body stays exact.
 ns=v.c.__dict__.copy();ns.update(__name__='_parallel_risk_cost_report',__file__=str(Path(__file__)),accelerated_costs=accelerated_costs);exec(compile(changed,str(Path(__file__))+'::original_tail_body','exec'),ns);ns['run']('vidstg')
 receipt=dict(status='original_scalar_tail_body_with_actual_parallel_costs_exported',original_reporter_sha256=v.c.sha(v.OLD/'root_tail_cost_v1.py'),code_sha256=v.c.sha(Path(__file__)),only_actual_worker_cost_block_replaced=True,no_new_GT_scoring=True,no_original_exit0_fabrication=True)
 v.put(v.D/'root_tail_cost_adapter_receipt_v2.json',receipt)
if __name__=='__main__':run()
