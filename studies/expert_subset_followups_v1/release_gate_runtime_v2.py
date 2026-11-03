"""Same queue/science gates, with explicit preserved zero-model parent recovery."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];D=R/'outputs/expert_subset_followups_v1';A=R/'outputs/expert_subset_expansion_v1';B=A/'runtime_v2'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def inspect():
 cfg=read(D/'gate_execution_v2.json')
 for group in ['code_sha256','input_sha256']:
  for rel,h in cfg[group].items():assert sha(R/rel)==h,rel
 q=read(D/'queue_v1.json');assert q['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in q.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
 for group in ['code_sha256','input_sha256']:
  for rel,h in q[group].items():assert sha(R/rel)==h,rel
 assert q['parent_scientific_fingerprint']==read(A/'plan.json')['fingerprint']==read(A/'execution_v2_plan.json')['scientific_fingerprint']
 # Preserve actual failed launch rather than falsely declaring original exit0.
 old=read(A/'controller_actual_exit.json');assert old['tool_session']==90139 and old['actual_tool_wait']and old['exit_code']==1
 failed=read(A/'exits/mttr_GPU.json');assert failed['actual_subprocess_wait']and failed['exit_code']==1
 preservation=read(A/'preserved/mttr_protected_ToDesk_gate_v1/preservation.json');assert preservation['MTTR_new_model_loads']==preservation['MTTR_new_expert']==preservation['MTTR_new_Source']==0
 waiting=[]
 for expert in ['referdino','referformer_r50','referformer_r101','mttr']:
  for ds in ['hc2','vidstg']:
   p=A/expert/ds/'reporting_receipt.json';rp=p.with_name('root_export_receipt.json')
   if not p.exists()or not rp.exists():waiting.append(expert+'/'+ds+':complete_audit_and_export');continue
   x=read(p);assert x['queries']==x['parents']==128 and x['metrics_sha256']==sha(p.with_name('metrics.json'))and x['report_sha256']==sha(R/'results/expert_subset_expansion_v1'/expert/ds/'report.json')
 if not(A/'root_multi_export_receipt.json').exists():waiting.append('all_four_experts_ROOT_analysis')
 prior=[p for p in(A/'exits').glob('*.json')if read(p)['expert']!='mttr'];new=list((B/'exits').glob('*.json'));assert len(prior)==27 and all(read(p)['actual_subprocess_wait']and read(p)['exit_code']==0 for p in prior)
 if len(new)!=9:waiting.append('remaining9_actual_MTTR_stages_success')
 else:assert all(read(p)['actual_subprocess_wait']and read(p)['exit_code']==0 for p in new)
 ctrl=read(B/'controller.json');stat=Path(f'/proc/{ctrl["pid"]}/stat')
 if stat.exists()and int(stat.read_text().split()[21])==ctrl['startticks']:waiting.append('current_runtime_v2_controller_exit')
 actual=B/'controller_actual_exit.json'
 if not actual.exists():waiting.append('current_tool6505_actual_wait_receipt')
 else:x=read(actual);assert x['tool_session']==6505 and x['actual_tool_wait']and x['exit_code']==0
 pub=A/'publication_receipt.json'
 if not pub.exists():waiting.append('current_four_experts_ALLremote_publication')
 else:
  x=read(pub);p=Path(x['completion_path']);assert sha(p)==x['completion_sha256']and read(p)['fresh_independent_bare_all_blob_bytes_verified']and read(p)['commit']==x['commit']
 return dict(status='waiting_predecessor'if waiting else'predecessor_complete_pending_new_adapter_and_scientific_execution_freeze',waiting=waiting,prior_failure_preserved_not_called_success=True,completed_stages_reused=27,remaining_successful_stages_required=9,new_GPU=0,new_Source=0,new_expert=0,new_GT=0,does_not_authorize_unfrozen_GPU_launch=True)
if __name__=='__main__':print(json.dumps(inspect(),indent=2))
