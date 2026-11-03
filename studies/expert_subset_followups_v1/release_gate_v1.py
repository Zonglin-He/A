"""Read-only release gate for two authorized followups; never launches a model."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];D=R/'outputs/expert_subset_followups_v1';PARENT=R/'outputs/expert_subset_expansion_v1'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def inspect():
 q=read(D/'queue_v1.json');assert q['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in q.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
 for field in ['code_sha256','input_sha256']:
  for f,h in q[field].items():assert sha(R/f)==h,f
 waiting=[]
 for expert in ['referdino','referformer_r50','referformer_r101','mttr']:
  for ds in ['hc2','vidstg']:
   p=PARENT/expert/ds/'reporting_receipt.json';root=PARENT/expert/ds/'root_export_receipt.json'
   if not p.exists()or not root.exists():waiting.append(f'{expert}/{ds}:complete_audit_and_export');continue
   a=read(p);assert a['queries']==a['parents']==128 and a['metrics_sha256']==sha(p.with_name('metrics.json'))
   assert a['report_sha256']==sha(R/'results/expert_subset_expansion_v1'/expert/ds/'report.json')
 if not(PARENT/'root_multi_export_receipt.json').exists():waiting.append('current_four_experts_ROOT_interpretation_and_paired_export')
 exits=list((PARENT/'exits').glob('*.json'))
 if len(exits)!=36:waiting.append('all36_current_worker_stages_actual_exit_zero')
 else:assert all(read(p)['actual_subprocess_wait']and read(p)['exit_code']==0 for p in exits)
 controller=q['parent_controller'];stat=Path(f'/proc/{controller["pid"]}/stat')
 if stat.exists()and int(stat.read_text().split()[21])==controller['startticks']:waiting.append('current_controller_actual_exit')
 actual=PARENT/'controller_actual_exit.json'
 if not actual.exists():waiting.append('current_controller_tool90139_actual_wait_receipt')
 else:
  a=read(actual);assert a['tool_session']==90139 and a['actual_tool_wait']and a['exit_code']==0
 pub=PARENT/'publication_receipt.json'
 if not pub.exists():waiting.append('current_scope_stage_publication_ALLremote_proof')
 else:
  x=read(pub);cp=Path(x['completion_path']);assert sha(cp)==x['completion_sha256'];assert read(cp)['fresh_independent_bare_all_blob_bytes_verified']and read(cp)['commit']==x['commit']
 return dict(status='waiting_predecessor'if waiting else'predecessor_complete_pending_new_adapter_and_scientific_execution_freeze',waiting=waiting,new_GPU=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT=0,does_not_authorize_unfrozen_GPU_launch=True)
if __name__=='__main__':print(json.dumps(inspect(),indent=2))
