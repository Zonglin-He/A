"""Root stage-specific comparisons using only whole audited scalar records."""
from pathlib import Path
import sys,json,hashlib,time
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'studies/expert_subset_followups_v1'))
from common_v1 import *
from full_score import cluster
def main():
 validate();guard();torch.set_num_threads(1);summary={}
 for ds in ['hc2','vidstg']:
  f=D/'onlinerefer_r50'/ds;a=read(f/'reporting_receipt.json');assert a['metrics_sha256']==sha(f/'metrics.json')and(f/'root_interpretation_v1.json').exists();ours=read(f/'metrics.json')['private_rows'];comparisons={}
  for expert in ['referdino','referformer_r50','referformer_r101','mttr']:
   d=R/'outputs/expert_subset_expansion_v1'/expert/ds;ack=read(d/'reporting_receipt.json');assert ack['metrics_sha256']==sha(d/'metrics.json');prior={r['id']:r for r in read(d/'metrics.json')['private_rows']};direct=[];total=[];spatial=[]
   for row in ours:
    b=prior[row['id']];assert b['parent']==row['parent']and b['duration_group']==row['duration_group'];direct.append(dict(parent=row['parent'],values=dict(OnlineRefer=row['values']['Current'],Previous=b['values']['Current'])))
    for records,arm in [(total,'PureSource'),(spatial,'PureOwn')]:
     v={};zero={}
     for k in row['values']['Current']:
      x=[row['values'][z][k]for z in ['Current',arm]]+[b['values'][z][k]for z in ['Current',arm]];v[k]=None if any(z is None for z in x)else(x[0]-x[1])-(x[2]-x[3]);zero[k]=None if v[k]is None else 0.
     records.append(dict(parent=row['parent'],values=dict(GainDifference=v,Zero=zero)))
   fields=[(k,100)for k in ['t','v','observed_actual_sIoU','unobserved_actual_sIoU']];comparisons[expert]=dict(Current_delta=cluster(direct,'OnlineRefer','Previous',fields),total_gain_difference=cluster(total,'GainDifference','Zero',fields),conditional_own_interval_spatial_gain_difference=cluster(spatial,'GainDifference','Zero',fields),prior_audit_sha256=sha(d/'reporting_receipt.json'))
  summary[ds]=dict(OnlineRefer_audit_sha256=sha(f/'reporting_receipt.json'),paired_comparisons=comparisons)
 atomic(P/'OnlineRefer_vs_four_completed_expert_configurations.json',summary);atomic(D/'OnlineRefer_cross_expert_root_receipt.json',dict(status='whole_two_domains_OnlineRefer_cross_previous_four_scalar_effects_exported',code_sha256=sha(Path(__file__)),summary_sha256=sha(P/'OnlineRefer_vs_four_completed_expert_configurations.json'),new_Source=0,new_expert=0,new_GT=0,not_causal_or_equal_compute=True,conditional_spatial_gains_use_each_experts_own_interval=True,time=time.time()));print(json.dumps({ds:{e:v['Current_delta']['v']for e,v in r['paired_comparisons'].items()}for ds,r in summary.items()}))
if __name__=='__main__':main()
