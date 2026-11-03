"""Root-only complete-six-configuration export of audited scalar evidence."""
from common_v1 import *
from full_score import cluster
import csv
PREVIOUS=['referdino','referformer_r50','referformer_r101','mttr'];ALL=PREVIOUS+EXPERTS
def main():
 validate();guard();torch.set_num_threads(1);joined={};costs=[];table='# Six expert configurations on the same exposed128parent panels\n\n'
 table+='Current is the unchanged soft E/expert observed-candidate implementation; this is not a hard-anchor fix. ReferFormerR50/R101 are one family. OnlineRefer carries native state over only eight sparse observations; SgMg has the declared joint-eight refined-mask-to-box adapter. No equal knowledge/compute or expert independence claim. All old expert scores are reused, not rescored.\n\n|Dataset|Expert|Pure@Source t / v %|Pure@Own v %|Current t / v %|R.3 / R.5 %|Unobserved IoU %|\n|---|---|---|---:|---|---|---:|\n'
 for ds in ['hc2','vidstg']:
  rr={};proof={};effects={}
  for expert in ALL:
   d=(R/'outputs/expert_subset_expansion_v1'if expert in PREVIOUS else D)/expert/ds;p=(R/'results/expert_subset_expansion_v1'if expert in PREVIOUS else P)/expert/ds;ack=read(d/'reporting_receipt.json');assert ack['metrics_sha256']==sha(d/'metrics.json')and ack['report_sha256']==sha(p/'report.json');assert(d/'root_export_receipt.json').exists();rows_=read(d/'metrics.json')['private_rows'];assert len(rows_)==len({r['parent']for r in rows_})==128;rr[expert]={r['id']:r for r in rows_};proof[expert]=dict(audit_sha256=sha(d/'reporting_receipt.json'),metrics_sha256=ack['metrics_sha256'],report_sha256=ack['report_sha256']);m=read(p/'report.json')['means'];c=m['Current'];q=m['PureSource'];z=m['PureOwn'];table+=f"|{ds}|{expert}|{100*q['t']:.6f} / {100*q['v']:.6f}|{100*z['v']:.6f}|{100*c['t']:.6f} / {100*c['v']:.6f}|{100*c['vIoU_ge_03']:.6f} / {100*c['vIoU_ge_05']:.6f}|{100*c['unobserved_actual_sIoU']:.6f}|\n"
   if expert in EXPERTS:
    seal=read(d/'sealed.json');load=read(D/expert/'model_load.json')
    for key,e in seal['files'].items():costs.append(dict(expert=expert,dataset=ds,query_id=key,worker_pid=e['worker']['pid'],worker_startticks=e['worker']['startticks'],model_steps=e['actual_model_steps'],logical_expert_queries=e['new_expert_queries'],empty_mask_conversions=e['empty_mask_to_full_frame_boxes'],allocated_peak_bytes=e['peak_allocated_GPU_bytes'],reserved_peak_bytes=e['peak_reserved_GPU_bytes'],model_load_record_sha256=sha(D/expert/'model_load.json'),**e['cost']))
  for expert in ALL[1:]:
   direct=[];gain=[];spatial=[]
   assert set(rr[expert])==set(rr['referdino'])
   for key,r in rr[expert].items():
    b=rr['referdino'][key];assert r['parent']==b['parent']and r['duration_group']==b['duration_group'];direct.append(dict(parent=r['parent'],values=dict(Alternative=r['values']['Current'],Original=b['values']['Current'])))
    for out,arm in [(gain,'PureSource'),(spatial,'PureOwn')]:
     a={};z={}
     for f in r['values']['Current']:
      v=[r['values'][k][f]for k in ['Current',arm]]+[b['values'][k][f]for k in ['Current',arm]];a[f]=None if any(x is None for x in v)else(v[0]-v[1])-(v[2]-v[3]);z[f]=None if a[f]is None else 0.
     out.append(dict(parent=r['parent'],values=dict(GainDifference=a,Zero=z)))
   fields=[(f,100)for f in ['t','v','unobserved_actual_sIoU']];effects[expert]=dict(Current_vs_ReferDINO=cluster(direct,'Alternative','Original',fields),total_gain_difference_vs_ReferDINO=cluster(gain,'GainDifference','Zero',fields),own_interval_spatial_gain_difference_vs_ReferDINO=cluster(spatial,'GainDifference','Zero',fields))
  joined[ds]=dict(experts=proof,effects=effects)
 assert len(costs)==512 and sum(x['model_steps']for x in costs if x['expert']=='onlinerefer_r50')==2048 and sum(x['model_steps']for x in costs if x['expert']=='sgmg_videoswin_t')==256
 with(P/'all512_new_query_costs.tsv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=list(costs[0]),delimiter='\t');w.writeheader();w.writerows(costs)
 table+='\nCurrent-minus-PureOwn isolates spatial strategy within each expert own interval; across experts that interval can change. Current-minus-PureSource includes temporal and spatial changes. Gain differences across experts are descriptive paired effects, not a causal isolation of expert quality. All signed gains/harms, severe/rescue union once, short<=5s/long groups, source-relative negatives and pointwise10000parent bootstrap intervals remain in each report; no target tuning/unknown-test/familywise/equivalence claim.\n\nActual OnlineRefer2048native causal steps and SgMg256joint-eight calls are512logical expert queries, not512equal-cost forward steps. Source/original ReferDINO/optimizer new calls0; inherited acquisition remains nonzero. Load once per actual PID/start, query/forward/checkpoint/finish costs nest and cannot be added or called uncached deployment. OnlineRefer step checkpoint writes lie within its measured forward window. All private raw masks/features/media/captions/coordinates/weights stay private.\n'
 (P/'ANALYSIS.md').write_text(table);atomic(P/'all_expert_paired_effects.json',joined);atomic(D/'root_multi_export_receipt.json',dict(status='all_six_configurations_ROOT_paired_scalar_cost_export_PENDING_interpretation_publication',code_sha256=sha(Path(__file__)),summary_sha256=sha(P/'all_expert_paired_effects.json'),analysis_sha256=sha(P/'ANALYSIS.md'),cost_sha256=sha(P/'all512_new_query_costs.tsv'),new_Source=0,new_expert=0,new_GT=0));print('all six configuration audited score/cost export complete')
if __name__=='__main__':main()
