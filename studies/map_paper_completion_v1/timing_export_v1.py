"""Read-only repeat-averaged paired timing analysis; no model/GT work."""
from common import *
import argparse,subprocess,platform
G=D/'deployment';Q=P/'deployment'
def main():
 import numpy as np
 torch_stage=validate('deployment');seal=read(G/'sealed.json');assert seal['measurement_sha256']==sha(G/'measurements.json')and seal['all_current_MAP_outputs_bitwise'];v=read(G/'measurements.json');items=v['items'];assert len(items)==192 and all(x['prediction_bitwise_verified']and not x['GT_read']for x in items)
 pause=read(D/'deployment_CPU_pause.json');resume=read(D/'deployment_CPU_resume.json');assert pause['groups']==resume['groups'];selection=read(D/'timing_selection.json');stats={};allrows=[];rng=np.random.default_rng(20260920)
 for ds,rows in selection.items():
  assert len(rows)==16 and len({r['metadata']['original_video_id']for r in rows})==16
  means={a:[]for a in ['Source','FullMAP']}
  for q in range(1,17):
   for arm in means:
    calls=[x for x in items if x['dataset']==ds and x['query']==q and x['arm']==arm];assert sorted(x['repeat']for x in calls)==[1,2,3]
    means[arm].append(float(np.mean([x['query_wall_seconds']for x in calls])))
    for x in calls:
     allrows.append(dict(dataset=ds,query=f'Q{q:02}',repeat=x['repeat'],arm=arm,seconds=x['query_wall_seconds'],RGB_seconds=x['RGB_seconds'],parse_seconds=x['parse_seconds'],**x['cost']))
  a=np.asarray(means['Source']);b=np.asarray(means['FullMAP']);indices=rng.integers(0,16,size=(10000,16));delta=(b-a)[indices].mean(1);ratio=b[indices].mean(1)/a[indices].mean(1)
  stats[ds]=dict(queries=16,parents=16,repeats=3,measured_calls=96,mean_Source_seconds=float(a.mean()),mean_FullMAP_seconds=float(b.mean()),paired_overhead_seconds=float((b-a).mean()),paired_overhead_95CI=np.quantile(delta,[.025,.975]).tolist(),mean_time_ratio=float(b.mean()/a.mean()),ratio_95CI=np.quantile(ratio,[.025,.975]).tolist(),unit='One parent/query averaged across its3repeats;16pairedunits. Repeats not counted as independentqueries.',query_averages=[dict(query=f'Q{i+1:02}',Source_seconds=float(x),FullMAP_seconds=float(y),overhead_seconds=float(y-x))for i,(x,y)in enumerate(zip(a,b))])
 keys=sorted({k for x in allrows for k in x});Q.mkdir(parents=True,exist_ok=True)
 with(Q/'all_measured_calls.tsv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=keys,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(allrows)
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip();cpu=json.loads(subprocess.check_output(['lscpu','-J'],text=True));hardware=dict(GPU=gpu,CPU=[x for x in cpu['lscpu']if x['field']in ['Architecture:','CPU(s):','Model name:','Thread(s) per core:','Core(s) per socket:','Socket(s):']],OS=platform.platform(),python=platform.python_version(),torch_threads=4,native_Source='Original224FP32twooffsets',same_expert_observations=8)
 result=dict(status='verified_repeat_averaged_real_current_MAP_deployment',statistics=stats,hardware=hardware,confidence='Pointwise pairedparent nonparametricbootstrap10000 seed20260920, eachdomain16units, nofamilywiseclaim',scope=v['report']['scope'],cache_rules=v['report']['interquery_RGB_visual_capture_teacher_cache'],own_CPU_groups_paused_for_matched_timing=True,all192outputs_bitwise=True,measurement_sha256=sha(G/'measurements.json'),seal_sha256=sha(G/'sealed.json'),export_code_sha256=sha(Path(__file__)),new_model=0,new_GT_scoring=0,actual_model_calls_provenance=v['report']['measured_calls'])
 atomic(Q/'paired_timing.json',result);atomic(G/'export_receipt.json',dict(status='complete_verified',measurement_sha256=sha(G/'measurements.json'),paired_report_sha256=sha(Q/'paired_timing.json'),calls_TSV_sha256=sha(Q/'all_measured_calls.tsv'),export_code_sha256=sha(Path(__file__)),new_model=0,new_GT_scoring=0));print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
