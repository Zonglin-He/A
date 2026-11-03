"""Root read-only scalar verification; never rerun a spatial function or score GT."""
from pathlib import Path
import csv,json,hashlib,math,collections,os,time
import numpy as np
R=Path(__file__).resolve().parents[2];D=R/'outputs/spatial_comparison_focus_v1';P=R/'results/spatial_comparison_focus_v1'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=read(D/'measurements.json');s=read(D/'sealed.json');worker=read(D/'measure_worker.json');a=read(D/'audit_receipt.json');summary=read(P/'timing_summary.json');items=m['items'];arms=['PureExpert','Matching','Current']
 assert sha(D/'measurements.json')==s['measurement_sha256']==a['measurement_sha256']==summary['measurement_sha256']
 assert len(items)==2304 and s['calls']==2304 and len(m['warmup'])==6
 assert s['worker']==worker and all(x['worker']==worker for x in items)
 stages={x:read(D/(x+'_exit.json'))for x in ['preflight','measure','audit','export']}
 assert all(x['actual_subprocess_wait']and x['exit_code']==0 for x in stages.values())
 assert stages['measure']['worker']==worker
 assert not Path(f"/proc/{worker['pid']}").exists(), 'Measurement PID unexpectedly exists; verify incarnation first'
 callrows=list(csv.DictReader((P/'all_measured_calls.tsv').open(),delimiter='\t'));assert len(callrows)==2304
 bykey={(x['dataset'],str(x['query']),str(x['repeat']),x['arm']):x for x in items}
 for row in callrows:
  item=bykey[row['dataset'],row['query'],row['repeat'],row['arm']]
  for k,v in row.items():assert str(item[k])==v,(k,row)
 perquery=list(csv.DictReader((P/'per_query_three_repeat_means.tsv').open(),delimiter='\t'));assert len(perquery)==256
 maxerr=0.;checks=0
 for ds in ['hc2','vidstg']:
  matrix=[]
  for q in range(1,129):
   row=[]
   for arm in arms:
    vals=[x for x in items if (x['dataset'],x['query'],x['arm'])==(ds,q,arm)]
    assert len(vals)==3 and {x['repeat']for x in vals}=={1,2,3} and {x['order_position']for x in vals}=={1,2,3}
    assert all(x['elapsed_ns']>0 and math.isfinite(x['elapsed_seconds']) and x['elapsed_seconds']==x['elapsed_ns']/1e9 and x['all_output_arrays_exact']and x['output_digest']==x['reference_array_digest']for x in vals)
    mean=math.fsum(x['elapsed_seconds']for x in vals)/3
    exported=next(x for x in perquery if x['dataset']==ds and int(x['query'])==q)
    assert abs(mean-float(exported[arm]))<1e-14;row.append(mean)
   matrix.append(row)
  matrix=np.array(matrix);w=np.random.default_rng(20261103).multinomial(128,[1/128]*128,size=10000);boot=w@matrix/128
  for j,arm in enumerate(arms):
   target=summary['summary'][ds]['methods'][arm]
   expected={'mean_seconds':float(matrix[:,j].mean()),'CI95_mean_seconds':np.quantile(boot[:,j],[.025,.975]).tolist(),'p50_query_mean_seconds':float(np.quantile(matrix[:,j],.5)),'p90_query_mean_seconds':float(np.quantile(matrix[:,j],.9))}
   for key,val in expected.items():
    delta=float(np.max(np.abs(np.array(val)-np.array(target[key]))));maxerr=max(maxerr,delta);assert delta<1e-13;checks+=1
  for name,target in summary['summary'][ds]['paired'].items():
   aa,bb=name.split('-');ia,ib=arms.index(aa),arms.index(bb)
   expected={'mean_seconds':float((matrix[:,ia]-matrix[:,ib]).mean()),'CI95_seconds':np.quantile(boot[:,ia]-boot[:,ib],[.025,.975]).tolist(),'ratio_of_means':float(matrix[:,ia].mean()/matrix[:,ib].mean()),'CI95_ratio_of_means':np.quantile(boot[:,ia]/boot[:,ib],[.025,.975]).tolist()}
   for key,val in expected.items():
    delta=float(np.max(np.abs(np.array(val)-np.array(target[key]))));maxerr=max(maxerr,delta);assert delta<1e-10;checks+=1
  accuracy=read(P/'reused_accuracy.json')[ds];old=R/'results/spatial_temporal_six_v1/matching_pilot'/ds/'report.json';assert sha(old)==accuracy['reused_report_sha256'];rep=read(old)
  for aa,bb in [('PureExpert','PureMatched'),('Matching','Matching'),('Current','Current')]:assert accuracy['means'][aa]==rep['means'][bb]
 proof=dict(status='ROOT_all_calls_worker_and_coverage_perquery_means_CI_p90_accuracy_provenance_verified',formal_calls=2304,warmups_excluded=6,actual_controller_tool_session=74706,actual_controller_tool_wait_exit_code=0,actual_stage_wait_receipts=stages,measurement_worker=worker,summary_scalar_checks=checks,max_scalar_recompute_error=maxerr,measurements_sha256=sha(D/'measurements.json'),summary_sha256=sha(P/'timing_summary.json'),accuracy_sha256=sha(P/'reused_accuracy.json'),new_Source=0,new_expert=0,new_GPU=0,new_GT=0,no_retiming=True,Current_changed=False,paper_changed=False,code_sha256=sha(Path(__file__)),time=time.time())
 (D/'root_validation_v1.json').write_text(json.dumps(proof,indent=2)+'\n');(P/'root_validation_v1.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
