"""Independent fixed cached128/domain expert comparison; no full-grid recovery."""
from eg_execution_common_v3 import *
import argparse
ARMS=['Source','CurrentSimple','CurrentOurs','AlternativeSimple','AlternativeOurs']
PAIRS=[('CurrentOurs','CurrentSimple'),('AlternativeOurs','AlternativeSimple'),('AlternativeSimple','CurrentSimple'),('AlternativeOurs','CurrentOurs')]
MAP={'Source':'Source','CurrentSimple':'CurrentSimple','CurrentOurs':'CurrentOurs','AlternativeSimple':'Simple','AlternativeOurs':'Ours'}

def validate_evaluation():
 validate_execution_v3();p=read(D/'small_evaluation_plan_v3.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p

def score(ds):
 import torch,numpy as np
 from full_score import metric
 torch.set_num_threads(1);validate_evaluation();folder=D/'pilot_v1'/ds;seal=read(folder/'sealed.json');audit=read(folder/'independent_audit.json');assert audit['status']=='verified_all128_pilot_outputs_context_checkpoints_and_fresh_math_no_GT'and audit['seal_sha256']==sha(folder/'sealed.json')and seal['queries']==128
 truth=rc.selected.labels(ds);rr=[];checks=0;maxerr=0.;start=time.perf_counter()
 for row in pilot_rows(ds):
  e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[row['id']];values={}
  with np.load(D/e['path'])as z:
   inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for a,stored in MAP.items():values[a],err=metric(z[stored],inp,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[stored+'_interval'].tolist(),z['positions8']);assert err<2e-10;checks+=1;maxerr=max(maxerr,err)
  assert values['CurrentSimple']['t']==values['AlternativeSimple']['t']==values['Source']['t']
  rr.append(dict(id=row['id'],parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
 out=P/'small_v3'/ds;report=rc.selected.suite().parent_report(rr,ARMS,PAIRS,out);assert report['parents']==128;rc.selected.suite().signed_tails(rr,PAIRS,out);dest=D/'small_v3'/ds;atomic(dest/'metrics.json',dict(private_rows=rr,report=report));rec=dict(status='scored_fixed128_expert_generalization_after_complete_pilot_seal_and_noGT_input_audit',queries=128,parents=128,checks=checks,max_error=maxerr,pilot_seal_sha256=sha(folder/'sealed.json'),input_audit_sha256=sha(folder/'independent_audit.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,full12205_complete=False);atomic(dest/'score_receipt.json',rec);atomic(out/'score_receipt.json',rec)

def audit(ds):
 import torch,numpy as np
 import eg_pilot_root_v1 as numerical
 torch.set_num_threads(1);validate_evaluation();rc.gt_guard();folder=D/'pilot_v1'/ds;dest=D/'small_v3'/ds;seal=read(folder/'sealed.json');sc=read(dest/'score_receipt.json');assert sc['pilot_seal_sha256']==sha(folder/'sealed.json');start=time.perf_counter()
 for row in pilot_rows(ds):numerical.checked(ds,row,seal['files'][row['id']])
 rr=read(dest/'metrics.json')['private_rows'];report=read(P/'small_v3'/ds/'report.json');assert len(rr)==128 and report['parents']==128
 for a,mm in report['means'].items():
  for k,v in mm.items():
   vals=[r['values'][a][k]for r in rr if r['values'][a][k]is not None];assert(v is None and not vals)or abs(v-float(np.mean(vals)))<1e-12
 rec=dict(status='verified_fixed128_expert_generalization_all_arrays_Source_teacher_and_means',queries=128,parents=128,all_arrays_fresh_cold_exact=True,all_means_recomputed=True,score_sha256=sha(dest/'score_receipt.json'),metrics_sha256=sha(dest/'metrics.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_GT_scoring=0,full12205_complete=False);atomic(dest/'reporting_receipt.json',rec);atomic(P/'small_v3'/ds/'reporting_receipt.json',rec)

def export(ds):
 import numpy as np,csv
 from full_score import cluster
 validate_evaluation();rc.gt_guard();dest=D/'small_v3'/ds;folder=P/'small_v3'/ds;audit=read(dest/'reporting_receipt.json');assert audit['status'].startswith('verified_fixed128_expert_generalization')and audit['score_sha256']==sha(dest/'score_receipt.json')and audit['metrics_sha256']==sha(dest/'metrics.json');rr=read(dest/'metrics.json')['private_rows'];report=read(folder/'report.json')
 synthetic=[]
 for r in rr:
  vals={}
  for f in r['values']['Source']:
   x=[r['values'][a][f]for a in ['AlternativeOurs','AlternativeSimple','CurrentOurs','CurrentSimple']];vals[f]=None if any(v is None for v in x)else(x[0]-x[1])-(x[2]-x[3])
  synthetic.append(dict(parent=r['parent'],values=dict(GainDifference=vals,Zero={k:None if v is None else 0. for k,v in vals.items()})))
 fields=[(k,100)for k in ['v','t','s','vIoU_ge_03','vIoU_ge_05','observed_actual_sIoU','unobserved_actual_sIoU']]
 interaction=cluster(synthetic,'GainDifference','Zero',fields);atomic(folder/'between_expert_gain_difference.json',dict(definition='(AlternativeOurs-AlternativeSimple)-(CurrentOurs-CurrentSimple)',paired=interaction,queries=128,parents=128,pointwise_parentcluster_10000=True,not_expert_independence_proof=True))
 tails={}
 for a,b in PAIRS:
  dt=np.asarray([100*(r['values'][a]['t']-r['values'][b]['t'])for r in rr]);dv=np.asarray([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rr]);tails[a+'-'+b]=dict(severe_t=int((dt< -5).sum()),severe_v=int((dv< -5).sum()),severe_union_once=int(((dt< -5)|(dv< -5)).sum()),rescue_union_once=int(((dt>5)|(dv>5)).sum()),v_improve=int((dv>0).sum()),v_harm=int((dv<0).sum()),v_tie=int((dv==0).sum()))
 atomic(folder/'all_paired_tails.json',tails)
 text=f'# Fixed cached128-parent expert comparison: {ds}\n\nSame original Source and eight indices, current ReferDINO versus official ReferFormer R50. Each expert payload is acquired once/query and shared by its simple interpolation and structured inference. Both simple rows use original native Source time; each Ours independently rebuilds that expert masks/geometry/E ROI features/prototypes/posterior/own interval/finalDP. No new Source acquisitions or full12205 completion. Same observation count is not equal compute.\n\n|Method|tIoU %|vIoU %|sIoU %|R.3 %|R.5 %|\n|---|---:|---:|---:|---:|---:|\n'
 for a,m in report['means'].items():text+='|'+a+'|'+'|'.join(f'{100*m[f]:.6f}'for f in ['t','v','s','vIoU_ge_03','vIoU_ge_05'])+'|\n'
 text+='\nBetween-expert gain difference:\n\n'+json.dumps(interaction,indent=2)+'\n\nAll within-expert/between-expert paired CIs, signed positives/negatives, severe union/rescues and short/long groups accompany the report. No target-score config selection or unsupported expert-independence claim. Actual saved expert/conditional numeric costs are separate from inherited Source cost, not summed into uncached deployment.\n';(folder/'ANALYSIS.md').write_text(text)
 atomic(dest/'root_export_receipt.json',dict(status='root_fixed128_expert_generalization_analysis_exported',dataset=ds,audit_sha256=sha(dest/'reporting_receipt.json'),report_sha256=sha(folder/'report.json'),gain_difference_sha256=sha(folder/'between_expert_gain_difference.json'),tails_sha256=sha(folder/'all_paired_tails.json'),code_sha256=sha(Path(__file__)),full12205_complete=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['score','audit','export']);a.add_argument('--dataset',choices=rc.COUNTS,required=True);q=a.parse_args();globals()[q.command](q.dataset)
