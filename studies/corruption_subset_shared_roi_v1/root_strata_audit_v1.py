"""Read-only report audit: independently recompute every published hit/identity stratum mean."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'outputs/corruption_subset_shared_roi_v1';P=R/'results/corruption_subset_shared_roi_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(ds,name):
 started=time.perf_counter();plan=read(D/'root_strata_audit_plan_v1.json');assert plan['code_sha256']==sha(__file__)and plan['original_scientific_plan_sha256']==sha(D/'plan.json');root=D/'units'/name/ds;out=P/name/ds;a=read(root/'reporting_receipt.json');assert a['status']=='verified_whole128_all_arrays_inputs_teacher_SHA_means'and a['score_sha256']==sha(root/'score_receipt.json');assert a['seal_sha256']==sha(root/'sealed.json');rr=read(root/'metrics.json')['private_rows'];assert len(rr)==128 and len({r['parent']for r in rr})==128;groups=[('whole',rr,out)]
 for field in ['sampled_hit','pixel_identity']:
  for value in [False,True]:
   sub=[r for r in rr if r[field]==value]
   if sub:groups.append((field+'_'+str(value),sub,out/(field+'_'+str(value))))
 checked=[];total=0
 for label,sub,folder in groups:
  rp=folder/'report.json';report=read(rp);assert report['queries']==len(sub)and report['parents']==len({r['parent']for r in sub});n=0
  for arm,fields in report['means'].items():
   for metric,mean in fields.items():
    vals=[r['values'][arm][metric]for r in sub if r['values'][arm][metric]is not None]
    assert (mean is None and not vals)or (vals and math.isfinite(mean)and abs(sum(vals)/len(vals)-mean)<1e-12),(label,arm,metric)
    n+=1
  checked.append(dict(stratum=label,queries=len(sub),parents=report['parents'],mean_checks=n,report_sha256=sha(rp)));total+=n
 worker=dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]));rec=dict(status='verified_all_whole_and_nonempty_hit_identity_stratum_means',dataset=ds,condition=name,groups=checked,mean_checks=total,reporting_receipt_sha256=sha(root/'reporting_receipt.json'),metrics_sha256=sha(root/'metrics.json'),seal_sha256=sha(root/'sealed.json'),code_sha256=sha(__file__),worker=worker,seconds=time.perf_counter()-started,new_GT=0,new_model=0,new_Source=0,new_expert=0,numerical_predictions_or_existing_reports_changed=False)
 for p in [root/'root_strata_audit_receipt_v1.json',out/'root_strata_audit_receipt_v1.json']:p.write_text(json.dumps(rec,indent=2)+'\n')
 print(json.dumps(rec),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['hc2','vidstg']);p.add_argument('--condition',required=True);a=p.parse_args();run(a.dataset,a.condition)
