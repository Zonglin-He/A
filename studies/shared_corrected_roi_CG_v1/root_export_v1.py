"""Read-only root analysis; execute only after whole-domain CG official score and audit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from cg_common import *
import argparse

def summarize(folder, rr, pairs):
 import numpy as np
 report=read(folder/'report.json');assert report['queries']==len(rr)
 for arm,means in report['means'].items():
  for metric,value in means.items():
   vals=[r['values'][arm][metric]for r in rr if r['values'][arm][metric]is not None]
   assert (value is None and not vals)or abs(value-float(np.mean(vals)))<1e-12
 tails={}
 for left,right in pairs:
  dt=np.asarray([100*(r['values'][left]['t']-r['values'][right]['t'])for r in rr]);dv=np.asarray([100*(r['values'][left]['v']-r['values'][right]['v'])for r in rr])
  tails[left+'-'+right]=dict(severe_t=int((dt< -5).sum()),severe_v=int((dv< -5).sum()),severe_union_once=int(((dt< -5)|(dv< -5)).sum()),rescue_union_once=int(((dt>5)|(dv>5)).sum()))
 atomic(folder/'all_paired_tails.json',tails)
 def ci(x):return f"{x['mean']:+.6f} [{x['CI95'][0]:.6f}, {x['CI95'][1]:.6f}]"
 text='|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs corrupted-or-native Source|\n|---|---:|---:|---:|---:|---|\n'
 for arm,m in report['means'].items():
  z=report['severe_vs_Source'][arm];text+=f"|{arm}|{100*m['t']:.6f}|{100*m['v']:.6f}|{100*m['vIoU_ge_03']:.6f}|{100*m['vIoU_ge_05']:.6f}|{z['t']}/{z['v']}/{z['union']}|\n"
 text+='\n|Signed comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|\n|---|---:|---:|---|---:|\n'
 for key,paired in report['paired'].items():
  z=tails[key];text+=f"|{key}|{ci(paired['t'])}|{ci(paired['v'])}|{z['severe_t']}/{z['severe_v']}/{z['severe_union_once']}|{z['rescue_union_once']}|\n"
 text+='\n## Duration groups\n\nShort is GT duration <=5 seconds, long is >5 seconds. All intervals are pointwise parent-cluster bootstrap95%, not familywise.\n\n|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|\n|---|---:|---:|---:|\n'
 for group,g in report['timegroups'].items():
  for key,paired in g['paired'].items():text+=f"|{group} / {key}|{g['queries']}|{ci(paired['t'])}|{ci(paired['v'])}|\n"
 text+='\nEvery signed positive/negative result, fixed-support observed/unobserved metric and boundary error remains in report.json and the signed TSVs. Severe means a loss greater than5pp; union counts a query once. Rescue means an increase greater than5pp in t or v. Source is native per-offset legal MAP plus physical hull. Historically exposed cohorts and user-directed method selection are not unseen confirmation; zero-crossing intervals are not equivalence proofs.\n'
 return report,text

def run(ds):
 validate();folder=P/ds;ap=D/f'{ds}_reporting_receipt.json';sp=D/f'{ds}_score_receipt.json';sealed=D/f'{ds}_sealed.json';audit=read(ap);score=read(sp);seal=read(sealed)
 assert audit['status']=='verified_whole_domain' and audit['all_compact_arrays_cold_rederived_exact'] and audit['all_available_Source_native_logits_q0_original_exact'] and audit['actual_E_ROI_pool_checked_before_raw_disposal'] and audit['all_means_recomputed']
 assert audit['score_sha256']==sha(sp) and score['seal_sha256']==sha(sealed) and seal['queries']==COUNTS[ds] and not seal['GT_read']
 rr=read(D/ds/'metrics.json')['private_rows'];report,body=summarize(folder,rr,PAIRS);assert len(rr)==COUNTS[ds] and report['parents']=={'hc2':227,'vidstg':732}[ds]
 counts=seal['counts'];assert counts['new_expert']==0 and counts['new_Source_forwards']==counts['new_Source']
 costs=dict(actual_counts=counts,actual_worker=seal['worker'],whole_worker_elapsed_seconds=seal['seconds'],model_load_once_for_actual_worker=seal['model_load'],actual_PID_start_segments=audit['worker_segments'],original_Source_teacher_acquisition_nonzero=True,selected_first_auxiliary_DP=False,scope='Source/pipeline/compatibility/selected-finish/load clocks are nested as recorded and must not be summed. Reused raw captures retain acquisition costs. These conditional research walls are not uncached deployment latency. Failed model/paid attempts remain separate evidence; no blind retries. Sufficient states cannot support arbitrary future repooling after only owned new raw disposal.')
 atomic(folder/'cost_segments.json',costs)
 profile='HC gamma0/u0/link0/beta.5/context1.5; unobserved box coordinates remain Source, so no full spatial-propagation efficacy claim.'if ds=='hc2'else'Vid gamma.5/u.5/link.25/beta1/context1; do not substitute the HC disabled-propagation profile.'
 text=f'# CG shared-corrected-box ROI transfer: {ds}\n\nWhole {len(rr)} queries / {report["parents"]} parents sealed before independent official labels. {score["checks"]} official checks, maximum error {score["max_error"]:.17g}; {audit["SHA_checks"]} prediction/compact/receipt SHA checks plus bound old inputs/teacher provenance, cold all-array replay and all means verified. Root export reads audited scalars and costs only, no model/Source/expert/optimizer/GT scoring.\n\n'+body
 text+='\n## Method and controls\n\n'+profile+' Kappa0 remains fixed. Direct actual corrected-box ROI pooling replaces the first auxiliary DP; final interval-conditioned trajectory DP remains. HistoricalTwoDP is the immutable earlier kappa0 method with exactly the same domain profile; SourceJointMAP separates readout from adaptation. Hidden original-array/pool compatibility is a prerequisite, not a new MBR experiment or selected auxiliary-DP cost. No new parameter search or independent-unseen validation.\n\n## Costs\n\n'+json.dumps(costs,indent=2)+'\n'
 (folder/'ANALYSIS.md').write_text(text)
 with (folder/'main_table.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['configuration','tIoU_percent','vIoU_percent','R03_percent','R05_percent']);w.writerows([[name,*[100*m[k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05']]]for name,m in report['means'].items()])
 receipt=dict(status='root_complete_CG_paired_tail_duration_cost_analysis',dataset=ds,code_sha256=sha(Path(__file__)),audit_sha256=sha(ap),score_sha256=sha(sp),seal_sha256=sha(sealed),report_sha256=sha(folder/'report.json'),analysis_sha256=sha(folder/'ANALYSIS.md'),new_model=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT_scoring=0)
 atomic(D/f'{ds}_root_export_receipt.json',receipt);atomic(folder/'root_export_receipt.json',receipt);print(ds,'CG root analysis complete',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);run(p.parse_args().dataset)
