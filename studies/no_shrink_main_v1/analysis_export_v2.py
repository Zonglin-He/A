"""Root read-only exports of scored and fully audited current-kappa0 domains."""
from ns_common import *
import argparse

def tail(rows,a,b):
 dt=[100*(r['values'][a]['t']-r['values'][b]['t'])for r in rows];dv=[100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]
 return dict(severe_t=sum(x< -5 for x in dt),severe_v=sum(x< -5 for x in dv),severe_union_once=sum(t< -5 or v< -5 for t,v in zip(dt,dv)),rescue_union_once=sum(t>5 or v>5 for t,v in zip(dt,dv)))
def run(task,ds):
 import numpy as np
 from full_score import cluster
 validate();auditpath=D/f'{task}_{ds}_reporting_receipt.json';audit=read(auditpath);scorepath=D/f'{task}_{ds}_score_receipt.json';score=read(scorepath);assert audit['status']=='verified_whole_domain'and audit['score_sha256']==sha(scorepath)and score['seal_sha256']==sha(D/f'{task}_{ds}_sealed.json');folder=P/task/ds;rr=read(D/task/ds/'metrics.json')['private_rows'];report=read(folder/'report.json');tails={k:tail(rr,*k.split('-'))for k in report['paired']};atomic(folder/'all_paired_tails.json',tails)
 costs=dict(new_Source=0,new_expert=0,new_optimizer=0,current_worker_CPU_seconds=audit['worker_CPU_seconds'],scope='Conditional cached-query CPU wall excludes setup/outer-loop, may include shared-host effects. Original Source/expert acquisitions remain nonzero; nested costs must not be added or used as current deployment latency.')
 if task=='CG':
  origin=R/f'outputs/cgstvg_full_v1/{ds}_reporting_receipt.json';old=read(origin);costs['inherited_acquisition_audit']=dict(path=str(origin.relative_to(R)),sha256=sha(origin),original_worker_segments=old.get('worker_segments',old.get('worker_cost_segments',{})));seal=read(D/f'CG_{ds}_sealed.json');first=next(iter(seal['files'].values()));atomic(folder/'fixed_domain_profile.json',dict(profile=first['provenance']['profile'],only_kappa_changed=True,all_HC_unobserved_boxes_retained_Source=(ds=='hc2')))
 if task=='components':
  old=read(R/f'results/map_components_v1/components/{ds}/report.json');firstfour=['Source','EightReplacement','SharedCorrection','SpatialInference'];assert all(report['means'][a]==old['means'][a]for a in firstfour)
  with(folder/'table3.tsv').open('w')as f:
   w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['configuration','tIoU_percent','vIoU_percent','R03_percent','R05_percent']);w.writerows([[a,*[100*report['means'][a][k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05']]]for a in firstfour+['FullMAP']])
  atomic(folder/'final_DP_control.json',dict(control='Same current-kappa0 auxiliary features/q1 jointMAP interval/candidate bank/validity/unary; final per-frame independent argmax versus exact path DP. Not removal of the first auxiliary DP or the entire spatial branch.',means={a:report['means'][a]for a in ['FinalIndependent','FullMAP']},paired=report['paired']['FullMAP-FinalIndependent'],tails=tails['FullMAP-FinalIndependent'],timegroups={g:x['paired']['FullMAP-FinalIndependent']for g,x in report['timegroups'].items()}))
 legacy=None
 if task=='CG':
  oldpath=MAP/'CG'/ds/'metrics.json';oldrows={r['id']:r for r in read(oldpath)['private_rows']};joined=[]
  for r in rr:
   o=oldrows[r['id']];assert o['parent']==r['parent']and o['values']['Source']==r['values']['Source'];joined.append({**r,'values':{**r['values'],'LegacyKappa64':o['values']['FullMAP']}})
  legacy=dict(paired=cluster(joined,'FullMAP','LegacyKappa64'),tails=tail(joined,'FullMAP','LegacyKappa64'),timegroups={g:cluster([r for r in joined if r['duration_group']==g],'FullMAP','LegacyKappa64')for g in ['short_le5s','long_gt5s']},legacy_audit_sha256=sha(MAP/f'CG_{ds}_reporting_receipt.json'),legacy_metrics_sha256=sha(oldpath),source='Historical kappa64 jointMAP+actualDP, only shrinkage is changed in this comparison. No MBR comparison introduced.');atomic(folder/'legacy_kappa_comparison.json',legacy);suite().signed_tails(joined,[('FullMAP','LegacyKappa64')],folder/'legacy_comparison'if False else folder)
  # Keep the native/current signed table authoritative; add the legacy comparison separately.
  (folder/'all_signed_severe_timegroups.tsv').rename(folder/'legacy_kappa_signed_severe_timegroups.tsv');suite().signed_tails(rr,pairs(task),folder)
 atomic(folder/'cost_segments.json',costs)
 def fmt(v):return f"{v['mean']:+.6f} [{v['CI95'][0]:.6f}, {v['CI95'][1]:.6f}]"
 s=f'# Current no-shrinkage {task}: {ds}\n\nComplete whole-domain seal → independent official scoring → exact array rederivation and all-mean audit: **{len(rr)} queries / {report["parents"]} parents**, {score["checks"]} official dense checks (max discrepancy {score["max_error"]:.17g}), {audit["prediction_receipt_SHA_checks"]} prediction/receipt SHA checks plus bound parent hashes verified during derivation. Zero new Source, expert, optimizer or GT scoring in this root export.\n\nKappa0 is the directly user-selected current method. Selection followed exposed TA full results; no claim of independent unseen confirmation. Old kappa64 scientific files and outputs remain immutable. Source uses native per-offset MAP and physical hull; SourceJointMAP isolates readout. Every changed posterior uses its own MAP interval and actually re-solves spatial DP.\n\n|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs native Source|\n|---|---:|---:|---:|---:|---|\n'
 for a,m in report['means'].items():
  v=report['severe_vs_Source'][a];s+=f"|{a}|{100*m['t']:.6f}|{100*m['v']:.6f}|{100*m['vIoU_ge_03']:.6f}|{100*m['vIoU_ge_05']:.6f}|{v['t']}/{v['v']}/{v['union']}|\n"
 s+='\n|Signed comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|\n|---|---:|---:|---|---:|\n'
 for key,v in report['paired'].items():
  z=tails[key];s+=f"|{key}|{fmt(v['t'])}|{fmt(v['v'])}|{z['severe_t']}/{z['severe_v']}/{z['severe_union_once']}|{z['rescue_union_once']}|\n"
 s+='\n## Duration groups\n\n|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|\n|---|---:|---:|---:|\n'
 for group,g in report['timegroups'].items():
  for key,v in g['paired'].items():s+=f"|{group} / {key}|{g['queries']}|{fmt(v['t'])}|{fmt(v['v'])}|\n"
 s+='\n## Fixed controls and interpretation\n\n'
 if task=='components':s+='The five progressive rows use unchanged native Source, eight-only expert replacement, shared initialization **E**, spatial path inference under q0 jointMAP, and current kappa0 FullMAP. The first four rows exactly equal the previous component experiment. SharedCorrection is E, not all observed frames forcibly overwritten by experts. The first native-Source→eight-replacement change also includes q0 jointMAP readout; SourceJointMAP is therefore separate. Shared−Eight vIoU CI crosses zero and observed-frame sIoU falls, while unobserved sIoU rises. Spatial−Shared vIoU is positive. Full−Spatial vIoU is positive on average, but short-event delta is negative with a CI crossing zero; unobserved spatial-IoU delta also crosses zero. All observed/unobserved outcomes remain in report.json.\n\nFinalDP versus independent selection holds the final interval and all upstream evidence/candidates/unaries fixed. It supports the final trajectory selection only, not an independent contribution of the first auxiliary DP. All improvements, harms, ties and severe losses remain in final_DP_control.json; no claim of universally beneficial DP.\n\n'
 if task=='CG':
  s+='CG uses its existing per-domain profile with **only kappa set to0**; the full profile is in fixed_domain_profile.json. HC gamma0/u0/link0 deliberately disables unobserved propagation and linkage, and every unobserved box remains the original Source box. Its unobserved fixed-support sIoU change is exactly zero, so this is not evidence that complete propagation works on HC. All227 HC parents were used in development; remaining queries do not create independent parent validation. Source weight/input/local precision qualifications are inherited from the original study.\n\n'
  s+=f"Relative to historical kappa64 jointMAP+DP, delta t {fmt(legacy['paired']['t'])} pp and delta v {fmt(legacy['paired']['v'])} pp. The mean change and all severe/rescue/short-long outcomes are in legacy_kappa_comparison.json and the anonymous signed TSV; neither historical negative outcomes nor tail damage are removed.\n\n"
 s+='## Costs and evidence\n\n'+json.dumps(costs,indent=2)+'\n\nConfidence intervals are pointwise parent-cluster bootstrap intervals, not familywise. Short means native-GT duration <=5s. Severe means a signed tIoU or vIoU loss >5pp, union counted once; rescues use a >5pp gain. All signed positives and negatives, all means, CIs and tails remain in the public report/TSVs. Private captions/media/coordinates/features/weights are excluded.\n';(folder/'ANALYSIS.md').write_text(s)
 rec=dict(status='root_complete_analysis_export',task=task,dataset=ds,audit_sha256=sha(auditpath),score_sha256=sha(scorepath),report_sha256=sha(folder/'report.json'),analysis_sha256=sha(folder/'ANALYSIS.md'),code_sha256=sha(Path(__file__)),new_Source=0,new_expert=0,new_GT_scoring=0);atomic(D/f'{task}_{ds}_root_export_receipt.json',rec);atomic(folder/'root_export_receipt.json',rec);print(task,ds,'root exported',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['components','CG'],required=True);p.add_argument('--dataset',choices=COUNTS,required=True);z=p.parse_args();run(z.task,z.dataset)
