"""Read-only root analysis of audited shared-corrected ROI main/component results."""
from shared_common import *
import argparse

def tails(rr,a,b):
 dt=[100*(r['values'][a]['t']-r['values'][b]['t'])for r in rr];dv=[100*(r['values'][a]['v']-r['values'][b]['v'])for r in rr]
 return dict(severe_t=sum(v< -5 for v in dt),severe_v=sum(v< -5 for v in dv),severe_union_once=sum(t< -5 or v< -5 for t,v in zip(dt,dv)),rescue_union_once=sum(t>5 or v>5 for t,v in zip(dt,dv)))

def run(ds):
 import numpy as np
 validate();task='components';folder=P/task/ds;ap=D/f'{task}_{ds}_reporting_receipt.json';sp=D/f'{task}_{ds}_score_receipt.json';audit=read(ap);score=read(sp)
 assert audit['status']=='verified_whole_domain'and audit['score_sha256']==sha(sp)and score['seal_sha256']==sha(D/f'{task}_{ds}_sealed.json')
 rr=read(D/task/ds/'metrics.json')['private_rows'];report=read(folder/'report.json');assert len(rr)==COUNTS[ds]
 for arm,mm in report['means'].items():
  for k,v in mm.items():
   vals=[r['values'][arm][k]for r in rr if r['values'][arm][k]is not None];assert(v is None and not vals)or abs(v-float(np.mean(vals)))<1e-12
 old=read(R/f'results/map_components_v1/components/{ds}/report.json');assert all(report['means'][a]==old['means'][a]for a in ['Source','EightReplacement','SharedCorrection','SpatialInference'])
 tt={a+'-'+b:tails(rr,a,b)for a,b in pairs(task)};atomic(folder/'all_paired_tails.json',tt)
 with(folder/'table3.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['configuration','tIoU_percent','vIoU_percent','R03_percent','R05_percent']);w.writerows([[a,*[100*report['means'][a][k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05']]]for a in ['Source','EightReplacement','SharedCorrection','SpatialInference','FullMAP']])
 atomic(folder/'final_DP_control.json',dict(control='Shared-corrected-box ROI features, newly computed posterior and jointMAP interval, identical final bank/validity/unary; independent per-frame choice versus final exact trajectory DP. No first auxiliary DP in either arm.',means={a:report['means'][a]for a in ['FinalIndependent','FullMAP']},paired=report['paired']['FullMAP-FinalIndependent'],tails=tt['FullMAP-FinalIndependent'],timegroups={g:r['paired']['FullMAP-FinalIndependent']for g,r in report['timegroups'].items()}))
 costs=dict(new_Source=0,new_expert=0,new_optimizer=0,cached_query_CPU_seconds_by_PID_start=audit['worker_CPU_seconds'],inherited_paid_acquisition_and_eight_expert_cost_nonzero=True,scope='Conditional cached CPU replay; excludes original acquisition and setup/outer loop. Timings may be nested and must not be added or used as deployment latency. A separate actual matched deployment experiment is required for this method.')
 atomic(folder/'cost_segments.json',costs)
 def fmt(v):return f"{v['mean']:+.6f} [{v['CI95'][0]:.6f}, {v['CI95'][1]:.6f}]"
 text=f'# Shared-corrected-box ROI main and components: {ds}\n\nAll {len(rr)} queries / {report["parents"]} parents sealed before independent official CPU scoring: {score["checks"]} dense checks, maximum discrepancy {score["max_error"]:.17g}. Separate audit verified {audit["prediction_receipt_SHA_checks"]} prediction/receipt SHAs, bound parent/compact/capture hashes, every cold-rederived array and all scalar means. This root export reads saved scalar results only, with zero model, Source, expert, optimizer or GT-scoring calls.\n\nThe user selected this simplification after both full structural ablations. The same eight expert observations inform shared-corrected per-frame boxes; their ROIs directly pool appearance features. Kappa0/beta1/one prototype update, jointMAP interval and final exact trajectory DP remain. The first auxiliary DP is removed. Old method files and results remain immutable. Full datasets were already exposed; this is not independent unseen confirmation.\n\n|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs native Source|\n|---|---:|---:|---:|---:|---|\n'
 for a,m in report['means'].items():
  z=report['severe_vs_Source'][a];text+=f"|{a}|{100*m['t']:.6f}|{100*m['v']:.6f}|{100*m['vIoU_ge_03']:.6f}|{100*m['vIoU_ge_05']:.6f}|{z['t']}/{z['v']}/{z['union']}|\n"
 text+='\n|Signed comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|\n|---|---:|---:|---|---:|\n'
 for key,v in report['paired'].items():
  z=tt[key];text+=f"|{key}|{fmt(v['t'])}|{fmt(v['v'])}|{z['severe_t']}/{z['severe_v']}/{z['severe_union_once']}|{z['rescue_union_once']}|\n"
 text+='\n## Duration groups\n\n|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|\n|---|---:|---:|---:|\n'
 for group,g in report['timegroups'].items():
  for key,v in g['paired'].items():text+=f"|{group} / {key}|{g['queries']}|{fmt(v['t'])}|{fmt(v['v'])}|\n"
 text+='\n## Interpretation\n\nThe first four progressive rows exactly retain their original values. Source uses native per-offset MAP plus physical hull; SourceJointMAP isolates readout. EightReplacement also changes the native readout to q0 jointMAP, so that increment is not solely an expert-box effect. SharedCorrection means globally corrected boxes, not raw expert overwrite. FullMAP-HistoricalAuxDP compares the selected direct-ROI method with the preserved two-DP method. Zero-crossing CIs do not establish equivalence or an average gain. The selection follows the user condition, rather than choosing the highest full-data score.\n\nFullMAP-FinalIndependent fixes the new posterior/interval and all final candidates/unaries. It measures final path selection, not the removed first auxiliary DP. All improvements, harms, ties, short/long results and severe tails are retained. Fixed-support observed/unobserved overlap and boundary errors are in report.json; mean improvements do not guarantee every query improves.\n\n'
 text+='## Costs and remaining scope\n\n'+json.dumps(costs,indent=2)+'\n\nCIs are pointwise parent-cluster bootstrap intervals, not familywise. Short means native-GT duration <=5s. Severe means signed tIoU or vIoU loss >5pp, union once; rescue means >5pp gain. All signed outcomes remain in public TSVs. Media, captions, coordinates, raw features and weights stay private. Structural/factorial, budget, CG transfer, old corrupt panels, actual timing and qualitative synchronization remain separate tasks; full219690corruption and extra generic TTA are not complete.\n'
 (folder/'ANALYSIS.md').write_text(text)
 rec=dict(status='root_complete_analysis_export',dataset=ds,code_sha256=sha(Path(__file__)),audit_sha256=sha(ap),score_sha256=sha(sp),report_sha256=sha(folder/'report.json'),analysis_sha256=sha(folder/'ANALYSIS.md'),all_signed_tails_and_duration_CIs=True,new_model=0,new_GT_scoring=0)
 atomic(D/f'{task}_{ds}_root_export_receipt.json',rec);atomic(folder/'root_export_receipt.json',rec)
 print(json.dumps(dict(dataset=ds,means={a:{k:100*v for k,v in report['means'][a].items()if k in ['t','v','vIoU_ge_03','vIoU_ge_05']}for a in ['Source','FullMAP','FinalIndependent']},paired={k:report['paired'][k]for k in ['FullMAP-HistoricalAuxDP','FullMAP-FinalIndependent','FullMAP-SpatialInference']},tails=tt,native_severe=report['severe_vs_Source']['FullMAP']),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);run(p.parse_args().dataset)
