"""Read-only current-kappa0 own-K budget tables, tails, costs and figures."""
from ns_common import *
import argparse

def run(ds):
 import numpy as np
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 validate();auditpath=D/f'budget_{ds}_reporting_receipt.json';a=read(auditpath);assert a['all_arrays_cold_rederived_exact']and a['all_means_recomputed']and a['score_sha256']==sha(D/f'budget_{ds}_score_receipt.json');score=read(D/f'budget_{ds}_score_receipt.json');assert score['seal_sha256']==sha(D/f'budget_{ds}_sealed.json');seal=read(D/f'budget_{ds}_sealed.json');rr=read(D/'budget'/ds/'metrics.json')['private_rows'];folder=P/'budget'/ds;r=read(folder/'report.json');assert len(rr)==128 and r['parents']==128;configs=[f'{s}{k}'for k in [2,4,8,16]for s in ['Hybrid','Uniform']];costs={};tails={}
 for name in configs:
  original=[e['provenance']['configs'][name]['original_cost']for e in seal['files'].values()];costs[name]=dict(mean_original_expert_forward_seconds=float(np.mean([x['expert_GPU_forward_seconds']for x in original])),actual_K_range=[min(x['actual_K']for x in original),max(x['actual_K']for x in original)],original_acquisition_not_zero=True,new_expert=0)
 for key in r['paired']:
  left,right=key.split('-');t=[100*(x['values'][left]['t']-x['values'][right]['t'])for x in rr];v=[100*(x['values'][left]['v']-x['values'][right]['v'])for x in rr];tails[key]=dict(severe_t=sum(z< -5 for z in t),severe_v=sum(z< -5 for z in v),severe_union_once=sum(x< -5 or y< -5 for x,y in zip(t,v)),rescue_union_once=sum(x>5 or y>5 for x,y in zip(t,v)))
 atomic(folder/'all_paired_tails.json',tails);atomic(folder/'cost_segments.json',dict(original_expert=costs,current_query_CPU_seconds=a['worker_CPU_seconds'],scope='Original per-K teacher forward costs inherited, notzero deployment. Cached own-K geometry/real repool/currentposterior/DP cost is conditional research wall, not totaldeployment. Startup/transport/request not inferred from forward; nestedtime notadded.'))
 curve=folder/'budget_curve.tsv'
 with curve.open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['schedule','nominal_K','queries','tIoU_percent','vIoU_percent','R03_percent','R05_percent','original_expert_forward_seconds']);w.writerows([[name.rstrip('0123456789'),int(name[len(name.rstrip('0123456789')):]),128,*[100*r['means'][name][k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05']],costs[name]['mean_original_expert_forward_seconds']]for name in configs])
 fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
 for schedule in ['Hybrid','Uniform']:
  names=[f'{schedule}{k}'for k in [2,4,8,16]];values=[100*r['means'][n]['v']for n in names];axes[0].plot([2,4,8,16],values,marker='o',label=schedule);axes[1].plot([costs[n]['mean_original_expert_forward_seconds']for n in names],values,marker='o',label=schedule)
 for ax in axes:ax.axhline(100*r['means']['Source']['v'],color='gray',ls='--',label='Native Source');ax.grid(alpha=.25);ax.legend(fontsize=8);ax.set_ylabel('vIoU (%)')
 axes[0].set(xlabel='Nominal expert observations K',xticks=[2,4,8,16]);axes[1].set_xlabel('Original expert forward (s/query; inherited)');fig.suptitle(f'{ds}: no shrinkage, fixed128queries /128parents');private=D/'figures';private.mkdir(exist_ok=True);svg=folder/'budget_curve.svg';fig.savefig(svg,metadata={'Date':None});fig.savefig(private/f'{ds}_budget_curve_kappa0.png',dpi=180);fig.savefig(private/f'{ds}_budget_curve_kappa0.pdf');plt.close(fig)
 def ci(v):return f"{v['mean']:+.6f} [{v['CI95'][0]:.6f}, {v['CI95'][1]:.6f}]"
 s=f'# Current no-shrinkage own-K budget: {ds}\n\nComplete128query/128parent historical fixed panel,8 independent configurations/query, whole seal→{score["checks"]}officialdensechecksmax{score["max_error"]:.17g}→{a["prediction_receipt_SHA_checks"]}predreceiptSHA checks plus parent/teacher/rawgrid hashes and all arrays rederived exactly/all means. These are1024budgetconfigurations/domain, not the full12205-query budget study. Current kappa0 choice followed exposed full results; no independent-unseen confirmation claim.\n\n|Schedule|K|tIoU %|vIoU %|R.3 %|R.5 %|Original expert forward s/query|Native-source severe t/v/union|\n|---|---:|---:|---:|---:|---:|---:|---|\n'
 for n in configs:
  m=r['means'][n];z=r['severe_vs_Source'][n];s+=f"|{n.rstrip('0123456789')}|{n[len(n.rstrip('0123456789')):]}|{100*m['t']:.6f}|{100*m['v']:.6f}|{100*m['vIoU_ge_03']:.6f}|{100*m['vIoU_ge_05']:.6f}|{costs[n]['mean_original_expert_forward_seconds']:.6f}|{z['t']}/{z['v']}/{z['union']}|\n"
 s+='\nEveryK uses its own original joint teacher boxes/masks/context and original raw grids; rebuild E/geometry/auxDP, genuinely repool, kappa0 posterior/MAP/newintervalDP. Hidden old kappa64 per-K consistency checks remain bitwise prerequisites; Hybrid8 is exactly current full NoShrink on the same queries. No K8 subset pretending to be smallK. Restore original Torch4 only during derive, otherstages1thread, sameCPU14-15. DefaultK8 is not changed from these curves. Absolute values must not be compared to full-target means.\n\n|Signed comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|\n|---|---:|---:|---|---:|\n'
 for key,v in r['paired'].items():
  z=tails[key];s+=f"|{key}|{ci(v['t'])}|{ci(v['v'])}|{z['severe_t']}/{z['severe_v']}/{z['severe_union_once']}|{z['rescue_union_once']}|\n"
 s+='\n## Duration groups\n\n|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|\n|---|---:|---:|---:|\n'
 for group,g in r['timegroups'].items():
  for key,v in g['paired'].items():s+=f"|{group} / {key}|{g['queries']}|{ci(v['t'])}|{ci(v['v'])}|\n"
 s+='\nPointwise parent-cluster CIs, no familywise correction. Allsigned positive/negative losses/rescues are retained. Source is native per-offset MAP+hull. Not every metric/schedule must increase with K; all exceptions/CIs crossing0 remain in tables. Originalexpert forward is only one nested component of deployment cost; source acquisition/requests/transport/startup are not zero or inferable from cachedCPUwall. No newSource/expert/GTscoring in this root export. Source/teacher costs inherited; current worker costs in cost_segments.json. PrivatePNG/PDF must be viewed by root before curve publication.\n';(folder/'ANALYSIS.md').write_text(s)
 rec=dict(status='root_exported_pending_PNG_visual_inspection',dataset=ds,audit_sha256=sha(auditpath),report_sha256=sha(folder/'report.json'),curve_TSV_sha256=sha(curve),SVG_sha256=sha(svg),PNG_sha256=sha(private/f'{ds}_budget_curve_kappa0.png'),PDF_sha256=sha(private/f'{ds}_budget_curve_kappa0.pdf'),code_sha256=sha(Path(__file__)),new_Source=0,new_expert=0,new_GT_scoring=0);atomic(D/f'budget_{ds}_root_export_receipt.json',rec);atomic(folder/'root_export_receipt.json',rec);print(ds,'budget root exported')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--dataset',choices=COUNTS,required=True);z=a.parse_args();run(z.dataset)
