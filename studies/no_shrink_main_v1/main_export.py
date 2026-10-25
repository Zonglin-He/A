"""Reuse fully audited NoShrink full-domain scalars and predictions; zero GT rescoring."""
from ns_common import *
import numpy as np

def run(ds):
 validate();audit=read(PREV/f'temporal_v2_{ds}_reporting_receipt.json');assert audit['status']=='verified_whole_domain' and audit['score_sha256']==sha(PREV/f'temporal_v2_{ds}_score_receipt.json');old=read(PREV/'temporal_v2_v2'/ds/'metrics.json');rows=[];lookup={}
 for x in old['private_rows']:
  values=dict(Source=x['values']['Source'],SourceJointMAP=x['values']['SourceJointMAP'],FullMAP=x['values']['NoShrink'],LegacyKappa64=x['values']['FullMAP']);rows.append({**x,'values':values});lookup[x['id']]=values
 arms=['Source','SourceJointMAP','FullMAP','LegacyKappa64'];pairs=[('FullMAP','Source'),('FullMAP','SourceJointMAP'),('FullMAP','LegacyKappa64')];folder=P/'TA'/ds;r=suite().parent_report(rows,arms,pairs,folder);suite().signed_tails(rows,pairs,folder);tails={}
 for a,b in pairs:
  dt=[100*(x['values'][a]['t']-x['values'][b]['t'])for x in rows];dv=[100*(x['values'][a]['v']-x['values'][b]['v'])for x in rows];tails[a+'-'+b]=dict(severe_t=sum(v< -5 for v in dt),severe_v=sum(v< -5 for v in dv),severe_union_once=sum(t< -5 or v< -5 for t,v in zip(dt,dv)),rescue_union_once=sum(t>5 or v>5 for t,v in zip(dt,dv)))
 atomic(folder/'all_paired_tails.json',tails)
 with(folder/'main_table.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['configuration','tIoU_percent','vIoU_percent','R03_percent','R05_percent','severe_t','severe_v','severe_union'])
  for a in arms:w.writerow([a,*[100*r['means'][a][k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05']],*[r['severe_vs_Source'][a][k]for k in ['t','v','union']]])
 s=f'# Selected no-shrinkage TA main: {ds}\n\n{r["queries"]} queries / {r["parents"]} parents. All NoShrink predictions were fully sealed, officially scored and exact-replay audited before this read-only export. Current FullMAP is exactly that kappa0 arm, with its own MAP interval and actual interval-conditioned spatial DP. Source remains the original native per-offset MAP and physical hull.\n\nUser explicitly selected this main method after both full results were exposed. This is a post-exposure method selection, not new independent confirmation. Historical kappa64 results and failures remain immutable.\n\n|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs native Source|\n|---|---:|---:|---:|---:|---|\n'
 for a in arms:
  m=r['means'][a];v=r['severe_vs_Source'][a];s+=f"|{a}|{100*m['t']:.6f}|{100*m['v']:.6f}|{100*m['vIoU_ge_03']:.6f}|{100*m['vIoU_ge_05']:.6f}|{v['t']}/{v['v']}/{v['union']}|\n"
 s+='\nMean gains do not remove negative tails. Both domains have more severe losses versus native Source than the historical kappa64 FullMAP; all signed differences, rescues, observed/unobserved changes and short/long pointwise parent-cluster CIs remain published. No new model, Source, teacher or GT scoring in this export. Cached replay times are not deployment latency.\n';(folder/'ANALYSIS.md').write_text(s)
 rec=dict(status='complete_from_previously_audited_NoShrink',queries=len(rows),parent_seal_sha256=sha(PREV/f'temporal_v2_{ds}_sealed.json'),parent_score_sha256=sha(PREV/f'temporal_v2_{ds}_score_receipt.json'),parent_reporting_sha256=sha(PREV/f'temporal_v2_{ds}_reporting_receipt.json'),parent_metrics_sha256=sha(PREV/'temporal_v2_v2'/ds/'metrics.json'),all_means_recomputed=True,main_table_sha256=sha(folder/'main_table.tsv'),new_model=0,new_GT_scoring=0);atomic(D/f'TA_{ds}_adoption_receipt.json',rec);atomic(folder/'adoption_receipt.json',rec)
 # Reuse optimizer baseline scalars; only Ours comparison changes.
 tta=read(PREV/'published_TTA'/ds/'metrics.json')if(PREV/'published_TTA'/ds/'metrics.json').exists()else None
 if tta:
  rr=[]
  for x in tta['private_rows']:rr.append({**x,'values':{**x['values'],'FullMAP':lookup[x['id']]['FullMAP']}})
  aa=list(rr[0]['values']);pp=[('FullMAP',a)for a in aa if a!='FullMAP'];suite().parent_report(rr,aa,pp,P/'TTA'/ds);suite().signed_tails(rr,pp,P/'TTA'/ds)
 print(ds,json.dumps(r['means']['FullMAP']),flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);z=p.parse_args();run(z.dataset)
