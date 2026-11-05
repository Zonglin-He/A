"""ROOT whole-domain scalar interpretation; never launch a model or score GT."""
from pathlib import Path
import sys,os,json,time,csv,numpy as np
R=Path(__file__).resolve().parents[3];sys.path[:0]=[str(R/'studies/source_sam21_tr_v1'),str(R),str(R/'tools')]
import common as c
def independent_boot(rr,a,b):
 if not rr:return dict(queries=0,parents=0,statistics={})
 names=['delta_t_pp','delta_v_pp','delta_unobserved_IoU_pp','improved_v_percent','harmed_v_percent','severe_t_percent','severe_v_percent','severe_union_once_percent','rescue_union_percent'];parents=sorted({r['parent']for r in rr});index={v:i for i,v in enumerate(parents)};s=np.zeros((len(parents),len(names)));n=np.zeros_like(s);vals=[[]for k in names]
 for r in rr:
  x,y=r['values'][a],r['values'][b];dt=100*(x['t']-y['t']);dv=100*(x['v']-y['v']);u=None if x['unobserved_actual_sIoU']is None or y['unobserved_actual_sIoU']is None else 100*(x['unobserved_actual_sIoU']-y['unobserved_actual_sIoU']);v=[dt,dv,u,100.*(dv>1e-10),100.*(dv< -1e-10),100.*(dt< -5),100.*(dv< -5),100.*(dt< -5 or dv< -5),100.*(dt>5 or dv>5)]
  for j,x in enumerate(v):
   if x is not None:s[index[r['parent']],j]+=x;n[index[r['parent']],j]+=1;vals[j].append(x)
 k=len(parents);w=np.random.default_rng(20261022).multinomial(k,np.full(k,1/k),10000);den=w@n;bs=np.divide(w@s,den,out=np.full_like(den,np.nan),where=den>0);stats={}
 for j,name in enumerate(names):
  z=bs[:,j];z=z[np.isfinite(z)];stats[name]=dict(mean=float(np.mean(vals[j]))if vals[j]else None,CI95=np.quantile(z,[.025,.975]).tolist()if len(z)else None,queries=len(vals[j]),parents=int((n[:,j]>0).sum()))
 return dict(queries=len(rr),parents=k,statistics=stats)
def run(stage):
 c.check(True);allout={};checks=0
 for ds in c.COUNTS:
  d=c.D/stage/ds;p=c.P/stage/ds;ack=c.read(d/'reporting_receipt.json');assert ack['report_sha256']==c.sha(p/'report.json')and ack['metrics_sha256']==c.sha(d/'metrics.json');exp=c.read(d/'export_receipt.json');assert exp['audit_sha256']==c.sha(d/'reporting_receipt.json');rr=c.read(d/'metrics.json')['private_rows'];rep=c.read(p/'report.json');assert len(rr)==(128 if stage=='pilot'else c.COUNTS[ds]);threshold_equal=0
  for a in c.ARMS:
   for f,value in rep['means'][a].items():
    vv=[r['values'][a][f]for r in rr if r['values'][a][f]is not None];actual=float(np.mean(vv))if vv else None;assert(actual is None and value is None)or abs(actual-value)<1e-12;checks+=1
   for q in [.3,.5]:
    eq=sum(r['values'][a]['v']==q for r in rr);threshold_equal+=eq;strict=sum(r['values'][a]['v']>q for r in rr)/len(rr);assert abs(strict-rep['means'][a]['R03'if q==.3 else'R05'])<1e-12,'Preserve original score, export a new strict-threshold report if equality occurs'
  for group,pairs in rep['paired'].items():
   qq=[r for r in rr if group=='all'or r['duration_group']==group]
   for name,want in pairs.items():
    a,b=name.split('-');got=independent_boot(qq,a,b);assert got['queries']==want['queries']and got['parents']==want['parents']
    for f,v in want['statistics'].items():
     actual=got['statistics'][f];assert abs(actual['mean']-v['mean'])<2e-12 and np.max(np.abs(np.asarray(actual['CI95'])-v['CI95']))<2e-12;checks+=1
  tails={}
  for a,b in c.PAIRS:
   v=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rr]);t=np.array([100*(r['values'][a]['t']-r['values'][b]['t'])for r in rr]);tails[a+'-'+b]=dict(improved=int((v>1e-10).sum()),harmed=int((v< -1e-10).sum()),tie=int((abs(v)<=1e-10).sum()),v_loss_gt5pp=int((v< -5).sum()),v_loss_ge10pp=int((v<=-10).sum()),v_gain_gt5pp=int((v>5).sum()),severe_t_or_v_union_once=int(((t< -5)|(v< -5)).sum()),rescue_t_or_v_union_once=int(((t>5)|(v>5)).sum()))
  out=dict(stage=stage,dataset=ds,queries=len(rr),parents=rep['parents'],means=rep['means'],paired=rep['paired'],signed_tails=tails,efficiency_and_empty=c.read(p/'efficiency_and_empty.json'),official_strict_recall_threshold_recheck=True,exact_threshold_values=threshold_equal,limitations=['SAM2TR uses original Source-box temporal-refinement interval; SAM2 outputs do not update its temporal evidence.','Tracking sees Source boxes and sampled RGB, no ReferDINO/text/E/Full-event/DP. Full and SAM2 are complete deployment alternatives with different evidence and all-frame external-model compute.','Original fixed panel is exposed; pointwise paired-parent CIs, not unknown confirmation or equivalence.','Native unobserved IoU over GT-supported sampled frames, not conditional dense event IoU.','Experimental query/request wall includes raw output and provenance I/O; inherited Source acquisition is nonzero, not matched deployment timing.','Current Full soft-anchor behavior unchanged. No production/paper or old prediction was edited.'])
  if stage=='pilot':
   td=c.R/'outputs/expert_guided_ttft_v1/target_pilot_v1'/ds;tack=c.read(td/'reporting_receipt.json');assert tack['metrics_sha256']==c.sha(td/'metrics.json');other={r['id']:r for r in c.read(td/'metrics.json')['private_rows']};joined=[]
   for r in rr:
    q=other[r['id']];assert q['parent']==r['parent']
    for a in ['Source','SourceTR']:
     for f in ['v','t']:assert abs(r['values'][a][f]-q['values'][a][f])<1e-12
    joined.append(dict(r,values={**r['values'],'TTFT':q['values']['TTFT']}))
   out['TTFT_panel_scalar_reuse']=dict(mean_t=100*float(np.mean([r['values']['TTFT']['t']for r in joined])),mean_v=100*float(np.mean([r['values']['TTFT']['v']for r in joined])),SAM2TR_minus_TTFT=independent_boot(joined,'SAM2TR','TTFT'),scope='Two already completed full pipelines, different external model/information and optimizer budgets. No new TTFT/model/GT scoring, no discarded TTFT outcome.')
  c.atomic(p/'ROOT_report.json',out);c.atomic(d/'ROOT_receipt.json',dict(status='ROOT_absolute_strictR_pairedCI_positive_negative_shortlong_empty_and_cost_review_complete',ROOT_report_sha256=c.sha(p/'ROOT_report.json'),audit_sha256=c.sha(d/'reporting_receipt.json'),new_GPU=0,new_Source=0,new_expert=0,new_GT=0,worker=c.identity()));allout[ds]=out
 lines=[f'# ROOT {stage} comparison: Source-prompt SAM2.1 + Source-box TR','', '|Method|HC tIoU|HC vIoU|Vid tIoU|Vid vIoU|','|---|---:|---:|---:|---:|']
 for a in c.ARMS:lines.append('|'+a+'|'+'|'.join(f"{100*allout[ds]['means'][a][f]:.6f}"for ds in c.COUNTS for f in ['t','v'])+'|')
 lines+=['','Full-SAM2TR:']
 for ds,z in allout.items():
  q=z['paired']['all']['Current-SAM2TR']['statistics']['delta_v_pp'];lines.append(f"{ds}: {q['mean']:+.6f} pp,95%CI {q['CI95']}.")
 lines+=['','The HC mean favors Current Full and Vid mean favors SAM2TR; both panel vIoU CIs cross zero. No established Full vIoU advantage in both domains. SAM2TR-SourceTR is null in HC and positive in Vid on a genuinely shared SourceTR interval. Primary source-prompt SAM2 is not a same-ReferDINO-evidence propagation control. All negative tails, empty masks, original TTFT results and Source acquisition costs remain.','', 'New stage remains LOCAL until a new verified publication receipt; ongoing full prefixes are not scored or used to select this configuration.']
 (c.P/stage/'ROOT_ANALYSIS.md').write_text('\n'.join(lines)+'\n');c.atomic(c.P/stage/'ROOT_summary.json',allout);c.atomic(c.D/stage/'ROOT_completion.json',dict(status='whole_stage_ROOT_interpretation_complete_pending_publication',domains=list(allout),absolute_and_grouped_paired_CI_checks=checks,summary_sha256=c.sha(c.P/stage/'ROOT_summary.json'),new_Source=0,new_expert=0,new_GT=0,worker=c.identity(),time=time.time()));print(json.dumps(dict(ROOT_stage_complete=stage,checks=checks)),flush=True)
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--stage',choices=['pilot','full'],required=True);x=a.parse_args();run(x.stage)
