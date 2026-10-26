"""All label variants and conservative semantic sensitivity after full metric audit."""
from common import *
import argparse
def export(ds):
 import numpy as np
 from full_score import cluster
 validate();start=time.perf_counter();d=D/ds;p=P/ds;audit=read(d/'reporting_receipt.json');assert audit['status']=='verified_whole_domain_all_arrays_parent_receipts_means_and_text_counts';assert sha(d/'metrics.json')==audit['metrics_sha256'];rows=read(d/'metrics.json')['private_rows'];label=read(D/'conservative_text_sensitivity_v1.json')['datasets'][ds]['queries'];pos=read(D/'semantic_annotation_v1.json')['datasets'][ds]['queries'];assert{r['id']for r in rows}==set(label)==set(pos)
 groups={name:[r for r in rows if label[r['id']]['group']==name]for name in ['event_consensus','conservative_static_visual_proxy','unclassified']};groups['all']=rows;summary={}
 for name,rr in groups.items():
  summary[name]=dict(queries=len(rr),parents=len({r['parent']for r in rr}),paired=cluster(rr,'FullMAP','NoVisualUpdate',[('t',100),('v',100)])if rr else{},severe_t=sum(100*(r['values']['FullMAP']['t']-r['values']['NoVisualUpdate']['t'])< -5 for r in rr),severe_v=sum(100*(r['values']['FullMAP']['v']-r['values']['NoVisualUpdate']['v'])< -5 for r in rr))
 pp=sorted({r['parent']for r in rows});idx={k:i for i,k in enumerate(pp)};draw=np.random.default_rng(20261026).multinomial(len(pp),[1/len(pp)]*len(pp),size=10000)
 def boot(rr):
  s=np.zeros(len(pp));n=np.zeros(len(pp))
  for r in rr:j=idx[r['parent']];s[j]+=100*(r['values']['FullMAP']['t']-r['values']['NoVisualUpdate']['t']);n[j]+=1
  den=draw@n;return np.divide(draw@s,den,out=np.full(10000,np.nan),where=den>0)
 supported=all(summary[g]['queries']>=30 and summary[g]['parents']>=10 for g in ['event_consensus','conservative_static_visual_proxy']);b1,b2=boot(groups['event_consensus']),boot(groups['conservative_static_visual_proxy']);valid=np.isfinite(b1)&np.isfinite(b2)
 contrast=dict(status='exploratory_proxy_supported_counts'if supported else'insufficient_static_or_event_support',definition='(Full-NoVisualUpdate event consensus t gain)-(Full-NoVisualUpdate conservative static visual proxy t gain)',CI95=np.quantile((b1-b2)[valid],[.025,.975]).tolist()if supported and valid.any()else None,pointwise_joint_parent_resampling=True,not_pure_action_vs_appearance_gold=True)
 if supported:contrast['mean']=summary['event_consensus']['paired']['t']['mean']-summary['conservative_static_visual_proxy']['paired']['t']['mean']
 report=read(p/'report.json');baseline=report['paired'];out=dict(status='root_conservative_semantic_sensitivity_exported_not_gold',groups=summary,event_vs_static_proxy_contrast=contrast,original_keyword_report_retained=True,POS_report_counts_retained=True,root62_single_AI_review_not_human_gold=True,classification_annotation_sha256=sha(D/'conservative_text_sensitivity_v1.json'),original_keyword_SHA=sha(D/'text_classification_v4.json'),POS_annotation_sha256=sha(D/'semantic_annotation_v1.json'))
 atomic(p/'conservative_semantic_report_v1.json',out)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,4.5));validgroups=[g for g in ['all','event_consensus','conservative_static_visual_proxy','unclassified']if summary[g]['queries']]
 for j,g in enumerate(validgroups):
  v=summary[g]['paired']['t'];lo,hi=v['CI95'];axes[0].errorbar(j,v['mean'],yerr=[[v['mean']-lo],[hi-v['mean']]],fmt='o',capsize=4);axes[0].text(j,v['mean'],f"  n={summary[g]['queries']}",fontsize=8)
 axes[0].set_xticks(range(len(validgroups)),[dict(all='All',event_consensus='Event\nconsensus',conservative_static_visual_proxy='Static visual\nproxy',unclassified='Unclassified')[g]for g in validgroups]);axes[0].axhline(0,color='gray',lw=1);axes[0].set_ylabel('Full - No visual update: tIoU pp');axes[0].set_title('Query-only conservative sensitivity, not gold')
 arms=['FullMAP','LengthMeanEvidence','TemperatureEvidence2','SmoothPrior3'];rr=read(d/'metrics.json')['private_rows'];pair={a:cluster(rr,a,'NoVisualUpdate',[('t',100)])['t']for a in arms}
 for j,a in enumerate(arms):
  v=pair[a];lo,hi=v['CI95'];axes[1].errorbar(j,v['mean'],yerr=[[v['mean']-lo],[hi-v['mean']]],fmt='o',capsize=4)
 axes[1].set_xticks(range(4),['Current\nupdate','Mean\nevidence','Evidence\nT=2','Prior\nsmoothing']);axes[1].axhline(0,color='gray',lw=1);axes[1].set_ylabel('Against no visual update: tIoU pp');axes[1].set_title('Full-domain fixed controls, parent 95% CI');fig.suptitle(ds.upper()+' temporal posterior risk — all signed effects retained');fig.tight_layout();fig.savefig(d/'semantic_diagnostics_v1.png',dpi=160);fig.savefig(p/'semantic_diagnostics_v1.svg');plt.close(fig)
 atomic(d/'semantic_root_export_receipt_v1.json',dict(status='root_numerical_and_semantic_sensitivity_exported_pending_actual_image_inspection_and_publication',audit_sha256=sha(d/'reporting_receipt.json'),semantic_report_sha256=sha(p/'conservative_semantic_report_v1.json'),PNG_sha256=sha(d/'semantic_diagnostics_v1.png'),SVG_sha256=sha(p/'semantic_diagnostics_v1.svg'),code_sha256=sha(Path(__file__)),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_GT_scoring=0,classification_gold_claim=False))
 print(json.dumps(dict(dataset=ds,groups={g:{k:v for k,v in x.items()if k!='paired'}for g,x in summary.items()},contrast=contrast)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);export(p.parse_args().dataset)
