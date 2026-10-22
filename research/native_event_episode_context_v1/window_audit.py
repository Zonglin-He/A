"""Separately bound postseal window/case audit; never inference selection."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_episode_context_v1.run import D,P,read,put,sha,records
import numpy as np,torch

def main():
 s=read(D/'sealed.json');p=read(D/'plan.json');r=read(P/'report.json');assert read(D/'score_receipt.json')['report_sha256']==sha(P/'report.json')
 labels=R/'data/evaluation/development_labels.jsonl';assert sha(labels)==p['labels_sha256'];gt={x['id']:x for x in map(json.loads,labels.read_text().splitlines())if x['dataset']=='hc2'};vals={x['query']:x for x in r['rows']};rows=[]
 for row in records():
  k=row['id'];e=s['files'][k];assert sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],map_location='cpu',weights_only=False)
  if not e['supported']:continue
  ids=np.asarray(row['metadata']['frame_ids']);win=tr['window'];target=gt[k]['interval'];start,end=target;length=end-start
  def cover(lo):return max(0,min(ids[lo+win['width']-1]+1,end)-max(ids[lo],start))/length
  v=vals[k];a='EpisodeClauseFusion';q=tr['q'];ii,jj=np.triu_indices(len(ids),1);span=ids[jj]+1-ids[ii];fps=row['metadata']['fps'];expected={arm:float((span*q[arm][ii,jj]).sum()/fps)for arm in [a,'GlobalClauseFusion','NativeG6','EpisodeSelfHullFusion']}
  rows.append(dict(query=k,duration_group=v['duration_group'],episode_gt_coverage=cover(win['lo']),shifted_gt_coverage=cover(win['shifted_lo']),delta_t_vs_global_pp=100*(v['values'][a]['t']-v['values']['GlobalClauseFusion']['t']),delta_t_vs_native_pp=100*(v['values'][a]['t']-v['values']['NativeG6']['t']),delta_t_vs_source_pp=100*(v['values'][a]['t']-v['values']['Source']['t']),expected_interval_seconds=expected,decoded_interval_seconds={arm:v['values'][arm]['length_seconds']for arm in expected},true_duration_seconds=length/fps))
 summary={}
 for name,rr in [('supported52',rows),('window_covers_event',[x for x in rows if x['episode_gt_coverage']>=1-1e-10]),('window_cuts_event',[x for x in rows if x['episode_gt_coverage']<1-1e-10]),('short',[x for x in rows if x['duration_group']=='short_le5s']),('long',[x for x in rows if x['duration_group']=='long_gt5s'])]:
  summary[name]=dict(count=len(rr),mean_window_GT_coverage=float(np.mean([x['episode_gt_coverage']for x in rr])),mean_shifted_GT_coverage=float(np.mean([x['shifted_gt_coverage']for x in rr])),dt_vs_global_pp=float(np.mean([x['delta_t_vs_global_pp']for x in rr])),dt_vs_source_pp=float(np.mean([x['delta_t_vs_source_pp']for x in rr])),decoded_length_seconds={arm:float(np.mean([x['decoded_interval_seconds'][arm]for x in rr]))for arm in ['EpisodeClauseFusion','GlobalClauseFusion','NativeG6']},true_duration_seconds=float(np.mean([x['true_duration_seconds']for x in rr])))
 out=dict(scope='Postseal offline case/window audit, no online GT gates; full-event GT not atomic-event truth',summary=summary,rows=rows);put(P/'window_audit.json',out)
 lines=['# Postseal episode coverage and case audit','','Window coverage below uses GT only AFTER prediction seal/scoring. No window or query is removed or selected using these values. A shifted window may still cover the actual event.','','|Offline cohort|N|episode event coverage|shifted coverage|t gain vs global clause pp|t gain vs Source pp|','|---|---:|---:|---:|---:|---:|']
 for name,x in summary.items():lines.append(f"|{name}|{x['count']}|{x['mean_window_GT_coverage']:.4f}|{x['mean_shifted_GT_coverage']:.4f}|{x['dt_vs_global_pp']:+.4f}|{x['dt_vs_source_pp']:+.4f}|")
 lines+=['','## Concrete same-query contrasts','','|Query|episode coverage|t change vs global clause pp|t change vs Source pp|','|---|---:|---:|---:|']
 ordered=sorted(rows,key=lambda x:x['delta_t_vs_global_pp'])
 for x in ordered[:4]+ordered[-4:]:lines.append(f"|{x['query']}|{x['episode_gt_coverage']:.4f}|{x['delta_t_vs_global_pp']:+.4f}|{x['delta_t_vs_source_pp']:+.4f}|")
 lines+=['','Coverage and case associations do not prove cropping is the sole cause: both fusion context and available interval support change. Short-event length summaries separate expansion from mere shifted-context quality. No GT-selected fallback is adopted. [Complete numeric audit](window_audit.json).']
 (P/'WINDOW_AUDIT.md').write_text('\n'.join(lines)+'\n');put(D/'window_audit_receipt.json',dict(code_sha256=sha(Path(__file__)),seal_sha256=sha(D/'sealed.json'),labels_sha256=sha(labels),scored_report_sha256=sha(P/'report.json'),audit_sha256=sha(P/'window_audit.json'),online_GT=False));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
