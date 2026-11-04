"""Offline GT oracle interval ceiling to separate sampling limits; not a method."""
from pathlib import Path
import sys,json,time
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'studies/temporal_query_semantics_v1'))
from common import *
def iou(a,b):
 t=max(0.,min(a[1],b[1])-max(a[0],b[0]));return t/(a[1]-a[0]+b[1]-b[0]-t)
def main():
 validate();sys.addaudithook(old.deny_gpu_media);torch.set_num_threads(1);oldroot=read(D/'ROOT_completion_v2.json');truth=old.selected.labels('vidstg');seal=read(D/'sealed.json');records={r['id']:r for r in read(D/'records.json')};rows=[];public=[];mx=0.
 for pair in read(D/'cohort.json'):
  values=[]
  for side in ['a','b']:
   key=pair[side];e=seal['files'][key];assert sha(D/e['path'])==e['sha256'];g=truth[key]['interval'];fps=records[key]['metadata']['fps']
   with np.load(D/e['path'])as z:
    ids=z['ids'];start=ids[:,None];end=ids[None,:]+1;valid=end>start;inter=np.maximum(0,np.minimum(end,g[1])-np.maximum(start,g[0]));score=np.divide(inter,np.maximum(end,g[1])-np.minimum(start,g[0]),out=np.zeros_like(inter,dtype=float),where=valid);grid=np.triu(np.ones_like(score,dtype=bool));support=z['input_q0']>0;bestg=float(score[grid].max());bestq=float(score[support].max());indg=max(iou([int(ids[i]),int(ids[j])+1],g)for i in range(len(ids))for j in range(i,len(ids)));indq=max(iou([int(ids[i]),int(ids[j])+1],g)for i,j in np.argwhere(support));mx=max(mx,abs(bestg-indg),abs(bestq-indq));assert mx<1e-12
    values.append(dict(grid=bestg,prior=bestq,gt_seconds=(g[1]-g[0])/fps,clip_seconds=(ids[-1]-ids[0]+1)/fps,native_sampled_frames_in_GT=int(((ids>=g[0])&(ids<g[1])).sum()),native_frames=len(ids),actual_t=iou(z['CorrectedROI_interval'],g)))
    public.append(dict(pair=pair['pair'],side=side,grid_oracle_t_percent=100*bestg,prior_support_oracle_t_percent=100*bestq,GT_event_duration_seconds=(g[1]-g[0])/fps,native_frames=len(ids),sampled_GT_frames=values[-1]['native_sampled_frames_in_GT'],Current_t_percent=100*values[-1]['actual_t']))
  a,b=values;rows.append(dict(pair=pair['pair'],grid_oracle_t_percent=50*(a['grid']+b['grid']),prior_support_oracle_t_percent=50*(a['prior']+b['prior']),grid_oracle_both_t03_percent=100.*(a['grid']>=.3 and b['grid']>=.3),prior_support_oracle_both_t03_percent=100.*(a['prior']>=.3 and b['prior']>=.3),GT_event_duration_seconds=.5*(a['gt_seconds']+b['gt_seconds']),GT_native_no_sample_queries_percent=50.*((a['native_sampled_frames_in_GT']==0)+(b['native_sampled_frames_in_GT']==0)),actual_both_t03_percent=100.*(a['actual_t']>=.3 and b['actual_t']>=.3)))
 report=dict(scope='Offline GT oracle upper bound on existing native grid and separately strictly-positive q0support. Nooracledatausedforinference,selectionorparameters;notdeployablebaseline.',summary=bootstrap(rows,[k for k in rows[0]if k!='pair']),grid_oracle_both_t03_pairs=sum(r['grid_oracle_both_t03_percent']>0 for r in rows),prior_support_oracle_both_t03_pairs=sum(r['prior_support_oracle_both_t03_percent']>0 for r in rows),actual_both_t03_pairs=sum(r['actual_both_t03_percent']>0 for r in rows),zero_sample_queries=sum(r['sampled_GT_frames']==0 for r in public),max_independent_enumeration_error=mx,queries=64,pairs=32,report_sha256=sha(P/'report.json'),new_Source=0,new_expert=0,new_GPU=0,new_model=0,new_official_tube_scoring=0)
 atomic(P/'sampling_ceiling.json',report);atomic(D/'sampling_ceiling_private_rows.json',rows)
 import csv
 with(P/'all64_sampling_ceiling.tsv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(public[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(public)
 text=(P/'ANALYSIS.md').read_text();text+='\n## Sampling/prior-support prerequisite check (offline GT oracle only)\n\n'
 for k,z in report['summary']['statistics'].items():text+=f'- {k}: {z["mean"]:.6f},95%videoCI{z["CI95"]}.\n'
 text+=f'\nBoth events can reach tIoU>=.3 on the existing grid in{report["grid_oracle_both_t03_pairs"]}/32pairs, and within the original strictly-positive q0support in{report["prior_support_oracle_both_t03_pairs"]}/32pairs; actual E-ROI update succeeds in{report["actual_both_t03_pairs"]}/32pairs. {report["zero_sample_queries"]}/64GT events have no native sampled frame inside. Thus distinguish absent sampled evidence from decoder support; intervals can bracket events even without an internal sample. Oracle calculations use GT after all predictions/separateaudit and do not alter predictions or selection. Independent full endpoint enumeration matches vectorized upper bounds, maxerror{mx}. No label-assisted adaptation or oracle-method performance claim.\n'
 (P/'ANALYSIS.md').write_text(text);oldroot.update(analysis_sha256=sha(P/'ANALYSIS.md'),sampling_ceiling_sha256=sha(P/'sampling_ceiling.json'),sampling_ceiling_code_sha256=sha(Path(__file__)),sampling_oracle_is_offline_only=True,time=time.time());atomic(D/'ROOT_completion_v3.json',oldroot);atomic(P/'completion_scalars.json',oldroot);print(json.dumps(report))
if __name__=='__main__':main()
