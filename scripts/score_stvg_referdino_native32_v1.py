"""GT-only-after-cost-pass-and-all32-seal metrics; never oracle-select outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_referdino_native32_v1 import D,P,PRE,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def mean(v):
 v=[float(x)for x in v if x is not None];return float(np.mean(v))if v else None

def main():
 import torch
 torch.set_num_threads(4);p=verify(True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert read(D/'COST_GATE.json')['status']=='pass'and bar['count']==32 and not torch.cuda.is_initialized();start=time.perf_counter()
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in ['hc2','vidstg']};gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in plans};rows=[];frames=[];node_rows=[]
 for r in p['records']:
  z=load(D/'predictions'/(r['alias']+'.pt'));ds=r['dataset'];q=r['query_ordinal'];dense,spans,_=gt[ds];row=plans[ds][q];m=metrics(z,row,dense[q],spans[q]);ids=z['frame_ids'];obs=set(z['positions8']);old=set(z['planned_positions']);node=[]
  for i,fid in enumerate(ids):
   known=int(fid)in dense[q]
   if not known:continue
   target=np.asarray(dense[q][int(fid)],float)/np.array([row['input']['width'],row['input']['height']]*2);wh=target[2:]-target[:2];center=(target[2:]+target[:2])/2;eb=z['outputs']['E_MBR'][i];far=bool(np.any(abs(eb[:2]-center)/np.maximum(wh,1e-12)>.5));rat=eb[2:]/np.maximum(wh,1e-12);scale=bool(np.any((rat>2)|(rat<.5)));vv={arm:iou_gt(b[i],dense[q][int(fid)],row,ds)for arm,b in z['outputs'].items()};node.append(dict(position=i,original4=i in old,observed8=i in obs,far_center_E=far,scale_E=scale,E_good=vv['E_MBR']>=.5,E_very_bad=vv['E_MBR']<.1,values=vv))
  rr=dict(alias=r['alias'],dataset=ds,formerly_skipped=r['alias']in p['configuration']['formerly_skipped'],selected_query=z['native']['selected_query'],expert_cost=z['native']['cost'],**m)
  rr['subsets']={}
  for subset,pred in [('original4',lambda n:n['original4']),('additional4',lambda n:n['observed8']and not n['original4']),('outside8',lambda n:not n['observed8']),('far_center_E',lambda n:n['far_center_E']),('scale_E',lambda n:n['scale_E']),('E_good',lambda n:n['E_good']),('E_very_bad',lambda n:n['E_very_bad'])]:
   nn=[n for n in node if pred(n)];rr['subsets'][subset]=dict(known_nodes=len(nn),means={arm:mean([n['values'][arm]for n in nn])for arm in z['outputs']},recovered_to_IoU_ge05={arm:sum(n['values'][arm]>=.5 for n in nn)for arm in z['outputs']})
  for j,pos in enumerate(z['positions8']):
   fid=int(ids[pos]);known=fid in dense[q];frames.append(dict(alias=r['alias'],dataset=ds,position=pos,original4=pos in old,GT_known=known,selected_query=rr['selected_query'],native_frame_score=z['native']['selected_frame_scores'][j].tolist(),teacher_IoU=iou_gt(z['native']['boxes'][j],dense[q][fid],row,ds)if known else None,E_IoU=iou_gt(z['outputs']['E_MBR'][pos],dense[q][fid],row,ds)if known else None,Ref_E_IoU=iou_gt(z['outputs']['Ref_E_MBR'][pos],dense[q][fid],row,ds)if known else None))
  node_rows.append(dict(alias=r['alias'],dataset=ds,known_nodes=node));rows.append(rr)
 groups={}
 for name,rr in [('HC16',[r for r in rows if r['dataset']=='hc2']),('Vid16',[r for r in rows if r['dataset']=='vidstg']),('Formerly_skipped5',[r for r in rows if r['formerly_skipped']])]:
  arms=['E_MBR','Ref_E_MBR','Ref_direct8_MBR'];pairs={}
  for arm in arms[1:]:pairs[arm+'_minus_E']={ep:paired([None if r['values'][arm][ep]is None or r['values']['E_MBR'][ep]is None else 100*(r['values'][arm][ep]-r['values']['E_MBR'][ep])for r in rr])for ep in ['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s']}
  anchor={}
  for mask in ['all8','original4','additional4']:
   vals=[]
   for r in rr:
    ff=[f for f in frames if f['alias']==r['alias']and f['GT_known']and(mask=='all8'or f['original4']==(mask=='original4'))];vals.append(mean([100*(f['teacher_IoU']-f['E_IoU'])for f in ff]))
   anchor[mask]=paired(vals)
  groups[name]=dict(parents=len(rr),means={arm:{ep:mean([r['values'][arm][ep]for r in rr])for ep in ['v','t','s','observed_actual_s','unobserved_actual_s']}for arm in arms},paired=pairs,teacher_actual_anchor_minus_E=anchor,subsets={key:{arm:paired([None if r['subsets'][key]['means'][arm]is None else 100*(r['subsets'][key]['means'][arm]-r['subsets'][key]['means']['E_MBR'])for r in rr])for arm in arms[1:]}for key in rr[0]['subsets']},cost_expert_seconds=[r['expert_cost']['expert_seconds']for r in rr])
 put(P/'RESULTS_PUBLIC.json',dict(status='completed',groups=groups,rows=rows,all32_sealed_before_GT=True,development_exposed=True,identity_not_inferred_from_IoU=True));put(P/'FRAME_DIAGNOSTICS_PUBLIC.json',dict(rows=frames,total_observed_slots=256,unknown_not_zero_filled=True));put(P/'NODE_ERROR_DIAGNOSTICS_PUBLIC.json',dict(rows=node_rows,GT_only_postseal_categories=True,identity_labels_unavailable=True))
 put(P/'COST_BREAKDOWN.json',dict(GPU_allocation_seconds=bar['GPU_allocation_seconds'],cost_gate=read(P/'COST_GATE_PUBLIC.json'),Refer_completed_queries=32,Refer_warmup_queries=1,Refer_total_expert_seconds=sum(r['expert_cost']['expert_seconds']for r in rows),Refer_total_forward_seconds=sum(r['expert_cost']['forward_seconds']for r in rows),Refer_max_peak_allocated_bytes=max(r['expert_cost']['peak_allocated_bytes']for r in rows),T_forwards=20,Source_forwards=0,training_OPD_updates=0,scoring_CPU_seconds=time.perf_counter()-start,RGB_decode_model_load_setup_cost_excluded_from_perquery_gate=True,old_cache_cost_excluded=True));print({k:{a:v['v']for a,v in g['paired'].items()}for k,g in groups.items()},flush=True)
if __name__=='__main__':main()
