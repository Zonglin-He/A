"""Full final-readout scoring and reference/observed-unobserved diagnostics after seals."""
import sys,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from scripts.stvg_temporal_caption_query_v1 import roster,catalog,read,sha,write
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal,_metric_functions,score_tube_xyxy
from vg_tta.tastvg_oracle_event5_v1 import DenseTube,box_iou
B=ROOT/'artifacts/stvg_video_conditioned_global_referent_v1';torch.set_num_threads(2)
def main():
 for phase in ['method','oracle']:
  z=read(B/f'{phase.upper()}_STUDENT_GLOBAL_BARRIER.json');assert z['count']==8 and z['GT_read']==(phase=='oracle')
  for f in z['files']:assert sha(ROOT/f['path'])==f['sha256']
 cases={r['case_id']:catalog(r)[0]for r in roster()};truths,bindings=_truths_after_seal(cases);off,ind,_=_metric_functions();rows=[];errors=[];candidates=[];inp={r['case_id']:r for r in read(B/'CANDIDATE_INPUT_BARRIER.json')['records']}
 def cx(a):return np.concatenate([(a[:,:2]+a[:,2:])/2,a[:,2:]-a[:,:2]],1)
 for idx,r in enumerate(roster()):
  cid=r['case_id'];c=cases[cid];gt=truths[cid];m=torch.load(B/'method/student'/f'{cid}.pt',map_location='cpu',weights_only=False);o=torch.load(B/'oracle/student'/f'{cid}.pt',map_location='cpu',weights_only=False);fid=m['frame_ids'];arms={};sp={};moves={};meta=dict(frame_ids=fid,input=dict(width=c['width'],height=c['height']));Source=DenseTube(cx(m['arms']['F']['boxes']),meta,gt['truth'],gt['span'],clip=r['dataset']=='hc2');poses=inp[cid]['positions'];mask=np.isin(Source.fids,[fid[p]for p in poses]);alias=('HC' if r['dataset']=='hc2'else'V')+f'{idx%4+1:02d}'
  for phase,z in [('method',m),('oracle',o)]:
   for name,a in z['arms'].items():
    if phase=='oracle' and name in ['F','T','Projection']:continue
    key=name;v=score_tube_xyxy(a['boxes'],fid,c['width'],c['height'],a['interval'],gt['truth'],gt['span'],r['dataset'],official=off,independent=ind);arms[key]=v['metrics'];errors.append(v['official_independent_max_abs_error']);tube=DenseTube(cx(a['boxes']),meta,gt['truth'],gt['span'],clip=r['dataset']=='hc2');delta=tube.iou-Source.iou;sp[key]=dict(observed_GT_count=int(mask.sum()),unobserved_GT_count=int((~mask).sum()),observed_sIoU_delta_pp=float(delta[mask].mean()*100)if mask.any()else None,unobserved_sIoU_delta_pp=float(delta[~mask].mean()*100)if(~mask).any()else None)
   for name,fit in z['fits'].items():
    ref=name.rsplit('_',1)[0];anchors=z['supervision_anchors'][ref];p=[a['position']for a in anchors];target=np.array([a['box_cxcywh']for a in anchors],float);before=z['source_boxes'].numpy();after=fit['final_cached_decoder_boxes'].numpy();initial=np.concatenate([v.numpy().reshape(-1)for v in fit['initial_state'].values()]);final=np.concatenate([v.numpy().reshape(-1)for v in fit['final_state'].values()]);bc=np.concatenate([before[:,:2]-before[:,2:]/2,before[:,:2]+before[:,2:]/2],1);ac=np.concatenate([after[:,:2]-after[:,2:]/2,after[:,:2]+after[:,2:]/2],1);tc=np.concatenate([target[:,:2]-target[:,2:]/2,target[:,:2]+target[:,2:]/2],1)if len(target)else np.empty((0,4));moves[name]=dict(reference_count=len(p),optimizer_steps=fit['optimizer_steps'],initial_reference_IoU=float(box_iou(bc[p],tc).mean())if p else None,final_reference_IoU=float(box_iou(ac[p],tc).mean())if p else None,parameter_delta_norm=float(np.linalg.norm(final-initial)),all_frame_box_RMS=float(np.sqrt(np.mean((after-before)**2))),reference_GT_oracle=phase=='oracle',fit_nominal_GT_read_false_means_no_label_file_open_not_label_free=phase=='oracle')
  for cand in inp[cid]['candidates']:
   values=[]
   for k,f in enumerate(inp[cid]['physical_frame_ids']):
    if f in gt['truth']:
     norm=np.array(gt['truth'][f],float)/[c['width'],c['height'],c['width'],c['height']];values.append(float(box_iou(cand['boxes_xyxy'][k],norm)))
   candidates.append(dict(case=alias,dataset=r['dataset'],candidate_id=cand['candidate_id'],window_id=cand['window_id'],available_GT_anchor_count=len(values),reference_mean_GT_IoU=float(np.mean(values))if values else None))
  rr=[a for a in candidates if a['case']==alias];rows.append(dict(case=alias,dataset=r['dataset'],arms=arms,spatial=sp,movement=moves,candidate_count=len(rr),candidate_Oracle_mean_GT_IoU=max((a['reference_mean_GT_IoU']for a in rr if a['reference_mean_GT_IoU']is not None),default=None),Source_mean_GT_IoU=rr[0]['reference_mean_GT_IoU'],GT_read=True,offline_oracle_separate=True))
 groups=[]
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds]
  for arm in rr[0]['arms']:
   mm={k:statistics.mean(r['arms'][arm][k]for r in rr)*100 for k in rr[0]['arms'][arm]};delta=[(r['arms'][arm]['m_vIoU']-r['arms']['F']['m_vIoU'])*100 for r in rr];groups.append(dict(dataset=ds,arm=arm,metrics_percent=mm,positive_count=sum(x>1e-8 for x in delta),negative_count=sum(x< -1e-8 for x in delta),neutral_count=sum(abs(x)<=1e-8 for x in delta),worst_delta_vIoU_pp=min(delta),severe_negative_definition='delta_vIoU<=-5pp fixed reporting bin not gate',severe_negative_count=sum(x<=-5 for x in delta)))
 write(B/'ROOT_POSTSEAL_REPORT.json',dict(status='pass',rows=rows,groups=groups,all_candidate_rows=candidates,GT_source_bindings=bindings,metric_count=len(errors),independent_metric_max_abs_error=max(errors),GT_read=True,GT_used_for_method_update=False,GT_oracle_diagnostic_update=True,observed_frame_subset_original_four_positions=True))
 for ds in ['hc2','vidstg']:print(ds,{g['arm']:round(g['metrics_percent']['m_vIoU'],6)for g in groups if g['dataset']==ds})
 print('metric max',max(errors))
if __name__=='__main__':main()
