"""Postseal diagnostic update direction and geometric pseudo-reference quality."""
import sys,statistics,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from scripts.stvg_temporal_caption_query_v1 import ROOT,BASE as OLD,sha,read,write,roster,catalog
from scripts.review_stvg_temporal_caption_adaptation_v1 import structures
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal,_metric_functions
from scripts.stvg_sparse_r1_student_v1 import _target_pair
from vg_tta.tastvg_oracle_event5_v1 import DenseTube
B=ROOT/'artifacts/stvg_support_aware_temporal_spatial_update_v1';torch.set_num_threads(2)
def cos(a,b):
 a,b=a.reshape(-1).double(),b.reshape(-1).double();na,nb=float(a.norm()),float(b.norm());return float(a@b/(a.norm()*b.norm())) if min(na,nb)>1e-12 else None
for n in ['TEMPORAL_GLOBAL_BARRIER.json','STUDENT_GLOBAL_BARRIER.json']:
 z=read(B/n);assert z['status']=='complete'and z['GT_read']is False
 for f in z['files']:assert sha(ROOT/f['path'])==f['sha256']
cases={r['case_id']:catalog(r)[0]for r in roster()};truths,binding=_truths_after_seal(cases);jobs=read(OLD/'CAPTION_JOB_LOCK.json')['records'];refs={r['case_id']:r for r in read(OLD/'SPATIAL_REFERENCE_LOCK.json')['records']};rows=[]
def cx(box):return np.concatenate(((box[:,:2]+box[:,2:])/2,box[:,2:]-box[:,:2]),-1)
for r in roster():
 cid=r['case_id'];case=cases[cid];old=torch.load(OLD/'student'/f'{cid}.pt',map_location='cpu',weights_only=False);t=torch.load(B/'temporal'/f'{cid}.pt',map_location='cpu',weights_only=False);s=torch.load(B/'student'/f'{cid}.pt',map_location='cpu',weights_only=False);g=truths[cid];local=[j for j in jobs if j['case_id']==cid];ev,hs,b,phi=structures(case,old['temporal_hidden'],[j['candidate']for j in local]);theta=torch.zeros(256,2,dtype=torch.float64,requires_grad=True)
 loss=theta.sum()*0
 for h,e,rr in zip(hs,ev['offsets'],case['records']):
  ij=e['ij'].cpu();start,end,overlap=_target_pair(rr['frame_ids'],g['span']);assert overlap>0;at=int(torch.nonzero((ij[0]==start)&(ij[1]==end))[0,0]);scores=e['score'].double().cpu()+h[ij[0]]@theta[:,0]+h[ij[1]]@theta[:,1];loss-=scores.log_softmax(0)[at]/2
 grad=torch.autograd.grad(loss,theta)[0];fit=old['temporal_fit'];valid=fit['valid_candidate_indices'];p0=b[valid].softmax(0);q=fit['target'];gqa=(phi[valid]*(p0-q)[:,None,None]).sum(0);updates={}
 for mode,v in t['results'].items():
  delta=v['path'][1]['theta']-v['path'][0]['theta'];updates[mode]=dict(first_delta_vs_negative_GT_NLL_cosine=cos(delta,-grad),first_delta_dot_negative_GT_NLL=float((delta*(-grad)).sum()),first_delta_norm=float(delta.norm()),final_vs_negative_GT_NLL_cosine=cos(v['path'][-1]['theta'],-grad))
 meta=dict(frame_ids=s['frame_ids'],input=dict(width=case['width'],height=case['height']));mask=np.isin(sorted(g['truth']),[s['frame_ids'][p]for p in s['reference_binding']['observed_positions']]);source=DenseTube(cx(s['arms']['S0_source']['boxes']),meta,g['truth'],g['span'],clip=r['dataset']=='hc2');sp={}
 target=np.stack([a['box_cxcywh_normalized'].numpy()for a in s['supervision_anchors']]);pos=s['reference_binding']['observed_positions']
 def iou(a,b):
  a,b=np.concatenate((a[...,:2]-a[...,2:]/2,a[...,:2]+a[...,2:]/2),-1),np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1);inter=np.maximum(np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2]),0).prod(-1);area=np.maximum(a[...,2:]-a[...,:2],0).prod(-1)+np.maximum(b[...,2:]-b[...,:2],0).prod(-1);return inter/np.maximum(area-inter,1e-7)
 for arm,v in s['arms'].items():
  tube=DenseTube(cx(v['boxes']),meta,g['truth'],g['span'],clip=r['dataset']=='hc2');delta=tube.iou-source.iou;sp[arm]=dict(supervised_GT_frame_count=int(mask.sum()),unobserved_GT_frame_count=int((~mask).sum()),supervised_sIoU_delta_pp=float(delta[mask].mean()*100)if mask.any()else None,unobserved_sIoU_delta_pp=float(delta[~mask].mean()*100)if (~mask).any()else None,pseudo_reference_mean_IoU_at_observations=float(iou(cx(v['boxes'])[pos],target).mean()),dense_full_sIoU_percent=float(tube.iou.mean()*100))
 reference=np.array(refs[cid]['candidate']['boxes_xyxy_normalized']);refTube=DenseTube(cx(reference),meta,g['truth'],g['span'],clip=r['dataset']=='hc2')
 rows.append(dict(case_id=cid,dataset=r['dataset'],QA_gradient_vs_GT_NLL_gradient_cosine=cos(gqa,grad),GT_gradient_norm=float(grad.norm()),QA_gradient_norm=float(gqa.norm()),updates=updates,spatial=sp,reference_selected_source_kind=refs[cid]['candidate']['opaque_geometry_binding']['source_candidate_id'],reference_dense_GT_sIoU_percent=float(refTube.iou.mean()*100),reference_is_student_pseudoreference=True,reference_is_independent_teacher_geometry=False))
write(B/'ROOT_DIRECTION_AND_REFERENCE_DIAGNOSTICS.json',dict(status='pass',count=8,rows=rows,GT_read=True,GT_gradient_used_for_update=False,GT_used_to_choose_step_or_configuration=False,scope='offline same512 gradient and all dense GT frame diagnosis'))
for ds in ['hc2','vidstg']:
 rr=[r for r in rows if r['dataset']==ds];print(ds,'QAcos',statistics.mean(r['QA_gradient_vs_GT_NLL_gradient_cosine']for r in rr));print('first update descent cos',{m:statistics.mean(r['updates'][m]['first_delta_vs_negative_GT_NLL_cosine']for r in rr)for m in rr[0]['updates']});print('spatial unobserved',{m:statistics.mean(r['spatial'][m]['unobserved_sIoU_delta_pp']for r in rr)for m in rr[0]['spatial']})
