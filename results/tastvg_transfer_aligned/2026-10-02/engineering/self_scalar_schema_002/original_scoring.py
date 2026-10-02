"""CPU-only saved-state transfer metrics and independent selection/state audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tastvg_transfer_aligned_common_v1 import *
def summarize(rows):
 from scripts.score_tastvg_best_quick_v1 import source_summary
 fields=['pre_v','post_v','delta_v','pre_s','post_s','delta_s','delta_free_v','delta_free_t','delta_common_v','cosine','lag']
 out={}
 for group in ['corruption','clean']:
  rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption')];out[group]={}
  for role in ['self','next','near','far']:
   out[group][role]={}
   for arm in ['A','R']:
    take=[r for r in rr if r['role']==role and r['arm']==arm];z=source_summary(take,fields)
    z['coverage']=dict(logical_pairs=len(take),actual_writes=sum(r['updated'] for r in take),noops=sum(not r['updated'] for r in take),donor_sources=len({r['donor_source_id'] for r in take}),target_sources=len({r['target_source_id'] for r in take}),scheduled_targets=sum(r['target_scheduled'] for r in take),positive=int(sum(r['delta_v']>1e-12 for r in take)),negative=int(sum(r['delta_v']< -1e-12 for r in take)),zero=int(sum(abs(r['delta_v'])<=1e-12 for r in take)),harm_gt5pp=int(sum(r['delta_v']<-.05 for r in take)),gross_gain_pp=100*float(np.mean([max(r['delta_v'],0) for r in take])),gross_loss_pp=100*float(np.mean([max(-r['delta_v'],0) for r in take])))
    z['target_clustered_sensitivity']=source_summary([{**r,'source_id':r['target_source_id']} for r in take],['delta_v','delta_s']);actual=[r for r in take if r['updated']];z['actual_write_only']=source_summary(actual,['delta_v','delta_s'])
    out[group][role][arm]=z
   aa={(r['condition'],r['order'],r['donor_arrival']):r for r in rr if r['role']==role and r['arm']=='A'};paired=[]
   for r in rr:
    if r['role']==role and r['arm']=='R':
     a=aa[r['condition'],r['order'],r['donor_arrival']];assert r['target_source_id']==a['target_source_id'];paired.append({**r,**{'delta_R_A_'+m:r['delta_'+m]-a['delta_'+m] for m in ['v','s','common_v','free_v']}})
   out[group][role]['R_minus_A']=source_summary(paired,['delta_R_A_v','delta_R_A_s','delta_R_A_common_v','delta_R_A_free_v'])
  out[group]['near_minus_far']={}
  for arm in ['A','R']:
   far={(r['condition'],r['order'],r['donor_arrival']):r for r in rr if r['role']=='far' and r['arm']==arm};z=[]
   for r in rr:
    if r['role']=='near' and r['arm']==arm:
     f=far[r['condition'],r['order'],r['donor_arrival']];z.append({**r,'delta_near_far_v':r['delta_v']-f['delta_v'],'delta_near_far_s':r['delta_s']-f['delta_s'],'delta_near_far_cosine':r['cosine']-f['cosine']})
   out[group]['near_minus_far'][arm]=source_summary(z,['delta_near_far_v','delta_near_far_s','delta_near_far_cosine'])
 return out
def run():
 import torch
 torch.set_num_threads(2)
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric
 from vg_tta.tastvg_paper_readouts_v1 import dense_official_metrics
 from scripts.diagnose_tastvg_pipeline_cpu_v1 import evaluator
 verify('TRANSFER');b=read(BASE/'transfer/PREDICTION_BARRIER.json');assert b['pairs']==768 and b['write_payloads']==192 and not b['GT_read']
 for f,h in b['files'].items():assert sha(BASE/f)==h;receipt((BASE/f).with_suffix('.pt'))
 t=read(BASE/'transfer/TARGET_LOCK.json');assert sha(BASE/'transfer/TARGET_LOCK.json')==b['target_lock_sha256'];z=load(BASE/'transfer/TEXT_CONTEXT.pt');np.testing.assert_allclose(z['vectors'].numpy()@z['vectors'].numpy().T,z['cosine'],atol=1e-14,rtol=0)
 p=read(OLD/'hc2/PLAN.json');checks=collections.Counter();lookup={};parents=z['parents'];sim=z['cosine'];maxerr=0
 for x in t['rows']:
  seq=p['splits']['search']['orders'][x['order']];i=x['donor_arrival'];f=list(range(i+1,32));ix=parents.index(seq[i]);expected={'self':i,'next':next(j for j in f if j%4!=0),'near':min(f,key=lambda j:(-float(sim[ix,parents.index(seq[j])]),j)),'far':min(f,key=lambda j:(float(sim[ix,parents.index(seq[j])]),j))}
  for role,j in expected.items():assert x['roles'][role]['arrival']==j and x['roles'][role]['parent']==seq[j];checks['target_selection']+=1
  lookup[x['order'],i]=x
 lab=POOL/'hc2/GT_LABELS_search.json';write(BASE/'transfer/SCORING_LOCK.json',dict(script_sha256=sha(Path(__file__)),labels_sha256=sha(lab),barrier_sha256=sha(BASE/'transfer/PREDICTION_BARRIER.json'),GT_input_to_model=False,time=time.time()))
 write(BASE/'transfer/GT_EXPOSURE.json',dict(after_prediction_barrier=True,historically_exposed=True,no_GT_selection=True,time=time.time()));gt=read(lab);ids={s:i for i,s in enumerate(sorted({p['rows'][j]['source'] for j in parents}))};metric=HC2DenseMetric();rows=[]
 def metrics(boxes,parent,idx):
  nonlocal maxerr
  row=p['rows'][parent];g=gt[str(parent)];truth={int(k):v for k,v in g['truth'].items()};span=g['span'];pix=np.maximum(xyxy(boxes,row['input']['width'],row['input']['height']),0);interval=[row['frame_ids'][idx[0]],row['frame_ids'][idx[1]]+1];out=metric(pix,row['frame_ids'],interval,truth,span);ind=dense_official_metrics(pix,row['frame_ids'],interval,truth,span);small=evaluator(boxes,row,truth,span,True)(idx)
  for key in ['m_vIoU','m_tIoU']:
   err=abs(out[key]-ind[key]);assert err<1e-10;maxerr=max(maxerr,err);checks['dense_v_t_independent']+=1
  assert abs(out['m_vIoU']-small['v'])<1e-10
  dense_ids=np.asarray(sorted(truth));pred=np.column_stack([np.interp(dense_ids,row['frame_ids'],pix[:,k]) for k in range(4)]);q=np.asarray([truth[int(fid)] for fid in dense_ids]);inter=np.maximum(np.minimum(pred[:,2:],q[:,2:])-np.maximum(pred[:,:2],q[:,:2]),0).prod(1);union=np.maximum(pred[:,2:]-pred[:,:2],0).prod(1)+np.maximum(q[:,2:]-q[:,:2],0).prod(1)-inter;iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0);iou[(dense_ids<row['frame_ids'][0])|(dense_ids>row['frame_ids'][-1])]=0;err=abs(out['sIoU_dense_GT']-float(iou.mean()));assert err<1e-10;maxerr=max(maxerr,err);checks['dense_s_independent']+=1
  return dict(v=out['m_vIoU'],s=out['sIoU_dense_GT'],t=out['m_tIoU'])
 for file in sorted((BASE/'transfer/pairs').rglob('*.pt')):
  x=load(file);oldf=OLD/'hc2'/x['arm']/'online'/x['condition']/x['order']/f'{x["donor_arrival"]:05}.pt';old=load(oldf);assert sha(oldf)==x['donor_payload_sha256'];assert state_hash(old['pre_state'])==x['pre_sha'] and state_hash(old['post_state'])==x['post_sha'];assert x['roles']==lookup[x['order'],x['donor_arrival']]['roles'];assert not x['GT_read'] and x['parameter_updates']==x['new_experts']==0;checks['state_pair_binding']+=1
  mm={}
  for parent,y in x['predictions'].items():
   parent=int(parent);cr=read(POOL/'hc2/capture'/x['condition']/f'{parent:05}.json');assert cr['sha256']==x['target_receipts'][parent]['capture_sha256'];assert cr['pixel_sha256']==x['target_receipts'][parent]['pixel_sha256'];checks['target_capture_binding']+=1
   mm[parent]={phase:metrics(y[phase]['boxes'],parent,x['frozen_intervals'][parent]) for phase in ['pre','post']};mm[parent]['free']={phase:metrics(y[phase]['boxes'],parent,y[phase]['indices']) for phase in ['pre','post']}
   if parent==x['donor_parent']:
    for phase,saved in [('pre','slow'),('post','post_prediction')]:assert torch.equal(y[phase]['boxes'],old[saved]['boxes']) and y[phase]['indices']==old[saved]['indices'];checks['self_bitwise']+=1
    mm[parent]['self']={phase:metrics(y[phase]['boxes'],parent,x['self_interval']) for phase in ['pre','post']}
   if not x['updated']:assert torch.equal(y['pre']['boxes'],y['post']['boxes']);checks['no_op_predictions']+=int(not x['updated'])
  for role,target in x['roles'].items():
   parent=target['parent'];m=mm[parent];readout=m['self'] if role=='self' else m;a=m['pre'];z0=m['post'];r=dict(arm=x['arm'],condition=x['condition'],order=x['order'],donor_arrival=x['donor_arrival'],target_arrival=target['arrival'],donor_source_id=ids[p['rows'][x['donor_parent']]['source']],target_source_id=ids[p['rows'][parent]['source']],source_id=ids[p['rows'][x['donor_parent']]['source']],role=role,updated=x['updated'],cosine=target['cosine'],lag=target['arrival']-x['donor_arrival'],target_scheduled=target['scheduled_in_original'],role_aliases=[k for k,v in x['roles'].items() if v['parent']==parent])
   for name,key in [('v','v'),('s','s')]:r.update({f'pre_{name}':readout['pre'][key],f'post_{name}':readout['post'][key],f'delta_{name}':readout['post'][key]-readout['pre'][key]})
   r.update(delta_common_v=z0['v']-a['v'],delta_free_v=m['free']['post']['v']-m['free']['pre']['v'],delta_free_t=m['free']['post']['t']-m['free']['pre']['t']);rows.append(r)
 assert len(rows)==768
 write(PUBLIC/'hc2/TRANSFER_ROWS.json',rows);write(PUBLIC/'hc2/TRANSFER_SUMMARY.json',summarize(rows));write(PUBLIC/'hc2/TRANSFER_CASES.json',{role:dict(positive=sorted([r for r in rows if r['role']==role and r['condition']!='clean'],key=lambda r:-r['delta_v'])[:5],negative=sorted([r for r in rows if r['role']==role and r['condition']!='clean'],key=lambda r:r['delta_v'])[:5]) for role in ['self','next','near','far']})
 # Matched self readback to previous local-update rows without changing labels.
 oldrows=read(ROOT/'results/tastvg_routed_online_token/2026-10-02/hc2/LOCAL_UPDATE_PAIRED_ROWS.json');oldmap={(r['condition'],r['order'],r['arrival']):r for r in oldrows}
 for r in rows:
  if r['role']=='self':
   saved=oldmap[r['condition'],r['order'],r['donor_arrival']];key=r['arm']+'_net';assert abs(r['delta_v']-saved[key])<1e-10;checks['old_local_scalar_parity']+=1
 audit=dict(status='pass',checks=dict(checks),max_metric_error=maxerr,rows=len(rows),new_GT_after_barrier=True,GPU_initialized=torch.cuda.is_initialized(),target_GT_free=True,time=time.time());assert not torch.cuda.is_initialized();write(BASE/'transfer/ROOT_READBACK.json',audit);write(PUBLIC/'TRANSFER_ROOT_READBACK.json',audit);write(PUBLIC/'TRANSFER_RESOURCES.json',read(BASE/'transfer/RESOURCES.json'));status(BASE/'transfer/STATUS.json',dict(status='scored_pending_publication',done=192,total=192,time=time.time()));print('TRANSFER scored',audit)
if __name__=='__main__':run()
