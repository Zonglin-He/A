"""CPU-only controlled withdrawals of frozen correspondence evidence."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,collections,math,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(4)
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap,soft_iou,propagate
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
D=ROOT/'artifacts/stvg_correspondence_diagnosis32_v1';OLD=ROOT/'artifacts/stvg_soft_correspondence32_v1';PUB=ROOT/'results/stvg_soft_correspondence/2026-10-18/cached_diagnosis32'
ARMS=['real','near_shift','interpolation','far_shift','uniform_affinity','static_box_masks','self_match_static','interpolation_raster','hard_correspondence','centered_correspondence','half_grid_correspondence']
def mean(v):
 a=[float(x)for x in v if x is not None and np.isfinite(x)];return float(np.mean(a))if a else None

def prepare():
 p=read(OLD/'PLAN_LOCK.json');b=read(OLD/'GLOBAL_FEEDBACK_BARRIER.json');assert b['count']==32
 own=['scripts/diagnose_stvg_correspondence32_v1.py','protocols/stvg_correspondence_diagnosis32_v1.md','vg_tta/stvg_soft_correspondence_v1.py']
 put(D/'PLAN_LOCK.json',dict(status='locked_before_control_outcomes',count=32,records=p['records'],arms=ARMS,old_feedback_barrier_sha256=sha(OLD/'GLOBAL_FEEDBACK_BARRIER.json'),files=b['files'],new_model_calls=0,new_training=0,new_GT_feedback=0,retrospective_exposed=True,deadline_epoch=time.time()+3600));put(D/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f)for f in own},registries={f:sha(ROOT/f)for f in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']}))

def matching(maps,valid,positions,boxes):
 n,c,h,w=maps.shape;flat=maps.float().flatten(2).transpose(1,2);features=F.normalize(flat,dim=-1);valid_flat=valid.flatten(1);globalmean=flat[valid_flat].mean(0);centered=F.normalize(flat-globalmean,dim=-1);soft=[];hard=[];center=[];selfcopy=[];stats=[]
 for a,box in zip(positions,boxes):
  mask=cell_overlap(box[None],h,w)[0].float().flatten()*valid_flat[a];logits=(features@features[a].T)/.07;logits[...,~valid_flat[a]]=-torch.inf;A=torch.softmax(logits,-1);sm=(A@mask).reshape(n,h,w)*valid;soft.append(sm);hard.append(mask[A.argmax(-1)].reshape(n,h,w)*valid)
  cl=(centered@centered[a].T)/.07;cl[...,~valid_flat[a]]=-torch.inf;center.append((torch.softmax(cl,-1)@mask).reshape(n,h,w)*valid);selfcopy.append(sm[a][None].expand(n,h,w)*valid)
  # Full attention statistics need only saved scalar outputs.
  ent=-(A*A.clamp_min(1e-30).log()).sum(-1);den=math.log(int(valid_flat[a].sum()));own=A[a];diag=own.diagonal();ownmap=sm[a];inside=mask.reshape(h,w);ni=np.count_nonzero(inside.numpy()>0)
  stat=dict(anchor_position=int(a),normalized_entropy=float(ent[valid_flat].mean()/den),effective_reference_patches=float(ent[valid_flat].exp().mean()),top1_attention=float(A.max(-1).values[valid_flat].mean()),self_diagonal_mass=float(diag[valid_flat[a]].mean()),self_mask_MAE=float(abs(ownmap-inside)[valid[a]].mean()),self_mask_mass_ratio=float(ownmap.sum()/mask.sum())if mask.sum()>0 else None,mask_mean=float(mask[valid_flat[a]].mean()),feature_cosine_all_patch_mean=float((features@features[a].T)[valid_flat].mean()),inside_cells=int(ni),reference_cells=int(valid_flat[a].sum()),raster_self_iou=float(soft_iou(box[None],inside[None])[0]),reference_width_cells=float(box[2]*w),reference_height_cells=float(box[3]*h))
  stats.append(stat)
 return [torch.stack(x)if x else torch.empty(0,n,h,w)for x in [soft,hard,center,selfcopy]],stats

def controls():
 p=read(D/'PLAN_LOCK.json');files=[];started=time.time();records=p['records'];plans={d:read(ROOT/'artifacts/decota_paper_experiments_v1'/d/'PLAN.json')['rows']for d in ['hc2','vidstg']};review={r['source_alias']:r['slots']for r in read(ROOT/'artifacts/stvg_dino_error_attribution32_v1/VISUAL_REVIEW.json')['rows']}
 for i,rec in enumerate(records):
  assert time.time()<p['deadline_epoch'];from main_method.runtime import resource_guard
  resource_guard();alias=rec['alias'];existing=D/'controls'/(alias+'.pt')
  if existing.exists():
   done=load(existing);assert set(done['rewards'])==set(ARMS) and not done['GT_read'];files.append(read(existing.with_suffix('.receipt.json')));print('REUSE_CONTROL',alias,flush=True);continue
  z=load(OLD/'predictions'/(alias+'.pt'));assert sha(OLD/'predictions'/(alias+'.pt'))==p['files'][i]['sha256'];maps=z['maps'];valid=z['valid'];n,c,h,w=maps.shape;ap=z['admitted_positions'];planned=z['planned_positions'];ids=np.asarray(z['frame_ids']);fps=plans[rec['dataset']][rec['query_ordinal']]['input']['fps'];orig=load(ROOT/'artifacts/stvg_dino_rollback32_v1/predictions'/(alias+'.pt'));anchors=orig['expert']['anchors']['single4'];boxes=[torch.tensor(a['box'])for a in anchors];assert ap==[a['position']for a in anchors]
  unobs=[j for j in range(n)if j not in planned];farperm=list(range(n));shift=max(1,len(unobs)//2)
  for j,k in enumerate(unobs):farperm[k]=unobs[(j+shift)%len(unobs)]
  outputs,astats=matching(maps,valid,ap,boxes);soft,hard,cent,selfcopy=outputs
  cpu_gpu_error=float(abs(soft-z['soft_maps_per_anchor']).max())if ap else 0.
  assert np.isfinite(cpu_gpu_error) and cpu_gpu_error<5e-5 # cross-device FP32 probe only; never replaces old maps
  soft=z['soft_maps_per_anchor'] # all old arms/far shift retain sealed original bits
  actions=z['actions'];rewards=dict(real=z['rewards']['real'].clone(),near_shift=z['rewards']['shuffled'].clone(),interpolation=z['rewards']['interpolation'].clone());peranchor={};geometry_loo=[];real_loo=[];selfaudit=[]
  if ap:
   means=cell_overlap(torch.stack(boxes),h,w).float().flatten(1).mean(1);uniform=means[:,None,None,None]*valid[None];static=cell_overlap(torch.stack(boxes),h,w).float()[:,None].expand(-1,n,-1,-1)*valid
   for name,sm in [('far_shift',soft[:,farperm]),('uniform_affinity',uniform),('static_box_masks',static),('self_match_static',selfcopy),('hard_correspondence',hard),('centered_correspondence',cent)]:
    score=torch.stack([soft_iou(actions,x[:,None])for x in sm]).float();rewards[name]=score.mean(0);peranchor[name]=score
   peranchor['real']=z['per_anchor_rewards'];order=np.argsort(ap);aid=ids[np.asarray(ap)[order]];ab=np.array([b.numpy()for b in boxes])[order];interp=np.stack([np.interp(ids,aid,ab[:,j])for j in range(4)],1);raster=cell_overlap(torch.tensor(interp),h,w).float();rewards['interpolation_raster']=soft_iou(actions,raster[:,None]).float()
   small=F.adaptive_avg_pool2d(maps,(max(1,h//2),max(1,w//2)));vv=F.adaptive_max_pool2d(valid[:,None].float(),small.shape[-2:])[:,0].bool();coarse=propagate(small,vv,ap,boxes);rewards['half_grid_correspondence']=torch.stack([soft_iou(actions,x[:,None])for x in coarse]).float().mean(0)
   for j in range(len(ap)):
    keep=[k for k in range(len(ap))if k!=j];real_loo.append(peranchor['real'][keep].mean(0)if keep else torch.zeros_like(rewards['real']))
    if keep:
     oo=sorted(keep,key=lambda k:ap[k]);xx=np.stack([np.interp(ids,ids[np.array(ap)[oo]],np.array([boxes[k].numpy()for k in oo])[:,d])for d in range(4)],1);geometry_loo.append(aligned_iou(actions,torch.tensor(xx,dtype=actions.dtype)[:,None]))
    else:geometry_loo.append(torch.zeros_like(rewards['real']))
   # Merely increasing map resolution by nearest replication adds no information.
   repeated=soft[0].repeat_interleave(2,-2).repeat_interleave(2,-1);err=float(abs(soft_iou(actions,soft[0,:,None])-soft_iou(actions,repeated[:,None])).max());assert err<1e-12;selfaudit.append(err)
  else:
   for name in ARMS:
    if name not in rewards:rewards[name]=torch.zeros_like(z['rewards']['real'])
  assert set(rewards)==set(ARMS);weights={a:torch.softmax(r/z['config']['tau'],-1)for a,r in rewards.items()};gap={}
  for name,perm in [('near_shift',z['permutation']),('far_shift',farperm)]:
   dt=abs(ids[np.asarray(perm)[unobs]]-ids[unobs]);gap[name]=dict(seconds_quantiles=np.quantile(dt/fps,[0,.25,.5,.75,.95,1]).tolist(),span_fraction_quantiles=np.quantile(dt/(ids[-1]+1-ids[0]),[0,.25,.5,.75,.95,1]).tolist(),fraction_shift_ge25percent=float(np.mean(dt>=.25*(ids[-1]+1-ids[0]))))
  # Compare actual/local versus shuffled visual evidence without annotation access.
  v=F.normalize(maps.flatten(2).transpose(1,2),dim=-1);gapsim={name:float((v[unobs]*v[np.asarray(perm)[unobs]]).sum(-1).mean())for name,perm in [('near_shift',z['permutation']),('far_shift',farperm)]};mapstats={}
  for name in ARMS:
   a=rewards[name];mapstats[name]=dict(candidate_reward_std=float(a.std(-1,unbiased=False)[unobs].mean()),ESS=float((1/weights[name].square().sum(-1))[unobs].mean()),flat_frames=int((a.max(-1).values==a.min(-1).values).sum()),reward_mean=float(a[unobs].mean()))
  codes=[review[alias][planned.index(pos)]for pos in ap]
  files.append(save(D/'controls'/(alias+'.pt'),dict(alias=alias,dataset=rec['dataset'],query_ordinal=rec['query_ordinal'],frame_ids=z['frame_ids'],rewards=rewards,weights=weights,per_anchor_rewards=peranchor,real_leave_one_out=real_loo,interpolation_leave_one_out=geometry_loo,anchor_codes=codes,planned_positions=planned,admitted_positions=ap,anchor_stats=astats,grid=[h,w],gap=gap,aligned_feature_cosine=gapsim,reward_stats=mapstats,upsample_invariance_max_error=max(selfaudit)if selfaudit else 0.,cpu_vs_original_CUDA_max_map_error=cpu_gpu_error,GT_read=False)))
  print('CONTROL',alias,i+1,'CPU_seconds',round(time.time()-started,2),flush=True)
 put(D/'CONTROL_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False,new_models=0,GPU_seconds=0,CPU_wall_seconds=time.time()-started))
if __name__=='__main__':globals()[sys.argv[1]]()
