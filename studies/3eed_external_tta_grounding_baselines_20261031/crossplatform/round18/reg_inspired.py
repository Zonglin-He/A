"""Reg-TTA3D-inspired grounding port. Not an official detector reproduction.
Preserves noise-stability supervision, class-wise high-confidence size guidance,
regression-only students and class-balanced EMA teachers. Explicit substitutions:
query positional box perturbation (no RoI head/yaw), token category proxies,
and a stated IoU/score-change stability gate due to ambiguous printed Eq2.
"""
import re,math
import numpy as np
import torch
import torch.nn.functional as F
from algorithms import scores,rawboxes,detach,grad_summary
from geometry import groups,iou_matrix,match_boxes

FAMILIES={'car':r'car|sedan|suv|taxi|van|jeep','truck':r'truck|trailer','bus':r'bus','person':r'pedestrian|person|man|woman|people|child|boy|girl|walker|worker','cyclist':r'cyclist|biker|bicycle|bike|motorcycle','vehicle':r'vehicle'}
REG_NAMES=tuple('prediction_heads.5.'+head+'.net.8.'+s for head in ('center_residual_head','size_pred_head') for s in ('weight','bias'))
def proxy_classes(model,out,text):
 tok=model.tokenizer([text],return_offsets_mapping=True,return_tensors='pt');offset=tok['offset_mapping'][0].numpy();prob=(out['last_proj_queries']@out['proj_tokens'].transpose(-1,-2)/.07).softmax(-1)[0];mass=[];names=[]
 for name,pattern in FAMILIES.items():
  m=np.zeros(len(offset),bool)
  for found in re.finditer(r'\b(?:'+pattern+r')\b',text,re.I):m|=(offset[:,1]>found.start())&(offset[:,0]<found.end())
  if m.any():mass.append(prob[:,torch.as_tensor(m,device=prob.device)].sum(-1));names.append(name)
 if not mass:return np.full(256,-1),np.zeros(256),names
 p=torch.stack(mass,-1).detach().cpu().numpy();cls=p.argmax(-1);confidence=p.max(-1);cls[confidence<.05]=-1;return cls,confidence,names
def head_snapshot(model):return {n:p.detach().clone() for n,p in model.named_parameters() if n in REG_NAMES}
def load_head(model,state):
 with torch.no_grad():
  for n,p in model.named_parameters():
   if n in state:p.copy_(state[n])
def representative_indices(boxes,confidence,classes):
 reps=[]
 for c in sorted(set(classes[classes>=0])):
  ix=np.flatnonzero((classes==c)&(boxes[:,3:]>0).all(1));g,r=groups(boxes[ix],confidence[ix]);reps.extend(ix[r].tolist())
 return np.array(sorted(reps,key=lambda q:(-confidence[q],q))[:64],int)
def run_reg(model,pc,text,mask,seed,teacher=None,student=None,optimizer_state=None):
 model.eval().requires_grad_(False);source_head=head_snapshot(model)
 if teacher is not None:load_head(model,teacher)
 teacher_before=head_snapshot(model)
 with torch.no_grad():original=model({'point_clouds':pc,'text':[text]});pre=detach(original,mask)
 cls,confidence,classnames=proxy_classes(model,original,text);base=rawboxes(original)[0].detach().cpu().numpy();reps=representative_indices(base,confidence,cls)
 # Size perturbation occurs at fixed upstream slots; point sampling unchanged.
 rng=np.random.default_rng(seed);scale=torch.as_tensor(rng.uniform(.9,1.1,(1,256,3)),dtype=pc.dtype,device=pc.device)
 def perturb(module,args):
  aa=list(args);pos=aa[3].clone();pos[...,3:]*=scale;aa[3]=pos;return tuple(aa)
 hook=model.decoder[-1].register_forward_pre_hook(perturb)
 with torch.no_grad():noisy=model({'point_clouds':pc,'text':[text]});noise=detach(noisy,mask)
 hook.remove();nc,nconf,_=proxy_classes(model,noisy,text);noisyb=noise['boxes'];pair_iou=np.diag(iou_matrix(base,noisyb));change=np.abs(confidence-nconf);literal=1-change/np.maximum(pair_iou,1e-6)
 stable=(pair_iou>=.5)&(change<=.1)&(cls==nc);npg=[];targets=[];counts=np.zeros(len(FAMILIES),float)
 for c in sorted(set(cls[reps])) if len(reps) else []:
  if c<0:continue
  ci=[q for q in reps if cls[q]==c];threshold=np.quantile([max(confidence[q],nconf[q]) for q in ci],.8)
  for q in ci:
   if stable[q] and max(confidence[q],nconf[q])>=threshold:
    npg.append(q);targets.append(noisyb[q] if nconf[q]>confidence[q] else base[q]);counts[list(FAMILIES).index(classnames[c])]+=1
 # Merge then NMS, again, since the winning noisy proposals may newly overlap.
 retained=[]
 for j,q in sorted(enumerate(npg),key=lambda item:(-max(confidence[item[1]],nconf[item[1]]),item[1])):
  if any(cls[q]==cls[npg[k]] and iou_matrix(np.array(targets[j])[None],np.array(targets[k])[None])[0,0]>=.5 for k in retained):continue
  retained.append(j)
 raw_npg=len(npg);npg=[npg[j] for j in retained];targets=[targets[j] for j in retained];counts[:]=0
 for q in npg:counts[list(FAMILIES).index(classnames[cls[q]])]+=1
 diag={'npg_candidates':len(reps),'npg_before_second_NMS':raw_npg,'npg_accepted':len(npg),'noise_stable_before_confidence':int(stable[reps].sum()) if len(reps) else 0,'printed_literal_tau1_5_rejects':int((literal[reps]>1.5).sum()) if len(reps) else 0,'classes_present_proxy':classnames,'steps':0,'teacher_steps':0,'CRR_low_boxes':0,'source_confidence_is_grounding_token_proxy':True,'not_official_RoI_yaw':True}
 if student is not None:load_head(model,student)
 else:load_head(model,teacher_before)
 selected=[]
 for n,p in model.named_parameters():
  if n in REG_NAMES:p.requires_grad_(True);selected.append((n,p))
 optimizer=torch.optim.Adam([p for n,p in selected],lr=.001)
 if optimizer_state is not None:optimizer.load_state_dict(optimizer_state)
 optimizer.zero_grad();jitter=torch.as_tensor(rng.normal(0,.01,pc.shape).clip(-.03,.03),dtype=pc.dtype,device=pc.device);aug=pc+jitter
 output=model({'point_clouds':aug,'text':[text]});studentbox=rawboxes(output)[0];student_cls,student_conf,_=proxy_classes(model,output,text);sb=studentbox.detach().cpu().numpy();losses=[]
 if npg:
  y=np.array(targets,np.float32);a,b,md=match_boxes(y,sb);diag['pseudo_geometry_match']=md
  if len(a):losses.append(F.smooth_l1_loss(studentbox[b],torch.as_tensor(y[a],device=pc.device),beta=.1))
 rr=representative_indices(sb,student_conf,student_cls);crr=[];direct_size=base[int(pre['scores'].argmax()),3:].copy();q0=int(pre['scores'].argmax());direct_box=base[q0].copy()
 if q0 in npg:direct_box=np.array(targets[npg.index(q0)],copy=True);direct_size=direct_box[3:].copy()
 for c in sorted(set(student_cls[rr])) if len(rr) else []:
  if c<0:continue
  ci=sorted([q for q in rr if student_cls[q]==c],key=lambda q:(-student_conf[q],q));k=max(1,math.ceil(.2*len(ci)));hi=ci[:k];lo=ci[k:]
  if not lo:continue
  mean=studentbox[hi,3:].detach().mean(0);diff=studentbox[lo,3:]-mean;term=torch.where(diff.abs()>=.1,diff.square(),torch.zeros_like(diff)).mean();crr.append(term);diag['CRR_low_boxes']+=len(lo)
  original_class=[q for q in reps if cls[q]==c]
  if cls[q0]==c and original_class and confidence[q0]<np.quantile(confidence[original_class],.8):direct_size=mean.detach().cpu().numpy()
 if crr:losses.append(torch.stack(crr).mean())
 if losses:
  loss=torch.stack(losses).sum();assert torch.isfinite(loss);loss.backward();diag.update(grad_summary(selected));optimizer.step();diag.update(steps=1,loss=float(loss.detach()))
 else:diag.update(selected_parameters=1734,gradient_parameters=0,gradient_L2=0.)
 new_student=head_snapshot(model)
 with torch.no_grad():student_post=detach(model({'point_clouds':pc,'text':[text]}),mask)
 alpha=.999
 if counts.sum():
  frequency=counts/counts.sum();var=float(np.var(frequency));maxvar=(len(counts)-1)/(len(counts)**2);alpha=.99+.009*min(var/maxvar,1.)
  teacher_after={n:teacher_before[n]*alpha+new_student[n]*(1-alpha) for n in teacher_before};diag['teacher_steps']=int(diag['steps']>0)
 else:teacher_after=teacher_before
 load_head(model,teacher_after)
 with torch.no_grad():teacher_post=detach(model({'point_clouds':pc,'text':[text]}),mask)
 diag.update(EMA_alpha=alpha,class_counts=counts.tolist(),student_delta_L2=float(torch.sqrt(sum((new_student[n]-teacher_before[n]).square().sum() for n in new_student))),teacher_delta_L2=float(torch.sqrt(sum((teacher_after[n]-teacher_before[n]).square().sum() for n in teacher_after))))
 direct_box[3:]=direct_size
 return pre,teacher_post,student_post,direct_box,diag,selected,teacher_after,new_student,optimizer.state_dict()
