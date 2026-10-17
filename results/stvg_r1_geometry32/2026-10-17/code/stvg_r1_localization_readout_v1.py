"""Independent R1 rectangular-grid adaptation of pinned LocalizationHeads single-sample recipe."""
import math
import numpy as np
def independent_row(module,h,pos,last):
 import torch
 from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb,repeat_kv
 B,T,_=h.shape;q=module.q_proj(h).view(B,T,-1,module.head_dim).transpose(1,2);k=module.k_proj(h).view(B,T,-1,module.head_dim).transpose(1,2);q,k=apply_multimodal_rotary_pos_emb(q,k,*pos,module.rope_scaling['mrope_section']);k=repeat_kv(k,module.num_key_value_groups);scores=q[:,:,last:last+1]@k.transpose(2,3)/math.sqrt(module.head_dim);return scores.softmax(-1,dtype=torch.float32).to(q.dtype)[0,:,0].detach()
def chord(v):
 y=np.sort(np.asarray(v,dtype=np.float64));x=np.arange(len(y));a=np.array([0,y[0]]);d=np.array([len(y)-1,y[-1]])-a;u=d/np.linalg.norm(d);z=np.column_stack((x,y))-a;distance=np.linalg.norm(z-(z@u)[:,None]*u,axis=1);return float(y[np.argmax(distance)])
def readout(attn,H,W,pixel_width,pixel_height):
 import torch
 from scipy.ndimage import gaussian_filter,label
 a=attn.detach().cpu().float().numpy();L,N,V=a.shape;assert V==H*W;mass=a.sum(-1);threshold=chord(mass.ravel());ranked=[];allrows=[]
 for l in range(L):
  for h in range(N):
   m=a[l,h].reshape(H,W);bottom=bool((m[-1]>.05).any());entropy=float('inf');components=0
   if mass[l,h]>=threshold:
    z=np.maximum(m-2*m.mean(),0);labs,components=label((z>.001).astype(np.int32),structure=np.ones((3,3)));total=float(z.sum())
    if total>0:
     probs=[float(z[labs==i].sum())/total for i in range(1,components+1)];entropy=-sum(p*math.log(p) for p in probs if p>0)
   row=dict(layer=l,head=h,attention_sum=float(mass[l,h]),entropy=entropy,bottom_row_focus=bottom,components=int(components));allrows.append(row)
   if l>1 and not bottom and math.isfinite(entropy) and mass[l,h]>=threshold:ranked.append(row)
 fallback=not ranked
 if fallback:ranked=sorted([r for r in allrows if not r['bottom_row_focus']],key=lambda r:-r['attention_sum'])[:1]
 ranked.sort(key=lambda r:r['entropy']);selected=ranked[:5];M=np.zeros((H,W),np.float32)
 for r in selected:M+=gaussian_filter(a[r['layer'],r['head']].reshape(H,W),sigma=1.)
 mask=M>M.mean();yy,xx=np.where(mask);box=None
 if len(xx):
  x0,y0=int(xx.min()*pixel_width/W),int(yy.min()*pixel_height/H);x1=min(pixel_width-1,int((xx.max()+1)*pixel_width/W-1));y1=min(pixel_height-1,int((yy.max()+1)*pixel_height/H-1));a0,b0,a1,b1=x0/pixel_width,y0/pixel_height,(x1+1)/pixel_width,(y1+1)/pixel_height
  if a1>a0 and b1>b0:box=[(a0+a1)/2,(b0+b1)/2,a1-a0,b1-b0]
 clean=[{k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in r.items()} for r in selected]
 return dict(status='valid' if box else 'empty_mask_abstain',box=box,selected_heads=clean,criterion_threshold=threshold,fallback=fallback,anchor_attention_mass=float(a.sum(-1).mean()),map=M)
