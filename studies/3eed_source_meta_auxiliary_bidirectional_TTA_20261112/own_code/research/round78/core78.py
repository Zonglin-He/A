"""Exact one-step functional SGD shared by meta-training and unlabeled inference."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round76'))
from common76 import *
from model76 import *
from torch.func import functional_call
H78=ROOT/'research/round78';B78=ROOT/'crossplatform/round78'
class Fusion78(Fusion):
 def forward(self,kind,base,z,d,pm=None,im=None,wrong=False,point_mask=None,image_mask=None):
  if kind=='main':return self.main(base,z,d,'bidirectional',point_mask=point_mask,image_mask=image_mask)
  assert kind=='aux';return torch.stack([self.auxiliary(z,d,'bidirectional',p,i,wrong)[0] for p,i in zip(pm,im)]).mean()
def auxiliary78(m,z,d,pm,im,wrong=False):return m('aux',None,z,d,pm,im,wrong)
def inner78(m,z,d,pm,im,create_graph=False,wrong=False,detach_gradient=False):
 L=auxiliary78(m,z,d,pm,im,wrong);nn=m.tta_parameters();gg=torch.autograd.grad(L,[v for _,v in nn],create_graph=create_graph,retain_graph=create_graph,allow_unused=True);gg=[torch.zeros_like(v) if g is None else g for (_,v),g in zip(nn,gg)]
 if detach_gradient:gg=[g.detach() for g in gg]
 norm=torch.sqrt(sum(g.square().sum() for g in gg)+1e-24);coefficient=(1/(norm+1e-6)).clamp_max(1.);fast={n:v for n,v in m.named_parameters()}
 for (n,v),g in zip(nn,gg):fast[n]=v-.5*coefficient*g
 return fast,L,norm
def main78(m,base,z,d,fast=None,point_mask=None,image_mask=None):
 args=('main',base,z,d)
 if fast is None:return m(*args,point_mask=point_mask,image_mask=image_mask)
 return functional_call(m,fast,args,{'point_mask':point_mask,'image_mask':image_mask},strict=True)
def ground78(pp,y):
 if y.max()<=0:return pp.sum()*0
 t=torch.softmax(y/.1,0);return -(t*pp.clamp_min(1e-300).log()).sum()
def maskpair(r,d,repeat=(76001,76002)):
 pairs=[masks(r,d,x) for x in repeat];return [v[0] for v in pairs],[v[1] for v in pairs]
def prepare_item(r):
 z0=prefix(r);return z0,tensor_prefix(z0),tensor_data(load(casepath(r)),r)
def source_y(r):return torch.tensor(load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'],dtype=torch.float64,device='cuda')
def enable_tta(m):
 m.requires_grad_(False)
 for _,p in m.tta_parameters():p.requires_grad_(True)
def evaluate_source(m,base,rr,savepath):
 rows=[];scores=[];values=[]
 for r in rr:
  _,z,d=prepare_item(r);y=source_y(r);pm,im=maskpair(r,d);enable_tta(m)
  with torch.no_grad():p0,q0,_=main78(m,base,z,d)
  fast,a,gn=inner78(m,z,d,pm,im)
  with torch.no_grad():
   p1,q1,_=main78(m,base,z,d,fast);p0i=float(y[p0.argmax()]);p1i=float(y[p1.argmax()]);best=torch.argmax(y);g0=ground78(p0,y);g1=ground78(p1,y);v=torch.stack([p0,p1]);scores.append(v.cpu().numpy());values.append([p0i,p1i]);s0=p0.clamp_min(1e-300).log();s1=p1.clamp_min(1e-300).log();other=torch.arange(len(y),device='cuda')!=best;rows.append({'row':len(rows)+1,'ground0':float(g0),'ground1':float(g1),'expected_IoU0':float((p0*y).sum()),'expected_IoU1':float((p1*y).sum()),'best_IoU_candidate_margin0':float(s0[best]-s0[other].max()),'best_IoU_candidate_margin1':float(s1[best]-s1[other].max()),'auxiliary':float(a.detach()),'inner_gradient_norm':float(gn.detach()),'actual_parameter_step_L2':float(torch.sqrt(sum((fast[n]-p).square().sum() for n,p in m.tta_parameters()))),'q_changed':int(p0.argmax())!=int(p1.argmax())})
  del fast
 v=np.asarray(values);np.savez_compressed(savepath,scores=np.stack(scores),IoU=v);return {'zero_mIoU_percent':float(v[:,0].mean()*100),'one_mIoU_percent':float(v[:,1].mean()*100),'gain_pp':float((v[:,1]-v[:,0]).mean()*100),'zero_Acc25_percent':float((v[:,0]>.25).mean()*100),'one_Acc25_percent':float((v[:,1]>.25).mean()*100),'zero_Acc50_percent':float((v[:,0]>.5).mean()*100),'one_Acc50_percent':float((v[:,1]>.5).mean()*100),'improved':int((v[:,1]-v[:,0]>1e-10).sum()),'worsened':int((v[:,1]-v[:,0]<-1e-10).sum()),'query_changes':sum(x['q_changed'] for x in rows),'mean_ground0':float(np.mean([x['ground0'] for x in rows])),'mean_ground1':float(np.mean([x['ground1'] for x in rows]))},rows
