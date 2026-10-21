"""Same saved initial 32 candidates and exact CPU ROI-integral coordinate gradients."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
ROOT=Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve();OUT=ROOT/'outputs/layout_reward_dev/research';sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT)]
from mask_completeness import validate_before_GT,unpack,read,sha,MaskScores,arr,summary,cxcywh
from stvg.policy import chart,unchart,diagonal
from stvg.geometry import local_cost
torch.set_num_threads(1)
SCORES=('mask_mean','mask_coverage','pixel_Dice','projection_Dice')
def cos(x,y):
 d=np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1);return np.divide((x*y).sum(-1),d,out=np.full_like(d,np.nan),where=d>1e-12)
def rho(x,y):
 x=rankdata(x,axis=-1);y=rankdata(y,axis=-1);x-=x.mean(-1,keepdims=True);y-=y.mean(-1,keepdims=True);return cos(x,y)
def iou(b,g):
 p=np.maximum(np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1),0);it=np.maximum(np.minimum(p[...,2:],g[...,2:])-np.maximum(p[...,:2],g[...,:2]),0).prod(-1);return it/np.maximum(np.maximum(p[...,2:]-p[...,:2],0).prod(-1)+(g[...,2:]-g[...,:2]).prod(-1)-it,1e-15)
class DifferentiableScores:
 def __init__(self,P):
  p=torch.tensor(P,dtype=torch.float64);self.h,self.w=p.shape[-2:];self.mass=p.sum((1,2));self.px=p.amax(1);self.py=p.amax(2);self.S=F.pad(p.cumsum(1).cumsum(2),(1,0,1,0))[:,None];self.X=F.pad(self.px.cumsum(1),(1,0))[:,None,None];self.Y=F.pad(self.py.cumsum(1),(1,0))[:,None,None]
 def sample2(self,xy):
  n=xy.shape[1];grid=2*xy-1;return F.grid_sample(self.S,grid.reshape(8,n*4,1,2),mode='bilinear',padding_mode='border',align_corners=True).reshape(8,n,4)
 def sample1(self,p,x):
  n=x.shape[1];grid=torch.stack((2*x-1,torch.zeros_like(x)),-1);return F.grid_sample(p,grid.reshape(8,n*2,1,2),mode='bilinear',padding_mode='border',align_corners=True).reshape(8,n,2)
 def __call__(self,b):
  p=torch.cat((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1).clamp(0,1);x0,y0,x1,y1=p.unbind(-1);xy=torch.stack((torch.stack((x1,y1),-1),torch.stack((x0,y1),-1),torch.stack((x1,y0),-1),torch.stack((x0,y0),-1)),-2)
  v=self.sample2(xy);mass=v[...,0]-v[...,1]-v[...,2]+v[...,3];area=(x1-x0)*(y1-y0)*self.w*self.h
  xv=self.sample1(self.X,torch.stack((x0,x1),-1));yv=self.sample1(self.Y,torch.stack((y0,y1),-1));mx=xv[...,1]-xv[...,0];my=yv[...,1]-yv[...,0]
  return dict(mask_mean=mass/area.clamp_min(1e-15),mask_coverage=mass/self.mass[:,None].clamp_min(1e-15),pixel_Dice=2*mass/(area+self.mass[:,None]).clamp_min(1e-15),projection_Dice=.5*(2*mx/((x1-x0)*self.w+self.px.sum(-1)[:,None]).clamp_min(1e-15)+2*my/((y1-y0)*self.h+self.py.sum(-1)[:,None]).clamp_min(1e-15)))

seal,records=validate_before_GT();T=ROOT/'outputs/layout_reward_dev';labelsfile=ROOT/'data/evaluation/development_labels.jsonl';labelsha=sha(labelsfile);labels={x['id']:x for x in map(json.loads,labelsfile.read_text().splitlines())if x['dataset']=='hc2'}
oldgrad=read(OUT/'gradient_signal.json');assert oldgrad['proof']['labels_sha256']==labelsha;oldgrad={q['id']:q for q in oldgrad['queries']};metrics={q['id']:q for q in read(T/'scope_hc2/metrics.json')['rows']}
query=[];maxerr=0.;finite_difference_maxerr=0.
for j,r in enumerate(records):
 k=r['id'];pack=torch.load(OUT/'native_masks'/seal['files'][k]['path'],map_location='cpu',weights_only=False);P,raw,selected,z,_,_=unpack(pack);c=torch.load(T/'student_cache'/f'{k}.pt',map_location='cpu',weights_only=False);tr=torch.load(T/'traces'/f'{k}.pt',map_location='cpu',weights_only=False);rr=tr['LayoutOPD']['rounds'][0]['rollout'];pos=c['planned_positions'];ids=c['frame_ids'];gt=labels[k]['boxes'];idx=np.array([i for i,p in enumerate(pos)if str(ids[p])in gt],dtype=int);wh=np.array([r['metadata']['width'],r['metadata']['height']]*2)
 ds=DifferentiableScores(P);b=torch.tensor(c['E'][pos],dtype=torch.float64);mu=chart(b).detach().requires_grad_(True);D=diagonal(b);point=ds(unchart(mu)[:,None]);cand=rr['boxes'].double();cv=ds(cand);grad={s:torch.autograd.grad(point[s].sum(),mu,retain_graph=s!=SCORES[-1])[0].detach()for s in SCORES};q=dict(id=k,public_id=f'H{j+1:03}',GTanchors=len(idx),rankings={},point_gradients={})
 # Exact equivalence to the independent NumPy fractional integral at all selected initial boxes.
 for ai in range(8):
  v=MaskScores(P[ai]).score(arr(b[ai]));maxerr=max(maxerr,max(abs(float(point[s][ai,0])-v[s])for s in SCORES))
 # Central finite difference only on the first query, fixed 1e-6 step in chart coordinates; no decoder.
 if j==0:
  for dim in range(4):
   plus=mu.detach().clone();minus=mu.detach().clone();plus[:,dim]+=1e-6;minus[:,dim]-=1e-6;pv=ds(unchart(plus)[:,None]);mv=ds(unchart(minus)[:,None])
   for s in SCORES:
    num=((pv[s]-mv[s])/(2e-6)).squeeze(1);finite_difference_maxerr=max(finite_difference_maxerr,float((num-grad[s][:,dim]).abs().max()))
 if len(idx):
  g=np.array([gt[str(ids[pos[a]])]for a in idx])/wh;truth=iou(arr(cand)[idx],g[:,None]);layout=arr(rr['rewards'])[idx];geometry=-arr(local_cost(cand,torch.as_tensor(z['boxes']).double()[:,None]))[idx]
  for s,v in [(s,arr(cv[s])[idx])for s in SCORES]+[('old_layout',layout),('geometry',geometry)]:
   rv=rho(v,truth);q['rankings'][s]=summary(rv)
  lookup={a['anchor']:a for a in oldgrad[k]['rows']};GTg=np.array([lookup[a]['GT_grad_chart']for a in idx]);gbox=np.stack([cxcywh(gg)for gg in g]);correction=(arr(chart(torch.as_tensor(gbox)))-arr(mu.detach())[idx])/arr(D)[idx]
  for s in SCORES:
   gz=arr(D*grad[s])[idx];iz=arr(D)[idx]*GTg;cg=cos(gz,iz);cc=cos(gz,correction)
   q['point_gradients'][s]=dict(GT_gradient_cos=summary(cg),GT_correction_cos=summary(cc),gradient_z_norm=summary(np.linalg.norm(gz,axis=-1)),gradient_zero=int((np.linalg.norm(gz,axis=-1)<1e-10).sum()),GT_gradient_zero=int((np.linalg.norm(iz,axis=-1)<1e-10).sum()),rows=[dict(anchor=int(a),GT_cos=float(cg[t]),correction_cos=float(cc[t]),gradient_norm=float(np.linalg.norm(gz[t])),gradient_chart=arr(grad[s])[a].tolist())for t,a in enumerate(idx)])
 query.append(q);del c,tr,pack,P,ds;gc.collect()
 if (j+1)%16==0:print('same32_mask_CPU',j+1,flush=True)

def qmacro(dpath):
 out=[]
 for q in query:
  v=q
  for key in dpath:v=v.get(key,{})
  if isinstance(v,dict)and v.get('n',0)>0:out.append(v['mean'])
 return summary(out)
def paired_rho(a,b):
 rr=[q for q in query if q['rankings'].get(a,{}).get('n',0)>0 and q['rankings'].get(b,{}).get('n',0)>0];d=np.array([q['rankings'][a]['mean']-q['rankings'][b]['mean']for q in rr]);rng=np.random.default_rng(20261021);ii=rng.integers(0,len(d),(10000,len(d)));return dict(query_n=len(d),mean=float(d.mean()),CI95=np.quantile(d[ii].mean(-1),[.025,.975]).tolist(),positive=int((d>1e-10).sum()),negative=int((d < -1e-10).sum()))
aggregate=dict(rankings={s:qmacro(['rankings',s])for s in SCORES+('old_layout','geometry')},paired_projection_minus_layout=paired_rho('projection_Dice','old_layout'),paired_projection_minus_geometry=paired_rho('projection_Dice','geometry'),point_gradients={s:{f:qmacro(['point_gradients',s,f])for f in ['GT_gradient_cos','GT_correction_cos','gradient_z_norm']}for s in SCORES},point_zero_gradients={s:sum(q.get('point_gradients',{}).get(s,{}).get('gradient_zero',0)for q in query)for s in SCORES})
# Annotated box center/size changes are proxies, not occlusion or true motion labels.
motion=[]
for r in records:
 k=r['id'];g=labels[k]['boxes'];meta=r['metadata'];ff=np.array([f for f in meta['frame_ids']if str(f)in g]);bb=np.array([g[str(f)]for f in ff],dtype=float);cc=(bb[:,:2]+bb[:,2:])/2;ww=bb[:,2:]-bb[:,:2];dt=np.diff(ff)/meta['fps'];speed=np.linalg.norm(np.diff(cc,axis=0)/((ww[:-1]+ww[1:])/2),axis=-1)/dt;size=np.linalg.norm(np.diff(np.log(ww),axis=0),axis=-1)/dt
 motion.append(dict(id=k,center_change_proxy=float(speed.mean()),logsize_change_proxy=float(size.mean()),metrics=metrics[k]['values']))
groups={}
for proxy in ['center_change_proxy','logsize_change_proxy']:
 edge=np.quantile([x[proxy]for x in motion],[.25,.5,.75]);bins=np.digitize([x[proxy]for x in motion],edge);gg=[]
 for b in range(4):
  rr=[x for i,x in enumerate(motion)if bins[i]==b];gg.append(dict(quartile=b+1,queries=len(rr),proxy_mean=float(np.mean([x[proxy]for x in rr])),tIoU_mean=float(np.mean([x['metrics']['WithoutOPD']['t']for x in rr])),arms={a:dict(v_delta_pp=float(np.mean([100*(x['metrics'][a]['v']-x['metrics']['WithoutOPD']['v'])for x in rr])),unobs_delta_pp=float(np.mean([100*(x['metrics'][a]['unobserved_actual_sIoU']-x['metrics']['WithoutOPD']['unobserved_actual_sIoU'])for x in rr])))for a in ['GeometryOPD','Full','LayoutDirect']}))
 groups[proxy]=dict(quartile_edges=edge.tolist(),groups=gg)
assert maxerr<1e-7 and finite_difference_maxerr<1e-4 and sha(labelsfile)==labelsha and not torch.cuda.is_initialized()
out=dict(status='CPU_same_saved_initial32_diagnostic_no_method_adaptation_no_temperature',proof=dict(native_mask_seal_sha256=sha(OUT/'native_masks/sealed.json'),GT_sha256=labelsha,independent_integral_max_score_error=maxerr,finite_difference_max_coordinate_gradient_error=finite_difference_maxerr),coordinates='mu=(cx,cy,logw,logh); D=(Ew,Eh,1,1); compare D*gradient_mu in normalized z coordinates',aggregate=aggregate,GT_box_change_proxy_groups=groups,queries=query)
(OUT/'mask_same_candidates.json').write_text(json.dumps(out,indent=2,allow_nan=True)+'\n');(OUT/'mask_same_candidates_public_summary.json').write_text(json.dumps({k:v for k,v in out.items()if k!='queries'},indent=2,allow_nan=True)+'\n');print(json.dumps(aggregate,indent=2));print('GT_BOX_CHANGE_PROXIES',json.dumps(groups,indent=2))
