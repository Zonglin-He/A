"""CPU saved-artifact coordinate-space gradient diagnosis, not student updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys,json,hashlib,gc
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
torch.set_num_threads(1)
ROOT=Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve();TRIAL=ROOT/'outputs/layout_reward_dev';OUT=TRIAL/'research'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/layout_reward'))
from layout_reward import LayoutReward
from stvg.policy import chart,unchart,diagonal,mean_gradient

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def a(x):return x.detach().numpy() if torch.is_tensor(x) else np.asarray(x)
def summary(x):
 z=np.asarray(x,dtype=float);z=z[np.isfinite(z)]
 return dict(n=len(z),mean=float(z.mean()),median=float(np.median(z)),positive_fraction=float((z>1e-10).mean()),negative_fraction=float((z < -1e-10).mean())) if len(z) else dict(n=0)
def cos(x,y):
 den=np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1)
 return np.divide((x*y).sum(-1),den,out=np.full_like(den,np.nan),where=den>1e-12)
def rho(x,y):
 x=rankdata(x,axis=-1);y=rankdata(y,axis=-1);x-=x.mean(-1,keepdims=True);y-=y.mean(-1,keepdims=True);return cos(x,y)
def tiou(box,gt):
 xy=torch.cat((box[...,:2]-box[...,2:]/2,box[...,:2]+box[...,2:]/2),-1).clamp_min(0)
 area=(xy[...,2:]-xy[...,:2]).clamp_min(0).prod(-1);area_g=(gt[...,2:]-gt[...,:2]).clamp_min(0).prod(-1)
 inter=(torch.minimum(xy[...,2:],gt[...,2:])-torch.maximum(xy[...,:2],gt[...,:2])).clamp_min(0).prod(-1)
 return inter/(area+area_g-inter).clamp_min(1e-15)
def unit(x):return x/torch.linalg.vector_norm(x,dim=-1,keepdim=True).clamp_min(1e-20)
seal=read(TRIAL/'scope_hc2/sealed.json');records=[json.loads(l) for l in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if l and json.loads(l)['dataset']=='hc2'];assert len(records)==128 and seal['queries']==128
for r in records:
 k=r['id'];receipt=read(TRIAL/'predictions'/f'{k}.json');assert sha(TRIAL/'predictions'/f'{k}.npz')==seal['files'][k]['sha256'] and sha(TRIAL/'traces'/f'{k}.pt')==receipt['trace_sha256'];cs=read(TRIAL/'student_cache'/f'{k}.json');assert sha(TRIAL/'student_cache'/f'{k}.pt')==cs['sha256']
labelpath=ROOT/'data/evaluation/development_labels.jsonl';labelsha=sha(labelpath);labels={x['id']:x for x in map(json.loads,labelpath.read_text().splitlines())if x['dataset']=='hc2'}
metrics={x['id']:x for x in read(TRIAL/'scope_hc2/metrics.json')['rows']};queries=[];anchors=[];reward_replay_maxerr=0;proposal_maxerr=0
for j,r in enumerate(records):
 k=r['id'];g=labels[k];meta=r['metadata'];wh=np.array([meta['width'],meta['height']]*2)
 c=torch.load(TRIAL/'student_cache'/f'{k}.pt',map_location='cpu',weights_only=False);tr=torch.load(TRIAL/'traces'/f'{k}.pt',map_location='cpu',weights_only=False)
 pos=c['planned_positions'];ids=c['frame_ids'];have=np.array([str(ids[p])in g['boxes'] for p in pos]);idx=np.flatnonzero(have)
 boxes=torch.as_tensor(c['E'][pos],dtype=torch.float64);D=diagonal(boxes).double();mu=chart(boxes).detach().requires_grad_(True)
 reward=LayoutReward(c['teacher']['layout_features'],c['teacher']['layout_mask'],c['teacher']['boxes'],device='cpu')
 value=reward(unchart(mu)[:,None]).squeeze(1);gR=torch.autograd.grad(value.sum(),mu)[0].detach();gRz=D*gR
 rr=tr['LayoutOPD']['rounds'][0]['rollout'];actions=rr['actions'].double();weights=rr['weights'].double();saved_mu=rr['proposal_mean'].double()
 proposal_maxerr=max(proposal_maxerr,float((mu.detach()-saved_mu).abs().max()));reward_replay_maxerr=max(reward_replay_maxerr,float((value.detach()-tr['LayoutOPD']['path'][0]['layout_reward']).abs().max()))
 # Mean NLL loss gradient; convert its ascent counterpart to normalized-coordinate covector.
 gLoss=mean_gradient(mu.detach(),actions,D,weights,(.05,.25));gOz=-D*gLoss
 q=dict(id=k,public_id=f'H{j+1:03}',GT_anchors=len(idx),metrics=metrics[k]['values'],rows=[])
 if len(idx):
  gtxy=torch.tensor(np.array([g['boxes'][str(ids[pos[t]])] for t in idx])/wh,dtype=torch.float64)
  iv=tiou(unchart(mu)[idx],gtxy);gI=torch.autograd.grad(iv.sum(),mu)[0].detach();gIz=D*gI
  gtbox=torch.cat(((gtxy[:,:2]+gtxy[:,2:])/2,gtxy[:,2:]-gtxy[:,:2]),-1);correction=(chart(gtbox)-mu.detach()[idx])/D[idx]
  R=rho(a(rr['rewards'])[idx],a(tiou(rr['boxes'].double()[idx],gtxy[:,None])))
  direct_gt=cos(a(gRz[idx]),a(gIz[idx]));opd_gt=cos(a(gOz[idx]),a(gIz[idx]));direct_corr=cos(a(gRz[idx]),a(correction));opd_corr=cos(a(gOz[idx]),a(correction))
  direct_step_mu=mu.detach()[idx]+.05*D[idx]*unit(gRz[idx]);opd_step_mu=mu.detach()[idx]+.05*D[idx]*unit(gOz[idx])
  direct_iou=tiou(unchart(direct_step_mu),gtxy);opd_iou=tiou(unchart(opd_step_mu),gtxy)
  for t,z in enumerate(idx):
   row=dict(id=k,public_id=q['public_id'],anchor=int(z),frame=int(ids[pos[z]]),initial_IoU=float(iv[t].detach()),initial_layout=float(value[z].detach()),point_grad_z_norm=float(gRz[z].norm()),GT_IoU_grad_z_norm=float(gIz[z].norm()),OPD_grad_z_norm=float(gOz[z].norm()),point_GT_cos=float(direct_gt[t]),OPD_GT_cos=float(opd_gt[t]),point_correction_cos=float(direct_corr[t]),OPD_correction_cos=float(opd_corr[t]),candidate_rho=float(R[t]),point_step_IoU_gain=float((direct_iou[t]-iv[t]).detach()),OPD_step_IoU_gain=float((opd_iou[t]-iv[t]).detach()),point_grad_chart=a(gR[z]).tolist(),OPD_grad_chart=a(-gLoss[z]).tolist(),GT_grad_chart=a(gI[z]).tolist(),D=a(D[z]).tolist())
   anchors.append(row);q['rows'].append(row)
  q['point_GT_cos']=summary(direct_gt);q['OPD_GT_cos']=summary(opd_gt);q['point_correction_cos']=summary(direct_corr);q['OPD_correction_cos']=summary(opd_corr);q['candidate_rho']=summary(R);q['point_step_IoU_gain']=summary(a(direct_iou-iv.detach()));q['OPD_step_IoU_gain']=summary(a(opd_iou-iv.detach()));q['zero_GT_grad']=int((gIz[idx].norm(dim=-1)<1e-10).sum());q['zero_point_grad']=int((gRz[idx].norm(dim=-1)<1e-10).sum())
 queries.append(q);del c,tr,reward;gc.collect()
 if (j+1)%16==0:print('initial_coordinate_scan',j+1,flush=True)

def paired(field1,field2):
 rows=[q for q in queries if q.get(field1,{}).get('n',0)>0 and q.get(field2,{}).get('n',0)>0];d=np.array([q[field1]['mean']-q[field2]['mean']for q in rows]);rng=np.random.default_rng(20261021);idx=rng.integers(0,len(d),(10000,len(d)))
 return dict(valid_queries=len(d),mean=float(d.mean()),CI95=np.quantile(d[idx].mean(-1),[.025,.975]).tolist(),positive=int((d>1e-10).sum()),negative=int((d < -1e-10).sum()))
agg=dict(queries=128,anchors_with_GT=len(anchors),queries_without_GT_anchors=sum(q['GT_anchors']==0 for q in queries),initial_zero_IoU=sum(x['initial_IoU']==0 for x in anchors),GT_IoU_zero_gradient=sum(x['GT_IoU_grad_z_norm']<1e-10 for x in anchors),point_zero_gradient=sum(x['point_grad_z_norm']<1e-10 for x in anchors),OPD_zero_gradient=sum(x['OPD_grad_z_norm']<1e-10 for x in anchors),query_macro={f:summary([q[f]['mean']for q in queries if q.get(f,{}).get('n',0)>0])for f in ['candidate_rho','point_GT_cos','OPD_GT_cos','point_correction_cos','OPD_correction_cos','point_step_IoU_gain','OPD_step_IoU_gain']},paired_OPD_minus_point_GT_cos=paired('OPD_GT_cos','point_GT_cos'),paired_OPD_minus_point_correction_cos=paired('OPD_correction_cos','point_correction_cos'),paired_OPD_minus_point_step_IoU_gain=paired('OPD_step_IoU_gain','point_step_IoU_gain'),query_macro_positive_anchor_fraction={f:summary([q[f]['positive_fraction']for q in queries if q.get(f,{}).get('n',0)>0])for f in ['point_GT_cos','OPD_GT_cos','point_correction_cos','OPD_correction_cos','point_step_IoU_gain','OPD_step_IoU_gain']})
rho_good=[q for q in queries if q.get('candidate_rho',{}).get('mean',-1)>0];agg['query_sign_cases']=dict(candidate_positive_rho=len(rho_good),positive_candidate_rho_point_negative_GT_mean=sum(q.get('point_GT_cos',{}).get('mean',0)<0 for q in rho_good),positive_candidate_rho_OPD_negative_GT_mean=sum(q.get('OPD_GT_cos',{}).get('mean',0)<0 for q in rho_good),point_negative_GT_mean_OPD_positive_GT_mean=sum(q.get('point_GT_cos',{}).get('mean',0)<0 and q.get('OPD_GT_cos',{}).get('mean',0)>0 for q in queries),OPD_negative_GT_mean_point_positive_GT_mean=sum(q.get('OPD_GT_cos',{}).get('mean',0)<0 and q.get('point_GT_cos',{}).get('mean',0)>0 for q in queries))
groups={}
for label,lo,hi in [('zero',0,0),('low_nonzero',0,.3),('mid',.3,.7),('high',.7,1.00001)]:
 rows=[x for x in anchors if (x['initial_IoU']==0 if label=='zero' else lo < x['initial_IoU'] < hi if label=='low_nonzero' else lo<=x['initial_IoU']<hi)];qgroup=[]
 for q in queries:
  rr=[x for x in rows if x['id']==q['id']]
  if rr:qgroup.append({f:float(np.nanmean([x[f]for x in rr]))for f in ['point_GT_cos','OPD_GT_cos','point_correction_cos','OPD_correction_cos','candidate_rho','point_step_IoU_gain','OPD_step_IoU_gain']})
 groups[label]=dict(anchors=len(rows),queries=len(qgroup),query_macro={f:summary([x[f]for x in qgroup])for f in ['point_GT_cos','OPD_GT_cos','point_correction_cos','OPD_correction_cos','candidate_rho','point_step_IoU_gain','OPD_step_IoU_gain']})
time_groups={}
for label,fun in [('t_zero',lambda t:t==0),('t_low_nonzero',lambda t:0<t<.3),('t_mid',lambda t:.3<=t<.5),('t_high',lambda t:t>=.5)]:
 qq=[q for q in queries if fun(q['metrics']['WithoutOPD']['t'])];time_groups[label]=dict(queries=len(qq),t_mean=float(np.mean([q['metrics']['WithoutOPD']['t']for q in qq])),arms={})
 for name,arm in [('LayoutOPD','Full'),('LayoutDirect','LayoutDirect'),('GeometryOPD','GeometryOPD')]:
  time_groups[label]['arms'][name]=dict(v_delta_pp=float(np.mean([100*(q['metrics'][arm]['v']-q['metrics']['WithoutOPD']['v'])for q in qq])),unobs_delta_pp=float(np.mean([100*(q['metrics'][arm]['unobserved_actual_sIoU']-q['metrics']['WithoutOPD']['unobserved_actual_sIoU'])for q in qq])),v_improved=sum(q['metrics'][arm]['v']>q['metrics']['WithoutOPD']['v']+1e-10 for q in qq),unobs_improved=sum(q['metrics'][arm]['unobserved_actual_sIoU']>q['metrics']['WithoutOPD']['unobserved_actual_sIoU']+1e-10 for q in qq))
assert sha(labelpath)==labelsha and not torch.cuda.is_initialized();assert proposal_maxerr<1e-12 and reward_replay_maxerr<1e-5
out=dict(status='sealed_HC128_CPU_coordinate_only_not_actual_updates',metric='normalized z coordinates mu=mu0+D*z; D=(Ew,Eh,1,1) frozen; compare covector-gradients D*d/dmu',OPD_direction='-D times analytical mean-gradient of frozen-candidate weighted mixture NLL at exact same E state',finite_step='mu_new=mu0+0.05 D unit(direction_z); no decoder/no GT-selected direction or magnitude',zero_IoU='Report separately: IoU local gradient undefined for diagnostic cosine when norm zero; correction direction remains available',proof=dict(seal_sha256=sha(TRIAL/'scope_hc2/sealed.json'),labels_sha256=labelsha,point_reward_CPU_saved_GPU_maxerr=reward_replay_maxerr,proposal_E_maxerr=proposal_maxerr,no_cuda_initialized=True),aggregate=agg,initial_IoU_groups=groups,time_groups=time_groups,queries=queries)
(OUT/'gradient_signal.json').write_text(json.dumps(out,indent=2,allow_nan=True)+'\n');print(json.dumps(agg,indent=2));print('TIME',json.dumps(time_groups,indent=2))
