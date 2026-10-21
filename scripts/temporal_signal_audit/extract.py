"""CPU-only frozen time-subgraph and spatial-cache audit. Never reads labels.

No Source/encoder/ReferDINO construction. This is diagnostic reconstruction,
not another inference queue. Do not import this module from production code.
"""
from pathlib import Path
import os,sys,json,hashlib,time,argparse
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from stvg.tube_replay import occupancy_prior
from scripts.tube_fusion.core import restore_geometry
from stvg.spatial import evidence_dp
OUT=ROOT/'outputs/temporal_signal_audit_v1'
PUBLIC=ROOT/'results/temporal_signal_audit_v1'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def arr(x):return x.detach().cpu().numpy()if torch.is_tensor(x)else np.asarray(x)

class Layer(nn.Module):
 def __init__(self):
  super().__init__();self.self_attn=nn.MultiheadAttention(256,8);self.cross_attn_image=nn.MultiheadAttention(256,8)
  self.linear1=nn.Linear(256,2048);self.linear2=nn.Linear(2048,256)
  self.norm1=nn.LayerNorm(256);self.norm3=nn.LayerNorm(256);self.norm4=nn.LayerNorm(256)
 def forward(self,x,i):
  q=x+i['query_time'];v,S=self.self_attn(q,q,x,key_padding_mask=i['query_mask'],average_attn_weights=False)
  x=self.norm1(x+v)
  v,A=self.cross_attn_image(x.permute(1,0,2),i['encoded_feature']+i['encoded_pos'],i['encoded_feature'],key_padding_mask=i['encoded_mask'],average_attn_weights=False)
  x=self.norm3(x+v.transpose(0,1));x=self.norm4(x+self.linear2(torch.relu(self.linear1(x))))
  return x,A[:,:,0,:].mean(1),S[0].mean(0)
class Decoder(nn.Module):
 def __init__(self,path):
  super().__init__();self.layers=nn.ModuleList([Layer()for _ in range(6)]);self.norm=nn.LayerNorm(256)
  c=torch.load(path,map_location='cpu',weights_only=False,mmap=True);s=c.get('model_ema',c.get('model'));prefix='ground_decoder.time_decoder.'
  self.load_state_dict({k[len(prefix):]:v for k,v in s.items()if k.startswith(prefix)},strict=True)
  self.eval().requires_grad_(False)
 @torch.no_grad()
 def recover(self,o,boxes,observed):
  i=o['inputs'];x=i['query_tgt'];n=len(x);h,w=o['grid_size'];nv=h*w;legal=~i['encoded_mask'][:,-nv:]
  M,_=occupancy_prior(boxes,o['grid_size'],legal);P,_=occupancy_prior(boxes,o['grid_size'],legal,permuted=True)
  area=M.sum(-1)/legal.sum(-1).clamp_min(1);signals=[];errors=[]
  for ell,layer in enumerate(self.layers):
   x,A,S=layer(x,i);hidden=self.norm(x).transpose(0,1)
   errors.append(float((hidden-o['hidden'][ell]).abs().max()))
   visual=A[:,-nv:].double();mass=visual.sum(-1);prob=visual/mass.clamp_min(1e-30)[:,None]
   target=(prob*M).sum(-1);perm=(prob*P).sum(-1)
   entropy=-(prob*prob.clamp_min(1e-30).log()).sum(-1)/legal.sum(-1).double().log().clamp_min(1)
   # Key salience: attention received from every time query. Mean over keys is 1.
   key=S.mean(0)*n;diagonal=S.diagonal();obs=torch.as_tensor(observed,dtype=torch.bool)
   anchor=S[:,obs].sum(-1)if obs.any()else torch.zeros(n)
   signals.append(torch.stack([mass,target,perm,target-area,target-perm,entropy,key,diagonal,anchor],-1).numpy())
  assert max(errors)<=1e-4,f'CPU reconstruction failed: {max(errors)} > 1e-4'
  return np.stack(signals),errors

def prepare():
 assert not(OUT/'plan.json').exists(),'Immutable plan already exists'
 dev=[json.loads(l)for l in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if l.strip()]
 lock=read(ROOT/'outputs/tube_temporal_hc64/lock.json');by={r['id']:r for r in dev};selection=read(ROOT/'outputs/interval_pool/confirmation_selection.json')['development_recheck_selection']
 cohorts={'HC64':[by[k]for k in lock['ids']], 'HC32':[x['record']for x in selection['hc2']], 'Vid32':[x['record']for x in selection['vidstg']]};items=[]
 for cohort,records in cohorts.items():
  base=ROOT/('outputs/tube_temporal_hc64'if cohort=='HC64'else 'outputs/interval_pool/recheck_'+('hc2'if cohort=='HC32'else 'vidstg'))
  for index,r in enumerate(records):
   key=r['id'];receipt=base/'receipts'/f'{key}.json';rec=read(receipt);p=base/rec['capture_path'];assert sha(p)==rec['capture_sha256']
   q=(ROOT/'outputs/tube_fusion_hc64/posteriors'/f'{key}.pt')if cohort=='HC64'else(base/rec['posterior_path'])
   items.append(dict(cohort=cohort,anonymous=cohort[0]+('D'if cohort=='HC64'else'R')+f'{index+1:03d}',id=key,dataset=r['dataset'],fps=r['metadata']['fps'],capture=str(p.relative_to(ROOT)),capture_sha256=sha(p),posterior=str(q.relative_to(ROOT)),posterior_sha256=sha(q),receipt=str(receipt.relative_to(ROOT)),receipt_sha256=sha(receipt)))
 weights={k:dict(path='checkpoints/'+name,sha256=sha(ROOT/'checkpoints'/name))for k,name in [('hc2','TASTVG_VidSTG.pth'),('vidstg','TASTVG_HCSTVG2.pth')]}
 labels={k:dict(path=p,sha256=sha(ROOT/p))for k,p in [('HC64','data/evaluation/development_labels.jsonl'),('HC32','data/evaluation/hc2_labels.jsonl'),('Vid32','data/evaluation/vidstg_labels.jsonl')]}
 # Hashing label files binds them but does not deserialize or inspect any label.
 sources={str(p.relative_to(ROOT)):sha(p)for p in sorted((ROOT/'scripts/temporal_signal_audit').glob('*.py'))}
 plan=dict(schema='cached_temporal_signal_audit_v1',status='locked_before_label_deserialization',items=items,weights=weights,labels=labels,sources=sources,
 primary_event=['confidence','eref_support','aux_support','attention_placement'],primary_boundary=['link_break','box_speed','attention_placement_change','hidden_change'],
 protocol=dict(near_boundary_seconds=1.,far_boundary_seconds=2.,minimum_per_class=2,bootstrap_queries=10000,bootstrap_seed=20261022,circular_nonzero_shifts=31,
 event_primary='unobserved nodes, GT interval [start,end); per-query AUC with average ties; report AUC-.5 and circular alignment null, no direction flipping',
 boundary='start/end separate and pooled; near <=1s vs >2s, boundary must be inside captured axis; no nearest-node substitution',
 nuisance='label-free residual from intercept, normalized time, time squared, nearest observation proximity exp(-distance/1s), log clipped ROI area; primary replication unchanged',
 attention='mean 8 heads, all 6 layers primary; 6 separate layers secondary, no GT layer selection; legal-token fixed half-roll control, area baseline',
 hidden_change='within-offset (stride2) cosine change assigned to center, not interleaved-offset artifact',
 visibility='Event annotations alone cannot establish target presence/identity outside event; no causal interpretation.',
 cpu_hidden_tolerance_maxabs=1e-4,not_pristine_confirmation=True,no_new_method_or_decoding=True),
 formal_plan_sha256=sha(ROOT/'outputs/boxfixed_full_v1/plan.json'),GT_read=False)
 write(OUT/'plan.json',plan);print('Locked',len(items),'queries',flush=True)

def gradient(v,t):
 return np.gradient(np.asarray(v,dtype=float),np.asarray(t,dtype=float),edge_order=1)if len(t)>1 else np.zeros_like(v)
def cosine_change(v):
 v=np.asarray(v,float);v=v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-30);z=np.zeros(len(v))
 if len(v)>2:
  d=1-(v[2:]*v[:-2]).sum(-1);z[1:-1]=d;z[0]=d[0];z[-1]=d[-1]
 return z

@torch.no_grad()
def extract():
 torch.set_num_threads(1);os.nice(15);p=read(OUT/'plan.json');assert not(OUT/'sealed.json').exists()
 for k,v in p['sources'].items():assert sha(ROOT/k)==v
 models={};done=[];start=time.perf_counter()
 for j,item in enumerate(p['items']):
  target=OUT/'signals'/f"{item['anonymous']}.npz"
  if target.exists():raise RuntimeError('No silent overwrite/resume')
  if item['dataset']not in models:
   w=p['weights'][item['dataset']];assert sha(ROOT/w['path'])==w['sha256'];models[item['dataset']]=Decoder(ROOT/w['path'])
  assert sha(ROOT/item['capture'])==item['capture_sha256'];assert sha(ROOT/item['posterior'])==item['posterior_sha256']
  c=torch.load(ROOT/item['capture'],map_location='cpu',weights_only=False);g=c['geometry'];T=g['frames'];ids=arr(g['frame_ids']);t=ids/item['fps'];obs=arr(g['observed']).astype(bool);geom=restore_geometry(g)
  dp=evidence_dp(geom.for_full_timeline());actions=arr(dp['actions']).astype(int);box=arr(dp['final']);aux=arr(g['u'])[np.arange(T),actions];feature=arr(g['features'])[np.arange(T),actions]
  links=arr(g['link'])[np.arange(T-1),actions[:-1],actions[1:]];link=np.r_[links[0],links];link[1:-1]=(links[:-1]+links[1:])/2
  area=box[:,2]*box[:,3];dist=np.abs(t[:,None]-t[obs][None,:]).min(-1)
  v=np.column_stack([np.gradient(box[:,0],t),np.gradient(box[:,1],t),np.gradient(np.log(box[:,2]),t),np.gradient(np.log(box[:,3]),t)])
  sig=dict(confidence=arr(g['confidence']),eref_support=arr(g['u'])[:,0],aux_support=aux,link_break=1-link,box_speed=np.linalg.norm(v,axis=1),appearance_change=cosine_change(feature),area=area,anchor_proximity=np.exp(-dist),time_position=(t-t[0])/max(t[-1]-t[0],1e-9))
  att=np.zeros((6,T,9));err=[];hidden=np.zeros((T,256));
  for o in c['temporal_cache']['offsets']:
   ix=np.asarray(o['frame_positions'],int);a,e=models[item['dataset']].recover(o,box[ix],obs[ix]);att[:,ix]=a;err+=e;hidden[ix]=arr(o['hidden'][-1,0])
  names=['visual_mass','roi_mass','permuted_roi_mass','roi_enrichment','attention_placement','visual_entropy','temporal_key_salience','temporal_self_diagonal','temporal_anchor_mass']
  for k,name in enumerate(names):
   sig[name]=att[:,:,k].mean(0)
   for ell in range(6):sig[name+f'_L{ell+1}']=att[ell,:,k]
  sig['attention_placement_change']=abs(gradient(sig['attention_placement'],t));sig['hidden_change']=cosine_change(hidden)
  q=torch.load(ROOT/item['posterior'],map_location='cpu',weights_only=False);q=arr(q['means']['G']if 'means'in q else q['hull']);assert q.shape==(T,T)
  sig['native_event_mass']=np.asarray([q[:k+1,k:].sum()for k in range(T)])
  sig['native_start_mass']=q.sum(-1);sig['native_end_mass']=q.sum(0);sig['native_boundary_mass']=sig['native_start_mass']+sig['native_end_mass']
  X=np.column_stack([np.ones(T),sig['time_position'],sig['time_position']**2,sig['anchor_proximity'],np.log(area.clip(1e-8))])
  for name in ['confidence','eref_support','aux_support','attention_placement','roi_enrichment','visual_mass','temporal_key_salience']:
   sig[name+'_residual']=sig[name]-X@np.linalg.lstsq(X,sig[name],rcond=None)[0]
  assert all(np.isfinite(v).all()and np.shape(v)==(T,)for v in sig.values())
  target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,frame_ids=ids,observed=obs,**sig)
  row=dict(anonymous=item['anonymous'],path=str(target.relative_to(OUT)),sha256=sha(target),max_hidden_error=max(err),hidden_checks=len(err),nodes=T,GT_read=False)
  done.append(row);write(OUT/'progress.json',dict(completed=j+1,total=len(p['items']),seconds=time.perf_counter()-start,GT_read=False,CUDA_initialized=torch.cuda.is_initialized()))
  if j%8==0:print(j+1,len(p['items']),'maxerr',max(err),'seconds',round(time.perf_counter()-start,2),flush=True)
 assert not torch.cuda.is_initialized();assert sha(ROOT/'outputs/boxfixed_full_v1/plan.json')==p['formal_plan_sha256']
 write(OUT/'sealed.json',dict(status='all_label_free_signals_sealed',plan_sha256=sha(OUT/'plan.json'),files=done,GT_read=False,CUDA_initialized=False,seconds=time.perf_counter()-start,new_RGB=0,new_experts=0,new_encoder=0,GPU_calls=0,CPU_time_subgraph_calls=2*len(done),parameter_updates=0))
 print('sealed',len(done),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','extract']);x=a.parse_args();globals()[x.action]()
