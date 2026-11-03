"""Independent official-decoder replay and loss/gradient audit, no fit/GT."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round51'))
from model51 import load_source,fingerprint,native_matching,PREFIX_KEYS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest_order(model,names):
 state=model.state_dict();assert set(state)==set(names);h=hashlib.sha256()
 for n in names:
  v=state[n].detach().cpu().contiguous();h.update(n.encode()+b'\0');h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
 return h.hexdigest()
class AuditQK(nn.Module):
 def __init__(self):
  super().__init__();self.A_q=nn.Parameter(torch.zeros((4,288),device='cuda'));self.B_q=nn.Parameter(torch.zeros((288,4),device='cuda'));self.A_k=nn.Parameter(torch.zeros((4,288),device='cuda'));self.B_k=nn.Parameter(torch.zeros((288,4),device='cuda'))
 def forward(self,source):return source+torch.cat((self.B_q@self.A_q,self.B_k@self.A_k,torch.zeros_like(source[576:])),0)
 def set(self,values):
  with torch.no_grad():
   for n,p in self.named_parameters():p.copy_(values[n]);p.grad=None
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan)
 for rel,h in {**p['programs_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/rel)==h
 torch.manual_seed(0);np.random.seed(0);start=time.perf_counter();model=load_source(read(ROOT/p['source_plan']));original_fp=fingerprint(model);order=list(model.state_dict());module=model.decoder[5].cross_v;adapter=AuditQK();parametrize.register_parametrization(module,'in_proj_weight',adapter);rows=read(ROOT/p['manifest']);selected=[]
 for pl in ['drone','quad']:selected+=sorted([r for r in rows if r['platform']==pl],key=lambda r:hashlib.sha256(('R54_independent_method_v1:'+r['id']).encode()).digest())[:2]
 score_error=0.;loss_error=0.;replayed=0;gradient_rows=[];gradient_scope=True;all_A_zero=True
 for row in selected:
  sid=row['id'];latent=ROOT/p['latent_root']/'samples_PRIVATE'/(sid+'.npz');obsfile=ROOT/p['observation_root']/'samples_PRIVATE'/(sid+'.npz');predfile=ROOT/p['prediction_root']/'samples_PRIVATE'/(sid+'.npz');mainfit=ROOT/p['main_root']/'fitted_PRIVATE'/(sid+'.npz');strongfit=ROOT/p['strong_root']/'fitted_PRIVATE'/(sid+'.npz')
  with np.load(latent,allow_pickle=False) as z:data={k:z[k].copy() for k in ['h','scores','logodds','tokens','token_mask',*PREFIX_KEYS] if k in z.files}
  with np.load(obsfile,allow_pickle=False) as z:obs={k:torch.as_tensor(z[k],device='cuda',dtype=torch.float32 if z[k].dtype.kind=='f' else None) for k in ['allowed','geometry_cost','RGB_cost','RGB_shuffle_cost','edge_pairs','edge_affinity','edge_shuffle_affinity','point_RGB_known','query_prior','view_A','view_B']}
  with np.load(predfile,allow_pickle=False) as z:arms=z['arm_names'].tolist();expected=z['scores'];expected_q=z['queries']
  with np.load(mainfit,allow_pickle=False) as z:fit={k:z[k].copy() for k in z.files}
  with np.load(strongfit,allow_pickle=False) as z:fit.update({k:z[k].copy() for k in z.files})
  tensor=lambda x:torch.as_tensor(x,device='cuda');args=[tensor(data[k]) if k in data else None for k in PREFIX_KEYS];tokens=tensor(data['tokens'])[None];mask=tensor(data['token_mask']).bool();baseq=int(np.argmax(data['scores']));z0=tensor(data['logodds']);source_values={n:tensor(fit['qk_1__'+n]) if n.startswith('A_') else torch.zeros_like(v) for n,v in adapter.named_parameters()};adapter.set(source_values);views=[torch.arange(1024,device='cuda'),obs['view_A'].long(),obs['view_B'].long()];original_forward=module.forward;captured={}
  def before(m,aa,kw):captured['q']=kw['query'];captured['k']=kw['key']
  def after(m,aa,value):captured['attention']=value[1][0]
  handles=[module.register_forward_pre_hook(before,with_kwargs=True),module.register_forward_hook(after)]
  def forward(indices,U=None):
   local=list(args);local[1]=args[1][:,indices]
   if U is not None:
    def injected(*aa,**kwargs):kwargs['attn_mask']=U[:,indices];return original_forward(*aa,**kwargs)
    module.forward=injected
   try:h=model.decoder[5](*local,detected_feats=None,detected_mask=None)
   finally:module.forward=original_forward
   query=F.normalize(model.contrastive_align_projection_image(h),dim=-1);proba,z=native_matching(query,tokens,mask);A=captured['attention'];fg=A*len(indices)*obs['query_prior'][:,None]*obs['allowed'][:,indices];allmass=torch.cat((fg,fg.new_full((1,len(indices)),.2)),0);S=allmass/allmass.sum(0,keepdim=True).clamp_min(1e-20);return proba,z,A,S,h[0]
  source=[]
  with torch.no_grad():
   for ix in views:
    proba,z,A,S,h=forward(ix);inverse=torch.full((1024,),-1,device='cuda',dtype=torch.long);inverse[ix]=torch.arange(len(ix),device='cuda');edges=obs['edge_pairs'].long();keep=(inverse[edges[:,0]]>=0)&(inverse[edges[:,1]]>=0);edge=inverse[edges[keep]];raw=(obs['edge_affinity'][keep]*(S[:,edge[:,0]]-S[:,edge[:,1]]).square().sum(0)).mean();source.append((S.detach(),S.mean(1).detach(),max(float(raw),.01),keep,edge))
   proba,z,A,S,h=forward(views[0]);assert torch.equal(proba,tensor(data['scores'])) and torch.equal(h,tensor(data['h']));qinput=captured['q'].detach();kinput=captured['k'].detach()
  shared_np=np.intersect1d(obs['view_A'].cpu().numpy(),obs['view_B'].cpu().numpy());shared=[torch.searchsorted(ix,tensor(shared_np)) for ix in views[1:]]
  def relative_entropy(P,Q,axis):return torch.sum(P*(torch.log(torch.clamp(P,min=1e-12))-torch.log(torch.clamp(Q,min=1e-12))),dim=axis)
  def independent_loss(U=None,shuffle=False,noobs=False):
   RGB=obs['RGB_shuffle_cost'] if shuffle else obs['RGB_cost'];aff=obs['edge_shuffle_affinity'] if shuffle else obs['edge_affinity'];terms=[];states=[];fullz=None
   for i,ix in enumerate(views):
    prob,z,A,S,h=forward(ix,U);states.append(S)
    if i==0:fullz=z
    S0,mass0,gscale,keep,edge=source[i];have=obs['allowed'][:,ix]&obs['point_RGB_known'][ix][None];scale=torch.sqrt(obs['RGB_cost'][:,ix][have].square().mean()).clamp_min(.1) if have.any() else S.new_tensor(.1);colorloss=(S[:-1]*RGB[:,ix]).sum()/max(int(obs['point_RGB_known'][ix].sum()),1)/scale;geoloss=(S[:-1]*obs['geometry_cost'][:,ix]).sum(0).mean();graphloss=(aff[keep]*(S[:,edge[:,0]]-S[:,edge[:,1]]).square().sum(0)).mean()/gscale
    pri=relative_entropy(S,S0,0).mean();mass=relative_entropy(S.mean(1),mass0,0);terms.append(pri+mass+(0. if noobs else colorloss+.25*geoloss+.1*graphloss))
   x=states[1][:,shared[0]];y=states[2][:,shared[1]];mean=(x+y)/2;stable=(relative_entropy(x,mean,0).mean()+relative_entropy(y,mean,0).mean())/2;keep=((fullz-fullz[baseq])-(z0-z0[baseq])).square().mean()
   if U is not None:delta=(U-U.mean(1,keepdim=True)).square().mean()
   else:
    def logits(weight):
     qq=F.linear(qinput,weight[:288],module.in_proj_bias[:288]).transpose(0,1).reshape(1,256,8,36).transpose(1,2);kk=F.linear(kinput,weight[288:576],module.in_proj_bias[288:576]).transpose(0,1).reshape(1,1024,8,36).transpose(1,2);return (qq/math.sqrt(36))@kk.transpose(-1,-2)
    d=logits(module.in_proj_weight)-logits(module.parametrizations.in_proj_weight.original);delta=(d-d.mean(-1,keepdim=True)).square().mean()
   return sum(terms)/3+.25*stable+keep+.01*delta
  record=read(ROOT/p['main_root']/'cases_PRIVATE'/(sid+'.json'))['statistics']['trajectories'];strongrec=read(ROOT/p['strong_root']/'cases_PRIVATE'/(sid+'.json'))['details']
  with torch.no_grad():
   for j,arm in enumerate(arms):
    adapter.set(source_values);U=None
    if arm.startswith('qk'):
     adapter.set({n:tensor(fit[arm+'__'+n]) for n,_ in adapter.named_parameters()})
    elif arm.startswith('free'):U=tensor(fit[arm])
    elif arm=='zero_step':U=torch.zeros((256,1024),device='cuda')
    prob,z,A,S,h=forward(views[0],U);err=float(np.max(np.abs(prob.cpu().numpy()-expected[j])));score_error=max(score_error,err);assert err==0. and int(prob.argmax())==int(expected_q[j]),(arm,err);replayed+=1
    if arm.endswith('_3') or arm.endswith('_20'):
     actual=float(independent_loss(U,'RGBshuffle' in arm,'noobs' in arm));saved=strongrec[arm]['final_loss'] if arm.endswith('_20') else record[arm.rsplit('_',1)[0]]['final_loss'];le=abs(actual-saved);assert le<1e-5,(arm,actual,saved);loss_error=max(loss_error,le)
    assert torch.equal(module.in_proj_weight[576:],module.parametrizations.in_proj_weight.original[576:])
  gradients=[]
  for shuffle in [False,True]:
   adapter.set(source_values);model.zero_grad(set_to_none=True);loss=independent_loss(None,shuffle,False);loss.backward();g=torch.cat([v.grad.reshape(-1) for _,v in adapter.named_parameters()]);assert torch.isfinite(g).all();gradients.append(g.detach().clone());all_A_zero&=bool(torch.count_nonzero(adapter.A_q.grad)==0 and torch.count_nonzero(adapter.A_k.grad)==0);gradient_scope&=all(x.grad is None for n,x in model.named_parameters() if '.parametrizations.in_proj_weight.0.' not in n)
  gradient_rows.append({'platform':row['platform'],'QK_source_gradient_norm':float(gradients[0].norm()),'shuffled_gradient_norm':float(gradients[1].norm()),'true_minus_shuffled_gradient_norm':float((gradients[0]-gradients[1]).norm())});model.zero_grad(set_to_none=True);adapter.set(source_values)
  for handle in handles:handle.remove()
 assert gradient_scope and all_A_zero
 parametrize.remove_parametrizations(module,'in_proj_weight',leave_parametrized=False);model.requires_grad_(False);assert digest_order(model,order)==original_fp
 result={'state':'PASS_R54_INDEPENDENT_OFFICIAL_DECODER_REPLAY_AND_OBJECTIVE_GRADIENT_AUDIT','hash_selected_observations':len(selected),'rows_per_platform':2,'readouts_replayed':replayed,'maximum_score_error':score_error,'maximum_independent_objective_error':loss_error,'only_QK_LoRA_gradient_scope':gradient_scope,'zero_B_initial_A_gradients_zero_expected':all_A_zero,'gradient_comparisons':gradient_rows,'source_model_restored_byteexact':True,'new_optimizer_steps':0,'new_full_grounder_or_backbone_forwards':0,'new_GT_reads':0,'prediction_seal_sha256':sha(ROOT/p['prediction_root']/'SEALED.json'),'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};out=ROOT/'research/round54/method_audit_attempt001';assert not out.exists();out.mkdir();(out/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
