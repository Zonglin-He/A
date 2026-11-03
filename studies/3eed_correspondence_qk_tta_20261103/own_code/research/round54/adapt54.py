"""Same-objective free attention correspondence versus Q/K-only rank4 LoRA.

Only the last native attention, FFN and existing matching head are replayed.
The source feature bank and queries are frozen; key subsamples share raw IDs.
No V/P/R scores, candidate GT, external model or geometry regression is used.
"""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round51'))
from model51 import load_source,fingerprint,PREFIX_KEYS,native_matching
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def fingerprint_in_order(model,names):
 state=model.state_dict();assert set(state)==set(names);digest=hashlib.sha256()
 for name in names:
  value=state[name].detach().cpu().contiguous();digest.update(name.encode()+b'\0');digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
 return digest.hexdigest()

class PackedQK(nn.Module):
 def __init__(self,weight):
  super().__init__();assert weight.shape==(864,288)
  with torch.random.fork_rng(devices=[weight.device.index] if weight.is_cuda else []):
   torch.manual_seed(5400)
   self.A_q=nn.Parameter(weight.new_empty(4,288));self.B_q=nn.Parameter(weight.new_zeros(288,4));self.A_k=nn.Parameter(weight.new_empty(4,288));self.B_k=nn.Parameter(weight.new_zeros(288,4));nn.init.kaiming_uniform_(self.A_q,a=math.sqrt(5));nn.init.kaiming_uniform_(self.A_k,a=math.sqrt(5))
 def forward(self,original):return original+torch.cat([self.B_q@self.A_q,self.B_k@self.A_k,torch.zeros_like(original[:288])],dim=0)

class Interface:
 def __init__(self,model):
  self.model=model;model.eval().requires_grad_(False);self.module=model.decoder[5].cross_v;self.adapter=PackedQK(self.module.in_proj_weight);parametrize.register_parametrization(self.module,'in_proj_weight',self.adapter);assert sum(x.numel() for x in self.adapter.parameters())==4608
  self.initial={n:x.detach().clone() for n,x in self.adapter.named_parameters()};self.versions={n:x._version for n,x in model.named_parameters() if not x.requires_grad};self.V=self.module.parametrizations.in_proj_weight.original[576:].detach().clone()
 def reset(self,trainable):
  with torch.no_grad():
   for n,x in self.adapter.named_parameters():x.copy_(self.initial[n]);x.grad=None;x.requires_grad_(trainable)
  self.model.eval();assert torch.equal(self.module.in_proj_weight,self.module.parametrizations.in_proj_weight.original)
 def check(self):
  for n,x in self.model.named_parameters():
   if n in self.versions:assert not x.requires_grad and x.grad is None and x._version==self.versions[n],n
  assert torch.equal(self.module.in_proj_weight[576:],self.V) and all(not m.training for m in self.model.modules())
 def delta(self):return float(torch.sqrt(sum((x.detach()-self.initial[n]).square().sum() for n,x in self.adapter.named_parameters())).item())

def make_context(model,arrays):
 device=next(model.parameters()).device;T=lambda key:torch.as_tensor(arrays[key],device=device);cache={};handles=[]
 def cross_pre(m,args,kwargs):
  for k in ['query','key','value']:cache[k]=kwargs[k].detach().clone()
 def norm_hook(m,args,value):cache['residual']=value.detach().clone()
 handles=[model.decoder[5].cross_v.register_forward_pre_hook(cross_pre,with_kwargs=True),model.decoder[5].norm_l.register_forward_hook(norm_hook)]
 try:
  with torch.no_grad():h=model.decoder[5](*[T(k) if k in arrays else None for k in PREFIX_KEYS],detected_feats=None,detected_mask=None)
 finally:
  for handle in handles:handle.remove()
 assert torch.equal(h[0],T('h')),'Original last-decoder feature bridge'
 cache.update(tokens=T('tokens')[None],mask=T('token_mask').bool(),original_scores=T('scores'),original_z=T('logodds'),original_h=T('h'),device=device)
 return cache

def attention_forward(model,context,indices,U=None):
 mask=None if U is None else U[:,indices]
 output,A=model.decoder[5].cross_v(query=context['query'],key=context['key'][indices],value=context['value'][indices],attn_mask=mask,key_padding_mask=None,need_weights=True)
 q=model.decoder[5].norm_v(context['residual']+output);q=model.decoder[5].norm2(q+model.decoder[5].ffn(q));h=q.transpose(0,1).contiguous()
 query=F.normalize(model.contrastive_align_projection_image(h),p=2,dim=-1);p,z=native_matching(query,context['tokens'],context['mask'])
 return {'A':A[0],'scores':p,'z':z,'h':h[0]}

def assignment(A,obs,indices):
 foreground=A*len(indices)*obs['query_prior'][:,None]*obs['allowed'][:,indices];unknown=foreground.new_full((1,len(indices)),.2);mass=torch.cat([foreground,unknown],0);return mass/mass.sum(0,keepdim=True).clamp_min(1e-20)

def kl(p,q,dim):return (p*(p.clamp_min(1e-12).log()-q.clamp_min(1e-12).log())).sum(dim)

def prepare_views(obs,device):
 views=[torch.arange(1024,device=device),obs['view_A'].long(),obs['view_B'].long()];result=[]
 for indices in views:
  inverse=torch.full((1024,),-1,device=device,dtype=torch.long);inverse[indices]=torch.arange(len(indices),device=device);edges=obs['edge_pairs'].long();keep=(inverse[edges[:,0]]>=0)&(inverse[edges[:,1]]>=0);local=inverse[edges[keep]]
  result.append({'indices':indices,'edges':local,'edge_keep':keep})
 shared=np.intersect1d(obs['view_A'].detach().cpu().numpy(),obs['view_B'].detach().cpu().numpy());a=torch.searchsorted(views[1],torch.as_tensor(shared,device=device));b=torch.searchsorted(views[2],torch.as_tensor(shared,device=device))
 return result,(a,b)

def source_states(model,ctx,obs,views):
 with torch.no_grad():
  result=[]
  for view in views:
   out=attention_forward(model,ctx,view['indices']);S=assignment(out['A'],obs,view['indices']);mass=S.mean(1);edges=view['edges'];raw_graph=(obs['edge_affinity'][view['edge_keep']]*(S[:,edges[:,0]]-S[:,edges[:,1]]).square().sum(0)).mean() if len(edges) else S.new_zeros(())
   result.append({'A':out['A'].detach(),'S':S.detach(),'mass':mass.detach(),'graph_scale':max(float(raw_graph),.01)})
  assert torch.equal(out['h'],out['h'])
 return result

def actual_logit_delta(module,ctx,U):
 if U is not None:return (U-U.mean(1,keepdim=True)).square().mean()
 W=module.in_proj_weight;W0=module.parametrizations.in_proj_weight.original;b=module.in_proj_bias;H=module.num_heads;D=288//H
 def score(weight):
  q=F.linear(ctx['query'],weight[:288],b[:288]).transpose(0,1).reshape(1,256,H,D).transpose(1,2);k=F.linear(ctx['key'],weight[288:576],b[288:576]).transpose(0,1).reshape(1,1024,H,D).transpose(1,2);return (q/math.sqrt(D))@k.transpose(-1,-2)
 delta=score(W)-score(W0);return (delta-delta.mean(-1,keepdim=True)).square().mean()

def objective(model,ctx,obs,views,shared,source,U=None,shuffle=False,no_observations=False):
 terms={k:ctx['query'].new_zeros(()) for k in ['RGB','geometry','graph','prior','mass','stability','keep','logit_keep']};states=[];full=None
 color=obs['RGB_shuffle_cost'] if shuffle else obs['RGB_cost'];edge_weights=obs['edge_shuffle_affinity'] if shuffle else obs['edge_affinity']
 for i,view in enumerate(views):
  ix=view['indices'];out=attention_forward(model,ctx,ix,U);S=assignment(out['A'],obs,ix);states.append(S)
  if i==0:full=out
  rgb_norm=max(int(obs['point_RGB_known'][ix].sum()),1);known_allowed=obs['allowed'][:,ix]&obs['point_RGB_known'][ix][None];rgb_scale=torch.sqrt(obs['RGB_cost'][:,ix][known_allowed].square().mean()).detach().clamp_min(.1) if known_allowed.any() else S.new_tensor(.1)
  terms['RGB']+=(S[:-1]*color[:,ix]).sum()/rgb_norm/rgb_scale
  terms['geometry']+=(S[:-1]*obs['geometry_cost'][:,ix]).sum(0).mean()
  edges=view['edges']
  if len(edges):terms['graph']+=(edge_weights[view['edge_keep']]*(S[:,edges[:,0]]-S[:,edges[:,1]]).square().sum(0)).mean()/source[i]['graph_scale']
  terms['prior']+=kl(S,source[i]['S'],0).mean();terms['mass']+=kl(S.mean(1),source[i]['mass'],0)
 for key in ['RGB','geometry','graph','prior','mass']:terms[key]/=len(views)
 left,right=shared;mean=(states[1][:,left]+states[2][:,right])/2;terms['stability']=(kl(states[1][:,left],mean,0).mean()+kl(states[2][:,right],mean,0).mean())/2
 q0=int(ctx['original_scores'].argmax());d=full['z']-full['z'][q0];d0=ctx['original_z']-ctx['original_z'][q0];terms['keep']=(d-d0).square().mean();terms['logit_keep']=actual_logit_delta(model.decoder[5].cross_v,ctx,U)
 total=terms['prior']+terms['mass']+.25*terms['stability']+terms['keep']+.01*terms['logit_keep']
 if not no_observations:total=total+terms['RGB']+.25*terms['geometry']+.1*terms['graph']
 return total,terms,full,states[0]

def diagnostics(terms,S,sourceS,ctx,current,parameter_delta,gradient_norm):
 mass=S.detach().mean(1);base=sourceS.mean(1);changed=abs(float(mass[-1]-base[-1]));collapse=bool((mass[-1]>.95 and mass[-1]-base[-1]>.1) or (mass[:-1].max()>.95 and mass[:-1].max()-base[:-1].max()>.1))
 return {'loss_terms':{k:float(v.detach()) for k,v in terms.items()},'unknown_mass':float(mass[-1]),'source_unknown_mass':float(base[-1]),'max_query_mass':float(mass[:-1].max()),'source_max_query_mass':float(base[:-1].max()),'new_mass_collapse_flag':collapse,'assignment_KL':float(kl(S.detach(),sourceS,0).mean()),'score_max_abs_change':float((current['scores'].detach()-ctx['original_scores']).abs().max()),'parameter_delta_L2':parameter_delta,'gradient_norm_before_clipping':gradient_norm}

def run_case(model,interface,arrays,obs_np,plan):
 device=next(model.parameters()).device;interface.reset(False);ctx=make_context(model,arrays);obs={k:torch.as_tensor(v,device=device,dtype=torch.float32 if v.dtype.kind=='f' else None) for k,v in obs_np.items()};obs['allowed']=obs['allowed'].bool();obs['point_RGB_known']=obs['point_RGB_known'].bool();obs['edge_pairs']=obs['edge_pairs'].long();views,shared=prepare_views(obs,device)
 with torch.no_grad():
  original=attention_forward(model,ctx,views[0]['indices']);assert torch.equal(original['h'],ctx['original_h']) and torch.equal(original['scores'],ctx['original_scores']);zero=attention_forward(model,ctx,views[0]['indices'],torch.zeros((256,1024),device=device));zero_error=float((zero['scores']-ctx['original_scores']).abs().max());assert zero_error<1e-6 and int(zero['scores'].argmax())==int(original['scores'].argmax())
 source=source_states(model,ctx,obs,views);scores={'source':arrays['scores'].copy(),'zero_step':zero['scores'].cpu().numpy()};stats={'source_bridge_bitexact':True,'direct_zero_scores_bitexact':torch.equal(zero['scores'],ctx['original_scores']),'direct_zero_score_max_error':zero_error,'direct_zero_Top1_preserved':True,'trajectories':{}};fitted={}
 trajectories=[('free',False,False),('qk',False,False),('free',True,False),('qk',True,False),('qk',False,True)]
 for kind,shuffle,noobs in trajectories:
  interface.reset(kind=='qk');U=nn.Parameter(torch.zeros((256,1024),device=device)) if kind=='free' else None;parameters=[U] if U is not None else list(interface.adapter.parameters());optimizer=torch.optim.Adam(parameters,lr=plan['learning_rate'],betas=(.9,.999),eps=1e-8);name=kind+('_RGBshuffle' if shuffle else '')+('_noobs' if noobs else '');stepstats=[]
  with torch.no_grad():initial_loss,_,_,_=objective(model,ctx,obs,views,shared,source,U,shuffle,noobs)
  for step in range(1,4):
   optimizer.zero_grad(set_to_none=True);loss,terms,_,S=objective(model,ctx,obs,views,shared,source,U,shuffle,noobs);assert torch.isfinite(loss);loss.backward();grad=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in parameters if p.grad is not None)));torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();interface.check()
   with torch.no_grad():current=attention_forward(model,ctx,views[0]['indices'],U);S_after=assignment(current['A'],obs,views[0]['indices']);delta=float(U.detach().norm()) if U is not None else interface.delta();record=diagnostics(terms,S_after,source[0]['S'],ctx,current,delta,grad);record.update(step=step,loss_before_step=float(loss.detach()));stepstats.append(record)
   if step in [1,3]:
    arm=name+'_'+str(step)
    if not noobs:scores[arm]=current['scores'].cpu().numpy().copy()
    elif step==3:scores[arm]=current['scores'].cpu().numpy().copy()
    if not noobs or step==3:fitted[arm]=U.detach().cpu().numpy().copy() if U is not None else {n:x.detach().cpu().numpy().copy() for n,x in interface.adapter.named_parameters()}
  with torch.no_grad():last_loss,last_terms,last_current,last_S=objective(model,ctx,obs,views,shared,source,U,shuffle,noobs)
  stats['trajectories'][name]={'initial_loss':float(initial_loss),'final_loss':float(last_loss),'steps':stepstats,'final_terms':{k:float(v) for k,v in last_terms.items()},'trainable_parameters':262144 if kind=='free' else 4608,'no_network_parameters_updated':kind=='free'}
  optimizer.zero_grad(set_to_none=True);del optimizer,U
 interface.reset(False);interface.check();return scores,stats,fitted

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--limit',type=int);ap.add_argument('--resume',action='store_true');a=ap.parse_args();p=read(a.plan)
 for rel,h in {**p['programs_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/rel)==h,rel
 rows=read(ROOT/p['manifest']);out=ROOT/p['output_root']
 if out.exists():assert a.resume and read(out/'RUN_BINDING.json')['plan_sha256'] in [sha(a.plan),p.get('pre_GT_technical_predecessor_plan_sha256')] and not (out/'SEALED.json').exists()
 else:out.mkdir(parents=True);dump(out/'RUN_BINDING.json',{'plan_sha256':sha(a.plan)})
 for d in ['samples_PRIVATE','cases_PRIVATE','fitted_PRIVATE']:(out/d).mkdir(exist_ok=True)
 torch.manual_seed(0);np.random.seed(0);start=time.perf_counter();model=load_source(read(ROOT/p['source_plan']));sourcefp=fingerprint(model);source_order=list(model.state_dict());interface=Interface(model);latent=ROOT/p['latent_root'];ls=read(latent/'SEALED.json');obsroot=ROOT/p['observation_root'];os=read(obsroot/'SEALED.json');done=[];status=[];used=[]
 try:
  for row in rows:
   sid=row['id'];casefile=out/'cases_PRIVATE'/(sid+'.json');samplefile=out/'samples_PRIVATE'/(sid+'.npz')
   if casefile.exists():
    record=read(casefile);assert a.resume and record['state']=='COMPLETED' and sha(samplefile)==record['sample_sha256'];done.append(sid);status.append(record['status_row']);continue
   rel='samples_PRIVATE/'+sid+'.npz';assert sha(latent/rel)==ls['files'][rel] and sha(obsroot/rel)==os['files'][rel]
   fields=['boxes','scores','logodds','h','tokens','token_mask',*PREFIX_KEYS]
   with np.load(latent/rel,allow_pickle=False) as z:arrays={k:z[k].copy() for k in fields if k in z.files}
   with np.load(obsroot/rel,allow_pickle=False) as z:obs={k:z[k].copy() for k in ['allowed','geometry_cost','RGB_cost','RGB_shuffle_cost','edge_pairs','edge_affinity','edge_shuffle_affinity','point_RGB_known','query_prior','view_A','view_B']}
   tick=time.perf_counter();result,info,fitted=run_case(model,interface,arrays,obs,p);arms=p['arms'];assert set(result)==set(arms),set(result)^set(arms);stack=np.stack([result[arm] for arm in arms]);queries=stack.argmax(1).astype(np.int64);assert np.isfinite(stack).all()
   np.savez_compressed(samplefile,arm_names=np.asarray(arms),scores=stack,queries=queries,selected_original_boxes=arrays['boxes'][queries],source_boxes=arrays['boxes'],source_scores=arrays['scores'])
   flat={}
   for key,v in fitted.items():
    if isinstance(v,dict):
     for n,value in v.items():flat[key+'__'+n]=value
    else:flat[key]=v
   np.savez_compressed(out/'fitted_PRIVATE'/(sid+'.npz'),**flat)
   statuses={}
   for arm in arms:
    fixed=arm in ['source','zero_step'] or arm.startswith('free');delta=0. if fixed else info['trajectories'][arm.rsplit('_',1)[0]]['steps'][int(arm.rsplit('_',1)[1])-1]['parameter_delta_L2']
    statuses[arm]={'unknown':bool(not obs['point_RGB_known'].any() or not obs['allowed'].any()) if arm not in ['source','zero_step','qk_noobs_3'] else False,'fallback':False,'no_parameter_update':bool(fixed or delta==0.)}
   st={'id':sid,'arm_statuses':statuses}
   record={'state':'COMPLETED','id_PRIVATE':sid,'platform':row['platform'],'sample_sha256':sha(samplefile),'observations_sha256':sha(obsroot/rel),'fitted_sha256':sha(out/'fitted_PRIVATE'/(sid+'.npz')),'statistics':info,'status_row':st,'seconds':time.perf_counter()-tick};dump(casefile,record);done.append(sid);status.append(st);used.append(record['seconds'])
   dump(out/'PROGRESS.json',{'n':len(done),'total':336,'seconds':time.perf_counter()-start})
   if len(done)%8==0:print('ADAPT54',len(done),336,'last_seconds',round(record['seconds'],3),flush=True)
   if a.limit is not None and len(used)>=a.limit:break
  interface.reset(False);interface.check();parametrize.remove_parametrizations(interface.module,'in_proj_weight',leave_parametrized=False);model.requires_grad_(False);assert fingerprint_in_order(model,source_order)==sourcefp,'Every tensor must match in original source order'
  if len(done)==336:
   dump(out/'READOUT_STATUS_PRIVATE.json',status);counts={'n':336,'arms':len(p['arms']),'optimizer_steps':336*15,'source_model_unchanged_after_all_resets':True,'source_fingerprint':sourcefp,'seconds_this_process':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'new_full_grounder_forwards':0,'new_source_backbone_forwards':0,'source_checkpoint_loads_this_process':1};dump(out/'COSTS.json',counts);dump(out/'SEALED.json',{'state':'R54_CORRESPONDENCE_AND_QK_TTA_PREDICTIONS_NO_GT',**counts,'arm_names':p['arms'],'plan_sha256':sha(a.plan),'files':{f.relative_to(out).as_posix():sha(f) for f in sorted(out.rglob('*')) if f.is_file()}});print(json.dumps(counts),flush=True)
  else:dump(out/('ENGINEERING_STOP_'+str(len(done))+'.json'),{'n':len(done),'no_GT_read':True,'seconds_this_process':time.perf_counter()-start,'model_source_restored':True,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())});print(json.dumps({'state':'ENGINEERING_STOP_VALID_OUTPUTS_REUSABLE','n':len(done),'seconds':time.perf_counter()-start}),flush=True)
 except BaseException as e:dump(out/('FAILED_'+str(time.time_ns())+'.json'),{'type':type(e).__name__,'error':str(e),'n':len(done),'no_GT_scored':True,'automatic_retry':False});raise
if __name__=='__main__':main()
