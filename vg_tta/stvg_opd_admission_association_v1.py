"""Fixed admission/association intervention. Original OPD update, explicit teachers/noise."""
import ast,copy,hashlib,inspect,itertools,types
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import detached
ARMS=['H_Top1','H_Path','R_Top1','R_Path','R_Add','R_Shuffle']

def iou(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 lo=np.maximum(a[...,:2]-a[...,2:]/2,b[...,:2]-b[...,2:]/2)
 hi=np.minimum(a[...,:2]+a[...,2:]/2,b[...,:2]+b[...,2:]/2)
 z=np.maximum(hi-lo,0).prod(-1)
 return z/np.maximum(a[...,2:].prod(-1)+b[...,2:].prod(-1)-z,1e-7)

def decompose(scores,edges,key):
 u=[np.asarray(x,float).copy() for x in scores];cc=[];sh=[];perms=[]
 for j,e in enumerate(edges):
  e=np.asarray(e,float);r=e.mean(1);c=e.mean(0);g=e.mean();C=e-r[:,None]-c[None,:]+g
  u[j]+=r-g/2;u[j+1]+=c-g/2
  n=e.shape[1];seed=int(hashlib.sha256(f'admission-association-shuffle-v1|{key}|{j}'.encode()).hexdigest()[:15],16)
  q=np.roll(np.arange(n),1+seed%(n-1)) if n>1 else np.arange(n)
  cc.append(C);sh.append(C[:,q]);perms.append(q.tolist())
 return u,cc,sh,perms

def choose(u,edges):
 if not u:return []
 paths=list(itertools.product(*[range(len(x)) for x in u]));assert len(paths)<=81
 values=[sum(u[j][v] for j,v in enumerate(p))+sum(e[p[j],p[j+1]] for j,e in enumerate(edges)) for p in paths]
 return list(paths[int(np.argmax(values))])

def select(pools,key):
 out={};details={}
 for admission in ['H','R']:
  pp=[p for p in pools if p['accepted'] or (admission=='R' and p['reason']=='ambiguous_distinct_instances' and max(p['scores'],default=-1)>=.35)]
  pp=sorted(pp,key=lambda p:p['frame_id']);assert len(pp)<=4 and [p['position'] for p in pp]==sorted(p['position'] for p in pp)
  scores=[np.asarray(p['scores'],float) for p in pp]
  edges=[iou(np.asarray(a['boxes'])[:,None,:],np.asarray(b['boxes'])[None,:,:]) for a,b in zip(pp,pp[1:])]
  u,C,sh,perms=decompose(scores,edges,key)
  choices={'Top1':[int(np.argmax(s)) for s in scores], 'Path':choose(scores,edges), 'Add':[int(np.argmax(s)) for s in u], 'Shuffle':choose(u,sh)}
  for path in itertools.product(*[range(len(s)) for s in scores]):
   raw=sum(s[j] for s,j in zip(scores,path))+sum(e[path[j],path[j+1]] for j,e in enumerate(edges))
   centered=sum(s[j] for s,j in zip(u,path))+sum(e[path[j],path[j+1]] for j,e in enumerate(C))
   assert abs(raw-centered)<1e-12
  for kind in (['Top1','Path'] if admission=='H' else ['Top1','Path','Add','Shuffle']):
   rows=[]
   for p,k in zip(pp,choices[kind]):
    rows.append(dict(position=p['position'],frame_id=p['frame_id'],candidate_index=k,candidate_id=p['candidate_ids'][k],
     box=p['boxes'][k],score=p['scores'][k],old_admitted=p['accepted'],context_active=p['context_active']))
   out[admission+'_'+kind]=dict(GT_assisted=False,rows=rows)
  details[admission]=dict(unary=[v.tolist() for v in u],edges=[e.tolist() for e in edges],centered=[e.tolist() for e in C],
   shuffled=[e.tolist() for e in sh],permutations=perms,choices=choices,
   degenerate_edges=sum(bool(np.max(np.abs(e))<=1e-12) for e in C),singleton_edges=sum(min(e.shape)==1 for e in C))
 return dict(arms=out,pools=pools,decomposition=details)

def pools_from_expert(expert,ids):
 pp=[]
 for (_,position),p in sorted(expert['observations'].items()):
  probe=p['probe'];boxes=probe['boxes'].detach().cpu().tolist();scores=list(map(float,probe['target_scores']))
  assert len(boxes)==len(scores)<=3 and all(b[2]>0 and b[3]>0 for b in boxes)
  pp.append(dict(position=int(position),frame_id=int(ids[position]),boxes=boxes,scores=scores,
   candidate_ids=list(map(int,probe['candidate_ids'])),accepted=bool(probe['accepted']),reason=probe['reason'],context_active=bool(p['receipt']['context_active'])))
 return pp

def fixed_support(expert,before,fixed):
 from vg_tta.decota_optimizer_posterior_r1_v1 import Energy
 if fixed is None:return Energy(expert,before,'top1')
 all_=Energy(expert,before,'all');lookup={int(f[0]):(f,m) for f,m in zip(all_.frames,all_.frame_metadata)};frames=[];meta=[]
 for row in fixed['rows']:
  (pos,boxes,logw),m=lookup[row['position']];k=row['candidate_index'];m=copy.deepcopy(m)
  assert torch.equal(boxes[k].cpu(),torch.tensor(row['box'],dtype=torch.float32)) and m['scores'][k]==row['score']
  assert m['old_admitted']==row['old_admitted'] and m['context_active']==row['context_active']
  frames.append((pos,boxes[k:k+1],logw.new_zeros(1)));m.update(proposals=1,weights=[1.],scores=[m['scores'][k]],evidence_entropy=0.);meta.append(m)
 return types.SimpleNamespace(frames=frames,frame_metadata=meta,empty=not frames)

def half_noise(key,k,positions,h_positions,ids):
 seed=int(hashlib.sha256(f'spatial-opd-v1|{key}|{k}'.encode()).hexdigest()[:15],16)
 original=torch.randn(len(h_positions),16,4,generator=torch.Generator().manual_seed(seed),dtype=torch.float32)
 hh={p:original[j] for j,p in enumerate(h_positions)};rows=[]
 for p in positions:
  if p in hh:rows.append(hh[p])
  else:
   s=int(hashlib.sha256(f'admission-association-extra-frame-v1|{key}|{ids[p]}|{k}'.encode()).hexdigest()[:15],16)
   rows.append(torch.randn(16,4,generator=torch.Generator().manual_seed(s),dtype=torch.float32))
 return torch.stack(rows) if rows else torch.empty(0,16,4)

class FixedInstallation:
 def __init__(self,rec,h_positions,ids):
  self.rec=rec;self.rows=[];self.h_positions=h_positions;self.ids=ids
  core=rec.core;tree=ast.parse(open(core.__file__).read());node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit'))
  node.args.kwonlyargs += [ast.arg(arg='fixed_evidence'),ast.arg(arg='noise_override')];node.args.kw_defaults += [ast.Constant(None),ast.Constant(None)]
  for n in ast.walk(node):
   if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='support' for t in n.targets):n.value=ast.parse('_fixed_support(expert,before,fixed_evidence)',mode='eval').body
   if isinstance(n,ast.Assert) and ast.unparse(n.test)=='error < 2e-05':n.test=ast.parse('_P4_scope_guard(mu,rr,gmu,analytic,error,cfg,k)',mode='eval').body
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='rollout':n.keywords += [ast.keyword(arg='_noise_override',value=ast.parse('noise_override(key,k,positions) if noise_override is not None else None',mode='eval').body)]
   if isinstance(n,ast.keyword) and n.arg=='GT_used':n.value=ast.parse("bool(fixed_evidence and fixed_evidence['GT_assisted'])",mode='eval').body
  roll=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rollout'))
  roll.args.kwonlyargs.append(ast.arg(arg='_noise_override'));roll.args.kw_defaults.append(ast.Constant(None))
  for n in ast.walk(roll):
   if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='noise' and 'torch.randn' in ast.unparse(n.value):n.value=ast.IfExp(test=ast.parse('_noise_override is not None',mode='eval').body,body=ast.parse('_noise_override.to(mean)',mode='eval').body,orelse=n.value)
  ns=dict(core.__dict__);ns.update(_fixed_support=fixed_support,_P4_scope_guard=rec.guard,mean_coordinates=rec.mean)
  exec(compile(ast.fix_missing_locations(ast.Module(body=[roll,node],type_ignores=[])),core.__file__,'exec'),ns)
  original=ns['optimizer_step'];ns['optimizer_step']=self.trace(original)
  self.fit=rec.wrap(ns['fit'])
 def noise(self,key,k,positions):return half_noise(key,k,positions,self.h_positions,self.ids)
 def trace(self,original):
  def step(opt,loss,params,informative):
   l=inspect.currentframe().f_back.f_locals;k=l['k'];rr=l['rr'];pos=l['positions'];cfg=l['cfg'];expert=l['expert'];before=l['boxes'];gmu=l['gmu']
   from vg_tta.decota_optimizer_posterior_r1_v1 import Energy
   from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
   all_=Energy(expert,before,'all');top={int(f[0]):f[1][int(np.argmax(m['scores']))] for f,m in zip(all_.frames,all_.frame_metadata)}
   target=torch.stack([top[p] for p in pos]);actions=self.rec.contexts[-1]['actions'][-1]['actions'].to(before)
   with torch.no_grad():rw=aligned_iou(actions,target[:,None,:]);ww=torch.softmax(rw/cfg['tau'],-1)
   feedback=dict(max_reward=rr['rewards'].max(1).values.cpu(),reward_variance=rr['rewards'].var(1,unbiased=False).cpu(),
    Top1_same_samples_weight_L1=(rr['weights']-ww).abs().sum(1).cpu(),mean_gradient_norm=gmu.detach().norm(dim=1).cpu(),
    per_frame_informative=(rr['rewards']!=rr['rewards'][:,:1]).any(1).cpu(),same_samples_top1_rewards=rw.cpu(),same_samples_top1_weights=ww.cpu())
   adamb=detached(opt.state_dict(),'cpu');update=original(opt,loss,params,informative)
   self.rows.append(dict(round=k,Adam_before=adamb,Adam_after=detached(opt.state_dict(),'cpu'),feedback=feedback));return update
  return step

def objective_audit(fit,expert,fixed):
 import vg_tta.stvg_opd_p4_joint_precision001 as p4
 import vg_tta.decota_spatial_opd_inline_gradient_precision004 as inline
 from vg_tta.decota_spatial_opd_action_readback_precision005 import audit as action
 source=inspect.getsource(p4.inline_audit).replace('support=top1_support(expert)',"support=[(r['position'],r['box']) for r in _fixed['rows']]")
 source=source.replace("not fit['GT_used']","fit['GT_used']==_fixed['GT_assisted']")
 ns=dict(p4.__dict__,_fixed=fixed);exec(compile(source,'<declared_fixed_evidence_objective_audit>','exec'),ns)
 old=inline.audit;inline.audit=lambda f,e,unused=None:ns['inline_audit'](f,e,lambda *a:None)
 try:result=action(fit,expert)
 finally:inline.audit=old
 return dict(result,original_component_absolute_audit_passed=None,original_component_failed_line=None,original_component_audit='not_executed_not_applicable_to_fixed_support',GT_assisted=fixed['GT_assisted'],scope='original objective and native chart with declared fixed unit-weight original-pool supervision; no independent full decoder Jacobian')

def math_audit(fit):
 import vg_tta.stvg_opd_common_state_geometry_arithmetic004 as ar
 source=inspect.getsource(ar.mathematical_readback).replace("not fit['GT_used']","isinstance(fit['GT_used'],bool)")
 ns=dict(ar.__dict__);exec(compile(source,'<declared_Oracle_or_label_free_Adam_readback>','exec'),ns)
 return dict(ns['mathematical_readback'](fit),GT_assisted=fit['GT_used'])

def noise_audit(fit,key,h_positions,ids):
 n=0
 for k,r in enumerate(fit['rounds']):
  if 'rollout' not in r:assert fit['empty'];continue
  half=half_noise(key,k,fit['positions'],h_positions,ids);noise=torch.cat((half,-half),1);rr=r['rollout']
  assert torch.equal(rr['mean'][:,None,:]+fit['config']['sigma']*noise,rr['samples']);n+=noise.numel()
 return dict(status='pass',sample_coordinates=n,original_H_noise_preserved=True,restored_frame_noise_physical_ID_bound=True)

def contracts():
 rng=np.random.default_rng(20261013);count=0
 for k in range(1000):
  sizes=rng.integers(1,4,size=int(rng.integers(1,5)));ss=[rng.normal(size=n) for n in sizes];ee=[rng.random((a,b)) for a,b in zip(sizes,sizes[1:])]
  u,c,sh,perms=decompose(ss,ee,str(k));assert choose(ss,ee)==choose(u,c)
  for e,C,S in zip(ee,c,sh):
   assert np.max(abs(C.mean(0)))<1e-14 and np.max(abs(C.mean(1)))<1e-14 and abs(np.linalg.norm(C)-np.linalg.norm(S))<1e-14
  for p in itertools.product(*[range(n) for n in sizes]):
   a=sum(s[j] for s,j in zip(ss,p))+sum(e[p[j],p[j+1]] for j,e in enumerate(ee));b=sum(s[j] for s,j in zip(u,p))+sum(e[p[j],p[j+1]] for j,e in enumerate(c));assert abs(a-b)<1e-12
  count+=1
 assert choose([],[])==[]
 for k in range(4):
  h=half_noise('test',k,[1,4],[1,4],[2,5,8,11,14]);r=half_noise('test',k,[0,1,3,4],[1,4],[2,5,8,11,14]);assert torch.equal(h,r[[1,3]])
 from vg_tta.decota_spatial_opd_tunable_v1 import likelihood_loss
 mu=torch.zeros(2,4,requires_grad=True);samples=torch.randn(2,32,4);loss=likelihood_loss(mu,mu.detach(),samples,torch.full((2,32),1/32),.1);assert not torch.count_nonzero(torch.autograd.grad(loss,mu)[0])
 assert not torch.cuda.is_initialized()
 return dict(status='pass',random_decomposition_contracts=count,noise_alignment_contracts=4,uniform_feedback_zero_gradient=True,CPU_only=True,GT_read=False)
