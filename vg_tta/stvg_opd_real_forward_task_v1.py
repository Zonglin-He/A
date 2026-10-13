"""Complete live OPD proposal selection; immutable likelihood and fresh Adam per query."""
import inspect,time
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import detached
from vg_tta.stvg_opd_forward_candidate_update_v1 import candidate_states,metrics,choose
ARMS=['A','SEL','HALF','LN']

class Policy:
 def __init__(self,recorder,arm,visited):
  assert arm in ARMS
  self.arm=arm;self.visited=visited;self.rows=[];self.originals=[];self.entering=None
  for module in [recorder.core,recorder.component]:
   compiled=None
   for c in recorder.installed[module]['fit'].__closure__ or []:
    v=c.cell_contents
    if inspect.isfunction(v) and v.__name__ in ['fit','component_fit']:compiled=inspect.unwrap(v)
   assert compiled is not None
   ns=compiled.__globals__;old=ns['optimizer_step'];self.originals.append((ns,old));ns['optimizer_step']=self.wrap(old)
 def wrap(self,original):
  def step(opt,loss,params,informative):
   loc=inspect.currentframe().f_back.f_locals;rp=loc['rp'];k=loc['k'];boxes=loc['boxes'];positions=loc['positions'];targets=loc['evidence']
   if self.entering is None:self.entering=boxes.detach().clone()
   before=detached(rp.state(),'cpu');adamb=detached(opt.state_dict(),'cpu')
   update=original(opt,loss,params,informative);after=detached(rp.state(),'cpu');adama=detached(opt.state_dict(),'cpu')
   item=dict(round=k,Adam_before=adamb,Adam_after=adama)
   if update is not None and self.arm!='A':
    cs=candidate_states(before,after);scores={};outputs={};U=[i for i in range(len(boxes)) if i not in set(self.visited)]
    choice={'HALF':'Half','LN':'LN'}.get(self.arm);feasible=[];reason='fixed_proposal_mask';elapsed=0.
    if self.arm=='SEL':
     torch.cuda.synchronize();tick=time.perf_counter()
     for name,st in cs.items():
      rp.restore(st)
      with torch.no_grad():b=rp.values()['boxes'].detach().clone()
      outputs[name]=b.cpu();scores[name]=metrics(b,boxes.detach(),self.entering,positions,targets,U)
     choice,feasible,reason=choose(scores);torch.cuda.synchronize();elapsed=time.perf_counter()-tick
    rp.restore(cs[choice]);raw=torch.cat([(p.detach().cpu()-before[n]).flatten() for n,p in rp.named])
    update={**update,'proposal_raw':update['raw'],'raw':raw}
    item['policy']=dict(before_state=before,proposal_state=after,selected_state=cs[choice],selected=choice,
     feasible=feasible,reason=reason,scores=scores,outputs=outputs,current_boxes=boxes.detach().cpu(),
     entering_boxes=self.entering.cpu(),positions=positions,unvisited=U,targets=targets.detach().cpu(),
     candidate_seconds=elapsed,actual_candidate_forwards=4 if self.arm=='SEL' else 0,
     moments_advanced_for_all1792=True,next_gradient_uses_committed_state=True)
   self.rows.append(item);return update
  return step
 def close(self):
  for ns,old in self.originals:ns['optimizer_step']=old

def objective_audit(fit,expert):
 """Reuse all original native action/reward/likelihood/chart checks; audit proposal before masking."""
 import vg_tta.stvg_opd_p4_joint_precision001 as p4
 import vg_tta.decota_spatial_opd_inline_gradient_precision004 as inline
 from vg_tta.decota_spatial_opd_action_readback_precision005 import audit as actions
 source=inspect.getsource(p4.inline_audit)
 old="raw=u['raw'].numpy();error=float(abs(delta-raw).max())"
 assert source.count(old)==1
 source=source.replace(old,"raw=u.get('proposal_raw',u['raw']).numpy();error=float(abs(delta-raw).max())")
 source=source.replace("assert np.array_equal(nxt-a,raw)","assert np.array_equal(nxt-a,u['raw'].numpy())")
 ns=dict(p4.__dict__);exec(compile(source,'<declared_forward_policy_proposal_audit>','exec'),ns)
 old_audit=inline.audit
 def checked(f,e,unused=None):return ns['inline_audit'](f,e,lambda *a:None)
 inline.audit=checked
 try:r=actions(fit,expert)
 finally:inline.audit=old_audit
 r['original_component_absolute_audit_passed']=None;r['original_component_failed_line']=None
 r['scope']='original native likelihood/action/reward/gradient/chart and Adam proposal; application independently checked by policy math'
 return r

def math_audit(f):
 from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import mathematical_readback
 from scripts.audit_stvg_opd_forward_candidate_v1 import iou
 m=mathematical_readback(f);n=0;choices={}
 for k,(h,row) in enumerate(zip(f['path'],f['geometry']['rounds'])):
  if 'policy' not in row:continue
  p=row['policy'];before=h['state'];proposal=p['proposal_state'];nxt=f['path'][k+1]['state']
  assert all(torch.equal(before[a],p['before_state'][a]) for a in before)
  cs=candidate_states(before,proposal);assert all(torch.equal(cs[p['selected']][a],nxt[a]) for a in before)
  assert all(torch.equal(nxt[a],p['selected_state'][a]) for a in before)
  raw=torch.cat([(proposal[a]-before[a]).flatten() for a in before]);assert torch.equal(raw,h['update']['proposal_raw'])
  n+=1792*3
  if f['policy_arm']=='SEL':
   cur=p['current_boxes'].numpy().astype(float);ent=p['entering_boxes'].numpy().astype(float);pos=p['positions'];U=p['unvisited'];targets=p['targets'].numpy().astype(float);ss={}
   for a,b in p['outputs'].items():
    b=b.numpy().astype(float);G=float((iou(b[pos],targets)-iou(cur[pos],targets)).mean()) if pos else None
    D=float(np.square(b[U]-ent[U]).sum()/len(U)) if U else None;ss[a]={'G':G,'D':D}
    for key,val in ss[a].items():assert val==p['scores'][a][key] if val is None else abs(val-p['scores'][a][key])<2e-12
   reason='empty_observed' if not pos else 'nonpositive_original_gain' if ss['A']['G']<=0 else 'empty_unobserved' if not U else 'positive_original_gain'
   feasible=[] if reason!='positive_original_gain' else [a for a in ['A','LN','Query','Half'] if ss[a]['G']>=.9*ss['A']['G']]
   choice='A' if not feasible else min(feasible,key=lambda a:(ss[a]['D'],['A','LN','Query','Half'].index(a)))
   assert choice==p['selected'] and feasible==p['feasible'] and reason==p['reason'];n+=11
   assert torch.equal(p['outputs'][choice],f['path'][k+1]['boxes'])
  choices[p['selected']]=choices.get(p['selected'],0)+1
 return dict(status='pass',Adam=m,policy_scalar_checks=n,choices=choices,full_decoder_Jacobian_independently_replayed=False,GT_read=False)

def contracts():
 from vg_tta.stvg_opd_forward_candidate_update_v1 import contracts as old
 c=old();p=torch.nn.Parameter(torch.ones(1792));opt=torch.optim.Adam([p],lr=.01)
 for k in range(2):
  prior=p.detach().clone();p.grad=torch.ones_like(p);opt.step();proposal=p.detach().clone()
  with torch.no_grad():p.copy_(prior+.5*(proposal-prior))
  assert int(opt.state[p]['step'])==k+1 and torch.isfinite(p).all()
 assert torch.equal(opt.state[p]['exp_avg'],torch.full_like(p,.19))
 return dict(c,count=7,scope='synthetic fixed selection/masking/fresh Adam/current committed state and preserved moments; not actual GPU qualification')
