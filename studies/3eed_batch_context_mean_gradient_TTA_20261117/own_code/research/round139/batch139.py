"""Batch8 mean-task episodic Up4896; no GT gradients or teacher-state import."""
from common139 import *
sys.path.insert(0,str(R/'research/round136'));from optim136 import *
from common139 import R,H,B,rd,sha,dump,check,utc
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def batchfit(base,m,S,entries,wrong,engineering):
 restore(m,S);freeze91(m);params=[dict(m.named_parameters())[k] for k in NAMES];radius=radius135(m,S);delta=torch.zeros(4896,device='cuda');xx=[];trace=[];states={};fd=[]
 for e in entries:
  z0,z,d,route,ev,groups,old=inputs136(e);assert bool(ev['eligible']);P=z['vis0'][0]
  with torch.no_grad():ref=F.normalize(m.point_adapter(P),dim=-1).detach()
  xx.append({'entry':e,'z':z,'d':d,'route':route,'ev':ev,'groups':groups,'P':P,'ref':ref,'target':torch.as_tensor(ev['wrong_target'] if wrong else ev['target'],device='cuda'),'Sg':torch.as_tensor(ev['S'],device='cuda'),'Cg':torch.as_tensor(ev['C'],device='cuda')})
 def one(x):
  lp,a,_=live135(m,base,x['z'],x['d'],x['P'],x['route']);pt=taskloss(lp,x['groups'],x['Sg'],x['Cg'],None,'partial');gm=torch.stack([lp[torch.as_tensor(g,device='cuda')].max() for g in x['groups']]);rank=(x['target']*(x['target'].clamp_min(1e-300).log()-gm[x['Cg']].log_softmax(0))).sum();keep=(F.normalize(m.point_adapter(x['P']),dim=-1)-x['ref']).square().sum(-1).mean()/2;return pt+rank+keep,pt,rank,keep,lp,a
 def evaluate():
  with torch.no_grad():return [float(one(x)[0]) for x in xx]
 for step in range(1,4):
  put123(m,S,delta)
  for p in params:p.requires_grad_(True)
  grads=[];beforevec=[]
  for x in xx:
   L,pt,rank,keep,*_=one(x);assert step!=1 or float(keep)==0.;beforevec.append(float(L.detach()));gg=torch.autograd.grad(L,params);grads.append(torch.cat([v.detach().flatten() for v in gg]))
  g=torch.stack(grads).mean(0);states['gradient'+str(step)]=g.cpu().numpy();assert torch.isfinite(g).all();before=float(np.mean(beforevec));old=delta.clone();direction=-radius*g/g.norm() if float(g.norm())>1e-12 and before>1e-12 else g*0
  for p in params:p.requires_grad_(False)
  if engineering and step==1 and float(g.norm())>1e-7:
   vals=[];eps=radius*.001
   for sign in [-1,1]:put123(m,S,old+sign*eps*g/g.norm());vals.append(float(np.mean(evaluate())))
   derivative=(vals[1]-vals[0])/(2*eps);err=abs(derivative-float(g.norm()))/float(g.norm());assert err<.25,(derivative,float(g.norm()),err);fd.append({'relative_error':err})
  accepted=False;fraction=0.;trials=[]
  if float(direction.norm())>0:
   for j in range(9):
    candidate=old+direction/(2**j);put123(m,S,candidate);v=evaluate();value=float(np.mean(v));bound=before+1e-4*float(g.double()@(candidate-old).double())+1e-12;ok=value<=bound;trials.append({'fraction':2.**(-j),'objective':value,'bound':bound,'accepted':ok})
    if ok:delta=candidate;accepted=True;fraction=2.**(-j);break
  if not accepted:delta=old
  put123(m,S,delta);after=evaluate();trace.append({'step':step,'before':before,'after':float(np.mean(after)),'per_case_before_PRIVATE':beforevec,'per_case_after_PRIVATE':after,'gradient_L2':float(g.norm()),'parameter_L2':float(delta.norm()),'accepted':accepted,'fraction':fraction,'trials':trials})
 states['delta']=delta.cpu().numpy();outputs=[]
 with torch.no_grad():
  for x in xx:
   L,pt,rank,keep,lp,a=one(x);outputs.append({'entry':x['entry'],'probability':lp.exp().cpu().numpy(),'features':{k:a[k].cpu().numpy() for k in FEATURES},'objective':float(L),'partial':float(pt),'rank':float(rank),'keep':float(keep)})
 assert {k for k,v in state(m).items() if not np.array_equal(v,S[k])}<=set(NAMES);restore(m,S);freeze91(m);return outputs,states,{'trace':trace,'radius':radius,'finite_difference':fd,'batch_size':len(xx),'scope_only_Up4896':True,'wrong':wrong,'mean_loss_not_individual_acceptance':True}
def main():
 check();assert rd(H/'experiment/BATCH_EXECUTION_LOCK.json')['enabled'];reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');parts=rd(H/'BATCH_PARTITIONS_PRIVATE.json');root=B/('engineering' if args.engineering else 'formal');root.mkdir(parents=True,exist_ok=False);index={(e['source'],e['target'],e['row']['id']):e for e in reg};outall={};n=0
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base);selected=[p for p in parts if p['source']==src]
  if args.engineering:
   q=[p for p in selected if p['order']==0 and p['n']>=2][0];selected=[q]
   for tgt in sorted({p['target'] for p in parts if p['source']==src}):
    p=next(p for p in parts if p['source']==src and p['target']==tgt);selected.append({'source':src,'target':tgt,'order':0,'n':1,'batch_PRIVATE':'singleton_'+src+'_to_'+tgt,'members_PRIVATE':p['members_PRIVATE'][:1]})
  for p in selected:
   entries=[index[(src,p['target'],sid)] for sid in p['members_PRIVATE']];name=src+'_to_'+p['target']
   for wrong in [False,True]:
    kind='wrong' if wrong else 'correct';bid=p['batch_PRIVATE']+'_'+kind;D=root/'batches_PRIVATE'/bid;D.mkdir(parents=True);outputs,ss,rec=batchfit(base,m,S,entries,wrong,args.engineering);rec.update(source=src,target=p['target'],order=p['order'],batch_PRIVATE=p['batch_PRIVATE'],members_PRIVATE=p['members_PRIVATE']);np.savez_compressed(D/'state.npz',**ss)
    for o in outputs:
     e=o.pop('entry');sid=e['row']['id'];prob=o.pop('probability');ff=o.pop('features');np.savez_compressed(D/(sid+'.npz'),probability=prob,**ff);o['id_PRIVATE']=sid;rec.setdefault('case_final_PRIVATE',[]).append(o);key=(src,p['target'],sid);outall.setdefault(key,{})[kind+'_order'+str(p['order'])]=prob
     if args.engineering and p['n']==1:
      old=np.load(R/'crossplatform/round136/method'/name/'samples_PRIVATE'/(sid+'.npz'));oldp=dict(zip(old['arm_names'],old['scores']))['wrong_C_B_A' if wrong else 'C_B_A'];err=float(abs(prob-oldp).max());assert err<1e-7,err;prior=np.load(R/'crossplatform/round136/method'/name/'states_PRIVATE'/(sid+'.npz'))[('wrong_C_B_A' if wrong else 'C_B_A')+'_delta'];assert np.array_equal(prior,ss['delta']);rec['singleton_R136_probability_bridge']=err;rec['singleton_R136_delta_exact']=True
    dump(D/'record.json',rec);n+=1
   if args.engineering or n%16==0:print('BATCH139',args.engineering,src,n,p['order'],p['n'],flush=True)
  assert fingerprint(base)==fp and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 if not args.engineering:
  for e in reg:
   src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(R/'crossplatform/round136/method'/name/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));scores={k:ps[k] for k in ['Frozen_A','Frozen_F','AB_mean','direct_C_B_A','C_B_A','wrong_C_B_A']};new=outall.get((src,tgt,sid),{})
   for order in [0,1,2]:
    for kind in ['correct','wrong']:scores[kind+'_order'+str(order)]=new.get(kind+'_order'+str(order),ps['Frozen_A']).copy()
   out=root/'samples_PRIVATE'/name;out.mkdir(parents=True,exist_ok=True);arr=np.stack(list(scores.values()));np.savez_compressed(out/(sid+'.npz'),arm_names=np.array(list(scores)),scores=arr,queries=arr.argmax(1),source_boxes=a['source_boxes'])
 dump(root/'SEALED.json',{'fit_count':n,'n':len(reg) if not args.engineering else None,'GT_used_for_updates':False,'FINAL_access':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('BATCH139_COMPLETE',args.engineering,n,flush=True)
if __name__=='__main__':main()
