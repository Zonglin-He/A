"""Single solver-only follow-up, original R94 primary remains unchanged."""
from solver94 import *
import argparse,time
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();rows=read(H94/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=[r for p in ['drone','quad'] for r in [x for x in rows if x['platform']==p][:4]]
P=B94/'prediction336_attempt001';O=B94/('engineering_solver_repair8_attempt001' if args.engineering else 'solver_repair336_attempt001')
for n in ['samples_PRIVATE','states_PRIVATE','records_PRIVATE']:(O/n).mkdir(parents=True,exist_ok=False)
m,S=model92();source_head=patch94(m);start=__import__('time').perf_counter();counts={'n':len(rows),'new_evidence_calls':0,'head_trial_calls':0,'source_training':False,'FINAL_access':False};check=0.
for nr,r in enumerate(rows):
 restore(m,S);f=feature_to92(load(P/'features_PRIVATE'/(r['id']+'.npz')));ev=load(P/'evidence_PRIVATE'/(r['id']+'.npz'));r0=torch.as_tensor(ev['r0_PRIVATE'],device='cuda');s0=torch.as_tensor(ev['correct_base_PRIVATE'][0]+ev['r0_PRIVATE'],device='cuda');target=torch.as_tensor(ev['binding_target_PRIVATE'],device='cuda').log();p0=s0.softmax(0);mix=((p0+target.exp())/2).log();shuffle=torch.as_tensor(ev['shuffle_target_PRIVATE'],device='cuda').log();gn,info=proposal_solve(m,f,r0,target,'gn');radius=float(np.sqrt(sum(np.square(v.astype(np.float64)).sum() for v in gn.values())));records={};out={}
 for name in METHOD_SOLVE:
  restore(m,S);enable94(m);t=mix if name=='gn_mixture' else shuffle if name=='gn_shuffle' else target;delta,meta=proposal_solve(m,f,r0,t,'gradient' if name=='gradient' else 'gn')
  if name=='gradient':norm=float(np.sqrt(sum(np.square(v.astype(np.float64)).sum() for v in delta.values())));delta={k:v*(radius/max(norm,1e-30)) for k,v in delta.items()}
  loss,p=score_loss_solve(m,f,r0,t);loss.backward();g={n:p.grad.detach().cpu().numpy().copy() for n,p in m.match[-1].named_parameters()};recs=[];accepted=False;before=float(loss.detach())
  if args.engineering and name=='gn':
   aa,gg,ss=components_solve(m,f,r0);vW=torch.randn_like(m.match[-1].weight)*.001;vb=torch.randn_like(m.match[-1].bias)*.001
   def func(w,b):
    raw=F.linear(aa.float(),w,b)+f['late'];q=F.normalize(raw,dim=-1);return native_scores(q,f['u'],f['mask'].bool()).double()+r0
   _,jvp=torch.autograd.functional.jvp(func,(m.match[-1].weight,m.match[-1].bias),(vW,vb));ana=(gg*(F.linear(aa,vW.double(),vb.double()))).sum(-1);check=max(check,float(abs(jvp-ana).max()));assert check<1e-5
  for k in range(9):
   scale=2.**(-k);st={n:source_head[n]+scale*delta[n] for n in source_head};put94(m,st)
   with torch.no_grad():l,p=score_loss_solve(m,f,r0,t)
   bound=before+1e-4*sum(float((g[n].astype(np.float64)*(st[n]-source_head[n])).sum()) for n in g);key=name+'_'+str(k);np.savez_compressed(O/'states_PRIVATE'/(r['id']+'_'+key+'.npz'),**st);ok=float(l)<=bound+1e-10;recs.append({'scale':scale,'loss':float(l),'bound':bound,'accepted':ok,'state_key':key,'parameter_L2':float(np.sqrt(sum(np.square((st[n]-source_head[n]).astype(np.float64)).sum() for n in st)))});counts['head_trial_calls']+=1
   if k==0:out[name+'_raw']=p.detach().cpu().numpy()
   if ok:accepted=True;out[name+'_bt']=p.detach().cpu().numpy();break
  if not accepted:out[name+'_bt']=p0.cpu().numpy()
  records[name]={'before':before,'after':recs[-1]['loss'] if accepted else before,'accepted':accepted,'accepted_scale':recs[-1]['scale'] if accepted else 0.,'trials':recs,'target_full_teacher_not_GT':True,**meta}
 arms=[n+'_'+k for n in METHOD_SOLVE for k in ['raw','bt']];scores=np.stack([out[k] for k in arms]);np.savez_compressed(O/'samples_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(arms),scores=scores,queries=scores.argmax(1));dump(O/'records_PRIVATE'/(r['id']+'.json'),records)
 if (nr+1)%16==0 or args.engineering:print('SOLVE94',nr+1,len(rows),flush=True)
counts.update(seconds=time.perf_counter()-start,analytic_JVP_max_error=check,peak_GPU_bytes=int(torch.cuda.max_memory_allocated()));dump(O/'COUNTS.json',counts);seal(O,'R94_DISCLOSED_SINGLE_POST_DEV_SCORE_JACOBIAN_REPAIR_SEALED',n=len(rows),arm_names=arms);print(counts)
