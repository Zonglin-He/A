"""Independently reconstruct interventions/targets and Adam rollback; replay all trial states natively."""
from core94 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();P=B94/('engineering_prediction336_attempt001' if args.engineering else 'prediction336_attempt001');se=read(P/'SEALED.json');rows=read(H94/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=[r for pl in ['drone','quad'] for r in [x for x in rows if x['platform']==pl][:4]]
base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);m=Fusion91(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=source92();restore(m,S);Shead=patch94(m);maxp=maxstate=maxloss=maxbound=0.;top=trials=checks=0;start=time.perf_counter()
def cached_p(f,r0):
 q=torch.nn.functional.normalize(m.match(f['h'][None])[0]+f['late'],dim=-1);logits=q@f['u'].T/.07;mk=f['mask'].bool();s=torch.logsumexp(logits[:,mk],-1)-torch.logsumexp(logits[:,~mk],-1);return (s.double()+r0).softmax(0)
def ownloss(p,t,p0):return torch.sum(torch.xlogy(t,t)-t*p.log())+torch.sum(torch.xlogy(p0,p0)-p0*p.log())
def native(z,d,route,r0):
 m.route=route;m.use_interface=False;_,q,e=m.main(base,z,d,'bidirectional');ss=q@e['projected_text'].T/.07;mk=z['token_mask'].bool();return ((torch.logsumexp(ss[:,mk],-1)-torch.logsumexp(ss[:,~mk],-1)).double()+r0).softmax(0)
for nr,r in enumerate(rows):
 restore(m,S);m.requires_grad_(False);z0,z,d,rt,route,pm,im,ws=data91(r);sv=load(P/'samples_PRIVATE'/(r['id']+'.npz'));ev=load(P/'evidence_PRIVATE'/(r['id']+'.npz'));f=feature_to92(load(P/'features_PRIVATE'/(r['id']+'.npz')));rr=read(P/'records_PRIVATE'/(r['id']+'.json'));assert np.array_equal(sv['source_boxes'],z0['boxes']) and np.array_equal(sv['selected_original_boxes'],z0['boxes'][sv['queries']]);stored=dict(zip(sv['arm_names'],sv['scores']));real=[];wrong=[]
 with torch.no_grad():
  for aa,bb in [(None,None)]+list(zip(pm,im)):
   ff=feature91(m,base,z,d,route,S,aa,bb);b,res,_,_=parts92(m,ff,route,S);real.append(b)
   if aa is None:r0=res
  correct=torch.stack(real);maxp=max(maxp,float(abs(correct.cpu().numpy()-ev['correct_base_PRIVATE']).max()));assert np.max(abs(r0.cpu().numpy()-ev['r0_PRIVATE']))<1e-6
  for c in range(2):
   inds=np.flatnonzero(d['visible'].cpu().numpy());seed=int(hashlib.sha256(('R94binding|'+r['id']).encode()).hexdigest()[:16],16);order=np.random.default_rng(seed).permutation(inds);perm=np.arange(1024)
   if len(order)>1:perm[order]=np.roll(order,1 if c==0 else max(1,len(order)//2))
   assert np.array_equal(perm,ev['permutations_PRIVATE'][c]);inv=np.argsort(perm);pp=torch.as_tensor(perm,device='cuda');ii=torch.as_tensor(inv,device='cuda');dd=dict(d);dd['p2i']=d['p2i'][pp];dd['p2i_weight']=d['p2i_weight'][pp];dd['i2p']=torch.where(d['i2p']>=0,ii[d['i2p'].clamp_min(0)],d['i2p']);ww=[]
   assert torch.equal(dd['p2i'].ge(0).any(-1),d['p2i'].ge(0).any(-1)) and torch.equal(d['visible'],d['visible'][pp]);assert torch.equal(d['depth_target'],dd['depth_target'])
   for aa,bb in [(None,None)]+list(zip(pm,im)):
    ff=feature91(m,base,z,dd,route,S,aa,bb);b,_,_,_=parts92(m,ff,route,S);ww.append(b)
   wrong.append(torch.stack(ww))
  wrong=torch.stack(wrong);maxp=max(maxp,float(abs(wrong.cpu().numpy()-ev['wrong_base_PRIVATE']).max()));delta=correct.mean(0)-wrong.mean((0,1));delta-=delta.mean();s0=correct[0]+r0;p0=s0.softmax(0);qp=np.random.default_rng(int(hashlib.sha256(('R94querycontrol|'+r['id']).encode()).hexdigest()[:16],16)).permutation(256);ts={'binding':(s0+delta).softmax(0),'reverse':(s0-delta).softmax(0),'shuffle':(s0+delta[qp]).softmax(0),'viewmean':(correct.mean(0)+r0).softmax(0)}
  for k,v in ts.items():maxp=max(maxp,float(abs(v.cpu().numpy()-ev[k+'_target_PRIVATE']).max()))
  expected={'binding_direct':ts['binding'],'binding_optimal_mixture':(p0+ts['binding'])/2,'reverse_direct':ts['reverse'],'shuffle_direct':ts['shuffle'],'viewmean_direct':ts['viewmean'],'viewmean_optimal_mixture':(p0+ts['viewmean'])/2}
  for k,v in expected.items():maxp=max(maxp,float(abs(v.cpu().numpy()-stored[k]).max()));top+=int(int(v.argmax())!=int(stored[k].argmax()));checks+=1
 for name in ['binding','shuffle','reverse','viewmean']:
  restore(m,S);m.requires_grad_(False)
  for p in head94(m):p.requires_grad_(True)
  opt=torch.optim.Adam(head94(m),lr=.0005);target=torch.as_tensor(ev[name+'_target_PRIVATE'],device='cuda');pzero=torch.as_tensor(stored['R91_frozen'],device='cuda')
  for rec in rr[name]:
   opt.zero_grad();p=cached_p(f,r0);loss=ownloss(p,target,pzero);loss.backward();torch.nn.utils.clip_grad_norm_(head94(m),1.);g=[p.grad.clone() for p in head94(m)];old=patch94(m);os=copy.deepcopy(opt.state_dict());opt.step();proposal=patch94(m);ds={n:proposal[n]-v for n,v in old.items()};maxloss=max(maxloss,abs(float(loss)-rec['before']));accepted=False
   for trial in rec['trials']:
    state=load(P/'states_PRIVATE'/(r['id']+'_'+name+'_'+trial['state_key']+'.npz'));ss={n:old[n]+trial['scale']*ds[n] for n in old};maxstate=max(maxstate,max(float(np.max(abs(ss[n]-state[n]))) for n in ss));put94(m,state)
    with torch.no_grad():pc=cached_p(f,r0);pn=native(z,d,route,r0);ll=float(ownloss(pc,target,pzero))
    maxp=max(maxp,float(abs(pc-pn).max()));maxloss=max(maxloss,abs(ll-trial['loss']));dot=sum(float((gg.double()*torch.as_tensor(state[n]-old[n],device='cuda').double()).sum()) for (n,_),gg in zip(m.match[-1].named_parameters(),g));bound=rec['before']+1e-4*dot;maxbound=max(maxbound,abs(bound-trial['bound']));assert (ll<=bound+1e-10)==trial['accepted'];trials+=1
    if rec['step']==1 and trial['scale']==1 and name=='binding':maxp=max(maxp,float(abs(pn.cpu().numpy()-stored['binding_raw1']).max()));top+=int(int(pn.argmax())!=int(stored['binding_raw1'].argmax()));checks+=1
    if trial['accepted']:accepted=True
   assert accepted==rec['accepted']
   if not accepted:put94(m,old);opt.load_state_dict(os)
   endpoint=load(P/'states_PRIVATE'/(r['id']+'_'+name+'_'+rec['endpoint_key']+'.npz'));maxstate=max(maxstate,max(float(np.max(abs(patch94(m)[n]-endpoint[n]))) for n in endpoint))
   with torch.no_grad():pe=native(z,d,route,r0)
   key=name+'_bt'+str(rec['step'])
   if key in stored:maxp=max(maxp,float(abs(pe.cpu().numpy()-stored[key]).max()));top+=int(int(pe.argmax())!=int(stored[key].argmax()));checks+=1
   assert rec['after']<=rec['before']+1e-10
 old=load(P93/'samples_PRIVATE'/(r['id']+'.npz'));ref=dict(zip(old['arm_names'],old['scores']))
 for k in REFS94:assert np.array_equal(stored[k],ref['competition_projected_bt' if k=='R93_bt' else 'competition_projected_raw' if k=='R93_raw' else k])
 assert maxp<1e-6 and maxstate<1e-7 and maxloss<1e-6 and maxbound<1e-10 and top==0,(nr,maxp,maxstate,maxloss,maxbound,top)
 if (nr+1)%16==0 or args.engineering:print('AUDIT94',nr+1,len(rows),flush=True)
assert fingerprint(base)==fp;O=H94/('engineering_native_audit' if args.engineering else 'independent_native_audit_attempt001');O.mkdir();out={'state':'PASS_R94_NATIVE_INTERVENTIONS_DENSE_TARGETS_TRANSACTIONAL_ADAM_AND_ALL_TRIAL_READOUTS','n':len(rows),'arms':len(ARMS94),'saved_endpoint_replays':checks,'trial_replays':trials,'max_probability_or_evidence_error':maxp,'max_parameter_state_error':maxstate,'max_loss_error':maxloss,'max_Armijo_bound_error':maxbound,'top1_changes':top,'accepted_total_increases':0,'prediction_seal_sha256':sha(P/'SEALED.json'),'new_task_GT_scoring':0,'FINAL_access':False,'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',out);print(out)
