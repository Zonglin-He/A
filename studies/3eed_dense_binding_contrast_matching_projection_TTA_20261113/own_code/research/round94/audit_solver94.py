"""Independent explicit Jacobian/Gram solve and full native replay."""
from solver94 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();P=B94/('engineering_solver_repair8_attempt001' if args.engineering else 'solver_repair336_attempt001');T=B94/'prediction336_attempt001';rows=read(H94/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=[r for pl in ['drone','quad'] for r in [x for x in rows if x['platform']==pl][:4]]
base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);m=Fusion91(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=source92();restore(m,S);sh=patch94(m);maxp=maxst=maxloss=maxbound=0.;top=checks=0;start=time.perf_counter()
for nr,r in enumerate(rows):
 restore(m,S);f=feature_to92(load(T/'features_PRIVATE'/(r['id']+'.npz')));ev=load(T/'evidence_PRIVATE'/(r['id']+'.npz'));r0=torch.as_tensor(ev['r0_PRIVATE'],device='cuda');ss0=torch.as_tensor(ev['correct_base_PRIVATE'][0]+ev['r0_PRIVATE'],device='cuda');p0=ss0.softmax(0);target=torch.as_tensor(ev['binding_target_PRIVATE'],device='cuda').log();mix=((p0+target.exp())/2).log();shuffle=torch.as_tensor(ev['shuffle_target_PRIVATE'],device='cuda').log();stored=load(P/'samples_PRIVATE'/(r['id']+'.npz'));scores=dict(zip(stored['arm_names'],stored['scores']));records=read(P/'records_PRIVATE'/(r['id']+'.json'));z0,z,d,rt,route,pm,im,ws=data91(r)
 with torch.no_grad():
  a=m.match[:-1](f['h'][None])[0].double();raw=F.linear(a,m.match[-1].weight.double(),m.match[-1].bias.double())+f['late'].double();n=raw.norm(dim=-1,keepdim=True);q=raw/n;u=f['u'].double();mask=f['mask'].bool();logit=q@u.T/.07;du=(logit[:,mask].softmax(-1)@u[mask]-logit[:,~mask].softmax(-1)@u[~mask])/.07;g=(du-q*(du*q).sum(-1,keepdim=True))/n;J=torch.cat([(g[:,:,None]*a[:,None,:]).reshape(256,-1),g],1);J=J-J.mean(0);G=J@J.T;damp=float(G.trace()/256)*.01;initial_s=torch.logsumexp(logit[:,mask],-1)-torch.logsumexp(logit[:,~mask],-1)+r0;limit=.05*float(np.sqrt(sum(np.square(v.astype(np.float64)).sum() for v in sh.values())))
  def delta_for(t,kind):
   err=t-initial_s;err-=err.mean();v=J.T@(err/256 if kind=='gradient' else torch.linalg.solve(G+damp*torch.eye(256,device='cuda',dtype=torch.float64),err));norm=float(v.norm());v*=min(1.,limit/max(norm,1e-30));return {'weight':v[:18432].reshape(64,288).float().cpu().numpy(),'bias':v[18432:].float().cpu().numpy()}
  gn=delta_for(target,'gn');radius=float(np.sqrt(sum(np.square(v.astype(np.float64)).sum() for v in gn.values())))
 for name in METHOD_SOLVE:
  restore(m,S);enable94(m);t=mix if name=='gn_mixture' else shuffle if name=='gn_shuffle' else target;dd=delta_for(t,'gradient' if name=='gradient' else 'gn')
  if name=='gradient':norm=float(np.sqrt(sum(np.square(v.astype(np.float64)).sum() for v in dd.values())));dd={k:v*(radius/max(norm,1e-30)) for k,v in dd.items()}
  p=read94(m,f,r0);e=p.log()-t;e=e-e.mean();l=e.square().mean()/2;l.backward();grad={n:p.grad.detach().cpu().numpy() for n,p in m.match[-1].named_parameters()};rec=records[name];maxloss=max(maxloss,abs(float(l)-rec['before']))
  for tr in rec['trials']:
   st=load(P/'states_PRIVATE'/(r['id']+'_'+tr['state_key']+'.npz'));expected={n:sh[n]+tr['scale']*dd[n] for n in sh};maxst=max(maxst,max(float(abs(expected[n]-st[n]).max()) for n in sh));put94(m,st)
   with torch.no_grad():
    p=read94(m,f,r0);err=p.log()-t;err-=err.mean();loss=float(err.square().mean()/2);m.route=route;m.use_interface=False;_,q,e=m.main(base,z,d,'bidirectional');pn=(native_scores(q,e['projected_text'],z['token_mask'].bool()).double()+r0).softmax(0)
   bound=rec['before']+1e-4*sum(float((grad[n].astype(np.float64)*(st[n]-sh[n])).sum()) for n in grad);maxp=max(maxp,float(abs(p-pn).max()));maxloss=max(maxloss,abs(loss-tr['loss']));maxbound=max(maxbound,abs(bound-tr['bound']));assert (loss<=bound+1e-10)==tr['accepted'];checks+=1
   for end in (['raw'] if tr['scale']==1 else [])+(['bt'] if tr['accepted'] else []):maxp=max(maxp,float(abs(pn.cpu().numpy()-scores[name+'_'+end]).max()));top+=int(int(pn.argmax())!=int(scores[name+'_'+end].argmax()))
  if not rec['accepted']:maxp=max(maxp,float(abs(p0.cpu().numpy()-scores[name+'_bt']).max()));top+=int(int(p0.argmax())!=int(scores[name+'_bt'].argmax()))
 assert maxp<1e-6 and maxst<1e-7 and maxloss<1e-6 and maxbound<1e-10 and top==0,(nr,maxp,maxst,maxloss,maxbound,top)
 if (nr+1)%16==0 or args.engineering:print('AUDITSOLVE94',nr+1,len(rows),flush=True)
assert fingerprint(base)==fp;O=H94/('engineering_solver_audit' if args.engineering else 'independent_solver_audit_attempt001');O.mkdir();out={'state':'PASS_R94_SECONDARY_EXPLICIT_JACOBIAN_NATIVE_REPLAY','n':len(rows),'trial_replays':checks,'max_probability_error':maxp,'max_parameter_state_error':maxst,'max_loss_error':maxloss,'max_Armijo_bound_error':maxbound,'top1_changes':top,'prediction_seal_sha256':sha(P/'SEALED.json'),'no_GT_or_new_evidence':True,'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',out);print(out)
