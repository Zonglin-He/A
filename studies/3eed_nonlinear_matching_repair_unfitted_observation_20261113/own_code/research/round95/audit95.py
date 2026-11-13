"""Fresh native F/C replay and independent gain-ratio/LM-state reconstruction."""
from core95 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();P=B95/('engineering_prediction8_attempt001' if args.engineering else 'prediction336_attempt001');se=read(P/'SEALED.json');rows=read(H95/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=[r for pl in ['drone','quad'] for r in [x for x in rows if x['platform']==pl][:4]]
base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);m=Fusion91(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=source92();restore(m,S);Shead=patch94(m);maxp=maxst=maxrho=maxloss=0.;top=checks=trialn=0;start=time.perf_counter()
def cached(m,f,r0):
 raw=m.match(f['h'][None])[0]+f['late'];q=F.normalize(raw,dim=-1);v=q@f['u'].T/.07;mk=f['mask'].bool();s=(torch.logsumexp(v[:,mk],-1)-torch.logsumexp(v[:,~mk],-1)).double()+r0;return s.softmax(0)
def native(m,z,d,route,r0,pm=None,im=None):
 m.route=route;m.use_interface=False;_,q,e=m.main(base,z,d,'bidirectional',point_mask=pm,image_mask=im);v=q@e['projected_text'].T/.07;mk=z['token_mask'].bool();return ((torch.logsumexp(v[:,mk],-1)-torch.logsumexp(v[:,~mk],-1)).double()+r0).softmax(0)
for nr,r in enumerate(rows):
 restore(m,S);f=feature_to92(load(P94/'features_PRIVATE'/(r['id']+'.npz')));ev=load(P94/'evidence_PRIVATE'/(r['id']+'.npz'));r0=torch.as_tensor(ev['r0_PRIVATE'],device='cuda');s0=torch.as_tensor(ev['correct_base_PRIVATE'][0]+ev['r0_PRIVATE'],device='cuda');p0=s0.softmax(0);qF=torch.as_tensor(ev['binding_target_PRIVATE'],device='cuda');mix=(p0+qF)/2;shuf=torch.as_tensor(ev['shuffle_target_PRIVATE'],device='cuda');sv=load(P/'samples_PRIVATE'/(r['id']+'.npz'));scores=dict(zip(sv['arm_names'],sv['scores']));rec=read(P/'records_PRIVATE'/(r['id']+'.json'));fc=feature_to92(load(P/'C_features_PRIVATE'/(r['id']+'.npz')));z0,z,d,rt,route,pm,im,ws=data91(r);hp,hi=masks(r,d,76003)
 with torch.no_grad():
  aC=native(m,z,d,route,r0,hp,hi);aF=native(m,z,d,route,r0);maxp=max(maxp,float(abs(aC-cached(m,fc,r0)).max()),float(abs(aF-p0).max()));delta=torch.as_tensor(ev['delta_PRIVATE'],device='cuda');qC=(aC.log()+delta).softmax(0);q4=torch.stack([(torch.as_tensor(b,device='cuda')+r0+delta).softmax(0) for b in ev['correct_base_PRIVATE']]+[qC]).mean(0);shC=(aC.log()+delta[torch.as_tensor(ev['query_control_PRIVATE'],device='cuda')]).softmax(0);expected={'frozen_C':aC,'direct_C':qC,'direct_FC':(qF+qC)/2,'direct_four':q4,'mixture_FC':(mix+(aC+qC)/2)/2,'shuffle_FC':(shuf+shC)/2}
  for k,v in expected.items():maxp=max(maxp,float(abs(v.cpu().numpy()-scores[k]).max()));top+=int(int(v.argmax())!=int(scores[k].argmax()));checks+=1
 for name in ['lm','lm_mixture','lm_shuffle','lm_capped']:
  restore(m,S);t=(mix if name=='lm_mixture' else shuf if name=='lm_shuffle' else qF).log();damping=None;initial=None;accepted_steps=0
  for it in range(10):
   with torch.no_grad():
    p=cached(m,f,r0);e=t-p.log();e-=e.mean();before=float(e.square().mean()/2)
    if initial is None:initial=before
    if before<=max(1e-12,initial*1e-3):break
    a=m.match[:-1](f['h'][None])[0].double();raw=F.linear(a,m.match[-1].weight.double(),m.match[-1].bias.double())+f['late'].double();norm=raw.norm(dim=-1,keepdim=True);q=raw/norm;u=f['u'].double();v=q@u.T/.07;mk=f['mask'].bool();g=(v[:,mk].softmax(-1)@u[mk]-v[:,~mk].softmax(-1)@u[~mk])/.07;g=(g-q*(g*q).sum(-1,keepdim=True))/norm
    if args.engineering:
     J=torch.cat([(g[:,:,None]*a[:,None]).reshape(256,-1),g],1);J-=J.mean(0);G=J@J.T
    else:G=(g@g.T)*(a@a.T+1);G=G-G.mean(0,keepdim=True)-G.mean(1,keepdim=True)+G.mean()
    if damping is None:damping=max(1e-12,.01*float(G.trace())/256)
    old=patch94(m);rr=[x for x in rec[name]['trials'] if x['iteration']==it+1];assert rr;accepted=False
    for tr,x in enumerate(rr):
     assert x['damping']==damping or abs(x['damping']-damping)<1e-8*max(damping,1e-20);lam=x['damping'];v1=torch.linalg.solve(G+lam*torch.eye(256,device='cuda',dtype=torch.float64),e);v1-=v1.mean();ds={'weight':((v1[:,None]*g).T@a).float().cpu().numpy(),'bias':(v1[:,None]*g).sum(0).float().cpu().numpy()};nn=float(np.sqrt(sum(np.square(y.astype(np.float64)).sum() for y in ds.values())))
     if name=='lm_capped':
      lim=.05*float(np.sqrt(sum(np.square(y.astype(np.float64)).sum() for y in Shead.values())))
      if nn>lim:ds={k:y*(lim/nn) for k,y in ds.items()}
     actual=load(P/'states_PRIVATE'/(r['id']+'_'+name+'_'+x['state_key']+'.npz'));maxst=max(maxst,max(float(abs((old[k]+ds[k])-actual[k]).max()) for k in old));put94(m,actual);pp=cached(m,f,r0);pn=native(m,z,d,route,r0);maxp=max(maxp,float(abs(pp-pn).max()));err=t-pp.log();err-=err.mean();after=float(err.square().mean()/2);pw=torch.as_tensor(actual['weight']-old['weight'],device='cuda').double();pb=torch.as_tensor(actual['bias']-old['bias'],device='cuda').double();pred=(g*F.linear(a,pw,pb)).sum(-1);pred-=pred.mean();red=float((e.square().mean()-(e-pred).square().mean())/2);rho=(before-after)/max(red,1e-30);maxrho=max(maxrho,abs(rho-x['rho']));maxloss=max(maxloss,abs(before-x['before']),abs(after-x['after']),abs(red-x['predicted_reduction']));ok=red>0 and np.isfinite(after) and after<before and rho>.1;assert ok==x['accepted'];trialn+=1
     if ok:
      accepted=True;accepted_steps+=1
      if rho>.75:damping=max(1e-15,lam/3)
      elif rho<.25:damping=lam*2
      else:damping=lam
      break
     put94(m,old);damping=lam*10
    if not accepted:break
  final=load(P/'states_PRIVATE'/(r['id']+'_'+name+'_final.npz'));maxst=max(maxst,max(float(abs(patch94(m)[k]-final[k]).max()) for k in final));assert accepted_steps==rec[name]['accepted_steps']
  with torch.no_grad():
   put94(m,final);pF=native(m,z,d,route,r0);pC=native(m,z,d,route,r0,hp,hi)
   for view,pp in [('F',pF),('C',pC),('FC',(pF+pC)/2)]:key=name+'_'+view;maxp=max(maxp,float(abs(pp.cpu().numpy()-scores[key]).max()));top+=int(int(pp.argmax())!=int(scores[key].argmax()));checks+=1
 old=load(P94/'samples_PRIVATE'/(r['id']+'.npz'));o=dict(zip(old['arm_names'],old['scores']));gn=load(B94/'solver_repair336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));gg=dict(zip(gn['arm_names'],gn['scores']));mp={'R94_direct':'binding_direct','R94_mixture':'binding_optimal_mixture','R94_Adam':'binding_bt1'}
 for k in REFS95:assert np.array_equal(scores[k],gg['gn_bt'] if k=='R94_GN' else o[mp.get(k,k)])
 assert np.array_equal(sv['source_boxes'],z0['boxes']) and np.array_equal(sv['selected_original_boxes'],z0['boxes'][sv['queries']]);assert maxp<1e-6 and maxst<1e-7 and maxloss<1e-6 and maxrho<1e-5 and top==0,(nr,maxp,maxst,maxloss,maxrho,top)
 if (nr+1)%16==0 or args.engineering:print('AUDIT95',nr+1,len(rows),flush=True)
assert fingerprint(base)==fp;O=H95/('engineering_native_audit' if args.engineering else 'independent_native_audit_attempt001');O.mkdir();out={'state':'PASS_R95_NONLINEAR_LM_GAIN_RATIO_STATES_AND_NATIVE_F_C_READOUTS','n':len(rows),'arms':len(ARMS95),'saved_endpoint_replays':checks,'trial_replays':trialn,'max_probability_error':maxp,'max_parameter_state_error':maxst,'max_loss_error':maxloss,'max_gain_ratio_error':maxrho,'top1_changes':top,'accepted_total_increases':0,'new_task_GT_scoring':0,'FINAL_access':False,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',out);print(out)
