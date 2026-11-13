"""Independent live native replay, original gradients, own acceptance and factorial identities."""
from common92 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();P=B92/('engineering_prediction336_attempt001' if args.engineering else 'prediction336_attempt001');se=read(P/'SEALED.json');rows=read(H92/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=[next(r for r in rows if r['platform']==p) for p in ['drone','quad']]
base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);m=Fusion91(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=source92();restore(m,S);worst=fe=stateerr=losserr=bounderr=0.;top=nc=checks=accepted=0;held=[];start=time.perf_counter();initialparity=0.
def ownparts(f,rt):
 q=F.normalize(m.match(f['h'][None])[0]+f['late'],dim=-1);b=native_scores(q,f['u'],f['mask'].bool()).double();d,_=m.interface(f['h'],f['local'],f['own'],f['text'],f['relation_text'],rt,'full');return b,d.double(),q
for nr,r in enumerate(rows):
 sid=r['id'];z0,z,d,rt,route,pm,im,ws=data91(r);a=load(P/'samples_PRIVATE'/(sid+'.npz'));assert sha(P/'samples_PRIVATE'/(sid+'.npz'))==se['files']['samples_PRIVATE/'+sid+'.npz'];scores=dict(zip(a['arm_names'],a['scores']));assert list(a['arm_names'])==ARMS92 and np.array_equal(a['source_boxes'],z0['boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1)) and np.array_equal(a['selected_original_boxes'],z0['boxes'][a['queries']]);meta=read(P/'records_PRIVATE'/(sid+'.json'));saved=load(P/'components_PRIVATE'/(sid+'.npz'));rec91=read(P91/'records_PRIVATE'/(sid+'.json'));needed={rr['feature_key'] for rr in meta.values()}|{t['feature_key'] for rr in meta.values() for t in rr.get('trials',[])}|{'view0'};fresh={};obj={};pathmap={}
 for rr in meta.values():
  pathmap[rr['feature_key']]=rr['state_path']
  for t in rr.get('trials',[]):pathmap[t['feature_key']]=t['state_path']
 for key in sorted(needed):
  restore(m,S);freeze91(m)
  if key!='view0':puttheta91(m,load(Path(pathmap[key])))
  with torch.no_grad():
   f=feature91(m,base,z,d,route,S);nc+=1;b1,r1,q1=ownparts(f,route);fresh[key]=(b1,r1,q1);oldpath=P91/'features_PRIVATE'/(sid+'_'+key+'.npz');ff=load(oldpath if oldpath.exists() else P/'extra_features_PRIVATE'/(sid+'_'+key+'.npz'))
   for k,v in f.items():
    if v.numel() and v.dtype!=torch.bool:fe=max(fe,float(abs(v.cpu().numpy()-ff[k]).max()))
   for label,v in [('base',b1),('residual',r1),('query',q1)]:worst=max(worst,float(abs(v.cpu().numpy()-saved[key+'_'+label]).max()))
   if key!='view0':obj[key]=float(uniform_aux91(m,z,d,pm,im,ws,key.startswith('wrong')))
 b0,r0,q0=fresh['view0'];p0=(b0+r0).softmax(0)
 def compare(p,arm):
  global worst,top,checks
  v=p.detach().cpu().numpy();worst=max(worst,float(abs(v-scores[arm]).max()));top+=int(int(v.argmax())!=int(scores[arm].argmax()));checks+=1
 compare(p0,'R91_frozen')
 olda=load(P91/'samples_PRIVATE'/(sid+'.npz'));oldd=dict(zip(olda['arm_names'],olda['scores']))
 for a0,a1 in REF_MAP92.items():assert np.array_equal(scores[a0],oldd[a1])
 for kind in ['raw','bt','wrong_raw','wrong_bt']:
  rr=meta['same_state_'+kind+'_static_residual'];b1,r1,q1=fresh[rr['feature_key']];compare((b1+r0).softmax(0),'same_state_'+kind+'_static_residual');compare((b0+r1).softmax(0),'same_state_'+kind+'_residual_only');compare((b1+r1).softmax(0),'R91_'+kind+'_dynamic')
  assert float(((b1+r1)-(b1+r0)-(b0+r1)+(b0+r0)).abs().max())<1e-10
 for wrong in [False,True]:
  g,raw,gg=physical_direction91(m,base,z,d,S,pm,im,ws,wrong);direction=-(1/(g.norm()+1e-6)).clamp_max(1)*g;arm='static_residual_formal_'+('wrong_bt' if wrong else 'bt');rr=meta[arm];succeeded=False
  for trial in rr['trials']:
   alpha=trial['alpha'];freeze91(m);theta_step91(m,S,raw if alpha==.5 else alpha*direction);ss=load(Path(trial['state_path']));stateerr=max(stateerr,max(float(abs(p.detach().cpu().numpy()-ss[n]).max()) for n,p in m.tta_parameters()));b1,r1,q1=fresh[trial['feature_key']];p1=(b1+r0).softmax(0);keep=(p0*(p0.clamp_min(1e-300).log()-p1.clamp_min(1e-300).log())).sum()+.05*(q1-q0).square().sum(-1).mean();total=obj[trial['feature_key']]+float(keep);bound=rr['before']+1e-4*float(g.double()@(raw if alpha==.5 else alpha*direction).double());losserr=max(losserr,abs(total-trial['objective']));bounderr=max(bounderr,abs(bound-trial['bound']));ok=total<=bound+1e-10;assert ok==trial['accepted'];assert not succeeded;succeeded=ok
  assert succeeded==rr['accepted_by_loss_rule'];key=rr['feature_key'];b1,r1,q1=fresh[key];compare((b1+r0).softmax(0),arm);accepted+=int(succeeded);assert not succeeded or rr['after']<=rr['before']+1e-10
  restore(m,S)
  if rr['state_path']!='initial':puttheta91(m,load(Path(rr['state_path'])))
  hp,hi=masks(r,d,76003);hw=weight_sets({'id':'UNUSED_UNIFORM'},d,[hp],[hi],np.zeros(1024),np.zeros(1536))[0]['uniform']
  with torch.no_grad():hh=float(uniform_aux91(m,z,d,[hp],[hi],hw))
  restore(m,S)
  with torch.no_grad():hb=float(uniform_aux91(m,z,d,[hp],[hi],hw))
  held.append({'row':'DEV'+str(nr+1).zfill(4),'platform':r['platform'],'arm':arm,'heldout_physical_before':hb,'heldout_physical_after':hh,'heldout_physical_change':hh-hb,'not_used_for_fitting_or_acceptance':True})
 # Explicit gradient parity on the two engineering observations, not just a zero-KL assertion.
 if args.engineering:
  g,_,_=physical_direction91(m,base,z,d,S,pm,im,ws);restore(m,S);enable_theta91(m);m.route=route;m.variant='full';m.use_interface=False;_,q,e=m.main(base,z,d,'bidirectional');bp=e['base_probability'].clamp_min(1e-300).log();sp=(bp+r0.detach()).softmax(0);loss=uniform_aux91(m,z,d,pm,im,ws)+keep91(sp,q,p0.detach(),q0.detach());ps=[p for n,p in m.tta_parameters()];sg=flat90(torch.autograd.grad(loss,ps,allow_unused=True),ps);initialparity=max(initialparity,float((sg-g).abs().max()))
 assert worst<1e-6 and fe<1e-6 and stateerr<=1e-7 and losserr<1e-6 and bounderr<1e-10 and top==0
 if (nr+1)%32==0 or args.engineering:print('AUDIT92',nr+1,len(rows),flush=True)
assert fingerprint(base)==fp;assert initialparity<1e-7;O=H92/('engineering_native_audit' if args.engineering else 'independent_native_audit_attempt001');O.mkdir();out={'state':'PASS_R92_NATIVE_FACTORIAL_READOUT_GRADIENT_STATE_OWN_ACCEPTANCE_AND_HELDOUT','n':len(rows),'arms':len(ARMS92),'saved_endpoint_replays':checks,'native_suffix_calls':nc,'max_probability_or_components_error':worst,'feature_max_error':fe,'max_parameter_state_error':stateerr,'max_loss_error':losserr,'max_Armijo_bound_error':bounderr,'engineering_initial_gradient_parity_error':initialparity,'top1_changes':top,'accepted_label_free_endpoints':accepted,'accepted_total_increases':0,'new_task_GT_scoring':0,'FINAL_access':False,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',out);dump(O/'HELDOUT_ROWS.json',held);print(out)
