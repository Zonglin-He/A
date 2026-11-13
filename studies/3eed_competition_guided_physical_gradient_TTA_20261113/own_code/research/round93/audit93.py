"""Independent real-view evidence, NumPy projection/radius and live state/objective replay."""
from core93 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();P=B93/('engineering_prediction336_attempt001' if args.engineering else 'prediction336_attempt001');se=read(P/'SEALED.json');rows=read(H93/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=[r for pl in ['drone','quad'] for r in [x for x in rows if x['platform']==pl][:4]]
base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);m=Fusion91(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=source92();worst=stateerr=losserr=bounderr=graderr=0.;top=accepted=checks=trials_n=0;held=[];start=time.perf_counter()
def own_eval(z,d,route,r0,refs,pm,im,ws,name,co,wphys,wrong,pairs,target,current):
 m.route=route;m.variant='full';m.use_interface=False;_,q,e=m.main(base,z,d,'bidirectional');bs=native_scores(q,e['projected_text'],z['token_mask'].bool()).double();s=bs+r0;p=s.softmax(0);lp=0.
 for a,b,w in zip(pm,im,ws):
  P0,I,D=m.local_losses(z,d,a,b,wrong);lp=lp+(.1*(P0*w['P']).sum()+.1*(I*w['I']).sum()+.01*(D*w['D']).sum())/2
 if name=='current_top1':lt=-p[current].clamp_min(1e-300).log()
 elif len(pairs):
  tar=torch.as_tensor(-target if name=='reverse_preference' else target,device='cuda');lt=(s[torch.as_tensor(pairs[:,0],device='cuda')]-s[torch.as_tensor(pairs[:,1],device='cuda')]-tar).square().mean()/2
 else:lt=s.sum()*0
 keep=(refs[0]*(refs[0].clamp_min(1e-300).log()-p.clamp_min(1e-300).log())).sum()+.05*(q-refs[1]).square().sum(-1).mean();return wphys*lp.double()+co*lt+keep,p
for nr,r in enumerate(rows):
 sid=r['id'];z0,z,d,rt,route,pm,im,ws=data91(r);a=load(P/'samples_PRIVATE'/(sid+'.npz'));assert sha(P/'samples_PRIVATE'/(sid+'.npz'))==se['files']['samples_PRIVATE/'+sid+'.npz'];assert list(a['arm_names'])==ARMS93 and np.array_equal(a['source_boxes'],z0['boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1)) and np.array_equal(a['selected_original_boxes'],z0['boxes'][a['queries']]);scores=dict(zip(a['arm_names'],a['scores']));stored=load(P/'evidence_PRIVATE'/(sid+'.npz'));meta=read(P/'records_PRIVATE'/(sid+'.json'));parts=[];restore(m,S);freeze91(m)
 with torch.no_grad():
  for aa,bb in [(None,None)]+list(zip(pm,im)):
   f=feature91(m,base,z,d,route,S,aa,bb);parts.append(parts92(m,f,route,S))
  pool,pairs,target,elig=evidence93(z0['boxes'],rt['own_support'],d['visible'].cpu().numpy(),parts);assert np.array_equal(pool,stored['pool_PRIVATE']) and np.array_equal(pairs,stored['pairs_PRIVATE']);assert np.max(abs(target-stored['targets_PRIVATE']),initial=0)<1e-6;r0=parts[0][1];s0=(parts[0][0]+r0).cpu().numpy();E=np.zeros((len(pairs),256))
  if len(pairs):E[np.arange(len(pairs)),pairs[:,0]]=1;E[np.arange(len(pairs)),pairs[:,1]]=-1
  for name,tt in [('competition_direct',target),('reverse_preference_direct',-target)]:
   # Independent minimum-norm least squares, rather than importing producer pinv code.
   dd=np.linalg.lstsq(E,tt-E@s0,rcond=1e-12)[0] if len(pairs) else np.zeros(256);ss=s0+dd;pp=torch.as_tensor(ss).softmax(0).numpy();worst=max(worst,float(abs(pp-scores[name]).max()));top+=int(int(pp.argmax())!=int(scores[name].argmax()));checks+=1
 old=load(P92/'samples_PRIVATE'/(sid+'.npz'));oo=dict(zip(old['arm_names'],old['scores']))
 for k in REFS93:assert np.array_equal(scores[k],oo['static_residual_formal_bt' if k=='R92_bt' else 'same_state_raw_static_residual' if k=='R92_raw_static' else k])
 g=gradients93(m,base,z,d,route,S,pm,im,ws,pairs,target,r0);saveg=load(P/'gradients_PRIVATE'/(sid+'.npz'));refs=(g['initial_p'],g['initial_q']);radius=float(g['original_raw'].norm())
 for k,v in g.items():
  if isinstance(v,torch.Tensor):graderr=max(graderr,float(abs(v.cpu().numpy()-saveg[k]).max()) if v.numel() else 0.)
 for name in METHODS93:
  rr=meta[name+'_bt'];co=rr['coefficient'];wp=rr['physical_weight'];wrong=name=='wrong_binding';gp=g['wrong_physical'] if wrong else g['physical'];gc=g['reverse_pair'] if name=='reverse_preference' else g['current'] if name=='current_top1' else g['pair'];active=len(pairs)>0 and float(gc.norm())>1e-12;assert active==rr['task_active'];assert co==(float(gp.norm()/gc.norm()) if active else 0.);aa=gp.cpu().numpy().astype(np.float64);bb=gc.cpu().numpy().astype(np.float64);dot=aa@bb
  if not active:dd=aa;gi=aa
  else:
   ap=aa-min(0.,dot)/max(bb@bb,1e-30)*bb;dd=co*bb if name=='competition_only' else aa+co*bb if name=='competition_sum' else ap+co*bb;gi=wp*aa+co*bb
  dn=np.linalg.norm(dd);proposal=(-radius*dd/dn if dn>1e-12 else np.zeros_like(dd)).astype(np.float32);graderr=max(graderr,float(abs(proposal-saveg[name+'_proposal']).max()));assert abs(float(np.linalg.norm(proposal))-radius)<1e-6 or float(np.linalg.norm(proposal))==0.;visited={};before=rr['before']
  for trial in rr['trials']:
   key=trial['state_key'];ss=load(P/'states_PRIVATE'/(sid+'_'+key+'.npz'));restore(m,S);freeze91(m);theta_step91(m,S,torch.as_tensor(saveg[name+'_proposal'],device='cuda')*trial['scale']);stateerr=max(stateerr,max(float(abs(p.detach().cpu().numpy()-ss[n]).max()) for n,p in m.tta_parameters()));puttheta91(m,ss)
   with torch.no_grad():val,pp=own_eval(z,d,route,r0,refs,pm,im,ws,name,co,wp,wrong,pairs,target,g['current_query_PRIVATE'])
   # Use the actual initial objective gradient and saved FP32 displacement.
   actualg=(wp*gp+co*gc).double() if active else gp.double();delta=torch.as_tensor(saveg[name+'_proposal'],device='cuda')*trial['scale'];bound=before+1e-4*float(actualg@delta.double());losserr=max(losserr,abs(float(val)-trial['objective']));bounderr=max(bounderr,abs(bound-trial['bound']));assert (float(val)<=bound+1e-10)==trial['accepted'];visited[key]=pp.cpu().numpy();trials_n+=1
  for ending in ['raw','bt']:
   r1=meta[name+'_'+ending];pp=refs[0].cpu().numpy() if r1['state_key']=='initial' else visited[r1['state_key']];worst=max(worst,float(abs(pp-scores[name+'_'+ending]).max()));top+=int(int(pp.argmax())!=int(scores[name+'_'+ending].argmax()));checks+=1;assert not r1['accepted_by_loss_rule'] or r1['after']<=r1['before']+1e-10;accepted+=int(r1['accepted_by_loss_rule'])
  if name=='competition_projected':
   restore(m,S)
   if rr['state_key']!='initial':puttheta91(m,load(P/'states_PRIVATE'/(sid+'_'+rr['state_key']+'.npz')))
   hp,hi=masks(r,d,76003);hw=weight_sets({'id':'UNUSED_UNIFORM'},d,[hp],[hi],np.zeros(1024),np.zeros(1536))[0]['uniform']
   with torch.no_grad():after=float(uniform_aux91(m,z,d,[hp],[hi],hw))
   restore(m,S)
   with torch.no_grad():beforeh=float(uniform_aux91(m,z,d,[hp],[hi],hw))
   held.append({'row':'DEV'+str(nr+1).zfill(4),'platform':r['platform'],'arm':'competition_projected_bt','heldout_physical_before':beforeh,'heldout_physical_after':after,'heldout_physical_change':after-beforeh,'not_used_for_acceptance':True})
 assert worst<1e-6 and stateerr<=1e-7 and losserr<1e-6 and bounderr<1e-10 and graderr<1e-6 and top==0
 if (nr+1)%16==0 or args.engineering:print('AUDIT93',nr+1,len(rows),flush=True)
assert fingerprint(base)==fp;O=H93/('engineering_native_audit' if args.engineering else 'independent_native_audit_attempt001');O.mkdir();out={'state':'PASS_R93_NATIVE_EVIDENCE_GRADIENT_PROJECTION_RADIUS_STATES_OBJECTIVES_AND_READOUTS','n':len(rows),'arms':len(ARMS93),'saved_endpoint_replays':checks,'trial_replays':trials_n,'max_probability_error':worst,'max_parameter_state_error':stateerr,'max_loss_error':losserr,'max_Armijo_bound_error':bounderr,'max_gradient_or_projection_error':graderr,'top1_changes':top,'accepted_label_free_endpoints':accepted,'accepted_total_increases':0,'new_task_GT_scoring':0,'FINAL_access':False,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',out);dump(O/'HELDOUT_ROWS.json',held);print(out)
