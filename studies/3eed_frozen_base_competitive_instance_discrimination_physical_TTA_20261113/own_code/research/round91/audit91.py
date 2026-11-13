"""Independent native theta states, physical/conditional objectives and complete readouts."""
from targets91 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--source',action='store_true');ap.add_argument('--engineering',action='store_true');args=ap.parse_args();label=('engineering_' if args.engineering else '')+('source128' if args.source else 'prediction336')+'_attempt001';P=B91/label;se=read(P/'SEALED.json');selection=read(B91/'trained_source_attempt001/SELECTED_SOURCE_ONLY.json');SS={n:load(ROOT/v['state_path']) for n,v in selection['models'].items()};base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);torch.manual_seed(0);m=Fusion91(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S0=loadbase91(m,'R76');rows=source_rows('VAL128') if args.source else read(H91/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=rows[:2] if args.source else [next(r for r in rows if r['platform']==pl) for pl in ['drone','quad']]
arms=NEW91 if args.source else LEGACY91+NEW91;assert se['n']==len(rows);worst=fe=stateerr=loss_error=0.;topchanges=0;endpoints=checks=calls=accepted=0;start=time.perf_counter();held=read(P/'HELDOUT_ROWS.json')
def fileload(path):assert sha(path)==se['files'][path.relative_to(P).as_posix()];return load(path)
def compare(x,y):
 global worst,topchanges,endpoints
 x=x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else x;worst=max(worst,float(abs(x-y).max()));topchanges+=int(int(x.argmax())!=int(y.argmax()));endpoints+=1
for nr,r in enumerate(rows):
 sid=r['id'];z0,z,d,rt,route,pm,im,ws=data91(r,args.source);a=fileload(P/'samples_PRIVATE'/(sid+'.npz'));scores=dict(zip(a['arm_names'],a['scores']));assert list(a['arm_names'])==arms and np.array_equal(a['source_boxes'],z0['boxes']) and np.array_equal(a['queries'],a['scores'].argmax(-1)) and np.array_equal(a['selected_original_boxes'],z0['boxes'][a['queries']]);assert np.array_equal(a['mask_point'],torch.stack(pm).cpu().numpy()) and np.array_equal(a['mask_image'],torch.stack(im).cpu().numpy());rec=read(P/'records_PRIVATE'/(sid+'.json'));restore(m,S0);freeze91(m);features={};refs={};directions={};auxvalues={}
 with torch.no_grad():
  for v,(p,i) in enumerate([(None,None)]+list(zip(pm,im))):
   f=feature91(m,base,z,d,route,S0,p,i);calls+=1;old=fileload(P/'features_PRIVATE'/(sid+'_view'+str(v)+'.npz'));features['view'+str(v)]=f
   for k,t in f.items():
    if t.dtype==torch.bool:assert np.array_equal(t.cpu().numpy(),old[k])
    elif t.numel():fe=max(fe,float(abs(t.cpu().numpy()-old[k]).max()))
  for b in BASES91:
   for h in HEADS91:
    n=b+'_'+h;zv=[features['view'+str(i)] for i in range(3)];p,mean,f,e=mean_reads91(m,zv,route,SS[n],b,h);compare(p,scores[n+'_frozen']);compare(mean,scores[n+'_mean3']);compare(f,scores[n+'_feature3']);refs[n]=(p.detach(),e['projected_query'].detach(),e['conditional'].detach(),e['conditional_valid'].detach());old=fileload(P/'initial_PRIVATE'/(sid+'_'+n+'.npz'));assert np.array_equal(old['valid'],refs[n][3].cpu().numpy());loss_error=max(loss_error,float(abs(old['conditional']-refs[n][2].cpu().numpy()).max()) if old['conditional'].size else 0.)
 if not args.source:
  for name,p in legacy91(r).items():assert np.array_equal(p,scores[name])
 for wrong in [False,True]:
  g,raw,info=physical_direction91(m,base,z,d,S0,pm,im,ws,wrong);direction=-(1/(g.norm()+1e-6)).clamp_max(1)*g;directions[wrong]=(g,raw,direction,info);keys=sorted({t['patch_key'] for n,rr in rec.items() if (rr['kind'].startswith('wrong_physical') if wrong else rr['kind'].startswith('physical')) for t in rr['trials']})
  for key in keys:
   k=int(key.rsplit('_',1)[1]);alpha=GRID91[k];freeze91(m);theta_step91(m,S0,raw if k==0 else alpha*direction);s=fileload(P/'states_PRIVATE'/(sid+'_'+key+'.npz'));assert set(s)=={n for n,p in m.tta_parameters()};stateerr=max(stateerr,max(float(np.max(abs(p.detach().cpu().numpy()-s[n]))) for n,p in m.tta_parameters()));puttheta91(m,s)
   with torch.no_grad():
    f=feature91(m,base,z,d,route,S0);calls+=1;old=fileload(P/'features_PRIVATE'/(sid+'_'+key+'.npz'));features[key]=f;auxvalues[key]=float(uniform_aux91(m,z,d,pm,im,ws,wrong))
   for kk,t in f.items():
    if t.dtype==torch.bool:assert np.array_equal(t.cpu().numpy(),old[kk])
    elif t.numel():fe=max(fe,float(abs(t.cpu().numpy()-old[kk]).max()))
  for n,rr in rec.items():
   if not (rr['kind'].startswith('wrong_physical') if wrong else rr['kind'].startswith('physical')):continue
   b=n.split('_',1)[0];head=n[len(b)+1:];head=head[:head.rfind('_wrong_physical')] if wrong else head[:head.rfind('_physical')];basehead=b+'_'+head
   with torch.no_grad():
    for trial in rr['trials']:
     key=trial['patch_key'];p,e,_=scorer91(m,features[key],route,SS[basehead],b,head);actual=auxvalues[key]+float(keep91(p,e['projected_query'],refs[basehead][0],refs[basehead][1]));loss_error=max(loss_error,abs(actual-trial['objective']));checks+=1;alpha=trial['alpha'];expected=rr['before']+1e-4*float(g.double()@(raw if alpha==.5 else alpha*direction).double());assert abs(expected-trial['bound'])<1e-10;assert trial['accepted']==(trial['objective']<=trial['bound']+1e-10)
    key=rr['patch_key'];p,e,_=scorer91(m,features['view0' if key=='initial' else key],route,SS[basehead],b,head);compare(p,scores[n]);assert not rr['accepted_by_loss_rule'] or rr['after']<=rr['before']+1e-10;accepted+=int(rr['accepted_by_loss_rule']);assert rr['parameter_L2']>=0
 # Independent live conditional gradients and every attempted loss/patch.
 for b in BASES91:
  n=b+'_full_disc';restore(m,S0);setmatch91(m,SS[n]);copy_head91(m,SS[n]);enable_theta91(m);L,_,_=conditional_objective91(m,base,z,d,route,SS[n],b,refs[n],pm,im,ws);ps=[p for n,p in m.tta_parameters()];g=flat90(torch.autograd.grad(L,ps,allow_unused=True),ps);direction=-(1/(g.norm()+1e-6)).clamp_max(1)*g;rr=rec[n+'_conditional_bt'];loss_error=max(loss_error,abs(float(L.detach())-rr['before']));values={}
  for trial in rr['trials']:
   alpha=trial['alpha'];theta_step91(m,S0,alpha*direction);s=fileload(P/'states_PRIVATE'/(sid+'_'+trial['patch_key']+'.npz'));stateerr=max(stateerr,max(float(np.max(abs(p.detach().cpu().numpy()-s[k]))) for k,p in m.tta_parameters()));puttheta91(m,s)
   with torch.no_grad():val,p,parts=conditional_objective91(m,base,z,d,route,SS[n],b,refs[n],pm,im,ws)
   calls+=3;loss_error=max(loss_error,abs(float(val)-trial['objective']));checks+=1;bound=rr['before']+1e-4*float(g.double()@(alpha*direction).double());assert abs(bound-trial['bound'])<1e-10 and trial['accepted']==(trial['objective']<=trial['bound']+1e-10);values[trial['patch_key']]=p.detach()
  for kind in ['conditional_raw','conditional_bt']:
   r1=rec[n+'_'+kind];p=refs[n][0] if r1['patch_key']=='initial' else values[r1['patch_key']];compare(p,scores[n+'_'+kind]);assert not r1['accepted_by_loss_rule'] or r1['after']<=r1['before']+1e-10;accepted+=int(r1['accepted_by_loss_rule'])
 # Independent unused third-mask recovery values for every recorded main endpoint.
 hp,hi=masks(r,d,76003);hw=weight_sets({'id':'UNUSED_UNIFORM'},d,[hp],[hi],np.zeros(1024),np.zeros(1536))[0]['uniform'];done={};matches=[x for x in held if x['row']==('SOURCEVAL' if args.source else 'DEV')+str(nr+1).zfill(4)]
 for x in matches:
  key=rec[x['arm']]['patch_key']
  if key not in done:
   restore(m,S0)
   if key!='initial':puttheta91(m,fileload(P/'states_PRIVATE'/(sid+'_'+key+'.npz')))
   with torch.no_grad():done[key]=float(uniform_aux91(m,z,d,[hp],[hi],hw))
  loss_error=max(loss_error,abs(done[key]-x['heldout_true_physical_error']));checks+=1
 assert worst<1e-6 and fe<1e-6 and stateerr<=1e-7 and loss_error<1e-6 and topchanges==0
 if (nr+1)%32==0 or args.engineering:print('AUDIT91',label,nr+1,len(rows),flush=True)
assert fingerprint(base)==fp;folder=('engineering_source_audit' if args.source else 'engineering_target_audit') if args.engineering else ('independent_source_aux_audit_attempt001' if args.source else 'independent_native_audit_attempt001');O=H91/folder;O.mkdir();out={'state':'PASS_R91_NATIVE_FIXED_PHYSICAL_DIRECTION_STATE_OBJECTIVE_ALL_READOUTS_AND_HELDOUT','n':len(rows),'arms':len(arms),'saved_endpoint_replays':endpoints,'native_suffix_calls':calls,'actual_loss_scalar_checks':checks,'accepted_label_free_endpoints':accepted,'accepted_total_increases':0,'max_probability_error':worst,'feature_max_error':fe,'max_parameter_state_error':stateerr,'max_loss_error':loss_error,'top1_changes':topchanges,'probability_and_loss_tolerance_FP32':1e-6,'state_tolerance_FP32':1e-7,'original_A256_unchanged':True,'new_task_GT_scoring':0,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',out);print(out)
