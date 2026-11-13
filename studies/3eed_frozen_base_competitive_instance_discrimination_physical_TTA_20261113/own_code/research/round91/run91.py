"""Sealed source/DEV endpoints; native features shared only across immutable readout heads."""
from targets91 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--source',action='store_true');ap.add_argument('--engineering',action='store_true');args=ap.parse_args();selection=read(B91/'trained_source_attempt001/SELECTED_SOURCE_ONLY.json');SS={n:load(ROOT/v['state_path']) for n,v in selection['models'].items()};base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);torch.manual_seed(0);m=Fusion91(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S0=loadbase91(m,'R76');rows=source_rows('VAL128') if args.source else read(H91/'experiment/DEV336_PRIVATE.json');arms=NEW91 if args.source else LEGACY91+NEW91
if args.engineering:rows=rows[:2] if args.source else [next(r for r in rows if r['platform']==pl) for pl in ['drone','quad']]
label=('engineering_' if args.engineering else '')+('source128' if args.source else 'prediction336')+'_attempt001';O=B91/label
for f in ['samples_PRIVATE','records_PRIVATE','states_PRIVATE','features_PRIVATE','initial_PRIVATE']:(O/f).mkdir(parents=True,exist_ok=False)
start=time.perf_counter();statuses=[];held=[];direction_checks=[];counts={'n':len(rows),'arms':len(arms),'native_feature_suffix_calls':0,'conditional_gradient_native_suffix_calls':0,'conditional_trial_native_suffix_calls':0,'actual_unique_saved_theta_states':0};allnames=[b+'_'+h for b in BASES91 for h in HEADS91]
for nr,r in enumerate(rows):
 sid=r['id'];z0,z,d,rt,route,pm,im,ws=data91(r,args.source);restore(m,S0);freeze91(m);results={} if args.source else legacy91(r);rec={};refs={};status={};savedpatch={};features={};phy_by_key={};unknown=len(rt['token_roles'])==0
 def savepatch(key):
  if key not in savedpatch:
   v=theta_patch91(m);savedpatch[key]=v;np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+key+'.npz'),**v);counts['actual_unique_saved_theta_states']+=1
 def normpatch(key):return float(np.sqrt(sum(np.sum((v-S0[n])**2) for n,v in savedpatch[key].items())))
 def storefeature(key,f):
  features[key]={k:v.detach().cpu().numpy().copy() for k,v in f.items()};np.savez_compressed(O/'features_PRIVATE'/(sid+'_'+key+'.npz'),**features[key])
 with torch.no_grad():
  views=[feature91(m,base,z,d,route,S0,a,b) for a,b in [(None,None)]+list(zip(pm,im))];counts['native_feature_suffix_calls']+=3
  for v,f in enumerate(views):storefeature('view'+str(v),f)
  aux0=float(uniform_aux91(m,z,d,pm,im,ws));
  for b in BASES91:
   for h in HEADS91:
    name=b+'_'+h;p,mean,feature,e=mean_reads91(m,views,route,SS[name],b,h);results[name+'_frozen']=p.cpu().numpy();results[name+'_mean3']=mean.cpu().numpy();results[name+'_feature3']=feature.cpu().numpy();refs[name]=(p.detach(),e['projected_query'].detach(),e['conditional'].detach(),e['conditional_valid'].detach());np.savez_compressed(O/'initial_PRIVATE'/(sid+'_'+name+'.npz'),probability=p.cpu().numpy(),query=e['projected_query'].cpu().numpy(),conditional=e['conditional'].cpu().numpy(),valid=e['conditional_valid'].cpu().numpy())
 # Every base has identical original theta/physical heads; one original R80 direction suffices.
 for wrong in [False,True]:
  g,raw,initial=physical_direction91(m,base,z,d,S0,pm,im,ws,wrong);direction=-(1/(g.norm()+1e-6)).clamp_max(1)*g;before=initial['objective_before'];names=[b+'_full_disc' for b in BASES91] if wrong else allnames;pending=set(names);traces={n:[] for n in names};rawkind='wrong_physical_raw' if wrong else 'physical_raw';btkind='wrong_physical_bt' if wrong else 'physical_bt';prefix='wrong' if wrong else 'physical'
  if not args.source and not wrong:
   old=load(ROOT/'crossplatform/round79/prediction464_attempt001/states_PRIVATE'/(sid+'_uniform_sgd1.npz'));theta_step91(m,S0,raw);err=max(float(np.max(abs(p.detach().cpu().numpy()-old[n]))) for n,p in m.tta_parameters());assert err<=1e-7;direction_checks.append({'row':'DEV'+str(nr+1).zfill(4),'native_R80_raw_parameter_max_error':err})
  for k,alpha in enumerate(GRID91):
   freeze91(m);theta_step91(m,S0,raw if k==0 else alpha*direction);key=prefix+'_'+str(k);savepatch(key)
   with torch.no_grad():
    f=feature91(m,base,z,d,route,S0);counts['native_feature_suffix_calls']+=1;storefeature(key,f);aux=float(uniform_aux91(m,z,d,pm,im,ws,wrong));phy_by_key[key]=aux
    for name in list(pending|set(names if k==0 else [])):
     b,h=name.split('_',1);p,e,_=scorer91(m,f,route,SS[name],b,h);keep=float(keep91(p,e['projected_query'],refs[name][0],refs[name][1]));total=aux+keep;bound=before+1e-4*float(g.double()@(raw if k==0 else alpha*direction).double());ok=np.isfinite(total) and total<=bound+1e-10;trial={'alpha':alpha,'physical':aux,'keep':keep,'objective':total,'bound':bound,'accepted':bool(ok),'patch_key':key,'feature_key':key}
     if k==0:results[name+'_'+rawkind]=p.cpu().numpy();rec[name+'_'+rawkind]={'kind':rawkind,'before':before,'after':total,'accepted_by_loss_rule':False,'update_applied':True,'alpha':alpha,'parameter_L2':normpatch(key),'patch_key':key,'feature_key':key,'gradient_norm':float(g.norm()),'trials':[trial]}
     if name in pending:
      traces[name].append(trial)
      if ok:
       results[name+'_'+btkind]=p.cpu().numpy();rec[name+'_'+btkind]={'kind':btkind,'before':before,'after':total,'accepted_by_loss_rule':True,'update_applied':True,'alpha':alpha,'parameter_L2':normpatch(key),'patch_key':key,'feature_key':key,'gradient_norm':float(g.norm()),'trials':traces[name]};pending.remove(name)
   if not pending:break
  for name in pending:
   results[name+'_'+btkind]=results[name+'_frozen'].copy();rec[name+'_'+btkind]={'kind':btkind,'before':before,'after':before,'accepted_by_loss_rule':False,'update_applied':False,'alpha':0.,'parameter_L2':0.,'patch_key':'initial','feature_key':'view0','gradient_norm':float(g.norm()),'trials':traces[name]}
 # Role-conditioned comparison: same source, views, theta scope, clip and alpha grid.
 for b in BASES91:
  name=b+'_full_disc';restore(m,S0);setmatch91(m,SS[name]);copy_head91(m,SS[name]);enable_theta91(m);L,_,parts=conditional_objective91(m,base,z,d,route,SS[name],b,refs[name],pm,im,ws);counts['conditional_gradient_native_suffix_calls']+=3;ps=[p for n,p in m.tta_parameters()];g=flat90(torch.autograd.grad(L,ps,allow_unused=True),ps);direction=-(1/(g.norm()+1e-6)).clamp_max(1)*g;before=float(L.detach());trials=[];accepted=False
  for k,alpha in enumerate(GRID91):
   theta_step91(m,S0,alpha*direction);key=b+'_conditional_'+str(k);savepatch(key)
   with torch.no_grad():val,p,pp=conditional_objective91(m,base,z,d,route,SS[name],b,refs[name],pm,im,ws)
   counts['conditional_trial_native_suffix_calls']+=3;bound=before+1e-4*float(g.double()@(alpha*direction).double());ok=np.isfinite(float(val)) and float(val)<=bound+1e-10;trial={'alpha':alpha,'objective':float(val),'bound':bound,'accepted':bool(ok),'patch_key':key,**{k:float(v) for k,v in pp.items()}};trials.append(trial)
   if k==0:results[name+'_conditional_raw']=p.cpu().numpy();rec[name+'_conditional_raw']={'kind':'conditional_raw','before':before,'after':float(val),'accepted_by_loss_rule':False,'update_applied':True,'alpha':alpha,'parameter_L2':normpatch(key),'patch_key':key,'gradient_norm':float(g.norm()),'trials':[trial]}
   if ok:
    results[name+'_conditional_bt']=p.cpu().numpy();rec[name+'_conditional_bt']={'kind':'conditional_bt','before':before,'after':float(val),'accepted_by_loss_rule':True,'update_applied':True,'alpha':alpha,'parameter_L2':normpatch(key),'patch_key':key,'gradient_norm':float(g.norm()),'trials':trials};accepted=True;break
  if not accepted:results[name+'_conditional_bt']=results[name+'_frozen'].copy();rec[name+'_conditional_bt']={'kind':'conditional_bt','before':before,'after':before,'accepted_by_loss_rule':False,'update_applied':False,'alpha':0.,'parameter_L2':0.,'patch_key':'initial','gradient_norm':float(g.norm()),'trials':trials}
 # Third original mask is diagnostic only, never used in fitting/acceptance.
 hp,hi=masks(r,d,76003);hw=weight_sets({'id':'UNUSED_UNIFORM'},d,[hp],[hi],np.zeros(1024),np.zeros(1536))[0]['uniform'];cacheheld={}
 for name in [b+'_full_disc' for b in BASES91]:
  for kind in ['physical_raw','physical_bt','conditional_raw','conditional_bt','wrong_physical_raw','wrong_physical_bt']:
   key=rec[name+'_'+kind]['patch_key']
   if key not in cacheheld:
    restore(m,S0)
    if key!='initial':puttheta91(m,savedpatch[key])
    with torch.no_grad():cacheheld[key]=float(uniform_aux91(m,z,d,[hp],[hi],hw))
   held.append({'row':('SOURCEVAL' if args.source else 'DEV')+str(nr+1).zfill(4),'platform':r['platform'],'arm':name+'_'+kind,'heldout_true_physical_error':cacheheld[key]})
 for a in arms:
  if a in NEW91:
   spatial=not ('_query_disc_' in a);noup=a.endswith(('_frozen','_mean3','_feature3'));rr=rec.get(a,{});status[a]={'unknown':bool(unknown and spatial),'fallback':bool(unknown and spatial) or (not noup and not rr.get('update_applied',False)),'no_parameter_update':noup or not rr.get('update_applied',False)}
  else:status[a]=legacy_status91(r)[a]
 restore(m,S0);freeze91(m);scores=np.stack([results[a] for a in arms]);assert np.isfinite(scores).all() and np.allclose(scores.sum(-1),1);q=scores.argmax(-1);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(arms),scores=scores,queries=q,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][q],mask_point=torch.stack(pm).cpu().numpy(),mask_image=torch.stack(im).cpu().numpy());dump(O/'records_PRIVATE'/(sid+'.json'),rec);statuses.append({'id':sid,'arm_statuses':status})
 if (nr+1)%16==0 or args.engineering:print('RUN91',label,nr+1,len(rows),flush=True)
assert fingerprint(base)==fp;counts.update(seconds=time.perf_counter()-start,peak_GPU_bytes=int(torch.cuda.max_memory_allocated()),target_task_GT_read=False,source_labels_for_inner_update=False,original_A256_unchanged=True,original_scope_parameters=sum(p.numel() for n,p in m.tta_parameters()),head_and_matching_parameters_updated=False,shared_native_features_only_when_theta_state_identical=True);dump(O/'COUNTS.json',counts);dump(O/'READOUT_STATUS_PRIVATE.json',statuses);dump(O/'HELDOUT_ROWS.json',held);dump(O/'R80_DIRECTION_PARITY_ROWS.json',direction_checks);seal(O,'R91_FIXED_FROZEN_BASE_DISCRIMINATION_PHYSICAL_AND_CONDITIONAL_ENDPOINTS_SEALED',n=len(rows),arm_names=arms);print(counts)
