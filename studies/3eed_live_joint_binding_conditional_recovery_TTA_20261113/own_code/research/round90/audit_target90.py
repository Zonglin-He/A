"""Independent native prepared readouts, all learned-router patches and actual objectives."""
from target90 import *
from views90 import frozen_reads90
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
P=B90/'prediction336_attempt001';se=read(P/'SEALED.json');sel=read(B90/'trained_source_attempt001/SELECTED_SOURCE_ONLY.json');S={b:load(ROOT/sel['models'][b]['state_path']) for b in BRANCHES90};base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);torch.manual_seed(0);m=Fusion90(base.contrastive_align_projection_image).cuda().eval();load_R76_90(m);initial=state(m);worst=loss_error=0.;endpoints=checks=accepted=0;held=[];start=time.perf_counter()
for nr,r in enumerate(read(H90/'experiment/DEV336_PRIVATE.json')):
 sid=r['id'];f=P/'samples_PRIVATE'/(sid+'.npz');assert sha(f)==se['files'][f.relative_to(P).as_posix()];a=load(f);scores=dict(zip(a['arm_names'],a['scores']));rec=read(P/'records_PRIVATE'/(sid+'.json'));z0,z,d,rt,route,pm,im,ws=data90(r);m.route=route;assert np.array_equal(a['source_boxes'],z0['boxes']) and np.array_equal(a['queries'],a['scores'].argmax(-1)) and np.array_equal(a['selected_original_boxes'],z0['boxes'][a['queries']]);assert np.array_equal(a['mask_point'],torch.stack(pm).cpu().numpy()) and np.array_equal(a['mask_image'],torch.stack(im).cpu().numpy())
 for name,b,variant,pool,kind in specs90():
  restore(m,S[b]);configure90(m,b,variant);m.pool_mode=pool;m.requires_grad_(False);perm=shuffle90(r) if kind=='shuffle' else None
  with torch.no_grad():refp,refq,e=m.main(base,z,d,'bidirectional');refs=(refp,refq,e['conditional'],e['conditional_valid'],e['local_reads'])
  if name in BRANCHES90+VARIANTS90:worst=max(worst,float(abs(refp.cpu().numpy()-scores[name+'_frozen']).max()));endpoints+=1
  if name in BRANCHES90:
   with torch.no_grad():_,mean,feature,_=frozen_reads90(m,base,z,d,pm,im)
   worst=max(worst,float(abs(mean.cpu().numpy()-scores[name+'_mean3']).max()),float(abs(feature.cpu().numpy()-scores[name+'_feature3']).max()));endpoints+=2
  enable_tta90(m)
  for j,t in enumerate(rec[name]):
   if j:assert abs(t['before']-rec[name][j-1]['after'])<1e-10
   for k,trial in enumerate(t['trials']):
    bound=t['before']-1e-4*.1*trial['scale']*t['directional_derivative'];assert trial['scale']==1/2**k and abs(trial['bound']-bound)<1e-10 and trial['accepted']==(trial['actual_total']<=bound+1e-10)
   if t['accepted']:assert t['parameter_L2']>0 and t['after']<=t['before']+1e-10;accepted+=1
   lv,_,_=objective90(m,base,z,d,*refs,pm,im,ws,kind,perm)
   loss_error=max(loss_error,abs(float(lv)-t['before']));checks+=1
   if j==1:
    enable_tta90(m);L,_,_=objective90(m,base,z,d,*refs,pm,im,ws,kind,perm);ps=[p for n,p in m.tta_parameters()];raw=flat90(torch.autograd.grad(L,ps,allow_unused=True),ps);g=raw*min(1.,1/(float(raw.norm())+1e-6));old=[p.detach().clone() for p in ps];apply90(m,old,-.1*t['scale']*g if t['accepted'] else g*0)
   else:
    arm=arm90(name,j+1);sf=P/'states_PRIVATE'/(sid+'_'+arm+'.npz');assert sha(sf)==se['files'][sf.relative_to(P).as_posix()];put90(m,load(sf))
   with torch.no_grad():val,p,parts=objective90(m,base,z,d,*refs,pm,im,ws,kind,perm)
   loss_error=max(loss_error,abs(float(val)-t['after']));checks+=1
   for k,v in parts.items():loss_error=max(loss_error,abs(float(v)-t['parts_after'][k]));checks+=1
   if j in [0,2]:worst=max(worst,float(abs(p.cpu().numpy()-scores[arm]).max()));endpoints+=1
  allowed={n for n,p in m.tta_parameters()};assert all(np.array_equal(v.detach().cpu().numpy(),S[b][k]) for k,v in m.state_dict().items() if k not in allowed)
 # Same prepared model, unused third missingness: no data-driven acceptance or tuning.
 restore(m,S['candidate_joint']);configure90(m,'candidate_joint');m.requires_grad_(False);hp,hi=mask89(r,d,{'point':rt['region_point'],'image':rt['region_image']},88003)
 with torch.no_grad():
  _,_,full=m.main(base,z,d,'bidirectional');_,_,bef=m.main(base,z,d,'bidirectional',point_mask=hp,image_mask=hi);put90(m,load(P/'states_PRIVATE'/(sid+'_candidate_joint_tta1.npz')));_,_,aft=m.main(base,z,d,'bidirectional',point_mask=hp,image_mask=hi)
 held.append({'row':'DEV'+str(nr+1).zfill(4),'platform':r['platform'],'third_mask_conditional_error_before':float(conditional_loss90(bef['conditional'],full['conditional'],bef['conditional_valid']&full['conditional_valid'])),'third_mask_conditional_error_after':float(conditional_loss90(aft['conditional'],full['conditional'],aft['conditional_valid']&full['conditional_valid']))});assert worst<1e-10 and loss_error<1e-10
 if (nr+1)%32==0:print('TARGET_AUDIT90',nr+1,flush=True)
assert fingerprint(base)==fp and accepted==read(P/'COUNTS.json')['accepted_updates'];O=H90/'independent_native_audit_attempt001';O.mkdir();out={'state':'PASS_R90_NATIVE_JOINT_PREPARATION_CONDITIONAL_TTA_AND_FIXED_BOXES','n':336,'arm_names':ARMS90,'saved_endpoint_replays':endpoints,'actual_loss_scalar_checks':checks,'accepted_updates':accepted,'accepted_total_increases':0,'max_probability_error':worst,'max_loss_error':loss_error,'original_A256_unchanged':True,'original_model_and_final_discriminator_frozen_at_test':True,'learned_router_updated_at_test':True,'all_unknown_and_fallback_retained':True,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start,'new_task_GT_scoring':0};dump(O/'AUDIT_RESULT.json',out);dump(H90/'HELDOUT_OBSERVATION_ROWS.json',held);print(out)
