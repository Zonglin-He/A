"""Independent native source adaptation replay and source-GT directional diagnosis."""
from adapt90 import *
from views90 import frozen_reads90
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
P=B90/'source_aux128_attempt001';seal0=read(P/'SEALED.json');sel=read(B90/'trained_source_attempt001/SELECTED_SOURCE_ONLY.json');S={b:load(ROOT/sel['models'][b]['state_path']) for b in BRANCHES90};base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);torch.manual_seed(0);m=Fusion90(base.contrastive_align_projection_image).cuda().eval();load_R76_90(m);start=time.perf_counter();error=loss_error=0.;checks=accepted=0;direction=[];held=[]
for i,r in enumerate(source_rows('VAL128')):
 z0,z,d,rt,route,pm,im,ws=data90(r,True);m.route=route;a=load(P/'samples_PRIVATE'/(r['id']+'.npz'));scores=dict(zip(a['arm_names'],a['scores']));rec=read(P/'records_PRIVATE'/(r['id']+'.json'))
 for name,tr in rec.items():
  b=name if name in BRANCHES90 else 'candidate_joint';kind='full' if name in BRANCHES90 else name.split('candidate_joint_',1)[1];restore(m,S[b]);configure90(m,b);m.requires_grad_(False);perm=shuffle90(r) if kind=='shuffle' else None
  with torch.no_grad():refp,refq,e=m.main(base,z,d,'bidirectional');refs=(refp,refq,e['conditional'],e['conditional_valid'],e['local_reads']);lv,_,_=objective90(m,base,z,d,*refs,pm,im,ws,kind,perm)
  loss_error=max(loss_error,abs(float(lv)-tr[0]['before']));checks+=1
  if name in BRANCHES90:
   error=max(error,float(abs(refp.cpu().numpy()-scores[b+'_frozen']).max()))
   with torch.no_grad():_,mean,feature,_=frozen_reads90(m,base,z,d,pm,im)
   error=max(error,float(abs(mean.cpu().numpy()-scores[b+'_mean3']).max()),float(abs(feature.cpu().numpy()-scores[b+'_feature3']).max()))
  for j,t in enumerate(tr):
   if j:assert abs(t['before']-tr[j-1]['after'])<1e-10
   for k,trial in enumerate(t['trials']):assert trial['scale']==1/2**k and abs(trial['bound']-(t['before']-1e-4*.1*trial['scale']*t['directional_derivative']))<1e-10
   if t['accepted']:assert t['after']<=t['before']+1e-10 and t['parameter_L2']>0;accepted+=1
   if j==1:
    enable_tta90(m);ll,_,_=objective90(m,base,z,d,*refs,pm,im,ws,kind,perm);ps=[p for n,p in m.tta_parameters()];raw=flat90(torch.autograd.grad(ll,ps,allow_unused=True),ps);g=raw*min(1.,1/(float(raw.norm())+1e-6));old=[p.detach().clone() for p in ps];apply90(m,old,-.1*t['scale']*g if t['accepted'] else g*0);m.requires_grad_(False)
   else:
    arm=name+'_tta'+str(j+1) if name in BRANCHES90 else name+'1';put90(m,load(P/'states_PRIVATE'/(r['id']+'_'+arm+'.npz')));m.requires_grad_(False)
   with torch.no_grad():val,p,parts=objective90(m,base,z,d,*refs,pm,im,ws,kind,perm)
   loss_error=max(loss_error,abs(float(val)-t['after']));checks+=1
   if j in [0,2]:error=max(error,float(abs(p.cpu().numpy()-scores[arm]).max()))
 for variant in ['no_ref','soft_pool']:
  restore(m,S['candidate_joint']);configure90(m,'candidate_joint','no_ref' if variant=='no_ref' else 'full');m.pool_mode='soft' if variant=='soft_pool' else 'strict';m.requires_grad_(False)
  with torch.no_grad():pp,_,_=m.main(base,z,d,'bidirectional')
  error=max(error,float(abs(pp.cpu().numpy()-scores[variant+'_frozen']).max()))
 # Once predictionseal exists, source labels may diagnose gradient alignment, never decide updates.
 b='candidate_joint';restore(m,S[b]);configure90(m,b);m.requires_grad_(False)
 with torch.no_grad():refp,refq,e=m.main(base,z,d,'bidirectional');refs=(refp,refq,e['conditional'],e['conditional_valid'],e['local_reads'])
 enable_tta90(m);ps=[p for n,p in m.tta_parameters()];y=torch.tensor(load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'],device='cuda',dtype=torch.float64);p,_,_=m.main(base,z,d,'bidirectional');gg=flat90(torch.autograd.grad(ground89(p,y),ps,allow_unused=True),ps)
 for kind in ['full','bypass','physical']:
  L,_,_=objective90(m,base,z,d,*refs,pm,im,ws,kind);ga=flat90(torch.autograd.grad(L,ps,allow_unused=True),ps);den=float(ga.norm()*gg.norm());direction.append({'row':'SOURCEVAL'+str(i+1).zfill(4),'kind':kind,'cosine_auxiliary_vs_source_ground_gradient':float(ga@gg)/den if den>1e-12 else None,'source_ground_gradient_norm':float(gg.norm()),'auxiliary_gradient_norm':float(ga.norm()),'source_labels_for_diagnostic_only':True})
 m.requires_grad_(False);hp,hi=mask89(r,d,{'point':rt['region_point'],'image':rt['region_image']},89003)
 with torch.no_grad():_,_,bef=m.main(base,z,d,'bidirectional',point_mask=hp,image_mask=hi);put90(m,load(P/'states_PRIVATE'/(r['id']+'_candidate_joint_tta1.npz')));_,_,aft=m.main(base,z,d,'bidirectional',point_mask=hp,image_mask=hi);held.append({'row':'SOURCEVAL'+str(i+1).zfill(4),'third_mask_conditional_error_before':float(conditional_loss90(bef['conditional'],refs[2],refs[3]&bef['conditional_valid'])),'third_mask_conditional_error_after':float(conditional_loss90(aft['conditional'],refs[2],refs[3]&aft['conditional_valid']))})
 assert error<1e-10 and loss_error<1e-10
 if (i+1)%32==0:print('SOURCE_AUX_AUDIT90',i+1,flush=True)
assert fingerprint(base)==fp and accepted==read(P/'COUNTS.json')['accepted_updates'];O=H90/'independent_source_aux_audit_attempt001';O.mkdir();out={'state':'PASS_R90_SOURCE_UNLABELED_INNER_UPDATES_NATIVE_REPLAY','n':128,'accepted_updates':accepted,'actual_loss_checks':checks,'max_probability_error':error,'max_loss_error':loss_error,'labels_never_used_for_inner_updates_or_selection':True,'source_GT_direction_rows':len(direction),'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',out);dump(H90/'SOURCE_AUX_DIRECTION_ROWS.json',direction);dump(H90/'SOURCE_HELDOUT_MASK_ROWS.json',held);print(out)
