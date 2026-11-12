"""All larger-cohort saved-state/native-score audit, without target labels."""
from extension_common78 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
plan=read(H78/'experiment/DESIGN_LOCK.json');pin(plan);pin(read(H78/'experiment/EXTENSION_CODE_LOCK.json'));selected=read(B78/'trained_source_attempt001/SELECTED_SOURCE_ONLY.json');base=load_source(read(ROOT/plan['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);net={};S={}
for b in ['joint','meta']:
 S[b]=load(ROOT/selected['models'][b]['state_path']);m=Fusion78(base.contrastive_align_projection_image).cuda().eval();restore(m,S[b]);enable_tta(m);net[b]=m
ref=Fusion78(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);restore(ref,load(ROOT/plan['init_state']));P=B78/'extension_prediction2304_attempt001';ss=read(P/'SEALED.json');rr=ext_rows();maxp=maxparam=0.;states=0;start=time.perf_counter();auxrows=[];calls=0
for f,h in ss['files'].items():assert sha(P/f)==h
for i,r in enumerate(rr):
 sid=r['id'];z0,z,d=ext_item(r);pm,im=maskpair(r,d);out=load(P/'samples_PRIVATE'/(sid+'.npz'));pred={};record=read(P/'records_PRIVATE'/(sid+'.json'))
 for b,m in net.items():
  restore(m,S[b]);enable_tta(m)
  with torch.no_grad():p0,q0,_=main78(m,base,z,d);views=[p0];pred[b+'0']=p0.cpu().numpy();calls+=1
  for a,c in zip(pm,im):
   with torch.no_grad():v,_,_=main78(m,base,z,d,point_mask=a,image_mask=c);views.append(v);calls+=1
  pred[b+'_direct3']=torch.stack(views).mean(0).cpu().numpy()
  for wrong in [False,True]:
   arm=b+('_wrong1' if wrong else '1');fast,L,gn=inner78(m,z,d,pm,im,wrong=wrong);patch=load(P/'states_PRIVATE'/(sid+'_'+arm+'.npz'));assert set(patch)=={n for n,_ in m.tta_parameters()}
   for n,p in m.tta_parameters():maxparam=max(maxparam,float(abs(patch[n]-fast[n].detach().cpu().numpy()).max()));assert maxparam<1e-7;p.data.copy_(torch.tensor(patch[n],device='cuda'))
   with torch.no_grad():
    p,q,_=main78(m,base,z,d);pred[arm]=p.cpu().numpy();calls+=1;after=float(auxiliary78(m,z,d,pm,im,wrong));delta=float(torch.sqrt(sum((v-torch.tensor(S[b][n],device='cuda')).square().sum() for n,v in m.tta_parameters())));assert abs(delta-record[arm]['parameter_step_L2'])<1e-9;auxrows.append({'row':f'E{i+1:05d}','platform':r['platform'],'arm':arm,'auxiliary_before':float(L.detach()),'auxiliary_after':after,'gradient_norm':float(gn.detach()),'parameter_step_L2':delta,'query_feature_change_L2':float((q-q0).norm()),'query_changed':int(p.argmax())!=int(p0.argmax())})
   assert all(np.array_equal(v.detach().cpu().numpy(),S[b][n]) for n,v in m.state_dict().items() if n not in patch);restore(m,S[b]);states+=1
 with torch.no_grad():v,_,_=main78(ref,base,z,d);pred['R76_bidirectional_frozen']=v.cpu().numpy();calls+=1
 for k,v in pred.items():maxp=max(maxp,float(abs(v-out['scores'][list(out['arm_names']).index(k)]).max()))
 assert maxp<1e-10 and np.array_equal(z0['boxes'],out['source_boxes']) and np.array_equal(out['queries'],out['scores'].argmax(1)) and np.array_equal(out['selected_original_boxes'],z0['boxes'][out['queries']]);prior=load(ROOT/'crossplatform/round69/predicted2304_attempt001/samples_PRIVATE'/(sid+'.npz'));pd=dict(zip(prior['arm_names'],prior['scores']))
 for a in ['source','frozen_ABC','R59']:assert np.array_equal(out['scores'][list(out['arm_names']).index(a)],pd[a])
 if(i+1)%128==0:print('AUDIT_EXT78',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
assert fingerprint(base)==fp;O=H78/'independent_extension_native_audit_attempt001';O.mkdir();dump(O/'AUDIT_RESULT.json',{'state':'PASS_R78_ALL2304_FIXED_EXTENSION_NATIVE_REPLAY_WITHOUT_TARGET_GT','n':2304,'target_states_replayed':states,'grounding_suffix_calls':calls,'max_probability_error':maxp,'max_regenerated_parameter_error':maxparam,'target_GT_or_FINAL_read':False,'source_selection_unchanged':True,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start});dump(H78/'ALL_EXTENSION_AUXILIARY_ENDPOINT_ROWS.json',auxrows);print(read(O/'AUDIT_RESULT.json'))
