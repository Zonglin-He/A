"""All-input source-prepared readout and real saved TTA state/objective replay."""
from common76 import *
from model76 import *
from target76 import objective,phash
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H/'experiment/TARGET_CODE_LOCK.json'));P=BASE/'target336_attempt001';ps=read(P/'SEALED.json');sr=BASE/'trained_source_attempt002';sel=read(sr/'SELECTED_SOURCE_ONLY.json');base=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);basefp=fingerprint(base);net={};S={}
for mode in MODES:
 m=Fusion(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S[mode]=load(ROOT/sel[mode]['state_path']);restore(m,S[mode]);net[mode]=m
rr=target_rows();by={pl:[r for r in rr if r['platform']==pl] for pl in ['drone','quad']};wrong={r['id']:by[pl][(i+1)%len(by[pl])] for pl in by for i,r in enumerate(by[pl])};maxprob=maxloss=0.;actualstates=accepted=0;calls=0;start=time.perf_counter();step_rows=[];aux_rows=[]
for i,r in enumerate(rr):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(P/rel)==ps['files'][rel];out=load(P/rel);z0=prefix(r);z=tensor_prefix(z0);d=tensor_data(load(casepath(r)),r);probs=dict(zip(out['arm_names'],out['scores']));predicted={};stepfile=P/'steps_PRIVATE'/(sid+'.json');assert sha(stepfile)==ps['files'][stepfile.relative_to(P).as_posix()];record=read(stepfile);pm=[];im=[]
 for rep in [76001,76002]:a,b=masks(r,d,rep);pm.append(a);im.append(b)
 for mode in MODES:
  pp,q,extras=net[mode].main(base,z,d,mode);calls+=1;predicted[mode+'_frozen']=pp.cpu().numpy();known=(d['p2i'][z['query_points_sample_inds'].long()]>=0).any(-1);assert (extras['late_RGB_delta'][~known]==0).all()
 bi=net['bidirectional'];refp=torch.tensor(predicted['bidirectional_frozen'],device='cuda');_,refq,_=bi.main(base,z,d,'bidirectional');calls+=1;beforehash=phash(bi);before_loss={w:float(objective(bi,base,z,d,refp,refq,pm,im,w)[0]) for w in [False,True]};calls+=2;pp,_,_=bi.main(base,z,d,'bidirectional',reverse_off=True);calls+=1;predicted['bidirectional_reverse_off']=pp.cpu().numpy();wi=tensor_data(load(casepath(wrong[sid])),wrong[sid])['image'];pp,_,_=bi.main(base,z,d,'bidirectional',image=wi);calls+=1;predicted['bidirectional_wrong_RGB']=pp.cpu().numpy();views=[refp]
 for a,b in zip(pm,im):pp,_,_=bi.main(base,z,d,'bidirectional',point_mask=a,image_mask=b);calls+=1;views.append(pp)
 predicted['bidirectional_frozen_budget3']=torch.stack(views).mean(0).cpu().numpy()
 for kind,recs in record['updates'].items():
  wh=kind=='wrong_interaction_binding';assert len(recs)==3 and recs[0]['before_sha256']==beforehash
  for j,rec in enumerate(recs):
   if j:assert rec['before_sha256']==recs[j-1]['after_sha256']
   if rec['accepted']:assert rec['loss_after']<=rec['loss_before']-max(1e-12,1e-6*abs(rec['loss_before'])) and rec['parameter_step_L2']>0
   else:assert rec['before_sha256']==rec['after_sha256'] and rec['optimizer_before_sha256']==rec['optimizer_after_sha256']
   assert not rec['target_GT_or_expert_used'];accepted+=int(rec['accepted']);step_rows.append({'row':f'R{i+1:05d}','platform':r['platform'],'branch':kind,'step':j+1,**{k:rec[k] for k in ['accepted','scale','loss_before','loss_after','aux_before','aux_after','keep_after','gradient_norm','parameter_step_L2']},'trial_count':len(rec['trials'])})
  assert abs(before_loss[wh]-recs[0]['loss_before'])<1e-9
  for arm,which in [('bidirectional_TTA1',0),('bidirectional_TTA3',2)] if not wh else [('bidirectional_TTA3_wrong_correspondence',2)]:
   restore(bi,S['bidirectional']);patchfile=P/'adapted_PRIVATE'/(sid+'_'+arm+'.npz');assert sha(patchfile)==ps['files'][patchfile.relative_to(P).as_posix()];patch=load(patchfile);tta=dict(bi.tta_parameters());assert set(patch)==set(tta)
   for n,v in tta.items():v.copy_(torch.tensor(patch[n],device='cuda'))
   assert phash(bi)==recs[which]['after_sha256'];actualstates+=1;L,a,k=objective(bi,base,z,d,refp,refq,pm,im,wh);calls+=1;maxloss=max(maxloss,abs(float(L)-recs[which]['loss_after']));assert maxloss<1e-9;pp,q,_=bi.main(base,z,d,'bidirectional');calls+=1;predicted[arm]=pp.cpu().numpy();aux_rows.append({'row':f'R{i+1:05d}','platform':r['platform'],'method':arm,'objective':float(L),'auxiliary':float(a),'preservation':float(k),'query_change_L2':float((q-refq).square().sum().sqrt())})
  restore(bi,S['bidirectional'])
 for k,v in predicted.items():maxprob=max(maxprob,float(abs(v-probs[k]).max()))
 assert maxprob<1e-10 and list(out['arm_names'])==ARMS;assert np.array_equal(out['source_boxes'],z0['boxes']) and np.array_equal(out['selected_original_boxes'],z0['boxes'][out['queries']]);assert np.array_equal(out['queries'],out['scores'].argmax(1));source=z0['scores'].astype(float);source/=source.sum();assert np.array_equal(source,probs['source'])
 hist=load(ROOT/'crossplatform/round66/predicted336_attempt001'/rel,['arm_names','scores']);hd=dict(zip(hist['arm_names'],hist['scores']));assert np.array_equal(probs['frozen_ABC'],hd['frozen_ABC']) and np.array_equal(probs['R59'],hd['R59'])
 if(i+1)%32==0:print('TARGET_AUDIT76',i+1,actualstates,'sec',round(time.perf_counter()-start,1),flush=True)
assert fingerprint(base)==basefp;out=H/'independent_target_audit_attempt001';assert not out.exists();out.mkdir();rec={'state':'PASS_R76_ALL336_FROZEN_BRANCHES_PHYSICAL_CONTROLS_AND_SAVED_ADAPTED_NATIVE_REPLAY','n':336,'arm_names':ARMS,'actual_adapted_states':actualstates,'optimizer_proposals':2016,'accepted':accepted,'accepted_objective_increases':0,'max_probability_error':maxprob,'max_objective_error':maxloss,'original_A256_unchanged':True,'unknown_late_RGB_delta_exact_zero':True,'source_selection_target_GT':False,'target_GT_or_expert_read':False,'prediction_seal_sha256':sha(P/'SEALED.json'),'audit_grounding_suffix_forwards':calls,'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',rec);dump(H/'ALL_OPTIMIZER_STEP_ROWS.json',step_rows);dump(H/'ALL_UNLABELED_ENDPOINT_ROWS.json',aux_rows);print(rec)
