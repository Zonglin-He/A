"""Native saved-state, direct-view and source selection audit; no target labels."""
from core78 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
plan=read(H78/'experiment/DESIGN_LOCK.json');pin(plan);pin(read(H78/'experiment/TARGET_CODE_LOCK.json'));tr=B78/'trained_source_attempt001';seal0=read(tr/'SEALED.json');sel=read(tr/'SELECTED_SOURCE_ONLY.json');hist=read(tr/'HISTORY.json')
for f,h in seal0['files'].items():assert sha(tr/f)==h
chosen=max(range(1,5),key=lambda ep:(np.mean([x['one_mIoU_percent'] for x in hist if x['epoch']==ep]),-ep));assert chosen==sel['shared_epoch'];base=load_source(read(ROOT/plan['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S={};net={};maxp=0.;start=time.perf_counter();source=[]
for b in ['joint','meta']:
 S[b]=load(ROOT/sel['models'][b]['state_path']);m=Fusion78(base.contrastive_align_projection_image).cuda().eval();restore(m,S[b]);net[b]=m;record,details=evaluate_source(m,base,source_rows('VAL128'),H78/(b+'_SOURCE_REPLAY_PRIVATE.npz'));stored=load(tr/'val_PRIVATE'/(b+'_e'+str(chosen)+'.npz'));rep=load(H78/(b+'_SOURCE_REPLAY_PRIVATE.npz'));assert np.max(abs(rep['scores']-stored['scores']))<1e-8 and np.array_equal(rep['scores'].argmax(-1),stored['scores'].argmax(-1)) and np.array_equal(rep['IoU'],stored['IoU']);source.append({'branch':b,**record})
P=B78/'target336_attempt001';ss=read(P/'SEALED.json')
for f,h in ss['files'].items():assert sha(P/f)==h
states=0;auxrows=[]
for i,r in enumerate(target_rows()):
 sid=r['id'];z0,z,d=prepare_item(r);pm,im=maskpair(r,d);saved=load(P/'samples_PRIVATE'/(sid+'.npz'));record=read(P/'records_PRIVATE'/(sid+'.json'));pred={}
 for b,m in net.items():
  restore(m,S[b]);enable_tta(m)
  with torch.no_grad():p0,q0,_=main78(m,base,z,d);views=[p0];pred[b+'0']=p0.cpu().numpy()
  for a,c in zip(pm,im):
   with torch.no_grad():v,_,_=main78(m,base,z,d,point_mask=a,image_mask=c);views.append(v)
  pred[b+'_direct3']=torch.stack(views).mean(0).cpu().numpy()
  for wrong in [False,True]:
   arm=b+('_wrong1' if wrong else '1');fast,L,gn=inner78(m,z,d,pm,im,wrong=wrong);patch=load(P/'states_PRIVATE'/(sid+'_'+arm+'.npz'));assert set(patch)=={n for n,_ in m.tta_parameters()}
   for n,p in m.tta_parameters():assert np.max(abs(patch[n]-fast[n].detach().cpu().numpy()))<1e-7;p.data.copy_(torch.tensor(patch[n],device='cuda'))
   with torch.no_grad():
    p,q,_=main78(m,base,z,d);pred[arm]=p.cpu().numpy();after=float(auxiliary78(m,z,d,pm,im,wrong));delta=float(torch.sqrt(sum((v-torch.tensor(S[b][n],device='cuda')).square().sum() for n,v in m.tta_parameters())));assert abs(delta-record[arm]['parameter_step_L2'])<1e-9;auxrows.append({'row':f'R{i+1:05d}','platform':r['platform'],'arm':arm,'auxiliary_before':float(L.detach()),'auxiliary_after':after,'gradient_norm':float(gn.detach()),'parameter_step_L2':delta,'query_feature_change_L2':float((q-q0).norm()),'query_changed':int(p.argmax())!=int(p0.argmax())})
   assert all(np.array_equal(v.detach().cpu().numpy(),S[b][n]) for n,v in m.state_dict().items() if n not in patch);restore(m,S[b]);states+=1
 for k,v in pred.items():maxp=max(maxp,float(abs(v-saved['scores'][list(saved['arm_names']).index(k)]).max()))
 assert maxp<1e-10 and np.array_equal(saved['source_boxes'],z0['boxes']) and np.array_equal(saved['queries'],saved['scores'].argmax(1)) and np.array_equal(saved['selected_original_boxes'],z0['boxes'][saved['queries']]);prior=load(ROOT/'crossplatform/round76/target336_attempt001/samples_PRIVATE'/(sid+'.npz'));pd=dict(zip(prior['arm_names'],prior['scores']))
 for a,b in [('source','source'),('frozen_ABC','frozen_ABC'),('R59','R59'),('R76_bidirectional_frozen','bidirectional_frozen')]:assert np.array_equal(saved['scores'][list(saved['arm_names']).index(a)],pd[b])
 if(i+1)%64==0:print('AUDIT78',i+1,round(time.perf_counter()-start,1),flush=True)
# Post-training source gradient diagnostics remain separate from target labels.
gradient_rows=[]
for b,m in net.items():
 restore(m,S[b]);enable_tta(m)
 for i,r in enumerate(source_rows('VAL128')):
  _,z,d=prepare_item(r);y=source_y(r);pm,im=maskpair(r,d);pp,_,_=main78(m,base,z,d);params=[p for _,p in m.tta_parameters()];gg=torch.autograd.grad(ground78(pp,y),params,allow_unused=True);g=torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p,v in zip(params,gg)]).detach();fast,aux,gn=inner78(m,z,d,pm,im);delta=torch.cat([(fast[n]-p).flatten() for n,p in m.tta_parameters()]).detach();den=float(g.norm()*delta.norm());dot=float(g.dot(delta));gradient_rows.append({'row':f'S{i+1:05d}','branch':b,'source_ground_gradient_L2':float(g.norm()),'actual_SGD_step_L2':float(delta.norm()),'ground_gradient_dot_SGD_delta':dot,'aux_ground_gradient_cosine':-dot/den if den>0 else None})
assert fingerprint(base)==fp;out=H78/'independent_native_audit_attempt001';assert not out.exists();out.mkdir();dump(out/'AUDIT_RESULT.json',{'state':'PASS_R78_ALL_SOURCE_SELECTED_ENDPOINTS_AND_ALL336_NATIVE_TTA_STATES','n':336,'source_val_inputs_replayed':256,'source_gradients_diagnosed':256,'source_selected_shared_epoch':chosen,'source_selection_target_labels':False,'target_states_replayed':states,'max_probability_error':maxp,'fixed_source_baselines_exact':True,'target_GT_or_FINAL_read':False,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start});dump(H78/'SOURCE_SELECTED_REPLAY_SUMMARY.json',source);dump(H78/'ALL_TARGET_AUXILIARY_ENDPOINT_ROWS.json',auxrows);dump(H78/'SOURCE_GRADIENT_SCALAR_ROWS.json',gradient_rows);print(read(out/'AUDIT_RESULT.json'))
