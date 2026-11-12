"""No task labels: four core endpoints on fixed exposed extension rosters."""
from core80 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H80/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H80/'experiment/CODE_LOCK.json'));assert read(H80/'engineering_attempt001/RESULT.json')['state'].startswith('PASS_');base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);O=B80/'prediction2640_attempt001';assert not O.exists();O.mkdir(parents=True)
for sub in ['samples_PRIVATE','states_PRIVATE','records_PRIVATE']:(O/sub).mkdir()
statuses=[];start=time.perf_counter();calls=gradcalls=updates=0
for i,r in enumerate(allrows80()):
 sid=r['id'];dev=sid in DEV_IDS;z0,z,d=item80(r);pm,im=maskpair79(r,d);ws=uniform_weights80(d,pm,im);old,od=old80(r);assert np.array_equal(old['source_boxes'],z0['boxes']);probs={k:od[k] for k in ['source','frozen_ABC','R59']};rec={}
 with torch.no_grad():refp,refq,_=m.main(base,z,d,'bidirectional');calls+=1
 if dev:
  assert np.max(abs(refp.cpu().numpy()-od['frozen_bi']))<1e-10
  probs.update({k:od[k] for k in ['frozen_bi','frozen_direct3','uniform_sgd1']});rec['uniform_sgd1']=read(ROOT/'crossplatform/round79/prediction464_attempt001/records_PRIVATE'/(sid+'.json'))['uniform_sgd1'];rec['uniform_sgd1']['reused_R79_sealed']=True
 else:
  probs['frozen_bi']=refp.cpu().numpy();views=[refp]
  with torch.no_grad():
   for a,b in zip(pm,im):v,_,_=m.main(base,z,d,'bidirectional',point_mask=a,image_mask=b);views.append(v);calls+=1
  probs['frozen_direct3']=torch.stack(views).mean(0).cpu().numpy()
 for arm,wrong in ([('uniform_wrong_sgd1',True)] if dev else [('uniform_sgd1',False),('uniform_wrong_sgd1',True)]):
  g,delta,before=gradient79(m,base,z,d,refp,refq,pm,im,ws,wrong);calls+=1;gradcalls+=1;assert torch.isfinite(g).all();set_step(m,S,delta)
  with torch.no_grad():
   pp,q,_=m.main(base,z,d,'bidirectional');calls+=1;L,aux,keep=objective79(m,base,z,d,refp,refq,pm,im,ws,wrong);calls+=1;actual=torch.cat([(v-torch.tensor(S[n],device='cuda')).flatten() for n,v in m.tta_parameters()]);probs[arm]=pp.cpu().numpy();rec[arm]={**before,'parameter_step_L2':float(actual.norm()),'objective_after':float(L),'auxiliary_after':float(aux),'preservation_after':float(keep),'query_feature_delta_L2':float((q-refq).norm()),'query_changed':int(pp.argmax())!=int(refp.argmax()),'wrong_binding':wrong,'reused_R79_sealed':False}
  np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+arm+'.npz'),**{n:v.detach().cpu().numpy() for n,v in m.tta_parameters()});updates+=1;restore(m,S)
 ss=np.stack([probs[k] for k in ARMS80]);qq=ss.argmax(1);assert np.isfinite(ss).all();np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS80),scores=ss,queries=qq,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][qq]);dump(O/'records_PRIVATE'/(sid+'.json'),rec);statuses.append({'id':sid,'arm_statuses':{a:{'unknown':not bool(d['visible'].any()) if a in ['frozen_bi','frozen_direct3','uniform_sgd1','uniform_wrong_sgd1'] else False,'fallback':bool(a in rec and not rec[a]['parameter_step_L2']>0),'no_parameter_update':not(a in rec and rec[a]['parameter_step_L2']>0)} for a in ARMS80}})
 if(i+1)%128==0:print('R80_PREDICT',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
assert fingerprint(base)==fp;assert all(np.array_equal(v.detach().cpu().numpy(),S[n]) for n,v in m.state_dict().items());cost={'n':2640,'new_extension_n':2304,'DEV336_core_endpoints_reused':3,'new_updated_states':updates,'grounding_suffix_calls':calls,'auxiliary_parameter_gradient_calls':gradcalls,'sensitivity_calls':0,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'source_training':False,'GT_read':False,'FINAL_access':False};dump(O/'COUNTS.json',cost);dump(O/'READOUT_STATUS_PRIVATE.json',statuses);seal(O,'R80_FIXED_R76_UNIFORM_RECOVERY_2640_SEALED_BEFORE_TASK_LABEL_JOIN',n=2640,arm_names=ARMS80);print(cost)
