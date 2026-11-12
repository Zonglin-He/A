"""Frozen source128+DEV336 unlabeled predictions; labels only in a later process."""
from core79 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H79/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H79/'experiment/RUN_CODE_LOCK.json'));assert read(H79/'engineering_attempt001/RESULT.json')['state'].startswith('PASS_');base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);out=B79/'prediction464_attempt001';assert not out.exists();out.mkdir()
for sub in ['samples_PRIVATE','states_PRIVATE','weights_PRIVATE','records_PRIVATE','source_deltas_PRIVATE']:(out/sub).mkdir()
rows=source_rows('VAL128')+target_rows();start=time.perf_counter();calls=0;backs=0;statuses=[]
for i,r in enumerate(rows):
 sid=r['id'];z0=prefix(r);z=tensor_prefix(z0);d=tensor_data(load(casepath(r)),r);pm,im=maskpair79(r,d);refp,refq,rp,ri,pr=sensitivity(m,base,z,d,z0['boxes']);calls+=1;backs+=pr['pair_count'];weights,fb=weight_sets(r,d,pm,im,rp,ri);probs={'source':z0['scores']/z0['scores'].sum(),'frozen_bi':refp.cpu().numpy()};views=[refp];records={'saliency':pr,'fallback_weight_sets':fb,'no_GT_for_sensitivity_or_update':True,'steps':{}}
 with torch.no_grad():
  for a,b in zip(pm,im):v,_,_=m.main(base,z,d,'bidirectional',point_mask=a,image_mask=b);views.append(v);calls+=1
 probs['frozen_direct3']=torch.stack(views).mean(0).cpu().numpy();gradient={};raw={}
 for kind,wrong in [('uniform',False),('query',False),('shuffle',False),('wrong',True)]:
  ws=weights['query' if wrong else kind];g,delta,record=gradient79(m,base,z,d,refp,refq,pm,im,ws,wrong);assert g.isfinite().all();gradient[kind]=g;raw[kind]=delta;records['steps'][kind]=record;calls+=1
 radius=raw['uniform'].norm();variants={'uniform_sgd1':raw['uniform'],'query_raw_sgd1':raw['query']}
 for name,key in [('query_radius_sgd1','query'),('shuffle_radius_sgd1','shuffle'),('wrong_binding_radius_sgd1','wrong')]:variants[name]=raw[key]*(radius/raw[key].norm()) if float(raw[key].norm())>0 else raw[key]
 for arm,delta in variants.items():
  set_step(m,S,delta)
  with torch.no_grad():
   pp,qq,_=m.main(base,z,d,'bidirectional');calls+=1;kind='uniform' if arm.startswith('uniform') else 'shuffle' if arm.startswith('shuffle') else 'wrong' if arm.startswith('wrong') else 'query';L,a,k=objective79(m,base,z,d,refp,refq,pm,im,weights['query' if kind=='wrong' else kind],kind=='wrong');calls+=1;actual=torch.cat([(v-torch.tensor(S[n],device='cuda')).flatten() for n,v in m.tta_parameters()]);probs[arm]=pp.cpu().numpy();records[arm]={'parameter_step_L2':float(actual.norm()),'uniform_radius':float(radius),'cosine_to_uniform_proposal':float(torch.nn.functional.cosine_similarity(delta.double(),raw['uniform'].double(),dim=0)) if float(delta.norm()*radius)>0 else None,'objective_after':float(L),'auxiliary_after':float(a),'preservation_after':float(k),'query_feature_delta_L2':float((qq-refq).norm()),'query_changed':int(pp.argmax())!=int(refp.argmax())}
  np.savez_compressed(out/'states_PRIVATE'/(sid+'_'+arm+'.npz'),**{n:v.detach().cpu().numpy() for n,v in m.tta_parameters()})
  if r['platform']=='waymo':np.savez_compressed(out/'source_deltas_PRIVATE'/(sid+'_'+arm+'.npz'),delta=actual.cpu().numpy())
  restore(m,S)
 if r['platform']=='waymo':
  old=load(ROOT/'crossplatform/round77/source128_attempt001/samples_PRIVATE'/(sid+'.npz'));od=dict(zip(old['arm_names'],old['scores']));assert np.max(abs(refp.cpu().numpy()-od['bidirectional_frozen']))<1e-10;probs['R76_Adam1']=od['aux1'];probs['R76_Adam3']=od['aux3'];pass
 else:
  old=load(ROOT/'crossplatform/round76/target336_attempt001/samples_PRIVATE'/(sid+'.npz'));od=dict(zip(old['arm_names'],old['scores']));assert np.max(abs(refp.cpu().numpy()-od['bidirectional_frozen']))<1e-10;probs.update(R76_Adam1=od['bidirectional_TTA1'],R76_Adam3=od['bidirectional_TTA3'],frozen_ABC=od['frozen_ABC'],R59=od['R59'])
 casearms=ARMS79[:-2] if r['platform']=='waymo' else ARMS79;ss=np.stack([probs[a] for a in casearms]);q=ss.argmax(1);np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(casearms),scores=ss,queries=q,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][q]);wp={'point_saliency':rp,'image_saliency':ri}
 for kind,ws in weights.items():
  for j,w in enumerate(ws):
   for key,v in w.items():wp[kind+'_'+str(j)+'_'+key]=v.cpu().numpy()
 np.savez_compressed(out/'weights_PRIVATE'/(sid+'.npz'),**wp);dump(out/'records_PRIVATE'/(sid+'.json'),records);statuses.append({'id':sid,'platform':r['platform'],'arm_statuses':{a:{'unknown':not bool(d['visible'].any()),'fallback':a in variants and not records[a]['parameter_step_L2']>0,'no_parameter_update':a not in variants or not records[a]['parameter_step_L2']>0} for a in casearms}})
 if(i+1)%16==0:print('PREDICT79',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
assert fingerprint(base)==fp;assert all(np.array_equal(v.detach().cpu().numpy(),S[n]) for n,v in m.state_dict().items());dump(out/'STATUS_PRIVATE.json',statuses);dump(out/'COUNTS.json',{'n':464,'source_n':128,'target_n':336,'arms':len(ARMS79),'grounding_suffix_calls':calls,'sensitivity_backward_calls':backs,'auxiliary_gradient_calls':464*4,'updated_states':464*5,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'GT_read':False,'source_training':False,'source_ABC_R59_not_available_not_emitted':True});seal(out,'R79_ALL_SOURCE_AND_TARGET_UNLABELED_PREDICTIONS_BEFORE_GT_ANALYSIS',n=464,arm_names=ARMS79);print(read(out/'COUNTS.json'))
