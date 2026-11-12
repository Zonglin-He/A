"""Independent all-input state regeneration/replay before task-label access."""
from core80 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H80/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H80/'experiment/AUDIT_CODE_LOCK.json'));O=B80/'prediction2640_attempt001';ss=read(O/'SEALED.json')
for f,h in ss['files'].items():assert sha(O/f)==h
sealed_cache_dirs=['crossplatform/round79/prediction464_attempt001','crossplatform/round69/predicted2304_attempt001','crossplatform/round69/AB2304_attempt001','crossplatform/round78/extension_geometry2304_attempt001']
cachepins={f:read(ROOT/f/'SEALED.json')['files'] for f in sealed_cache_dirs}
base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);start=time.perf_counter();maxp=maxparam=0.;states=0;endpoints=[]
for i,r in enumerate(allrows80()):
 sid=r['id'];dev=sid in DEV_IDS
 for cache in (sealed_cache_dirs[:1] if dev else sealed_cache_dirs[1:]):
  rel='samples_PRIVATE/'+sid+'.npz';assert sha(ROOT/cache/rel)==cachepins[cache][rel]
 z0,z,d=item80(r);pm,im=maskpair79(r,d);ws=uniform_weights80(d,pm,im);saved=load(O/'samples_PRIVATE'/(sid+'.npz'));rec=read(O/'records_PRIVATE'/(sid+'.json'));old,od=old80(r);pred={a:od[a] for a in ['source','frozen_ABC','R59']}
 with torch.no_grad():
  refp,refq,_=m.main(base,z,d,'bidirectional');pred['frozen_bi']=refp.cpu().numpy();views=[refp]
  for a,b in zip(pm,im):v,_,_=m.main(base,z,d,'bidirectional',point_mask=a,image_mask=b);views.append(v)
  pred['frozen_direct3']=torch.stack(views).mean(0).cpu().numpy()
 for arm,wrong in [('uniform_sgd1',False),('uniform_wrong_sgd1',True)]:
  # Independently form uniform native losses instead of calling gradient79.
  enable79(m);pp,q,_=m.main(base,z,d,'bidirectional');keep=(refp*(refp.clamp_min(1e-300).log()-pp.clamp_min(1e-300).log())).sum()+.1*(q-refq).square().sum(-1).mean()/2;aux=torch.stack([m.auxiliary(z,d,'bidirectional',a,b,wrong)[0] for a,b in zip(pm,im)]).mean();loss=aux.double()+keep;params=m.tta_parameters();gg=torch.autograd.grad(loss,[v for _,v in params],allow_unused=True);g=torch.cat([(torch.zeros_like(v) if a is None else a).flatten() for (_,v),a in zip(params,gg)]).detach();delta=-.5*(1/(g.norm()+1e-6)).clamp_max(1)*g
  patchpath=(ROOT/'crossplatform/round79/prediction464_attempt001/states_PRIVATE' if dev and arm=='uniform_sgd1' else O/'states_PRIVATE')/(sid+'_'+arm+'.npz');patch=load(patchpath);off=0
  with torch.no_grad():
   for n,v in params:
    nn=v.numel();expected=torch.tensor(S[n],device='cuda')+delta[off:off+nn].reshape_as(v);maxparam=max(maxparam,float(abs(expected.cpu().numpy()-patch[n]).max()));v.copy_(torch.tensor(patch[n],device='cuda'));off+=nn
   # Native mean and uniform-sum aggregation differ by small reduction roundoff.
   assert maxparam<1e-7;pp,q,_=m.main(base,z,d,'bidirectional');pred[arm]=pp.cpu().numpy();actual=torch.cat([(v-torch.tensor(S[n],device='cuda')).flatten() for n,v in params]);assert abs(float(actual.norm())-rec[arm]['parameter_step_L2'])<1e-8
   assert all(np.array_equal(v.cpu().numpy(),S[n]) for n,v in m.state_dict().items() if n not in patch)
  endpoints.append({'row':'R'+str(i+1).zfill(5),'platform':r['platform'],'arm':arm,**rec[arm]});restore(m,S);states+=1
 assert list(saved['arm_names'])==ARMS80 and np.array_equal(saved['source_boxes'],z0['boxes']) and np.array_equal(saved['queries'],saved['scores'].argmax(1)) and np.array_equal(saved['selected_original_boxes'],z0['boxes'][saved['queries']])
 for a,v in pred.items():maxp=max(maxp,float(abs(v-saved['scores'][ARMS80.index(a)]).max()))
 assert maxp<1e-10
 if dev:
  for a in ['frozen_bi','frozen_direct3','uniform_sgd1']:assert np.max(abs(saved['scores'][ARMS80.index(a)]-od[a]))<1e-10
 else:
  prior=load(ROOT/'crossplatform/round78/extension_prediction2304_attempt001/samples_PRIVATE'/(sid+'.npz'));v=prior['scores'][list(prior['arm_names']).index('R76_bidirectional_frozen')];assert np.max(abs(v-pred['frozen_bi']))<1e-10
 if(i+1)%128==0:print('AUDIT80',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
assert fingerprint(base)==fp;A=H80/'independent_native_audit_attempt001';A.mkdir();result={'state':'PASS_R80_ALL2640_NATIVE_SCORES_AND_5280_ACTUAL_STATES','n':2640,'arm_names':ARMS80,'actual_states_replayed':states,'new_states':4944,'reused_R79_uniform_states':336,'max_probability_error':maxp,'max_native_mean_regenerated_parameter_error':maxparam,'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'prediction_seal_sha256':sha(O/'SEALED.json'),'source_training':False,'GT_read':False,'seconds':time.perf_counter()-start};dump(A/'AUDIT_RESULT.json',result);dump(H80/'ALL_ENDPOINT_SCALAR_ROWS.json',endpoints);print(result)
