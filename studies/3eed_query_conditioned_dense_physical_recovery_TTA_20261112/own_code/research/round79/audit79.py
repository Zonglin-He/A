"""All-case independent sensitivity/weight/actual-state replay, no task labels."""
from core79 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H79/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H79/'experiment/AUDIT_CODE_LOCK.json'));P=B79/'prediction464_attempt001';seal0=read(P/'SEALED.json')
for f,h in seal0['files'].items():assert sha(P/f)==h
base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);start=time.perf_counter();maxp=maxparam=maxsal=maxradius=0.;scalars=[];sourcechecks=0;states=0
for i,r in enumerate(source_rows('VAL128')+target_rows()):
 sid=r['id'];z0=prefix(r);z=tensor_prefix(z0);d=tensor_data(load(casepath(r)),r);pm,im=maskpair79(r,d);saved=load(P/'samples_PRIVATE'/(sid+'.npz'));rec=read(P/'records_PRIVATE'/(sid+'.json'));wp=load(P/'weights_PRIVATE'/(sid+'.npz'));refp,refq,rp,ri,pr=sensitivity(m,base,z,d,z0['boxes']);maxsal=max(maxsal,float(abs(wp['point_saliency']-rp).max()),float(abs(wp['image_saliency']-ri).max()));assert maxsal<1e-5 and pr['pairs_PRIVATE']==[tuple(v) for v in rec['saliency']['pairs_PRIVATE']];weights,fb=weight_sets(r,d,pm,im,wp['point_saliency'],wp['image_saliency']);assert fb==rec['fallback_weight_sets'];pred={'frozen_bi':refp.cpu().numpy()};views=[refp]
 for kind,ws in weights.items():
  for j,w in enumerate(ws):
   for key,v in w.items():assert np.max(abs(v.cpu().numpy()-wp[kind+'_'+str(j)+'_'+key]))<1e-6;assert abs(float(v.sum())-1)<1e-6 or not bool(v.any())
 for a,b in zip(weights['query'],weights['shuffle']):
  for key in a:assert torch.equal(a[key].sort().values,b[key].sort().values)
 with torch.no_grad():
  for a,b in zip(pm,im):v,_,_=m.main(base,z,d,'bidirectional',point_mask=a,image_mask=b);views.append(v)
 pred['frozen_direct3']=torch.stack(views).mean(0).cpu().numpy();raw={}
 for kind,wrong in [('uniform',False),('query',False),('shuffle',False),('wrong',True)]:
  g,delta,_=gradient79(m,base,z,d,refp,refq,pm,im,weights['query' if wrong else kind],wrong);raw[kind]=delta
 radius=raw['uniform'].norm();proposals={'uniform_sgd1':raw['uniform'],'query_raw_sgd1':raw['query']}
 for name,key in [('query_radius_sgd1','query'),('shuffle_radius_sgd1','shuffle'),('wrong_binding_radius_sgd1','wrong')]:proposals[name]=raw[key]*(radius/raw[key].norm()) if float(raw[key].norm())>0 else raw[key]
 for arm,delta in proposals.items():
  patch=load(P/'states_PRIVATE'/(sid+'_'+arm+'.npz'));tta=dict(m.tta_parameters());assert set(patch)==set(tta);off=0
  with torch.no_grad():
   for n,v in tta.items():
    size=v.numel();wanted=torch.tensor(S[n],device='cuda')+delta[off:off+size].reshape_as(v);maxparam=max(maxparam,float(abs(wanted.cpu().numpy()-patch[n]).max()));v.copy_(torch.tensor(patch[n],device='cuda'));off+=size
   assert maxparam<1e-7;pp,q,_=m.main(base,z,d,'bidirectional');pred[arm]=pp.cpu().numpy();actual=torch.cat([(v-torch.tensor(S[n],device='cuda')).flatten() for n,v in m.tta_parameters()]);norm=float(actual.norm());assert abs(norm-rec[arm]['parameter_step_L2'])<1e-9
   if 'radius' in arm:maxradius=max(maxradius,abs(norm-float(radius)));assert maxradius<1e-6
  assert all(np.array_equal(v.detach().cpu().numpy(),S[n]) for n,v in m.state_dict().items() if n not in patch);scalars.append({'row':('S' if r['platform']=='waymo' else 'R')+str(i+1 if r['platform']=='waymo' else i-127).zfill(5),'platform':r['platform'],'arm':arm,**rec[arm]});restore(m,S);states+=1
 for a,v in pred.items():maxp=max(maxp,float(abs(v-saved['scores'][list(saved['arm_names']).index(a)]).max()))
 assert maxp<1e-10 and np.array_equal(saved['source_boxes'],z0['boxes']) and np.array_equal(saved['queries'],saved['scores'].argmax(1)) and np.array_equal(saved['selected_original_boxes'],z0['boxes'][saved['queries']])
 if r['platform']=='waymo':
  old=load(ROOT/'crossplatform/round77/source128_attempt001/samples_PRIVATE'/(sid+'.npz'));od=dict(zip(old['arm_names'],old['scores']));refs={'source':'source','R76_Adam1':'aux1','R76_Adam3':'aux3'};sourcechecks+=1
 else:
  old=load(ROOT/'crossplatform/round76/target336_attempt001/samples_PRIVATE'/(sid+'.npz'));od=dict(zip(old['arm_names'],old['scores']));refs={'source':'source','R76_Adam1':'bidirectional_TTA1','R76_Adam3':'bidirectional_TTA3','frozen_ABC':'frozen_ABC','R59':'R59'}
 for a,b in refs.items():
  got=saved['scores'][list(saved['arm_names']).index(a)]
  if a=='source':assert np.array_equal(got,z0['scores']/z0['scores'].sum()) and int(got.argmax())==int(od[b].argmax())
  else:assert np.max(abs(got-od[b]))<1e-10
 if(i+1)%32==0:print('AUDIT79',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
assert fingerprint(base)==fp;O=H79/'independent_native_audit_attempt002';O.mkdir();result={'state':'PASS_R79_ALL464_SENSITIVITY_WEIGHTS_SHUFFLES_AND_NATIVE_PARAMETER_STATES','n':464,'source_n':128,'target_n':336,'actual_updated_states':states,'max_probability_error':maxp,'max_regenerated_parameter_error':maxparam,'max_saliency_regeneration_error':maxsal,'max_matched_radius_error':maxradius,'original_A256_unchanged':True,'source_training':False,'GT_read':False,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',result);dump(H79/'ALL_ENDPOINT_SCALAR_ROWS.json',scalars);print(result)
