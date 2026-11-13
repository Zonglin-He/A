"""Independent source state/selection and native full/masked validation replay."""
from core90 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
start=time.perf_counter();P=B90/'trained_source_attempt001';se=read(P/'SEALED.json');hist=read(P/'HISTORY.json');choice=read(P/'SELECTED_SOURCE_ONLY.json');epoch=max([1,2],key=lambda e:(np.mean([r['source_val_full_mIoU_percent'] for r in hist if r['epoch']==e]),-e));assert choice['shared_epoch']==epoch;base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);torch.manual_seed(0);m=Fusion90(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);load_R76_90(m);initial=state(m);rows=source_rows('VAL128');tr=source_rows('TRAIN512');assert not {r['scene'] for r in rows}&{r['scene'] for r in tr} and not {r['pointpath'] for r in rows}&{r['pointpath'] for r in tr};worst=0.;calls=0
for b in BRANCHES90:
 configure90(m,b);allowed={n for n,p in m.source_parameters()}
 for e in [1,2]:
  f=P/'states_PRIVATE'/(b+'_e'+str(e)+'.npz');assert sha(f)==se['files'][f.relative_to(P).as_posix()];ss=load(f);assert all(np.array_equal(v,initial[k]) for k,v in ss.items() if k not in allowed);va=load(P/'val_PRIVATE'/(b+'_e'+str(e)+'.npz'));yy=[]
  for r,p in zip(rows,va['scores']):y=load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'];yy.append([float(y[int(v.argmax())]) for v in p])
  yy=np.array(yy);assert np.array_equal(yy,va['IoU']);h=next(x for x in hist if x['branch']==b and x['epoch']==e);assert h['source_val_full_mIoU_percent']==float(yy[:,0].mean()*100)
with torch.no_grad():
 for i,r in enumerate(rows):
  z0,z,d,rt,route,pm,im,ws=data90(r,True);m.route=route
  for b in BRANCHES90:
   configure90(m,b);restore(m,load(ROOT/choice['models'][b]['state_path']));expected=load(P/'val_PRIVATE'/(b+'_e'+str(epoch)+'.npz'))['scores'][i]
   for j,(a,bb) in enumerate([(None,None)]+list(zip(pm,im))):
    p,_,_=m.main(base,z,d,'bidirectional',point_mask=a,image_mask=bb);calls+=1;worst=max(worst,float(abs(p.cpu().numpy()-expected[j]).max()));assert worst<1e-10
  if (i+1)%32==0:print('SOURCE_AUDIT90',i+1,flush=True)
assert fingerprint(base)==fp;O=H90/'independent_source_audit_attempt001';O.mkdir();out={'state':'PASS_R90_NATIVE_SOURCE_SELECTION_AND_JOINT_UPDATE_SCOPE','n':128,'source_train_n':512,'branches':4,'source_epochs_each':2,'shared_epoch':epoch,'native_replays':calls,'max_probability_error':worst,'source_train_val_scene_and_frame_overlap':0,'postfusion_features_not_cached_as_constants':True,'original_3EED_frozen':True,'target_features_or_labels_read':False,'seconds':time.perf_counter()-start,'training_seal_sha256':sha(P/'SEALED.json')};dump(O/'AUDIT_RESULT.json',out);print(out)
