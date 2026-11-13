"""Independent native source validation, source-only selection and frozen-body audit."""
from core89 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
start=time.perf_counter();P=B89/'trained_source_attempt001';seal0=read(P/'SEALED.json');choice=read(P/'SELECTED_SOURCE_ONLY.json');history=read(P/'HISTORY.json');expected=max(range(1,5),key=lambda e:(np.mean([r['source_val_mIoU_percent'] for r in history if r['epoch']==e]),-e));assert choice['shared_epoch']==expected;source=source_rows('VAL128');tr=source_rows('TRAIN512');assert not {r['scene'] for r in source}&{r['scene'] for r in tr};assert all(r['platform']=='waymo' for r in source+tr);base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);torch.manual_seed(0);m=Fusion89(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=load_R76(m);init=state(m);frozen=0.;error=0.;rows=[];calls=0
for branch in ['continue','query','candidate']:
 for e in range(1,5):
  file=P/'states_PRIVATE'/(branch+'_e'+str(e)+'.npz');assert sha(file)==seal0['files'][file.relative_to(P).as_posix()];ss=load(file)
  for k in ss:
   if not k.startswith('match.' if branch=='continue' else 'interface.'):assert np.array_equal(ss[k],init[k]),('frozen_state_changed',branch,k)
  vv=load(P/'val_PRIVATE'/(branch+'_e'+str(e)+'.npz'));assert vv['scores'].shape[:2]==(128,3);scores=vv['scores'];ys=[]
  for r,p in zip(source,scores):y=load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'];ys.append([float(y[int(v.argmax())]) for v in p])
  yy=np.array(ys);assert np.array_equal(vv['IoU'],yy[:,0]);h=next(x for x in history if x['branch']==branch and x['epoch']==e);assert h['source_val_mIoU_percent']==float(yy[:,0].mean()*100) and h['source_val_masked_mIoU_percent']==float(yy[:,1:].mean(1).mean()*100)
with torch.no_grad():
 for nr,r in enumerate(source):
  cached=load(B89/'source_features640_attempt001/samples_PRIVATE'/(r['id']+'.npz'));rt=load(B89/'source_features640_attempt001/routes_PRIVATE'/(r['id']+'.npz'));z0=prefix(r);z=tensor_prefix(z0);d=tensor_data(load(casepath(r)),r);m.route=tensor_route89(rt);region={'point':rt['region_point'],'image':rt['region_image']};ab=[mask89(r,d,region,k) for k in [89001,89002]];modes=[(None,None)]+ab
  for branch in ['continue','query','candidate']:
   restore(m,load(ROOT/choice['models'][branch]['state_path']));m.use_interface=branch!='continue';m.variant='query_only' if branch=='query' else 'full';saved=load(P/'val_PRIVATE'/(branch+'_e'+str(expected)+'.npz'))['scores'][nr]
   for v,(pm,im) in enumerate(modes):
    p,_,extra=m.main(base,z,d,'bidirectional',point_mask=pm,image_mask=im);calls+=1;error=max(error,float(abs(p.cpu().numpy()-saved[v]).max()));assert error<1e-10
    if v==0 and branch=='candidate':
     y=load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'];a={k:torch.tensor(cached['0_'+k],device='cuda') for k in ['h','u','late','local','own','text','relation_text','baseline','mask']};row={'row':'SOURCEVAL'+str(nr+1).zfill(4),'prepared_full_IoU':float(y[int(p.argmax())]),'frozen_R76_IoU':float(y[int(cached['0_baseline'].argmax())])}
     for variant in ['no_ref','shared_mix','query_only']:row[variant+'_IoU']=float(y[int(cached_score89(m,a,m.route,'candidate',variant).argmax())])
     rows.append(row)
  if (nr+1)%32==0:print('AUDIT_SOURCE89',nr+1,flush=True)
assert fingerprint(base)==fp;O=H89/'independent_source_audit_attempt001';O.mkdir();out={'state':'PASS_R89_SOURCE_ONLY_SELECTION_NATIVE_FULL_MASKED_AND_FROZEN_BODY','source_train_n':512,'source_val_n':128,'source_branches':3,'shared_epoch':expected,'native_replays':calls,'max_probability_error':error,'original_R76_other_parameters_unchanged':True,'source_scene_and_frame_overlap':0,'target_inputs_or_labels_read':False,'training_seal_sha256':sha(P/'SEALED.json'),'source_VAL_frozen_R76_mIoU_percent':float(np.mean([r['frozen_R76_IoU'] for r in rows])*100),'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',out);dump(H89/'SOURCE_COMPONENT_ROWS.json',rows);print(out)
