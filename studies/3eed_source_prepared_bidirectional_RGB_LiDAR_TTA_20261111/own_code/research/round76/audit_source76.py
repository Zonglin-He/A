"""Source-only selection arithmetic and full selected-state GPU replay."""
from common76 import *
from model76 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H/'experiment/DESIGN_LOCK.json');sr=BASE/'trained_source_attempt002';ss=read(sr/'SEALED.json');hist=read(sr/'HISTORY.json');selected=read(sr/'SELECTED_SOURCE_ONLY.json');rr=source_rows('VAL128');assert all(r['platform']=='waymo' for r in rr);source0=[];sourceoracle=[]
for r in rr:
 z=prefix(r);y=load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'];source0.append(float(y[int(z['scores'].argmax())]));sourceoracle.append(float(y.max()))
for mode in MODES:
 hh=[x for x in hist if x['mode']==mode];assert len(hh)==8 and [x['epoch'] for x in hh]==list(range(1,9));best=max(hh,key=lambda x:(x['val_mIoU_percent'],-x['epoch']));assert selected[mode]['epoch']==best['epoch']
 for row in hh:
  f=sr/'val_PRIVATE'/(mode+'_e'+str(row['epoch']).zfill(2)+'.npz');a=load(f);assert a['scores'].shape==(128,256);assert np.array_equal(a['scores'].argmax(1),a['selected']);v=[]
  for r,q in zip(rr,a['selected']):v.append(float(load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'][q]))
  assert np.array_equal(np.array(v),a['IoU']) and abs(np.mean(v)*100-row['val_mIoU_percent'])<1e-12
base=load_source(read(ROOT/p['source_plan']));fp=fingerprint(base);start=time.perf_counter();maxerr=0;calls=0
for mode in MODES:
 sel=selected[mode];f=ROOT/sel['state_path'];assert sha(f)==sel['state_sha256']==ss['files'][f.relative_to(sr).as_posix()];m=Fusion(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);restore(m,load(f));prior=load(sr/'val_PRIVATE'/(mode+'_e'+str(sel['epoch']).zfill(2)+'.npz'))
 with torch.no_grad():
  for i,r in enumerate(rr):
   pp,q,_=m.main(base,tensor_prefix(prefix(r)),tensor_data(load(casepath(r)),r),mode);calls+=1;maxerr=max(maxerr,float(abs(pp.cpu().numpy()-prior['scores'][i]).max()));assert int(pp.argmax())==int(prior['selected'][i])
 assert maxerr<1e-10;del m
assert fingerprint(base)==fp;out=H/'independent_source_audit_attempt001';assert not out.exists();out.mkdir();rec={'state':'PASS_R76_ALL_SOURCE_CURVES_SELECTION_AND_SELECTED_VAL128_NATIVE_GPU_REPLAY','source_train_n':512,'source_val_n':128,'epochs_per_mode':8,'selected_epochs':{k:v['epoch'] for k,v in selected.items()},'selected_source_mIoU_percent':{k:v['source_val_mIoU_percent'] for k,v in selected.items()},'source_raw_mIoU_percent':float(np.mean(source0)*100),'source_candidate_oracle_mIoU_percent':float(np.mean(sourceoracle)*100),'selected_GPU_suffix_calls':calls,'max_probability_error':maxerr,'original3EED_unchanged':True,'target_labels_or_inputs_read':False,'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',rec);print(rec)
