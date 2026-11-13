"""Frozen-body, source-only selection, real native feature/readout and negative-pair audit."""
from core91 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
P=B91/'trained_source_attempt001';se=read(P/'SEALED.json');hist=read(P/'HISTORY.json');selection=read(P/'SELECTED_SOURCE_ONLY.json');choice=max(range(1,5),key=lambda e:(np.mean([r['source_val_mIoU_percent'] for r in hist if r['epoch']==e]),-e));assert selection['shared_epoch']==choice;base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);torch.manual_seed(0);m=Fusion91(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);initial={b:loadbase91(m,b) for b in BASES91};tr=source_rows('TRAIN512');val=source_rows('VAL128');assert not {r['scene'] for r in tr}&{r['scene'] for r in val};assert not {r['pointpath'] for r in tr}&{r['pointpath'] for r in val};assert all(r['platform']=='waymo' for r in tr+val);states={n:load(ROOT/v['state_path']) for n,v in selection['models'].items()};worst=fe=0.;topchanges=0;calls=0;start=time.perf_counter();pair_rows=[];baseline_rows=[]
for row in hist:
 ss=load(ROOT/row['state_path']);assert sha(ROOT/row['state_path'])==se['files'][row['state_path'].split('trained_source_attempt001/')[1]];assert all(np.array_equal(v,initial[row['base']][k]) for k,v in ss.items() if not k.startswith('interface.'));v=load(P/'val_PRIVATE'/(row['name']+'_e'+str(row['epoch'])+'.npz'));iy=[]
 for r,pp in zip(val,v['scores']):y=load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'];iy.append([float(y[int(p.argmax())]) for p in pp])
 iy=np.array(iy);assert np.array_equal(iy,v['IoU']) and row['source_val_mIoU_percent']==float(iy[:,0].mean()*100) and row['source_val_masked_mIoU_percent']==float(iy[:,1:].mean(1).mean()*100)
with torch.no_grad():
 for nr,r in enumerate(val):
  z0,z,d,rt,route,pm,im,ws=data91(r,True);m.route=route;restore(m,initial['R76']);freeze91(m);features=[native_feature91(m,base,z,d,a,b) for a,b in [(None,None)]+list(zip(pm,im))];calls+=3;old=load(B91/'source_features640_attempt001/samples_PRIVATE'/(r['id']+'.npz'))
  for j,f in enumerate(features):
   for k,t in f.items():
    if t.dtype==torch.bool:assert np.array_equal(t.cpu().numpy(),old[str(j)+'_'+k])
    elif t.numel():fe=max(fe,float(abs(t.cpu().numpy()-old[str(j)+'_'+k]).max()))
  for b in BASES91:
   restore(m,initial[b]);f=features[0];ns=native_scores(F.normalize(m.match(f['h'][None])[0]+f['late'],dim=-1),f['u'],f['mask'].bool()).double();y=load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'];baseline_rows.append({'row':'SOURCEVAL'+str(nr+1).zfill(4),'base':b,'IoU':float(y[int(ns.argmax())])})
   for h in HEADS91:
    n=b+'_'+h;restore(m,states[n]);freeze91(m);saved=load(P/'val_PRIVATE'/(n+'_e'+str(choice)+'.npz'))['scores'][nr];scores=[]
    for j,f in enumerate(features):
     p,e,s=cached91(m,f,route,h);worst=max(worst,float(abs(p.cpu().numpy()-saved[j]).max()));topchanges+=int(int(p.argmax())!=int(saved[j].argmax()));scores.append(s)
    if h=='full_disc':
     m.route=route;m.use_interface=True;m.variant=variant91(h);p,_,_=m.main(base,z,d,'bidirectional');calls+=1;worst=max(worst,float(abs(p.cpu().numpy()-saved[0]).max()));topchanges+=int(int(p.argmax())!=int(saved[0].argmax()))
    pairs=load(P/'pairs_PRIVATE'/(b+'_'+r['id']+'.npz'))['pairs_PRIVATE'];ng=load(B91/'source_competitors640_attempt001/samples_PRIVATE'/(r['id']+'.npz'));ass=ng['competitor_assignment_PRIVATE'];assert len(pairs)<=4
    if len(pairs):
     assert len(set(ass[pairs[:,1]].tolist()))==len(pairs) and (ass[pairs[:,1]]>=0).all() and (pairs[:,0]==int(ng['positive_query_PRIVATE'])).all();margin=(scores[0][torch.as_tensor(pairs[:,0],device='cuda')]-scores[0][torch.as_tensor(pairs[:,1],device='cuda')]).cpu().numpy();pair_rows.append({'row':'SOURCEVAL'+str(nr+1).zfill(4),'base':b,'head':h,'pair_count':len(pairs),'positive_margin_n':int((margin>0).sum()),'above_training_margin_n':int((margin>=.2).sum()),'mean_margin':float(margin.mean())})
  assert fe==0 and worst<1e-6 and topchanges==0
  if (nr+1)%32==0:print('SOURCE_AUDIT91',nr+1,flush=True)
assert fingerprint(base)==fp;O=H91/'independent_source_audit_attempt001';O.mkdir();out={'state':'PASS_R91_SOURCE_ONLY_SELECTION_FROZEN_STRONG_BASE_NATIVE_HEADS_AND_DISTINCT_NEGATIVE_POOLS','n':128,'source_train_n':512,'source_val_n':128,'source_branches':10,'epochs_checked':4,'shared_epoch':choice,'native_suffix_calls':calls,'cached_feature_max_error':fe,'max_probability_error':worst,'source_FP32_probability_tolerance':1e-6,'top1_changes':topchanges,'source_scene_and_frame_overlap':0,'original_R76_fusion_and_matching_bases_unchanged':True,'target_inputs_or_labels_read':False,'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',out);dump(H91/'SOURCE_PAIR_DISCRIMINATION_ROWS.json',pair_rows);dump(H91/'SOURCE_BASELINE_SCALAR_ROWS.json',baseline_rows);print(out)
