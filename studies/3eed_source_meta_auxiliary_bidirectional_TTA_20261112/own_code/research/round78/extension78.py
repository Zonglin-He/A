"""Source-selected joint/meta one-step SGD; target labels never imported."""
from extension_common78 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ARMS78=['source','frozen_ABC','R59','R76_bidirectional_frozen','joint0','joint1','joint_direct3','joint_wrong1','meta0','meta1','meta_direct3','meta_wrong1']
def main():
 plan=read(H78/'experiment/DESIGN_LOCK.json');pin(plan);pin(read(H78/'experiment/EXTENSION_CODE_LOCK.json'));tr=B78/'trained_source_attempt001';ts=read(tr/'SEALED.json');selected=read(tr/'SELECTED_SOURCE_ONLY.json');assert sha(tr/'SELECTED_SOURCE_ONLY.json')==ts['files']['SELECTED_SOURCE_ONLY.json'];base=load_source(read(ROOT/plan['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);out=B78/'extension_prediction2304_attempt001';assert not out.exists();out.mkdir()
 for s in ['samples_PRIVATE','states_PRIVATE','records_PRIVATE']:(out/s).mkdir()
 net={};S={}
 for b in ['joint','meta']:
  sel=selected['models'][b];sf=ROOT/sel['state_path'];assert sha(sf)==sel['state_sha256'];S[b]=load(sf);m=Fusion78(base.contrastive_align_projection_image).cuda().eval();restore(m,S[b]);net[b]=m
 ref=Fusion78(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);restore(ref,load(ROOT/plan['init_state']));start=time.perf_counter();calls=0;auxcalls=0;status=[]
 for i,r in enumerate(ext_rows()):
  sid=r['id'];z0,z,d=ext_item(r);pm,im=maskpair(r,d);scores={};recs={};st={}
  for b,m in net.items():
   enable_tta(m)
   with torch.no_grad():p0,q0,_=main78(m,base,z,d);views=[p0];calls+=1
   for a,c in zip(pm,im):
    with torch.no_grad():p,_,_=main78(m,base,z,d,point_mask=a,image_mask=c);views.append(p);calls+=1
   scores[b+'0']=p0.cpu().numpy();scores[b+'_direct3']=torch.stack(views).mean(0).cpu().numpy()
   for wrong in [False,True]:
    fast,L,gn=inner78(m,z,d,pm,im,wrong=wrong);auxcalls+=2
    with torch.no_grad():
     pp,qq,_=main78(m,base,z,d,fast);calls+=1;arm=b+('_wrong1' if wrong else '1');scores[arm]=pp.cpu().numpy();delta=float(torch.sqrt(sum((fast[n]-p).square().sum() for n,p in m.tta_parameters())));recs[arm]={'auxiliary_before':float(L.detach()),'gradient_norm':float(gn.detach()),'parameter_step_L2':delta,'query_feature_change_L2':float((qq-q0).norm()),'predicted_query_changed':int(pp.argmax())!=int(p0.argmax()),'target_GT_used':False,'step_rule':'functional_SGD_same_as_source_training_alpha0.5_clip1'}
    np.savez_compressed(out/'states_PRIVATE'/(sid+'_'+arm+'.npz'),**{n:fast[n].detach().cpu().numpy() for n,_ in m.tta_parameters()});del fast
  with torch.no_grad():prior_bi,_,_=main78(ref,base,z,d);calls+=1
  prior=load(ROOT/'crossplatform/round69/predicted2304_attempt001/samples_PRIVATE'/(sid+'.npz'));pd=dict(zip(prior['arm_names'],prior['scores']));assert np.array_equal(z0['boxes'],prior['source_boxes']);scores.update(source=pd['source'],frozen_ABC=pd['frozen_ABC'],R59=pd['R59'],R76_bidirectional_frozen=prior_bi.cpu().numpy());ss=np.stack([scores[k] for k in ARMS78]);qq=ss.argmax(1);np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS78),scores=ss,queries=qq,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][qq]);dump(out/'records_PRIVATE'/(sid+'.json'),recs)
  for a in ARMS78:
   active=a in recs and recs[a]['parameter_step_L2']>0;st[a]={'unknown':bool(not d['visible'].any()) if a.startswith(('joint','meta')) else False,'fallback':bool(a in recs and not active),'no_parameter_update':not active}
  status.append({'id':sid,'arm_statuses':st})
  if(i+1)%128==0:print('TARGET_EXT78',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
 assert fingerprint(base)==fp
 for b,m in net.items():assert all(np.array_equal(v.detach().cpu().numpy(),S[b][n]) for n,v in m.state_dict().items())
 dump(out/'READOUT_STATUS_PRIVATE.json',status);dump(out/'COUNTS.json',{'n':2304,'arms':12,'target_GT_read':False,'source_shared_epoch':selected['shared_epoch'],'source_parameters':338343,'target_parameters':102498,'grounding_suffix_calls':calls,'auxiliary_fusion_calls':auxcalls,'SGD_proposals':9216,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'original_A256_unchanged':True,'FINAL_access':False});seal(out,'R78_FIXED_SOURCE_SELECTED_TARGET_PREDICTIONS_SEALED_BEFORE_TARGET_LABEL_JOIN',n=2304,arm_names=ARMS78);print(read(out/'COUNTS.json'))
if __name__=='__main__':main()
