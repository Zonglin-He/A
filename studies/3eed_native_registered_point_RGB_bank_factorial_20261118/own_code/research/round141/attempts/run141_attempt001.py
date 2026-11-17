"""Frozen native-space/image factorial; full source reader, all A256 unchanged."""
from common141 import *
from core141 import *
from common141 import R,H,B,rd,sha,dump,check,utc
import argparse,shutil
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');ap.add_argument('--attempt',type=int,default=1);args=ap.parse_args();assert args.attempt>=1 and (args.engineering or args.attempt==1)
NEW_ARMS=['F_RGBB_only','F_nativeB_only','F_nativeB_RGBB','F_nativeB_RGBB_wrong','F_extraA_duplicateA']
def main():
 check();assert shutil.disk_usage(R).free>=20*1024**3;reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));root=B/(('engineering' if args.attempt==1 else 'engineering_attempt'+str(args.attempt).zfill(3)) if args.engineering else 'formal');root.mkdir(exist_ok=False);n=0;forwards=0;backbones=0;mx=0.
 if not args.engineering:assert rd(H/'ENGINEERING_AUDIT.json')['status']=='PASS'
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base)
  if args.engineering:
   module=base.cross_encoder.layers[-1].cross_layer.cross_lv;module_name=next(n for n,q in base.named_modules() if q is module);np.savez_compressed(root/(src+'_BASE_LASTCROSS_PRIVATE.npz'),**{k:v.detach().cpu().numpy() for k,v in module.state_dict().items()},module_name=np.array(module_name),n_heads=np.array(module.num_heads))
  for e in [e for e in reg if e['source']==src]:
   tick=time.perf_counter();tgt,sid=e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','banks_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   old=np.load(R/'crossplatform/round136/method'/name/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(old['arm_names'],old['scores']));scores={k:ps[k].copy() for k in ['Frozen_A','Frozen_F','R135_A','R135_F','C_B_A','direct_C_B_A']};features={};banks={};proof={'has_B':e['has_B'],'GT_read':False,'source_parameters_updated':0};restore(m,S);freeze91(m);z0=load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));gA,d0=routegeom135(e,z0);rt0=load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'));A=torch.as_tensor(z0['vis0'][0],device='cuda')
   if not e['has_B']:
    for arm in NEW_ARMS:scores[arm]=scores['Frozen_A'].copy()
    if args.engineering:
     p,a,g=read_bank141(base,m,tensor_prefix(z0),[gA],rt0,e['row']);err=float(abs(p-scores['Frozen_A']).max());mx=max(mx,err);assert err<1e-7;forwards+=1
   else:
    f=load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));valid=torch.as_tensor(f['valid_B'],device='cuda');P=torch.where(valid[:,None],(A+torch.as_tensor(f['B_vis'],device='cuda'))/2,A);zF=prefix_bank141(base,z0,P,np.empty((0,3),np.float32),A[:0]);p,a,g=read_bank141(base,m,zF,[gA],rt0,e['row']);err=float(abs(p-scores['Frozen_F']).max());mx=max(mx,err);assert err<1e-7;forwards+=1
    sensor=e['pair_PRIVATE']['B'];assert sha(e['A_sensor']['pointpath'])==e['A_sensor']['point_sha256'] and sha(e['A_sensor']['imagepath'])==e['A_sensor']['image_sha256'];assert sha(sensor['pointpath'])==sensor['point_sha256'] and sha(sensor['imagepath'])==sensor['image_sha256'];assert sha(R/e['registration_path'])==e['registration_sha256'];rr=rd(R/e['registration_path']);v=native_observation141(base,e['A_sensor'],e['pair_PRIVATE'],z0,e['registration_path'],'real');backbones+=1;bridge=v['bridge'];berr=float(abs(bridge['vis'].cpu().numpy()-f['B_vis']).max());assert np.array_equal(bridge['valid'].cpu().numpy(),f['valid_B']) and berr<1e-5;ids=v['selected_indices'];k=len(ids);xb=v['xyz'][ids];vb=v['features'][torch.as_tensor(ids,device='cuda')];zb=prefix_bank141(base,z0,P,xb,vb);rB=route_bank141(rt0,z0['boxes'],xb);gAb=graph_A_bank141(e['A_sensor'],z0,gA,xb,v['full_A_model']);gB0,pr0=geo141.additional(e['A_sensor'],sensor,z0,rr);znb={**z0,'xyz':zb['xyz'].cpu().numpy()};gBb,prb=geo141.additional(e['A_sensor'],sensor,znb,rr);wrong=wrong_extra_graph141(gBb,src+'|'+sid+'|R141wrong_extraRGB');proof['wrong_extra_RGB_effective']=bool(not np.array_equal(wrong['p2i'],gBb['p2i']) or not np.array_equal(wrong['i2p'],gBb['i2p']))
    specs={'F_RGBB_only':(zF,[gA,gB0],rt0),'F_nativeB_only':(zb,[gAb],rB),'F_nativeB_RGBB':(zb,[gAb,gBb],rB),'F_nativeB_RGBB_wrong':(zb,[gAb,wrong],rB)}
    pad=0
    if k:
     va=native_observation141(base,e['A_sensor'],e['pair_PRIVATE'],z0,e['registration_path'],'same');backbones+=1;proof['A_native_novel_count']=va['novel_count'];proof['A_resample_sample_sha256']=va['sample_sha256'];ja=va['eligible_indices'][:k];xa=va['xyz'][ja];fa=va['features'][torch.as_tensor(ja,device='cuda')];pad=k-len(ja)
     if pad:
      pp=np.arange(pad)%1024;xa=np.concatenate([xa,z0['xyz'][0,pp]],0);fa=torch.cat([fa,A[torch.as_tensor(pp,device='cuda')]],0)
     za=prefix_bank141(base,z0,P,xa,fa);ra=route_bank141(rt0,z0['boxes'],xa);gAa=graph_A_bank141(e['A_sensor'],z0,gA,xa,v['full_A_model']);banks.update(A_extra_xyz=xa,A_extra_features=fa.cpu().numpy());specs['F_extraA_duplicateA']=(za,[gAa,gAa],ra)
    else:specs['F_extraA_duplicateA']=(zF,[gA,gA],rt0);banks.update(A_extra_xyz=np.empty((0,3),np.float32),A_extra_features=np.empty((0,288),np.float32))
    banks.update(B_extra_xyz=xb,B_extra_features=vb.cpu().numpy(),B_all_xyz=v['xyz'],B_all_features=v['features'].cpu().numpy(),B_selected_indices=ids)
    for arm in NEW_ARMS:
     zz,gg,rt=specs[arm];seen=[]
     def observe(_m,inputs,out):
      assert tuple(inputs[0].shape)==(1,3,256,384) and tuple(out.shape)==(1,64,32,48)
      seen.append(out.detach().cpu().numpy() if args.engineering else True)
     hook=m.rgb.register_forward_hook(observe);attn_hook=None
     if args.engineering:
      def save_attention(_m,_a,kw,out):
       for key in ['query','key','value']:features[arm+'_cross_'+key]=kw[key].detach().cpu().numpy()
      attn_hook=base.cross_encoder.layers[-1].cross_layer.cross_lv.register_forward_hook(save_attention,with_kwargs=True)
     try:prob,a,mg=read_bank141(base,m,zz,gg,rt,e['row'])
     finally:
      hook.remove()
      if attn_hook is not None:attn_hook.remove()
     assert len(seen)==len(gg);scores[arm]=prob;forwards+=1
     if args.engineering:
      for j,feat in enumerate(seen):features[arm+'_CNN'+str(j)]=feat
     for key,value in a.items():features[arm+'_'+key]=value
     for key in ['p2i','p2i_weight','i2p','i2p_weight','visible']:banks[arm+'_'+key]=mg[key]
    if args.engineering:
     zz,gg,rt=specs['F_extraA_duplicateA'];single,_,_=read_bank141(base,m,zz,[gg[0]],rt,e['row']);dup_error=float(abs(single-scores['F_extraA_duplicateA']).max());assert dup_error<2e-6;proof['duplicate_image_single_bank_equivalence_error']=dup_error;forwards+=1
    if not k:assert abs(scores['F_nativeB_only']-scores['Frozen_F']).max()<1e-7 and abs(scores['F_nativeB_RGBB']-scores['F_RGBB_only']).max()<1e-7
    proof.update(extra_native_B_tokens=k,A_resample_padding=pad,native_B_static_count=v['static_count'],native_B_novel_count=v['novel_count'],B_sample_sha256=v['sample_sha256'],interpolation_bridge_max_error=berr,old_F_reader_bridge_error=err,RGBB_original_A_visible=pr0['visible_seeds'],RGBB_expanded_visible=prb['visible_seeds'],geometry_priors_frozen_A=True,point_backbone_calls=1+int(k>0),new_full_reader_calls=6)
   assert all(np.array_equal(v,state(m)[key]) for key,v in S.items());arr=np.stack(list(scores.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(list(scores)),queries=arr.argmax(1),source_boxes=z0['boxes']);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**features);np.savez_compressed(D/'banks_PRIVATE'/(sid+'.npz'),**banks);proof['seconds']=time.perf_counter()-tick;dump(D/'records_PRIVATE'/(sid+'.json'),proof);n+=1
   if args.engineering or n%32==0:print('FROZEN141',args.engineering,src,n,proof.get('extra_native_B_tokens'),round(proof['seconds'],2),flush=True)
   if n%32==0:assert shutil.disk_usage(R).free>=12*1024**3
  assert fingerprint(base)==fp;del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'new_complete_reader_forwards':forwards,'new_point_backbone_calls':backbones,'max_full_A_F_zero_extra_bridge_error':mx,'parameters_updated':0,'GT_read':False,'FINAL_access':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('FROZEN141_COMPLETE',n,forwards,backbones,flush=True)
if __name__=='__main__':main()
