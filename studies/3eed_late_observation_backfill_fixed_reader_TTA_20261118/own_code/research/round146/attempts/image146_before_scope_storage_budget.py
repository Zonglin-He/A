"""New physical B only for original missing-B; intact R145 reader everywhere."""
from common146 import *
sys.path.insert(0,str(R/'research/round141'));from core141 import *
sys.path.insert(0,str(R/'research/round145'));from position145 import normalize145,savepos145
from common146 import R,H,B,rd,sha,dump,check,utc
import argparse,shutil
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
MAPPING={'bf_raw_F_A':'Frozen_F','bf_raw_joint':'F_RGBB_only','bf_zero':'norm_joint_zero','bf_Aimage':'norm_Aimage','bf_Bimage':'norm_Bimage','bf_wrong_joint':'norm_joint_wrong','bf_wrong_B':'norm_B_wrong','bf_FA_FB_mean':'norm_FA_FB_mean','bf_FAB_FB_mean':'norm_FAB_FB_mean'}

def main():
 check();assert shutil.disk_usage(R).free>=18*1024**3;reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));met={(e['source'],e['target'],e['id_PRIVATE']):e for e in rd(H/'POSITION_ROWS_PRIVATE.json')};root=B/('image_engineering' if args.engineering else 'image');root.mkdir(exist_ok=False);n=nf=nb=0;maxbridge=maxB=0.
 if not args.engineering:assert rd(H/'IMAGE_ENGINEERING_AUDIT.json')['status']=='PASS'
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base);np.savez_compressed(root/(src+'_POSITION_MODULE_PRIVATE.npz'),**{k:v.detach().cpu().numpy() for k,v in base.pos_embed.state_dict().items()})
  for e in [e for e in reg if e['source']==src]:
   tick=time.perf_counter();tgt,sid=e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','banks_PRIVATE','position_PRIVATE','records_PRIVATE']:(D/sub).mkdir(exist_ok=True,parents=True)
   old=np.load(R/'crossplatform/round145/method'/name/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(old['arm_names'],old['scores']));ff={};bb={};pp={};rec={'backfilled':e['backfilled'],'original_has_B':e['original_has_B'],'has_B':e['has_B'],'parameters_updated':0,'GT_read':False};restore(m,S);freeze91(m)
   if not e['backfilled']:
    for k,v in MAPPING.items():ps[k]=ps[v].copy()
   else:
    nb+=1;assert not e['original_has_B'];z0=load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));a=load(R/e['backfill_features_path']);assert sha(R/e['backfill_features_path'])==e['backfill_features_sha256'];A=torch.as_tensor(z0['vis0'][0],device='cuda');valid=torch.as_tensor(a['valid_B'],device='cuda');P=torch.where(valid[:,None],(A+torch.as_tensor(a['B_vis'],device='cuda'))/2,A);gA,_=routegeom135(e,z0);gB,_=geo141.additional(e['A_sensor'],e['pair_PRIVATE']['B'],z0,rd(R/e['registration_path']));bad=wrong_extra_graph141(gB,src+'|'+sid+'|R142imageonlywrong');rt=load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'));zraw=dict(tensor_prefix(z0),vis0=P[None]);delta=met[src,tgt,sid]['delta'];z=normalize145(base,zraw,delta);savepos145(pp,z,z0,delta);rec.update(delta=delta,selected_rank=e['selected_B_rank'],wrong_B_binding_effective=bool(not np.array_equal(gB['p2i'],bad['p2i']) or not np.array_equal(gB['i2p'],bad['i2p'])))
    if args.engineering:
     v=native_observation141(base,e['A_sensor'],e['pair_PRIVATE'],z0,e['registration_path'],'real');err=float(abs(v['bridge']['vis'].cpu().numpy()-a['B_vis']).max());maxB=max(maxB,err);assert err<1e-5 and np.array_equal(v['bridge']['valid'].cpu().numpy(),a['valid_B']);bb.update(B_native_xyz=v['xyz'],B_native_vis=v['features'].cpu().numpy());rec.update(B_fresh_backbone_bridge_error=err,B_actual_sample_sha256=v['sample_sha256'],fresh_backbone_calls=1)
     prob,_,_=read_bank141(base,m,tensor_prefix(z0),[gA],rt,e['row']);err=float(abs(prob-ps['Frozen_A']).max());maxbridge=max(maxbridge,err);assert err<1e-7;rec['original_complete_A_bridge_error']=err;nf+=1
    specs={'bf_raw_F_A':(zraw,[gA]),'bf_raw_joint':(zraw,[gA,gB]),'bf_zero':(z,[gA,gB]),'bf_Aimage':(z,[gA]),'bf_Bimage':(z,[gB]),'bf_wrong_joint':(z,[gA,bad]),'bf_wrong_B':(z,[bad])}
    for arm,(zz,gs) in specs.items():
     prob,feat,mg=read_bank141(base,m,zz,gs,rt,e['row']);ps[arm]=prob;nf+=1
     for k,val in feat.items():ff[arm+'_'+k]=val
     for k in ['p2i','p2i_weight','i2p','i2p_weight','visible']:bb[arm+'_'+k]=mg[k]
    ps['bf_FA_FB_mean']=(ps['bf_Aimage']+ps['bf_Bimage'])/2;ps['bf_FAB_FB_mean']=(ps['bf_zero']+ps['bf_Bimage'])/2
   assert all(np.array_equal(v,state(m)[k]) for k,v in S.items());arr=np.stack(list(ps.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(list(ps)),queries=arr.argmax(1),source_boxes=old['source_boxes']);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'banks_PRIVATE'/(sid+'.npz'),**bb);np.savez_compressed(D/'position_PRIVATE'/(sid+'.npz'),**pp);rec['seconds']=time.perf_counter()-tick;dump(D/'records_PRIVATE'/(sid+'.json'),rec);n+=1
   if args.engineering or n%128==0:print('IMAGE146',args.engineering,src,n,nb,nf,flush=True)
  assert fingerprint(base)==fp;del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'backfilled':nb,'new_full_forwards':nf,'fresh_backbone_calls':nb if args.engineering else 0,'max_completeA_bridge':maxbridge,'max_engineering_rawB_bridge_error':maxB,'parameters_updated':0,'GT_read':False,'FINAL_access':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('IMAGE146_COMPLETE',n,nb,nf,flush=True)
if __name__=='__main__':main()
