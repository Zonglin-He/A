"""Image-only matched binding control; frozen model, original A1024/A256."""
from common142 import *
sys.path.insert(0,str(R/'research/round141'))
from core141 import *
from common142 import R,H,B,rd,sha,dump,check,utc
import argparse

ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
ARMS=['F_RGBB_only_wrong','F_Bimage_only_teacher','F_Bimage_only_wrong_teacher']

def main():
 check();reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));root=B/('image_engineering' if args.engineering else 'image');root.mkdir(exist_ok=False);n=0;forward=0;mx=0.;maxdup=0.
 if not args.engineering:assert rd(H/'IMAGE_ENGINEERING_AUDIT.json')['status']=='PASS'
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base)
  for e in [e for e in reg if e['source']==src]:
   tick=time.perf_counter();sid,tgt=e['row']['id'],e['target'];name=src+'_to_'+tgt;D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','banks_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   old=np.load(R/'crossplatform/round141/formal'/name/'samples_PRIVATE'/(sid+'.npz'));ref=dict(zip(old['arm_names'],old['scores']));scores={k:ref[k].copy() for k in ['Frozen_A','Frozen_F','R135_A','R135_F','C_B_A','direct_C_B_A','F_RGBB_only']};restore(m,S);freeze91(m);features={};banks={};rec={'source_parameters_updated':0,'GT_read':False,'FINAL_access':False,'has_B':e['has_B']}
   if e['has_B']:
    z0=load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));gA,_=routegeom135(e,z0);rt=load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'));f=load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));A=torch.as_tensor(z0['vis0'][0],device='cuda');valid=torch.as_tensor(f['valid_B'],device='cuda');P=torch.where(valid[:,None],(A+torch.as_tensor(f['B_vis'],device='cuda'))/2,A);z=dict(tensor_prefix(z0),vis0=P[None]);rr=rd(R/e['registration_path']);assert sha(R/e['registration_path'])==e['registration_sha256']
    for sensor in [e['A_sensor'],e['pair_PRIVATE']['B']]:
     assert sensor['eligible_for_extra_media'] and not sensor['protected_image_content_alias']
     for kind in ['point','image']:assert sha(sensor[kind+'path'])==sensor[kind+'_sha256']
    gB,proof=geo141.additional(e['A_sensor'],e['pair_PRIVATE']['B'],z0,rr);bad=wrong_extra_graph141(gB,src+'|'+sid+'|R142imageonlywrong');banks.update(B_graph_visible=gB['visible']);rec['wrong_effective']=bool(not np.array_equal(gB['p2i'],bad['p2i']) or not np.array_equal(gB['i2p'],bad['i2p']));rec['B_visible_points']=int(gB['visible'].sum())
    for tag,graphs in [('Frozen_F',[gA]),('F_RGBB_only',[gA,gB])]:
     pp,aa,gg=read_bank141(base,m,z,graphs,rt,e['row']);err=float(abs(pp-ref[tag]).max());mx=max(mx,err);assert err<1e-7,(name,tag,err);forward+=1
    specs={'F_RGBB_only_wrong':[gA,bad],'F_Bimage_only_teacher':[gB],'F_Bimage_only_wrong_teacher':[bad]}
    for arm,graphs in specs.items():
     pp,aa,gg=read_bank141(base,m,z,graphs,rt,e['row']);scores[arm]=pp;forward+=1
     for key,val in aa.items():features[arm+'_'+key]=val
     for key in ['p2i','p2i_weight','i2p','i2p_weight','visible']:banks[arm+'_'+key]=gg[key]
    if args.engineering:
     dup,_,_=read_bank141(base,m,z,[gA,gA],rt,e['row']);err=float(abs(dup-ref['Frozen_F']).max());maxdup=max(maxdup,err);assert err<2e-6;forward+=1;rec['duplicate_A_equivalence_error']=err
   else:
    for arm in ARMS:scores[arm]=ref['Frozen_A'].copy()
   assert all(np.array_equal(v,state(m)[key]) for key,v in S.items());arr=np.stack(list(scores.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(list(scores)),queries=arr.argmax(1),source_boxes=old['source_boxes']);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**features);np.savez_compressed(D/'banks_PRIVATE'/(sid+'.npz'),**banks);rec['seconds']=time.perf_counter()-tick;dump(D/'records_PRIVATE'/(sid+'.json'),rec);n+=1
   if args.engineering or n%64==0:print('IMAGE142',args.engineering,src,n,flush=True)
  assert fingerprint(base)==fp;del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'complete_reader_forwards':forward,'max_reference_bridge_error':mx,'max_engineering_duplicate_A_image_error':maxdup,'parameters_updated':0,'GT_read':False,'FINAL_access':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('IMAGE142_COMPLETE',n,forward,flush=True)

if __name__=='__main__':main()
