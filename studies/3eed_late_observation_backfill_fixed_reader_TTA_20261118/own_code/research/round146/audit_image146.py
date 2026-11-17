"""Independent physical graphs, fixed reference, raw B interpolation and full native head."""
from common146 import *
import argparse,numpy as np,torch
from PIL import Image
from scipy.spatial import cKDTree
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R/'research/round141'));import audit141 as A;sys.path.insert(0,str(R/'research/round144'));from audit144 import embed;sys.argv=argv
MAPPING={'bf_raw_F_A':'Frozen_F','bf_raw_joint':'F_RGBB_only','bf_zero':'norm_joint_zero','bf_Aimage':'norm_Aimage','bf_Bimage':'norm_Bimage','bf_wrong_joint':'norm_joint_wrong','bf_wrong_B':'norm_B_wrong','bf_FA_FB_mean':'norm_FA_FB_mean','bf_FAB_FB_mean':'norm_FAB_FB_mean'}
def main():
 check();torch.set_num_threads(4);reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));root=B/('image_engineering' if args.engineering else 'image');seal=rd(root/'SEALED.json');meta={(e['source'],e['target'],e['id_PRIVATE']):e for e in rd(H/'POSITION_ROWS_PRIVATE.json')};mx=gw=mp=mi=ms=0.;n=nb=0;W={s:A.tensors(dict(np.load(R/'crossplatform/round135'/(s+'_SOURCE_MODEL_PRIVATE.npz')))) for s in ['waymo','drone','quad']};PW={s:A.tensors(dict(np.load(root/(s+'_POSITION_MODULE_PRIVATE.npz')))) for s in W}
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name;a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));old=np.load(R/'crossplatform/round145/method'/name/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['queries'],a['scores'].argmax(1)) and np.array_equal(a['source_boxes'],old['source_boxes']);rec=rd(D/'records_PRIVATE'/(sid+'.json'));assert rec['parameters_updated']==0 and rec['backfilled']==e['backfilled']
  for arm in op:assert np.array_equal(ps[arm],op[arm])
  if not e['backfilled']:
   for k,v in MAPPING.items():assert np.array_equal(ps[k],op[v])
  else:
   nb+=1;assert not e['original_has_B'] and e['has_B'];z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));rt=A.tensors(dict(np.load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'))));key=hashlib.sha256(z['xyz'].tobytes()+z['point_indices'].tobytes()).hexdigest();ga=dict(np.load(R/'crossplatform/round97/shared_physical_geometry_PRIVATE'/(sid+'_'+key[:20]+'.npz')));ca,cb=A.cam(e['A_sensor']),A.cam(e['pair_PRIVATE']['B']);Ma,Mb=A.raw_to_model_transform(ca),A.raw_to_model_transform(cb);sensor=e['pair_PRIVATE']['B'];rr=rd(R/e['registration_path']);assert sha(R/e['registration_path'])==e['registration_sha256'];rawB=A.raw(sensor);xB=A.tx(A.tx(z['xyz'][0],np.linalg.inv(Ma)),np.linalg.inv(np.asarray(rr['forward']['T_raw_B_to_A'])));modB=A.tx(xB,Mb);wh=Image.open(sensor['imagepath']).size;gb=A.sparse_graph(A.project_model_points(modB,cb,wh),A.project_model_points(A.tx(rawB,Mb),cb,wh),wh,near=cKDTree(rawB).query(xB)[0]);bad={k:v.copy() for k,v in gb.items()};ids=np.flatnonzero((gb['p2i']>=0).any(1));perm=np.arange(1024);rng=np.random.default_rng(int(hashlib.sha256((src+'|'+sid+'|R142imageonlywrong').encode()).hexdigest()[:16],16));order=rng.permutation(ids)
   if len(order)>1:perm[order]=np.roll(order,1)
   inv=np.argsort(perm);bad['p2i']=gb['p2i'][perm];bad['p2i_weight']=gb['p2i_weight'][perm];bad['i2p']=np.where(gb['i2p']>=0,inv[gb['i2p'].clip(0)],-1);orig={k:ga[k] for k in ['p2i','p2i_weight','i2p','i2p_weight','visible']};specs={'bf_raw_F_A':[orig],'bf_raw_joint':[orig,gb],'bf_zero':[orig,gb],'bf_Aimage':[orig],'bf_Bimage':[gb],'bf_wrong_joint':[orig,bad],'bf_wrong_B':[bad]};pp=np.load(D/'position_PRIVATE'/(sid+'.npz'));delta=meta[src,tgt,sid]['delta'];assert float(pp['delta'])==rec['delta']==delta;xx=z['xyz'].copy();xx[:,:,2]+=delta;assert np.array_equal(xx,pp['pos_input']) and np.array_equal(pp['physical_xyz'],z['xyz']);err=float(abs(embed(xx,PW[src])-pp['pos0']).max());mp=max(mp,err);assert err<1e-4;po=np.stack([z['position'+str(i)] for i in range(6)]);po[:,:,:,2]+=delta;assert np.array_equal(po,pp['positions']);banks=np.load(D/'banks_PRIVATE'/(sid+'.npz'));ff=np.load(D/'features_PRIVATE'/(sid+'.npz'))
   for arm,gs in specs.items():
    ex=A.merged(gs)
    for k,v in ex.items():
     ac=banks[arm+'_'+k]
     if k.endswith('weight'):err=float(abs(ac-v).max(initial=0));gw=max(gw,err);assert err<1e-5
     else:assert np.array_equal(ac,v)
    feat=A.tensors({k:ff[arm+'_'+k] for k in ['h','u','late','local','own','text','relation_text','mask']});prob=A.read135(feat,rt,W[src]);err=float(abs(prob-ps[arm]).max());mx=max(mx,err);assert err<5e-6
   assert np.array_equal(ps['bf_FA_FB_mean'],(ps['bf_Aimage']+ps['bf_Bimage'])/2) and np.array_equal(ps['bf_FAB_FB_mean'],(ps['bf_zero']+ps['bf_Bimage'])/2)
   if args.engineering:
    cached=np.load(R/e['backfill_features_path']);X=banks['B_native_xyz'];V=banks['B_native_vis'];rawfull=A.tx(A.tx(rawB,rr['forward']['T_raw_B_to_A']),Ma);dist,idx=cKDTree(X).query(z['xyz'][0],k=3);near=cKDTree(rawfull).query(z['xyz'][0])[0];zero=dist[:,0]<=1e-8;valid=(near<=.25)&((dist[:,-1]<=1.6)|zero);ww=1/np.maximum(dist,1e-8);ww/=ww.sum(1,keepdims=True);ww[zero]=[1.,0.,0.];aligned=(V[idx]*ww.astype(np.float32)[:,:,None]).sum(1);aligned=np.where(valid[:,None],aligned,z['vis0'][0]);assert np.array_equal(valid,cached['valid_B']);err=float(abs(aligned-cached['B_vis']).max());mi=max(mi,err);assert err<2e-5
    rng=np.random.default_rng(int(hashlib.sha256(('R106sample|'+e['pair_PRIVATE']['pair_PRIVATE']+'|real|B').encode()).hexdigest()[:16],16));ind=rng.choice(len(rawB),16384,replace=len(rawB)<16384);sample=np.ascontiguousarray(A.tx(A.tx(rawB[ind],rr['forward']['T_raw_B_to_A']),Ma),np.float32);assert hashlib.sha256(sample.tobytes()).hexdigest()==rec['B_actual_sample_sha256'];err=float(cKDTree(sample).query(X)[0].max());ms=max(ms,err);assert err<1e-4
  if not e['has_B']:
   for k in MAPPING:assert np.array_equal(ps[k],ps['Frozen_A'])
  n+=1
  if n%128==0:print('AUDIT_IMAGE146',n,nb,flush=True)
 assert n==len(reg);out={'status':'PASS','n':n,'new_backfilled':nb,'max_CPU_full_reader_error':mx,'max_graph_weight_error':gw,'max_CPU_position_MLP_error':mp,'max_engineering_CPU_B_interpolation_error':mi,'max_B_native_to_exactsample_distance':ms,'all_parent_predictions_and_remaining_noB_exact':True,'source_parameters_updated':0,'fullTransformer_PointNet_not_CPU_recomputed':True,'FINAL_access':False};dump(H/('IMAGE_ENGINEERING_AUDIT.json' if args.engineering else 'IMAGE_AUDIT.json'),out);print(out,flush=True)
if __name__=='__main__':main()
