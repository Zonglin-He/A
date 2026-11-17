"""Independent CPU full reader and image-only projection/permutation checks."""
from common145 import *
import argparse,numpy as np,torch
from PIL import Image
from scipy.spatial import cKDTree
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R/'research/round141'));import audit141 as A;sys.argv=argv
sys.path.insert(0,str(R/'research/round144'));sys.argv=[argv[0]];from audit144 import embed;sys.argv=argv
ARMS=['norm_Aimage','norm_Bimage','norm_joint_wrong','norm_B_wrong']

def main():
 check();torch.set_num_threads(4);root=B/('image_engineering' if args.engineering else 'image');seal=rd(root/'SEALED.json');reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));W={s:A.tensors(dict(np.load(R/'crossplatform/round135'/(s+'_SOURCE_MODEL_PRIVATE.npz')))) for s in ['waymo','drone','quad']};mx=0.;gw=0.;n=0;nb=0;effective=0;mp=0.;meta={(e['source'],e['target'],e['id_PRIVATE']):e for e in rd(R/'research/round144/POSITION_ROWS_PRIVATE.json')};PW={s:A.tensors(dict(np.load(root/(s+'_POSITION_MODULE_PRIVATE.npz')))) for s in ['waymo','drone','quad']}
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name;a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));old=np.load(R/'crossplatform/round144/formal'/name/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['queries'],a['scores'].argmax(1)) and np.array_equal(a['source_boxes'],old['source_boxes']);rec=rd(D/'records_PRIVATE'/(sid+'.json'));assert rec['source_parameters_updated']==0 and rec['has_B']==e['has_B']
  for arm in op:assert np.array_equal(ps[arm],op[arm])
  assert np.array_equal(ps['norm_joint_zero'],op['aligned_reference']) and np.array_equal(ps['norm_FA_FB_mean'],(ps['norm_Aimage']+ps['norm_Bimage'])/2) and np.array_equal(ps['norm_FAB_FB_mean'],(ps['norm_joint_zero']+ps['norm_Bimage'])/2)
  if e['has_B']:
   nb+=1;effective+=rec['wrong_effective'];z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));rt=dict(np.load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz')));key=hashlib.sha256(z['xyz'].tobytes()+z['point_indices'].tobytes()).hexdigest();gA=dict(np.load(R/'crossplatform/round97/shared_physical_geometry_PRIVATE'/(sid+'_'+key[:20]+'.npz')));sensor=e['pair_PRIVATE']['B'];ca,cb=A.cam(e['A_sensor']),A.cam(sensor);Ma,Mb=A.raw_to_model_transform(ca),A.raw_to_model_transform(cb);rr=rd(R/e['registration_path']);rawB=A.raw(sensor);xB=A.tx(A.tx(z['xyz'][0],np.linalg.inv(Ma)),np.linalg.inv(np.asarray(rr['forward']['T_raw_B_to_A'])));modB=A.tx(xB,Mb);wh=Image.open(sensor['imagepath']).size;gB=A.sparse_graph(A.project_model_points(modB,cb,wh),A.project_model_points(A.tx(rawB,Mb),cb,wh),wh,near=cKDTree(rawB).query(xB)[0]);bad={k:v.copy() for k,v in gB.items()};visible=np.flatnonzero((gB['p2i']>=0).any(1));rng=np.random.default_rng(int(hashlib.sha256((src+'|'+sid+'|R142imageonlywrong').encode()).hexdigest()[:16],16));order=rng.permutation(visible);perm=np.arange(1024)
   if len(order)>1:perm[order]=np.roll(order,1)
   inv=np.argsort(perm);bad['p2i']=gB['p2i'][perm];bad['p2i_weight']=gB['p2i_weight'][perm];bad['i2p']=np.where(gB['i2p']>=0,inv[gB['i2p'].clip(0)],-1);orig={k:gA[k] for k in ['p2i','p2i_weight','i2p','i2p_weight','visible']};specs={'norm_Aimage':[orig],'norm_Bimage':[gB],'norm_joint_wrong':[orig,bad],'norm_B_wrong':[bad]};ff=np.load(D/'features_PRIVATE'/(sid+'.npz'));banks=np.load(D/'banks_PRIVATE'/(sid+'.npz'))
   pp=np.load(D/'position_PRIVATE'/(sid+'.npz'));delta=meta[src,tgt,sid]['delta'];assert rec['delta']==delta;xx=z['xyz'].copy();xx[:,:,2]+=delta;assert np.array_equal(pp['physical_xyz'],z['xyz']) and np.array_equal(pp['pos_input'],xx);err=float(abs(embed(xx,PW[src])-pp['pos0']).max());mp=max(mp,err);assert err<1e-4;pos=np.stack([z['position'+str(i)] for i in range(6)]);pos[:,:,:,2]+=delta;assert np.array_equal(pp['positions'],pos)
   for arm,gs in specs.items():
    ex=A.merged(gs)
    for k,v in ex.items():
     ac=banks[arm+'_'+k]
     if k.endswith('weight'):
      err=float(abs(v-ac).max(initial=0));gw=max(gw,err);assert err<1e-5
     else:assert np.array_equal(v,ac)
    f=A.tensors({k:ff[arm+'_'+k] for k in ['h','u','late','local','own','text','relation_text','mask']});p=A.read135(f,A.tensors(rt),W[src]);err=float(abs(p-ps[arm]).max());mx=max(mx,err);assert err<5e-6
  else:
   for arm in ARMS:assert np.array_equal(ps[arm],ps['Frozen_A'])
  n+=1
  if n%64==0:print('AUDIT_IMAGE145',n,nb,flush=True)
 assert n==(12 if args.engineering else 1798);out={'status':'PASS','n':n,'B_inputs':nb,'effective_wrong_B_inputs':effective,'max_CPU_full_reader_error':mx,'max_independent_graph_weight_error':gw,'max_CPU_position_MLP_error':mp,'native_A256_noB_and_reused_references_exact':True,'source_parameters_updated':0,'fullTransformer_not_independently_CPU_recomputed':True,'FINAL_access':False};dump(H/('IMAGE_ENGINEERING_AUDIT.json' if args.engineering else 'IMAGE_AUDIT.json'),out);print(out,flush=True)

if __name__=='__main__':main()
