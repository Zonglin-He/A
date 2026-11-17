"""CPU source full-reader and actual B-image correspondence audit."""
from common142 import *
import argparse,numpy as np,torch
from PIL import Image
from scipy.spatial import cKDTree
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R/'research/round141'));import audit141 as A;sys.argv=argv
def main():
 check();assert rd(H/'IMAGE_DECISION.json')['passed']
 for p,h in rd(H/'experiment/SOURCE_CODE_LOCK.json')['files'].items():assert sha(R/p)==h,p
 torch.set_num_threads(4);root=B/('source_engineering' if args.engineering else 'source');seal=rd(root/'SEALED.json');reg={(e['entry']['source'],e['entry']['row']['id']):e for e in rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json')};pairs={e['pair_PRIVATE']:e for e in rd(R/'research/round132/SOURCE_PAIRS_PRIVATE.json')};W={s:A.tensors(dict(np.load(R/'crossplatform/round135'/(s+'_SOURCE_MODEL_PRIVATE.npz')))) for s in ['waymo','drone','quad']};n=0;nb=0;mx=0.;gw=0.
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for src in ['waymo','drone','quad']:
  for p in sorted((root/src).glob('*.npz')):
   if p.name.endswith('_graph.npz'):continue
   sid=p.stem;e=reg[src,sid];a=np.load(p);record=rd(root/src/(sid+'.json'));rt=A.tensors(dict(np.load(R/e['entry']['files']['route']['path'])));old=np.load(R/'crossplatform/round140/source'/src/(sid+'.npz'));z=np.load(R/e['entry']['files']['prefix']['path']);assert np.array_equal(a['valid_B'],old['valid_B']) and np.array_equal(a['assignment'],old['assignment']) and np.array_equal(a['S'],old['S']);expected=np.where(a['valid_B'][:,None],(z['vis0'][0]+old['B_vis'])/2,z['vis0'][0]);assert np.array_equal(expected,a['F_vis']);assert record['source_parameters_updated']==0 and record['source_state_fingerprint_unchanged'];gA=dict(np.load(R/e['entry']['files']['geometry']['path']));banks=np.load(root/src/(sid+'_graph.npz'))
   if record['has_B']:
    nb+=1;ca,cb=A.cam(e['A']),A.cam(pairs[e['pair_PRIVATE']]['B']);Ma,Mb=A.raw_to_model_transform(ca),A.raw_to_model_transform(cb);sensor=pairs[e['pair_PRIVATE']]['B'];rr=rd(R/record['registration_path']);assert sha(R/record['registration_path'])==record['registration_sha256'];rawB=A.raw(sensor);xB=A.tx(A.tx(z['xyz'][0],np.linalg.inv(Ma)),np.linalg.inv(np.asarray(rr['forward']['T_raw_B_to_A'])));modB=A.tx(xB,Mb);wh=Image.open(sensor['imagepath']).size;ex=A.sparse_graph(A.project_model_points(modB,cb,wh),A.project_model_points(A.tx(rawB,Mb),cb,wh),wh,near=cKDTree(rawB).query(xB)[0])
   else:ex={k:gA[k] for k in ['p2i','p2i_weight','i2p','i2p_weight','visible']};assert np.array_equal(a['pA'],a['pB']) and abs(a['pA']-old['pA']).max()<1e-7
   for key,val in ex.items():
    if key.endswith('weight'):
     err=float(abs(val-banks[key]).max(initial=0));gw=max(gw,err);assert err<1e-5
    else:assert np.array_equal(val,banks[key])
   for tag in ['A','B']:
    f=A.tensors({k:a[tag+'_'+k] for k in ['h','u','late','local','own','text','relation_text','mask']});pr=A.read135(f,rt,W[src]);err=float(abs(pr-a['p'+tag]).max());mx=max(mx,err);assert err<5e-6 and np.isclose(pr.sum(),1,atol=1e-12)
   n+=1
 assert n==(3 if args.engineering else 313);out={'status':'PASS','n':n,'B_inputs':nb,'source_views':2*n,'max_CPU_full_reader_probability_error':mx,'max_independent_B_graph_weight_error':gw,'source_Fpoints_Aimage_Bimage_reader':True,'same_qualification_as_R132':True,'GT_read':False,'fullTransformer_not_CPU_regenerated':True,'FINAL_access':False};dump(H/('SOURCE_ENGINEERING_AUDIT.json' if args.engineering else 'SOURCE_AUDIT.json'),out);print(out,flush=True)
if __name__=='__main__':main()
