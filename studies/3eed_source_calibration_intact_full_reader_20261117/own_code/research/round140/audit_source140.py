"""Independent CPU full reader, same geometry and source-state verification."""
from common140 import *
import argparse,numpy as np,torch
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R/'research/round135'));from audit135 import read135,tensors;sys.argv=argv
def main():
 check();torch.set_num_threads(4);root=B/('source_engineering' if args.engineering else 'source');seal=rd(root/'SEALED.json');reg={(e['entry']['source'],e['entry']['row']['id']):e for e in rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json')};W={s:tensors(dict(np.load(R/'crossplatform/round135'/(s+'_SOURCE_MODEL_PRIVATE.npz')))) for s in ['waymo','drone','quad']};n=0;mx=0.
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for src in ['waymo','drone','quad']:
  for p in sorted((root/src).glob('*.npz')):
   sid=p.stem;e=reg[src,sid];a=np.load(p);rt=tensors(dict(np.load(R/e['entry']['files']['route']['path'])));old=np.load(R/'crossplatform/round132/source_calibration_PRIVATE'/src/(sid+'.npz'));assert np.array_equal(a['S'],old['S']) and np.array_equal(a['assignment'],old['assignment']) and np.array_equal(a['valid_B'],old['valid_B']);assert rd(root/src/(sid+'.json'))['source_state_fingerprint_unchanged']
   for v in ['A','B']:
    feat=tensors({k:a[v+'_'+k] for k in ['h','u','late','local','own','text','relation_text','mask']});prob=read135(feat,rt,W[src]);err=float(abs(prob-a['p'+v]).max());mx=max(mx,err);assert err<5e-6;assert np.isfinite(a['p'+v]).all() and np.isclose(a['p'+v].sum(),1,atol=1e-12)
   if not a['valid_B'].any():assert np.array_equal(a['pA'],a['pB'])
   n+=1
 assert n==(3 if args.engineering else 313);out={'status':'PASS','n':n,'source_views':2*n,'max_CPU_full_reader_probability_error':mx,'same_source_qualification_and_assignment_as_R132':True,'only_source_own_inputs':True,'GT_read':False,'FINAL_access':False,'full_transformer_not_CPU_regenerated':True};dump(H/('SOURCE_ENGINEERING_AUDIT.json' if args.engineering else 'SOURCE_AUDIT.json'),out);print(out,flush=True)
if __name__=='__main__':main()
