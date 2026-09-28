"""Independently recover the equivalent latent delta under unchanged W and gate."""
import sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,check_pins,verify_seal
from scripts.desta3d_v3_privileged_ptd_qualification import B1

def norm(a):
 a=np.asarray(a,dtype=np.float64).reshape(-1);return float(np.sqrt(np.dot(a,a)))

def run(name):
 d=OUT/name;verify_seal(d);check_pins(read(d/'LOCK.json')['pins']);assert read(d/'CONFIG.json')['mode']=='span'
 root=read(d/'ROOT_DECISION.json');st=torch.load(B1,weights_only=False,map_location='cpu')['adapter'];cases=[]
 for j,c in enumerate(root['cases']):
  b=c['branch'];ep=d/'episodes'/f'{j:02}';q=torch.load(ep/'BASIS.pt',weights_only=False,map_location='cpu');f=torch.load(ep/'FINAL.pt',weights_only=False,map_location='cpu')
  coeff=f['parameter'].double().numpy();rr=q['R'].numpy();gate=float(q['gate']);w=st['out_proj_'+b+'.weight'].double().numpy()
  dz=np.linalg.solve(rr,(coeff*q['scale']).reshape(-1,rr.shape[0]).T).T/gate
  reconstructed=gate*(dz@w.T);actual=f['token_delta'].double().numpy().reshape(reconstructed.shape)
  error=float(np.max(np.abs(reconstructed-actual)));assert np.allclose(reconstructed,actual,rtol=2e-5,atol=2e-5)
  lp=OUT/'layers001/episodes'/('00' if b=='event' else '02')/'LAYERS.pt'
  layers=torch.load(lp,weights_only=False,map_location='cpu');base=layers['original'];z=base['z'].double().numpy()
  assert z.shape==coeff.shape
  proposal=base['proposal'].double().numpy();updated=base['updated'].double().numpy();originalF=updated-gate*proposal
  ar='temporal' if b=='event' else 'spatial';mask_delta=layers[ar]['updated'].double().numpy()-updated
  entry={'branch':b,'unchanged_gate':gate,'equivalent_latent_delta_L2':norm(dz),'original_branch_latent_L2':norm(z),
   'equivalent_latent_relative_L2':norm(dz)/norm(z),'token_delta_L2':norm(actual),'token_delta_relative_to_base_F':norm(actual)/norm(originalF),
   'token_delta_to_old_correct_mask_L2_ratio':norm(actual)/norm(mask_delta),'FP64_gW_delta_reconstruction_max_error':error,
   'actual_tokendelta_shape':list(f['token_delta'].shape),'basis_orthogonality_max_error':float(np.max(np.abs(q['basis'].double().numpy().T@q['basis'].double().numpy()-np.eye(rr.shape[0]))))}
  cases.append(entry)
 result={'status':'passed','cases':cases,'interpretation':'Unconstrained latent reachability with absorbed gate/QR scale, not small reader updates or efficient learnability. Finite Adam is parameterization-dependent.',
    'new_GPU':False,'pins':{str(p):sha(p) for p in [Path(__file__),B1,d/'ROOT_DECISION.json']}}
 write(d/'ROOT_EQUIVALENT_LATENT_MAGNITUDE.json',result);print(json.dumps(result['cases']))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(p.parse_args().name)
