"""GT-free original-source input compatibility controls, locked before task scoring."""
from common138 import *
import numpy as np
def soft(x):e=np.exp(x-x.max());return e/e.sum()
def main():
 check();assert rd(B/'direct/SEALED.json')['n']==1798;out=B/'original_source_controls';out.mkdir(exist_ok=False);n=0
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  name=e['source']+'_to_'+e['target'];sid=e['row']['id'];D=B/'direct'/name;a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));p=dict(zip(a['arm_names'],a['scores']));f=np.load(D/'features_PRIVATE'/(sid+'.npz'));z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));ids=z['query_points_sample_inds'].astype(int);l=f['original_seed_logits'][ids].astype(float);hasB=rd(D/'records_PRIVATE'/(sid+'.json'))['has_B'];scores={'original_seed_only':soft(l),'A_original_seed_product':soft(np.log(np.maximum(p['Frozen_A'],1e-300))+l),'F_original_Aseed_product':soft(np.log(np.maximum(p['Frozen_F'],1e-300))+l)}
  if not hasB:scores={k:p['Frozen_A'].copy() for k in scores}
  d=out/name;d.mkdir(exist_ok=True);np.savez_compressed(d/(sid+'.npz'),arm_names=np.array(list(scores)),scores=np.stack(list(scores.values())),queries=np.array([x.argmax() for x in scores.values()]));n+=1
 dump(out/'SEALED.json',{'n':n,'GT_used':False,'not_retroactive_primary':True,'files':{str(p.relative_to(out)):sha(p) for p in out.rglob('*.npz')}});print('ORIGINAL_CONTROL138_SEALED',n,flush=True)
if __name__=='__main__':main()
