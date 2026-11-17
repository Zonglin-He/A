"""No-GT full-cohort initialization/readout bridge and immutable-preparation check."""
from common135 import *
import numpy as np
def main():
 basecheck();root=B135/'formal';seal=rd(root/'PREPARED_SEALED.json');assert seal['n']==1798
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for lock in ['MODEL_CODE_LOCK','FORMAL_LOCK']:
  for p,h in rd(H135/'experiment'/(lock+'.json'))['files'].items():assert sha(R135/p)==h,p
 n=0;noB=0;active=0;pairactive=0;pairs=0;maximum=0.;bridge=0.
 for D in sorted(root.glob('*_to_*')):
  for p in sorted((D/'zero_PRIVATE').glob('*.npz')):
   a=np.load(p);ps=dict(zip(a['arm_names'],a['scores']));old=np.load(R135/'crossplatform/round97'/D.name/'samples_PRIVATE'/p.name);op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['source_boxes'],old['source_boxes']);err=float(abs(ps['intact_live_A']-op['prepared_frozen']).max());maximum=max(maximum,err);assert err<1e-7 and ps['intact_live_A'].argmax()==op['prepared_frozen'].argmax()
   legacy=np.load(R135/'crossplatform/round134/formal'/D.name/'samples_PRIVATE'/p.name);lp=dict(zip(legacy['arm_names'],legacy['scores']));err=float(abs(ps['reset_fixed_F']-lp['zero_F']).max());bridge=max(bridge,err);assert err<1e-7
   rec=rd(D/'records_PRIVATE'/(p.stem+'.json'));active+=rec['active'];pairactive+=rec['pair_count']>0;pairs+=rec['pair_count']
   if not rec['has_B']:
    noB+=1
    for k in ['intact_live_F','AB_mean','direct_A','direct_F']:assert np.array_equal(ps[k],ps['intact_live_A'])
   n+=1
 assert n==1798;out={'status':'PASS','n':n,'no_B_complete_frozen_fallback':noB,'active':active,'pair_active':pairactive,'pair_count':pairs,'max_full_A_error':maximum,'max_legacy_bridge_error':bridge,'preparation_files_immutable':True,'GT_read':False,'FINAL_access':False};dump(H135/'PREPARATION_AUDIT.json',out);print(out,flush=True)
if __name__=='__main__':main()
