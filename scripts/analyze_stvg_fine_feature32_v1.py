"""Saved-result strata and validity readback, never online frame selection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_fine_feature32_v1 import D,P,PRE,verify
from scripts.run_stvg_joint_event_critic_v1 import read,load,put

def main():
 import torch,numpy as np
 import torch.nn.functional as F
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();z=read(P/'RESULTS_PUBLIC.json');meta=read(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json')['rows'];meta={ds:{x['query_ordinal']:x for x in v}for ds,v in meta.items()};record={x['alias']:x for x in p['records']};strata={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in z['rows']if r['dataset']==ds];mm={r['alias']:meta[ds][record[r['alias']]['query_ordinal']]for r in rr};out={}
  for key,field in [('target_area','target_area'),('relative_speed','speed_sizes_sec'),('query_form',None),('reference_available',None)]:
   values={r['alias']:(r['active']if key=='reference_available'else mm[r['alias']]['query_form']if key=='query_form'else mm[r['alias']]['bins'][field])for r in rr};out[key]={}
   for v in sorted(set(values.values()),key=str):
    sub=[r for r in rr if values[r['alias']]==v];out[key][str(v)]=dict(count=len(sub),parents=len(sub),
     paired={a+'_minus_'+b:{ep:paired([None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in sub])for ep in ['v','unobserved_actual_s']}for a,b in [('Fine_layer3','E_MBR'),('Fine_layer3','Coarse'),('Fine_layer3','Fine_downsampled')]})
  strata[ds]=out
 put(P/'STRATA_PUBLIC.json',dict(status='complete',shared_historical_bins=True,GT_offline_descriptive_only=True,no_online_rule=True,small_subgroups_not_confirmation=True,groups=strata))
 n=0
 for r in p['records']:
  a=r['alias'];f=load(D/'features'/(a+'.pt'));x=load(PRE/'features'/(a+'.pt'));assert f['maps'].shape[-2]>x['maps'].shape[-2]and f['maps'].shape[-1]>x['maps'].shape[-1];valid=F.interpolate(f['valid'][:,None].float(),size=x['maps'].shape[-2:],mode='nearest')[:,0].bool();assert torch.equal(valid,x['valid']);n+=1
 f=P/'ROOT_VALIDITY_ALIGNMENT.json'
 if not f.exists():put(f,dict(status='pass',count=n,actual_fine_height_and_width_larger=True,same_layer_downsampled_valid_equals_original_coarse=True,no_prediction_or_config_changed=True))
 print('saved strata and validity pass',n,flush=True)

if __name__=='__main__':main()
