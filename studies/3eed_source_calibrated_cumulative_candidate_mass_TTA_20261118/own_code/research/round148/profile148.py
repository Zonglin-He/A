"""Measure compressed feature/probe/state costs before models; no GT."""
from common148 import *
from storage148 import storage148
import numpy as np,io,collections

def main():
 check();counts=rd(H/'PREPARATION.json');samples=[];oldseal=rd(R/'crossplatform/round145/method/SEALED.json')['files'];by=collections.Counter();idx=np.linspace(0,1023,32).astype(np.int64)
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id']
  if by[src]>=3:continue
  name=src+'_to_'+tgt;p=R/'crossplatform/round145/method'/name/'features_PRIVATE'/(sid+'.npz');assert sha(p)==oldseal[str(p.relative_to(R/'crossplatform/round145/method'))];a=np.load(p)
  if 'norm_TTA_adapted_A' not in a.files:continue
  full=a['norm_TTA_adapted_A'];f={k:a[k] for k in a.files if k.startswith('norm_TTA_') and k!='norm_TTA_adapted_A'};f['norm_TTA_adapted_A_probe']=full[idx];f['norm_TTA_probe_indices']=idx;buf=io.BytesIO();np.savez_compressed(buf,**f);featbytes=len(buf.getvalue());sp=R/'crossplatform/round145/method'/name/'states_PRIVATE'/(sid+'.npz');assert sha(sp)==oldseal[str(sp.relative_to(R/'crossplatform/round145/method'))];b=np.load(sp);d={k:b[k] for k in b.files if k.startswith('norm_TTA_')};buf=io.BytesIO();np.savez_compressed(buf,**d);statebytes=len(buf.getvalue());samples.append({'source':src,'features_probe_bytes':featbytes,'state_bytes':statebytes,'full_adapted_array_uncompressed_bytes':full.nbytes});by[src]+=1
 assert dict(by)=={'waymo':3,'drone':3,'quad':3};nfit=counts['new_correct_fits']+counts['new_wrong_fits'];estimate=max(x['features_probe_bytes']+x['state_bytes'] for x in samples)*nfit+250*1024**2;cap=rd(H/'experiment/DESIGN_LOCK.json')['resource']['payload_cap_GiB']*1024**3;assert estimate<cap,(estimate,cap);used,free=storage148();dump(H/'RESOURCE_PROFILE.json',{'status':'PASS_BEFORE_MODEL','samples':samples,'max_fresh_fits':nfit,'conservative_estimate_bytes':estimate,'cap_bytes':cap,'observed_free_bytes':free,'probe_indices':idx.tolist(),'full_engineering_adapted_A_retained':True,'formal_full_adapted_A_not_stored_or_fully_compared':True,'formal_full_keep_reconstructed_and_native_final_heads_states_objectives_audited':True,'no_old_artifact_deleted':True});print(estimate,nfit,flush=True)
if __name__=='__main__':main()
