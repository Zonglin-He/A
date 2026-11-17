"""Measured storage profile, six symbolic cells with alpha1 aliases collapsed."""
from common149 import *
from storage149 import storage149
import numpy as np

def main():
 check();assert rd(H/'PREPARATION_AUDIT.json')['status']=='PASS';reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');sample=[]
 for e in reg:
  if not e['has_B']:continue
  name=e['source']+'_to_'+e['target'];sid=e['row']['id'];p=R/'crossplatform/round145/method'/name/'states_PRIVATE'/(sid+'.npz');f=R/'crossplatform/round145/method'/name/'features_PRIVATE'/(sid+'.npz');rec=R/'crossplatform/round145/method'/name/'records_PRIVATE'/(sid+'.json');pr=R/'crossplatform/round148/method'/name/'samples_PRIVATE'/(sid+'.npz');sample.append(dict(state=p.stat().st_size,feature=f.stat().st_size,record=rec.stat().st_size,prediction=pr.stat().st_size))
  if len(sample)>=12:break
 # Selected alphas are all1, so only plain A correct/wrong are actual newfits.
 assert all(v==1 for v in rd(H/'SOURCE_ALPHA_LOCK.json')['alpha'].values());counts=rd(H/'PREPARATION.json');actual_fits_max=2*counts['eligible'];mean={k:float(np.mean([x[k] for x in sample])) for k in sample[0]};est=1798*mean['prediction']*1.15+counts['eligible']*(mean['state']+mean['record'])*1.2+32*mean['feature']+40*1024**2;assert est<.75*1024**3,est;used,free=storage149();dump(H/'RESOURCE_PROFILE.json',dict(status='PASS_BEFORE_MODEL',profile_inputs=len(sample),measured_mean_bytes=mean,estimated_peak_bytes=est,newfits_max=actual_fits_max,existing_payload_bytes=used,free_bytes=free,scope_cap_GiB=.75,streamed_chunk=32,old_files_deleted=False));print('PROFILE149',est,actual_fits_max,flush=True)
if __name__=='__main__':main()
