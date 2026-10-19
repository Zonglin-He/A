"""All-source raw-field identity and independent scalar diagnostics before GT."""
from pathlib import Path
import sys,time,numpy as np
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_foreground_info128_v1 import D,P,FG,ST


def main():
 p=read(D/'PLAN_LOCK.json');b=read(D/'GLOBAL_DIAGNOSTIC_BARRIER.json');assert b['count']==128 and b['plan_sha256']==sha(D/'PLAN_LOCK.json')and not(D/'GT_EXPOSURE.json').exists();t=time.perf_counter();checks=0;ii,jj=np.triu_indices(65,1);maxerr=0.
 for r in p['records']:
  a=r['alias'];z=load(D/'diagnostics'/f'{a}.pt');f=load(FG/'predictions'/f'{a}.pt');s=load(ST/'predictions'/f'{a}.pt')
  for k in range(65):
   o=k%2;j=k//2
   assert z['fields']['Action'][k]==float(f['actions']['Native'][o].reshape(-1)[j])
   assert z['fields']['Motion'][k]==float(s['native_stage_traces'][o]['initial_motion_logits'].reshape(-1)[j])
   assert z['fields']['Appearance'][k]==float(s['native_stage_traces'][o]['initial_appearance_logits'].reshape(-1)[j])
  for key,field in z['fields'].items():
   for n in np.linspace(0,2079,19).astype(int):
    lo,hi=ii[n],jj[n];value=sum(float(field[k])for k in range(lo,hi+1));err=abs(value-z['mask_energies'][key][n]);assert err<1e-10;maxerr=max(maxerr,err);checks+=1
  assert np.max(abs(z['mask_energies']['ActionEOS']-z['mask_energies']['Action']-np.log(.3)*(jj-ii+1)))<1e-10
  for key in ['Action','Motion','Appearance']:
   assert np.array_equal(z['fields'][key+'Shift'],np.r_[z['fields'][key][-32:],z['fields'][key][:-32]])
   assert max(abs(z['fields'][key+'Flat']-sum(z['fields'][key])/65))<1e-12
 put(P/'INFORMATION_INDEPENDENT_AUDIT.json',dict(status='pass',no_GT=True,raw_source_fields_exact=3*65*128,independent_scalar_mask_sums=checks,max_error=maxerr,all_EOS_length_identities=True,all_Flat_Shift_controls_exact=True,CPU_seconds=time.perf_counter()-t,GPU_seconds=0,no_new_method_outputs=True));print('independent information audit pass',flush=True)

if __name__=='__main__':main()
