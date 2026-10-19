"""No-GT independent physical interpolation, moments and fixed donor audit."""
from pathlib import Path
import sys,numpy as np,time,math
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_foreground_info128_v1 import D,P


def main():
 from methods.stvg_referdino_opd_v1.runtime import setup
 setup();A=D/'revision001';p=read(A/'LOCK.json');b=read(A/'GLOBAL_CONTROL_BARRIER.json');assert b['count']==128 and b['lock_sha256']==sha(A/'LOCK.json')and not(A/'GT_EXPOSURE.json').exists();start=time.perf_counter();err=0.;checks=0;ii,jj=np.triu_indices(65,1);records=sorted(p['records'],key=lambda r:r['alias'])
 for n,r in enumerate(records):
  a=r['alias'];donor=records[(n+1)%128]['alias'];assert p['donors'][a]==donor and a!=donor;z=load(D/'diagnostics'/f'{a}.pt');d=load(D/'diagnostics'/f'{donor}.pt');control=load(A/'controls'/f'{a}.pt');ts=z['ids'];ds=d['ids'];x=np.array([(t-ts[0])/(ts[-1]-ts[0])for t in ts]);xx=np.array([(t-ds[0])/(ds[-1]-ds[0])for t in ds]);v=z['fields']['Action'];other=d['fields']['Action'];curve=[]
  for t in x:
   right=int(np.searchsorted(xx,t));left=max(right-1,0);right=min(right,64)
   value=other[left]if right==left else other[left]+(t-xx[left])/(xx[right]-xx[left])*(other[right]-other[left]);curve.append(float(value))
  mu=sum(v)/65;dm=sum(curve)/65;sd=math.sqrt(sum((t-mu)**2 for t in v)/65);dd=math.sqrt(sum((t-dm)**2 for t in curve)/65);expected=np.full(65,mu)if sd<=1e-12 or dd<=1e-12 else np.array([mu+(t-dm)*sd/dd for t in curve]);e=float(abs(expected-control['field']).max());assert e<1e-10;err=max(err,e)
  for k in np.linspace(0,2079,19).astype(int):assert abs(sum(control['field'][ii[k]:jj[k]+1])-control['energy'][k])<1e-10;checks+=1
 put(P/'WRONG_VIDEO_INDEPENDENT_AUDIT.json',dict(status='pass',all128_controls_before_GT=True,manual_normalized_physical_interpolations=128*65,donor_mapping_exact=True,independent_mask_sums=checks,max_field_error=err,CPU_seconds=time.perf_counter()-start,GPU_seconds=0,code_sha256=sha(Path(__file__))));print('wrong-video numeric audit pass',flush=True)

if __name__=='__main__':main()
