"""No-GT independent scalar interpolation and direct joint-risk sums."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import sys,bisect,numpy as np
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_joint_risk128_v1 import D,P,PRE,FIN,PRIORS
from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr


def box_at(t,ids,boxes):
 j=bisect.bisect_right(ids,t);j=max(1,min(len(ids)-1,j));i=j-1
 if t<=ids[0]:return boxes[0]
 if t>=ids[-1]:return boxes[-1]
 a=(t-ids[i])/(ids[j]-ids[i]);return (1-a)*boxes[i]+a*boxes[j]


def corners(b):
 return np.array([[max(0,x-w/2),max(0,y-h/2),max(0,x+w/2),max(0,y+h/2)]for x,y,w,h in b])


def main():
 p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')and not(D/'GT_EXPOSURE.json').exists();curves=nodes=checks=flatchecks=targets=fulls=0;maxerr=maxcurve=0.
 for r in p['records']:
  z=load(D/'predictions'/f"{r['alias']}.pt");inp=load(PRE/'inputs'/f"{r['alias']}.pt");sp=load(FIN/'Live/predictions'/f"{r['alias']}.pt");ids=z['frame_ids'];ii,jj=np.triu_indices(len(ids),1);start=np.asarray(ids)[ii];end=np.asarray(ids)[jj]+1;ti=np.asarray(ids)[z['positions8']].tolist();tb=corners(z['teacher_boxes']);fc=z['frame_count']
  for space,boxes in z['trajectories'].items():
   assert np.array_equal(boxes,inp['E']if space=='E'else sp['outputs'][f'T20_Live_{space[4:]}']);sb=corners(boxes);quality=[]
   for f in range(fc):
    x=box_at(f,ids,sb);y=box_at(f,ti,tb);a=max(0,x[2]-x[0])*max(0,x[3]-x[1]);b=max(0,y[2]-y[0])*max(0,y[3]-y[1]);inter=max(0,min(x[2],y[2])-max(x[0],y[0]))*max(0,min(x[3],y[3])-max(x[1],y[1]));u=a+b-inter;v=inter/u if u>0 else 0.
    if f<ti[0]or f>ti[-1]or b<=0:v=1.
    if f<ids[0]or f>ids[-1]:v=0.
    quality.append(v)
   quality=np.array(quality);err=float(abs(quality-z['quality_curves'][space]['curve']).max());assert err<1e-12;maxcurve=max(maxcurve,err);nodes+=fc;curves+=1;assert np.array_equal(z['quality_curves'][space]['shift'],np.roll(z['quality_curves'][space]['curve'],fc//2))
   for name in PRIORS:
    q=z['priors'][name];prob=q[ii,jj];rm=z['risk'][space][name];out=z['outputs'][space];base=final_mbr(ids,q);assert out[name+'_Time']['interval']==out[name+'_Flat']['interval']==base['interval'];flatchecks+=1
    for label,riskcurve,key in [('Risk',quality,'real_utility'),('Shift',np.roll(quality,fc//2),'shift_utility'),('Flat',np.full(fc,quality.mean()),'flat_utility')]:
     pref=np.r_[0.,np.cumsum(riskcurve)];u=rm[key][ii,jj];iv=out[name+'_'+label]['interval'];winner=np.where((start==iv[0])&(end==iv[1]))[0];picks=np.unique(np.r_[np.linspace(0,len(ii)-1,17).astype(int),winner]);do_full=space=='Live41'and name=='A'and label=='Risk'and np.ptp(quality)>1e-12 and fulls<3
     if do_full:picks=np.arange(len(ii));fulls+=1
     for chunk in np.array_split(picks,max(1,(len(picks)+63)//64)):
      a=start[chunk,None];b=end[chunk,None];lo=np.maximum(a,start[None]);hi=np.minimum(b,end[None]);over=np.maximum(0,hi-lo);num=pref[np.maximum(lo,hi)]-pref[lo];union=b-a+end[None]-start[None]-over;actual=(np.where(hi>lo,num,0)/union)@prob;err=float(abs(actual-u[chunk]).max());assert err<1e-11;maxerr=max(maxerr,err);checks+=len(chunk)
     if np.ptp(riskcurve)<=1e-12:assert iv==base['interval']
    for key,uk in [('target','real_utility'),('flat_target','flat_utility')]:
     weight=q*np.maximum(np.where(np.isfinite(rm[uk]),rm[uk],0),0);ex=q if weight.sum()<=0 else weight/weight.sum();assert abs(ex-rm[key]).max()<1e-14 and abs(rm[key].sum()-1)<1e-12;targets+=1
    assert out[name+'_Target']['interval']==final_mbr(ids,rm['target'])['interval']and out[name+'_FlatTarget']['interval']==final_mbr(ids,rm['flat_target'])['interval']
 put(P/'RISK_INDEPENDENT_AUDIT.json',dict(status='pass',all128_before_GT=True,independent_scalar_quality_curves=curves,actual_physical_frame_qualities=nodes,max_curve_error=maxcurve,direct_utility_values=checks,max_direct_utility_error=maxerr,full_primary_action_grids=fulls,flat_original_time_exact_checks=flatchecks,target_distributions=targets,no_GT=True,no_model_calls=True,all_spatial_arrays_exact=True));print('independent risk pass',checks,fulls,maxerr,flush=True)

if __name__=='__main__':main()
