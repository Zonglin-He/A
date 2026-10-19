"""Independent no-GT actual RGB-window, geometry and Source probability audit."""
from pathlib import Path
import sys,math,hashlib,numpy as np
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_tube_view128_v1 import D,P,PRE,FIN,DEV,HR


def main():
 from methods.stvg_referdino_opd_v1.runtime import setup,decode
 from scripts.stvg_temporal_marginal_decode_util256_v1 import independent_cdf
 setup();p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')and not(D/'GT_EXPOSURE.json').exists();assert read(D/'NATIVE_CONTROL.json')['status']=='pass';raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];checks=cdfs=actualRGB=changes=unique=0;maxerr=0
 for r in p['records']:
  z=load(D/'predictions'/f"{r['alias']}.pt");sp=load(FIN/'Live/predictions'/f"{r['alias']}.pt");hr=load(HR/'predictions'/f"{r['alias']}.pt");prior=load(DEV/'priors'/f"{r['alias']}.pt");inp=load(PRE/'inputs'/f"{r['alias']}.pt");H,W=z['original_RGB_shape'][1:3];ids=z['frame_ids'];boxes=np.asarray(sp['outputs']['T20_Live_41']);unique+=len(set(tuple(w)for w in r['windows'])-{(0,0,W,H)});changes+=r['windows'][0]!=r['windows'][1];assert z['original_RGB_sha256']==hr['observation']['RGB_sha256'];frames=None
  if r['alias']=='RC001':frames,physical=decode(raw[r['query_ordinal']]['input']);assert physical==ids and hashlib.sha256(frames.tobytes()).hexdigest()==z['original_RGB_sha256'];actualRGB+=1
  for state in [0,1]:
   s,e=r['state_intervals'][state];left=max([k for k,t in enumerate(ids)if t<=s]or[0]);right=min([k for k,t in enumerate(ids)if t>=e-1]or[len(ids)-1]);points=[]
   for k in range(left,right+1):
    x,y,w,h=boxes[k];x0=max(0,min(1,x-w/2));x1=max(0,min(1,x+w/2));y0=max(0,min(1,y-h/2));y1=max(0,min(1,y+h/2))
    if x1>x0 and y1>y0:points.append((x0,y0,x1,y1))
   window=[0,0,W,H]
   if points:
    lo=[min(v[k]for v in points)for k in [0,1]];hi=[max(v[k]for v in points)for k in [2,3]];bounds=[]
    for low,high,dim in zip(lo,hi,[W,H]):
     size=min(1,2*(high-low));center=(low+high)/2;low=min(max(center-size/2,0),1-size);bounds.append((math.floor(round(low*dim,10)),math.ceil(round((low+size)*dim,10))))
    window=[bounds[0][0],bounds[1][0],bounds[0][1],bounds[1][1]]
   obs=z['observations'][state];assert window==r['windows'][state]==obs['window'];assert r['window_meta'][state]['nodes']==list(range(left,right+1));checks+=1
   if frames is not None:
    x0,y0,x1,y1=window;pixels=frames[:,np.arange(y0,y1)[:,None],np.arange(x0,x1)[None,:],:];assert list(pixels.shape)==obs['shape']and hashlib.sha256(pixels.tobytes()).hexdigest()==obs['raw_RGB_sha256'];actualRGB+=1
   assert obs['shape']==[65,window[3]-window[1],window[2]-window[0],3]
   if obs['normalized_shape']is not None:assert obs['normalized_shape'][-2:]==[224,min(int(224*obs['shape'][2]/obs['shape'][1]),int(224*1.4))]
   grids=[]
   for rec,ll in zip(obs['records'],obs['logits']):
    ll=ll.numpy()[0].astype(float);ii,jj=np.triu_indices(len(ll),1);scores=ll[ii,0]+ll[jj,1];v=np.exp(scores-scores.max());v/=v.sum();grids.append(dict(ids=rec['frame_ids'],ij=np.stack([ii,jj]),p=v))
   q=independent_cdf(grids,ids);err=float(abs(q-obs['q']).max());assert err<1e-12;maxerr=max(maxerr,err);cdfs+=1
  qs={k:v['q']for k,v in z['outputs'].items()};assert np.array_equal(qs['Native'],prior['q'])and np.array_equal(qs['Crop0'],z['observations'][0]['q'])and np.array_equal(qs['Crop1'],z['observations'][1]['q'])
  assert np.array_equal(qs['Crop0Mix'],.5*qs['Native']+.5*qs['Crop0'])and np.array_equal(qs['Crop1Mix'],.5*qs['Native']+.5*qs['Crop1'])and np.array_equal(qs['SameS1Fixed'],.5*qs['Student1']+.5*qs['Crop0'])and np.array_equal(qs['SameS1Refresh'],.5*qs['Student1']+.5*qs['Crop1'])
  assert all(np.array_equal(v,inp['E']if k=='E'else sp['outputs'][f'T20_Live_{k[4:]}'])for k,v in z['trajectories'].items())
 assert unique==p['required_new_crop_Source_calls']and changes==p['changed_window_parents'];put(P/'VIEW_INDEPENDENT_AUDIT.json',dict(status='pass',all128_before_GT=True,independent_windows=checks,actual_first_video_and_crop_RGB_digests=actualRGB,independent_Source_envelope_CDFs=cdfs,max_CDF_error=maxerr,all_original_physical65_nodes=True,unique_nonfull_views=unique,changed_actual_student_windows=changes,all_frozen_spatial_arrays_exact=True,no_GT_read=True,new_model_calls=0));print('independent view audit pass',checks,actualRGB,maxerr,flush=True)

if __name__=='__main__':main()
