"""No-GT independent float64 audit of all cached feedback and actual attention."""
from pathlib import Path
import sys,math,numpy as np
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_motion_attribute128_v1 import D,P,CURVES


def main():
 p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['plan_sha256']==sha(D/'PLAN_LOCK.json');assert not (D/'GT_EXPOSURE.json').exists();headchecks=0;cells=protos=feedbacks=0;maxatt=maxgeom=maxq=0.
 for r in p['records']:
  z=load(D/'predictions'/f"{r['alias']}.pt");raw=z['raw_motion_class_logits'];ids=np.array(z['frame_ids'],float);ii,jj=np.triu_indices(65,1);a=ids[ii];b=ids[jj]+1;e=z['edges'];assert np.array_equal(e,np.r_[0.,(ids[:-1]+ids[1:])/2,r['frame_count']]);oc=[]
  for q in [z['q0'],z['q1']]:oc.append(np.array([sum(q[i,j]for i in range(t+1)for j in range(max(t,i+1),65))for t in range(65)]))
  assert max(abs(oc[0]-z['occupancy0']))<1e-12 and max(abs(oc[1]-z['occupancy1']))<1e-12
  qq={}
  for key,x,w in [('Global',raw['Global'],np.ones(65)),('Occ0',raw['Refer'],oc[0]),('Occ1',raw['Refer'],oc[1])]:
   mean=np.average(x.astype(float),weights=w,axis=0);v=1/(1+np.exp(-mean));v/=v.sum();assert abs(v-z['prototypes'][key]).max()<1e-12;qq[key]=v;protos+=1
  qq['ClassShift']=np.roll(qq['Global'],1);assert np.array_equal(qq['ClassShift'],z['prototypes']['ClassShift'])
  curves={c:1/(1+np.exp(-x.astype(float)))for c,x in raw.items()};expected=dict(GlobalQuery=curves['Global']@qq['Global'],GlobalMax=curves['Global'].max(-1),EQuery=curves['E']@qq['Global'],LiveQuery=curves['Live']@qq['Global'],ReferQuery=curves['Refer']@qq['Global'],ReferClassShift=curves['Refer']@qq['ClassShift'],ReferOcc0=curves['Refer']@qq['Occ0'],ReferOcc1=curves['Refer']@qq['Occ1']);expected['ReferTimeShift']=np.roll(expected['ReferQuery'],32)
  width=e[1:]-e[:-1];W=np.maximum(0,np.minimum(b[:,None],e[1:])-np.maximum(a[:,None],e[:-1]));inside=b-a;outside=e[-1]-inside;valid=outside>0
  for c in CURVES:
   x=expected[c];assert max(abs(x-z['curves'][c]))<1e-12;d=np.zeros(len(a));d[valid]=(W[valid]@x)/inside[valid]-((width-W[valid])@x)/outside[valid];zz=np.zeros_like(d)
   if d[valid].std()>1e-12 and np.ptp(x)>1e-12:zz[valid]=(d[valid]-d[valid].mean())/d[valid].std()
   positive=z['q0'][ii,jj]>0;target=np.zeros(len(a));v=np.log(z['q0'][ii[positive],jj[positive]])+.5*zz[positive];v=np.exp(v-v.max());target[positive]=v/v.sum();err=float(abs(target-z['outputs'][c]['q'][ii,jj]).max());assert err<1e-10,(r['alias'],c,err);maxq=max(maxq,err);feedbacks+=len(a)
  for tr in z['geometry_traces']:
   h,w=tr['h'],tr['w'];off=tr['offset'];xx=np.arange(w);yy=np.arange(h)
   for k,meta in tr['geometry'].items():
    bx=z['conditioning_boxes'][k][off::2];xmin=(bx[:,0]-bx[:,2]/2)*w;xmax=(bx[:,0]+bx[:,2]/2)*w;ymin=(bx[:,1]-bx[:,3]/2)*h;ymax=(bx[:,1]+bx[:,3]/2)*h
    dx=np.clip(np.minimum(xmax[:,None],xx+1)-np.maximum(xmin[:,None],xx),0,1);dy=np.clip(np.minimum(ymax[:,None],yy+1)-np.maximum(ymin[:,None],yy),0,1);area=(dy[:,:,None]*dx[:,None,:]).reshape(len(bx),-1);empty=area.sum(-1)<=0;area[empty]=1.;err=float(abs(area-meta['area'].numpy()).max());assert err<3e-8;maxgeom=max(maxgeom,err);assert np.array_equal(empty,meta['empty'].numpy());cells+=area.size
    for t in meta['attention_numeric_trace']or []:
     q=t['q'].numpy().astype(float);k=t['k'].numpy().astype(float);v=t['v'].numpy().astype(float);ar=t['area'].numpy().astype(float)
     with np.errstate(divide='ignore'):logits=q@k.swapaxes(-1,-2)/math.sqrt(q.shape[-1])+np.log(ar[:,None,None,:])
     logits-=logits.max(-1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(-1,keepdims=True);ctx=prob@v
     for actual,ex in [(t['prob'].numpy(),prob),(t['context'].numpy(),ctx)]:
      err=float((abs(actual-ex)/(1+abs(ex))).max());assert err<2e-5,err;maxatt=max(maxatt,err)
     headchecks+=1
 put(P/'NUMERIC_INDEPENDENT_AUDIT.json',dict(status='pass',all128_before_GT=True,attention_layers_actual_numeric_checks=headchecks,attention_scaled_maxerror=maxatt,fixed_attention_tolerance=2e-5,fractional_area_cells=cells,geometry_maxerror=maxgeom,prototype_checks=protos,interval_feedback_checks=feedbacks,posterior_maxerror=maxq,no_GT_read=True,no_new_model_calls=True))
 print('independent numeric pass',headchecks,maxatt,maxq,flush=True)

if __name__=='__main__':main()
