"""Independent NumPy expected-utility enumeration from original cache, CPU only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
B=ROOT/'artifacts/stvg_interval_decision_v1'
def run():
 import numpy as np,torch
 checks=0;worst=0.;shuffles=[]
 for rec in read(B/'PLAN_LOCK.json')['files']:
  assert sha(ROOT/rec['path'])==rec['sha256'];old=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);new=load(B/'readouts'/(rec['alias']+'.pt'))
  for off,(z,record)in enumerate(zip(old['capture']['zero']['logits'],old['capture']['records'])):
   zz=z.numpy().reshape(-1,2).astype(np.float64);ids=record['frame_ids'];pairs=[(s,e)for s in range(len(ids))for e in range(s+1,len(ids))];intervals=np.array([[ids[s],ids[e]+1]for s,e in pairs],np.float64);scores=np.array([zz[s,0]+zz[e,1]for s,e in pairs]);q=np.exp(scores-scores.max());q/=q.sum();util=np.zeros(len(pairs));length=intervals[:,1]-intervals[:,0]
   # Independently sum each possible true hypothesis's utility contribution.
   for (s,e),p in zip(intervals,q):
    overlap=np.maximum(0,np.minimum(e,intervals[:,1])-np.maximum(s,intervals[:,0]));util+=p*overlap/(e-s+length-overlap)
   selected=int(util.argmax());actual=new['offset_intervals']['native_MBR'][off];assert actual==intervals[selected].astype(int).tolist();err=abs(util[selected]-new['audit'][off]['MBR']['expected_tIoU']);worst=max(worst,float(err));assert err<1e-12;checks+=len(util)+2
   pr=new['offset_intervals']['projection'][off];same=np.where(length==pr[1]-pr[0])[0];idx=same[q[same].argmax()];assert new['offset_intervals']['projection_length_native_location'][off]==intervals[idx].astype(int).tolist();checks+=1
 for f,h in read(B/'RUNTIME_LOCK.json')['registries'].items():assert sha(ROOT/f)==h
 put(B/'ROOT_EXPECTED_UTILITY_READBACK.json',dict(status='pass',count=64,independent_interval_checks=checks,maximum_expected_tIoU_error=worst,all_MBР_and_exact_length_selections_recomputed=True,new_model_calls=0,registries_unchanged=True))
 print('Independent utility audit',checks,worst)
if __name__=='__main__':run()
