import sys,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch,numpy as np
from research.joint_event_v1.run import sha,read,put,records,B
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.temporal import expected_iou_fast
from research.event_presence_v1.audit import auc
from research.event_presence_v1.core import word_logits

def main():
 torch.set_num_threads(1);paper=read(R/'outputs/tastvg_paper_v1/plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for k,h in paper['code'].items():assert sha(R/k)==h
 for k,h in paper['provider_sources'].items():assert sha(R.parent/'stvg-opd'/k)==h
 v1=R/'outputs/event_presence_hc64_v1';v2=R/'outputs/event_presence_hc64_v2';s1=read(v1/'sealed.json');s2=read(v2/'sealed.json');assert s1['count']==s2['count']==64
 controls=['NativeG6','RawVerb','Ownership','Noun','Source'];exact=0;replay=0;denseerror=0.;zvals={1:[],2:[]};aucrows={1:[],2:[]}
 for row in records():
  k=row['id'];e1=s1['files'][k];e2=s2['files'][k]
  for d,s,e in [(v1,s1,e1),(v2,s2,e2)]:assert sha(d/e['path'])==e['sha256']and sha(d/e['trace_path'])==e['trace_sha256']
  with np.load(v1/e1['path'])as a,np.load(v2/e2['path'])as b:
   for c in controls:
    assert np.array_equal(a[c],b[c])and np.array_equal(a[c+'_interval'],b[c+'_interval']);exact+=1
   if k in ['RC009','RC016','RC042']:
    cap=torch.load(B/'captures'/f'{k}.pt',weights_only=False);geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry'])
    for d,e,z in [(v1,e1,a),(v2,e2,b)]:
     for c in ['NativeG6','Conditional','ShiftConditional']:
      pred=evidence_dp(geom.for_interval(z[c+'_interval'].tolist()))['final'].numpy();assert np.array_equal(pred,z[c]);replay+=1
      t=torch.load(d/e['trace_path'],weights_only=False);q=t['q'][c];ids=z['frame_ids'];pairs=t['pairs'];spans=np.c_[ids[pairs[:,0]],ids[pairs[:,1]]+1];inter=np.maximum(0,np.minimum(spans[:,None,1],spans[None,:,1])-np.maximum(spans[:,None,0],spans[None,:,0]));un=(spans[:,1]-spans[:,0])[:,None]+(spans[:,1]-spans[:,0])[None,:]-inter;dense=(inter/un)@q[pairs[:,0],pairs[:,1]];fast=expected_iou_fast(ids,q)[pairs[:,0],pairs[:,1]];err=float(abs(dense-fast).max());assert err<1e-12;denseerror=max(denseerror,err)
  for stage,d,e in [(1,v1,e1),(2,v2,e2)]:
   t=torch.load(d/e['trace_path'],weights_only=False);zvals[stage].append(max(abs(v['extrapolation_z'])for v in t['calibration']))
 result=dict(valid=True,queries=64,controls_compared=controls,bitwise_control_pairs=exact,independent_spatial_replays=replay,independent_dense_MBR_checks=18,max_dense_MBR_error=denseerror,paper_sources_unchanged=len(paper['code']),provider_sources_unchanged=len(paper['provider_sources']),paper_status='paused_by_user',stage1_queries_with_nominal_extrapolation_z_gt3=sum(z>3 for z in zvals[1]),stage2_queries_with_nominal_extrapolation_z_gt3=sum(z>3 for z in zvals[2]),stage1_v2_numeric_controls_exact=True)
 for d in [v1,v2]:put(d/'verification.json',result)
 put(R/'results/event_presence_hc64_v2/verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
