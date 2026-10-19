"""Extra no-GT actual nonfull crop control, because RC001 is full-image."""
from pathlib import Path
import sys,hashlib,numpy as np
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,load,put
from scripts.run_stvg_temporal_tube_view128_v1 import D,P
from methods.stvg_referdino_opd_v1.runtime import setup,decode
setup();assert not(D/'GT_EXPOSURE.json').exists();p=read(D/'PLAN_LOCK.json');raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows']
for r in p['records']:
 z=load(D/'predictions'/f"{r['alias']}.pt");H,W=z['original_RGB_shape'][1:3]
 states=[s for s in [0,1]if z['observations'][s]['window']!=[0,0,W,H]]
 if not states:continue
 frames,ids=decode(raw[r['query_ordinal']]['input']);assert ids==z['frame_ids'];checks=0
 for s in states:
  obs=z['observations'][s];x0,y0,x1,y1=obs['window'];pixels=np.stack([frame[y0:y1,x0:x1,:]for frame in frames]);assert list(pixels.shape)==obs['shape']and hashlib.sha256(pixels.tobytes()).hexdigest()==obs['raw_RGB_sha256'];checks+=1
 put(P/'NONFULL_RGB_AUDIT_PUBLIC.json',dict(status='pass',alias=r['alias'],nonfull_actual_crops=checks,first_native_control_was_full_image=True,first_nonfull_actual_RGB_separately_redecoded=True,no_model_calls=True,no_GT=True,added_before_first_GT=True));print('nonfull actual crop audit',r['alias'],checks);break
else:raise AssertionError('no nonfull crop')
