"""Corrected fresh-run entrypoint. Existing immutable experiment refuses rerun.

Requires the official data/checkpoints/source manifests described in protocols.
The historical 001/002/003 handoffs are evidence of this actual run; a fresh
run uses canonical source binding and integer memberships from the beginning.
"""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import BASE,read
def run(script,stage):subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts'/script),stage],cwd=ROOT,check=True)
if __name__=='__main__':
 assert not BASE.exists(),'Immutable existing experiment; preserve it, do not overwrite'
 main='run_stvg_joint_event_critic_v1.py';labels='stvg_joint_event_source_labels002.py'
 run(main,'prepare');run(labels,'bind');run(main,'target');run(main,'headroom')
 if read(BASE/'CANDIDATE_HEADROOM.json')['continue_source_pilot']:
  run(main,'source');run(labels,'labels')
  import numpy as np
  from scripts.run_stvg_joint_event_critic_v1 import load
  from vg_tta.stvg_joint_event_critic_v1 import candidate_tensor
  for group in ['source','target']:
   for path in (BASE/group).glob('*.pt'):
    z=load(path)
    for interval in z['candidates']['intervals']:
     x,q=candidate_tensor(z['features'][0],interval,z['input']['fps']);ids=np.array(z['frame_ids']);assert np.array_equal(x[:,-2].bool().numpy(),(ids>=interval[0])&(ids<interval[1]))
  for stage in ['train','infer','score']:run(main,stage)
