"""Replay descriptive mass diagnostic only; reads no annotation/model/media."""
from pathlib import Path
import numpy as np

def diagnose(plan,load_prediction):
 rows=[]
 for r in plan['records']:
  z=load_prediction(r['alias']);q=z['outputs']['Native']['q'];c=z['outputs']['Contrast']['q'];low=(q>0)&(q<.1*q.max())
  rows.append(dict(alias=r['alias'],native_mass_below_tenth_max=float(q[low].sum()),contrast_mass_below_tenth_max=float(c[low].sum())))
 a=np.array([r['native_mass_below_tenth_max']for r in rows]);b=np.array([r['contrast_mass_below_tenth_max']for r in rows])
 return dict(parents=len(rows),native_mass_mean=float(a.mean()),contrast_mass_mean=float(b.mean()),mass_increase_mean=float((b-a).mean()),parents_mass_increases_over_point1=int(((b-a)>.1).sum()),parents_contrast_mass_above_half=int((b>.5).sum()),rows=rows)

if __name__=='__main__':
 import sys,json
 root=Path(__file__).resolve().parents[4];sys.path.insert(0,str(root))
 from scripts.run_stvg_joint_event_critic_v1 import read,load
 d=root/'artifacts/stvg_temporal_motion_counterfactual128_v1'
 print(json.dumps(diagnose(read(d/'PLAN_LOCK.json'),lambda a:load(d/'predictions'/f'{a}.pt')),indent=2))
