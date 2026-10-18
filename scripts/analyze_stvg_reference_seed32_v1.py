"""Sealed raw-reference forward response, no new predictions or GT access."""
from pathlib import Path
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_reference_seed32_v1 import D,P,verify
from scripts.run_stvg_joint_event_critic_v1 import load,put,read

def main():
 import numpy as np,torch
 from scripts.run_stvg_dino_local_residual32_v1 import mean
 p=verify();rows=[]
 for r in p['records']:
  o=load(D/'predictions'/(r['alias']+'.pt'));sv=o['values']['Source_reference'];ev=o['values']['E_reference'];layers=[]
  for j in range(6):
   delta=torch.cat([(b['all_six_absolute_outputs'][j]-a['all_six_absolute_outputs'][j]).flatten(0,1)for a,b in zip(sv['traces'],ev['traces'])]);layers.append(dict(mean_box_L2=float(delta.norm(dim=-1).mean()),max_coordinate_abs=float(delta.abs().max()),mean_center_L2=float(delta[...,:2].norm(dim=-1).mean())))
  S=o['outputs']['Source_MBR'];E=o['outputs']['E_MBR'];rawS=sv['boxes'].numpy();rawE=ev['boxes'].numpy();rows.append(dict(alias=r['alias'],dataset=r['dataset'],active=o['active'],E_minus_S_reference_box_L2=float(np.linalg.norm(E-S,axis=-1).mean()),raw_D_E_minus_D_S_box_L2=float(np.linalg.norm(rawE-rawS,axis=-1).mean()),raw_D_E_minus_Source_box_L2=float(np.linalg.norm(rawE-S,axis=-1).mean()),per_layer=layers))
 groups={ds:dict(count=16,active=sum(r['active']for r in rows if r['dataset']==ds),query_means={k:mean([r[k]for r in rows if r['dataset']==ds])for k in ['E_minus_S_reference_box_L2','raw_D_E_minus_D_S_box_L2','raw_D_E_minus_Source_box_L2']},per_layer=[{k:mean([r['per_layer'][j][k]for r in rows if r['dataset']==ds])for k in rows[0]['per_layer'][0]}for j in range(6)])for ds in ['hc2','vidstg']}
 put(P/'REFERENCE_RESPONSE_PUBLIC.json',dict(status='complete',groups=groups,rows=rows,no_new_model_or_GT_reads=True,scope='forward response only; absolute output changes are not identity correctness or a local-refinement guarantee'))
if __name__=='__main__':main()
