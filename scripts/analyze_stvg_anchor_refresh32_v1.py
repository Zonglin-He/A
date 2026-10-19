"""No-GT per-layer counterfactual response from sealed native/DINO/E traces."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_anchor_refresh32_v1 import D,P,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def main():
 import torch,numpy as np
 p=verify();assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==32;rows=[]
 for r in p['records']:
  o=load(D/'predictions'/(r['alias']+'.pt'));raw=o['raw_calls'];E=o['E'].numpy();mask=np.ones(len(E),bool);mask[o['planned_positions']]=False;layer=[]
  if o['active']:
   for i in range(6):
    item=dict(layer=i)
    for schedule in ['initial','persistent']:
     finals=[]
     for v in range(2):
      b=raw[schedule+'_DINO']['traces'][v]['layers'][i]['raw_output'][:,0];c=raw[schedule+'_E']['traces'][v]['layers'][i]['raw_output'][:,0];finals.append((b-c).numpy())
     delta=np.stack([finals[j%2][j//2]for j in range(len(E))]);item[schedule]=dict(raw_center_delta_unobserved_mean=float(np.linalg.norm(delta[mask,:2],axis=1).mean())if mask.any()else None,raw_all_coordinate_delta_L2=float(np.linalg.norm(delta)),raw_changed_unobserved_nodes=int(np.any(delta[mask]!=0,axis=1).sum()),raw_max_absolute_coordinate_delta=float(abs(delta).max()))
    layer.append(item)
  rows.append(dict(alias=r['alias'],dataset=r['dataset'],active=o['active'],retained_anchors=len(o['positions']),unanchored_offsets=[v for v in range(2)if not any(p%2==v for p in o['positions'])],by_layer=layer))
 groups={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds and r['active']];groups[ds]=dict(active=len(rr),both_offsets_anchored=sum(not r['unanchored_offsets']for r in rr),by_layer=[])
  for i in range(6):groups[ds]['by_layer'].append(dict(layer=i,**{mode:{key:float(np.mean([r['by_layer'][i][mode][key]for r in rr]))for key in rr[0]['by_layer'][i][mode]}for mode in ['initial','persistent']}))
 assert not torch.cuda.is_initialized();put(P/'LAYER_RESPONSE_PUBLIC.json',dict(status='complete',count=32,only_saved_tensor_arithmetic=True,no_NEW_GT_model_or_candidate_calls=True,raw_coordinate_units_not_pp=True,groups=groups,rows=rows));print(groups,flush=True)
if __name__=='__main__':main()
