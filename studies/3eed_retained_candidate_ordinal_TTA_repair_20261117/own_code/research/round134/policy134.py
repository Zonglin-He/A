"""Pre-score task-relevant readout ablation; no new gradients or task labels."""
from common134 import *
import numpy as np
POLICY134=['tasklinked_pair_F','tasklinked_wrongpair_F','tasklinked_direct_F','tasklinked_partial3_direct_F']
def outputs(ps,ev):
 q0=int(ps['zero_F'].argmax());g0=int(ev['assignment'][q0]);eligible=bool((ev['lose']==g0).any())
 return {k:ps[a if eligible else b].copy() for k,a,b in [('tasklinked_pair_F','partial_pair_F','partial3_F'),('tasklinked_wrongpair_F','partial_wrongpair_F','partial3_F'),('tasklinked_direct_F','pair_direct_F','partial_direct_F'),('tasklinked_partial3_direct_F','partial3_pair_direct_F','partial3_F')]},eligible
def main():
 mat=rd(B134/'MATRIX_SEALED.json');reg=rd(H134/'INPUT_REGISTRY_PRIVATE.json');n=0;enabled=0;files={}
 for e in reg:
  name=e['source']+'_to_'+e['target'];sid=e['row']['id'];D=B134/'formal'/name
  with np.load(D/'samples_PRIVATE'/(sid+'.npz')) as f:ps=dict(zip(f['arm_names'],f['scores']))
  ev=np.load(R134/e['evidence_path']);out,gate=outputs(ps,ev);p=B134/'policy_PRIVATE'/name/(sid+'.npz');p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,scores=np.stack([out[k] for k in POLICY134]),arm_names=np.array(POLICY134),eligible=gate);files[str(p.relative_to(R134))]=sha(p);enabled+=gate;n+=1
 assert n==1798 and enabled==12;dump(B134/'POLICY_SEALED.json',{'n':n,'eligible':enabled,'arms':POLICY134,'files':files,'MATRIX_SEALED_sha256':sha(B134/'MATRIX_SEALED.json'),'task_GT_used':False,'before_task_scoring':True});print('POLICY134',n,enabled)
if __name__=='__main__':main()
