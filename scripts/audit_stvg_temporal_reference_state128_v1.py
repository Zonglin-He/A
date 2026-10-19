"""No-GT independent prefix-integral audit of every conditional query choice."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import sys,time,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.qualify_stvg_temporal_reference_state128_v1 import D,PRE,ST


def main():
 import torch
 torch.set_num_threads(4);beg=time.perf_counter();p=read(D/'QUALIFICATION_LOCK.json');out=read(D/'QUALIFICATION_PUBLIC.json');rows={r['alias']:r for r in out['rows']};assert sha(R/'scripts/qualify_stvg_temporal_reference_state128_v1.py')==p['script_sha256'];checks=0;affected=[]
 for r in p['records']:
  assert all(sha(R/f)==h for f,h in r['cache_pins'].items());a=r['alias'];inp=load(PRE/'inputs'/f'{a}.pt');teacher=load(PRE/'teacher0'/f'{a}.pt')['native'];state=load(ST/'predictions'/f'{a}.pt');ids=inp['frame_ids'];ti=np.array(ids)[r['positions8']];edges=np.r_[0.,(ti[:-1]+ti[1:])/2,float(r['frame_count'])];ll=teacher['trace']['all_logits'].double();prob=torch.sigmoid(ll).numpy()[:,:,0];width=np.diff(edges);prefix=np.vstack([np.zeros((1,15)),np.cumsum(width[:,None]*prob,axis=0)]);ii,jj=np.triu_indices(len(ids),1);iv=np.stack([np.array(ids)[ii],np.array(ids)[jj]+1],-1);I=np.vstack([[0,r['frame_count']],inp['interval'],state['outputs']['Live41']['interval'],iv]);scores=np.stack([(np.interp(I[:,1],edges,prefix[:,q])-np.interp(I[:,0],edges,prefix[:,q]))/(I[:,1]-I[:,0])for q in range(15)],1);chosen=scores.argmax(1);row=rows[a];assert chosen[:3].tolist()==[row['global_duration'],row['state0'],row['state1']];assert np.unique(chosen[3:]).tolist()==row['all_candidate_unique_queries'];native=teacher['selected_query'];assert np.array_equal(teacher['trace']['all_boxes'][:,native].numpy(),teacher['boxes'].numpy());checks+=len(iv)
  if (chosen[3:]!=native).any():affected.append(a)
 assert len(affected)==6
 result=dict(status='pass',independent_all_candidate_query_choices=checks,affected_parent_aliases=affected,affected_parents=6,native_selected_boxes_exact=128,all_cache_and_code_pins_unchanged=True,new_GT_reads=0,new_task_scoring=False,new_model_calls=0,CPU_seconds=time.perf_counter()-beg,scope='query and box variation, not teacher correctness')
 put(D/'INDEPENDENT_AUDIT.json',result);print('independent query coverage pass',checks,len(affected),flush=True)

if __name__=='__main__':main()
