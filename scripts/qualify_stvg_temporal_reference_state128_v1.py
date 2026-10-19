"""No-GT expression-capacity audit of cached native ReferDINO query selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import sys,time,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
D=R/'artifacts/stvg_temporal_reference_state128_v1';MEM=R/'artifacts/stvg_temporal_object_memory128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';ST=R/'artifacts/stvg_temporal_conditional_opd128_v1'


def weights(intervals,edges):
 iv=np.asarray(intervals);w=np.maximum(0,np.minimum(iv[:,1,None],edges[1:])-np.maximum(iv[:,0,None],edges[:-1]));assert (w.sum(-1)>0).all();return w/w.sum(-1,keepdims=True)


def main():
 import torch
 from scipy.special import expit
 from methods.stvg_referdino_opd_v1.runtime import setup
 setup();torch.set_num_threads(4);records=read(MEM/'PLAN_LOCK.json')['records'];protocol=read(D/'QUALIFICATION_PROTOCOL.json');beg=time.perf_counter();rs=[]
 for r in records:
  files=[PRE/'teacher0'/f"{r['alias']}.pt",PRE/'inputs'/f"{r['alias']}.pt",ST/'predictions'/f"{r['alias']}.pt"]
  rs.append(dict(alias=r['alias'],query_ordinal=r['query_ordinal'],positions8=r['positions8'],frame_count=r['frame_count'],cache_pins={str(f.relative_to(R)):sha(f)for f in files}))
 lock=dict(records=rs,count=128,created_epoch=time.time(),script_sha256=sha(Path(__file__)),protocol_sha256=sha(D/'QUALIFICATION_PROTOCOL.json'),no_GT=True,no_STVG_metric=True,new_model_calls=0);put(D/'QUALIFICATION_LOCK.json',lock);rows=[];maxerr=0
 for r in rs:
  assert time.perf_counter()-beg<900;a=r['alias'];teacher=load(PRE/'teacher0'/f'{a}.pt')['native'];inp=load(PRE/'inputs'/f'{a}.pt');state=load(ST/'predictions'/f'{a}.pt');tr=teacher['trace'];allboxes=tr['all_boxes'].numpy();ll=tr['all_logits'].numpy().astype(float);assert allboxes.shape==(8,15,4)and ll.shape==(8,15,1);native=int(teacher['selected_query']);assert np.array_equal(allboxes[:,native],teacher['boxes'].numpy())and native==int(teacher['native_query_scores'].argmax())
  prob=expit(ll);scores=prob.mean(0).max(-1);err=float(abs(scores-teacher['native_query_scores'].numpy()).max());assert err<1e-6;maxerr=max(maxerr,err)
  ids=inp['frame_ids'];ti=np.array(ids)[r['positions8']];edges=np.r_[0.,(ti[:-1]+ti[1:])/2,float(r['frame_count'])];iv0=inp['interval'];iv1=state['outputs']['Live41']['interval'];I=np.array([[0,r['frame_count']],iv0,iv1]);choices=np.einsum('it,tqc->iqc',weights(I,edges),prob).max(-1).argmax(-1);global_id,i0,i1=map(int,choices);ii,jj=np.triu_indices(len(ids),1);ivs=np.stack([np.array(ids)[ii],np.array(ids)[jj]+1],-1);all_choices=np.einsum('it,tqc->iqc',weights(ivs,edges),prob).max(-1).argmax(-1);diff=np.max(abs(allboxes-allboxes[:,native:native+1]),axis=(0,2));pairdiff=float(abs(allboxes[:,i0]-allboxes[:,i1]).max());q0=state['q0'];q1=state['outputs']['Live41']['q'];assert np.array_equal(q0,state['q0']);u=np.random.default_rng(np.random.SeedSequence([41,r['query_ordinal'],20261020])).random(32);samples=[]
  for q in [q0,q1]:
   mass=q[ii,jj];cdf=np.cumsum(mass);cdf[-1]=1.;picks=np.searchsorted(cdf,u,side='right');samples.append(all_choices[picks])
  rows.append(dict(alias=a,native=native,uniform64_argmax=int(scores.argmax()),global_duration=global_id,state0=i0,state1=i1,state1_vs_state0_max_box_difference=pairdiff,state0_vs_native_max_box_difference=float(diff[i0]),state1_vs_native_max_box_difference=float(diff[i1]),all_candidate_unique_queries=np.unique(all_choices).tolist(),all_candidate_mean_max_box_difference=float(diff[all_choices].mean()),posterior0_changed_reference_mass=float(q0[ii,jj][all_choices!=native].sum()),posterior1_changed_reference_mass=float(q1[ii,jj][all_choices!=native].sum()),paired32_sample_query_change_count=int((samples[0]!=samples[1]).sum()),paired32_sample_actual_box_change_count=int(sum(np.max(abs(allboxes[:,b]-allboxes[:,c]))>1e-6 for b,c in zip(*samples))),current_policy_teacher_reference_variation_is_not_teacher_accuracy=True))
 out=dict(status='completed_no_GT_interface_qualification',parents=128,uniform_sigmoid_mean_score_max_error=maxerr,native_boxes_and_selected_query_exact=True,native_uniform64_argmax_changed=sum(r['uniform64_argmax']!=r['native']for r in rows),global_duration_vs_native_changed=sum(r['global_duration']!=r['native']for r in rows),state0_vs_global_duration_changed=sum(r['state0']!=r['global_duration']for r in rows),state1_vs_state0_query_changed=sum(r['state0']!=r['state1']for r in rows),state1_vs_state0_actual_boxes_changed=sum(r['state1_vs_state0_max_box_difference']>1e-6 for r in rows),all_candidate_query_variable_parents=sum(len(r['all_candidate_unique_queries'])>1 for r in rows),any_paired_sample_actual_box_change_parents=sum(r['paired32_sample_actual_box_change_count']>0 for r in rows),mean_paired32_sample_actual_box_change_rate=float(np.mean([r['paired32_sample_actual_box_change_count']/32 for r in rows])),mean_state1_vs_state0_max_box_difference=float(np.mean([r['state1_vs_state0_max_box_difference']for r in rows])),mean_posterior0_changed_reference_mass=float(np.mean([r['posterior0_changed_reference_mass']for r in rows])),mean_posterior1_changed_reference_mass=float(np.mean([r['posterior1_changed_reference_mass']for r in rows])),rows=rows,CPU_seconds=time.perf_counter()-beg,GPU_seconds=0,new_model_calls=0,new_GT_reads=0,task_metrics_computed=False,spatial_updates=0,new_OPD_updates=0,teacher_advantage_not_established=True,no_fresh_RGB_or_decoder_recomputation=True,lock_sha256=sha(D/'QUALIFICATION_LOCK.json'))
 put(D/'QUALIFICATION_PUBLIC.json',out);print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2),flush=True)

if __name__=='__main__':main()
