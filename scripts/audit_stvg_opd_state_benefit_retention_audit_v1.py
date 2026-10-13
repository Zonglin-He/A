"""Independent cache/source/math/frame/parent statistic readback, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_state_benefit_retention_common_v1 import *

def near(a,b):
 if isinstance(a,np.ndarray):
  assert isinstance(b,np.ndarray) and a.shape==b.shape and np.allclose(a,b,rtol=1e-10,atol=2e-10,equal_nan=False)
  return a.size
 if isinstance(a,dict):assert set(a)==set(b),(set(a)-set(b),set(b)-set(a));return sum(near(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):assert len(a)==len(b);return sum(near(x,y) for x,y in zip(a,b))
 if a is None or isinstance(a,(str,bool,int)):assert a==b,(a,b)
 else:assert abs(a-b)<2e-10*(1+abs(a)),(a,b)
 return 1
def decision(summary):
 result={}
 for ds,d in summary['datasets'].items():
  result[ds]={}
  for a,b in PAIRS[:2]:
   key=a+'_to_'+b;c=d['contrasts'][key];ci=c['own_U_RMS_delta']['estimate']['ci95'];loss=c['own_U_correct_advantaged_persistent_loss_fraction']['estimate']
   joint=c['own_U_joint_agreement_erosion_persistent'];er=c['own_U_erosion']['estimate']['ci95']
   result[ds][key]=dict(mean_output_convergence_supported=ci is not None and ci[1]<0,
    persistent_absolute_loss_in_initially_correct_advantaged_U_observed=loss['mean'] is not None and loss['mean']>0,
    joint_parent_evidence_count=joint['true_parents'],advantage_erosion_supported=er is not None and er[1]<0,
    headroom_causality_established=False,previous_query_forgetting_tested=False)
 return dict(status='complete_scoped_retention_diagnostic',scope='current-query inherited spatial benefit retention/output convergence/expert agreement/actual saved gradients; previous-query forgetting and causal headroom untested',
  datasets=result,catastrophic_forgetting_established=False,deployable_unlabelled_predictor_established=False,
  new_gate_or_loss_implemented=False,new_method_frozen=False,production_promoted=False,automatic_next_experiment=False,CPU_only=True,GT_read=True,time=time.time())
def root():
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();bar=require_seal()
 from scripts.run_stvg_opd_state_benefit_retention_audit_v1 import cached
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 activate_old();counts=collections.Counter();scalars=0
 for r in read(BASE/'DESIGN_LOCK.json')['records']:
  computed,c=cached(r,True,True);saved=load(BASE/'readouts'/f"{r['index']:05}.pt")
  assert np.array_equal(computed['boxes'],saved['boxes']);scalars+=near(computed,saved);counts.update(c)
  status(BASE/'ROOT_STATUS.json',dict(status='independent_original_source_math_and_cached_output_readback',done=counts['fits'],total=192,CPU_only=True,GT_read=False,time=time.time()))
  print('RETENTION_ROOT_SOURCE',r['index']+1,64,flush=True)
 assert counts['fits']==192 and counts['updates']==4800 and counts['original_full_fits']==128 and counts['empty_fits']==21
 from scripts.score_stvg_opd_state_benefit_retention_audit_v1 import assemble,summarise
 with gzip.open(OUT/'FRAME_ROUND_ROWS.jsonl.gz','rt') as f:
  lines=iter(f);counter=[0,0]
  def check(row):
   counter[0]+=near(row,json.loads(next(lines)));counter[1]+=1
  pp,cc,c,provenance=assemble(True,check)
  assert next(lines,None) is None
 scalars+=counter[0]+near(pp,rows(OUT/'PARENT_ROWS.jsonl.gz'))+near(cc,rows(OUT/'ROUND_ROWS.jsonl.gz'))
 summary=summarise(pp,cc,True);scalars+=near(summary,read(OUT/'SUMMARY.json'))
 from scripts.score_stvg_opd_LN_state_intervention_v1 import assemble as dense
 endpoints,_,binding=dense(True);scalars+=near(endpoints,rows(OUT/'DENSE_ENDPOINT_ROWS.jsonl.gz'));provenance.update(binding)
 for parent,end in zip(pp,endpoints):
  assert parent['parent_id']==end['parent_id'] and parent['dataset']==end['dataset']
  for arm in ARMS:
   for mask in MASKS:
    scalars+=near(parent['arms'][arm][mask+'_before'],end[arm+'_'+mask+'_before'])
    scalars+=near(parent['arms'][arm][mask+'_after'],end[arm+'_'+mask+'_after'])
 assert c['frame_boundaries']==counter[1] and not torch.cuda.is_initialized();verify()
 write(BASE/'ROOT_READBACK.json',dict(status='pass',scope='all192 immutable fits/all4800 updates/all4992 arm boundaries/all genuine frame labels and output distances/128 original full-fit equality/source/state/pixels/evidence/common noise/Adam/native objective/full dense/independent paired parent bootstrap',
  counts=dict(counts),parents=64,fits=192,frame_round_rows=counter[1],parent_round_rows=1664,scalar_checks=scalars,GT_provenance=provenance,
  new_model_calls=0,new_optimizer_steps=0,independent_full_decoder_Jacobian=False,CPU_only=True,GT_read=True,time=time.time()))
 write(BASE/'ROOT_ANONYMOUS_SCALAR_READBACK.json',dict(status='pass',parents=64,frame_round_rows=counter[1],scalar_checks=scalars,
  all_positive_negative_neutral_unknown_included=True,empty_fits=21,CPU_only=True,GT_read=True,time=time.time()))
 write(BASE/'MECHANISM_DECISION.json',decision(summary))
if __name__=='__main__':root()
