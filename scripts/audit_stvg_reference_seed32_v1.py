"""Independent saved interface tensors, raw outputs and official scoring check."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_reference_seed32_v1 import D,P,PRE,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def main():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['all32_outputs_before_NEW_GT'];z=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in z['rows']};checks=collections.Counter()
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];o=load(D/'predictions'/(a+'.pt'));s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));S=s['Source'];E=torch.as_tensor(e['E']).double();assert all(record['frame_ids']==o['frame_ids'][view::2] for view,record in enumerate(s['records']));checks['actual_frame_offset_identity']+=2;assert np.array_equal(S,o['outputs']['Source_MBR'])and np.array_equal(E,o['outputs']['E_MBR']);assert torch.equal(o['values']['native']['boxes'],S);checks['native_reference_exact_Source']+=1
  for key,ref in [('Source_reference',S),('E_reference',E)]:
   v=o['values'][key];nat=o['values']['native']['traces'];parts=[]
   for view,t in enumerate(v['traces']):
    assert torch.equal(t['actual_reference'][:,0],ref[view::2].float());assert torch.equal(t['native_content'],nat[view]['native_content'])and torch.equal(t['native_reference'],nat[view]['actual_reference']);assert t['all_six_absolute_outputs'].shape[0]==6
    parts.append(t['all_six_absolute_outputs'].flatten(1,2)[-1]);checks['actual_reference_and_native_content']+=1;checks['all_six_saved_absolute_boxes']+=t['all_six_absolute_outputs'].numel()
   merged=torch.stack([parts[i%2][i//2]for i in range(len(S))]);assert torch.equal(merged,v['boxes']);actual=v['boxes'].numpy()if o['active']else E.numpy();assert np.array_equal(actual,o['outputs'][key]);checks['raw_readout_no_extra_E_transform']+=1
   if not o['active']:assert np.array_equal(o['outputs'][key],E);checks['no_ref_noop']+=1
  if not o['active']:assert torch.equal(o['values']['Source_reference']['boxes'],o['values']['E_reference']['boxes']);checks['same_reference_outputs_exact']+=1
  independent_metrics({**o,'outputs':{k:np.asarray(v,dtype=np.float64)for k,v in o['outputs'].items()}},plans[ds][r['query_ordinal']],gt[ds][0][r['query_ordinal']],gt[ds][1][r['query_ordinal']],rows[a],checks)
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds]
  for key,eps in g['paired'].items():
   aa,bb=key.split('_minus_')
   for ep,want in eps.items():
    got=paired([None if r['values'][aa][ep]is None or r['values'][bb][ep]is None else 100*(r['values'][aa][ep]-r['values'][bb][ep])for r in rr]);assert got==want;checks['independent_paired_bootstrap']+=1
 passed=all(z['groups'][ds]['paired'][c][ep]['mean']>0 for ds in gt for c in ['E_reference_minus_E_MBR','E_reference_minus_Source_reference']for ep in ['v','unobserved_actual_s']);assert passed==read(P/'GATE.json')['passed'];assert not torch.cuda.is_initialized();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),no_GPU_or_model_rerun=True,all_outputs_global_seal_before_NEW_GT=True,no_new_expert_training=True,production_registries_pinned=True));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),flush=True)
if __name__=='__main__':main()
