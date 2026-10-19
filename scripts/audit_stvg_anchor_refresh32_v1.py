"""Independent CPU trace intervention/carry, paired readout and localization audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_anchor_refresh32_v1 import D,P,PRE,CLOSED,FEAT,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def main():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_outputs_before_NEW_GT'];z=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in z['rows']};checks=collections.Counter();maxerr=0.
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 for row in p['records']:
  a=row['alias'];o=load(D/'predictions'/(a+'.pt'));x=load(FEAT/'features'/(a+'.pt'));rc=load(CLOSED/'replay_inputs'/(a+'.pt'));src=load(PRE/'source'/(a+'.pt'));E=x['E'].double().numpy();pos=[f['position']for f,k in zip(x['frames'],x['selected'])if k>=0];targets=np.array([f['boxes'][k].tolist()for f,k in zip(x['frames'],x['selected'])if k>=0]).reshape(-1,4);assert pos==o['positions']and np.array_equal(targets,o['targets']);checks['fixed_reference_set']+=1
  calls=o['raw_calls'];native=calls['native_copy'];assert torch.equal(native['boxes'],src['Source']);checks['zero_Source_exact']+=1
  for v in range(2):
   direct=calls['native_original']['all_six_outputs'][v];copied=torch.stack([t['raw_output']for t in native['traces'][v]['layers']]).transpose(1,2);assert torch.equal(direct,copied);checks['actual_native_all_six_exact']+=1
  for key,call in calls.items():
   if key=='native_original':continue
   is_native=key=='native_copy';persistent=key.startswith('persistent');isE=key in ['initial_E','persistent_E','persistent_E_repeat'];tt=E[pos]if isE else np.roll(targets,1,axis=0)if key=='persistent_shuffle'and len(pos)>1 else targets
   pieces=[]
   for v,tr in enumerate(call['traces']):
    ids=[]if is_native else[j for j,g in enumerate(pos)if g%2==v];local=[pos[j]//2 for j in ids];assert tr['local_positions']==local;prev=rc['spatial_inputs'][v]['pred_boxes']
    for t in tr['layers']:
     assert torch.equal(t['before_reference'],prev);want=prev.clone()
     if local and(persistent or t['layer']==0):want[local,0]=torch.as_tensor(tt[ids],dtype=want.dtype)
     assert torch.equal(t['used_reference'],want);assert torch.isfinite(t['raw_output']).all()and(t['raw_output'][...,2:]>0).all();prev=t['raw_output'];checks['reference_intervention_and_own_path_carry']+=1
    pieces.append(prev[:,0])
   assembled=torch.stack([pieces[i%2][i//2]for i in range(len(E))]);assert torch.equal(assembled,call['boxes']);checks['raw_final_from_six_layer_trace']+=1
  assert len(calls)==o['compute']['actual_trajectory_decoder_calls']and o['compute']['actual_PosDecoder_calls']==2*len(calls)and o['compute']['backward_calls']==0;checks['physical_no_backward_receipt']+=1
  for arm,dk,ek,local in [('Initial_delta','initial_DINO','initial_E',True),('Persistent_delta','persistent_DINO','persistent_E',True),('Persistent_shuffle','persistent_shuffle','persistent_E',True),('Persistent_allnodes','persistent_DINO','persistent_E',False)]:
   want=E.copy()
   if pos:
    b=calls[dk]['boxes'].double().numpy();c=calls[ek]['boxes'].double().numpy();want=np.concatenate((E[:,:2]+(b[:,:2]-c[:,:2]),E[:,2:]*(b[:,2:]/c[:,2:])),1)
    if local:want[o['planned_positions']]=E[o['planned_positions']]
   err=float(np.max(abs(want-o['outputs'][arm])));maxerr=max(maxerr,err);assert err<1e-12;checks['independent_paired_readout_scalars']+=want.size
  local=E.copy()
  if pos:local[pos]=targets
  assert np.array_equal(local,o['outputs']['Anchor_local']);checks['direct_anchor_control']+=1
  for arm in ['Initial_delta','Persistent_delta','Persistent_shuffle']:assert np.array_equal(o['outputs'][arm][o['planned_positions']],E[o['planned_positions']]);checks['planned_nodes_exact_E']+=1
  if not pos:
   for arm,b in o['outputs'].items():assert np.array_equal(b,E);checks['no_reference_all_outputs_noop']+=1
  else:
   for v in range(2):
    if not any(p%2==v for p in pos):assert torch.equal(calls['persistent_DINO']['boxes'][v::2],calls['persistent_E']['boxes'][v::2])and np.array_equal(o['outputs']['Persistent_delta'][v::2],E[v::2]);checks['unanchored_offset_exact_E']+=1
  independent_metrics(o,plans[row['dataset']][row['query_ordinal']],gt[row['dataset']][0][row['query_ordinal']],gt[row['dataset']][1][row['query_ordinal']],rows[a],checks)
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds]
  for key,eps in g['paired'].items():
   a,b=key.split('_minus_')
   for ep,want in eps.items():assert paired([None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in rr])==want;checks['independent_parent_bootstrap']+=1
 assert not torch.cuda.is_initialized();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),max_paired_readout_error=maxerr,no_GPU_model_rerun=True));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),maxerr,flush=True)
if __name__=='__main__':main()
