"""Old-A2 contract checks only. This tool rejects final mode and reads no GT."""
import sys
import numpy as np
from common46 import *
def main():
 a=args(__doc__);plan,rows,out=context(a);assert plan['execution_mode']=='engineering_oldA2';verify(out)
 import torch
 torch.set_num_threads(4);findings=[];oldGsame=True;numeric=verify(out/'oldG/evidenceM64_attempt001');dino=verify(out/'oldG/dinoM64_attempt001');assert numeric['new_calls']<=136 and numeric['non_numeric_VLM_calls']==0 and dino['calls']<=16
 steps=0
 for r in rows:
  z=npz(out/'source/samples_PRIVATE'/(r['id']+'.npz'));old=npz(ROOT/'crossplatform/round24/capture336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));p=npz(out/'main/predictions_PRIVATE'/(r['id']+'.npz'));names=p['arm_names'].tolist();assert np.array_equal(z['boxes'],old['boxes']) and np.array_equal(z['scores'],old['scores'])
  assert np.array_equal(p['queries'],p['scores'].argmax(-1)) and np.array_equal(p['boxes'],z['boxes'][p['queries']])
  b=npz(ROOT/'crossplatform/round23/bank272_attempt001/samples_PRIVATE'/(r['id']+'.npz'));FG=old['scores'].astype(float)+2*b['vision']+2*b['position']+2*b['relation'];qhist=int(FG.argmax());qnow=int(p['queries'][names.index('source_plus_current_oldG')]);same=qhist==qnow;oldGsame&=same
  newbank=npz(out/'oldG/bankM64_attempt001/samples_PRIVATE'/(r['id']+'.npz'));componenterrors={k:float(np.abs(newbank[k]-b[k]).max()) for k in ('vision','position','relation')}
  lib=npz(out/'main/memory_PRIVATE'/(r['platform']+'_library.npz'));token=z['tokens'][z['token_mask']].mean(0);token/=max(float(np.linalg.norm(token)),1e-12);key=z['query']*token;key/=np.maximum(np.linalg.norm(key,axis=-1,keepdims=True),1e-12)
  for name,vfield,ix in [('R24_memory_alpha1','V24_dense_G_probability_increment',np.arange(104)),('R30_memory_E_only','V30_preference_probability_increment',lib['E_indices']),('R30_memory_allA_zeroU','V30_preference_probability_increment',np.arange(104))]:
   nn=npz(out/'main/memory_PRIVATE'/(r['id']+'_'+name+'.npz'));K=torch.tensor(lib['A_keys'][ix]);V=torch.tensor(lib[vfield][ix]);sim=torch.einsum('qc,anc->qan',torch.tensor(key),K);near,which=sim.max(-1);best,obs=near.topk(min(4,len(ix)),-1);pos=which.gather(1,obs);w=((best-1)/.1).exp();corr=(V[obs,pos]*w).sum(-1)/w.sum(-1).clamp_min(1e-12)
   assert np.array_equal(ix[obs.numpy()],nn['A_row_indices']) and np.array_equal(pos.numpy(),nn['A_query_indices']) and np.array_equal(corr.numpy(),nn['correction_probability_units']);assert np.array_equal(p['scores'][names.index(name)],(z['scores']+corr.numpy()).astype(float))
  assert np.array_equal(p['scores'][names.index('source_plus_current_oldG')],p['oldG_F64'])
  for name in names[:6]:
   if name!='source':assert np.array_equal(p['scores'][names.index(name+'_plus_current_oldG')],p['scores'][names.index(name)].astype(float)+p['shared_deltaG_F64'])
  findings.append({'platform':r['platform'],'source_boxes_scores_bitexact':True,'literal_historical_oldG_query_exact':same,'historical_oldG_q':qhist,'current_oldG_q':qnow,'bank_component_maxerrors':componenterrors,'source_G_delta_addback_maxerror':float(np.abs(p['source_delta_addback_F64_diagnostic']-p['oldG_F64']).max()),'all12readouts_originalboxes_exact':True,'all3actualmemory_indices_and_corrections_recomputed_exact':True})
 for row in read(out/'external/TRACE_PRIVATE.json'):
  assert row['source_bridge_exact'];steps+=sum(d['steps'] for d in row['diagnostics'].values());assert all(d['nonselected_parameters_unchanged'] for d in row['diagnostics'].values())
 assert steps<=8
 result={'state':'PASS_OLD_A2_FULL_CONTROLLER_CONTRACTS' if oldGsame else 'MEASUREMENT_BRIDGE_STOP_OLDG_QUERY_NOT_REPLAYED','rows':findings,'numeric_Qwen_calls':numeric['new_calls'],'non_numeric_VLM_calls':0,'DINO_calls':dino['calls'],'external_optimizer_steps':steps,'predictions_sealed_before_audit':True,'task_GT_or_FINAL_access':False,'prediction_seal_sha256':sha(out/'SEALED.json')};dump(out/'ENGINEERING_CONTRACT_AUDIT.json',result);print(result['state'],flush=True)
 if not oldGsame:raise SystemExit('OldG identity readout bridge differs; do not access final before review')
if __name__=='__main__':main()
