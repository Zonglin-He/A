"""Postseal scalar attribution and independent original-E reconstruction."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_referdino_condition_hc128_v1 import D,P,verify

def main():
 import numpy as np,torch
 from scripts.run_stvg_joint_robust_correction32_v1 import input_pool
 from scripts.run_decota_paper_main_v1 import unpack_expert
 from vg_tta.stvg_error_state_correction_v1 import encode
 from vg_tta.stvg_dino_supported_correction32_v1 import estimate
 from vg_tta.stvg_joint_robust_correction32_v1 import apply
 assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass';p=verify(True);ff=read(P/'TEACHER_FRAME_DIAGNOSTICS_PUBLIC.json')['rows'];results=read(P/'RESULTS_PUBLIC.json');cost=read(P/'COST_BREAKDOWN.json');cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')['source_domains']['vidstg'];start=time.perf_counter();mx=0.;original_no_reference=0
 for r in p['records']:
  old=torch.load(ROOT/r['prior_input'],map_location='cpu',weights_only=False);z=load(D/'inputs'/(r['alias']+'.pt'));assert torch.equal(old['native_boxes'],z['Source']);fs=input_pool(unpack_expert(old['expert']));original_no_reference+=not bool(fs);pools=[encode(f['boxes'])-encode(z['Source'].double().numpy())[f['position']]for f in fs];res=estimate(pools,[f['scores']for f in fs],cal['scale'],cal['truncation'],1/16);E=apply(z['Source'].double().numpy(),res['delta']);err=float(np.max(abs(E-z['E'])));mx=max(mx,err);assert err<1e-14
 attribution={}
 for label,student in [('T0','E'),('T1','S1')]:
  attribution[label]={}
  for own in [True,False]:
   known=[x for x in ff if x['GT_known'] and x[label+'_intervened_selected']==own];bad=[x for x in known if x['values'][student]<.5];good=[x for x in known if x['values'][student]>=.5];attribution[label]['intervened_selected'if own else'other_lineage_selected']=dict(known=len(known),bad=len(bad),recovered=sum(x['values'][label]>=.5 for x in bad),good=len(good),damaged=sum(x['values'][label]<.5 for x in good))
  attribution[label]['near_native_candidate_IoU_ge099']=sum(x['nearest_native_candidate_IoU'][label]>=.99 for x in ff)
 times={k:dict(mean=float(np.mean([r[k]['expert_seconds']for r in cost['rows']])),median=float(np.median([r[k]['expert_seconds']for r in cost['rows']])),total=float(sum(r[k]['expert_seconds']for r in cost['rows'])))for k in ['native','T0','T1']}
 out=dict(status='pass',postseal_scalar_analysis=True,all_original_E_reconstructed=128,max_E_error=mx,original_E_no_admitted_reference_parents=original_no_reference,attribution=attribution,teacher_timing_seconds=times,trajectory_decoder_calls=sum(r[s]['decoder_calls']for r in cost['rows']for s in ['student1','student2']),offset_decoder_calls=sum(r[s]['offset_decoder_calls']for r in cost['rows']for s in ['student1','student2']),original_time_source_logits_not_retained=True,time_scope='canonical finalMBR computed during capture; fixed exactly across every arm; no independent temporal probability reconstruction claimed',CPU_seconds=time.perf_counter()-start)
 put(P/'POST_SEAL_ANALYSIS.json',out);print({k:v for k,v in out.items()if k not in ['time_scope']},flush=True)
if __name__=='__main__':main()
