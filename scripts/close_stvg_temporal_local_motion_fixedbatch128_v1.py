"""CPU-only closure of two pilots at prelocked resource stop; NEVER task GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_local_motion_fixedbatch128_v1 import D,P,OLD,EARLY,ARMS

def main():
 import torch
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_registered_motion128_v1 import tensor_sha
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 def reject(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):raise PermissionError('pilot closure never reads task GT')
 sys.addaudithook(reject);torch.set_num_threads(4);beg=time.perf_counter();p=read(D/'PLAN_LOCK.json');pilot=read(D/'PILOT2.json');cost=read(D/'COST.json');assert all(sha(R/f)==h for f,h in p['pins'].items());assert not(D/'GLOBAL_PREDICTION_BARRIER.json').exists()and not(D/'GT_EXPOSURE.json').exists();assert len(list((D/'predictions').glob('*.pt')))==2 and len(list((D/'qualification').glob('*.pt')))==2;assert pilot['estimated_GPU_seconds']>p['GPU_seconds_cap']and not pilot['GT_read']and not cost['success'];assert cost['RGB_slots']==2978
 assert all(cost['actual_calls'][k+'_'+s]==v for k,v in dict(motion=52,projection=52,fusion=12,decoder=24,head=12).items()for s in ['attempts','completed']);rows=[];files=[];spaces=oldexact=cdfs=indexchecks=inputchecks=permchecks=qualchecks=prefixchecks=utilitychecks=0;maxcdf=0.;ii,jj=np.triu_indices(65,1)
 for r in p['records'][:2]:
  resource_guard();a=r['alias'];z=load(D/'predictions'/f'{a}.pt');q=load(D/'qualification'/f'{a}.pt');old=load(OLD/'predictions'/f'{a}.pt');e=load(EARLY/'early_inputs'/f'{a}.pt');ids=z['frame_ids'];assert sha(OLD/'predictions'/f'{a}.pt')==r['registered_sha256']and sha(EARLY/'early_inputs'/f'{a}.pt')==r['early_sha256'];assert ids==old['frame_ids']and set(z['outputs'])==set(ARMS)and not z['GT_read']and not z['spatial_updated'];outputs={}
  for path in [D/'predictions'/f'{a}.pt',D/'qualification'/f'{a}.pt']:files.append(read(path.with_suffix('.receipt.json')))
  for arm in ['DenseLocal','SparseLocal']:
   ix=z['clip_indices'][arm];assert ix.shape==(65,8)and np.array_equal(ix[:,3],ids)
   for t in range(65):
    seq=ids[t%2::2];n=seq.index(ids[t])
    for j in range(8):
     expected=min(ids[-1],max(ids[0],ids[t]+2*(j-3)))if arm=='DenseLocal'else seq[min(len(seq)-1,max(0,n+j-3))];assert ix[t,j]==expected;indexchecks+=1
   feature=z['motion_features'][arm];assert feature.shape[:2]==(65,256)and torch.isfinite(feature).all()
   for batch in z['batches'][arm]:
    j=batch['start'];n=batch['count'];packed=batch['packed_clip_indices'];assert n==min(8,65-j)and packed.shape==(8,8)and batch['padded_slots']==8-n and np.array_equal(packed[:n],ix[j:j+n]);assert all(np.array_equal(row,packed[n-1])for row in packed[n:]);assert tensor_sha(feature[j:j+n])==batch['projected_center_sha256'];inputchecks+=1
   qc=q['controls'][arm]
   for stage in ['raw','projected']:
    assert torch.equal(qc['Batch'][stage],qc['Repeat'][stage]);qualchecks+=1
    for n in range(8):assert torch.equal(qc['Batch'][stage][n*8:(n+1)*8],qc['Reverse'][stage][(7-n)*8:(8-n)*8]);permchecks+=1
    for n in range(1,8):assert torch.equal(qc['Tail'][stage][:8],qc['Tail'][stage][n*8:(n+1)*8]);qualchecks+=1
   s=qc['Single']['projected'].float().numpy();b=qc['Batch']['projected'][:8].float().numpy();err=float(np.max(abs(s-b)/np.maximum(1,abs(s))));assert abs(err-q['diagnostics'][arm]['single_fullclip_max_relative_error'])<1e-7
  assert q['diagnostics']['original_R30_qualification_passed']is False
  for off in (0,1):
   cap=e['captured_encoder_inputs'][off];ov=e['original_views'][off];h,w=ov['info']['fea_map_size'];l=h*w;src=cap['src'];pr=q['controls']['Native'+str(off)]['projected'];want=src[-l:].permute(1,2,0).reshape(pr.shape);assert torch.equal(pr,want);qualchecks+=1
   for arm in ['Native','DenseLocal','SparseLocal']:
    x=src.clone()
    if arm!='Native':x[-l:]=z['motion_features'][arm][off::2].to(x).flatten(2).permute(2,0,1)
    m=z['metadata'][str(off)][arm];assert tensor_sha(x)==m['actual_input_sha256']and torch.equal(x[:-l],src[:-l])and m['appearance_text_input_exact']and m['position_padding_exact']and m['actual_encoder_entry_verified'];inputchecks+=1
    if arm=='Native':assert m['prefix_sha256']==tensor_sha(ov['prefix']);prefixchecks+=1
  for arm in ['Native','HiResAuxMix']:assert np.array_equal(z['outputs'][arm]['q'],old['outputs'][arm]['q'])and z['outputs'][arm]['interval']==old['outputs'][arm]['interval'];oldexact+=1
  assert all(torch.equal(x,y)for x,y in zip(z['logits']['Native'],old['logits']['Native']))
  for arm in ['Native','DenseLocal','SparseLocal']:
   gs=probabilities(z['records'],z['logits'][arm]);grid,v=mass_table(gs);other=independent_cdf(gs,grid);err=float(abs(v-other).max());assert grid==ids and err<2e-13 and np.array_equal(v,z['outputs'][arm]['q']);cdfs+=1;maxcdf=max(maxcdf,err)
   if arm!='Native':assert np.array_equal(.5*z['outputs']['Native']['q']+.5*v,z['outputs'][arm+'Mix']['q'])
  for arm,v in z['outputs'].items():
   d=final_mbr(ids,v['q']);assert d['interval']==v['interval'];sel=np.unique(np.r_[np.linspace(0,len(ii)-1,13).astype(int),np.flatnonzero((ii==d['indices'][0])&(jj==d['indices'][1]))]);bu=brute_utility(ids,v['q'],sel);assert abs(bu-d['utility_grid'][ii[sel],jj[sel]]).max()<1e-11;utilitychecks+=len(sel);flat=v['q'][ii,jj];outputs[arm]=dict(interval=v['interval'],posterior_sha256=__import__('hashlib').sha256(v['q'].tobytes()).hexdigest(),probability_total=float(flat.sum()),ESS=float(1/(flat*flat).sum()),full_vIoU=None,tIoU=None,GT_score_status='not_read_or_evaluated')
  for space,bx in old['trajectories'].items():assert np.array_equal(bx,z['trajectories'][space]);spaces+=1
  rows.append(dict(alias=a,outputs=outputs,new_exact_qualification=q['diagnostics'],all_spatial_arrays_exact=True))
 assert (spaces,oldexact,cdfs,indexchecks,permchecks,qualchecks,prefixchecks)==(8,4,6,2080,64,68,4)
 assert sha(R/p['source_checkpoint'])==p['source_checkpoint_sha256']and all(sha(R/f)==h for f,h in p['pins'].items());assert not(D/'GT_EXPOSURE.json').exists();put(D/'PILOT_CLOSURE_SEAL.json',dict(count=2,requested=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),no_GT=True,not_full_round_barrier=True));audit=dict(status='pass',scope='CPU independent pilot artifact audit only, not full128 or task-effect evaluation',pins=len(p['pins']),all_pins_exact=True,frozen_spatial_arrays_exact=spaces,old_posteriors_intervals_exact=oldexact,Native_prefix_logit_checks=prefixchecks,independent_CDF_checks=cdfs,max_CDF_error=maxcdf,independent_clip_indices=indexchecks,input_center_SHA_checks=inputchecks,fullclip_permutation_checks=permchecks,Native_repeat_tail_exact_checks=qualchecks,independent_MBR_sample_utilities=utilitychecks,new_GT=False,terminal_model_weight_flags_hash_check_reached=False,Source_checkpoint_file_exact=True)
 for f in ['ROOT_AUDIT.json','INDEPENDENT_AUDIT.json']:put(P/f,audit)
 decision=dict(status='stopped_at_prelocked_budget_qualification',requested_parents=128,completed_pilot_parents=2,full_round_completed=False,new_exact_pilot_qualification_passed=True,estimated_GPU_seconds=pilot['estimated_GPU_seconds'],budget_GPU_seconds=p['GPU_seconds_cap'],new_task_effect_measurement=False,GT_read=False,original_R30_qualification_passed=False,user_requested_stop_after_publication=True,no_further_exploration=True,goal_achieved=False);put(P/'DECISION_PUBLIC.json',decision);put(P/'RESULTS_PUBLIC.json',dict(status='two_sealed_pilots_unscored_budget_stop',parents=2,rows=rows,effect_metrics='unknown; no GT/task scoring',cost=cost,pilot=pilot,decision=decision));put(P/'COST_PUBLIC.json',cost);put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],data_role='first2 of prelocked repeatedly exposed RC128; no task GT',excluded=['private media','captions','parent IDs','GT','weights','raw feature caches']))
 lines=['# Fixed-batch local motion: budget stop after two complete pilots','','This round is closed at its prelocked resource gate. Two engineering pilots completed, all new exact checks passed, but the conservative128-parent estimate exceeds900s. No full128 barrier, GT access, effect scoring, static component/OPD claim or new experiment follows. User requested uploading existing results and stopping.','','| Item | Actual |','|---|---|',f"| Requested / completed parents |128 /2|",f"| Non-control pilot seconds | {pilot['case_seconds']} |",f"| Loading / control seconds | {pilot['loading']:.6f} / {pilot['controls_seconds']:.6f} |",f"| Conservative128 estimate / cap | {pilot['estimated_GPU_seconds']:.6f} /900s |",f"| Actual GPU lease+reserve | {cost['GPU_seconds']:.6f}s |",f"| Prior separately charged engineering | {cost['prior_engineering_GPU_seconds']:.6f}s |",'| New GT/effect scoring/expert/training/spatial or student updates |0|','','```json',json.dumps(decision,indent=2),json.dumps(audit,indent=2),json.dumps(cost,indent=2),'```','','Both pilots retained all six outputs, all four frozen spatial arrays and complete first2 qualification stages. Original motion/prefix/logits/q exact, fullclip permutation/repeat and tail duplicate checks passed. Single/batch differences and originalR30 failure remain disclosed; this revision does not retroactively pass the old8eps16 gate. Padding never entered65-node readout. The returned pilot intervals are included in RESULTS_PUBLIC with all GT metrics explicitly unknown. They are not performance results.','','Terminal whole-model weight/flags hash validation was after the full128 loop and was not reached because of the resource assertion. Source checkpoint/config/code pins are exact and the execution path has no optimizer/network/student/spatial updates; do not substitute that for an executed terminal runtime hash check. The CPU audit validates saved pilot tensors/receipts and declared exact controls only. No claim about temporal or OPD effectiveness is possible from this round. No budget extension, smaller scoring panel, parameter changes or continuation. Existing spatial configuration and CURRENT/MAIN/PAPER remain unchanged.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(D/'CPU_CLOSURE_COST.json',dict(seconds=time.perf_counter()-beg,GPU_seconds=0,GT_read=False,task_scoring=False));print(json.dumps(decision),flush=True)
if __name__=='__main__':main()
