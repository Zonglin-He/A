"""Independent all-parent local-clip/input/CDF audit, then sealed task scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,fcntl,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_local_motion_fixedbatch128_v1 import D,P,OLD,EARLY,PRE,FIN,DEV,STAGE,ARMS
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def main():
 import torch
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_registered_motion128_v1 import tensor_sha
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);beg=time.perf_counter();phase={'GT':False}
 def guard(event,args):
  if not phase['GT']and event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):raise PermissionError('local motion preaudit forbids GT')
 sys.addaudithook(guard);p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')and bar['weights_and_flags_unchanged']and bar['all_before_GT'];assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files']);assert len(bar['qualification_files'])==2 and all(sha(R/f['path'])==f['sha256']for f in bar['qualification_files']);raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];spaces=oldexact=cdfs=utils=indexchecks=inputchecks=0;maxcdf=0.;ii,jj=np.triu_indices(65,1);diagnostics={}
 def check():resource_guard();assert time.perf_counter()-beg<900
 for r in p['records']:
  check();a=r['alias'];assert sha(EARLY/'early_inputs'/f'{a}.pt')==r['early_sha256']and sha(OLD/'predictions'/f'{a}.pt')==r['registered_sha256'];z=load(D/'predictions'/f'{a}.pt');o=load(OLD/'predictions'/f'{a}.pt');e=load(EARLY/'early_inputs'/f'{a}.pt');ids=z['frame_ids'];diag=[]
  for arm in ['DenseLocal','SparseLocal']:
   ix=z['clip_indices'][arm];assert ix.shape==(65,8)and np.array_equal(ix[:,3],ids)
   for t in range(65):
    seq=ids[t%2::2];q=seq.index(ids[t])
    for j in range(8):
     want=min(ids[-1],max(ids[0],ids[t]+2*(j-3)))if arm=='DenseLocal'else seq[min(len(seq)-1,max(0,q+j-3))];assert int(ix[t,j])==want;indexchecks+=1
   feature=z['motion_features'][arm];assert feature.shape[:2]==(65,256)and torch.isfinite(feature).all()
   for batch in z['batches'][arm]:
    j=batch['start'];n=batch['count'];packed=batch['packed_clip_indices'];assert n==min(8,65-j)and packed.shape==(8,8)and batch['padded_slots']==8-n and np.array_equal(packed[:n],ix[j:j+n]);assert all(np.array_equal(row,packed[n-1])for row in packed[n:]);assert tensor_sha(feature[j:j+n])==batch['projected_center_sha256'];inputchecks+=1
   diag.append(dict(arm=arm,physical_unique_frames=len(set(ix.flatten().tolist())),new_unique_frames=len(set(ix.flatten().tolist())-set(ids)),clamp_repeats=int(sum(len(row)-len(set(row.tolist()))for row in ix)),mean_physical_clip_span=float(np.mean(ix[:,-1]-ix[:,0]))))
  for off in (0,1):
   cap=e['captured_encoder_inputs'][off];hh,ww=e['original_views'][off]['info']['fea_map_size'];l=hh*ww;src=cap['src'];assert all(z['metadata'][str(off)][k]['position_padding_exact']and z['metadata'][str(off)][k]['actual_encoder_entry_verified']and z['metadata'][str(off)][k]['appearance_text_input_exact']for k in ['Native','DenseLocal','SparseLocal'])
   for arm in ['Native','DenseLocal','SparseLocal']:
    x=src.clone()
    if arm!='Native':x[-l:]=z['motion_features'][arm][off::2].to(x).flatten(2).permute(2,0,1)
    assert torch.equal(x[:-l],src[:-l])and tensor_sha(x)==z['metadata'][str(off)][arm]['actual_input_sha256'];inputchecks+=1
  for arm in ['Native','HiResAuxMix']:assert np.array_equal(z['outputs'][arm]['q'],o['outputs'][arm]['q'])and z['outputs'][arm]['interval']==o['outputs'][arm]['interval'];oldexact+=1
  for arm in ['Native','DenseLocal','SparseLocal']:
   gs=probabilities(z['records'],z['logits'][arm]);grid,q=mass_table(gs);ind=independent_cdf(gs,grid);err=float(abs(ind-q).max());assert grid==ids and err<2e-13 and np.array_equal(q,z['outputs'][arm]['q']);cdfs+=1;maxcdf=max(maxcdf,err)
   if arm!='Native':assert np.array_equal(.5*z['outputs']['Native']['q']+.5*q,z['outputs'][arm+'Mix']['q'])
  for arm,v in z['outputs'].items():
   d=final_mbr(ids,v['q']);assert d['interval']==v['interval'];sel=np.unique(np.r_[np.linspace(0,len(ii)-1,13).astype(int),np.flatnonzero((ii==d['indices'][0])&(jj==d['indices'][1]))]);bu=brute_utility(ids,v['q'],sel);assert abs(bu-d['utility_grid'][ii[sel],jj[sel]]).max()<1e-11;utils+=len(sel)
  for space,bx in o['trajectories'].items():assert np.array_equal(bx,z['trajectories'][space]);spaces+=1
  diagnostics[a]=diag
 qualchecks=0;new_full_permutation_checks=0;single_errors=[]
 for rr,receipt in zip(p['records'][:2],bar['qualification_files']):
  qc=load(R/receipt['path']);first=load(EARLY/'early_inputs'/f"{rr['alias']}.pt");assert qc['diagnostics']['original_R30_qualification_passed']is False
  for off in (0,1):
   pr=qc['controls']['Native'+str(off)]['projected'];hh,ww=first['original_views'][off]['info']['fea_map_size'];want=first['captured_encoder_inputs'][off]['src'][-hh*ww:].permute(1,2,0).reshape(pr.shape);assert torch.equal(pr,want);qualchecks+=1
  for arm in ['DenseLocal','SparseLocal']:
   c=qc['controls'][arm];md=qc['diagnostics'][arm];assert md['same_shape_repeat_exact']and md['full_clip_permutation_equivariant_exact']
   for k in ['raw','projected']:
    assert torch.equal(c['Batch'][k],c['Repeat'][k]);qualchecks+=1
    for t in range(8):assert torch.equal(c['Batch'][k][t*8:(t+1)*8],c['Reverse'][k][(7-t)*8:(8-t)*8]);new_full_permutation_checks+=1
    for t in range(1,8):assert torch.equal(c['Tail'][k][:8],c['Tail'][k][t*8:(t+1)*8]);qualchecks+=1
   single=c['Single']['projected'].float().numpy();batch=c['Batch']['projected'][:8].float().numpy();err=float(np.max(abs(single-batch)/np.maximum(1,abs(single))));assert abs(err-md['single_fullclip_max_relative_error'])<1e-7;assert md['comparison_with_original_bound_passed']==(md['single_fullclip_max_relative_error']<md['original_bound']);single_errors.append(err)
 assert spaces==512 and oldexact==256 and cdfs==384 and indexchecks==133120
 audit=dict(status='pass',frozen_spatial_arrays_exact=spaces,old_posteriors_intervals_exact=oldexact,independent_clip_indices=indexchecks,actual_input_and_center_feature_SHA_checks=inputchecks,independent_CDF_checks=cdfs,max_CDF_error=maxcdf,independent_MBR_sample_utilities=utils,Native_motion_exact=True,same_batch_repeat_exact=True,batch_single_relative_errors=single_errors,original_R30_qualification_passed=False,new_exact_qualification_checks=qualchecks,independent_full_clip_permutation_checks=new_full_permutation_checks,tail_padding_outputs_excluded=True,new_GT=False);put(D/'AUDIT_BEFORE_GT.json',audit);assert all(sha(R/f)==h for f,h in p['pins'].items());put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')));phase['GT']=True
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));rows=[];officialchecks=scalarcalls=oldmetric=0;oldpublic={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-20/temporal_registered_motion128/RESULTS_PUBLIC.json')['rows']}
 for r in p['records']:
  check();a=r['alias'];z=load(D/'predictions'/f'{a}.pt');ids=np.array(z['frame_ids']);values={};curves={k:[]for k in ['v','t']};actual={}
  for space,boxes in z['trajectories'].items():
   tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True);sel=np.isin(tube.fids,np.array(sorted(set(ids)-set(r['physical_frame_ids8']))));actual[space]=dict(nodes=int(sel.sum()),sIoU=float(tube.iou[sel].mean())if sel.any()else None,all_temporal_arms_identical=True)
   for name,o in z['outputs'].items():
    iv=o['interval'];v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;officialchecks+=1;values[space+'_'+name]=dict(**v,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2)
    if name in ['Native','HiResAuxMix']:assert all(abs(values[space+'_'+name][k]-oldpublic[a]['values'][space+'_'+name][k])<2e-10 for k in ['v','t','s','boundary_MAE']);oldmetric+=1
   assert len({values[space+'_'+name]['s']for name in ARMS})==1
   if space!='E':
    vals=[tube.score([int(ids[i]),int(ids[j])+1])for i,j in zip(ii,jj)];scalarcalls+=len(vals)
    for k in curves:curves[k].append(np.array([v[k]for v in vals]))
  for name in ARMS:values['Mean_'+name]={k:avg([values[f'Live{s}_{name}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  quality={k:np.mean(v,0)for k,v in curves.items()};expected={name:{k:float(o['q'][ii,jj]@curve)for k,curve in quality.items()}for name,o in z['outputs'].items()};rows.append(dict(alias=a,values=values,expected_candidate_quality=expected,diagnostics=diagnostics[a],fixed_unobserved_actual_sIoU=actual));print('score_local_motion',a,flush=True)
 means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in ['Mean_'+a for a in ARMS]};ds={};expected={};pairs=[('DenseLocalMix',b)for b in ['Native','HiResAuxMix','SparseLocalMix','DenseLocal']]+[('DenseLocal',b)for b in ['Native','HiResAuxMix','SparseLocal']]+[('SparseLocalMix','Native'),('SparseLocal','Native'),('HiResAuxMix','Native')]
 for a,b in pairs:
  key=a+'_minus_'+b;delta=np.array([100*(r['values']['Mean_'+a]['v']-r['values']['Mean_'+b]['v'])for r in rows]);ds[key]={k:paired([(100 if k in ['v','t','s']else 1)*(r['values']['Mean_'+a][k]-r['values']['Mean_'+b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};ds[key]['tails']=dict(positive=int((delta>1e-10).sum()),negative=int((delta< -1e-10).sum()),flat=int((abs(delta)<=1e-10).sum()),below_minus3=int((delta< -3).sum()),below_minus5=int((delta< -5).sum()),leave_one_out_mean_range_pp=[float(((delta.sum()-delta)/127).min()),float(((delta.sum()-delta)/127).max())],worst=sorted([dict(alias=r['alias'],delta_pp=float(v))for r,v in zip(rows,delta)],key=lambda x:x['delta_pp'])[:8]);expected[key]={k:paired([100*(r['expected_candidate_quality'][a][k]-r['expected_candidate_quality'][b][k])for r in rows])for k in ['v','t']}
 check();assert all(sha(R/f)==h for f,h in p['pins'].items());cost=read(D/'COST.json');put(P/'RESULTS_PUBLIC.json',dict(status='completed_exposed_fixedbatch_local_motion_capability_not_OPD',parents=128,rows=rows,means=means,paired=ds,expected_quality=expected,cost=cost,spatial_frozen=True,original_R30_qualification_passed=False,execution_revision='fixed B8 with tail-repeat padding; exact fullclip permutation/replay gates'));hard=ds['DenseLocalMix_minus_Native']['v'];dist=expected['DenseLocal_minus_Native']['v'];control=expected['DenseLocal_minus_SparseLocal']['v'];go_hard=hard['mean']>=2 and hard['CI95'][0]>0;go_dist=dist['mean']>=2 and dist['CI95'][0]>0 and control['CI95'][0]>0;decision=dict(primary_hard=hard,pure_teacher_expected=dist,Dense_versus_Sparse_expected=control,decision='qualifies_one_actual_controlled_transfer'if go_dist else'eligible_static_component_confirmation_only'if go_hard else'close_this_local_motion_configuration',versus_Strong=ds['DenseLocalMix_minus_HiResAuxMix']['v'],not_OPD=True,not_independent_confirmation=True,no_clip_stride_mixture_temperature_sweep=True,goal_achieved=False);put(P/'DECISION_PUBLIC.json',decision)
 audit.update(new_GT=True,official_Dense_output_checks=officialchecks,scalar_candidate_quality_calls=scalarcalls,old_metrics_exact=oldmetric,GT_provenance=prov,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),all_temporal_arms_sIoU_identical=True);assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit);sc=dict(CPU_audit_and_scoring_seconds=time.perf_counter()-beg,GPU_seconds=0,publication_archive_cost_excluded=True);put(D/'SCORING_COST.json',sc);put(P/'SCORING_COST_PUBLIC.json',sc)
 lines=['Actual entry scripts/score_stvg_temporal_local_motion_fixedbatch128_v1.py. Engineering qualification and execution specification explicitly revised before any task scores: fixed B8 with repeat-tail padding, same-shape full-clip permutation/repeat exact. Original R30/recovery001 remain failed, old single tolerance is disclosed without passing it retroactively. Prior engineering costs separately retained.', '# Fixed-batch local motion, unchanged global context: RC128','','Source Vid frozen; original65 decoder timestamps, full-image appearance/text and spatial E/Live41/42/43 exact. Dense8×stride2 and Sparse8×same-offset-neighbors have the same T=8, center3, no temporal shift. Higher density and shorter physical context are coupled; no isolated aliasing or pure motion-vs-local-appearance claim. No spatial crop, new expert, training, student/spatial update. Not OPD.','','| Comparison | full vIoU pp [CI95] | expected candidate GTv pp [CI95] |','|---|---|---|']
 for name,x in ds.items():lines.append(f"| {name} | {x['v']['mean']:+.6f} {x['v']['CI95']} | {expected[name]['v']['mean']:+.6f} {expected[name]['v']['CI95']} |")
 lines+=['','All128 repeatedly exposed parents; spatial seeds averaged within parent, paired parent bootstrap. All losses/tails and no-change parents retained. Strong is an existing stronger development reference. Hard readout failure and transfer qualification are assessed separately; no method/OPD claim without matched actual transfer and independent confirmation.','','```json',json.dumps(decision,indent=2),json.dumps(ds['DenseLocalMix_minus_Native']['tails'],indent=2),json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','No clip/stride/mix/temperature rescue; close only exact negative configuration, not all OPD. No CURRENT/MAIN/PAPER or spatial config change. Candidate quality uses Dense scalar scoring, final outputs official/Dense double checked; MBR independent utility checks are predeclared samples, not all-action double calculation.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],exposure='HC2 RC128 repeatedly exposed; no independent confirmation',excluded=['captions','parentIDs','media','GT','weights','raw caches']))
 for f in P.glob('*.json'):
  if str(f.relative_to(R))not in p['pins']:f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 check();assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files']);print(json.dumps(decision,indent=2),flush=True);lock.close()
if __name__=='__main__':main()
