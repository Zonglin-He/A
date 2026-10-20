"""Audit actual DenseLocal OPD chains independently, then score sealed HC128."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,fcntl,signal,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_denselocal_opd128_v1 import D,P,OLD,OP,SEEDS,STATIC,FITS,forbid_GT
from scripts.score_stvg_temporal_conditional_opd128_v1 import audit_fit
from scripts.score_stvg_where_when_event128_v1 import avg,paired
from scripts.score_stvg_temporal_local_global_mix128_v1 import tails

def main():
 import torch
 from main_method.runtime import resource_guard
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 torch.set_num_threads(4);beg=time.perf_counter();phase={'GT':False}
 def block(event,args):
  if not phase['GT']:forbid_GT(event,args)
 sys.addaudithook(block);p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
 def check():resource_guard();assert time.perf_counter()-beg<900
 assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')
 assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files'])
 steps=changed=first=spacechecks=sourcechecks=mbrchecks=nonzero=0;maxerr=maxutility=0.;ii,jj=np.triu_indices(65,1)
 for r in p['records']:
  check();a=r['alias'];z=load(D/'predictions'/f'{a}.pt');old=load(OLD/'predictions'/f'{a}.pt');assert sha(OLD/'predictions'/f'{a}.pt')==r['dense_prediction_sha256']
  assert np.array_equal(z['q0'],old['outputs']['HiResAuxMix']['q'])and np.array_equal(z['teacher'],old['outputs']['DenseLocal']['q']);sourcechecks+=2
  assert z['frame_ids']==old['frame_ids']and set(z['outputs'])==set(p['arms'])
  for name,out in z['outputs'].items():
   q=out['q'];assert np.array_equal(q>0,z['q0']>0)and abs(q.sum()-1)<1e-10
   v=final_mbr(z['frame_ids'],q);assert v['interval']==out['interval']
   choices=np.unique(np.r_[np.linspace(0,2079,13).astype(int),np.flatnonzero((ii==v['indices'][0])&(jj==v['indices'][1]))])
   err=float(abs(brute_utility(z['frame_ids'],q,choices)-v['utility_grid'][ii[choices],jj[choices]]).max());assert err<1e-11;maxutility=max(maxutility,err);mbrchecks+=len(choices)
   if name in FITS:
    assert out['actual_updates']==len(out['trace'])and out['scheduled_updates']==40
    v=audit_fit(z['q0'],z['teacher'],out);steps+=v['steps'];changed+=v['Live_steps_differ_from_initial_sampler'];maxerr=max(maxerr,v['maxerror'])
    if out['trace']:assert not out['trace'][0]['eta_before'].any()
    nonzero+=int(not np.array_equal(q,z['q0']))
  for name,src in [('Native','Native'),('Strong','HiResAuxMix'),('TeacherDirect','DenseLocal'),('DenseLocalMix','DenseLocalMix')]:
   assert np.array_equal(z['outputs'][name]['q'],old['outputs'][src]['q'])and z['outputs'][name]['interval']==old['outputs'][src]['interval']
  for seed in SEEDS:
   f,l=z['outputs'][f'Fixed{seed}'],z['outputs'][f'Live{seed}']
   if f['trace']:
    for k in ['sampled_pairs','gradient','eta_after','uniforms','teacher_conditional_digest','loss']:assert np.array_equal(f['trace'][0][k],l['trace'][0][k])
   first+=1
  for name,b in z['trajectories'].items():assert np.array_equal(b,old['trajectories'][name]);spacechecks+=1
 cost=read(D/'COST.json');assert cost['success']and steps==cost['actual_updates']==cost['completed_trace_steps']==cost['completed_fit_updates']and first==384 and spacechecks==512
 audit=dict(status='pass',independent_numpy_gradient_Adam_sampling_steps=steps,maxerror=maxerr,actual_fits=cost['actual_fits'],changed_posterior_fits=nonzero,
 first_Fixed_Live_exact=first,Live_steps_differ_from_initial_sampler=changed,source_initial_and_teacher_exact=sourcechecks,spatial_arrays_exact=spacechecks,
 independent_MBR_sample_utilities=mbrchecks,max_direct_utility_error=maxutility,actual_updates=steps,new_model_or_GT=False)
 put(D/'POLICY_AUDIT_BEFORE_GT.json',audit)
 assert all((R/f['path']).stat().st_mtime<time.time()for f in bar['files']);check()
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')));phase['GT']=True
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows']
 previous={r['alias']:r for r in read(OP/'RESULTS_PUBLIC.json')['rows']};rows=[];officialchecks=oldmetrics=0
 for r in p['records']:
  check();a=r['alias'];z=load(D/'predictions'/f'{a}.pt');values={};actual={};diagnostics={}
  for name,f in z['outputs'].items():
   if name in FITS:diagnostics[name]=dict(actual_updates=f['actual_updates'],eta_norm=float(np.linalg.norm(f['eta'])),posterior_L1=float(abs(f['q']-z['q0']).sum()),
    initial_loss=f['trace'][0]['loss']if f['trace']else 0.,last_preupdate_loss=f['trace'][-1]['loss']if f['trace']else 0.,**f['fit_diagnostics'])
  for space,b in z['trajectories'].items():
   tube=DenseTube(b,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True)
   sel=np.isin(tube.fids,np.array(sorted(set(z['frame_ids'])-set(r['physical_frame_ids8']))));actual[space]=dict(nodes=int(sel.sum()),sIoU=float(tube.iou[sel].mean())if sel.any()else None,all_temporal_arms_identical=True)
   assert actual[space]==previous[a]['fixed_unobserved_actual_sIoU'][space]
   for name,o in z['outputs'].items():
    iv=o['interval'];v=tube.score(iv);off=official(b,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;officialchecks+=1
    v=dict(**v,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2,interval=iv)
    if name in STATIC:
     src={'Native':'Native','Strong':'HiResAuxMix','TeacherDirect':'DenseLocal','DenseLocalMix':'DenseLocalMix'}[name]
     assert all(abs(v[k]-previous[a]['values'][space+'_'+src][k])<2e-10 for k in ['v','t','s','boundary_MAE']);oldmetrics+=1
    values[space+'_'+name]=v
  for name in STATIC+['DirectJoint','DirectConditional']:
   values['Mean_'+name]={k:avg([values[f'Live{s}_{name}'][k]for s in SEEDS])for k in ['v','t','s','boundary_MAE']}
  for name in ['Fixed','Live']:
   values['Mean_'+name]={k:avg([values[f'Live{s}_{name}{s}'][k]for s in SEEDS])for k in ['v','t','s','boundary_MAE']}
   values['EMean_'+name]={k:avg([values[f'E_{name}{s}'][k]for s in SEEDS])for k in ['v','t','s','boundary_MAE']}
  rows.append(dict(alias=a,values=values,policy_updates=diagnostics,fixed_unobserved_actual_sIoU=actual));print('scored_DenseLocal_OPD',a,flush=True)
 names=STATIC+['DirectJoint','DirectConditional','Fixed','Live'];means={name:{k:avg([r['values']['Mean_'+name][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for name in names}
 contrasts=[('Mean_Live','Mean_'+b)for b in ['Strong','Native','TeacherDirect','DenseLocalMix','DirectJoint','DirectConditional','Fixed']]
 contrasts += [('Mean_'+a,'Mean_Strong')for a in ['DirectJoint','DirectConditional','Fixed']]+[(f'Live{s}_Live{s}',f'Live{s}_Fixed{s}')for s in SEEDS]
 stats={};aliases=[r['alias']for r in rows]
 for aa,bb in contrasts:
  d={k:paired([(100 if k!='boundary_MAE'else 1)*(r['values'][aa][k]-r['values'][bb][k])for r in rows])for k in ['v','t','s','boundary_MAE']}
  delta=np.array([100*(r['values'][aa]['v']-r['values'][bb]['v'])for r in rows]);d['tails']=dict(**tails(delta,aliases),positive_contribution_pp=float(np.maximum(delta,0).mean()),negative_contribution_pp=float(np.minimum(delta,0).mean()));stats[aa+'_minus_'+bb]=d
 res=dict(status='completed_actual_DenseLocal_temporal_OPD_exposed_development',parents=128,seeds=SEEDS,means=means,paired=stats,rows=rows,spatial_frozen=True,
 teacher='PURE cached DenseLocal',student_initial='Strong HiResAuxMix',teacher_RGB_refresh=False,student_state_conditioned_target_lookup=True,all_network_training=0)
 put(P/'RESULTS_PUBLIC.json',res)
 audit.update(new_target_GT_scored=True,official_DenseTube_checks=officialchecks,previous_static_metrics_exact=oldmetrics,GT_provenance=prov,
 all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']))
 assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit)
 net=stats['Mean_Live_minus_Mean_Strong']['v'];refs={b:stats['Mean_Live_minus_Mean_'+b]['v']for b in ['TeacherDirect','DirectJoint','DirectConditional','Fixed']}
 decision=dict(primary_net_over_current_unadapted_Strong=net,versus_same_teacher_controls=refs,
 point_gain_over_Strong=net['mean']>0,net_gain_interval_above_zero=net['CI95'][0]>0,
 independent_Live_value_intervals_above_zero=all(v['CI95'][0]>0 for v in refs.values()),
 actual_OPD_executed=True,not_confirmation=True,no_parameter_search=True,no_automatic_next_experiment=True,
 scope='one unchanged Strong-start conditional OPD configuration; only teacher replaced, no universal OPD verdict')
 put(P/'DECISION_PUBLIC.json',decision)
 fullcost=dict(**cost,CPU_audit_and_scoring_seconds=time.perf_counter()-beg,publication_and_archive_excluded=True,upstream_cache_cost=read(OLD/'COST.json'))
 put(P/'COST_PUBLIC.json',fullcost)
 lines=['# Pure DenseLocal driven temporal OPD: actual HC128 experiment','','Only supervision changes from the existing Strong-start temporal OPD. Initial B=exact saved HiResAuxMix; T=exact pure DenseLocal final-layer posterior, no new mixture or teacher. Actual130-dimensional temporary state,40 Adam0.1 updates,32 intervals per stochastic step,seeds41/42/43. Fixed freshly samples B each step; Live freshly samples CURRENT p_eta and queries the fixed teacher conditional tables at its sampled boundaries. Original fitter, loss, optimizer and finalMBR reused unchanged.','','| Output | full vIoU percent | tIoU percent | boundaryMAE frames |','|---|---:|---:|---:|']
 for name,v in means.items():lines.append(f"| {name} | {100*v['v']:.6f} | {100*v['t']:.6f} | {v['boundary_MAE']:.6f} |")
 lines+=['','| Comparison | full-v pp [CI95] | tIoU pp [CI95] | losses below-5pp |','|---|---|---|---:|']
 for k,v in stats.items():lines.append(f"| {k} | {v['v']['mean']:+.6f} {v['v']['CI95']} | {v['t']['mean']:+.6f} {v['t']['CI95']} | {v['tails']['below_minus5']} |")
 lines+=['','All128 development parents included, repeatedly exposed; seed paired temporal/spatial then mean within parent. Static/Direct time outputs average the same three frozen spatial trajectories. Other crossseed/E values retained and never selected. No independent generalization claim; no hard-readout/previous diagnostic prerequisite gate. Fixed/Live equal update/candidate budget;32 sampled intervals each ask TWO full teacher conditional vectors (up to65 classes), not32 scalar rewards or teacher network forwards. Teacher conditional labels are derived posterior tables, not native autoregressive network predictions.','','```json',json.dumps(decision,indent=2),json.dumps(fullcost,indent=2),json.dumps(audit,indent=2),'```','','All predictions, actual samples/teacher digests/gradients/eta chains sealed before GT; independent NumPy audit reconstructs the existing gradient and Adam/sampling algorithm. Spatial arrays and fixed unobserved actual-node sIoU identical. Network weights never loaded or updated; CPU probability-state adaptation is real, with perquery/fit optimizer reset. Complete signed means, each seed, no-change cases and severe tails retained. No new RGB, GPU, models or experts. CPU adaptation latency excludes prior1531.960485s multi-arm Dense-cache generation and other upstream Strong/spatial costs; it is not an end-to-end deployment latency. No scan, automatic confirmation, new route or registry/spatial promotion follows this result.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],spatial_config_sha256=sha(R/'methods/stvg_referdino_opd_v1/config.json'),teacher='PURE DenseLocal cached final-layer posterior',exposure='RC128 repeatedly exposed development',excluded=['media','captions','parentIDs','GT coordinates','weights','raw policy/teacher tensors/traces']))
 assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files']);check()
 print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':
 def timeout(signum,frame):raise TimeoutError('finite900s DenseLocal OPD audit/score')
 signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,timeout);signal.alarm(900)
 lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:main()
 except BaseException as e:put(D/f'SCORING_FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e)));raise
 finally:lock.close()
