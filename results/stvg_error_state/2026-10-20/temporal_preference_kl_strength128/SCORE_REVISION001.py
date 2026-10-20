"""Coefficient-only OPD: independent weighted-gradient replay, then sealed evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,fcntl,signal,numpy as np
from scipy.special import logsumexp,expit,xlogy
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_preference_kl_strength128_v1 import D,P,OLD,OP,COEFFICIENTS,ALL,FITS,ARMS,SEEDS,forbid_GT
from scripts.score_stvg_temporal_strong_relative_preference_opd128_v1 import independent_feedback
from scripts.score_stvg_where_when_event128_v1 import avg,paired
from scripts.score_stvg_temporal_local_global_mix128_v1 import tails

def replay(Bgrid,dd,bb,maps,support,out):
 coefficient=out['kl_coefficient'];B=Bgrid[tuple(support.T)];r=np.zeros(len(B));m1=r.copy();m2=r.copy();rng=np.random.default_rng(out['seed']);u=rng.random((40,32,2));cdf=np.cumsum(B);cdf[-1]=1.
 bank=np.stack([np.searchsorted(cdf,u[:,:,0],side='right'),np.searchsorted(cdf,u[:,:,1],side='right')],axis=-1);assert np.array_equal(bank,out['bank'])
 maximum=0.;changed=admitted=total=collisions=0;seen=set();accepted=set();pref=[];globalKL=[]
 def close(a,b):
  nonlocal maximum
  e=float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))));maximum=max(maximum,e);assert e<2e-10,e
 for k,tr in enumerate(out['trace']):
  p=B.copy()if not r.any()else np.exp(np.log(B)+r-logsumexp(np.log(B)+r))
  pairs=bank.reshape(-1,2)if out['mode']=='Direct'else bank[k].copy()
  if out['mode']=='Live':
   cp=np.cumsum(p);cp[-1]=1.;pairs[:,0]=np.searchsorted(cp,u[k,:,0],side='right');changed+=int(not np.array_equal(pairs,bank[k]))
  assert np.array_equal(pairs,tr['pairs'])and np.array_equal(u[k],tr['uniforms'])
  fb=independent_feedback(dd,bb,maps,pairs)
  for key in fb:close(fb[key],tr['feedback'][key])
  # Raw-log arithmetic orders can differ at roundoff; the strict admitted sign must remain exact.
  assert np.array_equal(fb['w'],tr['feedback']['w'])
  v=r[pairs[:,0]]-r[pairs[:,1]];t=fb['t'];w=fb['w'];a=(expit(v)-t)*w/len(pairs)
  g=coefficient*p*(r-p@r);np.add.at(g,pairs[:,0],a);np.add.at(g,pairs[:,1],-a)
  pk=float(np.mean(w*(np.logaddexp(0,v)-t*v+xlogy(t,t)+xlogy(1-t,1-t))))
  kk=float(p@r-logsumexp(np.log(B)+r)+logsumexp(np.log(B)))
  close(r,tr['eta_before']);close(p,tr['pi_before']);close(g,tr['gradient']);close(pk,tr['preference_KL']);close(kk,tr['global_KL']);close(pk+coefficient*kk,tr['loss'])
  # Gradient checked independently above; replay recorded Adam using its audited exact gradient.
  actual_g=tr['gradient'];m1=.9*m1+.1*actual_g;m2=.999*m2+.001*actual_g*actual_g;r-=.1*(m1/(1-.9**(k+1)))/(np.sqrt(m2/(1-.999**(k+1)))+1e-8);r-=r.mean()
  close(m1,tr['adam_first']);close(m2,tr['adam_second']);close(r,tr['eta_after'])
  admitted+=int(w.sum());total+=len(w);collisions+=int(fb['collision'].any(1).sum());seen.update(pairs.ravel().tolist());accepted.update(pairs[w>0].ravel().tolist());pref.append(pk);globalKL.append(kk)
 close(r,out['eta']);expected=np.zeros_like(Bgrid);expected[tuple(support.T)]=B if not r.any()else np.exp(np.log(B)+r-logsumexp(np.log(B)+r));close(expected,out['q'])
 return dict(steps=len(out['trace']),maxerror=maximum,Live_batches_different_from_initial=changed,admitted=admitted,pairs_evaluated=total,
  admission_fraction=admitted/total,unique_sampled_intervals=len(seen),unique_admitted_intervals=len(accepted),native_collision_pairs=collisions,
  initial_preference_KL=pref[0],final_preupdate_preference_KL=pref[-1],initial_global_KL=globalKL[0],final_preupdate_global_KL=globalKL[-1],final_preupdate_weighted_global_KL=coefficient*globalKL[-1],
  eta_norm=float(np.linalg.norm(r)),posterior_L1=float(abs(out['q']-Bgrid).sum()))


def main():
 import torch
 from main_method.runtime import resource_guard
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 torch.set_num_threads(4);beg=time.perf_counter();phase={'GT':False}
 def guard(event,args):
  if not phase['GT']:forbid_GT(event,args)
 sys.addaudithook(guard);plan=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
 def check():resource_guard();assert time.perf_counter()-beg<880
 assert all(sha(R/f)==h for f,h in read(D/'EXECUTION_REVISION001_LOCK.json')['pins'].items())
 assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')
 assert all(sha(R/f)==h for f,h in plan['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files'])
 cost=read(D/'COST_REVISION001.json');assert cost['success'];steps=first=spacechecks=baselinechecks=mbrchecks=0;maxerr=maxutility=0.;diags={};ii,jj=np.triu_indices(65,1)
 previous={r['alias']:r for r in read(OP/'RESULTS_PUBLIC.json')['rows']}
 for rec in plan['records']:
  check();a=rec['alias'];z=load(D/'predictions'/f'{a}.pt');old=load(OLD/'predictions'/f'{a}.pt');assert np.array_equal(z['q0'],old['q0'])and z['frame_ids']==old['frame_ids'];ev=z['evidence'];old_ev=old['evidence']
  for k in ['logD','logB','mapping','support']:
   if isinstance(ev[k],list):assert all(np.array_equal(u,v)for u,v in zip(ev[k],old_ev[k]))
   else:assert np.array_equal(ev[k],old_ev[k])
  assert set(z['outputs'])==set(ARMS);diags[a]={}
  for name,out in z['outputs'].items():
   q=out['q'];assert np.array_equal(q>0,z['q0']>0)and abs(q.sum()-1)<1e-10;v=final_mbr(z['frame_ids'],q);assert v['interval']==out['interval']
   choices=np.unique(np.r_[np.linspace(0,2079,9).astype(int),np.flatnonzero((ii==v['indices'][0])&(jj==v['indices'][1]))]);e=float(abs(brute_utility(z['frame_ids'],q,choices)-v['utility_grid'][ii[choices],jj[choices]]).max());assert e<1e-11;maxutility=max(maxutility,e);mbrchecks+=len(choices)
   if name in FITS:
    assert out['mode']=='Live'and out['actual_updates']==40 and out['kl_coefficient']==COEFFICIENTS[name.split('Live')[0]]
    vv=replay(z['q0'],ev['logD'],ev['logB'],ev['mapping'],ev['support'],out);diags[a][name]=vv;steps+=vv['steps'];maxerr=max(maxerr,vv['maxerror'])
    ref=old['outputs']['Live'+str(out['seed'])]['trace'][0]
    for key in ['pairs','uniforms']:assert np.array_equal(out['trace'][0][key],ref[key])
    assert float(abs(out['trace'][0]['gradient']-ref['gradient']).max())<1e-14
    assert float(abs(out['trace'][0]['eta_after']-ref['eta_after']).max())<1e-8
    first+=1
   else:
    src=old['outputs']['Strong'if name=='Strong'else 'Live'+name.split('Live')[1]];assert np.array_equal(q,src['q'])and out['interval']==src['interval'];baselinechecks+=1
  for key,b in z['trajectories'].items():assert np.array_equal(b,old['trajectories'][key]);spacechecks+=1
  print('KL_pre_GT_audit',a,flush=True)
 assert steps==cost['actual_updates']==30720 and first==768 and spacechecks==512 and baselinechecks==512
 audit=dict(status='pass',independent_numpy_weighted_gradient_Adam_steps=steps,maxerror=maxerr,source_teacher_evidence_and_mapping_byte_exact=True,
 original_source_evidence_validation='reused prior independently validated sealed native-logit evidence; not re-encoded',
 first_step_matches_saved_coefficient1_with_declared_tolerance=first,spatial_arrays_exact=spacechecks,reused_Strong_and_coefficient1_posteriors_exact=baselinechecks,
 independent_MBR_utilities=mbrchecks,max_direct_utility_error=maxutility,pre_GT_audit_new_model_or_GT=False)
 put(D/'POLICY_AUDIT_BEFORE_GT.json',audit);put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')));phase['GT']=True
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in plan['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];rows=[];officialchecks=oldmetrics=0
 for rec in plan['records']:
  check();a=rec['alias'];z=load(D/'predictions'/f'{a}.pt');values={};actual={}
  for space,b in z['trajectories'].items():
   tube=DenseTube(b,raw[rec['query_ordinal']],gt[rec['query_ordinal']],spans[rec['query_ordinal']],clip=True)
   sel=np.isin(tube.fids,np.array(sorted(set(z['frame_ids'])-set(rec['physical_frame_ids8']))));actual[space]=dict(nodes=int(sel.sum()),sIoU=float(tube.iou[sel].mean())if sel.any()else None,all_temporal_arms_identical=True);assert actual[space]==previous[a]['fixed_unobserved_actual_sIoU'][space]
   for name,out in z['outputs'].items():
    iv=out['interval'];v=tube.score(iv);off=official(b,raw[rec['query_ordinal']],gt[rec['query_ordinal']],spans[rec['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;officialchecks+=1
    v=dict(**v,boundary_MAE=(abs(iv[0]-spans[rec['query_ordinal']][0])+abs(iv[1]-spans[rec['query_ordinal']][1]))/2,interval=iv)
    if name=='Strong'or name.startswith('KL1Live'):
     src='Strong'if name=='Strong'else 'Live'+name.split('Live')[1];assert all(abs(v[k]-previous[a]['values'][space+'_'+src][k])<2e-10 for k in ['v','t','s','boundary_MAE']);oldmetrics+=1
    values[space+'_'+name]=v
  values['Mean_Strong']={k:avg([values[f'Live{s}_Strong'][k]for s in SEEDS])for k in ['v','t','s','boundary_MAE']}
  for name in ALL:
   values['Mean_'+name]={k:avg([values[f'Live{s}_{name}Live{s}'][k]for s in SEEDS])for k in ['v','t','s','boundary_MAE']}
   values['EMean_'+name]={k:avg([values[f'E_{name}Live{s}'][k]for s in SEEDS])for k in ['v','t','s','boundary_MAE']}
  assert all(abs(values['Mean_KL1'][k]-previous[a]['values']['Mean_Live'][k])<2e-10 for k in ['v','t','s','boundary_MAE'])
  rows.append(dict(alias=a,values=values,policy_updates=diags[a],fixed_unobserved_actual_sIoU=actual));print('scored_KL_strength',a,flush=True)
 means={n:{k:avg([r['values']['Mean_'+n][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for n in ['Strong']+ALL}
 contrasts=[('Mean_'+n,'Mean_Strong')for n in ALL]+[('Mean_'+n,'Mean_KL1')for n in COEFFICIENTS]+[('Mean_KL0p1','Mean_KL0p3')]+[(f'Live{s}_{n}Live{s}',f'Live{s}_KL1Live{s}')for n in COEFFICIENTS for s in SEEDS]
 stats={};aliases=[r['alias']for r in rows]
 for a,b in contrasts:
  v={k:paired([(100 if k!='boundary_MAE'else 1)*(r['values'][a][k]-r['values'][b][k])for r in rows])for k in ['v','t','s','boundary_MAE']}
  delta=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]);v['tails']=dict(**tails(delta,aliases),positive_contribution_pp=float(np.maximum(delta,0).mean()),negative_contribution_pp=float(np.minimum(delta,0).mean()));stats[a+'_minus_'+b]=v
 process={n:{k:avg([diags[a][n+'Live'+str(s)][k]for a in diags for s in SEEDS])for k in ['admission_fraction','unique_sampled_intervals','unique_admitted_intervals','initial_preference_KL','final_preupdate_preference_KL','final_preupdate_global_KL','final_preupdate_weighted_global_KL','eta_norm','posterior_L1']}for n in COEFFICIENTS}
 prior_process=read(OP/'RESULTS_PUBLIC.json')['process']['Live'];process['KL1']={k:prior_process[k]for k in process['KL0p1']if k in prior_process};process['KL1']['final_preupdate_weighted_global_KL']=prior_process['final_preupdate_global_KL']
 deltas={n:np.asarray([100*(r['values']['Mean_'+n]['v']-r['values']['Mean_Strong']['v'])for r in rows])for n in ALL};base=deltas['KL1'];decomp={}
 for n in COEFFICIENTS:
  x=deltas[n];groups={}
  for group,mask in [('coefficient1_positive',base>1e-10),('coefficient1_negative',base< -1e-10),('coefficient1_flat',abs(base)<=1e-10)]:
   groups[group]=dict(parents=int(mask.sum()),coefficient1_contribution_to_all128_pp=float(base[mask].sum()/128),new_contribution_to_all128_pp=float(x[mask].sum()/128),
    new_positive=int((x[mask]>1e-10).sum()),new_negative=int((x[mask]< -1e-10).sum()),new_flat=int((abs(x[mask])<=1e-10).sum()))
  decomp[n]=dict(groups=groups,positive_contribution_increase_pp=float((np.maximum(x,0)-np.maximum(base,0)).mean()),negative_contribution_change_pp=float((np.minimum(x,0)-np.minimum(base,0)).mean()),
   net_change_pp=float((x-base).mean()),posthoc_GT_explanation_only=True)
 result=dict(status='completed_full128_finite_KL_strength_development',parents=128,seeds=SEEDS,coefficients={'KL0p1':.1,'KL0p3':.3,'KL1':1.},means=means,paired=stats,process=process,contribution_decomposition=decomp,rows=rows,
 new_fits=768,new_updates=30720,reused_coefficient1=True,spatial_frozen=True,not_independent_confirmation=True,new_on_policy_controls_run=False)
 put(P/'RESULTS_PUBLIC.json',result)
 best=max(['KL1','KL0p3','KL0p1'],key=lambda n:means[n]['v']);net={n:stats['Mean_'+n+'_minus_Mean_Strong']['v']for n in ALL}
 decision=dict(development_selection_only=best,selection_rule=plan['selection'],method_net_over_Strong=net,
 selected_point_gain_at_least2pp=net[best]['mean']>=2,selected_gain_CI_above_zero=net[best]['CI95'][0]>0,
 no_new_independent_on_policy_contribution_claim=True,new_Direct_or_Fixed_controls=False,no_promotion=True,
 no_automatic_next_experiment=True,no_coefficients_beyond_predeclared_three=True,not_confirmation=True,
 scope='only protection-strength range0.1/0.3/1, all other algorithm choices fixed; contribution expansion examined, not universal OPD verdict')
 put(P/'DECISION_PUBLIC.json',decision)
 audit.update(new_target_GT_scored=True,official_DenseTube_checks=officialchecks,previous_baseline_metrics_exact=oldmetrics,GT_provenance=prov,
 all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']));assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit)
 fullcost=dict(**cost,CPU_audit_and_scoring_seconds=time.perf_counter()-beg,prior_coefficient1_experiment_cost=read(OLD/'COST_REVISION001.json'),
 original_source_cache_generation_cost=read(R/'artifacts/stvg_temporal_local_motion_completion128_v1/COST.json'),publication_archive_excluded=True);put(P/'COST_PUBLIC.json',fullcost)
 lines=['# Finite Strong KL protection-strength verification: full HC128','','Only the continuous global Strong KL coefficient changes. Native offset evidence, binary agreement target, learner initialization/state, sampling/seed streams, optimizer40Adam0.1, spatial arrays and final MBR are frozen. New actual Live fits at0.1/0.3 for128parents x3seeds;1 is reused from the completed previous experiment. This is development sensitivity/selection, not confirmation. No new Direct/Fixed experiment and no claim of independent on-policy improvement at newly selected strength.','','| Coefficient | complete vIoU percent | tIoU percent | full-v delta vs Strong pp [CI95] | positive contribution pp | negative contribution pp | below-5pp parents |','|---|---:|---:|---|---:|---:|---:|',f"| Strong | {means['Strong']['v']*100:.6f} | {means['Strong']['t']*100:.6f} | reference | - | - | - |"]
 for n in ['KL1','KL0p3','KL0p1']:
  v=means[n];st=stats['Mean_'+n+'_minus_Mean_Strong'];t=st['tails'];lines.append(f"| {result['coefficients'][n]} | {100*v['v']:.6f} | {100*v['t']:.6f} | {st['v']['mean']:+.6f} {st['v']['CI95']} | {t['positive_contribution_pp']:+.6f} | {t['negative_contribution_pp']:+.6f} | {t['below_minus5']} |")
 lines+=['','| Comparison | complete-v pp [CI95] | tIoU pp [CI95] |','|---|---|---|']
 for n in COEFFICIENTS:
  st=stats['Mean_'+n+'_minus_Mean_KL1'];lines.append(f"| {n} minus KL1 | {st['v']['mean']:+.6f} {st['v']['CI95']} | {st['t']['mean']:+.6f} {st['t']['CI95']} |")
 lines+=['','All128 parents retained, including no-change/time-no-overlap cases. Match temporal/spatial seeds then mean within parent; bootstrap parents10000times, not384 independent videos. Fixed-unobserved actual-node sIoU and all512 spatial arrays remain exact. Posthoc sign groups never choose output/feedback/parameters online. Development selection maximizes complete-v only, exact ties choose stronger protection; this does not promote a method or authorize more coefficients/confirmation.','','Process and contribution expansion:','```json',json.dumps(process,indent=2),json.dumps(decomp,indent=2),json.dumps(decision,indent=2),json.dumps(fullcost,indent=2),json.dumps(audit,indent=2),'```','','All new predictions/traces sealed before new GT; independent audit verifies every weighted gradient, sampled pair/label, Adam update and MBR decision. Raw0.1/0.3 signed outputs, all seeds, negative tails, worst cases and leave-one-out retained. Zero new experts/models/checkpoints/RGB/visual forwards/GPU/neural weight training or spatial updates. CPU-only cached adaptation/scoring time excludes existing source/expert cache generation and is not end-to-end deployment latency. Both offsets remain correlated and B_o not a marginal of global Strong. No follow-up search or registry promotion.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=plan['pins'],reused_coefficient1_commit='b65d1d6250d23fa71b3a38c19c5ce1ca0c031f3c',exposure='RC128 repeatedly exposed development',excluded=['media','captions','parentIDs','GT coordinates','weights','raw policy traces and features']))
 assert all(sha(R/f)==h for f,h in plan['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files']);check();print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':
 def timeout(signum,frame):raise TimeoutError('finite900s KL strength audit and score')
 signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,timeout);signal.alarm(900);lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:main()
 except BaseException as e:put(D/f'SCORING_FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e)));raise
 finally:lock.close()
