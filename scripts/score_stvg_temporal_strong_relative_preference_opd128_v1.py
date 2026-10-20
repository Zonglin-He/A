"""Independent NumPy reconstruction before GT, official sealed-output evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,fcntl,signal,numpy as np
from scipy.special import logsumexp,expit,xlogy
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_strong_relative_preference_opd128_v1 import D,P,OLD,OP,STAGE,HR,SEEDS,MODES,STATIC,FITS,forbid_GT
from scripts.score_stvg_where_when_event128_v1 import avg,paired
from scripts.score_stvg_temporal_local_global_mix128_v1 import tails


def independent_evidence(old,stage,hr):
 def joints(logits):
  out=[]
  for rec,z in zip(old['records'],logits):
   a=np.asarray(z,dtype=np.float64)[0];i,j=np.triu_indices(len(a),1);v=a[i,0]+a[j,1];out.append(v-logsumexp(v))
  return out
 dd=joints(old['logits']['DenseLocal']);hh=joints(hr['observation']['logits']);ll=[joints([v[k]for v in stage['pass_logits'][1]])for k in range(6)]
 bb=[logsumexp(np.stack([hh[o]-np.log(2)]+[l[o]-np.log(12)for l in ll]),axis=0)for o in range(2)]
 support=np.argwhere(old['outputs']['HiResAuxMix']['q']>0);ids=old['frame_ids'];maps=[]
 intervals=np.array([[ids[i],ids[j]+1]for i,j in support])
 for rec in old['records']:
  f=rec['frame_ids'];native=np.array([[f[i],f[j]+1]for i in range(len(f))for j in range(i+1,len(f))]);out=[]
  for s,e in intervals:
   overlap=np.maximum(0,np.minimum(e,native[:,1])-np.maximum(s,native[:,0]));union=np.maximum(e,native[:,1])-np.minimum(s,native[:,0])
   # Disjoint intervals have zero overlap; union formula is immaterial to zero-IoU ties.
   out.append(np.argmax(overlap/union))
  maps.append(out)
 return dd,bb,np.asarray(maps),support


def independent_feedback(dd,bb,maps,pairs):
 a,b=pairs.T;terms=[];coll=[]
 for o in range(2):
  x,y=maps[o,a],maps[o,b];h=dd[o][x]-dd[o][y];terms.extend([h,h-(bb[o][x]-bb[o][y])]);coll.append(x==y)
 v=np.stack(terms,axis=-1)[:,[0,2,1,3]];pos=(v>0).all(1);neg=(v<0).all(1)
 m=np.where(pos,np.min(v,axis=1),np.where(neg,np.max(v,axis=1),0.))
 return dict(terms=v,t=expit(m),w=(pos|neg).astype(float),m=m,
 mapped=np.stack([maps[:,a],maps[:,b]],axis=-1).transpose(1,0,2),collision=np.stack(coll,axis=1))


def replay(Bgrid,dd,bb,maps,support,out):
 B=Bgrid[tuple(support.T)];r=np.zeros(len(B));m1=r.copy();m2=r.copy();rng=np.random.default_rng(out['seed']);u=rng.random((40,32,2));cdf=np.cumsum(B);cdf[-1]=1.
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
  g=p*(r-p@r);np.add.at(g,pairs[:,0],a);np.add.at(g,pairs[:,1],-a)
  pk=float(np.mean(w*(np.logaddexp(0,v)-t*v+xlogy(t,t)+xlogy(1-t,1-t))))
  kk=float(p@r-logsumexp(np.log(B)+r)+logsumexp(np.log(B)))
  close(r,tr['eta_before']);close(p,tr['pi_before']);close(g,tr['gradient']);close(pk,tr['preference_KL']);close(kk,tr['global_KL']);close(pk+kk,tr['loss'])
  m1=.9*m1+.1*g;m2=.999*m2+.001*g*g;r-=.1*(m1/(1-.9**(k+1)))/(np.sqrt(m2/(1-.999**(k+1)))+1e-8);r-=r.mean()
  close(m1,tr['adam_first']);close(m2,tr['adam_second']);close(r,tr['eta_after'])
  admitted+=int(w.sum());total+=len(w);collisions+=int(fb['collision'].any(1).sum());seen.update(pairs.ravel().tolist());accepted.update(pairs[w>0].ravel().tolist());pref.append(pk);globalKL.append(kk)
 close(r,out['eta']);expected=np.zeros_like(Bgrid);expected[tuple(support.T)]=B if not r.any()else np.exp(np.log(B)+r-logsumexp(np.log(B)+r));close(expected,out['q'])
 return dict(steps=len(out['trace']),maxerror=maximum,Live_batches_different_from_initial=changed,admitted=admitted,pairs_evaluated=total,
  admission_fraction=admitted/total,unique_sampled_intervals=len(seen),unique_admitted_intervals=len(accepted),native_collision_pairs=collisions,
  initial_preference_KL=pref[0],final_preupdate_preference_KL=pref[-1],initial_global_KL=globalKL[0],final_preupdate_global_KL=globalKL[-1],
  eta_norm=float(np.linalg.norm(r)),posterior_L1=float(abs(out['q']-Bgrid).sum()))


def main():
 import torch
 from main_method.runtime import resource_guard
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 torch.set_num_threads(4);beg=time.perf_counter();phase={'GT':False}
 def guard(event,args):
  if not phase['GT']:forbid_GT(event,args)
 sys.addaudithook(guard);p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
 def check():resource_guard();assert time.perf_counter()-beg<880
 assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')
 assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files'])
 cost=read(D/'COST.json');assert cost['success'];diags={};steps=spacechecks=first=mbrchecks=0;maxerr=maxutility=0.;ii,jj=np.triu_indices(65,1)
 for rec in p['records']:
  check();a=rec['alias'];z=load(D/'predictions'/f'{a}.pt');old=load(OLD/'predictions'/f'{a}.pt');st=load(STAGE/'predictions'/f'{a}.pt');hr=load(HR/'predictions'/f'{a}.pt')
  dd,bb,maps,support=independent_evidence(old,st,hr);assert np.array_equal(support,z['evidence']['support'])and np.array_equal(maps,z['evidence']['mapping'])
  for o in range(2):assert np.max(abs(dd[o]-z['evidence']['logD'][o]))<1e-12 and np.max(abs(bb[o]-z['evidence']['logB'][o]))<1e-12
  assert np.array_equal(z['q0'],old['outputs']['HiResAuxMix']['q']);diags[a]={}
  for name,out in z['outputs'].items():
   q=out['q'];assert np.array_equal(q>0,z['q0']>0)and abs(q.sum()-1)<1e-10
   v=final_mbr(z['frame_ids'],q);assert v['interval']==out['interval']
   choices=np.unique(np.r_[np.linspace(0,2079,9).astype(int),np.flatnonzero((ii==v['indices'][0])&(jj==v['indices'][1]))]);e=float(abs(brute_utility(z['frame_ids'],q,choices)-v['utility_grid'][ii[choices],jj[choices]]).max());assert e<1e-11;maxutility=max(maxutility,e);mbrchecks+=len(choices)
   if name in FITS:
    v=replay(z['q0'],dd,bb,maps,support,out);steps+=v['steps'];maxerr=max(maxerr,v['maxerror']);diags[a][name]=v
  for s in SEEDS:
   f,l=z['outputs'][f'Fixed{s}'],z['outputs'][f'Live{s}']
   for key in ['pairs','gradient','eta_after','uniforms','loss']:assert np.array_equal(f['trace'][0][key],l['trace'][0][key])
   first+=1
  for key,b in z['trajectories'].items():assert np.array_equal(b,old['trajectories'][key]);spacechecks+=1
  print('preference_pre_GT_audit',a,flush=True)
 assert steps==cost['actual_updates']==46080 and first==384 and spacechecks==512
 audit=dict(status='pass',independent_numpy_steps=steps,maxerror=maxerr,independent_native_logp_and_physical_mapping=True,first_Fixed_Live_exact=first,
 spatial_arrays_exact=spacechecks,independent_MBR_utilities=mbrchecks,max_direct_utility_error=maxutility,pre_GT_audit_new_model_or_GT=False)
 put(D/'POLICY_AUDIT_BEFORE_GT.json',audit);put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')));phase['GT']=True
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows']
 previous={r['alias']:r for r in read(OP/'RESULTS_PUBLIC.json')['rows']};whole={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-20/temporal_denselocal_opd128/RESULTS_PUBLIC.json')['rows']}
 rows=[];officialchecks=oldmetrics=0
 for rec in p['records']:
  check();a=rec['alias'];z=load(D/'predictions'/f'{a}.pt');values={};actual={}
  for space,b in z['trajectories'].items():
   tube=DenseTube(b,raw[rec['query_ordinal']],gt[rec['query_ordinal']],spans[rec['query_ordinal']],clip=True)
   sel=np.isin(tube.fids,np.array(sorted(set(z['frame_ids'])-set(rec['physical_frame_ids8']))));actual[space]=dict(nodes=int(sel.sum()),sIoU=float(tube.iou[sel].mean())if sel.any()else None,all_temporal_arms_identical=True)
   assert actual[space]==previous[a]['fixed_unobserved_actual_sIoU'][space]
   for name,o in z['outputs'].items():
    iv=o['interval'];v=tube.score(iv);off=official(b,raw[rec['query_ordinal']],gt[rec['query_ordinal']],spans[rec['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;officialchecks+=1
    v=dict(**v,boundary_MAE=(abs(iv[0]-spans[rec['query_ordinal']][0])+abs(iv[1]-spans[rec['query_ordinal']][1]))/2,interval=iv)
    if name in STATIC:
     src={'Native':'Native','Strong':'HiResAuxMix','TeacherDirect':'DenseLocal','DenseLocalMix':'DenseLocalMix'}[name];assert all(abs(v[k]-previous[a]['values'][space+'_'+src][k])<2e-10 for k in ['v','t','s','boundary_MAE']);oldmetrics+=1
    values[space+'_'+name]=v
  for name in STATIC:
   values['Mean_'+name]={k:avg([values[f'Live{s}_{name}'][k]for s in SEEDS])for k in ['v','t','s','boundary_MAE']}
  for mode in MODES:
   values['Mean_'+mode]={k:avg([values[f'Live{s}_{mode}{s}'][k]for s in SEEDS])for k in ['v','t','s','boundary_MAE']}
   values['EMean_'+mode]={k:avg([values[f'E_{mode}{s}'][k]for s in SEEDS])for k in ['v','t','s','boundary_MAE']}
  values['Mean_WholeDenseLive']=whole[a]['values']['Mean_Live']
  rows.append(dict(alias=a,values=values,policy_updates=diags[a],fixed_unobserved_actual_sIoU=actual));print('scored_preference',a,flush=True)
 names=STATIC+MODES+['WholeDenseLive'];means={n:{k:avg([r['values']['Mean_'+n][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for n in names}
 contrasts=[('Mean_'+m,'Mean_Strong')for m in MODES]+[('Mean_Live','Mean_'+b)for b in ['Direct','Fixed','Native','TeacherDirect','WholeDenseLive']]+[(f'Live{s}_Live{s}',f'Live{s}_Fixed{s}')for s in SEEDS]
 stats={};aliases=[r['alias']for r in rows]
 for aa,bb in contrasts:
  v={k:paired([(100 if k!='boundary_MAE'else 1)*(r['values'][aa][k]-r['values'][bb][k])for r in rows])for k in ['v','t','s','boundary_MAE']};delta=np.array([100*(r['values'][aa]['v']-r['values'][bb]['v'])for r in rows]);v['tails']=dict(**tails(delta,aliases),positive_contribution_pp=float(np.maximum(delta,0).mean()),negative_contribution_pp=float(np.minimum(delta,0).mean()));stats[aa+'_minus_'+bb]=v
 process={m:{k:avg([diags[a][m+str(s)][k]for a in diags for s in SEEDS])for k in ['admission_fraction','unique_sampled_intervals','unique_admitted_intervals','initial_preference_KL','final_preupdate_preference_KL','initial_global_KL','final_preupdate_global_KL','eta_norm','posterior_L1']}for m in MODES}
 res=dict(status='completed_actual_selective_preference_OPD_exposed_development',parents=128,seeds=SEEDS,means=means,paired=stats,rows=rows,process=process,spatial_frozen=True,new_network_training=0)
 put(P/'RESULTS_PUBLIC.json',res)
 audit.update(new_target_GT_scored=True,official_DenseTube_checks=officialchecks,previous_static_metrics_exact=oldmetrics,GT_provenance=prov,
 all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']))
 assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit)
 net={m:stats['Mean_'+m+'_minus_Mean_Strong']['v']for m in MODES};controls={m:stats['Mean_Live_minus_Mean_'+m]['v']for m in ['Direct','Fixed']}
 decision=dict(method_net_over_Strong=net,Live_independent_value=controls,method_point_gain={m:v['mean']>0 for m,v in net.items()},method_CI_above_zero={m:v['CI95'][0]>0 for m,v in net.items()},
 method_two_pp_goal={m:v['mean']>=2 for m,v in net.items()},on_policy_CI_above_both_controls=all(v['CI95'][0]>0 for v in controls.values()),
 actual_OPD_executed=True,not_confirmation=True,no_parameter_search=True,no_automatic_next_experiment=True,no_registry_promotion=True,
 scope='one binary-weight cross-offset preference and continuing Strong KL configuration; method gain is distinct from on-policy gain')
 put(P/'DECISION_PUBLIC.json',decision);fullcost=dict(**cost,CPU_audit_and_scoring_seconds=time.perf_counter()-beg,upstream_cache_cost=read(OLD/'COST.json'),publication_and_archive_excluded=True);put(P/'COST_PUBLIC.json',fullcost)
 lines=['# Strong-relative DenseLocal preference OPD: actual HC128 experiment','','One user-revised configuration: strict four-sign agreement, t=sigmoid(weakest signed log preference), binary admitted w=1; otherwise abstain. Joint interval residual r with continuing KL(pi||Strong), coefficient1;40 Adam0.1 steps,32 pairs per stochastic step, seeds41/42/43. No new models, visual observations, experts or neural training. Local B_o is a reference recipe, NOT the marginal of global Strong. Both offsets are correlated.','','| Output | full vIoU percent | tIoU percent | boundary MAE frames |','|---|---:|---:|---:|']
 for n,v in means.items():lines.append(f"| {n} | {100*v['v']:.6f} | {100*v['t']:.6f} | {v['boundary_MAE']:.6f} |")
 lines+=['','| Comparison | full-v pp [CI95] | tIoU pp [CI95] | losses below-5pp |','|---|---|---|---:|']
 for k,v in stats.items():lines.append(f"| {k} | {v['v']['mean']:+.6f} {v['v']['CI95']} | {v['t']['mean']:+.6f} {v['t']['CI95']} | {v['tails']['below_minus5']} |")
 lines+=['','Method net gain and independent on-policy contribution are assessed separately. No Direct/Fixed gain is erased because Live does not beat it. Goal +2–3pp is relative to Strong, not Native; no independent confirmation or universal OPD verdict. All128 parents retained; temporal/spatial seeds paired then averaged within each parent. Repeatedly exposed development, not384 independent examples. Spatial arrays and actual unobserved-node sIoU identical across temporal arms.','','Fixed draws fresh BxB pairs per step; Live draws I from current student and J from B using paired uniforms. Direct uses the same1280-pair Fixed bank every step and has larger pair arithmetic; it is not compute-FLOP matched. All outputs use unchanged2080-action MBR. Only B-positive support adapted; no epsilon support. Teacher targets, mappings, sampling and candidate coordinates detached.','','Process (mean per fit; Direct revisits full bank each step):','```json',json.dumps(process,indent=2),json.dumps(decision,indent=2),json.dumps(fullcost,indent=2),json.dumps(audit,indent=2),'```','','Predictions and complete actual samples/feedback/gradients/Adam chains sealed before GT, then independent NumPy reconstruction of logits, physical mapping, strict signs, losses and every update; official task metrics cross-checked. New CPU timing excludes upstream cache generation, spatial and Source inference and is not deployment latency. Complete signed query results, negative tails and all three seeds retained. No scan, automatic next round, or CURRENT/MAIN/PAPER promotion.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],spatial_config_sha256=sha(R/'methods/stvg_referdino_opd_v1/config.json'),exposure='RC128 repeatedly exposed development',excluded=['media','captions','parentIDs','GT coordinates','weights','raw traces and features']))
 assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files']);check();print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':
 def timeout(signum,frame):raise TimeoutError('finite900s preference OPD audit/score')
 signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,timeout);signal.alarm(900)
 lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:main()
 except BaseException as e:put(D/f'SCORING_FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e)));raise
 finally:lock.close()
