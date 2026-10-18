"""Postseal CPU scoring of final-interval decisions and budget allocation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_posterior_decision_tree_v1 import D,P,ARMS,LAYOUTS

def avg(xs):
 a=[float(x)for x in xs if x is not None];return float(np.mean(a))if a else None

def paired(rows,fn):
 if not rows:return dict(mean=None,CI95=None,parents=0)
 # one query per parent in this fixed cohort; parent-cluster resampling
 v=np.array([fn(r)for r in rows],float);idx=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)))
 return dict(mean=float(v.mean()),CI95=np.quantile(v[idx].mean(1),[.025,.975]).tolist(),parents=len(v),positive=int((v>1e-12).sum()),negative=int((v< -1e-12).sum()),neutral=int((abs(v)<=1e-12).sum()),per_parent=v.tolist())

def temporal_summary(rows,geo):
 vals={a:{k:avg([r['temporal'][geo][a][k]for r in rows])for k in rows[0]['temporal'][geo][a]}for a in ARMS}
 pairs={f'final_MBR_minus_{a}':paired(rows,lambda r,a=a:100*(r['temporal'][geo]['final_MBR']['v']-r['temporal'][geo][a]['v']))for a in ARMS if a!='final_MBR'}
 strata={}
 for key in ['event_length_bin','event_start_bin','motion_bin','target_area_bin','native_overlap_group']:
  strata[key]={}
  for v in sorted({r[key]for r in rows}):
   rr=[r for r in rows if r[key]==v];strata[key][v]=dict(queries=len(rr),parents=len(rr),metrics={a:{m:avg([r['temporal'][geo][a][m]for r in rr])for m in ['v','t','zero_intersection']}for a in ARMS},paired_vs_Native=paired(rr,lambda r:100*(r['temporal'][geo]['final_MBR']['v']-r['temporal'][geo]['Native']['v'])),paired_vs_projection=paired(rr,lambda r:100*(r['temporal'][geo]['final_MBR']['v']-r['temporal'][geo]['projection']['v'])))
 return dict(queries=len(rows),parents=len(rows),metrics=vals,paired=pairs,strata=strata)

def coverage_summary(rows):
 vals={a:{k:avg([r['coverage'][a][k]for r in rows])for k in rows[0]['coverage'][a]}for a in LAYOUTS}
 pairs={}
 for a in ['N','U']:
  pairs['P_minus_'+a]=paired(rows,lambda r,a=a:r['coverage']['P']['score']-r['coverage'][a]['score'])
  pairs['P_zero_minus_'+a]=paired(rows,lambda r,a=a:r['coverage']['P']['zero_hit']-r['coverage'][a]['zero_hit'])
 strata={}
 for key in ['event_length_bin','event_start_bin','motion_bin','target_area_bin','native_overlap_group']:
  strata[key]={}
  for v in sorted({r[key]for r in rows}):
   rr=[r for r in rows if r[key]==v];strata[key][v]=dict(queries=len(rr),parents=len(rr),metrics={a:{m:avg([r['coverage'][a][m]for r in rr])for m in ['score','hit1','hit2','hits','zero_hit','within_event_spread_fraction']}for a in LAYOUTS},paired_vs_N=paired(rr,lambda r:r['coverage']['P']['score']-r['coverage']['N']['score']),paired_vs_U=paired(rr,lambda r:r['coverage']['P']['score']-r['coverage']['U']['score']))
 return dict(queries=len(rows),parents=len(rows),metrics=vals,paired=pairs,strata=strata)

def risk_groups(rows,kind):
 out={}
 for name,key,value in [('short','event_length_bin','Q1'),('fast','motion_bin','Q4'),('small','target_area_bin','Q1')]:
  rr=[r for r in rows if r[key]==value];pairs={}
  for control in (['Native','projection']if kind=='T0'else['N','U']):
   pairs[control]=paired(rr,lambda r,c=control:100*(r['temporal']['After']['final_MBR']['v']-r['temporal']['After'][c]['v'])if kind=='T0'else r['coverage']['P']['score']-r['coverage'][c]['score'])
  enough=len(rr)>=8;boundary=-1. if kind=='T0'else-.05
  risk=bool(enough and any(v['mean']<boundary for v in pairs.values()))
  out[name]=dict(parents=len(rr),status='UNKNOWN' if insufficient(enough) else'RISK'if risk else'PASS',comparisons=pairs)
 return out

def insufficient(x):return not x

def decisions(rows,phase):
 temporal={};space={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];tr=[r for r in rr if 'After'in r['temporal']]
  t=temporal_summary(tr,'After');p=t['paired'];risk=risk_groups(tr,'T0');goodmeans=p['final_MBR_minus_Native']['mean']>=.5 and p['final_MBR_minus_projection']['mean']>=-.5
  dominated=any(p['final_MBR_minus_'+a]['CI95'][1]<0 for a in ['marginal_MAP','per_offset_MBR']);samerule=all(r['decision_equalities']['final_equals_old_marginal']for r in tr)or all(r['decision_equalities']['final_equals_old_offset_MBR']for r in tr)
  passed=goodmeans and not dominated and not samerule and all(v['status']=='PASS'for v in risk.values())
  temporal[ds]=dict(mean_gate=goodmeans,dominated_by_old_actionness_free=dominated,identical_all_decisions_to_old=samerule,subgroups=risk,passed=passed,primary_queries=len(tr),extension_After_unavailable=True)
  cs=coverage_summary(rr);cp=cs['paired'];riskS=risk_groups(rr,'S0');meanOK=cp['P_minus_N']['mean']>=.05 and cp['P_minus_U']['mean']>0;zeroOK=all(cp['P_zero_minus_'+a]['mean']<=.05 for a in ['N','U']);passS=meanOK and zeroOK and all(v['status']=='PASS'for v in riskS.values());sameruleS=all(r['coverage']['P']['score']==r['coverage']['U']['score']for r in rr)
  uncertain=all(cp['P_minus_'+a]['CI95'][1]>=threshold for a,threshold in [('N',.05),('U',0.)])and any(cp['P_minus_'+a]['CI95'][0]<threshold<cp['P_minus_'+a]['CI95'][1]for a,threshold in [('N',.05),('U',0.)])
  space[ds]=dict(mean_gate=meanOK,zero_hit_risk_pass=zeroOK,subgroups=riskS,passed=passS,identical_coverage_all_to_uniform=sameruleS,uncertain_at_practical_threshold=uncertain,queries=len(rr))
 Tpassed=all(t['passed']for t in temporal.values());Spassed=all(s['passed']for s in space.values())
 # Exactly one expansion only if no clear domain failure; never fish after a clear practical no-go.
 need=phase=='initial'and not Spassed and all((s['mean_gate']and s['zero_hit_risk_pass'])or s['uncertain_at_practical_threshold']for s in space.values())and not any(s['identical_coverage_all_to_uniform']for s in space.values())and not any(v['status']=='RISK'for s in space.values()for v in s['subgroups'].values())
 return dict(status='development_screen_complete',phase=phase,T0=temporal,S0=space,T0_passed=Tpassed,S0_passed=Spassed,S0_extension_required=need,S1_authorized_to_dispatch=Spassed,J0_time_candidate_authorized=Tpassed,next_action='one_fixed_S0_extension'if need else'S1_fixed_expert_test'if Spassed else'time_confirmation_only'if Tpassed else'stop_this_distribution_decision_route',no_fullset_launch=True,no_GT_routing=True)

def run(extension=False):
 import torch
 torch.set_num_threads(4)
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_READOUT_BARRIER.json');assert bar['count']==64 and not bar['GT_read'];files=list(bar['files'])
 if extension:
  e=read(D/'EXTENSION_READOUT_BARRIER.json');assert e['count']==64 and not e['GT_read'];files+=e['files']
 for f in files:assert sha(ROOT/f['path'])==f['sha256']
 zs=[load(ROOT/f['path'])for f in files];assert len({(z['dataset'],z['query_ordinal'])for z in zs})==len(zs)
 features=ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json';assert sha(features)==read(D/'RUNTIME_LOCK.json')['strata_sha256'];fby={ds:{r['query_ordinal']:r for r in rr}for ds,rr in read(features)['rows'].items()};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in ['hc2','vidstg']};targets={ds:truths(ds,dict(parents=[z['query_ordinal']for z in zs if z['dataset']==ds]))for ds in plans};rows=[];checks=0
 for z in zs:
  ds,q=z['dataset'],z['query_ordinal'];row=plans[ds][q];gt=targets[ds][0][q];span=targets[ds][1][q];physical_span=[span[0],span[1]+(1 if ds=='hc2'else 0)];feat=fby[ds][q];assert physical_span[1]-physical_span[0]==feat['event_physical_frames'];ids=np.array(z['frame_ids']);timevalues={}
  for geo,boxes in z['boxes'].items():
   tube=DenseTube(boxes.numpy(),row,gt,span,clip=ds=='hc2');values={};niv=z['intervals']['Native'];old=(tube.fids>=niv[0])&(tube.fids<niv[1])
   for arm,iv in z['intervals'].items():
    v=official(boxes.numpy(),row,gt,span,iv,ds);ev=tube.score(iv);assert all(abs(v[k]-ev[k])<2e-10 for k in ['v','t','s']);checks+=3;now=(tube.fids>=iv[0])&(tube.fids<iv[1]);add=now&~old;lost=old&~now;overlap=max(0,min(iv[1],niv[1])-max(iv[0],niv[0]));added=iv[1]-iv[0]-overlap;removed=niv[1]-niv[0]-overlap
    v.update(R03=float(v['v']>.3),R05=float(v['v']>.5),zero_intersection=float(v['t']==0),length_input_fraction=(iv[1]-iv[0])/(ids[-1]+1-ids[0]),added_known_frames=int(add.sum()),lost_known_frames=int(lost.sum()),added_unknown_frames=int(added-add.sum()),lost_unknown_frames=int(removed-lost.sum()),added_known_IoU=avg(tube.iou[add]),lost_known_IoU=avg(tube.iou[lost]));values[arm]=v
   timevalues[geo]=values
  cov={}
  for layout,positions in z['layouts'].items():
   hits=ids[positions];inside=hits[(hits>=physical_span[0])&(hits<physical_span[1])];k=len(inside);cov[layout]=dict(score=min(1,k/2),hit1=float(k>=1),hit2=float(k>=2),hits=k,zero_hit=float(k==0),within_event_spread_fraction=(int(inside[-1])-int(inside[0]))/(physical_span[1]-physical_span[0])if k>=2 else None,model_coverage=z['model_coverage'][layout])
  mb=z['intervals']['final_MBR'];eq=dict(final_equals_native=mb==z['intervals']['Native'],final_equals_old_marginal=mb==z['intervals']['marginal_MAP'],final_equals_old_offset_MBR=mb==z['intervals'].get('per_offset_MBR'))
  prob=z['q'];nnz=prob[prob>0];rows.append(dict(alias=z['alias'],dataset=ds,parent_alias=z['alias'],phase='extension'if'After'not in timevalues else'initial',event_length_bin=feat['bins']['event_fraction'],event_start_bin=feat['bins']['event_start'],motion_bin=feat['bins']['speed_sizes_sec'],target_area_bin=feat['bins']['target_area'],native_overlap_group='zero'if timevalues['Source']['Native']['t']==0 else'nonzero',coverage=cov,temporal=timevalues,decision_equalities=eq,distribution=dict(states=len(nnz),q_sum=float(prob.sum()),max_mass=float(nnz.max()),entropy=float(-(nnz*np.log(nnz)).sum()),final_MBR_model_expected_tIoU=z['MBR_expected_tIoU'],old_MAP_model_expected_tIoU=z['mode_expected_tIoU']),compute=z['compute']))
  print('SCORED',z['alias'],flush=True)
 phase='extended'if extension else'initial';temporal={ds:{geo:temporal_summary([r for r in rows if r['dataset']==ds and r['phase']=='initial'],geo)for geo in ['Source','Before','After']}for ds in plans};coverage={ds:coverage_summary([r for r in rows if r['dataset']==ds])for ds in plans};decision=decisions(rows,phase);P.mkdir(parents=True,exist_ok=True);filename='RESULTS_EXTENDED.json'if extension else'RESULTS_INITIAL.json';put(P/filename,dict(status='complete',count=len(rows),T0_primary_count=64,temporal=temporal,coverage=coverage,rows=rows,GT_after_all_predictions=True,physical_coverage_not_identity_correctness=True,parent_macro_equals_query_macro=True,extension_primary_After_not_mixed=True));put(D/('FINAL_DECISION.json'if extension else'INITIAL_DECISION.json'),decision);put(D/('CPU_COMPLETION_EXTENDED.json'if extension else'CPU_COMPLETION_INITIAL.json'),dict(status='complete',count=len(rows),GT_read=True,metric_checks=checks,results_sha256=sha(P/filename)));print('DECISION',decision['next_action'],flush=True)

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--extension',action='store_true');run(a.parse_args().extension)
