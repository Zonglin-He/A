"""Independent scalar audit plus explicitly post-core conditional cached analysis.

No new predictions, candidate generation, teacher queries or fitting. GT is read
only to verify existing offline diagnostic arrays; missing labels stay unknown.
"""
import sys,time,collections,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_pair_direction_dev64_v1 import D,P,EPS,verify,read,put,load,sha

def main():
 import numpy as np,torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);start=time.perf_counter();plan=verify(full=True)
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in plan['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
 plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 bar=read(D/'DECOMPOSITION_BARRIER.json');assert bar['no_GT_read']and bar['count']==64
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 pub=read(P/'RESULTS_PUBLIC.json');out=[];checks=collections.Counter();errors=collections.Counter();nodepublic=[]
 for r in plan['records']:
  dec=load(D/'decomposition'/(r['alias']+'.pt'));diag=load(D/'diagnostics'/(r['alias']+'.pt'));source=load(ROOT/r['path']);fit=source['fits'][r['arm']];truth=gt[r['dataset']][0][r['query_ordinal']];meta=plans[r['dataset']][r['query_ordinal']];lookup={(n['step'],n['node']):n for n in diag['nodes']};strata={s:collections.Counter()for s in ['has_improver','both_harm','both_equal','one_equal_harm']};mom=[];counts=collections.Counter();allnorm=collections.Counter();sgains=[]
  den=diag['summary']['known_unobserved_nodes'];unobs=~np.isin(np.arange(len(dec['frame_ids'])),dec['planned_positions'])
  for j,st in enumerate(dec['steps']):
   saved=fit['rounds'][j];eps=saved['standardized_candidates'].numpy()-saved['eta_before'].numpy()[:,None];weights=st['q']-st['p'];individual=r['step']*weights[...,None]*eps;individual[~st['active']]=0
   reconstructed=individual.sum(1);errors['individual_update']=max(errors['individual_update'],float(abs(reconstructed-st['delta']).max()));assert np.allclose(reconstructed,st['delta'],rtol=0,atol=1e-12)
   independently_paired=individual[:,1:9]+individual[:,9:17];assert np.allclose(independently_paired,st['pair_contribution'],rtol=0,atol=1e-12);checks['query_steps']+=1;stepgain=0.
   for t,fid in enumerate(dec['frame_ids']):
    known=int(fid)in truth
    def scalar(b):return float(iou_gt(b,truth[int(fid)],meta,r['dataset']))
    if known and unobs[t]:stepgain+=(scalar(st['after'][t])-scalar(st['current'][t]))/max(den,1)
    if not st['active'][t]:continue
    n=lookup[(st['step'],t)];assert n['known_GT']==known;checks['all_active_nodes']+=1
    delta=st['delta'][t];unit=delta/np.linalg.norm(delta)if np.linalg.norm(delta)>EPS else np.zeros(2);q=st['q'][t];p=st['p'][t];valid=st['valid'][t].numpy()
    qual=np.array([scalar(b)for b in st['candidates'][t]])if known else None;cur=scalar(st['current'][t])if known else None
    if known:
     er=float(abs(qual-n['candidate_quality']).max());errors['scalar_candidate_IoU']=max(errors['scalar_candidate_IoU'],er);assert er<1e-12;assert abs(cur-n['current_quality'])<1e-12;assert abs(scalar(st['after'][t])-n['after_quality'])<1e-12;checks['scalar_candidate_IoUs']+=17;checks['known_active_nodes']+=1
     eq=float(q@qual)-cur;ep=float(p@qual)-cur;assert abs(eq-ep-n['expected_quality_tilt_pp']/100)<1e-12
     counts['valid_support_has_improver']+=int(qual[valid].max()>cur+EPS);counts['valid_support_headroom5pp']+=int(qual[valid].max()>cur+.05);counts['known_active']+=1
     counts['q_expected_above_current']+=int(eq>EPS);counts['q_expected_below_current']+=int(eq<-EPS);mom.append((100*eq,100*ep))
    cc=collections.Counter();nm=collections.Counter();pr=collections.Counter()
    for k,pair in enumerate(n['pairs']):
     a,b=k+1,k+9;v=independently_paired[t,k];alpha=float(weights[t,a]-weights[t,b]);both=bool(valid[a]and valid[b]);gap=None if qual is None else float(qual[a]-qual[b])
     cl='support_limited'if not both else'unknown_GT'if not known else'GT_tie'if abs(gap)<=EPS else'teacher_tie'if abs(alpha)<=EPS else'correct'if alpha*gap>0 else'wrong'
     assert cl==pair['classification'];assert abs(alpha-pair['coefficient'])<1e-12;assert np.allclose(v,pair['contribution'],rtol=0,atol=1e-12);cc[cl]+=1;norm=float(np.linalg.norm(v));nm[cl]+=norm;pr[cl]+=float(v@unit);allnorm[cl]+=norm;checks['pair_classifications']+=1
     if not(known and both):continue
     qa,qb=qual[a],qual[b]
     s='has_improver'if max(qa,qb)>cur+EPS else'both_harm'if max(qa,qb)<cur-EPS else'both_equal'if max(abs(qa-cur),abs(qb-cur))<=EPS else'one_equal_harm';ss=strata[s];ss['pairs']+=1;ss[cl]+=1;ss['norm_'+cl]+=norm;ss['all_norm']+=norm
     if cl in ['correct','wrong']:
      ss['preferred_improves']+=int(qual[a if alpha>0 else b]>cur+EPS)
    for c in ['support_limited','unknown_GT','GT_tie','teacher_tie','correct','wrong']:
     assert cc[c]==n['counts'].get(c,0);assert abs(nm[c]-n['norm_mass'].get(c,0))<1e-12;assert abs(pr[c]-n['signed_projection_onto_actual'].get(c,0))<1e-12
    nodepublic.append({k:n[k]for k in ['alias','dataset','panel','step','node','known_GT','expected_quality_tilt_pp','actual_IoU_delta_pp','correct_component_cosine','cancellation_ratio']})
   sgains.append(stepgain*100)
  assert np.allclose(sgains,diag['summary']['step_all_known_unobserved_delta_pp'],rtol=0,atol=1e-10)
  original=next(z for z in read(ROOT/r['result_path'])['rows']if z['alias']==r['alias']);expected=100*(original['values'][r['arm']]['unobserved_actual_s']-original['values']['E_MBR']['unobserved_actual_s']);assert abs(sum(sgains)-expected)<1e-10;checks['saved_final_spatial_telescopes']+=1
  def ratio(a,b):return a/b if b>EPS else None
  m=dict(teacher_expected_IoU_minus_current_pp=float(np.mean([z[0]for z in mom]))if mom else None,prior_expected_IoU_minus_current_pp=float(np.mean([z[1]for z in mom]))if mom else None,valid_support_has_improver_rate=ratio(counts['valid_support_has_improver'],counts['known_active']),valid_support_headroom5pp_rate=ratio(counts['valid_support_headroom5pp'],counts['known_active']),q_expected_above_current_rate=ratio(counts['q_expected_above_current'],counts['known_active']))
  for cl in ['correct','wrong','GT_tie','teacher_tie','unknown_GT','support_limited']:m[cl+'_norm_fraction_all']=ratio(allnorm[cl],sum(allnorm.values()))
  for s,ss in strata.items():
   m[s+'_direction_accuracy']=ratio(ss['correct'],ss['correct']+ss['wrong']);m[s+'_wrong_norm_fraction_classified']=ratio(ss['norm_wrong'],ss['norm_correct']+ss['norm_wrong']);m[s+'_preferred_improves_rate']=ratio(ss['preferred_improves'],ss['correct']+ss['wrong']);m[s+'_norm_fraction_known_pairs']=ratio(ss['all_norm'],sum(z['all_norm']for z in strata.values()))
  out.append(dict(alias=r['alias'],dataset=r['dataset'],panel=r['panel'],counts=dict(counts),strata={k:dict(v)for k,v in strata.items()},all_norm=dict(allnorm),metrics=m))
 groups={}
 for panel in ['original','selection','combined']:
  groups[panel]={}
  for ds in ['hc2','vidstg']:
   rr=[r for r in out if r['dataset']==ds and(panel=='combined'or r['panel']==panel)];groups[panel][ds]=dict(parents=len(rr),metrics={k:paired([r['metrics'][k]for r in rr])for k in out[0]['metrics']},strata_counts={s:dict(sum((collections.Counter(r['strata'][s])for r in rr),collections.Counter()))for s in strata})
   old=[r for r in pub['rows']if r['dataset']==ds and(panel=='combined'or r['panel']==panel)]
   for k in pub['groups'][panel][ds]['metrics']:assert paired([r['metrics'][k]for r in old])==pub['groups'][panel][ds]['metrics'][k];checks['group_bootstrap_metrics']+=1
 put(P/'CONDITIONAL_ANALYSIS.json',dict(status='complete',post_core_supplement=True,no_new_predictions=True,GT_offline_only=True,groups=groups,rows=out,core_all17_headroom_is_geometric_not_valid_masked=True,pair_strata_are_finite_candidates_not_local_gradient=True,undefined_parent_rates_are_null=True))
 with(P/'ALL_ACTIVE_NODE_AUDIT.csv').open('w')as f:
  writer=csv.DictWriter(f,fieldnames=list(nodepublic[0]));writer.writeheader();writer.writerows(nodepublic)
 verify(full=True);put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=dict(checks),max_absolute_errors=dict(errors),all64_source_and_formal_protection_pins_verified=True,formal64_not_scored_or_used=True,scalar_official_IoU_checked=True,parent_bootstrap_recomputed=True,unknown_GT_preserved=True,CPU_seconds=time.perf_counter()-start,new_GPU_model_expert_DINO_training_OPD=0));print('independent audit and conditional supplement passed',dict(checks),dict(errors),flush=True)
if __name__=='__main__':main()
