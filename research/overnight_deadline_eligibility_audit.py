"""Independent eligibility reconstruction from already sealed/scored anonymous rows.
No labels, visual inputs, model, posterior fitting, or new final predictions.
"""
import json,hashlib,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];L=R/'outputs/overnight_event_loop_20261023';P=R/'results/overnight_event_loop_20261023'
def read(f):return json.loads(f.read_text())
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def panel(rows,primary):
 means={base:{m:float(np.mean([100*(r['values'][primary][m]-r['values'][base][m])for r in rows]))for m in ['t','v']}for base in ['Source','NativeG6']}
 groups={g:float(np.mean([100*(r['values'][primary]['t']-r['values']['Source']['t'])for r in rows if r['duration_group']==g]))for g in ['short_le5s','long_gt5s']};delta=np.array([r['values'][primary]['t']-r['values']['NativeG6']['t']for r in rows]);win=int((delta>1e-12).sum());harm=int((delta< -1e-12).sum());basic=all(v>0 for x in means.values()for v in x.values())and all(x>0 for x in groups.values())and win>harm
 return dict(queries=len(rows),gains_pp=means,Source_duration_t_pp=groups,time_improved=win,time_harmed=harm,basic_necessary_conditions_pass=basic,not_sufficient_for_matched_controls_or_uncertainty=True)
def main():
 start=time.perf_counter();raw=(L/'state.json').read_bytes();(L/'eligibility_ledger_input.json').write_bytes(raw);s=json.loads(raw);studies=[]
 for branch in s['branches']:
  d=R/branch['path'];pf=d/'plan.json';rp=R/(branch.get('report')or f"results/{branch['name']}/report.json")
  if not pf.exists()or not rp.exists():continue
  plan=read(pf);rep=read(rp);primary=plan.get('primary');domain=rep.get('domains');entry=dict(name=branch['name'],primary=primary,plan_sha256=sha(pf),scored_report_sha256=sha(rp),fingerprint=plan.get('fingerprint'),ledger_qualified=bool(branch.get('qualified',False)))
  if domain:
   entry['panels']={k:panel(v['rows'],primary)for k,v in domain.items()};entry['has_fixed_both_domains']=any(k in domain for k in ['hc2','hc2_recheck'])and any(k in domain for k in ['vidstg','vidstg_recheck']);entry['all_basic_pass']=all(x['basic_necessary_conditions_pass']for x in entry['panels'].values());assert rep.get('passed_both_domains')is False;assert not(entry['has_fixed_both_domains']and entry['all_basic_pass']),entry
  else:
   if primary is not None and rep.get('rows')and primary in rep['rows'][0]['values']:entry['panels']={'HC64':panel(rep['rows'],primary)}
   entry['has_fixed_both_domains']=False;entry['all_basic_pass']=False
  studies.append(entry)
 assert len(studies)==29 and not any(x['ledger_qualified']for x in studies)
 out=dict(kind='Existing scored-evidence audit, NOTnewmethod or newGTscoring',code_sha256=sha(Path(__file__)),ledger_sha256=sha(L/'eligibility_ledger_input.json'),recorded_studies=len(studies),studies=studies,no_fixed_dual_domain_qualified_candidate=True,new_model_calls=0,new_GT_read=0,new_predictions=0,no_thresholds_or_duration_groups_changed=True,exposed_panels_remain_exposed=True,seconds=time.perf_counter()-start)
 P.mkdir(parents=True,exist_ok=True);(P/'ELIGIBILITY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(studies=len(studies),fixed_dual_studies=sum(x['has_fixed_both_domains']for x in studies),no_fixed_candidate=True,seconds=out['seconds']),indent=2))
if __name__=='__main__':main()
