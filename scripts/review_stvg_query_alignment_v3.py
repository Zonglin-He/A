"""Seal-gated CPU audit; freezes output-only choices before opening labels."""
import sys,json,time,statistics,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stvg_query_alignment_inputs_v3 import BASE,ROOT,OLD_SPARSE,sha,read,write,roster,catalog
from scripts import stvg_query_alignment_schema_v3 as schema
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal,score_tube_xyxy,_metric_functions

def response_preference(row,protocol,branch):
 if not row or row.get('output_truncated_at_cap') or row['parsed'].get('status')=='invalid_output':return None
 p=row['parsed'];key='preference' if protocol=='B' else ('boundary_preference' if branch=='temporal' else 'geometry_preference')
 choice=p[key]
 if choice in ['A','B']:
  if protocol=='C' and p['identity_preference'] in ['A','B'] and p['identity_preference']!=choice:return 'Uncertain'
  return p['candidate_ids_in_order'][0 if choice=='A' else 1]
 return choice

def main():
 seal=read(BASE/'P0_B_GLOBAL_READOUT_BARRIER.json');runtime=read(BASE/'RUNTIME.json')
 assert seal['status']=='complete' and seal['GT_read'] is False
 assert seal['runtime_sha256']==sha(BASE/'RUNTIME.json') and seal['roster_sha256']==sha(BASE/'PAIR_ROSTER.json')
 for n,h in {**runtime['pins'],**runtime['protected']}.items():assert sha(ROOT/n)==h,n
 outputs={};calls=[];classification=[]
 for f in seal['files']:
  path=ROOT/f['path'];assert sha(path)==f['sha256'];row=read(path);outputs[path.stem]=row
  if 'raw_output' not in row:continue
  parser={'direct':schema.parse_B_output,'facts':schema.parse_C_facts_output,'alignment':schema.parse_C_alignment_output}[row['kind']]
  try:actual=parser(row['raw_output'],binding=row['request_metadata']['binding']);assert actual==row['parsed'];valid=True
  except (ValueError,TypeError,KeyError) as exc:
   assert row['parsed'].get('status')=='invalid_output' and row['parsed']['error_type']==type(exc).__name__;valid=False
  assert row['runtime_sha256']==sha(BASE/'RUNTIME.json') and row['GT_read'] is False
  classification.append(dict(output_key=path.stem,kind=row['kind'],valid=valid,truncated=row['output_truncated_at_cap']))
  calls.append(row)
 assert sum(not c.get('exact_prior_B_reuse',False) for c in calls)==seal['actual_model_calls']<=112
 jobs=read(BASE/'PAIR_ROSTER.json')['records'];pairrows=[];selects={}
 for job in jobs:
  prefix=f"{job['case_id']}_{job['branch']}_{job['pair_index']}";baseline=job['left'].get('canonical_id',job['left'].get('candidate_id'));other=job['right'].get('canonical_id',job['right'].get('candidate_id'))
  assert all(prefix+'_'+c+'_B' in outputs and prefix+'_'+c+'_Cfacts' in outputs and prefix+'_'+c+'_C' in outputs for c in ['base','id_swap','order_swap'])
  for protocol in ['B','C']:
   result=[response_preference(outputs.get(prefix+'_'+c+'_'+protocol),protocol,job['branch']) for c in ['base','id_swap','order_swap']]
   stable=all(r==result[0] for r in result) and result[0] is not None
   wrong=outputs.get(prefix+'_wrong_query_'+protocol);wrongpref=response_preference(wrong,protocol,job['branch'])
   pairrows.append(dict(case_id=job['case_id'],dataset=job['dataset'],branch=job['branch'],pair_index=job['pair_index'],protocol=protocol,
    baseline=baseline,other=other,control_preferences=result,stable=stable,stable_preference=result[0] if stable else None,
    wrong_query_preference=wrongpref,wrong_query_call_executed=wrong is not None))
 for caseid in sorted({j['case_id'] for j in jobs}):
  for branch in ['temporal','spatial']:
   for protocol in ['B','C']:
    rows=[r for r in pairrows if r['case_id']==caseid and r['branch']==branch and r['protocol']==protocol]
    assert len(rows)==1
    winners=[r['other'] for r in rows if r['stable'] and r['stable_preference']==r['other']]
    acceptable=all(r['stable'] and r['stable_preference'] in [r['baseline'],r['other'],'Tie'] for r in rows)
    selects[f'{caseid}_{branch}_{protocol}']=dict(candidate_id=winners[0] if len(winners)==1 and acceptable else rows[0]['baseline'],
       update_signal_eligible=len(winners)==1 and acceptable,rule='unique stable alternative; all pair controls supported; else baseline')
 write(BASE/'OUTPUT_ONLY_SELECTION_BARRIER.json',dict(status='sealed_before_GT',records=selects,pair_controls=pairrows,
   response_barrier_sha256=sha(BASE/'P0_B_GLOBAL_READOUT_BARRIER.json'),GT_read=False,time=time.time()))
 # Opening old diagnostic GT or source truth is forbidden until every new response and choice above is sealed.
 cases={r['case_id']:catalog(r)[0] for r in roster()};targets,bindings=_truths_after_seal(cases);official,independent,_=_metric_functions()
 scores={};maxerr=0.;candidatecount=0
 for rec in roster():
  case,cat=catalog(rec);native=next(r for r in cat['spatial']['candidates'] if r['candidate_id']==cat['native_mapped_candidate_id'])
  def score(boxes,interval):
   nonlocal maxerr,candidatecount
   z=score_tube_xyxy(boxes,cat['frame_ids'],case['input']['width'],case['input']['height'],interval,targets[rec['case_id']]['truth'],targets[rec['case_id']]['span'],rec['dataset'],official=official,independent=independent)
   maxerr=max(maxerr,z['official_independent_max_abs_error']);candidatecount+=1;return z['metrics']
  row=dict(dataset=rec['dataset'],raw_Native=score(case['native_boxes'],case['Native_interval']),
   raw_Projection=score(case['native_boxes'],case['Projection_interval']),Native=score(native['boxes_xyxy_normalized'],case['Native_interval']),
   Projection=score(native['boxes_xyxy_normalized'],case['Projection_interval']))
  row['temporal']={r['canonical_id']:score(native['boxes_xyxy_normalized'],r['interval_frames']) for r in cat['temporal']}
  row['spatial']={r['candidate_id']:score(r['boxes_xyxy_normalized'],case['Projection_interval']) for r in cat['spatial']['candidates']}
  scores[rec['case_id']]=row
 for r in pairrows:
  pool=scores[r['case_id']][r['branch']];a=pool[r['baseline']]['m_vIoU'];b=pool[r['other']]['m_vIoU'];d=b-a
  r['other_minus_baseline_vIoU_pp']=100*d;r['informative_GT_pair']=abs(d)>1e-8
  pref=r['stable_preference'];r['stable_decisive']=r['stable'] and pref in [r['baseline'],r['other']]
  r['correct_if_informative_decisive']=(pref==(r['other'] if d>0 else r['baseline'])) if r['informative_GT_pair'] and r['stable_decisive'] else None
 groups=[]
 for dataset in ['hc2','vidstg']:
  ids=[k for k,v in scores.items() if v['dataset']==dataset]
  for branch in ['temporal','spatial']:
   for protocol in ['B','C']:
    sel=[selects[f'{k}_{branch}_{protocol}'] for k in ids];delta=[100*(scores[k][branch][s['candidate_id']]['m_vIoU']-scores[k]['Projection']['m_vIoU']) for k,s in zip(ids,sel)]
    p=[r for r in pairrows if r['dataset']==dataset and r['branch']==branch and r['protocol']==protocol]
    correct=[r['correct_if_informative_decisive'] for r in p if r['correct_if_informative_decisive'] is not None]
    groups.append(dict(dataset=dataset,branch=branch,protocol=protocol,cases=len(ids),directional_selection_count=sum(s['update_signal_eligible'] for s in sel),
      mean_selected_minus_mapped_Projection_vIoU_pp=statistics.mean(delta),per_case_delta_pp=delta,
      pool_oracle_headroom_pp=statistics.mean([100*(max(v['m_vIoU'] for v in scores[k][branch].values())-scores[k]['Projection']['m_vIoU']) for k in ids]),
      pairs=len(p),stable_pair_count=sum(r['stable'] for r in p),informative_stable_decisive_count=len(correct),correct_count=sum(correct),wrong_count=len(correct)-sum(correct),
      ranking_accuracy=sum(correct)/len(correct) if correct else None))
 valid_by_kind={k:dict(total=sum(r['kind']==k for r in classification),valid=sum(r['kind']==k and r['valid'] for r in classification),truncated=sum(r['kind']==k and r['truncated'] for r in classification)) for k in ['direct','facts','alignment']}
 factsrows=[v for v in calls if v['kind']=='facts' and v['parsed'].get('status')!='invalid_output']
 dimensions={f:dict(valid_pairs=len(factsrows),different_candidate_text_count=sum(v['parsed']['facts']['A'][f]!=v['parsed']['facts']['B'][f] for v in factsrows),
    Unknown_value_count=sum(v['parsed']['facts'][label][f].lower()=='unknown' for v in factsrows for label in ['A','B'])) for f in schema.FACT_FIELDS}
 triggers=[]
 for branch in ['temporal','spatial']:
  for protocol in ['B','C']:
   gg=[g for g in groups if g['branch']==branch and g['protocol']==protocol]
   if statistics.mean([g['mean_selected_minus_mapped_Projection_vIoU_pp'] for g in gg])>1e-8 and sum(g['correct_count'] for g in gg)>sum(g['wrong_count'] for g in gg):triggers.append(dict(branch=branch,protocol=protocol,status='feedback_development_signal_P0C_or_branch_support_review_required'))
 report=dict(status='complete_postseal_P0B_diagnostic',scope='fixed_old8_candidates_BvsC_with_A_cached_original_diagnostic',scores=scores,pair_rows=pairrows,groups=groups,
   output_validity=valid_by_kind,fact_dimension_discrimination=dimensions,conditional_feedback_triggers=triggers,
   calls=seal['actual_model_calls'],logical_packets=len(calls),exact_B_reuse_count=sum(c.get('exact_prior_B_reuse',False) for c in calls),maximum_calls=112,GT_read=True,GT_after_all_responses_and_selection_seal=True,official_independent_max_abs_error=maxerr,candidate_metric_count=candidatecount,
   actual_cost=dict(seconds=sum(r['total_seconds'] for r in calls),input_tokens=sum(r['input_token_count'] for r in calls),output_tokens=sum(r['output_token_count'] for r in calls),peak_GPU_bytes=max(r['GPU_peak_allocated_bytes'] for r in calls)),
   A_cache_report_sha256=sha(OLD_SPARSE/'P0_POSTSEAL_REVIEW.json'),truth_binding_metadata=bindings,
   P1_student_fits_executed=0,online_carry_executed=False,historically_exposed=True,fresh_unseen_claim=False,time=time.time())
 write(BASE/'P0_B_POSTSEAL_REVIEW.json',report)
 print(json.dumps({k:report[k] for k in ['status','groups','output_validity','conditional_feedback_triggers','calls','actual_cost']},indent=2))

if __name__=='__main__':main()
