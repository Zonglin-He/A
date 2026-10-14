"""Independent reparsing, inference-only eligibility sealing, then diagnostic GT."""
import sys,time,json,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stvg_evidence_controls_inputs_v1 import ROOT,BASE,sha,read,write,roster,pairs,CONTROLS
from scripts import stvg_evidence_controls_schema_v1 as schema
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal,score_tube_xyxy,_metric_functions

def canonical_choice(row,case):
 if row['output_truncated_at_cap'] or row['parsed']['status']!='valid':return None
 p=row['parsed'];iv=p.get('predicted_interval_seconds')
 if iv is not None:
  duration=(case['video_bounds'][1]-case['video_bounds'][0])/case['input']['fps']
  if iv[0]<0 or iv[1]>duration+1e-6:return None
 return row['request_metadata']['visible_ID_to_candidate'].get(p['preference'],p['preference'])
def main():
 assert read(BASE/'ADAPTATION_INTERFACE_LOCK.json')['review_source_sha256']==sha(Path(__file__))
 seal=read(BASE/'GLOBAL_READOUT_BARRIER.json');rt=read(BASE/'RUNTIME.json');assert seal['status']=='complete' and seal['count']==rt['model_call_cap']==232
 assert seal['runtime_sha256']==sha(BASE/'RUNTIME.json') and seal['job_roster_sha256']==sha(BASE/'JOB_ROSTER.json')
 for n,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/n)==h,n
 out={};packets=[];cases={};cats={};pairmap={};controlrows=[];selections=[]
 for rec in roster():cases[rec['case_id']],cats[rec['case_id']],pairmap[rec['case_id']]=pairs(rec)
 for f in seal['files']:
  path=ROOT/f['path'];assert sha(path)==f['sha256'];r=read(path);assert r['GT_read'] is False and r['runtime_sha256']==sha(BASE/'RUNTIME.json')
  try:p=schema.parse(r['raw_output'],r['kind']);assert p==r['parsed']
  except (AssertionError,ValueError,TypeError,KeyError) as e:assert r['parsed']['status']=='invalid_output' and r['parsed']['error_type']==type(e).__name__
  meta=r['request_metadata'];cid=meta['case_id'];branch=meta['branch'];pair=pairmap[cid][branch]
  canonical=[x.get('canonical_id',x.get('candidate_id')) for x in pair]
  expected={'1':canonical[0],'2':canonical[1]} if meta['control']!='id_swap' else {'1':canonical[1],'2':canonical[0]}
  assert meta['visible_ID_to_candidate']==expected, 'response ID binding differs from locked actual pair/control'
  rec=next(x for x in roster() if x['case_id']==cid);assert meta['source_case_sha256']==rec['source_case_sha256'] and meta['dataset']==rec['dataset']
  out[path.stem]=r;packets.append(r)
 assert len(out)==232
 for rec in roster():
  cid=rec['case_id'];case=cases[cid]
  for branch in ['temporal','spatial','coarse']:
   for rep in (['paired_small','independent_high'] if branch!='coarse' else ['independent_high']):
    for task in (['native_style','alignment'] if rep=='independent_high' and branch!='coarse' else ['native_style']):
     pref=[canonical_choice(out[f'{cid}_{branch}_{rep}_{c}_{task}'],case) for c in CONTROLS];stable=pref[0] is not None and all(v==pref[0] for v in pref)
     a,b=pairmap[cid][branch];ids=[v.get('canonical_id',v.get('candidate_id')) for v in [a,b]]
     controlrows.append(dict(case_id=cid,dataset=rec['dataset'],branch=branch,representation=rep,task=task,controls=pref,stable=stable,stable_choice=pref[0] if stable else None,baseline=ids[0],alternative=ids[1],decisive=stable and pref[0] in ids))
 for rec in roster():
  cid=rec['case_id'];rr=[r for r in controlrows if r['case_id']==cid];coarse=next(r for r in rr if r['branch']=='coarse')
  for branch in ['temporal','spatial']:
   rows=[r for r in rr if r['branch']==branch and r['representation']=='independent_high'];assert len(rows)==2
   stable_agree=all(r['decisive'] for r in rows) and rows[0]['stable_choice']==rows[1]['stable_choice']
   chosen=rows[0]['stable_choice'] if stable_agree else rows[0]['baseline']
   if branch=='temporal':
    # Native is already the original MAP decode: fitting it is an explicit no-op.
    eligible=stable_agree and chosen==rows[0]['baseline']
   else:eligible=stable_agree and chosen==rows[0]['alternative'] and coarse['stable'] and coarse['stable_choice']==coarse['baseline']
   selections.append(dict(case_id=cid,dataset=rec['dataset'],branch=branch,eligible_without_GT=bool(eligible),candidate_id=chosen,coarse_query_gate=coarse['stable'] and coarse['stable_choice']==coarse['baseline'],Native_target_no_op=branch=='temporal' and stable_agree and chosen==rows[0]['alternative']))
 # The fit list is fixed before opening ANY labels, at most two eligible per direction/branch.
 fits=[]
 for d in ['hc2','vidstg']:
  for b in ['temporal','spatial']:
   fits.extend([s for s in selections if s['dataset']==d and s['branch']==b and s['eligible_without_GT']][:2])
 write(BASE/'OUTPUT_ONLY_ADAPTATION_SELECTION_BARRIER.json',dict(status='sealed_before_GT',records=selections,conditional_fit_case_roster=fits,control_rows=controlrows,GT_read=False,response_barrier_sha256=sha(BASE/'GLOBAL_READOUT_BARRIER.json'),time=time.time()))
 target,bindings=_truths_after_seal(cases);official,indep,_=_metric_functions();scores={};maxerr=0.;count=0
 for rec in roster():
  cid=rec['case_id'];c=cases[cid];cat=cats[cid];native=pairmap[cid]['spatial'][0]
  def score(box,iv):
   nonlocal maxerr,count
   s=score_tube_xyxy(box,cat['frame_ids'],c['width'],c['height'],iv,target[cid]['truth'],target[cid]['span'],rec['dataset'],official=official,independent=indep);maxerr=max(maxerr,s['official_independent_max_abs_error']);count+=1;return s['metrics']
  row={'dataset':rec['dataset'],'Projection':score(native['boxes_xyxy_normalized'],c['Projection_interval']),'Native':score(native['boxes_xyxy_normalized'],c['Native_interval'])}
  for branch in ['temporal','spatial','coarse']:
   row[branch]={}
   for candidate in pairmap[cid][branch]:
    identifier=candidate.get('canonical_id',candidate.get('candidate_id'));box=native['boxes_xyxy_normalized'] if branch=='temporal' else candidate['boxes_xyxy_normalized'];iv=candidate['interval_frames'] if branch=='temporal' else c['Projection_interval'];row[branch][identifier]=score(box,iv)
  scores[cid]=row
 for r in controlrows:
  pool=scores[r['case_id']][r['branch']];a=pool[r['baseline']]['m_vIoU'];b=pool[r['alternative']]['m_vIoU'];r['alternative_minus_baseline_vIoU_pp']=100*(b-a);r['informative_GT_pair']=abs(a-b)>1e-8
  better=r['alternative'] if b>a else r['baseline']
  r['base_correct_when_informative']=r['controls'][0]==better if r['informative_GT_pair'] and r['controls'][0] in pool else None
  r['stable_correct_when_informative']=r['stable_choice']==better if r['informative_GT_pair'] and r['decisive'] else None
 groups=[]
 for d in ['hc2','vidstg']:
  for branch in ['temporal','spatial','coarse']:
   for rep in ['paired_small','independent_high']:
    for task in ['native_style','alignment']:
     rows=[r for r in controlrows if (r['dataset'],r['branch'],r['representation'],r['task'])==(d,branch,rep,task)]
     if not rows:continue
     validbase=[r for r in rows if r['base_correct_when_informative'] is not None];validstable=[r for r in rows if r['stable_correct_when_informative'] is not None]
     groups.append(dict(dataset=d,branch=branch,representation=rep,task=task,pairs=len(rows),ID_same_canonical_count=sum(r['controls'][0] is not None and r['controls'][0]==r['controls'][1] for r in rows),order_same_canonical_count=sum(r['controls'][0] is not None and r['controls'][0]==r['controls'][2] for r in rows),stable_count=sum(r['stable'] for r in rows),stable_decisive_count=sum(r['decisive'] for r in rows),base_informative_decisive=len(validbase),base_correct=sum(r['base_correct_when_informative'] for r in validbase),stable_informative_decisive=len(validstable),stable_correct=sum(r['stable_correct_when_informative'] for r in validstable)))
 validity={k:dict(total=sum(p['kind']==k for p in packets),valid=sum(p['kind']==k and p['parsed']['status']!='invalid_output' for p in packets),truncated=sum(p['kind']==k and p['output_truncated_at_cap'] for p in packets)) for k in ['native_style','facts','alignment']}
 report=dict(status='complete_P0_1_P0_2_postseal_readback',scope='fixed_old8_representations_and_tasks_only',count=232,groups=groups,control_rows=controlrows,selections=selections,conditional_fit_case_roster=fits,scores=scores,output_validity=validity,actual_cost=dict(seconds=sum(p['total_seconds'] for p in packets),input_tokens=sum(p['input_token_count'] for p in packets),output_tokens=sum(p['output_token_count'] for p in packets),peak_GPU_bytes=max(p['GPU_peak_allocated_bytes'] for p in packets)),candidate_metric_count=count,official_independent_max_abs_error=maxerr,truth_binding_metadata=bindings,GT_after_selection_seal=True,GT_read=True,actual_student_fits=0,historically_exposed=True,fresh_unseen_claim=False,time=time.time())
 write(BASE/'POSTSEAL_REVIEW.json',report)
 write(BASE/'ROOT_READBACK.json',dict(status='pass',scope='complete_response_reparse_clock_choice_metric_cost_source_custody_not_feedback_qualification',count=232,bytes=sum((ROOT/f['path']).stat().st_size for f in seal['files']),hash=sha(BASE/'POSTSEAL_REVIEW.json'),GT_read=True,CPU_only=True,time=time.time()))
 print(json.dumps({k:report[k] for k in ['count','groups','output_validity','conditional_fit_case_roster','actual_cost']},indent=2))
if __name__=='__main__':main()
