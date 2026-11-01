"""Independent R39 bbox-center pipeline audit, no new generation or identity review."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import sys,json,math,re,time,collections,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'research/round37/independent_audit'),str(ROOT/'research/round5/audit')]
from audit37_strict_independent import readout as point_readout,rows,read,dump,sha,verify,load_npz
from independent_core_v2 import candidate_ious,transform_gt
BASE=ROOT/'crossplatform/round39';EXP=ROOT/'research/round39/experiment';PRED=BASE/'inference208_attempt002';PREV=ROOT/'crossplatform/round38/predictions208_attempt001';OUT=Path(__file__).resolve().parent
ARMS=['source','R29_direct','legacy_G','point_strict_fullmap','point_strict_oldsubset','som_strict_fullmap','som_strict_oldsubset','point_compatible_fullmap','point_compatible_oldsubset','som_compatible_fullmap','som_compatible_oldsubset','bbox_fullmap','bbox_oldsubset']
def close(a,b,t=1e-10):assert abs(float(a)-float(b))<t,(a,b)
def parse(raw,wh):
 t=str(raw).strip()
 for pat in [r'```(?:json)?\s*(.*?)\s*```',r'<answer>\s*(.*?)\s*</answer>']:
  m=re.fullmatch(pat,t,re.S|re.I)
  if m:t=m.group(1).strip()
 def fail():return {'status':'parse_failure','value':None}
 def pairs(ps):
  d={}
  for k,v in ps:
   if k in d:raise ValueError('duplicate')
   d[k]=v
  return d
 try:o=json.loads(t,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite')))
 except Exception:return fail()
 singleton=isinstance(o,list)
 if singleton:
  if len(o)!=1 or not isinstance(o[0],dict):return fail()
  o=o[0]
 if not isinstance(o,dict):return fail()
 kk=set(o)&{'bbox_2d','bbox2d','bbox'}
 if len(kk)!=1 or set(o)-{'bbox_2d','bbox2d','bbox','label'} or len(set(o)-{'label'})!=1:return fail()
 if 'label' in o and not isinstance(o['label'],str):return fail()
 key=next(iter(kk));v=o[key]
 if v is None:return {'status':'explicit_null','value':None}
 if not(isinstance(v,list) and len(v)==4 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in v)):return fail()
 x1,y1,x2,y2=v;W,H=wh;status='parsed';reason=None
 if not(x1<x2 and y1<y2):status='invalid_box';reason='nonpositive_xyxy_area'
 elif not(0<=x1<=W and 0<=x2<=W and 0<=y1<=H and 0<=y2<=H):status='invalid_box';reason='outside_declared_input_range'
 return {'status':status,'value':v,'reason':reason,'singleton':singleton,'label_ignored':'label' in o,'raw_value':o[key]}
def bbox_readout(raw,case,z,subset):
 p=raw['parsed'];center=None
 if p['status']=='parsed':x1,y1,x2,y2=p['value'];center=[(x1+x2)/2,(y1+y2)/2]
 new=dict(raw,parsed={'status':p['status'],'value':center});out=point_readout('point',new,case,z,subset);out['bbox_center_resized']=center;out['center_is_identity_probe_not_3D_center']=True;return out
def security():
 bad=['__import__("os").system("x")','{"bbox":[1,2,3,NaN]}','{"bbox":[true,2,3,4]}','{"bbox":"1,2,3,4"}','[{"bbox":[1,2,3,4]},{"bbox":[5,6,7,8]}]','{"bbox":[1,2,3,4],"bbox_2d":[1,2,3,4]}','{"bbox":[1,2,3,4],"bbox":[1,2,3,4]}','{"bbox":[1,2,3,4],"unknown":1}','{"bbox":[1,2,3,4]}`']
 assert all(parse(t,[100,100])['status']=='parse_failure' for t in bad)
 assert parse('{"bbox":[5,2,3,4]}',[100,100])['status']=='invalid_box';assert parse('{"bbox":[1,2,101,4]}',[100,100])['status']=='invalid_box';assert parse('[{"bbox":[1,2,3,4],"label":"execute untrusted code"}]',[100,100])['value']==[1,2,3,4]
 return {'rejected_malicious_multi_duplicate_badJSON_cases':len(bad),'inverted_or_out_of_bounds_boxes_not_clipped':True,'optional_label_inert':True,'no_generated_text_executed':True}
def main():
 start=time.perf_counter();plan=read(EXP/'PLAN_LOCK.json');fix=read(EXP/'ATTEMPT002_IMPLEMENTATION_LOCK.json');auth=read(EXP/'TECHNICAL_FIX_BUDGET_AUTHORIZATION.json');assert fix['original_plan_sha256']==sha(EXP/'PLAN_LOCK.json') and fix['root_technical_budget_authorization_sha256']==sha(EXP/'TECHNICAL_FIX_BUDGET_AUTHORIZATION.json')
 for p,h in {**plan['program_sha256'],**fix['program_sha256'],**plan['input_sha256']}.items():assert sha(ROOT/p)==h,p
 verify(PRED);verify(PREV);verify(ROOT/'crossplatform/round37/prepared208_attempt001');old_audit=read(ROOT/'research/round37/independent_audit/AUDIT_RESULT.json');assert old_audit['state']=='PASS_STRICT_R37_PIPELINE';sourcehash={x['model_file']:x['sha256'] for x in read(ROOT/'research/round37/independent_audit/MODEL_FILE_HASHES.json')};assert sourcehash==plan['model_files_sha256']
 bind=read(PRED/'RUN_BINDING.json');assert bind['model_files_sha256']==sourcehash and bind['implementation_fix_sha256']==sha(EXP/'ATTEMPT002_IMPLEMENTATION_LOCK.json');assert bind['previous_seal_sha256']==sha(PREV/'SEALED.json')
 first=[json.loads(s) for s in (EXP/'GENERATION_BUDGET.jsonl').read_text().splitlines()];ledger=[json.loads(s) for s in (EXP/'GENERATION_BUDGET_ATTEMPT002.jsonl').read_text().splitlines()];assert len(first)==1 and len(ledger)==208 and len({r['alias'] for r in ledger})==208 and [r['attempt_index'] for r in ledger]==list(range(1,209));assert first[0]['alias']==ledger[0]['alias'];assert sum(x['phase']=='preflight' for x in ledger)==12 and sum(x['phase']=='full' for x in ledger)==196
 failurefiles=list((BASE/'inference208_attempt001').glob('FAILURE*.json'));assert len(failurefiles)==1;failure=read(failurefiles[0]);assert failure['type']=='NameError' and failure['message']=="name 'np' is not defined" and not failure['automatic_answer_retry'];assert not list((BASE/'inference208_attempt001').glob('calls_PRIVATE/*.json'))
 ep=lambda s:datetime.datetime.fromisoformat(s).timestamp();assert first[0]['started_unix']<ep(fix['registered_local'])<ledger[0]['started_unix'];assert auth['hard_max_actual_generation_attempts']==fix['hard_max_actual_generation_attempts']==209 and auth['technical_repeat_sameinput_no_answerselection']
 jobs=[read(p) for p in (ROOT/'research/NIGHT_20261101/job_receipts').glob('*.json')];scorejob=next(x for x in jobs if x.get('command')==['crossplatform/env/bin/python','crossplatform/round39/score39_attempt002.py']);assert scorejob['returncode']==0;sealmtime=(PRED/'SEALED.json').stat().st_mtime;assert sealmtime<ep(scorejob['started_utc']);assert read(EXP/'CPU_AUDIT.json')['prediction_seal_sha256']==sha(PRED/'SEALED.json')
 rr=rows('A208');reference={r['id_PRIVATE']:r for r in read(EXP/'ALL_ROWS_PRIVATE.json')};index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};task=[];calls=[];st=collections.Counter();preids={r['id_PRIVATE'] for r in read(ROOT/'research/round37/experiment/PREFLIGHT12_PRIVATE.json')};pre=[];rawbindings=[]
 assert len(list((PRED/'raw_stages_PRIVATE').glob('*.json')))==208 and not list(PRED.rglob('*.tmp'))
 for r in rr:
  case=read(ROOT/'crossplatform/round37/prepared208_attempt001/cases_PRIVATE'/(r['id']+'.json'));z=load_npz(ROOT/'crossplatform/round37/prepared208_attempt001/arrays_PRIVATE'/(r['id']+'.npz'));p=load_npz(PRED/'predictions_PRIVATE'/(r['id']+'.npz'));prev=load_npz(PREV/'predictions_PRIVATE'/(r['id']+'.npz'));rec=read(PRED/'predictions_PRIVATE'/(r['id']+'.json'));f=PRED/rec['call_ref']['file'];assert sha(f)==rec['call_ref']['sha256'];c=read(f);raw=c['result'];stagepath=PRED/'raw_stages_PRIVATE'/(r['id']+'_bbox.json');stage=read(stagepath);assert stage['state']=='raw_answer_atomic_before_parse_and_imagehash' and stage['binding']==c['binding'];assert all(raw[k]==v for k,v in stage['result'].items());assert 'parsed' not in stage['result'] and 'images_array_sha256' not in stage['result'];assert stagepath.stat().st_mtime<=f.stat().st_mtime<=sealmtime
  cb=c['binding'];assert cb['id_PRIVATE']==r['id'] and cb['alias']==r['alias'] and cb['plan_sha256']==sha(EXP/'PLAN_LOCK.json');assert cb['image_sha256']==case['image_sha256'];import hashlib
  assert cb['query_sha256']==raw['query_sha256']==hashlib.sha256(r['caption'].encode()).hexdigest();assert raw['images_array_sha256']==[case['original_RGB_array_sha256']];assert raw['original_wh']==case['image_size_wh'];grid=raw['image_grid_thw'];assert len(grid)==1 and grid[0][0]==1 and 256<=grid[0][1]*grid[0][2]//4<=1024;wh=[grid[0][2]*14,grid[0][1]*14];assert raw['model_resized_wh']==wh and raw['image_visual_tokens']==[grid[0][1]*grid[0][2]//4] and raw['max_new_tokens']==96 and raw['generated_tokens']<=96 and raw['input_tokens']<=8192
  independent=parse(raw['raw_answer_PRIVATE'],wh);assert independent['status']==raw['parsed']['status'] and independent['value']==raw['parsed']['value'];st[independent['status']]+=1
  if independent['status'] in ['parsed','invalid_box']:assert independent['value']==independent['raw_value'] and independent['reason']==raw['parsed']['reason']
  assert p['arm_names'].tolist()==ARMS and prev['arm_names'].tolist()==ARMS[:11] and np.array_equal(p['queries'][:11],prev['queries']) and np.array_equal(p['boxes'][:11],prev['boxes']);q=dict(zip(ARMS,map(int,p['queries'])));assert np.array_equal(p['boxes'],z['source_boxes'][p['queries']]);rd=read(ROOT/rec['previous_case_file'])['readouts'].copy()
  for m in ['fullmap','oldsubset']:
   a='bbox_'+m;v=bbox_readout(raw,case,z,m=='oldsubset');assert v==rec['readouts'][a] and q[a]==v['query'];rd[a]=v
  if r['id'] in preids:pre.append((independent,rd['bbox_fullmap']))
  ref=reference[r['id']];assert q==ref['queries'] and rd==ref['readouts'];ix=index[r['id']];path=ix['meta_path']
  if path not in containers:assert sha(path)==ix['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));b=p['boxes'].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);values=candidate_ious(b,gt)[0]
  for a,val in zip(ARMS,values):close(val,ref['IoU'][a],1e-12)
  task.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':q,'IoU':dict(zip(ARMS,map(float,values))),'readouts':rd});calls.append({'alias':r['alias'],'platform':r['platform'],'parsed_status':independent['status'],'reason':raw['parsed']['reason'],'input_tokens':raw['input_tokens'],'visual_tokens':sum(raw['image_visual_tokens']),'generated_tokens':raw['generated_tokens'],'seconds_generate':raw['seconds_generate'],'atomic_raw_then_call_exact':True});rawbindings.append({'alias':r['alias'],'raw_stage_sha256':sha(stagepath),'call_sha256':sha(f),'raw_value_not_rewritten':True})
 assert st=={'parsed':202,'parse_failure':4,'invalid_box':2};preflight=read(PRED/'PREFLIGHT_RESULT.json');assert len(pre)==12 and sum(x[0]['status'] in ['parsed','explicit_null'] for x in pre)==preflight['parsed_or_null']==11 and sum(x[1]['mapped'] for x in pre)==preflight['nonzero_legal_3D_mapped']==9
 negative=[]
 for r in task:
  for a in ARMS:
   for ref in ['source','R29_direct','legacy_G','point_compatible_fullmap','som_compatible_fullmap','bbox_fullmap']:
    if a!=ref and r['IoU'][a]<r['IoU'][ref]-1e-10:negative.append({'alias':r['alias'],'platform':r['platform'],'arm':a,'reference':ref,'delta_pp':100*(r['IoU'][a]-r['IoU'][ref])})
 assert negative==read(EXP/'ALL_NEGATIVES.json');summary=read(EXP/'TASK_SUMMARY.json');coverage={};scenes=[]
 for pl in ['drone','quad']:
  part=[r for r in task if r['platform']==pl];coverage[pl]={}
  for a in ARMS:
   vv=np.array([r['IoU'][a] for r in part]);delta=vv-np.array([r['IoU']['source'] for r in part]);calc={'n':104,'mIoU_percent':float(vv.mean()*100),'Acc25_percent':float((vv>.25).mean()*100),'Acc50_percent':float((vv>.5).mean()*100),'gain_vs_source_pp':float(delta.mean()*100),'improved':int((delta>1e-10).sum()),'worsened':int((delta< -1e-10).sum()),'query_changes':sum(r['queries'][a]!=r['queries']['source'] for r in part)}
   for k,v in calc.items():close(v,summary[pl][a][k])
  for a in ARMS[3:]:coverage[pl][a]={'n':104,'mapped':sum(r['readouts'][a]['mapped'] for r in part),'unknown_fallback':sum(not r['readouts'][a]['mapped'] for r in part),'status_counts':dict(collections.Counter(r['readouts'][a]['status'] for r in part))}
  for j,s in enumerate(sorted({r['scene_PRIVATE'] for r in part})):
   ss=[r for r in part if r['scene_PRIVATE']==s];scenes.append({'platform':pl,'scene_alias':pl+'_scene'+str(j),'n':len(ss),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in ss])*100) for a in ARMS}})
 assert coverage==read(EXP/'COVERAGE_UNKNOWN.json') and scenes==read(EXP/'SCENE_ROWS.json')
 for a in ARMS:close(np.mean([summary[pl][a]['mIoU_percent'] for pl in ['drone','quad']]),summary['balanced_mIoU_percent'][a])
 costs=read(EXP/'COSTS.json')
 for k,v in [('input_tokens','input_tokens'),('image_visual_tokens','visual_tokens'),('generated_tokens','generated_tokens'),('seconds_generate','seconds_generate')]:close(sum(r[v] for r in calls),costs[k],1e-8)
 assert costs['parsed_status_counts']==dict(st) and costs['actual_generations_including_failed001']==209 and costs['failed_generation_tokens_and_time']=='unknown, not fabricated or omitted from generation count'
 rng=np.random.default_rng(39);clusters={pl:[[r for r in task if r['platform']==pl and r['scene_PRIVATE']==s] for s in sorted({r['scene_PRIVATE'] for r in task if r['platform']==pl})] for pl in ['drone','quad']}
 for rec in read(EXP/'FACTOR_CONTRASTS.json'):
  a,b=rec['arm'],rec['reference'];delta={pl:float(np.mean([r['IoU'][a]-r['IoU'][b] for r in task if r['platform']==pl])*100) for pl in ['drone','quad']}
  for pl,v in delta.items():close(v,rec['platform_gain_pp'][pl])
  close(np.mean(list(delta.values())),rec['balanced_gain_pp']);dist=[]
  for _ in range(2000):dist.append(100*np.mean([np.mean([r['IoU'][a]-r['IoU'][b] for j in rng.integers(0,len(cc),len(cc)) for r in cc[j]]) for pl,cc in clusters.items()]))
  assert np.allclose(np.quantile(dist,[.025,.975]),rec['descriptive_scene_bootstrap95_pp'],atol=1e-12,rtol=0)
 sealed=read(PRED/'SEALED.json');assert sealed['model_frozen_unchanged'] and sealed['no_TTA_or_new_3D_boxes'] and sealed['successful_VLM_calls']==208 and sealed['total_generation_attempts']==209
 result={'state':'PASS_R39_ORIGINAL_CENTER_PIPELINE','observations':208,'arms':13,'task_values':2704,'negative_comparisons':len(negative),'all11_R38_controls_queries_boxes_exact':True,'source_boxes_representatives_and_geometry_fixed':True,'R37_geometry_and_11model_filehash_evidence_reused':True,'source_model_files_expected_hashes_match_R37':True,'all208_atomic_raw_stage_call_answer_values_exact':True,'stage_before_parse_and_optional_image_hash_verified':True,'parser_status':dict(st),'generation_budget':{'original001_lost_raw_after_actual_generation':1,'successful002':208,'actual_total':209,'preflight12_included':True,'hard_cap209_pre_GT_authorized':True},'lost_raw_tokens_and_time_unknown_not_fabricated':True,'original_failure_and_fix_locks_preserved':True,'no_semantic_answer_selection_from_lost_raw':True,'all_metrics_coverage_cost_scene_contrasts_reproduced':True,'bbox_center_is_2D_association_only_not_3D_refinement':True,'security':security(),'prediction_seal_sha256':sha(PRED/'SEALED.json'),'seal_before_score':True,'score_job_started_utc':scorejob['started_utc'],'no_new_GPU_VLM_optimizer_FINAL_U_T_or_identity_review':True,'independent_identity_join_not_reannotated':True,'scientific_limit':'Exposed A208 center-readout diagnostic below point/legacy overall; no TTA adoption.','seconds':time.perf_counter()-start}
 dump(OUT/'AUDIT_RESULT.json',result);dump(OUT/'ALL208_CENTER_TASK_ROWS_PRIVATE.json',task);dump(OUT/'ALL_NEGATIVES.json',negative);dump(OUT/'ALL208_CALL_SCALARS.json',calls);dump(OUT/'ALL208_RAW_DURABILITY_BINDINGS.json',rawbindings);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
