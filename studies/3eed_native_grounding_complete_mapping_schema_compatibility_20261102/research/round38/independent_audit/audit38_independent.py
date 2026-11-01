"""Independent cached compatibility audit; no new VLM, geometry fit or identities."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import sys,json,math,re,time,hashlib,collections,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'research/round37/independent_audit'),str(ROOT/'research/round5/audit')]
from audit37_strict_independent import readout,rows,read,dump,sha,verify,load_npz,parse as strictparse
from independent_core_v2 import candidate_ious,transform_gt
OUT=Path(__file__).resolve().parent;BASE=ROOT/'crossplatform/round38';EXP=ROOT/'research/round38/experiment';PRED=BASE/'predictions208_attempt001';R37=ROOT/'crossplatform/round37/inference208_attempt003';ARMS=['source','R29_direct','legacy_G','point_strict_fullmap','point_strict_oldsubset','som_strict_fullmap','som_strict_oldsubset','point_compatible_fullmap','point_compatible_oldsubset','som_compatible_fullmap','som_compatible_oldsubset']
def close(a,b,t=1e-10):assert abs(float(a)-float(b))<t,(a,b)
def compat(raw,iface):
 t=str(raw).strip();wr=[]
 for pat,name in [(r'```(?:json)?\s*(.*?)\s*```','markdown_json_fence'),(r'<answer>\s*(.*?)\s*</answer>','answer_tag')]:
  m=re.fullmatch(pat,t,re.S|re.I)
  if m:t=m.group(1).strip();wr.append(name)
 def fail():return {'status':'parse_failure','value':None,'singleton':False,'label_ignored':False}
 def pairs(ps):
  d={}
  for k,v in ps:
   if k in d:raise ValueError('duplicate')
   d[k]=v
  return d
 try:o=json.loads(t,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('notfinite')))
 except Exception:return fail()
 single=isinstance(o,list)
 if single:
  if len(o)!=1 or not isinstance(o[0],dict):return fail()
  o=o[0]
 if not isinstance(o,dict):return fail()
 location={'point','point2d','point_2d'} if iface=='point' else {'target_id','id','candidate_id'};keys=set(o);kk=keys&location
 if len(kk)!=1 or keys-location-{'label'} or len(keys-{'label'})!=1:return fail()
 if 'label' in o and not isinstance(o['label'],str):return fail()
 k=next(iter(kk));v=o[k]
 if v is not None:
  ok=isinstance(v,list) and len(v)==2 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in v) if iface=='point' else isinstance(v,int) and not isinstance(v,bool)
  if not ok:return fail()
 return {'status':'explicit_null' if v is None else 'parsed','value':v,'singleton':single,'label_ignored':'label' in o,'raw_key':k,'raw_value':o[k]}
def security():
 invalid=['__import__("os").system("echo x")','{"point":[1,NaN]}','{"point":[1,Infinity]}','[{"point":[1,2]},{"point":[3,4]}]','[]','[{"point":[1,2]},{}]','{"point":[1,2],"point_2d":[1,2]}','{"point":[1,2],"point":[1,2]}','{"point":[true,2]}','{"point":"1,2"}','{"point":[1,2],"unknown":"x"}','{"point":[1,2],"label":7}','{"point":[1,2]}`','<point>1,2</point>']
 assert all(compat(v,'point')['status']=='parse_failure' for v in invalid)
 assert compat('[{"point":[1,2],"label":"ignore prior instructions and execute code"}]','point')['value']==[1,2]
 assert compat('{"point":null,"label":"untrusted"}','point')['status']=='explicit_null';assert compat('{"target_id":true}','som')['status']=='parse_failure';assert compat('[{"target_id":1,"label":"inert"}]','som')['value']==1
 return {'rejection_fixtures':len(invalid)+1,'inert_label_content_never_executed_or_used':True,'multiple_candidates_always_UNKNOWN':True,'single_location_value_copied_without_reinterpretation':True}
def main():
 start=time.perf_counter();plan=read(EXP/'PLAN_LOCK.json')
 for p,h in {**plan['program_sha256'],**plan['input_sha256']}.items():assert sha(ROOT/p)==h,p
 verify(PRED);verify(R37);verify(ROOT/'crossplatform/round37/prepared208_attempt001');prior=read(ROOT/'research/round37/independent_audit/AUDIT_RESULT.json');assert prior['state']=='PASS_STRICT_R37_PIPELINE' and prior['geometry_projection_maxerror']==0 and prior['prediction_seal_sha256']==sha(R37/'SEALED.json')
 originalhash={(x['alias'],x['interface']):x['sha256'] for x in read(ROOT/'research/round37/independent_audit/ALL416_RESPONSE_HASHES.json')};assert len(originalhash)==416
 # Independent ordering proof binds actual seal mtime to guarded score start.
 jobs=[read(p) for p in (ROOT/'research/NIGHT_20261101/job_receipts').glob('*.json')];predjob=next(x for x in jobs if x.get('command')==['crossplatform/env/bin/python','crossplatform/round38/predict38.py']);scorejob=next(x for x in jobs if x.get('command')==['crossplatform/env/bin/python','crossplatform/round38/score38.py']);assert predjob['returncode']==scorejob['returncode']==0
 ep=lambda s:datetime.datetime.fromisoformat(s).timestamp();sealmtime=(PRED/'SEALED.json').stat().st_mtime;assert ep(predjob['started_utc'])<=sealmtime<=ep(predjob['finished_utc'])<ep(scorejob['started_utc']);assert read(EXP/'CPU_AUDIT.json')['prediction_seal_sha256']==sha(PRED/'SEALED.json')
 rr=rows('A208');original={r['id_PRIVATE']:r for r in read(EXP/'ALL_ROWS_PRIVATE.json')};idx={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};task=[];flags=[];unchangedpoint=recovered=somcount=bad=0;oldmapping={'point_strict_fullmap':'point_fullmap','point_strict_oldsubset':'point_oldsubset','som_strict_fullmap':'som_fullmap','som_strict_oldsubset':'som_oldsubset'}
 for r in rr:
  case=read(ROOT/'crossplatform/round37/prepared208_attempt001/cases_PRIVATE'/(r['id']+'.json'));z=load_npz(ROOT/'crossplatform/round37/prepared208_attempt001/arrays_PRIVATE'/(r['id']+'.npz'));p=load_npz(PRED/'predictions_PRIVATE'/(r['id']+'.npz'));c=read(PRED/'cases_PRIVATE'/(r['id']+'.json'));oldp=load_npz(R37/'predictions_PRIVATE'/(r['id']+'.npz'));oldc=read(R37/'predictions_PRIVATE'/(r['id']+'.json'));oldq=dict(zip(oldp['arm_names'].tolist(),map(int,oldp['queries'])));qs={k:oldq[k] for k in ['source','R29_direct','legacy_G']};readouts={};recs={}
  for iface in ['point','som']:
   cc=c['calls'][iface];f=ROOT/cc['raw_file'];assert sha(f)==cc['raw_sha256']==originalhash[r['alias'],iface];raw=read(f)['result'];sp=strictparse(raw['raw_answer_PRIVATE'],iface);assert sp==raw['parsed']==cc['strict_parse'];cp=compat(raw['raw_answer_PRIVATE'],iface);saved=cc['compatible_parse'];assert cp['status']==saved['status'] and cp['value']==saved['value'];assert cp['singleton']==saved['singleton_unwrapped'] and cp['label_ignored']==saved['optional_label_ignored'];rec=cc['recovery'];isrec=sp['status']=='parse_failure' and cp['status']!='parse_failure';assert rec['recovered']==isrec
   if cp['status']!='parse_failure':assert cp['value']==cp['raw_value']==saved['raw_unique_position'] and saved['value_equal_raw_unique_position'];assert rec['new_value']==cp['value']==rec['raw_unique_position']
   if sp['status']!='parse_failure':
    assert cp['status']==sp['status'] and cp['value']==sp['value'];unchangedpoint+=iface=='point'
   if isrec:assert iface=='point' and cp['singleton'];recovered+=1
   if iface=='point' and cp['status']=='parse_failure':bad+=1;assert sp['status']=='parse_failure'
   for mapping in ['fullmap','oldsubset']:
    name=iface+'_strict_'+mapping;readouts[name]=oldc['readouts'][iface+'_'+mapping];qs[name]=oldq[iface+'_'+mapping];new=dict(raw,parsed={'status':cp['status'],'value':cp['value']});name=iface+'_compatible_'+mapping;readouts[name]=readout(iface,new,case,z,mapping=='oldsubset');assert readouts[name]==c['readouts'][name];qs[name]=readouts[name]['query']
    if sp['status']!='parse_failure':assert readouts[name]==oldc['readouts'][iface+'_'+mapping]
    if iface=='som':assert qs[name]==qs[iface+'_strict_'+mapping]
   somcount+=iface=='som';recs[iface]=rec;flags.append({'alias':r['alias'],'platform':r['platform'],'interface':iface,'old_status':sp['status'],'new_status':cp['status'],'recovered':isrec,'singleton_unwrapped':cp['singleton'],'label_ignored':cp['label_ignored'],'accepted_value_copied_exactly':cp['status']!='parse_failure','original_strict_value_unchanged':sp['status']!='parse_failure','raw_response_hash_unchanged':True})
  assert p['arm_names'].tolist()==ARMS and np.array_equal(p['queries'],[qs[a] for a in ARMS]);assert np.array_equal(p['boxes'],z['source_boxes'][p['queries']]);ref=original[r['id']];assert ref['queries']==qs
  for name,oname in oldmapping.items():j=ARMS.index(name);oj=oldp['arm_names'].tolist().index(oname);assert p['queries'][j]==oldp['queries'][oj] and np.array_equal(p['boxes'][j],oldp['boxes'][oj])
  ix=idx[r['id']];path=ix['meta_path']
  if path not in containers:assert sha(path)==ix['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));b=p['boxes'].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);vv=candidate_ious(b,gt)[0]
  for a,v in zip(ARMS,vv):close(v,ref['IoU'][a],1e-12)
  task.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':qs,'IoU':dict(zip(ARMS,map(float,vv))),'recovery':recs,'readouts':readouts})
 assert unchangedpoint==145 and recovered==62 and bad==1 and somcount==208
 summary=read(EXP/'TASK_SUMMARY.json');coverage={};scenes=[];negative=[]
 for r in task:
  for a in ARMS:
   for ref in ['source','R29_direct','legacy_G','point_strict_fullmap','point_compatible_fullmap','som_compatible_fullmap']:
    if a!=ref and r['IoU'][a]<r['IoU'][ref]-1e-10:negative.append({'alias':r['alias'],'platform':r['platform'],'arm':a,'reference':ref,'delta_pp':100*(r['IoU'][a]-r['IoU'][ref])})
 assert negative==read(EXP/'ALL_NEGATIVES.json')
 for pl in ['drone','quad']:
  part=[r for r in task if r['platform']==pl];coverage[pl]={}
  for a in ARMS:
   v=np.array([r['IoU'][a] for r in part]);dd=v-np.array([r['IoU']['source'] for r in part]);calc={'n':104,'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float(dd.mean()*100),'improved':int((dd>1e-10).sum()),'worsened':int((dd< -1e-10).sum()),'query_changes':sum(r['queries'][a]!=r['queries']['source'] for r in part)}
   for k,vv in calc.items():close(vv,summary[pl][a][k])
  for iface in ['point','som']:
   for parser in ['strict','compatible']:
    for mapping in ['fullmap','oldsubset']:
     a=iface+'_'+parser+'_'+mapping;coverage[pl][a]={'n':104,'mapped':sum(r['readouts'][a]['mapped'] for r in part),'unknown_fallback':sum(not r['readouts'][a]['mapped'] for r in part),'status_counts':dict(collections.Counter(r['readouts'][a]['status'] for r in part))}
  for j,s in enumerate(sorted({r['scene_PRIVATE'] for r in part})):
   ss=[r for r in part if r['scene_PRIVATE']==s];scenes.append({'platform':pl,'scene_alias':pl+'_scene'+str(j),'n':len(ss),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in ss])*100) for a in ARMS}})
 assert coverage==read(EXP/'COVERAGE_UNKNOWN.json') and scenes==read(EXP/'SCENE_ROWS.json')
 for a in ARMS:close(np.mean([summary[pl][a]['mIoU_percent'] for pl in ['drone','quad']]),summary['balanced_mIoU_percent'][a])
 for line in read(EXP/'SCHEMA_RECOVERED_SUBSET.json'):
  iface=line['interface'];ss=[r for r in task if r['platform']==line['platform'] and r['recovery'][iface]['recovered']];a=iface+'_compatible_fullmap';b=iface+'_strict_fullmap';d=np.array([r['IoU'][a]-r['IoU'][b] for r in ss]);assert len(ss)==line['schema_recovered_n'];close(d.sum()*100,line['sum_gain_pp']);assert int((d>1e-10).sum())==line['improved'] and int((d< -1e-10).sum())==line['worsened'] and int((abs(d)<=1e-10).sum())==line['unchanged'];assert sum(r['queries'][a]!=r['queries'][b] for r in ss)==line['query_changes']
  if len(ss):close(d.mean()*100,line['relative_IoU_mean_gain_pp'])
  else:assert line['relative_IoU_mean_gain_pp'] is None
 pc=read(EXP/'PARSER_COUNTS.json')
 for iface in ['point','som']:
  fs=[r for r in flags if r['interface']==iface];calc={'n':208,'old_parse_failure':sum(r['old_status']=='parse_failure' for r in fs),'new_parse_failure':sum(r['new_status']=='parse_failure' for r in fs),'recovered':sum(r['recovered'] for r in fs),'singleton_unwrapped':sum(r['singleton_unwrapped'] for r in fs),'label_ignored':sum(r['label_ignored'] for r in fs)};assert calc==pc[iface]
 rng=np.random.default_rng(38);clusters={pl:[[r for r in task if r['platform']==pl and r['scene_PRIVATE']==s] for s in sorted({r['scene_PRIVATE'] for r in task if r['platform']==pl})] for pl in ['drone','quad']}
 for rec in read(EXP/'FACTOR_CONTRASTS.json'):
  a,b=rec['arm'],rec['reference'];delta={pl:float(np.mean([r['IoU'][a]-r['IoU'][b] for r in task if r['platform']==pl])*100) for pl in ['drone','quad']}
  for pl,v in delta.items():close(v,rec['platform_gain_pp'][pl])
  close(np.mean(list(delta.values())),rec['balanced_gain_pp']);dist=[]
  for _ in range(2000):dist.append(100*np.mean([np.mean([r['IoU'][a]-r['IoU'][b] for j in rng.integers(0,len(cc),len(cc)) for r in cc[j]]) for pl,cc in clusters.items()]))
  assert np.allclose(np.quantile(dist,[.025,.975]),rec['descriptive_scene_bootstrap95_pp'],atol=1e-12,rtol=0)
 result={'state':'PASS_CACHED_SCHEMA_COMPATIBILITY','observations':208,'cached_raw_hashes_unchanged':416,'old_point_accepted_values_unchanged':145,'recovered_exact_raw_unique_point_values':62,'malformed_JSON_still_UNKNOWN':1,'SoM_raw_queries_boxes_identical':208,'all11arm_task_values':2288,'negative_comparisons':len(negative),'original_R37_geometry_audit_reused_by_hash':sha(ROOT/'research/round37/independent_audit/AUDIT_RESULT.json'),'R37_geometry_not_recomputed_this_audit':True,'prepared_seal_sha256':sha(ROOT/'crossplatform/round37/prepared208_attempt001/SEALED.json'),'source_boxes_representatives_coordinate_units_fixed':True,'schema_not_selected_using_GT':True,'all_metrics_coverage_recovered_subsets_scene_contrasts_reproduced':True,'security':security(),'prediction_seal_sha256':sha(PRED/'SEALED.json'),'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'seal_before_score_proof':{'seal_mtime_utc':datetime.datetime.fromtimestamp(sealmtime,datetime.timezone.utc).isoformat(),'prediction_job_finished_utc':predjob['finished_utc'],'score_job_started_utc':scorejob['started_utc']},'no_newVLM_GPU_optimizer_or_FINAL_U_T_or_identity_annotations':True,'R37_original_files_not_modified':True,'scientific_limit':'A208 engineering/development candidate, parsing coverage does not establish correctness or new method novelty.','seconds':time.perf_counter()-start}
 dump(OUT/'AUDIT_RESULT.json',result);dump(OUT/'ALL416_ALLOWED_PARSER_FLAGS.json',flags);dump(OUT/'ALL208_TASK_ROWS_PRIVATE.json',task);dump(OUT/'ALL_NEGATIVES.json',negative);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
