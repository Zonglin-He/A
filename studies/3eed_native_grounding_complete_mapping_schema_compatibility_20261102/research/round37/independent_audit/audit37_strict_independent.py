"""Independent strict R37 replay and post-seal task audit. No R38/identity/FINAL reads."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='4'
import sys,time,json,hashlib,collections,re,math
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'crossplatform/round30'),str(ROOT/'crossplatform/round27'),str(ROOT/'crossplatform/round13'),str(ROOT/'crossplatform/round7'),str(ROOT/'research/round5/audit')]
from common30 import read,dump,sha,verify,load_npz,rows,feature
from observations27 import load as load_observation
from camera_projection import project_model_points
from independent_core_v2 import candidate_ious,transform_gt
BASE=ROOT/'crossplatform/round37';EXP=ROOT/'research/round37/experiment';OUT=Path(__file__).resolve().parent;PRED=BASE/'inference208_attempt003';ARMS=['source','R29_direct','legacy_G','point_fullmap','point_oldsubset','som_fullmap','som_oldsubset']
def close(a,b,t=1e-10):assert abs(float(a)-float(b))<t,(a,b)
def filehash(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def parse(raw,interface):
 s=str(raw).strip();wr=[]
 for pattern,name in [(r'```(?:json)?\s*(.*?)\s*```','markdown_json_fence'),(r'<answer>\s*(.*?)\s*</answer>','answer_tag')]:
  m=re.fullmatch(pattern,s,re.S|re.I)
  if m:s=m.group(1).strip();wr.append(name)
 def pairs(kv):
  d={}
  for k,v in kv:
   if k in d:raise ValueError('duplicate')
   d[k]=v
  return d
 try:o=json.loads(s,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite')))
 except Exception as e:return {'status':'parse_failure','reason':type(e).__name__,'value':None,'wrappers':wr}
 if not isinstance(o,dict) or len(o)!=1:return {'status':'parse_failure','reason':'exact_one_key_object_required','value':None,'wrappers':wr}
 key=next(iter(o))
 if key not in ({'point','point2d','point_2d'} if interface=='point' else {'target_id','id','candidate_id'}):return {'status':'parse_failure','reason':'unknown_key','value':None,'wrappers':wr}
 val=o[key];canonical='point' if interface=='point' else 'target_id'
 if val is None:return {'status':'explicit_null','reason':None,'value':None,'wrappers':wr,'canonical_key':canonical,'input_key':key}
 valid=(isinstance(val,list) and len(val)==2 and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in val)) if interface=='point' else isinstance(val,int) and not isinstance(val,bool)
 if not valid:return {'status':'parse_failure','reason':'invalid_value_type_or_shape','value':None,'wrappers':wr}
 return {'status':'parsed','reason':None,'value':val,'wrappers':wr,'canonical_key':canonical,'input_key':key}
def readout(interface,result,case,z,subset):
 q0=int(z['source_scores'].argmax());pr=result['parsed'];m=None;pt=None;status=pr['status']
 if status=='parsed':
  if interface=='point':
   grid=result['image_grid_thw'][0];assert len(grid)==3 and grid[0]==1;wp,hp=grid[2]*14,grid[1]*14;x,y=map(float,pr['value']);W,H=case['image_size_wh'];pt=[x*W/wp,y*H/hp] if 0<=x<wp and 0<=y<hp else None
   if pt is None or not(0<=pt[0]<W and 0<=pt[1]<H):status='coordinate_outside_image'
   else:
    hits=np.flatnonzero(z['masks_bool'][:,math.floor(pt[1]),math.floor(pt[0])]);status='single_independent_mask' if len(hits)==1 else 'no_mask' if len(hits)==0 else 'overlapping_multiple_masks';m=int(hits[0]) if len(hits)==1 else None
  else:
   cc=[c for c in case['candidates'] if c['target_id']==pr['value']]
   if not cc:status='unlisted_target_id'
   else:m=int(cc[0]['mask_index']);status='listed_independent_mask'
 table=case['old_subset_groups'] if subset else case['full_groups'];g=table.get(str(m),[]) if m is not None else []
 base={'mask_index':m,'original_point':pt,'old_subset_contains_selected_mask':m is not None and str(m) in case['old_subset_groups'],'full_group_nonempty':m is not None and bool(case['full_groups'].get(str(m),[]))}
 if not g:return {'query':q0,'mapped':False,'status':status if m is None else 'absent_from_old_subset' if subset and str(m) not in table else 'mask_without_3D_group',**base,'group_size':0,'query_changed':False}
 q=max(g,key=lambda j:(float(z['source_scores'][j]),-j));return {'query':int(q),'mapped':True,'status':'mapped_original_query',**base,'group_size':len(g),'query_changed':q!=q0}
def geometry(scene,boxes):
 pts=scene['visible_points_model'];depth=scene['depth_camera'];inds=[]
 for obj in scene['objects']:
  ids=np.asarray(obj['exclusive_visible_indices'],int)
  if len(ids)>=4:
   q1,q3=np.quantile(depth[ids],[.25,.75]);spread=max(q3-q1,.1);ids=ids[(depth[ids]>=q1-1.5*spread)&(depth[ids]<=q3+1.5*spread)]
  inds.append(ids)
 S=np.zeros((256,len(inds)),int);C=np.zeros_like(S,dtype=float);I=np.zeros_like(S,dtype=float);signs=np.array([[(-1 if not i&(1<<j) else 1) for j in range(3)] for i in range(8)])
 for q,b in enumerate(boxes):
  if (b[3:]<=0).any():continue
  proj=project_model_points(b[:3]+signs*b[3:]/2,scene['camera_spec'],scene['image_size_wh']);okay=(proj['depth']>.1).all() and np.isfinite(proj['uv']).all();rect=np.r_[proj['uv'].min(0),proj['uv'].max(0)] if okay else None
  for m,ids in enumerate(inds):
   S[q,m]=int((np.abs(pts[ids]-b[:3])<=b[3:]/2+.1).all(1).sum());C[q,m]=S[q,m]/max(len(ids),1)
   if okay:
    target=np.asarray(scene['objects'][m]['detector_box_xyxy'],float);inter=np.maximum(np.minimum(rect[2:],target[2:])-np.maximum(rect[:2],target[:2]),0).prod();union=np.maximum(rect[2:]-rect[:2],0).prod()+np.maximum(target[2:]-target[:2],0).prod()-inter;I[q,m]=inter/union if union>0 else 0.
 return S,C,I,[len(i) for i in inds]
def main():
 start=time.perf_counter();plan=read(EXP/'PLAN_LOCK.json');amend=read(EXP/'MEASUREMENT_AMENDMENT_LOCK.json');fix=read(EXP/'ATTEMPT003_IMPLEMENTATION_LOCK.json');reassess=read(EXP/'MEASUREMENT_GATE_REASSESSMENT_BEFORE_DATASET_LOCK.json');assert amend['original_plan_sha256']==fix['original_plan_sha256']==sha(EXP/'PLAN_LOCK.json');assert fix['root_gate_reassessment_sha256']==sha(EXP/'MEASUREMENT_GATE_REASSESSMENT_BEFORE_DATASET_LOCK.json')
 for p,h in {**plan['program_sha256'],**amend['program_sha256'],**fix['program_sha256']}.items():assert sha(ROOT/p)==h,p
 for p,h in plan['input_sha256'].items():assert sha(EXP/'ORIGINAL_REVIEW_SOURCES_SNAPSHOT.json' if p.endswith('native_grounding_G/SOURCES.json') else ROOT/p)==h,p
 for d in [BASE/'prepared208_attempt001',PRED,BASE/'unit_control_attempt002',BASE/'unit_diagnostic_attempt003']:verify(d)
 binding=read(PRED/'RUN_BINDING.json');assert binding['implementation_sha256']==sha(EXP/'ATTEMPT003_IMPLEMENTATION_LOCK.json') and binding['plan_sha256']==sha(EXP/'PLAN_LOCK.json') and binding['prepared_seal_sha256']==sha(BASE/'prepared208_attempt001/SEALED.json')
 modelhash=[]
 for name,h in plan['model_files_sha256'].items():assert filehash(Path(plan['model_dir'])/name)==h;modelhash.append({'model_file':name,'sha256':h})
 rr=rows('A208');assert len(rr)==208;ledger=[json.loads(s) for s in (EXP/'GENERATION_BUDGET.jsonl').read_text().splitlines()];assert len(ledger)==426 and [x['attempt_index'] for x in ledger]==list(range(1,427));assert sum(x['interface'] in ['point','som'] for x in ledger)==416;assert sum(x['interface']=='point_unit_control' for x in ledger)==2;assert sum(x['interface']=='point_unit_diagnostic' for x in ledger)==8
 assert len({(x['alias'],x['interface']) for x in ledger if x['interface'] in ['point','som']})==416;assert sum(x.get('phase')=='preflight' for x in ledger)==24 and sum(x.get('phase')=='full' for x in ledger)==392
 firstdata=min(x['started_unix'] for x in ledger if x['interface'] in ['point','som']);from datetime import datetime
 assert datetime.fromisoformat(fix['registered_local']).timestamp()<firstdata and datetime.fromisoformat(reassess['registered_local']).timestamp()<firstdata
 original={x['id_PRIVATE']:x for x in read(EXP/'ALL_ROWS_PRIVATE.json')};index={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};scene_cache={};task=[];grows=[];calls=[];failshape=collections.Counter();callhashes=[];matrixmax=0.;preids={r['id_PRIVATE'] for r in read(EXP/'PREFLIGHT12_PRIVATE.json')};precollections={i:[] for i in ['point','som']}
 for j,r in enumerate(rr):
  c=read(BASE/'prepared208_attempt001/cases_PRIVATE'/(r['id']+'.json'));z=load_npz(BASE/'prepared208_attempt001/arrays_PRIVATE'/(r['id']+'.npz'));src=feature(r);assert np.array_equal(src['boxes'],z['source_boxes']) and np.array_equal(src['scores'],z['source_scores']);M=len(z['masks_bool']);assert M==len(c['candidates'])
  v=(z['support_count']>=3)&(z['support_coverage']>=.15)&(z['projection_iou']>=.1)&(z['source_boxes'][:,3:]>0).all(1)[:,None];uniq=v.sum(1)==1;full={str(m):np.flatnonzero(v[:,m]&uniq).tolist() for m in range(M)};assert full==c['full_groups'];old=read(ROOT/'crossplatform/round29/groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['groups_PRIVATE'];assert old==c['old_subset_groups'] and all(full[m]==g for m,g in old.items());reps={m:max(g,key=lambda q:(float(z['source_scores'][q]),-q)) for m,g in full.items() if g};assert reps==c['representatives_full']
  ck=(r['platform'],r['imagepath'],r['pointpath'])
  if ck not in scene_cache:
   if len(scene_cache)>3:scene_cache.clear()
   scene_cache[ck]=load_observation(r)
  frame,scene,obsbinding=scene_cache[ck];assert np.array_equal(z['masks_bool'],frame['instances']['masks_bool']) and np.array_equal(z['mask_boxes'],frame['instances']['boxes_xyxy']);assert list(scene['image_size_wh'])==c['image_size_wh'];assert obsbinding==c['bindings'];assert hashlib.sha256(np.ascontiguousarray(frame['rgb']).tobytes()).hexdigest()==c['original_RGB_array_sha256'];assert filehash(r['imagepath'])==c['image_sha256'];assert filehash(BASE/'prepared208_attempt001/panels_PRIVATE'/(r['id']+'.png'))==c['panel_sha256']
  S,C,I,trim=geometry(scene,z['source_boxes']);assert np.array_equal(S,z['support_count']);assert abs(C-z['support_coverage']).max(initial=0)<1e-12;err=float(abs(I-z['projection_iou']).max(initial=0));matrixmax=max(matrixmax,err);assert err<1e-12;assert trim==c['depth_trim_counts']
  order=sorted(range(M),key=lambda m:(hashlib.sha256(('R13_full_description_candidate_order_v1|'+r['id']+'|'+hashlib.sha256(np.ascontiguousarray(z['masks_bool'][m]).tobytes()).hexdigest()).encode()).hexdigest(),m));assert [d['mask_index'] for d in c['candidates']]==order and [d['target_id'] for d in c['candidates']]==list(range(1,M+1))
  rec=read(PRED/'predictions_PRIVATE'/(r['id']+'.json'));q={'source':int(z['source_scores'].argmax())};legacy=load_npz(ROOT/'crossplatform/round29/direct208_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));names=legacy['arm_names'].tolist();q['R29_direct']=int(legacy['queries'][names.index('full_pair_group')]);q['legacy_G']=int(legacy['queries'][names.index('legacy_G')]);assert q['source']==c['source_q0'];rd={}
  for interface in ['point','som']:
   path=PRED/rec['call_refs'][interface]['file'];assert filehash(path)==rec['call_refs'][interface]['sha256'];call=read(path);res=call['result'];assert call['state']=='completed_single_greedy_generation_no_retry';assert call['binding']['id_PRIVATE']==r['id'] and call['binding']['alias']==r['alias'] and call['binding']['interface']==interface and call['binding']['image_sha256']==c['image_sha256'];assert call['binding']['plan_sha256']==sha(EXP/'PLAN_LOCK.json');assert res['query_sha256']==call['binding']['query_sha256']==hashlib.sha256(r['caption'].encode()).hexdigest();assert call['binding']['panel_sha256']==(c['panel_sha256'] if interface=='som' else None)
   parsed=parse(res['raw_answer_PRIVATE'],interface);assert parsed==res['parsed'];assert len(res['image_grid_thw'])==(1 if interface=='point' else 2);assert all(g[0]==1 and 256<=g[1]*g[2]//4<=1024 for g in res['image_grid_thw']);assert res['model_resized_wh']==[res['image_grid_thw'][0][2]*14,res['image_grid_thw'][0][1]*14];assert res['original_wh']==c['image_size_wh'];assert res['image_visual_tokens']==[g[1]*g[2]//4 for g in res['image_grid_thw']];assert res['input_tokens']<=8192 and res['generated_tokens']<=96 and res['max_new_tokens']==96
   assert res['images_array_sha256'][0]==c['original_RGB_array_sha256']
   if interface=='som':assert res['images_array_sha256'][1]==hashlib.sha256(np.ascontiguousarray(np.asarray(Image.open(BASE/'prepared208_attempt001/panels_PRIVATE'/(r['id']+'.png')).convert('RGB'))).tobytes()).hexdigest()
   if interface=='point' and parsed['status']=='parse_failure':
    try:
     text=res['raw_answer_PRIVATE'].strip()
     for pattern in [r'```(?:json)?\s*(.*?)\s*```',r'<answer>\s*(.*?)\s*</answer>']:
      matched=re.fullmatch(pattern,text,re.S|re.I)
      if matched:text=matched.group(1).strip()
     obj=json.loads(text);kind='singleton_list_object' if isinstance(obj,list) and len(obj)==1 else 'object_with_extra_fields' if isinstance(obj,dict) and len(obj)>1 else 'other_valid_JSON';failshape[kind]+=1
    except Exception:failshape['invalid_JSON']+=1
   for subset in [False,True]:
    arm=interface+('_oldsubset' if subset else '_fullmap');rd[arm]=readout(interface,res,c,z,subset);assert rd[arm]==rec['readouts'][arm];q[arm]=rd[arm]['query']
   scal={'alias':r['alias'],'platform':r['platform'],'interface':interface,'parsed_status':parsed['status'],'parse_reason':parsed['reason'],'input_tokens':res['input_tokens'],'visual_tokens':sum(res['image_visual_tokens']),'generated_tokens':res['generated_tokens'],'seconds_generate':res['seconds_generate']};calls.append(scal);callhashes.append({'alias':r['alias'],'interface':interface,'sha256':filehash(path)})
   if r['id'] in preids:precollections[interface].append((parsed,rd[interface+'_fullmap']))
  p=load_npz(PRED/'predictions_PRIVATE'/(r['id']+'.npz'));qs=np.array([q[a] for a in ARMS]);assert p['arm_names'].tolist()==ARMS and np.array_equal(p['queries'],qs) and np.array_equal(p['boxes'],z['source_boxes'][qs]);saved=original[r['id']];assert saved['queries']==q and saved['readouts']==rd
  ix=index[r['id']];metap=ix['meta_path']
  if metap not in containers:assert filehash(metap)==ix['metadata_container_sha256'];containers[metap]=read(metap)
  gt=transform_gt(containers[metap]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));boxes=z['source_boxes'][qs].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);vals=candidate_ious(boxes,gt)[0]
  for arm,val in zip(ARMS,vals):close(val,saved['IoU'][arm],1e-12)
  task.append({'alias':r['alias'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'id_PRIVATE':r['id'],'queries':q,'IoU':dict(zip(ARMS,map(float,vals))),'readouts':rd});grows.append({'alias':r['alias'],'platform':r['platform'],'masks':M,'full_nonempty_groups':sum(bool(g) for g in full.values()),'old_subset_groups':len(old),'unique_queries':int(uniq.sum()),'source_and_representatives_fixed':True,'raw_scene_matrices_recomputed':True})
  if (j+1)%32==0:print('INDEPENDENT_R37',j+1,round(time.perf_counter()-start,2),flush=True)
 summary=read(EXP/'TASK_SUMMARY.json');neg=[];coverage={};scenes=[]
 for r in task:
  for arm in ARMS:
   for ref in ['source','R29_direct','legacy_G','point_fullmap','som_fullmap']:
    if arm!=ref and r['IoU'][arm]<r['IoU'][ref]-1e-10:neg.append({'alias':r['alias'],'platform':r['platform'],'arm':arm,'reference':ref,'delta_pp':100*(r['IoU'][arm]-r['IoU'][ref])})
 assert neg==read(EXP/'ALL_NEGATIVES.json')
 for pl in ['drone','quad']:
  part=[r for r in task if r['platform']==pl];coverage[pl]={}
  for arm in ARMS:
   vals=np.array([r['IoU'][arm] for r in part]);dd=vals-np.array([r['IoU']['source'] for r in part]);calc={'n':104,'mIoU_percent':float(vals.mean()*100),'Acc25_percent':float((vals>.25).mean()*100),'Acc50_percent':float((vals>.5).mean()*100),'delta_vs_source_pp':float(dd.mean()*100),'improved':int((dd>1e-10).sum()),'worsened':int((dd< -1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries']['source'] for r in part)}
   for k,vv in calc.items():close(vv,summary[pl][arm][k])
  for arm in ['point_fullmap','point_oldsubset','som_fullmap','som_oldsubset']:coverage[pl][arm]={'n':104,'mapped':sum(r['readouts'][arm]['mapped'] for r in part),'sourcefallback_unknown':sum(not r['readouts'][arm]['mapped'] for r in part),'mapped_can_keep_q0':sum(r['readouts'][arm]['mapped'] and not r['readouts'][arm]['query_changed'] for r in part),'status_counts':dict(collections.Counter(r['readouts'][arm]['status'] for r in part))}
  for j,sc in enumerate(sorted({r['scene_PRIVATE'] for r in part})):
   ss=[r for r in part if r['scene_PRIVATE']==sc];scenes.append({'platform':pl,'scene_alias':pl+'_scene'+str(j),'n':len(ss),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in ss])*100) for a in ARMS}})
 assert coverage==read(EXP/'COVERAGE_UNKNOWN.json') and scenes==read(EXP/'SCENE_ROWS.json')
 for arm in ARMS:close(np.mean([summary[p][arm]['mIoU_percent'] for p in ['drone','quad']]),summary['balanced_mIoU_percent'][arm])
 costs=read(EXP/'COSTS.json')
 for cc in costs:
  subset=[r for r in calls if r['interface']==cc['interface']];assert len(subset)==cc['calls']==208
  for field,source in [('input_tokens_total','input_tokens'),('image_visual_tokens_total','visual_tokens'),('generated_tokens_total','generated_tokens'),('generate_seconds_total','seconds_generate')]:close(sum(r[source] for r in subset),cc[field],1e-8)
  assert sum(r['parsed_status']=='parse_failure' for r in subset)==cc['parse_failure'] and sum(r['parsed_status']=='explicit_null' for r in subset)==cc['explicit_null']
 preflight=read(PRED/'PREFLIGHT_RESULT.json')
 for iface,pairs in precollections.items():
  target=preflight['interfaces'][iface];assert len(pairs)==12;assert sum(p[0]['status']!='parse_failure' for p in pairs)==target['parsed_JSON_or_null'];assert sum(p[1]['mapped'] for p in pairs)==target['nonzero_legal_3D_mapped'];assert sum(p[1]['query_changed'] for p in pairs)==target['query_changes']
 # Preserve the old2 control and new8 capability results; no unit calibration.
 synthetic=[]
 for folder,count in [('unit_control_attempt002',2),('unit_diagnostic_attempt003',8)]:
  data=read(BASE/folder/'RESULT.json');inside=[];abs_err=[];alt_err=[]
  for r in data['records']:
   res=r['result'];assert parse(res['raw_answer_PRIVATE'],'point')==res['parsed'];W,H=r['canvas_wh'];assert res['model_resized_wh']==[W,H] and res['image_grid_thw'][0]==[1,H//14,W//14];val=res['parsed']['value'];assert res['parsed']['status']=='parsed'
   if count==8:name=f"{W}x{H}_{int(r['center_norm'][0]*100)}_{int(r['center_norm'][1]*100)}";im=Image.open(BASE/folder/(name+'.png')).convert('RGB');side=.1*min(W,H);close(r['square_side_pixels'],side);center=r['center_norm'];absolute=np.array(val)/[W,H];counter=np.array(val)/1000;ae=float(np.linalg.norm(absolute-center));ce=float(np.linalg.norm(counter-center));close(ae,r['absolute_normalized_L2_to_center']);close(ce,r['counterfactual1000_normalized_L2_to_center']);abs_err.append(ae);alt_err.append(ce)
   else:im=Image.open(BASE/folder/(str(W)+'x'+str(H)+'.png')).convert('RGB')
   hit=bool(0<=val[0]<W and 0<=val[1]<H and im.getpixel((math.floor(val[0]),math.floor(val[1])))==(0,0,255));assert hit==r['output_inside_actual_blue_region'];inside.append(hit)
  assert len(inside)==count
  if count==8:assert sum(inside)==data['inside_blue_count']==1;close(np.mean(abs_err),data['mean_absolute_normalized_L2']);close(np.mean(alt_err),data['mean_counterfactual1000_normalized_L2']);assert data['state']=='COMPLETED_MODEL_CAPABILITY_DIAGNOSTIC' and data['perfect_localization_not_technical_gate']
  else:assert sum(inside)==1 and data['state']=='FAILED_MODEL_COORDINATE_UNIT_CONTROL'
  synthetic.append({'attempt':folder,'calls':count,'inside_blue':sum(inside),'not_software_pass':True})
 # Reproduce reported scene resampling with same fixed deterministic design.
 rng=np.random.default_rng(37);clusters={pl:[[r for r in task if r['platform']==pl and r['scene_PRIVATE']==sc] for sc in sorted({r['scene_PRIVATE'] for r in task if r['platform']==pl})] for pl in ['drone','quad']}
 for row in read(EXP/'FACTOR_CONTRASTS.json'):
  a,b=row['arm'],row['reference'];d={pl:float(np.mean([r['IoU'][a]-r['IoU'][b] for r in task if r['platform']==pl])*100) for pl in ['drone','quad']}
  for pl,v in d.items():close(v,row['platform_delta_pp'][pl]);close(np.mean(list(d.values())),row['balanced_delta_pp'])
  dist=[]
  for _ in range(2000):
   dist.append(100*np.mean([np.mean([r['IoU'][a]-r['IoU'][b] for j in rng.integers(0,len(cc),len(cc)) for r in cc[j]]) for pl,cc in clusters.items()]))
  assert np.allclose(np.quantile(dist,[.025,.975]),row['descriptive_scene_bootstrap95_pp'],atol=1e-12,rtol=0)
 seal=read(PRED/'SEALED.json');assert seal['model_frozen_unchanged'] and seal['successful_VLM_calls']==416 and seal['total_generation_attempts']==426 and seal['preflight_calls_reused']==24 and seal['newcalls_this_full_phase']==392 and seal['greedy'] and seal['no_TTA_or_new_boxes'];assert read(BASE/'inference208_attempt001/FAILURE_LOAD.json')['successful_or_attempted_generations']==0
 result={'state':'PASS_STRICT_R37_PIPELINE','observations':208,'VLM_cached_responses':416,'strict_task_values':1456,'negative_comparisons':len(neg),'raw_scene_geometry_recomputed':208,'geometry_projection_maxerror':matrixmax,'oldsubset_same_masks_groups_exact':True,'source256boxes_scores_and_representatives_fixed':True,'strict_parse_fail_shape_counts':dict(failshape),'preflight12x2_reused_no_retry':True,'generation_ledger':{'dataset':416,'synthetic_old':2,'synthetic_new':8,'total':426,'budget':440},'synthetic_capability_controls':synthetic,'actual_model_files_hashes_rechecked':len(modelhash),'frozen_model_runtime_evidence':'Reviewed eval/requires_gradFalse, parameter version/buffer digest snapshots and final unchanged receipt; CPU does not independently inspect GPU parameter bytes.','model_identity_hashes_bound_to_plan':True,'all_platform_scene_contrast_cost_coverage_tables_reproduced':True,'R37_predictions_not_changed':True,'no_R38_or_independent_identity_annotations_read':True,'no_FINAL_U_T_access':True,'no_GPU_newVLM_optimizer_training':True,'limitations':['Strict parser rejects valid singleton-list/extra-field JSON; outcome is strict pipeline, not isolated VLM semantic ability.','Synthetic capability1/8 is not software coordinate correctness or a passed hit-rate gate.','A208 exposed development,2Drone/3Quad scenes; descriptive intervals are not independent confirmatory significance.','Raw point/mask coordinates audited locally only, not independent identity ground truth.'],'prediction_seal_sha256':sha(PRED/'SEALED.json'),'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'amendment003_sha256':sha(EXP/'ATTEMPT003_IMPLEMENTATION_LOCK.json'),'seconds':time.perf_counter()-start}
 dump(OUT/'AUDIT_RESULT.json',result);dump(OUT/'ALL208_STRICT_TASK_ROWS_PRIVATE.json',task);dump(OUT/'ALL_NEGATIVES.json',neg);dump(OUT/'ALL208_GEOMETRY_COVERAGE_ROWS.json',grows);dump(OUT/'ALL416_RESPONSE_SCALARS.json',calls);dump(OUT/'ALL416_RESPONSE_HASHES.json',callhashes);dump(OUT/'MODEL_FILE_HASHES.json',modelhash);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
