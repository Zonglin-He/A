"""Independent R15 full204 post-score audit. PREPARED; no execution authority.

Before a root-hashed separate audit release and all ten actual source/helper/
semantic/direct seals, this code never decodes GT index/annotations or score rows.
No model, raw sensor decode, adaptation, prediction repair or native-GPU rerank.
"""
from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,math,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
_spec=importlib.util.spec_from_file_location('R15_frozen_reference_core',ROOT/'research/round5/audit/independent_core_v2.py')
core=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(core)
ARMS=('source','proposal_assoc','proposal_geo','graph_assoc','graph_geo','graph_geo_noRGB','graph_geo_relation_disabled','graph_geo_relation_flipped_diagnostic')
ROLES={'fresh64_recent_content_relative_only':64,'R14_regression140_exposed':140}
SOURCE_SHA='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
DEPENDENCIES={'research/round5/audit/independent_core_v2.py':'b4937fdd89f719fe1d9560f770c01597ff6807c74a75013720b470c43393379a','research/round12/evaluation/metrics.py':'4f696eecb3a7b46e237dc13cb5354a512e34f10799590b41bce579f8c04ca81f','tools/obb_iou.py':'45896d29d1f95167b95a27b044fcc06b4e6fe5f9cb98749096c2f707292d4083'}
SCALAR_ATOL=1e-10;POLYGON_IOU_ATOL=1e-7;BOOT_SEED=1201;BOOT_DRAWS=2000

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def resolve(p):
 p=Path(p);return p if p.is_absolute() else ROOT/p

def seal(directory,digest):
 base=resolve(directory);p=base/'SEALED.json'
 if sha(p)!=digest:raise ValueError('Artifact seal mismatch')
 s=read(p)
 if not isinstance(s.get('files'),dict) or not s['files']:raise ValueError('Nonempty finite sealed files required')
 for name,h in s['files'].items():
  q=Path(name)
  if q.is_absolute() or '..' in q.parts or not (base/q).resolve().is_relative_to(base.resolve()) or sha(base/q)!=h:raise ValueError('Unsafe/corrupt sealed member')
 return base,s

def clamp32(raw):
 x=np.array(raw,np.float32,copy=True);x[...,3:6]=np.maximum(x[...,3:6],np.float32(1e-6));return x

def polygon_yaw0_iou(box6,target7):
 """Independent clipped target rectangle x interval-height, all float64.

 This never calls author convex geometry and only applies to pred yaw0/GTyaw7.
 Shift target to prediction center to limit global-coordinate cancellation.
 """
 b=np.asarray(box6,float);t=np.asarray(target7,float)
 if b.shape!=(6,) or t.shape!=(7,) or not np.isfinite(b).all() or not np.isfinite(t).all() or (b[3:]<=0).any() or (t[3:6]<=0).any():raise ValueError('Valid eval6/official7 required')
 dz=max(0.,min(b[2]+b[5]/2,t[2]+t[5]/2)-max(b[2]-b[5]/2,t[2]-t[5]/2))
 if dz==0:return 0.
 c=math.cos(float(t[6]));s=math.sin(float(t[6]));hx,hy=t[3]/2,t[4]/2
 dx,dy=t[0]-b[0],t[1]-b[1]
 poly=[np.array([c*x-s*y+dx,s*x+c*y+dy]) for x,y in [(-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy)]]
 for axis,sgn,limit in [(0,1.,b[3]/2),(0,-1.,b[3]/2),(1,1.,b[4]/2),(1,-1.,b[4]/2)]:
  if not poly:return 0.
  out=[];last=poly[-1];dl=limit-sgn*last[axis]
  for point in poly:
   dp=limit-sgn*point[axis]
   if (dl>=0)!=(dp>=0):
    den=dl-dp
    if den==0:raise ValueError('Unexpected zero clip denominator')
    out.append(last+(dl/den)*(point-last))
   if dp>=0:out.append(point)
   last=point;dl=dp
  poly=out
 if len(poly)<3:return 0.
 area=abs(math.fsum(float(poly[i][0]*poly[(i+1)%len(poly)][1]-poly[i][1]*poly[(i+1)%len(poly)][0]) for i in range(len(poly))))/2
 va=float(np.prod(b[3:]));vb=float(np.prod(t[3:6]));intersection=min(max(area*dz,0.),va,vb)
 return intersection/(va+vb-intersection)

def errors(raw,target):
 d=[float(x)-float(y) for x,y in zip(raw,target[:6])]
 return {'center_L2_m':math.sqrt(math.fsum(x*x for x in d[:3])),'size_L2_m':math.sqrt(math.fsum(x*x for x in d[3:])),'abs_error6_m':[abs(x) for x in d],'size_parameter_not_global_yaw0_enclosure':True}
def mean(v):return math.fsum(map(float,v))/len(v) if len(v) else None

def scene_stats(rows,values):
 by=collections.defaultdict(list)
 for r,v in zip(rows,values):by[(r['platform'],r['scene'])].append(float(v))
 gm=np.array([mean(by[k]) for k in sorted(by)])
 ci=np.quantile(np.random.default_rng(BOOT_SEED).choice(gm,(BOOT_DRAWS,len(gm))).mean(1),[.025,.975]).tolist() if len(gm)>1 else None
 return {'n':len(values),'scene_clusters':len(by),'micro_mean':mean(values),'equal_scene_mean':mean(gm.tolist()),'equal_scene_descriptive_CI95':ci,'bootstrap_seed':BOOT_SEED,'bootstrap_draws':BOOT_DRAWS,'few_scene_not_significance_or_population_proof':True}
def signed(v,tol):
 v=list(map(float,v));pos=[x for x in v if x>tol];neg=[x for x in v if x< -tol]
 return {'n':len(v),'mean':mean(v),'sum':math.fsum(v),'positive_count':len(pos),'negative_count':len(neg),'ties_within_tol':sum(abs(x)<=tol for x in v),'positive_sum':math.fsum(pos),'negative_sum':math.fsum(neg),'gain_sum_exceeds_abs_harm_sum':math.fsum(pos)>abs(math.fsum(neg)),'min':min(v,default=None),'max':max(v,default=None),'tolerance':tol}

class Checker:
 def __init__(self):self.n=0;self.max_abs=0.
 def eq(self,a,b,path='',atol=SCALAR_ATOL):
  self.n+=1
  if isinstance(a,dict):
   if not isinstance(b,dict):raise AssertionError(path+' type')
   for k,v in a.items():
    if k not in b:raise AssertionError(path+'/'+k+' missing')
    self.eq(v,b[k],path+'/'+k,atol)
  elif isinstance(a,(list,tuple)):
   if not isinstance(b,(list,tuple)) or len(a)!=len(b):raise AssertionError(path+' length')
   for j,(x,y) in enumerate(zip(a,b)):self.eq(x,y,path+'/'+str(j),atol)
  elif isinstance(a,bool) or a is None or isinstance(a,str):
   if type(a)!=type(b) or a!=b:raise AssertionError(path+' exact value')
  elif isinstance(a,(float,int,np.number)):
   if isinstance(b,bool) or not isinstance(b,(float,int)) or not math.isfinite(float(b)):raise AssertionError(path+' numeric type')
   delta=abs(float(a)-float(b));self.max_abs=max(self.max_abs,delta)
   if delta>atol:raise AssertionError(path+' numeric mismatch: '+str(delta))
  else:raise TypeError(type(a))

def check_prelabel(a):
 if sha(a.audit_release)!=a.audit_release_sha256:raise ValueError('Explicit root audit release hash required')
 ar=read(a.audit_release)
 if ar.get('CPU_AUDIT_RELEASE') is not True or ar.get('stage')!='R15_independent_full204_postscore':raise ValueError('No independent post-score audit release')
 for key,path in [('auditor_sha256',Path(__file__)),('score_release_sha256',a.score_release),('scored_seal_sha256',a.scored/'SEALED.json'),('prediction_seal_sha256',a.predictions/'SEALED.json'),('manifest_sha256',a.manifest),('cases_sha256',a.cases),('scoring_index_sha256',a.scoring_index)]:
  if sha(path)!=ar[key]:raise ValueError('Audit root binding '+key)
 sr=read(a.score_release)
 if sr.get('CPU_SCORE_RELEASE') is not True or sr.get('stage')!='R15_unified_full_description_direct8' or sr.get('all_direct_predictions_and_helpers_sealed_before_GT') is not True:raise ValueError('Original score release scope')
 for key in ('prediction_seal_sha256','manifest_sha256','cases_sha256','scoring_index_sha256'):
  if sr[key]!=ar[key]:raise ValueError('Audit and original score release mismatch '+key)
 if sr['source_checkpoint_sha256']!=SOURCE_SHA or sr['geometry_dependency_sha256']!=DEPENDENCIES:raise ValueError('Frozen source/geometry provenance')
 for rel,digest in DEPENDENCIES.items():
  if sha(ROOT/rel)!=digest:raise ValueError('Frozen geometric source changed')
 for label in ('program_bindings','input_bindings','prompt_bindings'):
  if not isinstance(sr[label],dict) or not sr[label]:raise ValueError('Nonempty finite binding '+label)
  for p,digest in sr[label].items():
   if sha(resolve(p))!=digest:raise ValueError('Frozen original binding '+label)
 if sha(ROOT/'research/round15/evaluation/score_direct.py')!=sr['scorer_sha256']:raise ValueError('Actual author scorer binding')
 artifacts={}
 for item in sr['sealed_artifacts']:
  if set(item)!={'role','directory','seal_sha256'} or item['role'] in artifacts:raise ValueError('Artifact role schema')
  base,s=seal(item['directory'],item['seal_sha256']);artifacts[item['role']]=(base,s)
  prod=s.get('program_sha256',{})
  if isinstance(prod,dict):
   for rel,digest in prod.items():
    if sha(resolve(rel))!=digest:raise ValueError('Actual producer code changed')
  elif isinstance(prod,str):
   if prod not in sr['program_bindings'].values():raise ValueError('Mask producer must be explicitly bound')
  else:raise ValueError('Unsupported producer hash schema')
 if len(artifacts)!=10 or set(artifacts)!=set(sr['required_seal_roles']):raise ValueError('Exactly eight captures + semantic + direct')
 _,ps=seal(a.predictions,ar['prediction_seal_sha256']);_,scs=seal(a.scored,ar['scored_seal_sha256'])
 if ps['state']!='sealed_R15_eight_direct_arms_before_GT' or ps['n_samples']!=204 or ps['n_arms']!=8 or tuple(ps['arm_names'])!=ARMS:raise ValueError('Full formal direct204 only')
 if scs['state']!='sealed_R15_postrelease_CPU_scoring' or scs['n_samples']!=204 or scs['release_sha256']!=ar['score_release_sha256'] or scs['prediction_seal_sha256']!=ar['prediction_seal_sha256']:raise ValueError('Actual complete postrelease scores')
 if set(scs['files'])!={'SCORER_LOCK.json','RESULTS_PRIVATE.json','SCALAR_SUMMARY.json'}:raise ValueError('Exact score artifact set')
 if ps['source_checkpoint_sha256']!=SOURCE_SHA:raise ValueError('Source checkpoint identity')
 for key in ('all_direct_no_model_parameter_update_GT_or_new_taskIoU','no_old_G_endpoint_used','graph_geometry_increment_same_associated_anchor_query_gate_support','scene_metadata_rebuilt_matches_actual_semantic_inputs'):
  if ps.get(key) is not True:raise ValueError('Actual direct protocol flag '+key)
 rows=read(a.manifest);cases=read(a.cases);cm={r['id']:r for r in cases}
 if len(rows)!=204 or len(cm)!=204 or len({r['id'] for r in rows})!=204 or set(cm)!={r['id'] for r in rows}:raise ValueError('Complete204 source/case join')
 role_counts=collections.Counter(c['role'] for c in cases)
 if dict(role_counts)!=ROLES or sr['expected_case_role_counts']!=ROLES:raise ValueError('Fresh64/regression140 not mixed')
 populations=collections.Counter((c['role'],c['platform']) for c in cases)
 if populations!={('fresh64_recent_content_relative_only','drone'):32,('fresh64_recent_content_relative_only','quad'):32,('R14_regression140_exposed','drone'):64,('R14_regression140_exposed','quad'):76}:raise ValueError('Frozen platform/cohort denominators')
 if len({c['case_alias'] for c in cases})!=204 or {p.name for p in (a.predictions/'cases_PRIVATE').glob('*.npz')}!={c['case_alias']+'.npz' for c in cases}:raise ValueError('Exact204 NPZ/alias coverage')
 expected={(resolve(c[k])/'SEALED.json').resolve() for c in cases for k in ('source_dir','mask_dir')}
 present={(base/'SEALED.json').resolve() for base,_ in artifacts.values()}
 if len(expected)!=8 or not expected<=present or (a.predictions/'SEALED.json').resolve() not in present:raise ValueError('Actual source/mask/direct closure')
 semantic=[(base,s) for base,s in artifacts.values() if s.get('state')=='sealed_R15_semantic204_three_calls_before_GT']
 if len(semantic)!=1 or semantic[0][1]['n_cases']!=204 or semantic[0][1]['n_model_calls']!=612 or sha(semantic[0][0]/'SEALED.json')!=ps['semantic_seal_sha256']:raise ValueError('Formal204 semantic612 not engineering')
 data={}
 for row in rows:
  sid=row['id'];case=cm[sid]
  if row['platform']!=case['platform'] or row['scene']!=case['scene_PRIVATE']:raise ValueError('Input platform/scene join')
  row['case_role']=case['role'];p=a.predictions/'cases_PRIVATE'/(case['case_alias']+'.npz')
  if ps['files'].get(str(p.relative_to(a.predictions)))!=sha(p):raise ValueError('Actual NPZ membership')
  with np.load(p,allow_pickle=False) as z:d={k:z[k].copy() for k in ('sample_id','arm_names','arm_query_index','arm_boxes_raw6','arm_boxes_eval6','arm_status','source_boxes_raw6','source_text_scores256','q_source','case_role','arm_geometry_anchor_raw6','actual_changed_axes','geometry_increment_axes','arm_support_sha256','arm_mask_sha256','arm_association_accepted')}
  sd=resolve(case['source_dir']);ss=read(sd/'SEALED.json');cp=sd/'source_cache'/(sid+'.npz')
  if not ss['source_unchanged'] or not ss['source_repeat_bitexact'] or ss['source_checkpoint_sha256']!=SOURCE_SHA or ss['files'][str(cp.relative_to(sd))]!=sha(cp):raise ValueError('Immutable original source cache')
  with np.load(cp,allow_pickle=False) as z:
   if not np.array_equal(z['boxes_raw6'][0],d['source_boxes_raw6']) or not np.array_equal(z['text_scores'][0],d['source_text_scores256']) or int(z['query_index'])!=int(d['q_source']):raise ValueError('Original pool/scores/q0 bit bridge')
  raw=d['arm_boxes_raw6'];pool=d['source_boxes_raw6'];q=int(d['q_source']);aq=d['arm_query_index']
  if str(d['sample_id'])!=sid or str(d['case_role'])!=case['role'] or tuple(d['arm_names'])!=ARMS or raw.shape!=(8,6) or pool.shape!=(256,6) or raw.dtype!=np.float32 or pool.dtype!=np.float32 or not np.isfinite(raw).all() or not np.isfinite(pool).all():raise ValueError('Selected/source schema')
  if d['arm_boxes_eval6'].dtype!=np.float32 or d['source_text_scores256'].shape!=(256,) or not np.isfinite(d['source_text_scores256']).all() or d['arm_status'].shape!=(8,):raise ValueError('Native eval/scores/status schema')
  if not np.array_equal(clamp32(raw),d['arm_boxes_eval6']) or q!=int(d['source_text_scores256'].argmax()) or not np.array_equal(raw[0],pool[q]) or aq.shape!=(8,) or not np.issubdtype(aq.dtype,np.integer) or ((aq<0)|(aq>=256)).any():raise ValueError('Native float32 clamp/readout')
  if not np.array_equal(d['arm_geometry_anchor_raw6'],pool[aq]) or not np.array_equal(raw[1],pool[aq[1]]) or not np.array_equal(raw[3],pool[aq[3]]) or not (aq[3]==aq[4]==aq[5]):raise ValueError('Association/increment anchor bridge')
  if not np.array_equal(d['actual_changed_axes'],raw!=raw[0]) or not np.array_equal(d['geometry_increment_axes'],raw!=pool[aq]):raise ValueError('Actual changed flags')
  for j in (2,4,5,6,7):
   if np.all(pool[aq[j],3:]>0) and np.any(raw[j,3:]<pool[aq[j],3:]):raise ValueError('R15 source-or-expansion dimension prior violated')
  for field in ('arm_support_sha256','arm_mask_sha256','arm_association_accepted'):
   if not np.array_equal(d[field][3:6],np.repeat(d[field][3],3)):raise ValueError('Common main q1/support/gate')
  data[sid]=d
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 write(a.output/'AUDIT_PRELABEL_LOCK.json',{'state':'all204_source_direct_semantic_capture_program_hashes_passed_before_GT_index_annotations_or_score_rows','audit_release_sha256':sha(a.audit_release),'score_release_sha256':sha(a.score_release),'prediction_seal_sha256':ar['prediction_seal_sha256'],'scored_seal_sha256':ar['scored_seal_sha256'],'n':204,'cohort_counts':dict(role_counts),'auditor_sha256':sha(__file__),'no_model_weight_payload_or_raw_sensor_decode':True})
 return ar,sr,rows,cm,data

def summary_rows(rows):
 out=[]
 for role in ROLES:
  for domain in ('drone','quad','all_platforms'):
   group=[r for r in rows if r['case_role']==role and (domain=='all_platforms' or r['platform']==domain)]
   for arm in ARMS:
    aa=[r['arms'][arm] for r in group];iv=[r['iou'] for r in aa];ce=[r['coordinate_errors']['center_L2_m'] for r in aa];se=[r['coordinate_errors']['size_L2_m'] for r in aa];ae=np.array([r['coordinate_errors']['abs_error6_m'] for r in aa])
    out.append({'case_role':role,'platform':domain,'arm':arm,'population':'all_cases_in_cohort','n':len(group),'scene_clusters':len({(r['platform'],r['scene']) for r in group}),'mean_iou':mean(iv),'equal_scene_mean_iou':scene_stats(group,iv)['equal_scene_mean'],'Acc25':mean([v>.25 for v in iv]),'Acc50':mean([v>.5 for v in iv]),'equal_scene_Acc25':scene_stats(group,[v>.25 for v in iv])['equal_scene_mean'],'equal_scene_Acc50':scene_stats(group,[v>.5 for v in iv])['equal_scene_mean'],'center_L2_m_mean':mean(ce),'size_L2_m_mean':mean(se),'equal_scene_center_L2_m_mean':scene_stats(group,ce)['equal_scene_mean'],'equal_scene_size_L2_m_mean':scene_stats(group,se)['equal_scene_mean'],'abs_error6_m_mean':ae.mean(0).tolist(),'equal_scene_abs_error6_m_mean':[scene_stats(group,ae[:,j])['equal_scene_mean'] for j in range(6)],'actual_changed_n':sum(a['actual_changed'] for a in aa),'query_changed_n':sum(a['query_changed'] for a in aa),'actual_changed_axis_counts':[sum(a['actual_changed_axes'][j] for a in aa) for j in range(6)],'status_histogram':dict(collections.Counter(a['status'] for a in aa))})
 return out

PAIRS=[(name,'source') for name in ARMS if name!='source']+[('proposal_geo','proposal_assoc'),('graph_assoc','proposal_assoc'),('graph_geo','graph_assoc'),('graph_geo','proposal_geo'),('graph_geo','graph_geo_noRGB'),('graph_geo','graph_geo_relation_disabled'),('graph_geo','graph_geo_relation_flipped_diagnostic')]
def progression(summaries):
 out={}
 for p in ('drone','quad'):
  ss={r['arm']:r for r in summaries if r['case_role']=='fresh64_recent_content_relative_only' and r['platform']==p};g=ss['graph_geo'];checks={}
  if any(ss[a]['n']!=32 for a in ('source','graph_assoc','graph_geo')):raise ValueError('All fresh32/platform denominator required')
  for ref in ('source','graph_assoc'):
   checks[ref+'_micro_iou_positive']=g['mean_iou']-ss[ref]['mean_iou']>1e-8;checks[ref+'_equal_scene_iou_positive']=g['equal_scene_mean_iou']-ss[ref]['equal_scene_mean_iou']>1e-8
   for metric in ('Acc25','Acc50','equal_scene_Acc25','equal_scene_Acc50'):checks[ref+'_'+metric+'_not_decreased']=g[metric]>=ss[ref][metric]
  out[p]={'passed':all(checks.values()),'checks':checks,'n':32,'scene_clusters':g['scene_clusters'],'population':'all_fresh32_no_accepted_or_changed_filter','regression140_does_not_rescue_gate':True,'scene_micro_and_equal_are_not_independent_evidence':True,'next_stage_only_if_root_separate_conditional_TTA_release':True}
 return out

def main(a):
 start=time.perf_counter();ar,sr,manifest,cases,data=check_prelabel(a);checker=Checker();allrows=[];private=[];recomputed_contrasts=[];recomputed_oracles=[];maxpoly=0.;poolmaxerror=0.;fixed=set()
 try:
  # First numerical label/score payload access occurs after AUDIT_PRELABEL_LOCK.
  actual=read(a.scored/'RESULTS_PRIVATE.json');checker.eq(actual['n'],204);observed={r['id']:r for r in actual['rows_PRIVATE']}
  if len(actual['rows_PRIVATE'])!=204 or len(observed)!=204 or set(observed)!=set(data):raise ValueError('All204 scored rows retained')
  checker.eq(actual['case_role_counts'],ROLES);checker.eq(actual['source_checkpoint_sha256'],SOURCE_SHA)
  if set(actual['progression_gate'])!={'drone','quad'}:raise ValueError('Both frozen platform gates only')
  index_rows=read(a.scoring_index);index={r['id']:r for r in index_rows}
  if len(index)!=len(index_rows):raise ValueError('Unique private scoring index')
  for domain in ('drone','quad'):
   selected=sorted((r for r in manifest if r['platform']==domain),key=lambda r:(hashlib.sha256(('R15_independent_pool_check_v1|'+r['id']).encode()).hexdigest(),r['id']))[:3]
   fixed.update(r['id'] for r in selected)
  meta_cache={}
  for row in manifest:
   sid=row['id'];ix=index[sid]
   if ix['platform']!=row['platform'] or ix['scene']!=row['scene']:raise ValueError('Official target index exact join')
   mp=Path(ix['meta_path'])
   if sha(mp)!=ix['metadata_container_sha256']:raise ValueError('Original target metadata hash')
   if str(mp) not in meta_cache:meta_cache[str(mp)]=read(mp)
   rawgt=meta_cache[str(mp)]['ground_info'][ix['object_index']]['bbox_3d'];t=core.transform_gt(rawgt,row['platform'],ix.get('waymo_view_suffix'))
   # Independent D/Q transform math, preserving official float32 target delivery.
   copygt=np.asarray(rawgt).copy()
   if copygt.shape!=(9,) or row['platform'] not in ('drone','quad'):raise ValueError('DQ raw9/official first7 scope')
   if row['platform']=='drone':copygt[2]+=1.8
   if not np.array_equal(t,copygt[:7].astype(np.float32)):raise ValueError('Independent first7/model-coordinate transform bridge')
   d=data[sid];raw=d['arm_boxes_raw6'];values,_=core.candidate_ious(d['arm_boxes_eval6'],t)
   r={'id':sid,'platform':row['platform'],'scene':row['scene'],'case_role':row['case_role'],'q_source':int(d['q_source']),'arms':{}}
   checker.eq({k:r[k] for k in ('id','platform','scene','case_role','q_source')},observed[sid],'row_identity')
   polygon=[]
   for j,name in enumerate(ARMS):
    poly=polygon_yaw0_iou(d['arm_boxes_eval6'][j],t);polygon.append(poly);err=abs(poly-values[j]);maxpoly=max(maxpoly,err)
    if err>POLYGON_IOU_ATOL or (poly>.25)!=(values[j]>.25) or (poly>.5)!=(values[j]>.5):raise AssertionError('Independent polygon/general3D/strict-threshold disagreement')
    e=errors(raw[j],t);change=raw[j]!=raw[0]
    arm={'iou':float(values[j]),'Acc25':bool(values[j]>.25),'Acc50':bool(values[j]>.5),'coordinate_errors':e,'query_index':int(d['arm_query_index'][j]),'query_changed':int(d['arm_query_index'][j])!=int(d['q_source']),'actual_changed':bool(change.any()),'actual_changed_axes':change.tolist(),'raw_nonpositive_sizes':int((raw[j,3:]<=0).sum()),'status':str(d['arm_status'][j]),'center_shift_L2_m':math.dist(raw[j,:3].astype(float),raw[0,:3].astype(float)),'size_shift_L2_m':math.dist(raw[j,3:].astype(float),raw[0,3:].astype(float))}
    checker.eq(arm,observed[sid]['arms'][name],sid+'/'+name);r['arms'][name]=arm
   pool=clamp32(d['source_boxes_raw6']);pv=np.array([polygon_yaw0_iou(b,t) for b in pool]);oo=observed[sid]['offline_source_pool_oracle'];best=float(pv.max());poolmaxerror=max(poolmaxerror,abs(best-oo['max_iou_all256']))
   checker.eq(best,oo['max_iou_all256'],sid+'/oracle_max',POLYGON_IOU_ATOL);checker.eq(bool(best>.25),oo['all256_Acc25']);checker.eq(bool(best>.5),oo['all256_Acc50']);checker.eq(float(pv[int(d['q_source'])]),oo['source_q0_iou'],atol=POLYGON_IOU_ATOL)
   checker.eq(bool(values[0]<=.25 and best>.25),oo['rank_recoverable_25']);checker.eq(bool(values[0]<=.5 and best>.5),oo['rank_recoverable_50'])
   if sid in fixed:
    p3,_=core.candidate_ious(pool,t)
    if np.max(np.abs(pv-p3))>POLYGON_IOU_ATOL:raise AssertionError('Fixed hash6 whole256 independent pool geometry disagreement')
   # CPU retrieval diagnostics stay explicitly non-native. Never replace q_source
   # or a cached native GPU tied ordering with this diagnostic ordering.
   order=sorted(range(256),key=lambda q:(-float(d['source_text_scores256'][q]),q))
   r['oracle_retrieval']={}
   for k in (1,5,10,256):
    z=oo['score_retrieval_'+str(k)];v=float(pv[order[:k]].max());r['oracle_retrieval'][str(k)]=v;checker.eq(v,z['best_iou_GTassisted'],atol=POLYGON_IOU_ATOL);checker.eq(z['native_GPU_rank_claimed'],False);checker.eq(bool(v>.25),z['Acc25']);checker.eq(bool(v>.5),z['Acc50'])
   checker.eq(256,oo['candidate_count']);checker.eq(int(np.sum(d['source_text_scores256']==d['source_text_scores256'].max())),oo['top_score_tie_count'])
   for threshold,label in ((.25,'25'),(.5,'50')):
    good=np.flatnonzero(pv>threshold)
    if len(good):
     strongest=max(float(d['source_text_scores256'][i]) for i in good);interval=[1+int(np.sum(d['source_text_scores256']>strongest)),int(np.sum(d['source_text_scores256']>=strongest))]
    else:interval=[None,None]
    checker.eq(interval,oo['first_threshold_candidate_source_score_rank_interval_'+label])
   r['oracle_values']=pv;r['oracle_record']=oo;allrows.append(r)
   private.append({'case_alias':cases[sid]['case_alias'],'platform':row['platform'],'case_role':row['case_role'],'arms':r['arms'],'independent_polygon_iou':dict(zip(ARMS,map(float,polygon))),'all256_oracle_max_iou':best,'fixed_hash_pool3D_check':sid in fixed})
  summary=summary_rows(allrows);sm={(x['case_role'],x['platform'],x['arm']):x for x in actual['summary']}
  if len(actual['summary'])!=48 or len(sm)!=48:raise ValueError('All2cohorts x3views x8arms summaries')
  for s in summary:checker.eq(s,sm[(s['case_role'],s['platform'],s['arm'])],'summary')
  contrast={(x['case_role'],x['platform'],x['arm'],x['reference']):x for x in actual['paired_contrasts']}
  if len(actual['paired_contrasts'])!=84 or len(contrast)!=84 or len(actual['signed_tails_PRIVATE'])!=84 or len({(x['case_role'],x['platform'],x['arm'],x['reference']) for x in actual['signed_tails_PRIVATE']})!=84:raise ValueError('All84 unique paired contrasts and signed tails')
  if len(actual['offline_source_pool_oracle'])!=6 or len({(x['case_role'],x['platform']) for x in actual['offline_source_pool_oracle']})!=6:raise ValueError('All6 unique source oracle summaries')
  for role in ROLES:
   for domain in ('drone','quad','all_platforms'):
    rr=[r for r in allrows if r['case_role']==role and (domain=='all_platforms' or r['platform']==domain)]
    for arm,ref in PAIRS:
     iv=[r['arms'][arm]['iou']-r['arms'][ref]['iou'] for r in rr];ce=[r['arms'][ref]['coordinate_errors']['center_L2_m']-r['arms'][arm]['coordinate_errors']['center_L2_m'] for r in rr];se=[r['arms'][ref]['coordinate_errors']['size_L2_m']-r['arms'][arm]['coordinate_errors']['size_L2_m'] for r in rr]
     ae=np.array([np.array(r['arms'][ref]['coordinate_errors']['abs_error6_m'])-r['arms'][arm]['coordinate_errors']['abs_error6_m'] for r in rr])
     expect={'n':len(rr),'IoU_gain':signed(iv,1e-8),'center_error_reduction_m':signed(ce,1e-6),'size_error_reduction_m':signed(se,1e-6),'abs6_error_reduction_m':[signed(ae[:,j],1e-6) for j in range(6)],'IoU_scene':scene_stats(rr,iv),'center_scene':scene_stats(rr,ce),'size_scene':scene_stats(rr,se)}
     checker.eq(expect,contrast[(role,domain,arm,ref)],'paired');recomputed_contrasts.append({'case_role':role,'platform':domain,'arm':arm,'reference':ref,**expect})
     tail=next(x for x in actual['signed_tails_PRIVATE'] if x['case_role']==role and x['platform']==domain and x['arm']==arm and x['reference']==ref)
     ordered=sorted(zip(rr,iv),key=lambda z:(z[1],z[0]['id']))
     checker.eq([{'id':r['id'],'delta_iou':v} for r,v in ordered[:5]],tail['most_negative_PRIVATE'],'tails_negative');checker.eq([{'id':r['id'],'delta_iou':v} for r,v in ordered[-5:][::-1]],tail['most_positive_PRIVATE'],'tails_positive')
    os=next(x for x in actual['offline_source_pool_oracle'] if x['case_role']==role and x['platform']==domain)
    bests=[float(r['oracle_values'].max()) for r in rr]
    expected={'n':len(rr),'max_all256_iou_mean':mean(bests),'equal_scene_max_all256_iou_mean':scene_stats(rr,bests)['equal_scene_mean'],'all256_Acc25':mean([v>.25 for v in bests]),'all256_Acc50':mean([v>.5 for v in bests]),'equal_scene_all256_Acc25':scene_stats(rr,[v>.25 for v in bests])['equal_scene_mean'],'equal_scene_all256_Acc50':scene_stats(rr,[v>.5 for v in bests])['equal_scene_mean']}
    expected['rank_recoverable25_n']=int(sum(r['arms']['source']['iou']<=.25 and r['oracle_values'].max()>.25 for r in rr));expected['rank_recoverable50_n']=int(sum(r['arms']['source']['iou']<=.5 and r['oracle_values'].max()>.5 for r in rr))
    expected['first_qualified_source_rank25_defined_n']=int(sum(r['oracle_values'].max()>.25 for r in rr));expected['first_qualified_source_rank50_defined_n']=int(sum(r['oracle_values'].max()>.5 for r in rr))
    expected['CPU_stable_score_retrieval_diagnostic']={str(k):{'best_iou_within_k_mean_GTassisted':mean([r['oracle_retrieval'][str(k)] for r in rr]),'Acc25':mean([r['oracle_retrieval'][str(k)]>.25 for r in rr]),'Acc50':mean([r['oracle_retrieval'][str(k)]>.5 for r in rr]),'native_GPU_rank_claimed':False} for k in (1,5,10,256)}
    checker.eq(expected,os,'oracle_summary',POLYGON_IOU_ATOL);recomputed_oracles.append({'case_role':role,'platform':domain,**expected})
  gate=progression(summary);checker.eq(gate,actual['progression_gate'],'progression')
  write(a.output/'CASES_PRIVATE.json',private)
  write(a.output/'RECOMPUTED_TABLES.json',{'summary':summary,'paired_contrasts':recomputed_contrasts,'source_pool_oracle':recomputed_oracles})
  report={'state':'PASS_R15_INDEPENDENT_FULL204_ROW_GEOMETRY_REDUCTIONS_AND_GATE_AUDIT','n_cases':204,'n_selected_arms_checked':1632,'independent_polygon_pool_boxes_checked':52224,'fixed_input_hash_cases_full256_reference3D_checked':len(fixed),'checks':checker.n,'max_author_scalar_difference':checker.max_abs,'max_independent_polygon_vs_reference_selected_iou_difference':maxpoly,'max_independent_pool_oracle_difference':poolmaxerror,'all204_zero_NA_and_negative_cases_retained':True,'cohorts_separate':ROLES,'all48_micro_sceneequal_summary_rows_checked':True,'all_paired_signed_tails_and_progression_gates_checked':True,'no_native_GPU_tie_replacement_CPU_retrieval_only_explicit_diagnostic':True,'fresh_progression_gate':gate,'source_checkpoint_sha256':SOURCE_SHA,'auditor_sha256':sha(__file__),'score_release_sha256':ar['score_release_sha256'],'scored_seal_sha256':ar['scored_seal_sha256'],'prediction_seal_sha256':ar['prediction_seal_sha256'],'new_GT_access_only_after_actual_root_audit_release_and_prelabel_lock':True,'no_model_GPU_new_predictions_method_or_threshold_changes':True,'seconds':time.perf_counter()-start}
  write(a.output/'SUMMARY.json',report);write(a.output/'SEALED.json',{'state':'sealed_R15_independent_full204_postscore_audit','files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'audit_release_sha256':sha(a.audit_release),'scored_seal_sha256':ar['scored_seal_sha256'],'prediction_seal_sha256':ar['prediction_seal_sha256'],'auditor_sha256':sha(__file__)})
  print(json.dumps({'state':report['state'],'checks':checker.n,'seal_sha256':sha(a.output/'SEALED.json')}))
 except Exception as exc:
  write(a.output/'FAILURE.json',{'state':'failed_independent_R15_audit_preserved','error':str(exc),'traceback_PRIVATE':traceback.format_exc(),'no_tolerance_or_method_change':True});raise

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('predictions','scored','manifest','cases','scoring-index','score-release','audit-release','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--audit-release-sha256',required=True);main(p.parse_args())
