"""R13 post-complete-seal CPU task score; no GT access before integrity gate.

Nine direct arms, new-content/regression cohorts kept separate. Selected boxes
use native yaw0 against transformed official7D targets, float64 general3D IoU.
Pool oracle is offline label-assisted coverage, never a deployed selection arm.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import candidate_ious,transform_gt,validate_official_clamp
_spec=importlib.util.spec_from_file_location('R13_reused_R12_metrics',ROOT/'research/round12/evaluation/metrics.py')
_metrics=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_metrics)
coordinate_errors=_metrics.coordinate_errors
signed_stats=_metrics.signed_stats
def paired_scene(rows,values):
 # Scene names need not be globally unique across platforms.
 namespaced=[dict(r,scene=(r['platform'],r['scene'])) for r in rows]
 return _metrics.paired_scene(namespaced,values)
mean=_metrics.mean
ARMS=('source','A_full_consensus','A_full_order0','A_noun_consensus',
      'A_deranged_consensus','B_inequality_full','B_inequality_noRGB',
      'B_inequality_half','B_old_surface_equality')
A_ARMS=ARMS[1:5]
B_ARMS=ARMS[5:]
DEPENDENCIES={
 'research/round12/evaluation/metrics.py':'4f696eecb3a7b46e237dc13cb5354a512e34f10799590b41bce579f8c04ca81f',
 'research/round5/audit/independent_core_v2.py':'b4937fdd89f719fe1d9560f770c01597ff6807c74a75013720b470c43393379a',
 'tools/obb_iou.py':'45896d29d1f95167b95a27b044fcc06b4e6fe5f9cb98749096c2f707292d4083'}
REQUIRED_NPZ=('sample_id','arm_names','arm_query_index','arm_boxes_raw6','arm_boxes_eval6',
              'arm_status','source_boxes_raw6','source_text_scores256','q_source','case_role')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def member(base,name):
 p=Path(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe seal member')
 result=(base/p).resolve()
 if not result.is_relative_to(base.resolve()):raise ValueError('Seal member escaped root')
 return result

def bind_map(mapping,label):
 if not isinstance(mapping,dict) or not mapping:raise ValueError(label+' must be nonempty path->SHA mapping')
 for path,digest in mapping.items():
  p=Path(path);p=p if p.is_absolute() else ROOT/p
  if not isinstance(digest,str) or len(digest)!=64 or sha(p)!=digest:raise ValueError(label+' mismatch: '+str(path))

def verify_artifact(artifact,program_bindings):
 if set(artifact)!={'role','directory','seal_sha256'}:raise ValueError('Sealed artifact schema')
 base=Path(artifact['directory']);base=base if base.is_absolute() else ROOT/base
 sp=base/'SEALED.json'
 if sha(sp)!=artifact['seal_sha256']:raise ValueError('Artifact seal mismatch: '+artifact['role'])
 seal=read(sp)
 if not isinstance(seal.get('files'),dict) or not seal['files']:raise ValueError('Empty artifact file allowlist')
 for rel,digest in seal['files'].items():
  if sha(member(base,rel))!=digest:raise ValueError('Artifact member mismatch: '+rel)
 producer=seal.get('program_sha256',{})
 if isinstance(producer,dict):
  for rel,digest in producer.items():
   if sha(member(ROOT,rel))!=digest:raise ValueError('Artifact program mismatch: '+rel)
 elif isinstance(producer,str):
  # Whole-image mask captures store a single producer SHA rather than a map.
  # Its corresponding source path must already be explicit and byte-verified
  # in the release. Never infer a filename or silently skip this proof.
  if len(producer)!=64 or producer not in program_bindings.values():
   raise ValueError('Artifact string producer missing from explicit verified program bindings')
 else:
  raise ValueError('Artifact program_sha256 must be path->SHA dictionary or explicit producer SHA string')
 return base,seal

def eval_boxes(raw):
 result=np.array(raw,copy=True)
 result[...,3:6]=np.maximum(result[...,3:6],np.asarray(1e-6,dtype=result.dtype))
 return result

def unlabelled_gate(a):
 if sha(a.release)!=a.release_sha256:raise ValueError('Release SHA mismatch')
 release=read(a.release)
 if not release.get('CPU_SCORE_RELEASE') or release.get('stage')!='R13_association_vs_inequality_geometry_direct':
  raise ValueError('No R13 CPU score release')
 if not release.get('all_A_B_predictions_and_helper_outputs_sealed_before_GT'):raise ValueError('Incomplete prediction seal')
 for key,path in [('prediction_seal_sha256',a.predictions/'SEALED.json'),
                  ('manifest_sha256',a.manifest),('cases_sha256',a.cases),('scoring_index_sha256',a.scoring_index),
                  ('scorer_sha256',Path(__file__))]:
  if sha(path)!=release[key]:raise ValueError('Binding mismatch: '+key)
 for path,digest in DEPENDENCIES.items():
  if sha(ROOT/path)!=digest:raise ValueError('Frozen geometry dependency mismatch')
 if release.get('geometry_dependency_sha256')!=DEPENDENCIES:raise ValueError('Release must bind exact transform/metrics/OBB dependencies')
 checkpoint=Path(release['source_checkpoint_path'])
 if sha(checkpoint)!=release['source_checkpoint_sha256']:raise ValueError('Source checkpoint mismatch')
 for key in ('program_bindings','input_bindings','prompt_bindings'):bind_map(release[key],key)
 if not isinstance(release['sealed_artifacts'],list) or not release['sealed_artifacts']:raise ValueError('All source/helper artifacts required')
 artifacts={}
 for art in release['sealed_artifacts']:
  if art['role'] in artifacts:raise ValueError('Duplicate sealed artifact role')
  artifacts[art['role']]=verify_artifact(art,release['program_bindings'])
  asource=artifacts[art['role']][1].get('source_checkpoint_sha256')
  if asource is not None and asource!=release['source_checkpoint_sha256']:raise ValueError('Artifact source checkpoint mismatch')
 if set(release['required_seal_roles'])!=set(artifacts):raise ValueError('Required source/helper artifact role closure mismatch')
 if not any(base.resolve()==a.predictions.resolve() for base,_ in artifacts.values()):raise ValueError('Composite not among verified artifacts')
 seal=read(a.predictions/'SEALED.json')
 if tuple(seal['arm_names'])!=ARMS or seal['n_arms']!=len(ARMS):raise ValueError('Composite arm protocol mismatch')
 if seal['source_checkpoint_sha256']!=release['source_checkpoint_sha256']:raise ValueError('Composite source mismatch')
 if seal['manifest_sha256']!=sha(a.manifest):raise ValueError('Composite manifest mismatch')
 manifest=read(a.manifest)
 if not isinstance(manifest,list) or not manifest:raise ValueError('Nonempty composite manifest list required')
 ids=[str(row['id']) for row in manifest]
 if len(set(ids))!=len(ids) or len(ids)!=seal['n_samples']:raise ValueError('Manifest count/duplicate mismatch')
 case_rows=read(a.cases)
 if not isinstance(case_rows,list) or len(case_rows)!=len(manifest):raise ValueError('Input-only cases list count mismatch')
 cases={str(r['id']):r for r in case_rows}
 if len(cases)!=len(case_rows) or set(cases)!=set(ids):raise ValueError('Input-only cases duplicate/join mismatch')
 role_map={'new_content':'new_content108','exposed_R12_all_changed_regression':'regression8'}
 for row in manifest:
  c=cases[str(row['id'])]
  if row['platform']!=c['platform'] or row['scene']!=c['scene_PRIVATE']:raise ValueError('Input-only case platform/scene mismatch')
  role=role_map[c['role']]
  if 'case_role' in row and row['case_role']!=role:raise ValueError('Optional manifest role mismatch')
  row['case_role']=role
 expected_roles=release['expected_case_role_counts']
 counts=dict(collections.Counter(row['case_role'] for row in manifest))
 if counts!=expected_roles:raise ValueError('Frozen cohort counts mismatch')
 if {p.name for p in a.predictions.glob('*.npz')}!={sid+'.npz' for sid in ids}:raise ValueError('NPZ sample coverage mismatch')
 predictions={}
 for row in manifest:
  sid=str(row['id']);path=member(a.predictions,sid+'.npz')
  if path.name not in seal['files'] or sha(path)!=seal['files'][path.name]:raise ValueError('Prediction not in composite seal')
  with np.load(path,allow_pickle=False) as z:
   if not set(REQUIRED_NPZ)<=set(z.files):raise ValueError('Missing composite NPZ fields')
   d={k:z[k].copy() for k in REQUIRED_NPZ}
  case=cases[sid];source_dir=Path(case['source_dir']);source_dir=source_dir if source_dir.is_absolute() else ROOT/source_dir
  if source_dir.resolve() not in {base.resolve() for base,_ in artifacts.values()}:raise ValueError('Per-case source directory not in verified artifacts')
  ss=read(source_dir/'SEALED.json');cp=source_dir/'source_cache'/(sid+'.npz')
  if not ss['source_unchanged'] or not ss['source_repeat_bitexact']:raise ValueError('Source frozen/repeat evidence missing')
  if ss['files'][str(cp.relative_to(source_dir))]!=sha(cp):raise ValueError('Source cache is not sealed')
  with np.load(cp,allow_pickle=False) as z:
   source_cache={k:z[k].copy() for k in ('sample_id','boxes_raw6','text_scores','query_index')}
  if str(source_cache['sample_id'].item())!=sid or source_cache['boxes_raw6'].shape!=(1,256,6) or source_cache['text_scores'].shape!=(1,256):raise ValueError('Source cache sample/schema mismatch')
  if not np.array_equal(d['source_boxes_raw6'],source_cache['boxes_raw6'][0]) or not np.array_equal(d['source_text_scores256'],source_cache['text_scores'][0]) or int(d['q_source'])!=int(source_cache['query_index']):raise ValueError('Composite pool/scores/q0 differ from immutable source cache')
  if str(d['sample_id'].item())!=sid or str(d['case_role'].item())!=row['case_role']:raise ValueError('Sample/cohort scalar mismatch')
  if tuple(d['arm_names'].tolist())!=ARMS or d['arm_status'].shape!=(len(ARMS),):raise ValueError('NPZ arm mismatch')
  raw=d['arm_boxes_raw6'];pool=d['source_boxes_raw6'];scores=d['source_text_scores256']
  if raw.shape!=(len(ARMS),6) or pool.shape!=(256,6) or scores.shape!=(256,):raise ValueError('Raw pool/selected shape mismatch')
  if raw.dtype!=np.float32 or pool.dtype!=np.float32 or not all(np.isfinite(x).all() for x in (raw,pool,scores)):
   raise ValueError('Finite native float32 raw boxes and finite scores required')
  validate_official_clamp(raw,d['arm_boxes_eval6'])
  q=int(d['q_source'].item());aq=d['arm_query_index']
  if q!=int(np.argmax(scores)) or aq.shape!=(len(ARMS),) or not np.issubdtype(aq.dtype,np.integer) or np.any((aq<0)|(aq>=256)):
   raise ValueError('Query/readout mismatch')
  if int(aq[0])!=q or not np.array_equal(raw[0],pool[q]):raise ValueError('Source q0 box mismatch')
  for j in range(1,5):
   if not np.array_equal(raw[j],pool[int(aq[j])]):raise ValueError('A must select unchanged original source proposal')
  if not np.array_equal(aq[5:],np.full(4,q)):raise ValueError('B must keep q0')
  if not np.array_equal(raw[7],raw[0]+np.float32(.5)*(raw[5]-raw[0])):raise ValueError('Half interpolation mismatch')
  for j in (5,6,7):
   if np.all(raw[0,3:]>0) and np.any(raw[j,3:]<raw[0,3:]):raise ValueError('Positive-only B may not shrink source sizes')
  predictions[sid]=d
 # No GT/index payload/annotation read has occurred. Hashing index bytes above is
 # only integrity verification. Write this durable boundary before label IO.
 a.output.mkdir(parents=True,exist_ok=False)
 dump(a.output/'SCORER_LOCK.json',{'state':'all_composite_source_helper_prompt_program_hashes_and_unlabelled_contracts_passed_before_GT',
      'prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'release_sha256':sha(a.release),
      'source_checkpoint_sha256':release['source_checkpoint_sha256'],'n_samples':len(manifest),
      'case_role_counts':counts,'geometry_dependency_sha256':DEPENDENCIES,
      'no_literal_identity_diagnostic':True,'nativeyaw0_float64_general3D_geometry':True})
 return release,manifest,predictions


def pool_oracle(pool,scores,target,source_q):
 values,cost=candidate_ious(eval_boxes(pool),target)
 order=np.lexsort((np.arange(256),-scores))
 best=float(values.max());best_ids=np.flatnonzero(values==best)
 out={'max_iou_all256':best,'all256_Acc25':bool(best>.25),'all256_Acc50':bool(best>.5),
      'source_q0_iou':float(values[source_q]),'source_q0_index':int(source_q),
      'candidate_count':256,'best_candidate_indices_GTassisted_PRIVATE':best_ids.tolist(),
      'scope':'offline GT-assisted upper bound; not method selection or literal object identity',
      'retrieval_tie_rule':'CPU diagnostic score descending/index ascending, NOT native GPU ties',
      'top_score_tie_count':int(np.sum(scores==scores.max()))}
 for k in (1,5,10,256):
  v=float(values[order[:k]].max())
  out['score_retrieval_'+str(k)]={'best_iou_GTassisted':v,'Acc25':bool(v>.25),'Acc50':bool(v>.5),
      'GT_selection_within_pool':k>1,'native_GPU_rank_claimed':False}
 for threshold,label in ((.25,'25'),(.5,'50')):
  good=np.flatnonzero(values>threshold)
  if len(good):
   strongest=float(scores[good].max())
   lower=1+int(np.sum(scores>strongest));upper=int(np.sum(scores>=strongest))
  else:lower=upper=None
  out['first_threshold_candidate_source_score_rank_interval_'+label]=[lower,upper]
  out['rank_recoverable_'+label]=bool(values[source_q]<=threshold and best>threshold)
 return out,cost


def evaluate_row(row,data,target):
 raw=data['arm_boxes_raw6'];eval6=data['arm_boxes_eval6']
 ious,cost=candidate_ious(eval6,target)
 result={'id':row['id'],'platform':row['platform'],'scene':row['scene'],
         'case_role':row['case_role'],'q_source':int(data['q_source']), 'arms':{}}
 for j,name in enumerate(ARMS):
  changes=raw[j]!=raw[0]
  result['arms'][name]={'iou':float(ious[j]),'Acc25':bool(ious[j]>.25),'Acc50':bool(ious[j]>.5),
      'coordinate_errors':coordinate_errors(raw[j],target),'query_index':int(data['arm_query_index'][j]),
      'query_changed':int(data['arm_query_index'][j])!=int(data['q_source']),
      'actual_changed':bool(changes.any()),'actual_changed_axes':changes.tolist(),
      'raw_nonpositive_sizes':int((raw[j,3:]<=0).sum()),'status':str(data['arm_status'][j]),
      'center_shift_L2_m':float(np.linalg.norm(raw[j,:3].astype(float)-raw[0,:3].astype(float))),
      'size_shift_L2_m':float(np.linalg.norm(raw[j,3:].astype(float)-raw[0,3:].astype(float)))}
 oracle,ocost=pool_oracle(data['source_boxes_raw6'],data['source_text_scores256'],target,int(data['q_source']))
 result['offline_source_pool_oracle']=oracle;cost=collections.Counter(cost);cost.update(ocost)
 return result,cost


def summarize(rows,roles):
 summaries=[];contrasts=[];oracle_summaries=[];tails=[]
 pairs=[(name,'source') for name in ARMS if name!='source']+[
   ('A_full_consensus','A_full_order0'),('A_full_consensus','A_noun_consensus'),
   ('A_full_consensus','A_deranged_consensus'),('B_inequality_full','B_inequality_noRGB'),
   ('B_inequality_full','B_inequality_half'),('B_inequality_full','B_old_surface_equality')]
 for role in roles:
  for platform in ('drone','quad','all_platforms'):
   group=[r for r in rows if r['case_role']==role and (platform=='all_platforms' or r['platform']==platform)]
   if not group:continue
   for name in ARMS:
    arm=[r['arms'][name] for r in group];ious=[x['iou'] for x in arm]
    ce=[x['coordinate_errors']['center_L2_m'] for x in arm];se=[x['coordinate_errors']['size_L2_m'] for x in arm]
    abs6=np.array([x['coordinate_errors']['abs_error6_m'] for x in arm])
    summaries.append({'case_role':role,'platform':platform,'arm':name,'population':'all_cases_in_cohort',
      'n':len(group),'scene_clusters':len({(r['platform'],r['scene']) for r in group}),
      'mean_iou':mean(ious),'equal_scene_mean_iou':paired_scene(group,ious)['equal_scene_mean'],
      'Acc25':mean([float(x>.25) for x in ious]),'Acc50':mean([float(x>.5) for x in ious]),
      'equal_scene_Acc25':paired_scene(group,[float(x>.25) for x in ious])['equal_scene_mean'],
      'equal_scene_Acc50':paired_scene(group,[float(x>.5) for x in ious])['equal_scene_mean'],
      'center_L2_m_mean':mean(ce),'size_L2_m_mean':mean(se),
      'equal_scene_center_L2_m_mean':paired_scene(group,ce)['equal_scene_mean'],
      'equal_scene_size_L2_m_mean':paired_scene(group,se)['equal_scene_mean'],
      'abs_error6_m_mean':abs6.mean(0).tolist(),
      'equal_scene_abs_error6_m_mean':[paired_scene(group,abs6[:,j])['equal_scene_mean'] for j in range(6)],
      'actual_changed_n':sum(x['actual_changed'] for x in arm),
      'query_changed_n':sum(x['query_changed'] for x in arm),
      'actual_changed_axis_counts':[sum(x['actual_changed_axes'][j] for x in arm) for j in range(6)],
      'status_histogram':dict(collections.Counter(x['status'] for x in arm))})
   for name,ref in pairs:
    delta=[r['arms'][name]['iou']-r['arms'][ref]['iou'] for r in group]
    ce=[r['arms'][ref]['coordinate_errors']['center_L2_m']-r['arms'][name]['coordinate_errors']['center_L2_m'] for r in group]
    se=[r['arms'][ref]['coordinate_errors']['size_L2_m']-r['arms'][name]['coordinate_errors']['size_L2_m'] for r in group]
    ae=np.array([np.array(r['arms'][ref]['coordinate_errors']['abs_error6_m'])-r['arms'][name]['coordinate_errors']['abs_error6_m'] for r in group])
    contrasts.append({'case_role':role,'platform':platform,'arm':name,'reference':ref,'n':len(group),
      'IoU_gain':signed_stats(delta,1e-8),'center_error_reduction_m':signed_stats(ce,1e-6),
      'size_error_reduction_m':signed_stats(se,1e-6),
      'abs6_error_reduction_m':[signed_stats(ae[:,j],1e-6) for j in range(6)],
      'IoU_scene':paired_scene(group,delta),'center_scene':paired_scene(group,ce),'size_scene':paired_scene(group,se),
      'positive_means_improvement':True})
    ordered=sorted(zip(group,delta),key=lambda x:(x[1],x[0]['id']))
    tails.append({'case_role':role,'platform':platform,'arm':name,'reference':ref,
      'most_negative_PRIVATE':[{'id':r['id'],'delta_iou':v} for r,v in ordered[:5]],
      'most_positive_PRIVATE':[{'id':r['id'],'delta_iou':v} for r,v in ordered[-5:][::-1]],
      'selection':'fixed smallest/largest signed delta, ID ties; may include zero if no positive/negative'})
   oo=[r['offline_source_pool_oracle'] for r in group]
   oracle_summaries.append({'case_role':role,'platform':platform,'n':len(group),
      'max_all256_iou_mean':mean([x['max_iou_all256'] for x in oo]),
      'equal_scene_max_all256_iou_mean':paired_scene(group,[x['max_iou_all256'] for x in oo])['equal_scene_mean'],
      'all256_Acc25':mean([float(x['all256_Acc25']) for x in oo]),
      'all256_Acc50':mean([float(x['all256_Acc50']) for x in oo]),
      'equal_scene_all256_Acc25':paired_scene(group,[float(x['all256_Acc25']) for x in oo])['equal_scene_mean'],
      'equal_scene_all256_Acc50':paired_scene(group,[float(x['all256_Acc50']) for x in oo])['equal_scene_mean'],
      'rank_recoverable25_n':sum(x['rank_recoverable_25'] for x in oo),
      'rank_recoverable50_n':sum(x['rank_recoverable_50'] for x in oo),
      'CPU_stable_score_retrieval_diagnostic':{
          str(k):{'best_iou_within_k_mean_GTassisted':mean([x['score_retrieval_'+str(k)]['best_iou_GTassisted'] for x in oo]),
                  'Acc25':mean([float(x['score_retrieval_'+str(k)]['Acc25']) for x in oo]),
                  'Acc50':mean([float(x['score_retrieval_'+str(k)]['Acc50']) for x in oo]),
                  'native_GPU_rank_claimed':False} for k in (1,5,10,256)},
      'first_qualified_source_rank25_defined_n':sum(x['first_threshold_candidate_source_score_rank_interval_25'][0] is not None for x in oo),
      'first_qualified_source_rank50_defined_n':sum(x['first_threshold_candidate_source_score_rank_interval_50'][0] is not None for x in oo),
      'scope':'offline GT-assisted source-pool coverage; neither a method nor instance-identity labels'})
 return summaries,contrasts,oracle_summaries,tails


def main(a):
 start=time.perf_counter()
 release,manifest,predictions=unlabelled_gate(a)
 # LABEL BOUNDARY: all composite/source/helper/input/program integrity checks and
 # source readout/selection/fixed-q0/half checks completed above, lock saved.
 raw_index=read(a.scoring_index)
 if not isinstance(raw_index,list):raise ValueError('Private R5 index must be a list')
 index={str(row['id']):row for row in raw_index}
 if len(index)!=len(raw_index):raise ValueError('Duplicate scoring index IDs')
 rows=[];metadata_cache={};geometry_counts=collections.Counter()
 for sample in manifest:
  sid=str(sample['id']);ix=index[sid]
  if ix['platform']!=sample['platform'] or ix['scene']!=sample['scene']:raise ValueError('Private index join mismatch')
  mp=Path(ix['meta_path'])
  if sha(mp)!=ix['metadata_container_sha256']:raise ValueError('Original annotation SHA mismatch')
  if str(mp) not in metadata_cache:metadata_cache[str(mp)]=read(mp)
  raw_box=metadata_cache[str(mp)]['ground_info'][ix['object_index']]['bbox_3d']
  target=transform_gt(raw_box,sample['platform'],ix.get('waymo_view_suffix'))
  row,cost=evaluate_row(sample,predictions[sid],target);rows.append(row);geometry_counts.update(cost)
 summary,contrasts,oracles,tails=summarize(rows,list(release['expected_case_role_counts']))
 result={'state':'R13_nine_direct_arms_scored_only_after_complete_seal','n':len(rows),
    'case_role_counts':dict(collections.Counter(r['case_role'] for r in rows)),
    'summary':summary,'paired_contrasts':contrasts,'offline_source_pool_oracle':oracles,
    'rows_PRIVATE':rows,'signed_tails_PRIVATE':tails,'geometry_counts':dict(geometry_counts),
    'seconds':time.perf_counter()-start,'source_checkpoint_sha256':release['source_checkpoint_sha256'],
    'scope':'New-content and exposed regression cohorts remain separate; fixed-q0 B vs original-pool selection A. Nativeyaw0, official source-size clamp only for taskIoU, strict IoU>.25/.5, float64 general3D geometry; no literal identity GT diagnosis, no neural TTA. Oracle uses GT only after seal.'}
 dump(a.output/'RESULTS_PRIVATE.json',result)
 dump(a.output/'SCALAR_SUMMARY.json',{k:v for k,v in result.items() if not k.endswith('_PRIVATE')})
 dump(a.output/'SEALED.json',{'state':'sealed_R13_postrelease_CPU_scoring',
   'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'n_samples':len(rows),
   'prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'release_sha256':sha(a.release),
   'scorer_sha256':sha(Path(__file__))})
 print(json.dumps({'n':len(rows),'case_role_counts':result['case_role_counts'],'seconds':result['seconds']}))


if __name__=='__main__':
 parser=argparse.ArgumentParser()
 for name in ('predictions','manifest','cases','scoring-index','release','output'):
  parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--release-sha256',required=True)
 main(parser.parse_args())
