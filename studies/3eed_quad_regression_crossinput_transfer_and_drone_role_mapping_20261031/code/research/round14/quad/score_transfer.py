"""R14 Quad seven-arm score, blocked until BOTH branches' explicit GT release.

Only B labels after every bound source/A pseudo/Quad/Drone prediction seal passes.
No A labels, new predictions, identity judgments, model/GPU or method tuning.
"""
from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import transform_gt,candidate_ious,validate_official_clamp
_spec=importlib.util.spec_from_file_location('R14_quad_frozen_R12_metrics',ROOT/'research/round12/evaluation/metrics.py')
_metrics=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_metrics)
mean=_metrics.mean;coordinate_errors=_metrics.coordinate_errors;signed_stats=_metrics.signed_stats
ARMS=('source','direct_additive_Aresidual','direct_size_normalized_Aresidual','parameter_half','kernel_half','parameter_full','kernel_full')
DRONE_ARMS=('source','old_full_order0','old_full_consensus','old_noun_consensus','semantic_only','mapping_only','combined')
SOURCE_SHA='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
DEPENDENCIES={
 'research/round5/audit/independent_core_v2.py':'b4937fdd89f719fe1d9560f770c01597ff6807c74a75013720b470c43393379a',
 'research/round12/evaluation/metrics.py':'4f696eecb3a7b46e237dc13cb5354a512e34f10799590b41bce579f8c04ca81f',
 'tools/obb_iou.py':'45896d29d1f95167b95a27b044fcc06b4e6fe5f9cb98749096c2f707292d4083'}
NPZ_KEYS=('pair_id','A_id','B_id','arm_names','arm_query_index','arm_boxes_raw6','arm_boxes_eval6','arm_status',
  'source_boxes_raw6','source_text_scores256','q_source','A_q_source','A_source6','A_target6','A_frozen_G_status',
  'A_gate','A_nonzero_residual','A_source_cache_sha256','B_source_cache_sha256','A_geometry_sha256',
  'B_point_sample_sha256','B_raw_indices_sha256')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def bound(path,digest):
 if sha(path)!=digest:raise ValueError('Hash mismatch '+str(path))
def member(directory,name):
 p=Path(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe seal member')
 path=(directory/p).resolve()
 if not path.is_relative_to(directory.resolve()):raise ValueError('Escaping member')
 return path

def bind_map(mapping):
 if not isinstance(mapping,dict) or not mapping:raise ValueError('Nonempty explicit program bindings required')
 for rel,h in mapping.items():
  p=Path(rel);bound(p if p.is_absolute() else ROOT/p,h)

def verify_artifact(item,bindings):
 if set(item)!={'directory','seal_sha256'}:raise ValueError('Generic artifact strict directory/seal_sha256 required')
 base=Path(item['directory']);bound(base/'SEALED.json',item['seal_sha256']);seal=read(base/'SEALED.json')
 if not isinstance(seal.get('files'),dict) or not seal['files']:raise ValueError('Nonempty sealed files required')
 for name,h in seal['files'].items():bound(member(base,name),h)
 programs=seal.get('program_sha256',{})
 if isinstance(programs,dict):
  for rel,h in programs.items():bound(member(ROOT,rel),h)
 elif isinstance(programs,str):
  if programs not in bindings.values():raise ValueError('String producer not explicitly byte-bound')
 else:raise ValueError('Producer schema')
 if seal.get('source_checkpoint_sha256',SOURCE_SHA)!=SOURCE_SHA:raise ValueError('Source provenance mismatch')
 return base.resolve(),seal


def validate_drone_final(seal,expected_n):
 # Exact actual sibling-branch schema; prefixes/partial generic counts ineligible.
 if expected_n!=64 or seal.get('state')!='sealed_R14_Drone64_seven_direct_arms_before_GT' or seal.get('n_cases')!=expected_n:
  raise ValueError('Drone final64 coverage/state')
 if tuple(seal.get('arm_names',()))!=DRONE_ARMS or not seal.get('no_parameter_update_GT_or_newbox') or not seal.get('model_state_unchanged'):
  raise ValueError('Drone seven-arm frozen direct-output contract')

def prelabel_gate(a):
 bound(a.release,a.release_sha256);r=read(a.release)
 if not r.get('CPU_SCORE_RELEASE') or not r.get('both_branches_complete_and_sealed') or r.get('stage')!='R14_both_branches_complete_before_target_GT':raise ValueError('No BOTH-branches target GT release')
 for key,path in [('quad_prediction_seal_sha256',a.predictions/'SEALED.json'),('pairs_sha256',a.pairs),
     ('B_manifest_sha256',a.B_manifest),('scoring_index_sha256',a.scoring_index),('quad_scorer_sha256',Path(__file__))]:bound(path,r[key])
 drone=Path(r['drone_prediction_dir']);bound(drone/'SEALED.json',r['drone_prediction_seal_sha256'])
 if r['source_checkpoint_sha256']!=SOURCE_SHA or r['geometry_dependency_sha256']!=DEPENDENCIES:raise ValueError('Source/geometry dependencies')
 bound(r['source_checkpoint_path'],SOURCE_SHA)
 for rel,h in DEPENDENCIES.items():bound(ROOT/rel,h)
 bind_map(r['program_bindings'])
 artifacts={}
 for item in r['sealed_artifacts']:
  base,seal=verify_artifact(item,r['program_bindings'])
  if base in artifacts:raise ValueError('Duplicate artifact directory')
  artifacts[base]=seal
 if a.predictions.resolve() not in artifacts or drone.resolve() not in artifacts:raise ValueError('Both branch final artifacts required')
 seal=artifacts[a.predictions.resolve()];ds=artifacts[drone.resolve()]
 if seal['state']!='sealed_R14_Quad76_Aonly_transfer7arms_before_B_GT' or seal['n_pairs']!=76 or seal['n_arms']!=7 or tuple(seal['arm_names'])!=ARMS:raise ValueError('Formal76 seven-arm seal required, engineering prefix is ineligible')
 if not seal['source_state_restored'] or not seal['full_B_forward_check_done'] or not seal['all_B_fixed_own_source_q'] or seal['B_in_loss_gate_LR_or_G']:raise ValueError('Independent B boundary')
 validate_drone_final(ds,r['expected_Drone_n'])
 pairs=read(a.pairs);manifest=read(a.B_manifest);brows={x['id']:x for x in manifest}
 if len(pairs)!=76 or len({p['pair_id'] for p in pairs})!=76 or len({p['A_id'] for p in pairs})!=76 or len({p['B_id'] for p in pairs})!=76:raise ValueError('One-to-one76')
 if set(brows)!={p['B_id'] for p in pairs} or any(x['platform']!='quad' for x in manifest):raise ValueError('B76 Quad manifest')
 if seal['pairs_sha256']!=sha(a.pairs):raise ValueError('Registered pairing changed')
 if {p.name for p in a.predictions.glob('*.npz')}!={p['pair_id']+'.npz' for p in pairs}:raise ValueError('Selected output coverage')
 def member_sealed(path):
  path=Path(path).resolve()
  return any(path.is_relative_to(base) and str(path.relative_to(base)) in s['files'] and sha(path)==s['files'][str(path.relative_to(base))] for base,s in artifacts.items())
 traces=read(a.predictions/'TRACE_PRIVATE.json');traces={x['pair_id']:x for x in traces}
 if set(traces)!={p['pair_id'] for p in pairs}:raise ValueError('Trace coverage')
 first=read(a.predictions/'FIRST_NONZERO_A_FULLMODEL_B_CHECK.json')
 if not first['passed'] or first['coordinate_atol_m']!=2e-4:raise ValueError('Full-model B cache bridge')
 data={};nonzero=0
 for pair in pairs:
  pid=pair['pair_id'];row=brows[pair['B_id']]
  if row['scene']!=pair['scene']:raise ValueError('Scene join')
  for key in ('A_cache_path','B_cache_path','A_geometry_path'):
   if not member_sealed(pair[key]):raise ValueError('Source/pseudo path not sealed')
  with np.load(a.predictions/(pid+'.npz'),allow_pickle=False) as z:
   if not set(NPZ_KEYS)<=set(z.files):raise ValueError('Missing score NPZ contract')
   d={key:z[key].copy() for key in NPZ_KEYS}
  if str(d['pair_id'].item())!=pid or str(d['A_id'].item())!=pair['A_id'] or str(d['B_id'].item())!=pair['B_id']:raise ValueError('Pair output join')
  if tuple(d['arm_names'].tolist())!=ARMS or d['arm_boxes_raw6'].shape!=(7,6) or d['arm_status'].shape!=(7,):raise ValueError('Arm dimensions')
  raw=d['arm_boxes_raw6']
  if raw.dtype!=np.float32 or not np.isfinite(raw).all():raise ValueError('Finite native float32 selected boxes')
  validate_official_clamp(raw,d['arm_boxes_eval6'])
  q=int(d['q_source']);pool=d['source_boxes_raw6'];scores=d['source_text_scores256']
  if pool.shape!=(256,6) or scores.shape!=(256,) or not np.isfinite(pool).all() or not np.isfinite(scores).all() or q!=int(scores.argmax()):raise ValueError('B source shape/readout')
  if not np.array_equal(d['arm_query_index'],np.full(7,q)) or not np.array_equal(raw[0],pool[q]):raise ValueError('B source query immutable')
  with np.load(pair['B_cache_path'],allow_pickle=False) as z:
   if not np.array_equal(pool,z['boxes_raw6'][0]) or not np.array_equal(scores,z['text_scores'][0]) or q!=int(z['query_index']):raise ValueError('Immutable B cache bridge')
   for kk,outkey in [('point_sample_sha256','B_point_sample_sha256'),('raw_indices_sha256','B_raw_indices_sha256')]:
    if not np.array_equal(z[kk],d[outkey]):raise ValueError('B source input hashes')
  with np.load(pair['A_cache_path'],allow_pickle=False) as z:
   aq=int(z['query_index']);sa=z['boxes_raw6'][0,aq].copy()
   if aq!=int(d['A_q_source']) or not np.array_equal(sa,d['A_source6']):raise ValueError('A immutable source')
  with np.load(pair['A_geometry_path'],allow_pickle=False) as z:
   gn=z['arm_names'].tolist();gt=z['arm_boxes_raw6'][gn.index('B_inequality_full')].copy()
   if not np.array_equal(gt,d['A_target6']):raise ValueError('A frozen G target')
  for key,pathkey in [('A_source_cache_sha256','A_cache_path'),('B_source_cache_sha256','B_cache_path'),('A_geometry_sha256','A_geometry_path')]:
   if str(d[key].item())!=sha(pair[pathkey]):raise ValueError('Original member hash')
  gate=bool(d['A_gate']);signal=bool(gate and np.any(d['A_target6']!=sa) and np.all(sa[3:]>0) and np.all(d['A_target6'][3:]>0))
  if signal!=bool(d['A_nonzero_residual']):raise ValueError('A-input-only nonzero subset')
  nonzero+=signal;delta=d['A_target6']-sa
  if signal:
   if not np.array_equal(raw[1],raw[0]+delta) or not np.array_equal(raw[2],raw[0]+np.tile(raw[0,3:]/sa[3:],2)*delta):raise ValueError('Same-info additive/normalized controls')
  elif not np.array_equal(raw,np.tile(raw[0],(7,1))):raise ValueError('Zero or NA A must retain all B source arms')
  for pj,kj in ((3,4),(5,6)):
   if float(np.max(np.abs(raw[pj].astype(float)-raw[kj].astype(float))))>2e-4:raise ValueError('Parameter/kernel selected coordinate bridge')
  tr=traces[pid]
  if tr['A_update_executed']!=signal or tr['B_gate_loss_LR_or_clip_used'] or tr['B_local_G_constructed']:raise ValueError('A-only trace boundary')
  data[pid]=d
 if nonzero!=r['expected_A_nonzero_n'] or nonzero!=seal['nonzero_A_n']:raise ValueError('Locked A signal count')
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 dump(a.output/'SCORER_LOCK.json',{'state':'both_branches_and_all_Quad_source_pseudo_pair_control_contracts_checked_before_GT_index_payload',
   'quad_prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'drone_prediction_seal_sha256':sha(drone/'SEALED.json'),
   'release_sha256':sha(a.release),'n_pairs':76,'A_nonzero_n':nonzero,'scorer_sha256':sha(Path(__file__)),
   'subset_is_presealed_A_input_signal_not_B_task_selected':True})
 return r,seal,pairs,brows,data,traces


def evaluate_case(pair,row,d,target):
 raw=d['arm_boxes_raw6'];values,cost=candidate_ious(d['arm_boxes_eval6'],target)
 result={'pair_id':pair['pair_id'],'A_id_PRIVATE':pair['A_id'],'B_id_PRIVATE':pair['B_id'],
   'scene':row['scene'],'A_nonzero_signal':bool(d['A_nonzero_residual']),'A_gate':bool(d['A_gate']),
   'A_G_status':str(d['A_frozen_G_status'].item()),'q_source_B':int(d['q_source']),'arms':{}}
 for j,name in enumerate(ARMS):
  delta=raw[j]!=raw[0]
  result['arms'][name]={'iou':float(values[j]),'Acc25':bool(values[j]>.25),'Acc50':bool(values[j]>.5),
    'coordinate_errors':coordinate_errors(raw[j],target),'status':str(d['arm_status'][j]),
    'actual_changed':bool(delta.any()),'actual_changed_axis6':delta.tolist(),
    'raw_nonpositive_size_count':int((raw[j,3:]<=0).sum()),
    'center_shift_L2_m':float(np.linalg.norm(raw[j,:3].astype(float)-raw[0,:3].astype(float))),
    'size_shift_L2_m':float(np.linalg.norm(raw[j,3:].astype(float)-raw[0,3:].astype(float)))}
 result['parameter_kernel_coordinate_max_abs_m']={label:float(np.max(np.abs(raw[p].astype(float)-raw[k].astype(float)))) for label,p,k in [('half',3,4),('full',5,6)]}
 return result,cost


def summarize(rows):
 summary=[];contrasts=[];tails=[]
 compare=[(a,'source') for a in ARMS if a!='source']
 compare += [(a,ref) for a in ('parameter_half','kernel_half','parameter_full','kernel_full') for ref in ('direct_additive_Aresidual','direct_size_normalized_Aresidual')]
 compare += [('parameter_half','kernel_half'),('parameter_full','kernel_full'),('parameter_full','parameter_half')]
 for population,rr in [('all76_primary',rows),('A_nonzero_signal_descriptive',[r for r in rows if r['A_nonzero_signal']])]:
  for arm in ARMS:
   aa=[r['arms'][arm] for r in rr];vv=[x['iou'] for x in aa]
   ce=[x['coordinate_errors']['center_L2_m'] for x in aa];se=[x['coordinate_errors']['size_L2_m'] for x in aa]
   abs6=np.array([x['coordinate_errors']['abs_error6_m'] for x in aa]).reshape(-1,6)
   summary.append({'population':population,'arm':arm,'n':len(rr),'scene_clusters':len({r['scene'] for r in rr}),
     'mean_iou':mean(vv),'equal_scene_mean_iou':_metrics.paired_scene(rr,vv)['equal_scene_mean'],
     'Acc25':mean([float(v>.25) for v in vv]),'Acc50':mean([float(v>.5) for v in vv]),
     'equal_scene_Acc25':_metrics.paired_scene(rr,[float(v>.25) for v in vv])['equal_scene_mean'],
     'equal_scene_Acc50':_metrics.paired_scene(rr,[float(v>.5) for v in vv])['equal_scene_mean'],
     'center_L2_m_mean':mean(ce),'size_L2_m_mean':mean(se),
     'equal_scene_center_L2_m_mean':_metrics.paired_scene(rr,ce)['equal_scene_mean'],
     'equal_scene_size_L2_m_mean':_metrics.paired_scene(rr,se)['equal_scene_mean'],
     'abs6_error_m_mean':abs6.mean(0).tolist() if len(rr) else None,
     'equal_scene_abs6_error_m_mean':[_metrics.paired_scene(rr,abs6[:,j])['equal_scene_mean'] for j in range(6)] if len(rr) else None,
     'actual_changed_n':sum(x['actual_changed'] for x in aa),
     'raw_nonpositive_size_count':sum(x['raw_nonpositive_size_count'] for x in aa),
     'status_histogram':dict(collections.Counter(x['status'] for x in aa))})
  for arm,ref in compare:
   iv=[r['arms'][arm]['iou']-r['arms'][ref]['iou'] for r in rr]
   ce=[r['arms'][ref]['coordinate_errors']['center_L2_m']-r['arms'][arm]['coordinate_errors']['center_L2_m'] for r in rr]
   se=[r['arms'][ref]['coordinate_errors']['size_L2_m']-r['arms'][arm]['coordinate_errors']['size_L2_m'] for r in rr]
   ae=np.array([np.array(r['arms'][ref]['coordinate_errors']['abs_error6_m'])-r['arms'][arm]['coordinate_errors']['abs_error6_m'] for r in rr]).reshape(-1,6)
   contrasts.append({'population':population,'arm':arm,'reference':ref,'n':len(rr),
     'IoU_gain':signed_stats(iv,1e-8),'center_error_reduction_m':signed_stats(ce,1e-6),
     'size_error_reduction_m':signed_stats(se,1e-6),
     'abs6_error_reduction_m':[signed_stats(ae[:,j],1e-6) for j in range(6)],'IoU_scene':_metrics.paired_scene(rr,iv),
     'center_scene':_metrics.paired_scene(rr,ce),'size_scene':_metrics.paired_scene(rr,se),
     'positive_direction_means_improvement':True})
   ordered=sorted(zip(rr,iv),key=lambda x:(x[1],x[0]['pair_id']))
   tails.append({'population':population,'arm':arm,'reference':ref,
      'largest_negative_PRIVATE':[{'pair_id':r['pair_id'],'delta_iou':v} for r,v in ordered[:5]],
      'largest_positive_PRIVATE':[{'pair_id':r['pair_id'],'delta_iou':v} for r,v in ordered[-5:][::-1]],
      'selection':'fixed signed extremes, may contain zeros/opposite sign if tail absent; not method selection'})
 numerical={}
 for label,p,k in [('half','parameter_half','kernel_half'),('full','parameter_full','kernel_full')]:
  differences=[r['parameter_kernel_coordinate_max_abs_m'][label] for r in rows]
  numerical[label]={'coordinate_Linf_max_m':max(differences,default=None),
     'coordinate_Linf_mean_m':mean(differences),'engineering_atol_m':2e-4,
     'task_IoU_parameter_minus_kernel':signed_stats([r['arms'][p]['iou']-r['arms'][k]['iou'] for r in rows],1e-8),
     'mathematical_equivalence_not_a_contribution_gate':True}
 return summary,contrasts,tails,numerical


def main(a):
 started=time.perf_counter();release,seal,pairs,brows,data,traces=prelabel_gate(a)
 # Only this point opens label-bearing scoring-index/metadata payloads.
 index={x['id']:x for x in read(a.scoring_index)};rows=[];cost=collections.Counter()
 for pair in pairs:
  row=brows[pair['B_id']];ix=index[pair['B_id']]
  if ix['platform']!='quad' or ix['scene']!=row['scene']:raise ValueError('B target join')
  mp=Path(ix['meta_path']);bound(mp,ix['metadata_container_sha256'])
  box=read(mp)['ground_info'][ix['object_index']]['bbox_3d'];target=transform_gt(box,'quad')
  result,counts=evaluate_case(pair,row,data[pair['pair_id']],target);rows.append(result);cost.update(counts)
 summary,contrasts,tails,numerical=summarize(rows)
 per_scene=[]
 for j,scene in enumerate(sorted({x['scene'] for x in rows})):
  group=[x for x in rows if x['scene']==scene];scene_summary,scene_contrasts,_,_=summarize(group)
  per_scene.append({'scene_pseudonym':f'scene_{j+1:02d}','n':len(group),
    'A_nonzero_signal_n':sum(x['A_nonzero_signal'] for x in group),
    'summary':scene_summary,'paired_contrasts':scene_contrasts})
 mechanism_counts={'A_gate_true_n':sum(x['A_gate'] for x in rows),
   'A_nonzero_signal_n':sum(x['A_nonzero_signal'] for x in rows),
   'A_update_executed_n':sum(bool(t['A_update_executed']) for t in traces.values()),
   'true_parameter_change_half_n':sum(bool(t.get('fraction_traces',{}).get('0.5',{}).get('observed_parameter_change',False)) for t in traces.values()),
   'true_parameter_change_full_n':sum(bool(t.get('fraction_traces',{}).get('1.0',{}).get('observed_parameter_change',False)) for t in traces.values()),
   'B_selected_changed_n_by_arm':{arm:sum(x['arms'][arm]['actual_changed'] for x in rows) for arm in ARMS}}
 result={'state':'R14_Quad76_unused_input_transfer_scored_after_both_branch_seals',
   'n':len(rows),'A_nonzero_signal_n':sum(x['A_nonzero_signal'] for x in rows),
   'summary':summary,'per_scene':per_scene,'mechanism_counts':mechanism_counts,
   'paired_contrasts':contrasts,'parameter_kernel_numerical':numerical,
   'rows_PRIVATE':rows,'signed_tails_PRIVATE':tails,'geometry_cost_counts':dict(cost),
   'seconds_scoring':time.perf_counter()-started,'source_checkpoint_sha256':SOURCE_SHA,
   'scope':['Quad only, same source100/scenes as locked input plan; physically distinct image+point A/B contents, same target/different instances not certified.',
    'All76 primary; presealed A-nonzero49 is descriptive signal subset, never B-task selection.',
    'B q0 fixed, no B G/RGB training/loss/gate/LR, no teacher/EMA/full retraining.',
    'Official yaw-only target transform, native prediction yaw0, original size clamp; float64 general3D IoU strict>.25/.5.',
    'All zero/negative/NA/failure statuses retained; source center/size parameter errors are not amodal-yaw-enclosure truth.',
    'Parameter and frozen-feature kernel are mathematically equivalent; tiny numerical task differences are not adaptation-specific contribution.',
    'All targetVAL had prior historical research/model/scoring exposure; current predictions sealed before current scoring, not pristine holdout.']}
 dump(a.output/'RESULTS_PRIVATE.json',result)
 dump(a.output/'SCALAR_SUMMARY.json',{k:v for k,v in result.items() if not k.endswith('_PRIVATE')})
 report={'state':'machine_report_for_root_review_not_publication_or_adoption_decision',
   'primary_all76':[x for x in summary if x['population']=='all76_primary'],
   'descriptive_A_signal49':[x for x in summary if x['population']=='A_nonzero_signal_descriptive'],
   'paired_contrasts':contrasts,'per_scene':per_scene,'mechanism_counts':mechanism_counts,
   'parameter_kernel_numerical':numerical,
   'actual_execution_cost':{'Quad_GPU_process_seconds':seal.get('seconds'),
     'pair_end_to_end_seconds_sum':sum(float(t.get('seconds',0)) for t in traces.values()),
     'peak_allocated_bytes_max':max((int(t.get('peak_allocated_bytes',0)) for t in traces.values()),default=None),
     'scoring_seconds':result['seconds_scoring']},
   'A_signal_is_not_all_real_parameter_or_B_changes':True,
   'true_parameter_change_episode_full_n':sum(bool(t.get('fraction_traces',{}).get('1.0',{}).get('observed_parameter_change',False)) for t in traces.values()),
   'scope':result['scope'],'new_adoption_decision_executed':False}
 dump(a.output/'REPORT.json',report)
 dump(a.output/'SEALED.json',{'state':'sealed_R14_Quad_post_both_branches_CPU_score',
   'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'n':len(rows),
   'release_sha256':sha(a.release),'quad_prediction_seal_sha256':sha(a.predictions/'SEALED.json'),
   'drone_prediction_seal_sha256':release['drone_prediction_seal_sha256'],'scorer_sha256':sha(Path(__file__))})
 print(json.dumps({'state':result['state'],'n':len(rows),'A_signal_n':result['A_nonzero_signal_n'],'seal_sha256':sha(a.output/'SEALED.json')}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ('predictions','pairs','B-manifest','scoring-index','release','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--release-sha256',required=True);main(p.parse_args())
