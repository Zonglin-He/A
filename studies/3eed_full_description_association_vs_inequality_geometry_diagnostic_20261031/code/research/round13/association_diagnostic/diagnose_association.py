"""R13 POST-SCORE descriptive annotated spatial compatibility, release gated.

No literal target identity labels. No query/G/prediction/metric modification.
Numeric annotation read/IoU occurs only after a separate hash-bound v2 release.
"""
from pathlib import Path
import argparse,collections,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'research/round5/audit'),str(ROOT/'crossplatform/round13')]
from independent_core_v2 import transform_gt,candidate_ious,validate_official_clamp
from support import parse_target_family
DEPENDENCIES={
 'research/round5/audit/independent_core_v2.py':'b4937fdd89f719fe1d9560f770c01597ff6807c74a75013720b470c43393379a',
 'tools/obb_iou.py':'45896d29d1f95167b95a27b044fcc06b4e6fe5f9cb98749096c2f707292d4083',
 'crossplatform/round13/support.py':'50b8cb9bf47d647239951e7b1bd9c976ac1049a81885a13de5789086a16576e4'}
READOUTS={'q0':'source','fullA':'A_full_consensus','nounA':'A_noun_consensus'}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def bound(path,digest):
 if sha(path)!=digest:raise ValueError('Binding mismatch')
def verify(directory,digest):
 bound(directory/'SEALED.json',digest);seal=read(directory/'SEALED.json')
 if not isinstance(seal.get('files'),dict) or not seal['files']:raise ValueError('Empty artifact')
 for name,h in seal['files'].items():
  p=Path(name)
  if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe member')
  bound(directory/p,h)
 return seal

def scalar_stats(values):
 xs=[float(v) for v in values if v is not None]
 return {'defined_n':len(xs),'NA_n':len(values)-len(xs),'mean':float(np.mean(xs)) if xs else None,
    'median':float(np.median(xs)) if xs else None,'min':min(xs) if xs else None,'max':max(xs) if xs else None}

def transform_other7(raw7,platform,view_suffix=None):
 """Explicit released raw yaw7D -> source model coords; no fabricated tails.

 JSON geometry is copied to float64, Drone raw z receives +1.8 once, Quad
 remains unchanged, then the official seven-vector is delivered float32.
 """
 if platform not in ('drone','quad'):raise ValueError('R13 other7 adapter only Drone/Quad')
 if view_suffix is not None:raise ValueError('No camera suffix transform in D/Q other7 adapter')
 raw=np.asarray(raw7)
 if raw.shape!=(7,) or not np.issubdtype(raw.dtype,np.number):raise ValueError('Other must be numeric raw yaw7D')
 b=np.array(raw,np.float64,copy=True)
 if not np.isfinite(b).all() or np.any(b[3:6]<=0):raise ValueError('Invalid annotated-other7D geometry')
 if platform=='drone':b[2]+=1.8
 return b.astype(np.float32)

def annotated_geometry(target_raw,others,platform,view_suffix=None):
 target=transform_gt(target_raw,platform,view_suffix)
 groups=[target];members=[[{'source':'target'}]];malformed=0
 if not isinstance(others,list):return target,[],{'schema':'NA_others_not_list','malformed_other_n':1,'input_other_n':None,'deduplicated_other_n':0}
 for i,obj in enumerate(others):
  try:
   if not isinstance(obj,dict) or 'bbox_3d_other' not in obj:raise ValueError('Missing other bbox')
   b=transform_other7(obj['bbox_3d_other'],platform,view_suffix)
  except (ValueError,TypeError):malformed+=1;continue
  group=next((j for j,g in enumerate(groups) if np.array_equal(g,b)),None)
  if group is None:groups.append(b);members.append([]);group=len(groups)-1
  members[group].append({'source':'other_annotation','position':i})
 return target,groups[1:],{'schema':'defined' if malformed==0 else 'NA_partial_other_schema',
    'input_other_n':len(others),'malformed_other_n':malformed,
    'deduplicated_other_n':len(groups)-1,'target_exact_duplicate_annotation_n':len(members[0])-1,
    'group_member_counts':[len(v) for v in members],
    'dedup':'exact elementwise transformed float32 official7D before IoU; no IDs inferred',
    'target_source_schema':'target9','other_source_schema':'other7_no_fabricated_tails'}

def classify(target_iou,other_iou,schema):
 if schema.startswith('NA_'):return schema
 if other_iou is None:return 'NA_no_distinct_annotated_other_geometry'
 margin=target_iou-other_iou
 if target_iou>.25 and margin>.10:return 'target_spatially_dominant'
 if other_iou>.25 and -margin>.10:return 'annotated_other_spatially_dominant'
 return 'unconfirmed_spatial_relation'

def mask_relation(a,b):
 if a is None and b is None:return 'both_undefined'
 if a is None:return 'A_undefined'
 if b is None:return 'B_undefined'
 return 'same_defined_mask' if a==b else 'different_defined_masks'

def predicted_category_diagnostic(selected,class_ids,family):
 if selected is None:return {'defined':False,'COCO_predicted_class_id':None,'firstnoun_family_compatible':None}
 if not 0<=int(selected)<len(class_ids):raise ValueError('Selected mask outside frozen COCO classes')
 cls=int(class_ids[int(selected)])
 return {'defined':True,'COCO_predicted_class_id':cls,
     'firstnoun_family_compatible':bool(cls in family) if family else None,
     'scope':'frozen category prediction versus input firstnoun family; not GT identity/category'}

def prepare(a):
 bound(a.release,a.release_sha256);r=read(a.release)
 if not r.get('POSTSCORE_DIAGNOSTIC_RELEASE') or r.get('stage')!='R13_postscore_annotated_spatial_association_diagnostic':raise ValueError('No explicit new diagnostic release')
 if not r['all_task_predictions_and_main_scores_sealed'] or r['method_or_primary_metric_modification_permitted']:raise ValueError('Scope')
 if r['annotation_source']!='target_ground_info_index_plus_its_others_bbox_3d_other_raw7D':raise ValueError('v2 source adapter not explicitly locked')
 for rel,h in DEPENDENCIES.items():bound(ROOT/rel,h)
 for rel,h in r['program_bindings'].items():
  p=Path(rel);bound(p if p.is_absolute() else ROOT/p,h)
 bound(r['source_checkpoint_path'],r['source_checkpoint_sha256'])
 if r['geometry_dependency_sha256']!=DEPENDENCIES:raise ValueError('Dependency map')
 for key in ('program','plan_v1','plan_v2','manifest','cases','scoring_index'):
  bound(r[key+'_path'],r[key+'_sha256'])
 if Path(r['program_path']).resolve()!=Path(__file__).resolve():raise ValueError('Program path')
 artifacts={}
 for item in r['sealed_artifacts']:
  if item['role'] in artifacts:raise ValueError('Duplicate role')
  artifacts[item['role']]=(Path(item['directory']),verify(Path(item['directory']),item['seal_sha256']))
 if not {'composite','A','B','scores','schema','nested_schema','count_schema'}<=set(artifacts):raise ValueError('Artifact closure')
 if artifacts['composite'][1]['source_checkpoint_sha256']!=r['source_checkpoint_sha256']:raise ValueError('Source checkpoint mismatch')
 manifest=read(r['manifest_path']);cases=read(r['cases_path']);ci={c['id']:c for c in cases}
 if any(row['platform'] not in ('drone','quad') for row in manifest):raise ValueError('R13 diagnostic only D/Q')
 if len(ci)!=len(cases) or len(manifest)!=116 or set(ci)!={x['id'] for x in manifest}:raise ValueError('Bound116 case coverage')
 selected={};A=artifacts['A'][0];B=artifacts['B'][0];C=artifacts['composite'][0]
 for row in manifest:
  sid=row['id'];case=ci[sid];alias=case['case_alias']
  md=Path(case['mask_dir']);md=md if md.is_absolute() else ROOT/md
  if md.resolve() not in {base.resolve() for base,_ in artifacts.values()}:raise ValueError('All used frozen mask seals must be explicitly bound')
  with np.load(C/(sid+'.npz'),allow_pickle=False) as z:
   d={k:z[k].copy() for k in ('sample_id','arm_names','arm_query_index','arm_boxes_raw6','arm_boxes_eval6','source_boxes_raw6','source_text_scores256','q_source','case_role')}
  names=d['arm_names'].tolist();pool=d['source_boxes_raw6'];q0=int(d['q_source'])
  if str(d['sample_id'].item())!=sid or q0!=int(d['source_text_scores256'].argmax()):raise ValueError('Source query bridge')
  validate_official_clamp(d['arm_boxes_raw6'],d['arm_boxes_eval6'])
  for arm in READOUTS.values():
   j=names.index(arm);q=int(d['arm_query_index'][j])
   if not np.array_equal(pool[q],d['arm_boxes_raw6'][j]):raise ValueError('Original proposal only')
  atrace=read(A/'cases'/(alias+'.json'))
  if atrace['id_PRIVATE']!=sid:raise ValueError('A join')
  with np.load(B/(sid+'.npz'),allow_pickle=False) as z:bmask=int(z['selected_mask_index'])
  bmask=None if bmask<0 else bmask
  selected[sid]={'data':d,'A_mask':atrace['consensus']['full']['selected_mask_index'],
       'A_noun_mask':atrace['consensus']['noun']['selected_mask_index'],'B_mask':bmask}
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 dump(a.output/'DIAGNOSTIC_LOCK.json',{'state':'all_hashes_selected_queries_and_v2_annotation_source_bound_before_numeric_GT',
    'release_sha256':sha(a.release),'plan_v1_sha256':r['plan_v1_sha256'],'plan_v2_sha256':r['plan_v2_sha256'],
    'n_cases':116,'dedup':'exact transformed float32 official7D','query_selection_performed':False})
 return r,manifest,ci,selected,artifacts


def summaries(rows):
 output=[]
 for role in ('new_content108','regression8'):
  for platform in ('drone','quad','all_platforms'):
   base=[x for x in rows if x['case_role']==role and (platform=='all_platforms' or x['platform']==platform)]
   if not base:continue
   strata={'all':base,'same_defined_mask_and_fullA_query_changed':[x for x in base if x['A_B_mask_relation']=='same_defined_mask' and x['fullA_query_changed']],
       'same_defined_mask':[x for x in base if x['A_B_mask_relation']=='same_defined_mask'],
       'different_defined_masks':[x for x in base if x['A_B_mask_relation']=='different_defined_masks'],
       'one_or_both_masks_undefined':[x for x in base if x['A_B_mask_relation'] not in ('same_defined_mask','different_defined_masks')]}
   for stratum,group in strata.items():
    out={'case_role':role,'platform':platform,'stratum':stratum,'n':len(group),
      'scene_clusters':len({x['scene_pseudonym'] for x in group}),
      'mask_relation_counts':dict(collections.Counter(x['A_B_mask_relation'] for x in group)),
      'query_changed_n':sum(x['fullA_query_changed'] for x in group),
      'box_changed_n':sum(x['fullA_box_changed'] for x in group),'readouts':{},
      'annotation_other_count':scalar_stats([x['annotation_schema']['deduplicated_other_n'] for x in group]),
      'spatial_category_transition_q0_to_fullA':dict(collections.Counter(x['readouts']['q0']['spatial_category']+' -> '+x['readouts']['fullA']['spatial_category'] for x in group))}
    for label in READOUTS:
     values=[x['readouts'][label] for x in group]
     out['readouts'][label]={'target_IoU':scalar_stats([x['target_IoU'] for x in values]),
       'annotated_other_max_IoU':scalar_stats([x['annotated_other_max_IoU'] for x in values]),
       'target_minus_other_margin':scalar_stats([x['target_minus_other_margin'] for x in values]),
       'spatial_category_counts':dict(collections.Counter(x['spatial_category'] for x in values))}
    out['fullA_minus_q0_target_IoU']=scalar_stats([x['readouts']['fullA']['target_IoU']-x['readouts']['q0']['target_IoU'] for x in group])
    out['fullA_minus_q0_margin']=scalar_stats([x['readouts']['fullA']['target_minus_other_margin']-x['readouts']['q0']['target_minus_other_margin'] if x['readouts']['fullA']['target_minus_other_margin'] is not None and x['readouts']['q0']['target_minus_other_margin'] is not None else None for x in group])
    out['predicted_COCO_category_not_GT']={}
    for key in ('A_full','A_noun','B_support'):
     vv=[x['predicted_category_addendum'][key] for x in group]
     out['predicted_COCO_category_not_GT'][key]={'defined_n':sum(x['defined'] for x in vv),
        'compatible_n':sum(x['firstnoun_family_compatible'] is True for x in vv),
        'incompatible_n':sum(x['firstnoun_family_compatible'] is False for x in vv),
        'compatibility_NA_n':sum(x['firstnoun_family_compatible'] is None for x in vv)}
    output.append(out)
 return output


def main(a):
 start=time.perf_counter();release,manifest,cases,selected,artifacts=prepare(a)
 # NUMERIC LABEL BOUNDARY: separate post-score release+v2 source adapter checked.
 index={x['id']:x for x in read(release['scoring_index_path'])};rows=[];counts=collections.Counter()
 primary={x['id']:x for x in read(artifacts['scores'][0]/'RESULTS_PRIVATE.json')['rows_PRIVATE']};max_primary_error=0.
 scene_names=sorted({(x['platform'],x['scene']) for x in manifest});scene_alias={s:f'scene_{i+1:02d}' for i,s in enumerate(scene_names)}
 for row in manifest:
  sid=row['id'];case=cases[sid];ix=index[sid];d=selected[sid]['data'];names=d['arm_names'].tolist()
  if row['platform']!=ix['platform'] or row['scene']!=ix['scene']:raise ValueError('Annotation join')
  mp=Path(ix['meta_path']);bound(mp,ix['metadata_container_sha256']);metadata=read(mp)
  entry=metadata['ground_info'][ix['object_index']]
  if not isinstance(entry.get('others'),list) or not isinstance(entry.get('others_num'),int) or entry['others_num']!=len(entry['others']):raise ValueError('Declared other count changed from schema qualification')
  target,others,schema=annotated_geometry(entry['bbox_3d'],entry.get('others'),row['platform'],ix.get('waymo_view_suffix'))
  readouts={}
  for key,arm in READOUTS.items():
   j=names.index(arm);box=d['arm_boxes_eval6'][j:j+1];tv,gc=candidate_ious(box,target);counts.update(gc)
   other_values=[]
   for t in others:
    ov,gc=candidate_ious(box,t);counts.update(gc);other_values.append(float(ov[0]))
   other=max(other_values) if other_values else None;target_iou=float(tv[0])
   primary_error=abs(target_iou-primary[sid]['arms'][arm]['iou']);max_primary_error=max(max_primary_error,primary_error)
   if primary_error>1e-12:raise ValueError('Target IoU diagnostic differs from immutable primary score; do not replace primary')
   readouts[key]={'query_index':int(d['arm_query_index'][j]),'target_IoU':target_iou,
       'annotated_other_max_IoU':other,'target_minus_other_margin':target_iou-other if other is not None else None,
       'spatial_category':classify(target_iou,other,schema['schema']),
       'literal_identity_GT_claimed':False}
  # This addendum uses only frozen predicted class IDs and the original input
  # firstnoun, not GT class strings, other annotations or the spatial categories.
  maskdir=Path(case['mask_dir']);maskdir=maskdir if maskdir.is_absolute() else ROOT/maskdir
  ms=read(maskdir/'SEALED.json');maskpath=maskdir/(case['frame_alias']+'.npz');bound(maskpath,ms['files'][maskpath.name])
  with np.load(maskpath,allow_pickle=False) as z:classes=z['class_ids'].copy()
  noun,family,scope=parse_target_family(row['caption']);sel=selected[sid]
  addendum={k:predicted_category_diagnostic(v,classes,family) for k,v in
     [('A_full',sel['A_mask']),('A_noun',sel['A_noun_mask']),('B_support',sel['B_mask'])]}
  j0=names.index('source');ja=names.index('A_full_consensus')
  rows.append({'case_alias':case['case_alias'],'id_PRIVATE':sid,'case_role':str(d['case_role'].item()),
      'platform':row['platform'],'scene_pseudonym':scene_alias[(row['platform'],row['scene'])],
      'A_B_mask_relation':mask_relation(sel['A_mask'],sel['B_mask']),
      'A_full_mask_index':sel['A_mask'],'B_support_mask_index':sel['B_mask'],
      'fullA_query_changed':int(d['arm_query_index'][ja])!=int(d['q_source']),
      'fullA_box_changed':bool(np.any(d['arm_boxes_raw6'][ja]!=d['arm_boxes_raw6'][j0])),
      'annotation_schema':schema,'readouts':readouts,'predicted_category_addendum':addendum,
      'scope':'annotated spatial compatibility only; no object IDs inferred, no complete scene inventory, no literal identity ground truth'})
 result={'state':'R13_postscore_descriptive_spatial_association_completed','n':len(rows),
   'summary':summaries(rows),'geometry_cost_counts':dict(counts),'seconds':time.perf_counter()-start,
   'target_IoU_existing_primary_max_abs_error':max_primary_error,'target_IoU_bridge_atol':1e-12,
   'scope':['All116 bound; cohorts separate; no new method prediction or GT query selection.',
      'Same selected 2D mask plus worse original 3D proposal is compatible with mapping/readout error; it does not prove the mask is the literal target.',
      'Unconfirmed or NA does not equal wrong; annotated-other dominance is annotation-relative spatial evidence, not literal identity GT.',
      'Frozen COCO category compatibility is a separate predicted-category diagnostic, not annotation truth.',
      'No G, original task metric, threshold or adoption rule modified.']}
 dump(a.output/'SUMMARY.json',result);dump(a.output/'CASES_PRIVATE.json',rows)
 dump(a.output/'SEALED.json',{'state':'sealed_R13_postscore_spatial_diagnostic','n':len(rows),
  'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'release_sha256':sha(a.release),
  'program_sha256':sha(Path(__file__))})
 print(json.dumps({'state':result['state'],'n':len(rows),'seal_sha256':sha(a.output/'SEALED.json')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
