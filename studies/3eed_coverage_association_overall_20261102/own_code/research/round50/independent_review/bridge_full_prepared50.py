"""UNEXECUTED cheap pre-DINO bridge: old DEV336 vs sealed full preparation.

Only JSON/NPZ artifacts and pinned seals are read after the dedicated root
grant. No image, raw point, original metadata container, model, GT, geometry
recomputation, method selection, or protected FINAL artifact is opened.
"""
from pathlib import Path
import collections,hashlib,json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'research/round50/experiment'
HERE=Path(__file__).resolve().parent
PLAN=EXP/'FULL_PREPARED_DEV_BRIDGE_PLAN_LOCK.json'
AUTH=EXP/'ROOT_INDEPENDENT_FULL_PREPARED_BRIDGE_AUTHORIZATION.json'
OLDPREP=ROOT/'crossplatform/round50/prepared336_attempt001'
OLDROWS=ROOT/'research/round24/input/ALL336_PRIVATE.json'
FULLROWS=ROOT/'research/round8/experiment/full5912_gpu_inputs.json'
FREEZE=EXP/'FULL_METHOD_FREEZE_BEFORE_FULL_MEDIA.json'
OUT=HERE/'full_prepared_dev336_bridge_attempt001'
# Full has an explicit strict_counts alias; the old preparation stored only
# mask_counts. Both full aliases must match the same original array exactly.
FIELDS=[
 ('source_boxes','source_boxes'),('source_scores','source_scores'),
 ('old_pool','old_pool'),('wide_pool','wide_pool'),
 ('old_weights','old_weights'),('wide_old_weights','wide_old_weights'),
 ('legacy_support','legacy_support'),('legacy_coverage','legacy_coverage'),
 ('legacy_projection_iou','legacy_projection_iou'),
 ('legacy_depth_trim_count','legacy_depth_trim_count'),
 ('legacy_quality','legacy_quality'),('legacy_valid','legacy_valid'),
 ('strict_counts','mask_counts'),('mask_counts','mask_counts'),
 ('allvisible','allvisible'),('unassigned','unassigned'),
 ('ambiguous','ambiguous'),('crop_valid','crop_valid'),
 ('mask_sha256','mask_sha256'),('detector_boxes','detector_boxes'),
 ('native_detector_class','native_detector_class'),
 ('historical_taxonomy_compatible','historical_taxonomy_compatible'),
 ('corrected_taxonomy_compatible','corrected_taxonomy_compatible')]
SEALS={};CHECKED={}

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def local(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 assert 'round46/final' not in p.as_posix() and 'FINAL384' not in p.as_posix()
 return ROOT/p
def seal(folder):
 folder=Path(folder)
 if folder not in SEALS:SEALS[folder]=read(folder/'SEALED.json')
 return SEALS[folder]
def member(folder,rel):
 folder=Path(folder);p=folder/rel
 assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
 if p not in CHECKED:
  digest=sha(p);assert digest==seal(folder)['files'][rel],('sealed_member_hash',rel)
  CHECKED[p]=digest
 return p
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy(order='K') for k in z.files}
def exact(a,b,label):
 a=np.asarray(a);b=np.asarray(b)
 assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),label
def identity(r):return tuple(r[k] for k in ('imagepath','pointpath','metapath','caption'))

def main():
 start=time.perf_counter();auth=read(AUTH);plan=read(PLAN)
 assert auth['state']=='ROOT_AUTHORIZED_R50_INDEPENDENT_FULLPREP_DEV336_BRIDGE_NO_GT'
 assert auth['program_sha256']==sha(__file__) and auth['bridge_plan_sha256']==sha(PLAN)
 assert plan['n']==336 and plan['full_n']==5912 and plan['no_GT_rawmedia_or_models'] is True
 assert plan['old_rows']==OLDROWS.relative_to(ROOT).as_posix()
 assert plan['full_rows']==FULLROWS.relative_to(ROOT).as_posix()
 assert plan['old_prepared']==OLDPREP.relative_to(ROOT).as_posix()
 assert plan['output']==OUT.relative_to(ROOT).as_posix()
 pins={**plan['programs_sha256'],**plan['inputs_sha256']}
 for rel,digest in pins.items():assert sha(local(rel))==digest,('input_pin',rel)
 run=local(plan['full_runroot'])
 assert run.is_relative_to(ROOT/'crossplatform/round50')
 required=[OLDROWS,FULLROWS,FREEZE,OLDPREP/'SEALED.json',OLDPREP/'BINDINGS_PRIVATE.json',
           run/'prepared/SEALED.json',run/'text/SEALED.json',run/'frames/SEALED.json']
 assert all(p.relative_to(ROOT).as_posix() in plan['inputs_sha256'] for p in required)
 assert sha(FREEZE)==plan['scientific_freeze_sha256']
 assert read(FREEZE)['selected_configuration']=='wideentry_oldW'
 assert seal(OLDPREP)['n']==336 and seal(run/'prepared')['n']==5912
 assert seal(run/'text')['n']==5912 and seal(run/'frames')['n']==5912
 assert member(OLDPREP,'BINDINGS_PRIVATE.json')==OLDPREP/'BINDINGS_PRIVATE.json'
 bindings=read(OLDPREP/'BINDINGS_PRIVATE.json')
 # The dedicated grant is checked before decoding either cohort. IDs and
 # aliases never define the cross-cohort relationship.
 oldrows=read(OLDROWS);fullrows=read(FULLROWS)
 assert len(oldrows)==336 and len(fullrows)==5912
 old_by={identity(r):r for r in oldrows};full_by={identity(r):r for r in fullrows}
 assert len(old_by)==336 and len(full_by)==5912
 assert len({r['id'] for r in oldrows})==336 and len({r['id'] for r in fullrows})==5912
 assert collections.Counter(r['platform'] for r in fullrows)=={'drone':2984,'quad':2928}
 assert set(old_by)<=set(full_by)
 assert not OUT.exists();OUT.mkdir(parents=True)
 rows=[];totals=collections.Counter();perplatform=collections.Counter();active=None
 try:
  for i,oldrow in enumerate(oldrows):
   active=f'B{i+1:04d}';fullrow=full_by[identity(oldrow)]
   assert fullrow['platform']==oldrow['platform'] and fullrow['scene']==oldrow['scene'],('platform_scene',active)
   oid,fid=oldrow['id'],fullrow['id']
   a=load(member(OLDPREP,'arrays_PRIVATE/'+oid+'.npz'))
   b=load(member(run/'prepared','arrays_PRIVATE/'+fid+'.npz'))
   oc=read(member(OLDPREP,'cases_PRIVATE/'+oid+'.json'))
   fc=read(member(run/'prepared','cases_PRIVATE/'+fid+'.json'))
   assert oc['id_PRIVATE']==oid and fc['id_PRIVATE']==fid
   assert oc['platform']==fc['platform']==oldrow['platform']
   for fullkey,oldkey in FIELDS:
    exact(b[fullkey],a[oldkey],(active,fullkey,oldkey))
    totals['field_pairs_bitexact']+=1;totals['array_values_bitexact']+=int(a[oldkey].size)
   assert fc['mask_pool_PRIVATE']==b['old_pool'].tolist() and fc['wide_pool_PRIVATE']==b['wide_pool'].tolist()
   assert oc['old_pool_count']==fc['old_pool_count']==len(a['old_pool'])
   assert oc['wide_pool_count']==fc['wide_pool_count']==len(a['wide_pool'])
   # Old text/preparation files are pinned by the already sealed DEV336
   # preparation binding registry, rather than trusting paths from new data.
   suffix='M64' if oldrow['stage']=='M' else '272'
   base=ROOT/'crossplatform'/('round24' if oldrow['stage']=='M' else 'round23')
   otp=base/('text'+suffix+'_attempt001')/'cases_PRIVATE'/(oid+'.json')
   opp=base/('prepared'+suffix+'_attempt001')/'cases_PRIVATE'/(oid+'.json')
   for p in (otp,opp):assert sha(p)==bindings[p.relative_to(ROOT).as_posix()]
   assert oc['actual_text_source_PRIVATE']==otp.relative_to(ROOT).as_posix()
   ftp=member(run/'text','cases_PRIVATE/'+fid+'.json')
   assert ftp.read_bytes()==otp.read_bytes(),('text_graph_bytes',active)
   assert fc['actual_text_source_PRIVATE']==ftp.relative_to(ROOT).as_posix()
   assert fc['actual_text_sha256']==sha(ftp)
   oldgraph=read(otp);subjectsha=hashlib.sha256(oldgraph['subject_query'].encode()).hexdigest()
   assert subjectsha==oc['actual_subject_query_sha256']==fc['actual_subject_query_sha256']
   op=read(opp);oldbinding=op['bindings']
   framecase=read(member(run/'frames','cases_PRIVATE/'+fid+'.json'))
   for key in ('image_sha256','point_sha256','mask_cache_sha256'):
    assert oldbinding[key]==fc['bindings'][key]==framecase[key],('frame_binding',active,key)
   assert framecase['id_PRIVATE']==fid and framecase['platform']==oldrow['platform']
   # Hash only the registered mask artifact: no image/XYZ/GT or mask decode.
   maskpath=member(run/'frames','masks_PRIVATE/'+fid+'.npz')
   assert sha(maskpath)==oldbinding['mask_cache_sha256']
   seed_equal=oldrow.get('seed')==fullrow.get('seed')
   rows.append({'row':active,'platform':oldrow['platform'],
    'old_id_PRIVATE':oid,'full_id_PRIVATE':fid,'exact_rawtriple_caption_platform_scene_join':True,
    'all_requested_array_pairs_shape_dtype_bytes_exact':True,
    'field_pairs_bitexact':len(FIELDS),'old_text_graph_bytes_exact':True,
    'image_point_mask_hashes_equal_to_original_bindings':True,'metadata_seed_equal':seed_equal,
    'full_text_graph_sha256':sha(ftp),'old_prepared_array_sha256':sha(member(OLDPREP,'arrays_PRIVATE/'+oid+'.npz')),
    'full_prepared_array_sha256':sha(member(run/'prepared','arrays_PRIVATE/'+fid+'.npz'))})
   totals['seed_equal']+=seed_equal;perplatform[oldrow['platform']]+=1
   if (i+1)%32==0:print('R50_FULL_PREP_DEV_BRIDGE',i+1,flush=True)
 except BaseException as exc:
  dump(OUT/'PARTIAL_ROWS_PRIVATE.json',rows)
  dump(OUT/'FAILED.json',{'state':'FAILED_R50_FULL_PREPARED_DEV336_CHEAP_BRIDGE_NO_GT',
   'error_type':type(exc).__name__,'error':str(exc),'active_anonymous_row':active,
   'completed_n':len(rows),'no_new_model_or_GT_or_automatic_retry':True});raise
 assert len(rows)==336 and totals['field_pairs_bitexact']==336*len(FIELDS)
 dump(OUT/'BRIDGE_ROWS_PRIVATE.json',rows)
 result={'state':'PASS_R50_INDEPENDENT_FULL_PREPARED_DEV336_CHEAP_BRIDGE_NO_GT',
  'n':336,'full_n':5912,'per_platform':dict(perplatform),'array_field_pairs_per_case':len(FIELDS),
  'array_field_pairs_shape_dtype_bytes_exact':totals['field_pairs_bitexact'],
  'array_values_bitexact_including_the_explicit_strict_counts_alias':totals['array_values_bitexact'],
  'old_text_graphs_bytes_exact':336,'original_image_point_mask_binding_triples_exact':336,
  'join_uses_exact_rawtriple_caption_and_platform_scene_not_ID_or_alias':True,
  'metadata_seed_equal_n':totals['seed_equal'],'metadata_seed_mismatch_n':336-totals['seed_equal'],
  'seed_difference_if_any_is_metadata_and_requires_explanation_not_new_source_resampling':True,
  'fields_full_to_old':FIELDS,'no_original_rawgeometry_or_maskpixel_ownership_recomputation':True,
  'no_GT_rawRGB_rawXYZ_models_or_protected_FINAL_specific_artifacts_read':True,
  'not_an_identity_truth_or_task_quality_audit':True,
  'scientific_freeze_sha256':sha(FREEZE),'bridge_plan_sha256':sha(PLAN),'program_sha256':sha(__file__),
  'seconds':time.perf_counter()-start}
 dump(OUT/'AUDIT_RESULT.json',result)
 dump(OUT/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
 print(json.dumps(result))

if __name__=='__main__':main()
