"""UNEXECUTED narrow first-row CPU diagnostic; no model/optimizer/task GT.

Compare frozen B geometry using the original independent auditor's IoU,
an explicit float64 IoU, and an original-corner-order control. The original
auditor, producer, states, predictions, tolerances and locks are not edited.
"""
from pathlib import Path
import ast,hashlib,itertools,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'research/round50/experiment'
PLAN=EXP/'EPISODIC_PROJECTION_DTYPE_DIAGNOSTIC_PLAN_LOCK.json'
AUTH=EXP/'ROOT_EPISODIC_PROJECTION_DTYPE_DIAGNOSTIC_AUTHORIZATION.json'
OUT=Path(__file__).resolve().parent/'episodic_projection_dtype_diagnostic_attempt001'
OLD_AUDITOR=Path(__file__).with_name('audit_episodic50_draft.py')
ORIGINAL_AUDIT_PLAN=EXP/'EPISODIC_INDEPENDENT_AUDIT_PLAN_LOCK.json'
OLD_ROWS=ROOT/'research/round24/input/ALL336_PRIVATE.json'
HASH={};SEALS={};IO=[]
def sha(p):
 p=Path(p)
 if p not in HASH:HASH[p]=hashlib.sha256(p.read_bytes()).hexdigest()
 return HASH[p]
def read(p):
 p=Path(p);IO.append(p.relative_to(ROOT).as_posix())
 return json.loads(p.read_text())
def local(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 assert 'FINAL384' not in rel and 'round46/final' not in rel and 'full5912_gpu_inputs' not in rel
 return ROOT/p
def seal(folder):
 folder=Path(folder)
 if folder not in SEALS:SEALS[folder]=read(folder/'SEALED.json')
 return SEALS[folder]
def member(folder,rel):
 p=Path(folder)/rel;assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
 assert sha(p)==seal(folder)['files'][rel]
 return p
def load_selected(p,keys):
 IO.append(Path(p).relative_to(ROOT).as_posix())
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy(order='K') for k in keys}
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def equal(a,b):
 a=np.asarray(a);b=np.asarray(b)
 return bool(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes())
def difference(a,b):
 a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
 return {'shape':list(a.shape),'dtype_left':str(a.dtype),'dtype_right':str(b.dtype),
  'shape_dtype_bytes_exact':equal(a,b),'numeric_different_cells':int(np.count_nonzero(a!=b)),
  'maximum_abs_difference':float(np.max(abs(a.astype(float)-b.astype(float)),initial=0))}
def rect_float64(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float)
 inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
 union=np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter
 return float(inter/union) if union>0 else 0.
def weight_from_raw(pool,S,C,I,boxes):
 valid=(S>=3)&(C>=.15)&(I>=.1)&(boxes[:,3:]>0).all(1)[:,None]
 quality=np.where(valid,C*np.sqrt(np.maximum(I,0)),0.)
 w=quality[:,pool].copy() if len(pool) else np.zeros((256,0));w/=np.maximum(w.sum(1,keepdims=True),1e-12)
 return w,valid

def main():
 start=time.perf_counter();auth=read(AUTH);plan=read(PLAN)
 assert auth['state']=='ROOT_AUTHORIZED_R50_FIRSTROW_EPISODIC_PROJECTION_DTYPE_DIAGNOSTIC_NO_GT'
 assert auth['program_sha256']==sha(__file__) and auth['plan_sha256']==sha(PLAN)
 assert plan['n']==1 and plan['anonymous_row']=='E0001' and plan['models_optimizers_GT_or_rescoring_allowed'] is False
 for rel,h in {**plan['programs_sha256'],**plan['inputs_sha256']}.items():assert sha(local(rel))==h
 assert sha(OLD_AUDITOR)==plan['original_independent_auditor_sha256']
 assert OLD_AUDITOR.relative_to(ROOT).as_posix() in plan['programs_sha256']
 oldplan=read(ORIGINAL_AUDIT_PLAN);rows=read(OLD_ROWS);assert len(rows)==336;row=rows[0];sid=row['id']
 cap=local(oldplan['capture_root']);fit=local(oldplan['fit_root'])
 prep=ROOT/'crossplatform/round50/prepared336_attempt001';previous=ROOT/'crossplatform/round50/predicted336_attempt001'
 own=ROOT/'crossplatform/round49/ownership336_attempt001';stage=ROOT/'crossplatform/round49/stage336_attempt001'
 for p in [ORIGINAL_AUDIT_PLAN,OLD_ROWS,*[x/'SEALED.json' for x in (cap,fit,prep,previous,own,stage)]]:
  assert p.relative_to(ROOT).as_posix() in plan['inputs_sha256']
 assert plan['failed_audit_result'] in plan['inputs_sha256']
 failure=read(local(plan['failed_audit_result']));assert failure['active_anonymous_row']=='E0001' and failure['completed_n']==0
 assert 'B_projection_iou' in failure['error']
 v=load_selected(member(cap,'samples_PRIVATE/'+sid+'.npz'),['A_boxes','A_scores','B_boxes','B_scores'])
 actual=load_selected(member(fit,'samples_PRIVATE/'+sid+'.npz'),['B_weights','B_legacy_support','B_legacy_coverage','B_projection_iou','A_fixed_VPR'])
 pc=read(member(fit,'cases_PRIVATE/'+sid+'.json'))
 a=load_selected(member(prep,'arrays_PRIVATE/'+sid+'.npz'),['wide_pool','legacy_valid'])
 previous_values=load_selected(member(previous,'samples_PRIVATE/'+sid+'.npz'),['image_e','raw_mask_PR'])
 AC={x['id']:x for x in read(ROOT/'research/round17/input/A_CASES_PRIVATE.json')}
 AF=read(ROOT/'research/round17/input/A_FRAME_INDEX_PRIVATE.json');RF=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json')
 for rel in ['crossplatform/round7/camera_projection.py','research/round17/input/A_CASES_PRIVATE.json',
  'research/round17/input/A_FRAME_INDEX_PRIVATE.json','research/round12/data_audit/FRAME_INDEX_PRIVATE.json']:
  assert rel in {**plan['programs_sha256'],**plan['inputs_sha256']}
 # Reuse only the locked camera mathematical bridge; this import does not
 # enter a model, original metadata reader, evaluator or experiment main.
 sys.path.insert(0,str(ROOT/'crossplatform/round7'));from camera_projection import CameraSpec,project_model_points
 source=ast.parse(OLD_AUDITOR.read_text());defs={n.name:n for n in source.body if isinstance(n,ast.FunctionDef)}
 mathnames=['box_iou','groups','matching','group_scores','choose_group','unique_mask','structure','old_read','rect_iou']
 mathns={'np':np}
 exec(compile(ast.Module(body=[defs[n] for n in mathnames],type_ignores=[]),'<original-unmodified-independent-pure-functions>','exec'),mathns)
 owncase=read(member(own,'cases_PRIVATE/'+sid+'.json'));stagecase=read(member(stage,'cases_PRIVATE/'+sid+'.json'))
 framepath=member(own,'frames_PRIVATE/'+owncase['frame_key_PRIVATE']+'.npz');maskpath=local(stagecase['mask_cache_path_PRIVATE'])
 assert sha(maskpath)==owncase['frame_mask_cache_sha256']==stagecase['mask_cache_sha256']
 allowed_npz={framepath:('visible_points_model','point_owner','depth_camera'),maskpath:('original_hw','class_ids','boxes_xyxy')}
 def selective_geometry_load(p):
  p=Path(p);assert p in allowed_npz
  return load_selected(p,allowed_npz[p])
 base={'np':np,'itertools':itertools,'ROOT':ROOT,'read':read,'member':member,'load':selective_geometry_load,'local':local}
 variants={}
 for name,iou in [('original_auditor',mathns['rect_iou']),('float64_cast_only',rect_float64)]:
  ns=dict(base,rect_iou=iou);exec(compile(ast.Module(body=[defs['B_geometry']],type_ignores=[]),'<single-old-geometry-function>','exec'),ns)
  variants[name]=ns['B_geometry'](row,v['B_boxes'],a['wide_pool'],own,stage,AC,AF,RF,CameraSpec,project_model_points)
 # Competing row-order explanation: independently reorder the same eight
 # corners as original association13, preserving every operation otherwise.
 original_corner_node=ast.parse(ast.unparse(defs['B_geometry'])).body[0]
 class Corners(ast.NodeTransformer):
  def visit_Assign(self,node):
   if any(isinstance(x,ast.Name) and x.id=='signs' for x in node.targets):
    return ast.parse('signs=np.asarray([[(-1 if not i&(1<<j) else 1) for j in range(3)] for i in range(8)])').body[0]
   return self.generic_visit(node)
 original_corner_node=ast.fix_missing_locations(Corners().visit(original_corner_node))
 ns=dict(base,rect_iou=rect_float64);exec(compile(ast.Module(body=[original_corner_node],type_ignores=[]),'<corner-order-control>','exec'),ns)
 variants['float64_and_original_corner_order']=ns['B_geometry'](row,v['B_boxes'],a['wide_pool'],own,stage,AC,AF,RF,CameraSpec,project_model_points)
 stored=(actual['B_weights'],None,actual['B_legacy_support'],actual['B_legacy_coverage'],actual['B_projection_iou'])
 stored_weights,stored_valid=weight_from_raw(a['wide_pool'],stored[2],stored[3],stored[4],v['B_boxes'])
 assert equal(stored_weights,actual['B_weights'])
 FA=v['A_scores'].astype(float)+2*actual['A_fixed_VPR'][0]+2*actual['A_fixed_VPR'][1]+2*actual['A_fixed_VPR'][2]
 results={};structures={};readouts={}
 for name,values in variants.items():
  w,valid,S,C,I=values;F,components=mathns['old_read'](v['B_scores'],w,a['wide_pool'],previous_values['image_e'],previous_values['raw_mask_PR'])
  st=mathns['structure'](v,a['legacy_valid'],valid,FA,F);structures[name]=st;readouts[name]=(F,components)
  results[name]={'projection_iou_vs_stored':difference(I,stored[4]),'support_vs_stored':difference(S,stored[2]),
   'coverage_vs_stored':difference(C,stored[3]),'weights_vs_stored':difference(w,stored[0]),
   'valid_mask_flags_vs_stored_raw':difference(valid,stored_valid),
   'acceptance':st['accepted'],'reason':st['reason'],'A_precondition':st['pre_A_condition']}
 cast=variants['float64_cast_only'];old=variants['original_auditor'];control=variants['float64_and_original_corner_order']
 gatekeys=['accepted','reason','pre_A_condition','source_group','positive_group','q0_group_has_B_mapping','mask_source_A','mask_positive_A','strict_A_group_gap']
 gate_equal=all(structures['original_auditor'][k]==structures['float64_cast_only'][k] for k in gatekeys)
 same_support=equal(old[2],stored[2]) and equal(cast[2],stored[2]);same_coverage=equal(old[3],stored[3]) and equal(cast[3],stored[3])
 assert same_support and same_coverage
 assert all(structures['float64_cast_only'][k]==pc[k] for k in gatekeys), 'Corrected geometric dtype does not reproduce frozen feedback contract'
 for variant in variants:
  for name in ('match_A','match_B','repsA','geometry_repsA','repsB'):
   assert equal(structures[variant][name],structures['float64_cast_only'][name])
 masks=selective_geometry_load(maskpath);det=masks['boxes_xyxy'];old_area=np.maximum(det[:,2:]-det[:,:2],0).prod(-1);new_area=np.maximum(det.astype(float)[:,2:]-det.astype(float)[:,:2],0).prod(-1)
 fixed_exact=all(equal(cast[k],stored[k]) for k in (0,2,3,4))
 result={'state':'PASS_R50_FIRSTROW_EPISODIC_IoU_FLOAT64_CAST_CAUSE_DIAGNOSTIC_NO_GT' if fixed_exact else 'UNRESOLVED_R50_FIRSTROW_PROJECTION_DIAGNOSTIC_NO_GT',
  'anonymous_row':'E0001','n':1,'original_auditor_sha256':sha(OLD_AUDITOR),'variants':results,
  'float64_cast_only_recovers_all_stored_projection_IoU_support_coverage_and_W_bytes':fixed_exact,
  'only_corner_order_control_projection_difference':difference(cast[4],control[4]),
  'old_vs_cast_G_B_readout_difference':difference(readouts['original_auditor'][0],readouts['float64_cast_only'][0]),
  'old_vs_cast_local_preference_gates_equal':gate_equal,
  'original_detector_box_dtype':str(det.dtype),'original_detector_area_dtype':str(old_area.dtype),'explicit_float64_detector_area_dtype':str(new_area.dtype),
  'original_vs_cast_detector_area_difference':difference(old_area.astype(float),new_area),
  'no_3D_VLM_detector_or_tinyhead_model_forwards':True,'optimizer_steps':0,'GT_reads':0,'task_scores':0,
  'no_producer_prediction_auditor_or_lock_modified':True,'no_comparison_tolerance_weakened':True,
  'not_an_all336_audit_PASS_or_task_quality_result':True,'plan_sha256':sha(PLAN),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start}
 assert not OUT.exists();OUT.mkdir(parents=True);dump(OUT/'DIAGNOSTIC.json',result)
 dump(OUT/'READ_BINDINGS_PRIVATE.json',{p:sha(ROOT/p) for p in sorted(set(IO))})
 dump(OUT/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
 print(json.dumps(result))

if __name__=='__main__':main()
