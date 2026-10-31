"""R13 sealed-output-only mechanism reduction. No task GT/IoU/media/model IO.

Only selected-box NPZ fields and final sealed A/B scalar diagnostics are decoded.
Other seal members are byte-hashed only. No optimization, response retry or arm
selection is performed; response diversity is not semantic identity correctness.
"""
from pathlib import Path
import argparse,collections,hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
ROLE_MAP={'new_content':'new_content108','exposed_R12_all_changed_regression':'regression8'}
A_ARMS=('source','full_order0_raw','full_two_numbering_consensus','noun_two_numbering_consensus','deranged_two_numbering_consensus')
B_ARMS=('source','B_inequality_full','B_inequality_noRGB','B_inequality_half','B_old_surface_equality')
CONDITIONS=('full','noun','deranged')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def member(base,name):
 p=Path(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe seal member')
 out=(base/p).resolve()
 if not out.is_relative_to(base.resolve()):raise ValueError('Escaping seal member')
 return out

def bindings(mapping,label):
 if not isinstance(mapping,dict) or not mapping:raise ValueError('Nonempty '+label+' required')
 for path,digest in mapping.items():
  path=Path(path);path=path if path.is_absolute() else ROOT/path
  if sha(path)!=digest:raise ValueError(label+' mismatch')

def verify_seal(directory,digest):
 if sha(directory/'SEALED.json')!=digest:raise ValueError('Seal hash mismatch')
 seal=read(directory/'SEALED.json')
 if not isinstance(seal.get('files'),dict) or not seal['files']:raise ValueError('Empty files allowlist')
 for name,h in seal['files'].items():
  if sha(member(directory,name))!=h:raise ValueError('Sealed member mismatch '+name)
 producers=seal.get('program_sha256',{})
 if not isinstance(producers,dict):raise ValueError('A/B producer closure must be explicit mapping')
 for name,h in producers.items():
  if sha(member(ROOT,name))!=h:raise ValueError('Producer source mismatch')
 return seal


def stats(values):
 xs=[float(v) for v in values if v is not None]
 return {'defined_n':len(xs),'missing_n':len(values)-len(xs),'sum':float(sum(xs)),
         'mean':float(np.mean(xs)) if xs else None,'median':float(np.median(xs)) if xs else None,
         'min':min(xs) if xs else None,'max':max(xs) if xs else None}

def histogram(values):return dict(collections.Counter(str(x) for x in values))

def scalar_npz(path,keys):
 with np.load(path,allow_pickle=False) as z:
  if not set(keys)<=set(z.files):raise ValueError('Missing selected-only NPZ fields')
  return {k:z[k].copy() for k in keys}


def reduce_B(trace,npz):
 support=trace['support'];diag=support['diagnostics'];g=trace['inequality']
 raw=npz['arm_boxes_raw6'];source=raw[0];active=npz['active_coordinate_mask']
 if tuple(npz['arm_names'].tolist())!=B_ARMS or raw.shape!=(5,6) or active.shape!=(5,6):raise ValueError('B selected schema')
 if not np.array_equal(npz['arm_query_index'],np.full(5,int(npz['q_source']))):raise ValueError('B query changed')
 if not np.array_equal(npz['actual_changed_axes'],raw!=source):raise ValueError('B actual changes mismatch')
 if not np.array_equal(active[1],active[2]):raise ValueError('B main/off active domains differ')
 if not np.array_equal(raw[3],source+np.float32(.5)*(raw[1]-source)):raise ValueError('B half mismatch')
 init=g.get('initial_terms');point_satisfied=None;rgb_satisfied=None;violation_class='NA_gate_or_projection'
 if init is not None:
  point_satisfied=bool(init['point_containment']<=0)
  rgb_satisfied=bool(init['RGB_positive_containment']<=1e-16)
  violation_class=('neither_violated' if point_satisfied and rgb_satisfied else
      'RGB_only_violated' if point_satisfied else 'point_only_violated' if rgb_satisfied else 'both_violated')
 solvers=g.get('solver',{})
 per_arm={}
 for j,name in enumerate(B_ARMS):
  change=raw[j]!=source
  solver=solvers.get('direct_full' if j==1 else 'direct_no_rgb' if j==2 else '',{})
  per_arm[name]={'changed':bool(change.any()),'center_changed':bool(change[:3].any()),
      'size_changed':bool(change[3:].any()),'changed_axis6':change.tolist(),
      'center_shift_L2_m':float(np.linalg.norm(raw[j,:3].astype(float)-source[:3].astype(float))),
      'size_shift_L2_m':float(np.linalg.norm(raw[j,3:].astype(float)-source[3:].astype(float))),
      'status':str(npz['arm_status'][j]),'solver_success':solver.get('success'),
      'solver_iterations':solver.get('iterations'),
      'finite_nonconverged':bool(solver.get('finite_nonconverged_retained',False)),
      'half_has_no_solver':j==3,
      'half_derived_from_nonconverged_full':bool(j==3 and solvers.get('direct_full',{}).get('finite_nonconverged_retained',False))}
 return {'unique_mask':bool(support['segmentation_valid']),
     'support_status':support['status'],
     'unique_support_n':diag.get('selected_unique_model_XYZ',0),
     'min8_met':bool(diag.get('selected_unique_model_XYZ',0)>=8),
     'candidate_mask_n':len(diag.get('candidates',[])),
     'source_roi_unique_visible':diag.get('visible_in_source_ROI'),
     'shared_gate_passed':bool(g['gate_passed']),'gate_reason':g['gate_reason'],
     'initial_terms_defined':init is not None,'point_already_satisfied':point_satisfied,
     'RGB_already_satisfied':rgb_satisfied,'initial_violation_class':violation_class,
     'initial_point_violation_fraction':init.get('point_violation_fraction') if init else None,
     'initial_RGB_boundary_violation_fraction':init.get('RGB_boundary_violation_fraction') if init else None,
     'all_full_off_search_domains_equal':True,
     'full_off_final_boxes_equal':bool(np.array_equal(raw[1],raw[2])),
     'arms':per_arm,
     'solver_terms':{k:{name:v.get(name) for name in ('initial_terms','final_terms','initial_total','final_total','numerical_invalid','status')} for k,v in solvers.items()},
     'scope':'Containment/response mechanism only. Active coordinates are search domain, not observed amodal parameters.'}


def reduce_A(trace,npz):
 if tuple(npz['arm_names'].tolist())!=A_ARMS:raise ValueError('A arm schema')
 raw=npz['arm_boxes_raw6'];query=npz['arm_query_index'];q0=int(npz['source_q0'])
 if raw.shape!=(5,6) or int(query[0])!=q0:raise ValueError('A selected shape/query')
 mapping=trace['mapping'];groups=mapping['groups'];membership=mapping['membership256']
 if len(membership)!=256 or sorted(q for g in groups for q in g['query_indices'])!=list(range(256)):
  raise ValueError('A dedup must cover all256 exactly')
 if any(membership[q]!=g['group_id'] for g in groups for q in g['query_indices']):raise ValueError('Membership/group disagreement')
 masks=mapping['masks'];conditions={};call_rows=[]
 for cond in CONDITIONS:
  calls=trace['calls'][cond]
  if len(calls)!=2:raise ValueError('Two fixed numberings required')
  rr=trace['consensus'][cond]
  for order,call in enumerate(calls):
   call_rows.append({'condition':cond,'numbering':order,'parse_status':call['parsed']['parse_status'],
       'explicit_UNKNOWN':call['parsed']['parse_status']=='explicit_UNKNOWN',
       'raw_answer_sha256':hashlib.sha256(call['raw_answer_PRIVATE'].encode()).hexdigest(),
       'input_token_n':int(call['input_token_n']),'generated_token_n':int(call['generated_token_n']),
       'visual_token_n':sum(call['image_visualtoken_n']),
       'visual_tokens_per_image':list(map(int,call['image_visualtoken_n'])),
       'ended_with_eos':bool(call['ended_with_eos']),
       'hit_max_new_tokens':bool(call['generated_token_n']>=call['max_new_tokens']),
       'truncation_policy':'no text truncation, cap violations fail process; image native bounded resizing',
       'model_generate_seconds':float(call['seconds_model_generate'])})
  conditions[cond]={'same_mask_across_numberings':bool(rr['two_numberings_same_physical_mask']),
     'selected_mask_index':rr['selected_mask_index'],
     'selected_mask_index_by_order':rr['selected_mask_index_by_order'],
     'readout_status':rr['readout_status'],'q_selected':int(rr['q_selected']),
     'query_changed':bool(rr['query_changed']),
     'same_raw_answer_across_orders':calls[0]['raw_answer_PRIVATE']==calls[1]['raw_answer_PRIVATE']}
 per_arm={}
 for j,name in enumerate(A_ARMS):
  per_arm[name]={'query_changed':int(query[j])!=q0,'box_changed':bool(np.any(raw[j]!=raw[0])),
      'center_changed':bool(np.any(raw[j,:3]!=raw[0,:3])),
      'size_changed_by_original_query_selection_not_geometry':bool(np.any(raw[j,3:]!=raw[0,3:])),
      'selected_query':int(query[j])}
 full,noun,deranged=[conditions[c]['selected_mask_index'] for c in CONDITIONS]
 raw_readout=trace['full_order0_raw']
 return {'group_n':len(groups),'all256_covered_once':True,
    'group_size_histogram':histogram([len(g['query_indices']) for g in groups]),
    'source_group_size':len(groups[mapping['source_group']]['query_indices']),
    'mask_n':len(masks),'mapped_mask_n':sum(m['mapping_status']=='mapped_unique_geometry_hypothesis' for m in masks),
    'mask_mapping_status_histogram':histogram([m['mapping_status'] for m in masks]),
    'mask_eligible_group_counts':[m['eligible_group_n'] for m in masks],
    'conditions':conditions,'calls':call_rows,'arms':per_arm,
    'full_order0_readout_status':raw_readout['readout_status'],
    'full_order0_selected_mask_index':raw_readout['selected_mask_index'],
    'all_six_raw_answers_identical':len({c['raw_answer_sha256'] for c in call_rows})==1,
    'semantic_mask_comparisons':{
      'full_vs_noun':{'both_defined':full is not None and noun is not None,'different_including_NA':full!=noun,
                     'different_when_both_defined':full is not None and noun is not None and full!=noun},
      'full_vs_deranged':{'both_defined':full is not None and deranged is not None,'different_including_NA':full!=deranged,
                     'different_when_both_defined':full is not None and deranged is not None and full!=deranged},
      'full_consensus_vs_order0':{'both_defined':full is not None and raw_readout['selected_mask_index'] is not None,
                     'different_including_NA':full!=raw_readout['selected_mask_index']}},
    'deranged_same_literal_caption_as_full':bool(trace['deranged_same_caption_as_full']),
    'scope':'Mask/query response differences or identical raw strings are not semantic identity correctness. Size changes here are source-candidate selection, not geometric regression.'}


def summarize(rows):
 summaries=[]
 for role in ('new_content108','regression8'):
  for platform in ('drone','quad','all_platforms'):
   rr=[r for r in rows if r['case_role']==role and (platform=='all_platforms' or r['platform']==platform)]
   if not rr:continue
   bb=[r['B'] for r in rr];aa=[r['A'] for r in rr];n=len(rr)
   b={'case_n':n,'unique_independent_mask_n':sum(r['unique_mask'] for r in bb),
      'min8_support_n':sum(r['min8_met'] for r in bb),'shared_gate_passed_n':sum(r['shared_gate_passed'] for r in bb),
      'support_status_histogram':histogram([r['support_status'] for r in bb]),
      'gate_reason_histogram':histogram([r['gate_reason'] for r in bb]),
      'unique_support_count':stats([r['unique_support_n'] for r in bb]),
      'initial_terms_defined_n':sum(r['initial_terms_defined'] for r in bb),
      'point_already_satisfied_n':sum(r['point_already_satisfied'] is True for r in bb),
      'RGB_already_satisfied_n':sum(r['RGB_already_satisfied'] is True for r in bb),
      'initial_violation_class_histogram':histogram([r['initial_violation_class'] for r in bb]),
      'full_off_boxes_identical_n':sum(r['full_off_final_boxes_equal'] for r in bb),
      'full_off_boxes_different_n':sum(not r['full_off_final_boxes_equal'] for r in bb),'arms':{}}
   for arm in B_ARMS:
    v=[r['arms'][arm] for r in bb]
    b['arms'][arm]={'denominator':n,'changed_n':sum(x['changed'] for x in v),
       'center_changed_n':sum(x['center_changed'] for x in v),'size_changed_n':sum(x['size_changed'] for x in v),
       'changed_axis6_counts':[sum(x['changed_axis6'][j] for x in v) for j in range(6)],
       'center_shift_L2_m':stats([x['center_shift_L2_m'] for x in v]),
       'size_shift_L2_m':stats([x['size_shift_L2_m'] for x in v]),
       'finite_nonconverged_n':sum(x['finite_nonconverged'] for x in v),
       'half_derived_from_nonconverged_full_n':sum(x['half_derived_from_nonconverged_full'] for x in v),
       'status_histogram':histogram([x['status'] for x in v])}
   a={'case_n':n,'dedup_group_count':stats([r['group_n'] for r in aa]),
      'dedup_covers_all256_once_n':sum(r['all256_covered_once'] for r in aa),
      'source_group_size':stats([r['source_group_size'] for r in aa]),
      'masks_total':sum(r['mask_n'] for r in aa),'mapped_masks_total':sum(r['mapped_mask_n'] for r in aa),
      'mask_mapping_status_histogram':dict(sum((collections.Counter(r['mask_mapping_status_histogram']) for r in aa),collections.Counter())),
      'all_six_raw_answers_identical_n':sum(r['all_six_raw_answers_identical'] for r in aa),
      'deranged_same_literal_caption_as_full_n':sum(r['deranged_same_literal_caption_as_full'] for r in aa),
      'conditions':{},'arms':{},'semantic_mask_comparisons':{}}
   for cond in CONDITIONS:
    values=[r['conditions'][cond] for r in aa];calls=[c for r in aa for c in r['calls'] if c['condition']==cond]
    a['conditions'][cond]={'case_denominator':n,'call_denominator':len(calls),
        'strict_parse_status_histogram':histogram([c['parse_status'] for c in calls]),
        'explicit_UNKNOWN_n':sum(c['explicit_UNKNOWN'] for c in calls),
        'same_physical_mask_across_numberings_n':sum(x['same_mask_across_numberings'] for x in values),
        'same_raw_answer_across_orders_n':sum(x['same_raw_answer_across_orders'] for x in values),
        'readout_status_histogram':histogram([x['readout_status'] for x in values])}
   for arm in A_ARMS:
    values=[r['arms'][arm] for r in aa]
    a['arms'][arm]={'denominator':n,**{k:sum(x[k] for x in values) for k in
       ('query_changed','box_changed','center_changed','size_changed_by_original_query_selection_not_geometry')}}
   for comparison in ('full_vs_noun','full_vs_deranged','full_consensus_vs_order0'):
    values=[r['semantic_mask_comparisons'][comparison] for r in aa]
    a['semantic_mask_comparisons'][comparison]={'case_denominator':n,
       **{k:sum(x[k] for x in values) for k in values[0]}}
   summaries.append({'case_role':role,'platform':platform,'n':n,'B_geometry':b,'A_association':a})
 return summaries


def call_cost(cases):
 calls=[c for row in cases for c in row['calls']]
 return {'n_cases':len(cases),'n_calls':len(calls),
    'model_generate_seconds':stats([c['model_generate_seconds'] for c in calls]),
    'input_tokens':stats([c['input_token_n'] for c in calls]),
    'generated_tokens':stats([c['generated_token_n'] for c in calls]),
    'visual_tokens_both_images':stats([c['visual_token_n'] for c in calls]),
    'ended_with_eos_n':sum(c['ended_with_eos'] for c in calls),
    'hit_max_new_tokens_n':sum(c['hit_max_new_tokens'] for c in calls),
    'hit_cap_without_eos_n':sum(c['hit_max_new_tokens'] and not c['ended_with_eos'] for c in calls),
    'text_truncation_enabled':False,'image_resizing_is_not_text_truncation':True}


def main(a):
 started=time.perf_counter()
 if sha(a.release)!=a.release_sha256:raise ValueError('Release hash')
 release=read(a.release)
 if not release.get('mechanism_reduction_permitted') or release.get('GT_read_permitted') is not False or release.get('stage')!='R13_all116_noGT_mechanism_reduction':raise ValueError('No no-GT mechanism release')
 for key,path in [('reducer_sha256',Path(__file__)),('manifest_sha256',a.manifest),('cases_sha256',a.cases)]:
  if sha(path)!=release[key]:raise ValueError('Release binding '+key)
 bindings(release['program_bindings'],'program_bindings');bindings(release['input_bindings'],'input_bindings')
 ass=verify_seal(a.association,release['association_seal_sha256'])
 bs=verify_seal(a.geometry,release['geometry_seal_sha256'])
 eng=verify_seal(a.engineering,release['engineering_association_seal_sha256'])
 if ass['state']!='sealed_R13_all116_frozen_association6calls_before_GT' or not ass['model_frozen_parameters_buffers_unchanged'] or ass['GT_or_optimizer_or_neural_update_executed']:raise ValueError('Formal A state')
 if not bs['no_neural_update_or_GPU_or_GT'] or not bs['all_q_fixed']:raise ValueError('B state')
 if eng['state']!='sealed_R13_frozen_association_first2_engineering_no_GT':raise ValueError('Engineering scope')
 manifest=read(a.manifest);cases=read(a.cases);ci={r['id']:r for r in cases}
 if len(ci)!=len(cases) or len(manifest)!=len(cases) or len(manifest)!=ass['n_samples'] or len(manifest)!=bs['n']:raise ValueError('Case count')
 if ass['n_model_calls']!=len(manifest)*6 or eng['n_model_calls']!=12:raise ValueError('Formal/engineering call count')
 if ass['manifest_sha256']!=sha(a.manifest) or ass['case_index_sha256']!=sha(a.cases) or bs['input_manifest_sha256']!=sha(a.manifest):raise ValueError('Manifest/cases binding')
 trace=read(a.geometry/'TRACE_PRIVATE.json');bt={r['id']:r for r in trace}
 if len(bt)!=len(manifest):raise ValueError('B trace count')
 rows=[]
 bkeys=('sample_id','arm_names','arm_query_index','arm_boxes_raw6','arm_status','q_source','active_coordinate_mask','actual_changed_axes')
 akeys=('sample_id','arm_names','arm_query_index','arm_boxes_raw6','source_q0')
 for row in manifest:
  sid=row['id'];case=ci[sid];alias=case['case_alias'];role=ROLE_MAP[case['role']]
  at=read(a.association/'cases'/(alias+'.json'))
  if at['id_PRIVATE']!=sid or at['role']!=case['role'] or at['platform']!=row['platform'] or at['scene_PRIVATE']!=row['scene']:raise ValueError('A trace join')
  bp=scalar_npz(a.geometry/(sid+'.npz'),bkeys);ap=scalar_npz(a.association/'cases'/(alias+'.npz'),akeys)
  if str(bp['sample_id'].item())!=sid or str(ap['sample_id'].item())!=sid or int(bp['q_source'])!=int(ap['source_q0']):raise ValueError('A/B sample/source mismatch')
  if not np.array_equal(bp['arm_boxes_raw6'][0],ap['arm_boxes_raw6'][0]):raise ValueError('A/B source box mismatch')
  if bt[sid]['platform']!=row['platform'] or bt[sid]['cohort']!=case['role']:raise ValueError('B trace join')
  rows.append({'id':sid,'case_alias':alias,'case_role':role,'platform':row['platform'],
       'A':reduce_A(at,ap),'B':reduce_B(bt[sid],bp)})
 eng_rows=[]
 for path in sorted((a.engineering/'cases').glob('*.json')):
  tr=read(path);npz=scalar_npz(path.with_suffix('.npz'),akeys);eng_rows.append(reduce_A(tr,npz))
 if len(eng_rows)!=2:raise ValueError('Engineering2 coverage')
 summary={'state':'R13_sealed_output_noGT_mechanism_reduction','n_cases':len(rows),
    'case_role_counts':dict(collections.Counter(r['case_role'] for r in rows)),
    'stratified':summarize(rows),'formal_call_cost':call_cost([r['A'] for r in rows]),
    'engineering_call_cost_separate':call_cost(eng_rows),
    'formal_actual_calls':ass['n_model_calls'],'engineering_actual_calls':eng['n_model_calls'],
    'total_actual_calls_both_scopes':ass['n_model_calls']+eng['n_model_calls'],
    'formal_process_seconds':ass.get('seconds_this_process'),'engineering_process_seconds':eng.get('seconds_this_process'),
    'B_process_seconds':bs.get('seconds'),'formal_peak_allocated_bytes':ass.get('peak_allocated_bytes'),
    'seconds_reducer':time.perf_counter()-started,
    'scope':['No GT/task IoU/score/index payload/point/image/feature/model decoded.',
      'Sealed media members were integrity byte-hashed only; no image or point reader invoked.',
      'A size changes are original proposal selection, not geometric regression.',
      'B active mask is optimizer search domain, not independently observed full-box coordinates.',
      'UNKNOWN/disagreement/mapping failures and zero movement retain original cohort denominators.',
      'Changed query, changed mask, answer diversity or identical strings do not prove literal target identity or task utility.',
      'No new optimization, inference, response repair, candidate selection or threshold choice.']}
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 dump(a.output/'SUMMARY.json',summary);dump(a.output/'CASE_MECHANISM_PRIVATE.json',rows)
 dump(a.output/'SEALED.json',{'state':'sealed_R13_noGT_mechanism_reduction',
    'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'n_cases':len(rows),
    'release_sha256':sha(a.release),'reducer_sha256':sha(Path(__file__)),
    'association_seal_sha256':sha(a.association/'SEALED.json'),'geometry_seal_sha256':sha(a.geometry/'SEALED.json'),
    'engineering_association_seal_sha256':sha(a.engineering/'SEALED.json')})
 print(json.dumps({'state':summary['state'],'n':len(rows),'seal_sha256':sha(a.output/'SEALED.json')}))


if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ('association','geometry','engineering','manifest','cases','release','output'):
  p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--release-sha256',required=True);main(p.parse_args())
