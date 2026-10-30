"""R10 sealed four-arm G audit; no targetGT/index/IoU/model API."""
from pathlib import Path
import argparse,collections,json
import numpy as np
from score_geometry import sha,read,dump,bounded,prediction,ARMS

def main(a):
 sp=a.predictions/'SEALED.json';assert sha(sp)==a.expected_seal_sha256;ps=read(sp);ss=read(a.source/'SEALED.json')
 assert ps['state']=='sealed_R10_fixedq0_direct_geometry_before_GT' and ps['n_arms']==4 and ps['source_seal_sha256']==sha(a.source/'SEALED.json')
 assert ps['same_support_and_acceptance_full_noRGB'] and ps['all_queries_fixed_original_q0'] and ps['no_GT_or_parameter_update'] and ps['n_samples']==ss['n_samples']
 for root,s in [(a.predictions,ps),(a.source,ss)]:
  for rel,h in s['files'].items():assert sha(bounded(root,rel))==h
 for rel,h in ps['program_sha256'].items():assert sha(bounded(a.program_root,rel))==h
 assert ps['manifest_sha256']==ss['manifest_sha256']==sha(a.manifest)
 samples=read(a.manifest);trace={t['id']:t for t in read(a.predictions/'trace.json')};assert len(samples)==ps['n_samples'] and set(trace)=={r['id'] for r in samples}
 cfg=ps['geometry_config'];counts=collections.defaultdict(collections.Counter);statuses=collections.defaultdict(collections.Counter);provenance=collections.defaultdict(collections.Counter);gates=collections.Counter();maxcenterexcess=0.;maxsizeexcess=0.;sourceunchangedfilecount=0
 for sample in samples:
  sid=sample['id'];d=prediction(a.predictions/(sid+'.npz'),a.source/'source_cache'/(sid+'.npz'));t=trace[sid];source=d['arm_boxes_raw6'][0];raw=d['arm_boxes_raw6'];axes=d['acceptance_mask6'];g=t['geometry'];diag=g.get('diagnostics',{});q=int(d['q_source']);assert t['q_source']==q and t['camera_fields_only_allowlist'] and t['language_conditions_original_frozen_q0_only_no_new_text_localizer']
  assert t['source_cache_file_sha256']==ss['files']['source_cache/'+sid+'.npz'] and t['observation_file_sha256']==ss['files']['observations/'+sid+'.npz']
  with np.load(a.source/'observations'/(sid+'.npz'),allow_pickle=False) as z:pc=z['sampled_points_model'];ix=z['raw_indices'];unique=np.unique(pc,axis=0)
  assert t['model_input_rows']==len(pc) and t['unique_raw_indices']==len(np.unique(ix)) and t['geometry_unique_XYZ_rows']==len(unique) and t['geometry_unique_XYZ_sha256']==__import__('hashlib').sha256(np.ascontiguousarray(unique).tobytes()).hexdigest()
  assert np.array_equal(d['pre_constraint_geometry6'][2].astype(np.float32),raw[2]) and not np.any(axes[:,3:]&~axes[:,:3])
  if 'support_ids_by_arm' in g:
   ids=[np.asarray(v,np.int64) for v in g['support_ids_by_arm']];assert len(ids)==4 and np.array_equal(ids[1],ids[2]) and set(ids[1])<=set(ids[3])
   for v in ids:assert len(v)==len(np.unique(v)) and set(v)<=set(ix)
  if not d['RGB_mask2d_private'].any():assert not axes.any() and np.array_equal(raw,np.broadcast_to(source,raw.shape))
  gates[sample['platform']+'|'+str(diag.get('global_gate_reason',diag.get('reason','unknown')))]+=1
  for j,arm in enumerate(ARMS):
   key=sample['platform']+'|'+arm;counts[key]['n']+=1;counts[key]['accepted_any']+=int(axes[j].any());counts[key]['actual_changed']+=int(d['actual_changed_axes'][j].any());counts[key]['center_changed']+=int(d['actual_changed_axes'][j,:3].any());counts[key]['size_changed']+=int(d['actual_changed_axes'][j,3:].any());statuses[key][str(d['arm_status'][j])]+=1
   for axis,label in enumerate(d['provenance'][j].tolist()):provenance[key+'|axis'+str(axis)][str(label)]+=1
   if (source[3:]>0).all():
    limit=np.minimum(cfg['center_bound_fraction']*source[3:],cfg['center_bound_cap_m']);excess=float(np.maximum(abs(raw[j,:3].astype(float)-source[:3])-limit,0).max());maxcenterexcess=max(maxcenterexcess,excess);assert excess<=5e-5
    sizeex=float(np.maximum(abs(np.log(raw[j,3:].astype(float)/source[3:]))-cfg['size_log_bound'],0).max());maxsizeexcess=max(maxsizeexcess,sizeex);assert sizeex<=5e-5
   if j>0 and 'solver' in diag:
    sol=diag['solver'][arm];assert str(d['arm_status'][j])==sol['status']
    if sol['numerical_invalid']:assert np.array_equal(raw[j],source)
    elif not sol['success']:assert str(d['arm_status'][j])=='finite_nonconverged_bounded_result'
  sourceunchangedfilecount+=1
 assert ps['source_model_state_fingerprint']==ss['source_model_state_fingerprint'] and ps['strict_FP32_flags']==ss['strict_FP32_flags']
 out={'state':'independent_actual_R10_geometry4_unlabelled_audit_pass','auditor_sha256':sha(Path(__file__)),'prediction_contract_scorer_sha256':sha(Path(__file__).with_name('score_geometry.py')),'n':ps['n_samples'],'prediction_seal_sha256':sha(sp),'source_capture_seal_sha256':sha(a.source/'SEALED.json'),'source_files_unchanged_cases':sourceunchangedfilecount,'counts':{k:dict(v) for k,v in counts.items()},'status_histograms':{k:dict(v) for k,v in statuses.items()},'gate_histogram':dict(gates),'provenance_counts':{k:dict(v) for k,v in provenance.items()},'max_center_bound_FP32_excess_m':maxcenterexcess,'max_logsize_bound_FP32_excess':maxsizeexcess,'common_q0_support_acceptance_preconstraint_verified':True,'physical_unique_support_not_replacementmultiplicity_verified':True,'targetGT_scoringindex_or_metadata_contents_read':False,'no_IoU_GPU_model_or_G_reexecution':True,'scope':'Actual engineering and provenance audit only, not pseudo-quality or physical object/outerface correctness. Wholeobservations, no heldout claim.'};assert not a.output.exists();dump(a.output,out);print(json.dumps({k:v for k,v in out.items() if k!='provenance_counts'}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('predictions','source','manifest','program-root','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--expected-seal-sha256',required=True);main(p.parse_args())
