"""Independent R11 no-label audit. No annotation-index/GT/model/GPU API."""
from pathlib import Path
import argparse,json,collections,time
import numpy as np
from score_reliability import sha,read,dump,bounded,verify_seal,prediction,validate_cohort,CONTROLS

def main(a):
 started=time.perf_counter();ps=read(a.predictions/'SEALED.json');ss=read(a.source/'SEALED.json')
 assert ps['state']=='sealed_R11_reliability11_and_atomic6_before_GT' and ss['state']=='sealed_R11_source_forward_and_input_observations_no_G_no_GT'
 assert ps['n_samples']==ss['n_samples']==320 and ps['n_arms']==11 and ps['atomic_arms_per_sample']==6
 assert ps['source_seal_sha256']==sha(a.source/'SEALED.json') and ps['manifest_sha256']==ss['manifest_sha256']==sha(a.manifest)
 assert ps['source_checkpoint_sha256']==ss['source_checkpoint_sha256']==sha(a.source_checkpoint)
 assert ps['source_model_state_fingerprint']==ss['source_model_state_fingerprint']
 assert ps['all_queries_fixed_original_q0'] and ps['all_reliability_and_rank_weights_sealed_before_GT'] and not ps['neural_parameter_updates_executed']
 for k in ('cudnn_allow_tf32','matmul_allow_tf32','cudnn_benchmark'):assert ss['strict_FP32_flags'][k] is False and ps['strict_FP32_flags'][k] is False
 verify_seal(a.predictions,ps);verify_seal(a.source,ss)
 assert ps['program_sha256_base']=='workspace_root'
 for name,digest in ps['program_sha256'].items():assert sha(bounded(a.program_root,name))==digest
 for name,digest in read(a.source/'registered_plan.json')['capture_program_sha256'].items():assert sha(bounded(a.program_root,name))==digest
 manifest=read(a.manifest);assert len(manifest)==320 and len({r['id'] for r in manifest})==320
 assert {r['id']+'.npz' for r in manifest}=={p.name for p in a.predictions.glob('*.npz')}
 preds={r['id']:prediction(a.predictions/(r['id']+'.npz'),a.source/'source_cache'/(r['id']+'.npz')) for r in manifest}
 control=read(a.predictions/'COHORT_CONTROL_LOCK.json');tr=read(a.predictions/'trace.json');means=validate_cohort(preds,manifest,control,tr)
 bridge=read(a.predictions/'ACTUAL_WEIGHT_BRIDGES.json');C={sid for sid,p in preds.items() if p['atomic_active6'].any()}
 assert {r['id'] for r in bridge}==C and len(bridge)==len(C)==ps['actual_allone_allzero_bridge_cases']
 assert all(r['allone_base_bitexact'] and r['allzero_source_bitexact'] for r in bridge)
 summary={}
 for platform in ('drone','quad'):
  rr=[preds[r['id']] for r in manifest if r['platform']==platform];counts=collections.Counter(str(r['association_status']) for r in rr)
  summary[platform]={'n':len(rr),'original_G_changed_n':sum(r['atomic_active6'].any() for r in rr),'original_coordinate_application_n_by_axis':[sum(r['atomic_active6'][j] for r in rr) for j in range(6)],'coordinate_score_defined_n_by_axis':[sum(r['atomic_active6'][j] and r['coordinate_defined6'][j] for r in rr) for j in range(6)],'association_histogram':dict(counts),'case_reliability_defined_n':sum(bool(r['case_defined']) for r in rr),'RGBterm_actual_changed_n':sum(not np.array_equal(r['arm_boxes_raw6'][1],r['arm_boxes_raw6'][2]) for r in rr),'RGB_score_defined_n':sum(bool(r['RGB_defined']) for r in rr),'weighted_changed_n':sum(not np.array_equal(r['arm_boxes_raw6'][3],r['arm_boxes_raw6'][0]) for r in rr),'uniform_changed_n':sum(not np.array_equal(r['arm_boxes_raw6'][4],r['arm_boxes_raw6'][0]) for r in rr),'weighted_changed_outside_originalC_n':sum(not r['atomic_active6'].any() and not np.array_equal(r['arm_boxes_raw6'][3],r['arm_boxes_raw6'][0]) for r in rr),'uniform_changed_outside_originalC_n':sum(not r['atomic_active6'].any() and not np.array_equal(r['arm_boxes_raw6'][4],r['arm_boxes_raw6'][0]) for r in rr),'policy_selected_n':{k:sum(r['gate_selection'][j] for r in rr) for j,k in enumerate(CONTROLS)}}
 out={'state':'PASS_R11_ACTUAL_UNLABELLED_PREDICTION_CONTRACT','n':320,'prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'source_capture_seal_sha256':sha(a.source/'SEALED.json'),'manifest_sha256':sha(a.manifest),'source_checkpoint_sha256':sha(a.source_checkpoint),'auditor_program_sha256':sha(Path(__file__)),'reader_program_sha256':sha(Path(__file__).with_name('score_reliability.py')),'code_files_and_all_sealed_files_rehashed':True,'fixedq0_original_full_off_atomic6_case_reliability_uniformmeans_exactK_bridges_pass':True,'uniform_means_input_only':means,'platforms':summary,'no_GT_index_annotation_media_model_or_GPU_reads':True,'counts_not_correctness_or_identity_evidence':True,'seconds':time.perf_counter()-started}
 def primitive(x):
  if isinstance(x,np.generic):return x.item()
  raise TypeError(type(x).__name__)
 a.output.parent.mkdir(parents=True,exist_ok=True);assert not a.output.exists();a.output.write_text(json.dumps(out,indent=2,default=primitive,allow_nan=False)+'\n');print(json.dumps({'state':out['state'],'output_sha256':sha(a.output)}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('predictions','source','manifest','source-checkpoint','program-root','output'):p.add_argument('--'+k,type=Path,required=True)
 main(p.parse_args())
