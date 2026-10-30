"""Postseal finite mechanism/gate review; no new IoU or annotation access."""
from pathlib import Path
import json,hashlib,collections,math,argparse
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main(a):
 ps=read(a.predictions/'SEALED.json');tr=read(a.predictions/'trace.json');mr=read(a.mechanism/'SUMMARY.json');ms=read(a.mechanism/'SEALED.json');res=read(a.scored/'RESULTS_PRIVATE.json');audit=read(a.score_audit)
 assert audit['state']=='PASS_R11_INDEPENDENT_SCALAR_REDUCTION_AUDIT' and audit['results_sha256']==sha(a.scored/'RESULTS_PRIVATE.json')
 for n,h in ms['files'].items():assert sha(a.mechanism/n)==h
 assert mr['binding']['geometry_seal_sha256']==sha(a.predictions/'SEALED.json')==audit['prediction_seal_sha256']
 checks=0;maxerr=0.
 def compare(x,y):
  nonlocal checks,maxerr
  checks+=1
  if isinstance(x,dict):
   for k,v in x.items():compare(v,y[k])
  elif isinstance(x,list):
   assert len(x)==len(y)
   for u,v in zip(x,y):compare(u,v)
  elif isinstance(x,(float,int)) and isinstance(y,(float,int)):
   e=abs(x-y);maxerr=max(maxerr,e);assert e<1e-12,(x,y)
  else:assert x==y,(x,y)
 for platform in ('all','drone','quad'):
  tt=[r for r in tr if platform=='all' or r['platform']==platform];C=[r for r in tt if r['base_actual_change']];r=[x['reliability'] for x in C];out=mr['all'] if platform=='all' else mr['by_platform'][platform]
  effective=[x['case_reliability'] for x in r];raw=[x.get('raw_case_reliability') for x in r]
  def stats(vals):
   defined=[v for v in vals if v is not None];return {'n':len(vals),'NA':len(vals)-len(defined),'defined':len(defined),'zero_count':sum(v==0 for v in defined),'distinct_exact_values':len(set(defined))}
  expected={'n':len(tt),'baseC':len(C),'association_all':dict(collections.Counter(x['reliability']['association_status'] for x in tt)),'association_in_C':dict(collections.Counter(x['association_status'] for x in r)),'case_effective_in_C':stats(effective),'case_raw_in_C':stats(raw),'component_conflict_cases':sum(x['features']['component_ambiguity']['conflict_observed'] for x in r),'zero_substantial_component_cases':sum(x['features']['component_ambiguity']['substantial_component_count']==0 for x in r),'raw_positive_RGB_effective_zero_cases':sum(x['raw_RGB_weight']>0 and x['RGB_weight']==0 for x in r),'actual_coordinate_axes6_in_C':[sum(x['features']['base_actual_changed_axis6'][j] for x in r) for j in range(6)]}
  expected['raw_positive_face_effective_zero_count']=sum(x['raw_face_weights'][k]>0 and x['face_weights'][k]==0 for x in r for k in x['active_face_keys'])
  expected['effective_defined_actual_axes']=sum(bool(d) and bool(ch) for x in r for d,ch in zip(x['axis_reliability_defined6'],x['features']['base_actual_changed_axis6']))
  expected['raw_defined_actual_axes']=sum(bool(d) and bool(ch) for x in r for d,ch in zip(x['raw_axis_reliability_defined6'],x['features']['base_actual_changed_axis6']))
  compare(expected,out)
 compare(mr['uniform_means_and_exact_denominators_recomputed'],ps['uniform_means_perplatform'])
 bridge=read(a.predictions/'ACTUAL_WEIGHT_BRIDGES.json');compare({'n':len(bridge),'allone_base_bitexact_pass':sum(x['allone_base_bitexact'] for x in bridge),'allzero_source_bitexact_pass':sum(x['allzero_source_bitexact'] for x in bridge)},mr['actual_weight_bridges'])
 table={(x['platform'],x['arm']):x for x in res['summary'] if x['population']=='all'};paired={(x['platform'],x['arm'],x['reference']):x for x in res['paired_contrasts'] if x['population']=='all'}
 gates={};numbers={}
 for platform in ('drone','quad'):
  u=res['discrimination'][platform]['coordinate_primary_IoU'];case=res['discrimination'][platform]['case_originalG_IoU'];risks={x['policy']:x for x in res['coverage_risk_primary50'] if x['platform']==platform};r50=risks['reliable_50'];rpair=paired[platform,'reliable_50','source'];w=paired[platform,'weighted_constraints','source'];wu=paired[platform,'weighted_constraints','uniform_weighted_constraints']
  improve=lambda p:p['IoU_gain']['mean']>0 and (p['center_error_reduction_m']['mean']>0 or p['size_error_reduction_m']['mean']>0)
  baselines={name:r50['risk']['harmful_mass']<v['risk']['harmful_mass'] for name,v in risks.items() if name!='reliable_50'}
  gates[platform]={'effective_defined_two_task_classes':u['ranking']['both_classes'],'reliable50_positive_task_and_coordinate':improve(rpair),'reliable50_strictly_lower_harmful_mass_than_each_control':all(baselines.values()),'percontrol_harmful_mass_pass':baselines,'weighted_positive_task_coordinate_and_task_over_uniform':improve(w) and wu['IoU_gain']['mean']>0}
  numbers[platform]={'n':table[platform,'source']['n'],'source_mIoU_percent':100*table[platform,'source']['mean_iou'],'weighted_minus_source_pp':100*w['IoU_gain']['mean'],'weighted_minus_uniform_pp':100*wu['IoU_gain']['mean'],'reliable50_minus_source_pp':100*rpair['IoU_gain']['mean'],'effective_coordinate_task_classes':u['effect_counts_defined'],'undefined_coordinate_task_classes':u['effect_counts_undefined'],'primary_AUC':u['ranking']['helpful_positive_AUROC_casebalanced'],'reliable50_help_harm_zero':{k:r50['risk'][k] for k in ('beneficial_n','harmful_n','zeroeffect_n')},'RGB_effect_classes':res['discrimination'][platform]['RGB_primary_matched_IoU']['effect_counts_all_candidate']}
 passed=all(v['effective_defined_two_task_classes'] and v['reliable50_positive_task_and_coordinate'] and v['reliable50_strictly_lower_harmful_mass_than_each_control'] and v['weighted_positive_task_coordinate_and_task_over_uniform'] for v in gates.values())
 assert not passed
 out={'state':'PASS_INDEPENDENT_MECHANISM_AND_PRELOCKED_GATE_REVIEW','prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'scored_seal_sha256':sha(a.scored/'SEALED.json'),'score_audit_sha256':sha(a.score_audit),'mechanism_seal_sha256':sha(a.mechanism/'SEALED.json'),'mechanism_summary_sha256':sha(a.mechanism/'SUMMARY.json'),'progression_preplan_sha256':sha(a.preplan),'auditor_sha256':sha(Path(__file__)),'mechanism_numeric_or_exact_checks':checks,'mechanism_max_difference':maxerr,'gate_by_platform':gates,'all_prerequisites_pass':passed,'decision':'Current reliability formulation fails prelocked resource gate; no neuralTTA stage. Do not change primary scores or coverage after labels.','key_numbers':numbers,'interpretation':['Drone taskAUC1 is based on onlyone defined helpful andone defined harmful unit, with4defined zeroeffects and5undefined units; not robust discrimination proof.','Quad solehelpful primary taskeffect is scoreundefined; defined subset lacks helpfulclass, AUROC/AP NA rather than0 or1.','Both RGB matchedterm assays have no harmful taskeffect, so primary two-class discrimination is unidentifiable.','Component conflict factor did not activate on currentC;4association conflicts originate from zero crosspartition maskIoU. Zero substantialcomponents and wrongsourceprior remain important undefined boundaries.','Weighted recovers relative to harmful unconditionalG but not source+uniform on bothplatforms; shrinkage is not established reliability benefit.','Centererror improvements can coexist with worse taskIoU; report both without switching primary label.','Only2Drone/3Quad scenes; fixedcohort randomreference not pvalue and no broad reliability impossibility claim.'],'no_new_GT_metadata_IoU_model_or_GPU_reads':True}
 a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({'state':out['state'],'gate_pass':passed,'checks':checks,'sha256':sha(a.output)}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('predictions','scored','mechanism','score-audit','preplan','output'):p.add_argument('--'+k,type=Path,required=True)
 main(p.parse_args())
