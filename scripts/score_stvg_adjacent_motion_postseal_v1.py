"""All-arm, all-candidate official/independent post-seal scoring; no model calls."""
import sys,copy,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from scripts.stvg_temporal_caption_query_v1 import read,write,sha
from scripts import score_stvg_motion_guided_postseal_v1 as legacy
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal,_metric_functions
from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
B=ROOT/'artifacts/stvg_adjacent_motion_opd_v1';OLD=ROOT/'artifacts/stvg_motion_guided_instance_opd_v1'
ALGORITHMS=('direct','iou','motion_iou')
def main():
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();legacy.ALGORITHMS=ALGORITHMS
 roster={r['case_id']:r for r in read(B/'ROSTER_LOCK.json')['records']};assert len(roster)==8
 lock=read(B/'METHOD_REFERENCE_LOCK.json');refs={r['case_id']:r for r in lock['records']};bar=read(B/'METHOD_STUDENT_GLOBAL_BARRIER.json');rt=read(B/'STUDENT_RUNTIME.json')
 assert bar['status']=='complete' and bar['count']==8 and bar['GT_read'] is False and bar['actual_fit_count']==lock['expected_fit_count'] and bar['runtime_sha256']==sha(B/'STUDENT_RUNTIME.json') and bar['reference_lock_sha256']==sha(B/'METHOD_REFERENCE_LOCK.json')
 assert read(B/'ROOT_METHOD_MATH_READBACK.json')['status']=='pass'
 for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/p)==h,p
 inpbar=read(B/'CANDIDATE_INPUT_BARRIER.json');inputs={r['case_id']:dict(r,candidates=r['candidate_geometry']) for r in inpbar['records']};assert set(inputs)==set(roster)
 for f in inpbar['files']:assert sha(ROOT/f['path'])==f['sha256'] and (ROOT/f['path']).stat().st_size==f['bytes']
 outputs={};sources={};oldseal=legacy._verify_old_temporal_seal()
 for f in bar['files']:
  p=ROOT/f['path'];assert sha(p)==f['sha256'] and p.stat().st_size==f['bytes'];cid=p.stem;assert cid in roster and cid not in outputs
  z=torch.load(p,map_location='cpu',weights_only=False);outputs[cid]=z;r=roster[cid];assert sha(ROOT/r['source_case_path'])==r['source_case_sha256'];c=load_source_case(ROOT/r['source_case_path']);sources[cid]=c
  legacy._check_output_identity(z,cid,r,c,sha(B/'METHOD_REFERENCE_LOCK.json'));legacy._assert_cpu_tree(z)
  prior=torch.load(oldseal['files'][cid]['path'],map_location='cpu',weights_only=False)
  for arm in ['F','Projection','T']:
   legacy._tensor_equal(z['arms'][arm]['boxes'],prior['arms'][arm]['boxes'],cid+arm);assert z['arms'][arm]['interval']==prior['arms'][arm]['interval']
  for alg in ALGORITHMS:
   arm='T_teacher_'+alg;assert z['arms'][arm]['interval']==prior['arms']['T']['interval']
   expected=prior['arms']['T']['boxes'] if refs[cid]['references']['teacher']['no_update'] else z['arms']['teacher_'+alg]['boxes'];legacy._tensor_equal(z['arms'][arm]['boxes'],expected,cid+arm)
 assert set(outputs)==set(roster)
 truths,bindings=_truths_after_seal({cid:dict(dataset=c['dataset'],query_ordinal=c['query_ordinal']) for cid,c in sources.items()});official,independent,contract=_metric_functions()
 cases=[];metrics=[];comparisons=[];candidate_rows=[];display_rows=[];oracle=[];motion=[]
 for cid in sorted(roster):
  r=roster[cid];z=outputs[cid];c=sources[cid];inp=inputs[cid]
  row,m,cmp=legacy._score_case(cid,'old8',r,c,z,refs[cid],inp,truths[cid],official,independent,'method');cases.append(row);metrics+=m;comparisons+=cmp
  display_rows+=legacy._score_candidates(cid,r,c,inp,z,truths[cid],'method')
  geo=copy.deepcopy(inp)
  for cand in geo['candidates']:
   cand['boxes_xyxy']=cand['geometry_boxes_xyxy'];cand['valid_observation_count']=sum(b is not None for b in cand['boxes_xyxy'])
  cr=legacy._score_candidates(cid,r,c,geo,z,truths[cid],'method');candidate_rows+=cr
  scorekey='common_denominator_original4_GT_known_missing_as_zero_candidate_IoU'
  valid=[x for x in cr if x[scorekey] is not None];display=[x for x in valid if x['candidate_id'] in inp['displayed_candidate_ids']];sel=refs[cid]['references']['teacher']['selected_candidate_id'];chosen=next((x for x in cr if x['candidate_id']==sel),None)
  oracle.append(dict(case_id=cid,dataset=r['dataset'],candidate_oracle_known_GT_common_support=max((x[scorekey] for x in valid),default=None),display_oracle_known_GT_common_support=max((x[scorekey] for x in display),default=None),chosen_geometry_known_GT_common_support=chosen[scorekey] if chosen else None,Source_original_known_GT_common_support=cr[0]['F_source_GT_IoU_mean_all_GT_known_original4'],GT_used_only_after_all_method_seals=True))
  for name,fit in z['fits'].items():
   if fit.get('algorithm')=='motion_iou':
    for rr in fit['rounds']:motion.append(dict(case_id=cid,fit=name,step=rr['step'],opd_loss=rr['opd_loss'],motion_loss=rr['motion_loss'],total_loss=rr['total_loss'],lambda_motion=rr['lambda_motion'],updated=rr['updated'],motion_valid_pair_count=rr['motion_valid_pair_count'],raw_gradient_norm=rr['raw_gradient_norm']))
 # Old V03 counterfactual is a clearly separated exact, already-scored reuse.
 oldbar=read(OLD/'REFERENCE_CONTROL_STUDENT_GLOBAL_BARRIER.json')
 for f in oldbar['files']:assert sha(ROOT/f['path'])==f['sha256'] and (ROOT/f['path']).stat().st_size==f['bytes']
 oldroot=read(OLD/'ROOT_POSTSEAL_REPORT.json');assert read(OLD/'ROOT_CLOSING.json')['status']=='complete'
 reused=dict(status='exact_prior_evidence_reuse',new_fits_executed=0,barrier_sha256=sha(OLD/'REFERENCE_CONTROL_STUDENT_GLOBAL_BARRIER.json'),report_sha256=sha(OLD/'ROOT_POSTSEAL_REPORT.json'),groups=oldroot['V03_reference_control_groups'],anchor_rows=oldroot['reference_control_locked_anchor_rows'],scope='old Source/ID2/ID3 geometry only; not new tracklet identity')
 out=dict(status='pass',schema='stvg_adjacent_motion_postseal_v1',GT_read=True,GT_used_for_method_update=False,model_loaded=False,GPU_initialized=False,case_rows=cases,metric_rows=metrics,arm_comparison_rows=comparisons,fit_movement_rows=[f for r in cases for f in r['fit_movement_rows']],candidate_geometry_rows=candidate_rows,candidate_display_rows=display_rows,candidate_oracle_rows=oracle,motion_round_rows=motion,groups=legacy._group_summaries(cases,comparisons),V03_prior_control=reused,metric_count=len(metrics),candidate_row_count=len(candidate_rows),actual_fit_count=bar['actual_fit_count'],optimizer_steps=sum(r['optimizer_steps'] for r in cases),metric_max_abs_error=max(r['official_independent_max_abs_error'] for r in metrics),GT_source_bindings=bindings,student_barrier_sha256=sha(B/'METHOD_STUDENT_GLOBAL_BARRIER.json'),scorer_sha256=sha(Path(__file__)),time=time.time())
 write(B/'ROOT_POSTSEAL_REPORT.json',out);print(json.dumps(dict(metric_count=len(metrics),fit_count=out['actual_fit_count'],optimizer_steps=out['optimizer_steps'],groups=out['groups'],max_error=out['metric_max_abs_error']),indent=2))
if __name__=='__main__':main()
