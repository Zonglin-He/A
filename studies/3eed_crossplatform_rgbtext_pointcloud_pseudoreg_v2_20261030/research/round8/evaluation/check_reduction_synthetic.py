"""Run independent reducer on the existing synthetic score fixture, no oldfile edits."""
from pathlib import Path
import copy,json
import score_crossdomain as S
import check_scoring_synthetic as F
import audit_scored_results as A

def main():
 captured=[];old_score=S.score_sample;old_dump=S.dump;directory=Path(__file__).resolve().parent
 def collect(*args,**kwargs):
  out=old_score(*args,**kwargs);captured.append(out[0]);return out
 def no_fixture_overwrite(path,obj):
  if Path(path)==directory/'SYNTHETIC_SCORING_FIXTURE.json':return
  old_dump(path,obj)
 try:S.score_sample=collect;S.dump=no_fixture_overwrite;F.main()
 finally:S.score_sample=old_score;S.dump=old_dump
 result={'n':len(captured),'rows_PRIVATE':captured,'summary':S.summaries(captured),'candidate_and_group_coverage':S.coverage_summaries(captured),'strict_threshold_disagreements':0,'max_selected_3D_vs_batch_abs_error':0.};checked=A.review(result);assert checked['aggregate_cells_checked']>0
 corrupt=copy.deepcopy(result);corrupt['summary'][0]['mean_iou']+=.01
 try:A.review(corrupt);raise RuntimeError('Wrong mean accepted')
 except AssertionError:pass
 receipt={'state':'synthetic_independent_score_reducer_pass','scorer_sha256':S.sha(Path(S.__file__)),'auditor_sha256':S.sha(Path(A.__file__)),'fixture_sha256':S.sha(Path(__file__)),'checks':{k:v for k,v in checked.items() if k!='signed_negative_counts_by_mode_arm_contrast'},'corrupt_summary_rejected':True,'no_real_model_GPU_targetGT_or_data_access':True};old_dump(directory/'SYNTHETIC_REDUCTION_FIXTURE.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
