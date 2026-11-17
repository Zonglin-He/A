"""Posthoc all-case decomposition from ALREADY SCORED fixed predictions.
No new models, geometry/GT evaluation, labels for adaptation, or method selection.
"""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2];H=R/'research/round127'
def main():
 assert json.loads((H/'SCALAR_AUDIT.json').read_text())['n']==2304
 rows=[r for p in (H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json') for r in json.loads(p.read_text())];out=[]
 for co in ['DEV','EXP']:
  rr=[r for r in rows if r['cohort']==co];stats=dict(cohort=co,n=len(rr),harmful_updates=0,harmful_with_direct_teacher_lower_than_zero=0,harmful_with_direct_teacher_not_lower_than_zero=0,harmful_with_direct_teacher_same_query_as_zero=0,helpful_updates=0,helpful_beyond_direct_teacher_IoU=0,helpful_not_beyond_direct_teacher_IoU=0,helpful_at_most_original_source_IoU=0,helpful_exceeds_original_source_IoU=0)
  for r in rr:
   a=r['arms'];z=a['zero_F']['IoU'];m=a['fab_full_F']['IoU'];t=a['direct_full_F']['IoU'];s=a['original_source']['IoU']
   if m<z-1e-12:
    stats['harmful_updates']+=1;stats['harmful_with_direct_teacher_lower_than_zero' if t<z-1e-12 else 'harmful_with_direct_teacher_not_lower_than_zero']+=1;stats['harmful_with_direct_teacher_same_query_as_zero']+=int(a['direct_full_F']['query_PRIVATE']==a['zero_F']['query_PRIVATE'])
   if m>z+1e-12:
    stats['helpful_updates']+=1;stats['helpful_beyond_direct_teacher_IoU' if m>t+1e-12 else 'helpful_not_beyond_direct_teacher_IoU']+=1;stats['helpful_exceeds_original_source_IoU' if m>s+1e-12 else 'helpful_at_most_original_source_IoU']+=1
  assert stats['harmful_updates']==stats['harmful_with_direct_teacher_lower_than_zero']+stats['harmful_with_direct_teacher_not_lower_than_zero'];assert stats['helpful_updates']==stats['helpful_beyond_direct_teacher_IoU']+stats['helpful_not_beyond_direct_teacher_IoU'];out.append(stats)
 doc={'state':'POSTHOC_DESCRIPTIVE_ONLY_EXISTING_SCORED_ROWS','no_new_task_GT_evaluation':True,'predictions_or_method_changed':False,'identity_truth':'IoU ordering and query identity do not certify physical instance identity','cohorts':out}
 (H/'TEACHER_EXECUTION_DIAGNOSTIC.json').write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps(doc,indent=2),flush=True)
if __name__=='__main__':main()
