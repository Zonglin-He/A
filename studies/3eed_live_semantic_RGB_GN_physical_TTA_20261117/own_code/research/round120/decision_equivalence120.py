"""Postseal categorical equivalence checks, no model/GT calculation."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round120';B=R/'crossplatform/round120';rd=lambda p:json.loads(Path(p).read_text());mat=rd(B/'MATRIX_SEALED.json');out=[]
comparisons=[('depth_trust_live_F','context_zero_F'),('depth_trust_live_F','depth_trust_frozenR_F'),('depth_trust_live_F','depth_trust_frozenLate_F'),('depth_trust_live_F','wrong_depth_trust_live_F'),('depth_trust_live_F','random_GN_trust_live_F'),('depth_trust_live_C','context_zero_C')]
for co in ['DEV','EXP']:
 rows=[r for f in (H/'scores_attempt001').glob(co+'_*_ALL_ROWS_PRIVATE.json') for r in rd(f)];n=len(rows)
 for a,b in comparisons:
  qdiff=sum(r['arms'][a]['query_PRIVATE']!=r['arms'][b]['query_PRIVATE'] for r in rows);ydiff=sum(abs(r['arms'][a]['IoU']-r['arms'][b]['IoU'])>1e-12 for r in rows);out.append({'cohort':co,'method':a,'control':b,'n':n,'query_changes':qdiff,'IoU_changes':ydiff,'reused_values_only':True})
(H/'DECISION_EQUIVALENCE_SUMMARY.json').write_text(json.dumps({'rows':out,'new_GT_scoring':0,'new_models':0,'FINAL_access':False},indent=2)+'\n');print(out,flush=True)
