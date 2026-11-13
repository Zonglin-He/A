"""Postseal controlled reader/update interactions from actual saved parameter patches."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round89';P=R/'crossplatform/round89/prediction336_attempt001';read=lambda p:json.loads(p.read_text());tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');rr=[]
for t in tasks:
 sid=t['id_PRIVATE'];a=np.load(P/'states_PRIVATE'/(sid+'_candidate_tta1.npz'),allow_pickle=False)
 for control in ['no_residual_tta1','query_tta1','continue_tta1','candidate_physical1','candidate_no_correspondence1']:
  b=np.load(P/'states_PRIVATE'/(sid+'_'+control+'.npz'),allow_pickle=False);diff=max(float(np.max(abs(a[k]-b[k]))) for k in a.files);rr.append({'row':t['row'],'platform':t['platform'],'control':control,'max_same_initial_parameter_difference':diff,'exact_same_body_update':diff==0,'candidate_tta1_IoU':t['IoU']['candidate_tta1'],'control_IoU':t['IoU'][control],'candidate_minus_control_pp':100*(t['IoU']['candidate_tta1']-t['IoU'][control]),'same_original_A256_boxes':True,'does_not_certify_identity':True})
(H/'SAVED_STEP_EFFECT_ROWS.json').write_text(json.dumps(rr,indent=2)+'\n');print({'rows':len(rr),'exact_same_update_different_IoU':sum(r['exact_same_body_update'] and abs(r['candidate_minus_control_pp'])>1e-8 for r in rr)})
