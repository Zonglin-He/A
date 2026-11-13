"""Independent post-completion matrix summary reduction, all rows and source identity checks."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round96';B=R/'crossplatform/round96';rd=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 S=H/'frozen_scores_attempt001';seal=rd(S/'SEALED.json')
 for f,v in seal['files'].items():assert sha(S/f)==v
 summary=rd(S/'ALL_METHOD_ROWS.json');receipt=rd(S/'SCORING_RECEIPT.json');matrix=rd(B/'frozen_matrix/MATRIX_SEALED.json');expected={(s,t) for s in ['waymo','drone','quad'] for t in ['waymo','drone','quad'] if s!=t};assert {(r['source'],r['target']) for r in matrix['directions']}==expected;total=0
 for d in matrix['directions']:
  src,tgt=d['source'],d['target'];rr=rd(S/(src+'_to_'+tgt+'_ALL_ROWS_PRIVATE.json'));assert len(rr)==d['n'] and all(r['source']==src and r['target']==tgt for r in rr);total+=len(rr)
  vals=[r for r in summary if r['source']==src and r['target']==tgt];assert len(vals)==10
  for v in vals:
   y=np.array([r['arms'][v['arm']]['IoU'] for r in rr]);assert v['n']==len(y) and v['mIoU_percent']==float(y.mean()*100) and v['Acc25_percent']==float((y>.25).mean()*100) and v['Acc50_percent']==float((y>.5).mean()*100)
   assert v['GT_class_span_assisted']==('_official_' in v['arm']) and v['TTA'] is False
   for r in rr:assert 0<=r['arms'][v['arm']]['IoU']<=1
 assert total==receipt['direction_task_inputs']==18620 and receipt['independent_general_3D_selected_checks']==total and receipt['max_abs_yaw_vs_general_3D_error']<1e-6 and receipt['FINAL_scored'] is False
 out={'state':'PASS_R96_SIX_DIRECTION_FULL_ROW_SUMMARIES_AND_ASSISTED_READOUT_SEPARATION','directions':6,'direction_task_inputs':total,'method_rows':len(summary),'primary':'last_text_logodds','FINAL_access':False,'TTA':False};(H/'MATRIX_SUMMARY_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
