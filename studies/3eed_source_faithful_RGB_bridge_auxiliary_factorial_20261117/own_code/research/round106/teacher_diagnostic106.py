"""Postseal teacher/endpoint mechanism diagnostic, not a label-based filter."""
from pathlib import Path
import json,hashlib,sys,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round106';B=R/'crossplatform/round106';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 matrix=rd(B/'MATRIX_SEALED.json');assert rd(H/'SCALAR_AUDIT.json')['n']==768;summary=[];allchanges=[]
 for name in matrix['directions']:
  src,_,tgt=name.split('_');rr=rd(H/'scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'));rows={r['id']:r for r in rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json')};gt=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');teacherI=[];fallbackI=[];signed=[]
  for r in rr:
   sid=r['id_PRIVATE'];s=rd(B/name/'records_PRIVATE'/(sid+'.json'));q0=r['arms']['real_frozen_fusedABC']['query_PRIVATE'];q1=r['arms']['real_TTA_fusedABC']['query_PRIVATE'];fit=s['fits'].get('real');q=np.array(fit['target']) if fit else None
   if q is None:teacherI.append(r['arms']['prepared_frozen']['IoU']);fallbackI.append(True);continue
   z=np.load(B/name/'samples_PRIVATE'/(sid+'.npz'));bs=z['source_boxes'].astype(float);bs[:,3:]=np.maximum(bs[:,3:],1e-6);ys=many_iou(np.c_[bs,np.zeros(256)],np.array(gt[rows[sid]['anno_index']]['source_GT7_float32']));teacherI.append(float(ys[q.argmax()]));fallbackI.append(False)
   if q0!=q1:
    delta=r['arms']['real_TTA_fusedABC']['IoU']-r['arms']['real_frozen_fusedABC']['IoU'];margin=float(np.log(q[q1])-np.log(q[q0]));allchanges.append({'direction':name,'row_PRIVATE':sid,'change_IoU':delta,'teacher_log_preference_for_updated_over_frozen':margin,'teacher_top1_IoU':float(ys[q.argmax()]),'frozen_IoU':r['arms']['real_frozen_fusedABC']['IoU'],'updated_IoU':r['arms']['real_TTA_fusedABC']['IoU'],'teacher_directly_supports_updated':margin>0,'fit':r['fits']['real']});signed.append((delta,margin))
  summary.append({'direction':name,'teacher_mean_AB_mIoU_percent_all_with_prepared_fallback':float(np.mean(teacherI)*100),'teacher_available_n':sum(not x for x in fallbackI),'Top1_switches_vs_frozen_fusedABC':len(signed),'harmful_changes_teacher_supports_new':sum(d< -1e-12 and m>0 for d,m in signed),'harmful_changes_teacher_opposes_new':sum(d< -1e-12 and m<=0 for d,m in signed),'beneficial_changes_teacher_supports_new':sum(d>1e-12 and m>0 for d,m in signed),'beneficial_changes_teacher_opposes_new':sum(d>1e-12 and m<=0 for d,m in signed)})
 dump(H/'TEACHER_DIAGNOSTIC_SUMMARY.json',{'groups':summary,'postseal_descriptive_not_method_selection':True,'teacher_targets_already_sealed_before_GT':True,'candidate_changes_not_identity_truth':True});dump(H/'TEACHER_CHANGES_PRIVATE.json',allchanges);p=H/'REPORT.md';p.write_text(p.read_text()+'\n## 已封存监督与最终读出的分解诊断\n\n'+json.dumps(rd(H/'TEACHER_DIAGNOSTIC_SUMMARY.json'),indent=2)+'\n\n这里只评价正式预测前已封存的q目标，属于事后机制诊断；不按GT筛选、反转监督或选择逐样本输出。\n');print('TEACHER106_DIAG',len(allchanges),flush=True)
if __name__=='__main__':main()
