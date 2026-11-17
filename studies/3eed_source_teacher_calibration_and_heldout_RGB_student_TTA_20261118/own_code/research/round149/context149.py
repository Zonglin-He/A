"""Postseal matched-eligible diagnostics only. No new predictor or GT geometry."""
from common149 import *
import numpy as np,re

def main():
 check();assert rd(H/'SCALAR_AUDIT.json')['status']=='PASS';rows=rd(H/'ALL_ROWS_PRIVATE.json');di={(r['source'],r['target'],r['id_PRIVATE']):r for r in rd(H/'DIAGNOSTIC_ROWS_PRIVATE.json')};out=[]
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)];rec=dict(source=src,target=tgt,eligible=0,recoverable_Azero_errors=0,good_C_recoverable=0,teacher_C_good_recoverable=0,TTA_good_on_teacher_recoverable=0,teacher_C_harms_Azero_good=0,A_zero_outside_S=0,TTA_outside_S=0,insideS_to_outsideS=0,outsideS_to_insideS=0,parent_Cgroup_contains_good_but_selected_bad=0,teacher_member_better_parent=0,teacher_member_worse_parent=0)
  for r in rr:
   d=di[src,tgt,r['id_PRIVATE']]
   if not d['eligible']:continue
   rec['eligible']+=1;ev=np.load(B/'evidence_PRIVATE'/(src+'_to_'+tgt)/(r['id_PRIVATE']+'.npz'));q0=r['queries_PRIVATE']['norm_Aimage'];q1=r['queries_PRIVATE']['cal_A_TTA'];x=bool(ev['S'][ev['assignment'][q0]]);y=bool(ev['S'][ev['assignment'][q1]]);rec['A_zero_outside_S']+=not x;rec['TTA_outside_S']+=not y;rec['insideS_to_outsideS']+=x and not y;rec['outsideS_to_insideS']+=not x and y;recover=bool(d['A256_has_good25'] and not d['A_zero_good25']);rec['recoverable_Azero_errors']+=recover;rec['good_C_recoverable']+=recover and d['good25_in_C'];rec['teacher_C_good_recoverable']+=recover and d['teacher_C_good25'];rec['TTA_good_on_teacher_recoverable']+=recover and d['teacher_C_good25'] and d['adapted_good25'];rec['teacher_C_harms_Azero_good']+=d['A_zero_good25'] and not d['teacher_C_good25'];rec['parent_Cgroup_contains_good_but_selected_bad']+=d['parent_group_in_C'] and d['parent_group_has_good25'] and not d['parent_selected_good25'];rec['teacher_member_better_parent']+=d['parent_group_in_C'] and d['teacher_member_better'];rec['teacher_member_worse_parent']+=d['parent_group_in_C'] and d['teacher_member_worse']
  out.append({k:int(v) if isinstance(v,(int,np.integer,np.bool_)) else v for k,v in rec.items()})
 totals={k:sum(r[k] for r in out) for k in out[0] if k not in ['source','target']};obj=dict(status='POSTSEAL_DESCRIPTIVE_ONLY',rows=out,totals=totals,no_new_predictor_or_task_geometry=True,GT_IoU_not_verified_instance_identity=True,denominator='original B/eligible764; sourceAimage zero is reference for recoverable counts',no_GT_selected_deployment_filter=True);dump(H/'MATCHED_TEACHER_DIAGNOSTIC.json',obj)
 note='\n\n## Matched eligible teacher diagnostic\n\n'+json.dumps(obj,indent=2)+'\n\nThese postseal counts are descriptive, not new method outputs. A poor teacher conditional leader does not imply every probability component is wrong. Outside-S outputs are not automatically update-created errors; before/after transitions are explicitly separated. Within-group member information is weak on this cohort; do not launch a broader member KL merely because the omitted term is computable.\n'
 O=H/'publication/public_export_attempt001'
 for p in [H/'REPORT.md',O/'REPORT.md']:p.write_text(p.read_text()+note)
 zh='\n\n## 本轮进一步定位到的监督缺口\n\n在同一 764 条可更新输入中，A 图像零步模型有 '+str(totals['recoverable_Azero_errors'])+' 条错误在 A256 中仍有 Acc@25 达标框；其中 '+str(totals['good_C_recoverable'])+' 条的保留集合 C 包含达标框，但教师在 C 内真正选中达标框的只有 '+str(totals['teacher_C_good_recoverable'])+' 条。TTA 在这批教师可救回的输入中最终达标 '+str(totals['TTA_good_on_teacher_recoverable'])+' 条。另一方面，教师会将 '+str(totals['teacher_C_harms_Azero_good'])+' 条零步达标输入指向不达标框。这些是几何代理统计，不是独立核验的物体身份标签。\n\n最终第一名在 S 外的数量，零步为 '+str(totals['A_zero_outside_S'])+' 条，更新后为 '+str(totals['TTA_outside_S'])+' 条；从 S 内切到 S 外 '+str(totals['insideS_to_outsideS'])+' 条，反向切入 '+str(totals['outsideS_to_insideS'])+' 条。因此不能将所有集合外输出都说成更新制造的错误，也不能将未知候选强行当负例。\n\n旧 R145 已选中的 C 内候选组，只有 '+str(totals['parent_Cgroup_contains_good_but_selected_bad'])+' 条存在组内达标框却选中不达标成员；冻结教师的组内成员偏好相对旧 R145 有 '+str(totals['teacher_member_better_parent'])+' 条 IoU 改善、'+str(totals['teacher_member_worse_parent'])+' 条恶化。当前不支持把新增组内 KL 当作优先修复。核心缺口仍是：保留了可能正确的候选，不等于已有教师能分清这些候选。\n'
 p=O/'REPORT_ZH.md';p.write_text(p.read_text()+zh);dump(O/'MATCHED_TEACHER_DIAGNOSTIC.json',obj);q=O/'own_code/research/round149/context149.py';q.parent.mkdir(parents=True,exist_ok=True);q.write_text(Path(__file__).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'));tokens=set()
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  for ob in [e['row'],e['A_sensor']]+([e['pair_PRIVATE']['B']] if e['has_B'] else []):
   for k in ['id','caption','scene','imagepath','pointpath','metapath']:
    v=ob.get(k)
    if isinstance(v,str) and len(v)>=12:tokens.add(v)
  tokens.add(e['content_group_PRIVATE'])
 for e in rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json'):
  for k in ['id','caption','scene','imagepath','pointpath','metapath']:
   v=e['entry']['row'].get(k)
   if isinstance(v,str) and len(v)>=12:tokens.add(v)
 for p in O.rglob('*'):
  if p.is_file():
   t=p.read_text();assert 'LOCAL_HOME/' not in t and not any(x in t for x in tokens),p;assert not re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})',t)
 dump(H/'POSTSEAL_EXPORT_AUDIT.json',dict(status='PASS',private_literals_scanned=len(tokens),all_public_files_rescanned=True));dump(O/'MANIFEST.json',dict(prefix='studies/3eed_source_teacher_calibration_and_heldout_RGB_student_TTA_20261118',files=[dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']));print('MATCHED149',totals,flush=True)
if __name__=='__main__':main()
