"""Actual postseal image review; no identity labels or case-specific method selection."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2];H=R/'research/round110';P=R/'crossplatform/round110/atlas_PRIVATE/INDEX_PRIVATE.json'
def main():
 notes={'CASE0001':'Viewed: CE/oldpool favor an elevated tree/background region, whereas transported pool returns to the visible near van already supported by frozen BBF. This partly restores a useful fusion choice and changes the overlapping box; not independent identity certification.','CASE0002':'Viewed: old and new choices overlap the same visible SUV vicinity. The decrease is consistent with different overlapping boxes rather than a certified different-instance switch; exact 3D identity UNKNOWN.','CASE0007':'Viewed: old CE/geometric favors a different region near the neighboring vehicle, while transport returns toward the partly out-of-image SUV supported by prepared/fused BBF. Out-of-image geometry and best-candidate overlap limit interpretation; identity not certified.','CASE0011':'Viewed: old CE/geometric favors the left-rear dark vehicle region, while transport returns to a spatially separated prepared/fused choice farther in BEV and loses overlap. This erases useful CE correction; not merely smaller movement within the same box.'}
 idx=json.loads(P.read_text())
 for r in idx:
  if r['case'] in notes:r['identity_review']=notes[r['case']]
 P.write_text(json.dumps(idx,indent=2,ensure_ascii=False)+'\n');summary={'generated_pages':len(idx),'actually_viewed_pages':len(notes),'not_gold_identity_annotations':True,'unviewed_UNKNOWN':True,'observations':list(notes.values()),'no_case_specific_filter_or_GT_gate':True};(H/'ATLAS_REVIEW_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');f=H/'REPORT.md';f.write_text(f.read_text()+'\n## 逐例实际看图（封存后解释，不筛分母）\n\n'+json.dumps(summary,indent=2)+'\n\n正例同时包括从背景区域回到可见车辆、恢复冻结融合选择；负例既有同一车辆附近重复框的切换，也有抹掉原CE纠错后回到不同区域。不能将全部变化归为实例修复，也不能把矩配对有效等同于语义身份正确。\n')
if __name__=='__main__':main()
