"""Actual full finite-scope closing only after root/view/public/archive."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_LN_state_intervention_common_v1 import *
from scripts.close_stvg_opd_p7_phase_v1 import archive
def run():
 verify();names=['ROOT_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_QUALIFICATION.json','ROOT_VISUAL_REVIEW.json','FINAL_GITHUB_RECEIPT.json','PUBLIC_SCALAR_AUDIT.json']
 assert all(read(BASE/n)['status']=='pass' for n in names)
 assert read(BASE/'ROOT_VISUAL_REVIEW.json')['actual_pixel_view'] and read(BASE/'FINAL_GITHUB_RECEIPT.json')['actual_remote_bytes_SHA256_Git_blob_tree_verified']
 for rel,b in read(BASE/'ROOT_VISUAL_REVIEW.json')['PDF_files'].items():assert sha(ROOT/rel)==b['sha256'] and (ROOT/rel).stat().st_size==b['bytes']
 from scripts.run_stvg_opd_LN_state_intervention_v1 import standby
 standby();s=read(OUT/'SUMMARY.json');public=read(BASE/'FINAL_GITHUB_RECEIPT.json');root=read(BASE/'ROOT_READBACK.json');decision=read(BASE/'MECHANISM_DECISION.json')
 pieces=[]
 for ds in ['hc2','vidstg']:
  v=s['datasets'][ds]['own_U']['contrasts']['Inherited_order1_to_Inherited_order2'];pieces.append(ds+'两继承状态更新效用绝对差'+str(round(v['absolute_difference']['mean']*100,3))+'pp/符号反转'+str(v['sign_flips'])+'/32/双方至少1pp反转'+str(v['flip_both_at_least1pp'])+'/32')
 note='**2026-10-13｜单次固定LN State Intervention实际root/view/全部匿名公开完成，当前scope关闭。** 人类明确选原P9/P0每域32历史曝光parent，64query×Source/Inherited1/Inherited2=192新完整fit/4800轮，全global后GT；24完整资格8fresh capture/精确repeat及128旧完整继承fit复现、12资格-formal完整复现、source/state/input/math/实际native-head VJP与独立真实U/dense/bootstrap根通过。仅替LN1536初态、QR256零、Adam新建、Ω0证据/共同noise/原配置固定；'+ '；'.join(pieces)+'。有限历史panel状态移植因果结果，不等于heldout泛化/新gate/可部署state选择。完整64三臂正负neutralunknown/21空证据fit保留，2PDF3页实际pixel view；'+str(public['file_count'])+'文件/'+str(public['bytes'])+'bytes在main/research共同'+public['commit']+'逐远端bytes/SHA256/Gitblob/mode/tree实际pass。无新DINO/teacher/调参/论文/CURRENT/main自动启动；P7-A冻结、S剩余formal取消与原13747prefix/同PID optimizer/lease T standby保持，原十scope真正FINAL不变。未来方法及main具体协议仍未闭合，30min同一Luna ACTIVE。见[新完整报告](</home/wwww/visual grounding/results/stvg_opd_LN_state_intervention/2026-10-13/REPORT.md>)、[独立根](</home/wwww/visual grounding/artifacts/stvg_opd_LN_state_intervention_v1/ROOT_READBACK.json>)。\n\n'
 p=ROOT/'docs/RESEARCH_HISTORY.md';text=p.read_text()
 for anchor in ['## 1. 当前状态：先读这一节\n\n','## 7. 候选路线总登记与重开条件\n\n','### 9.4 更新日志（新记录加在最上面）\n\n']:
  assert text.count(anchor)==1;text=text.replace(anchor,anchor+note,1)
 p.write_text(text);archive(BASE/'archive_closing','ARCHIVE_VERIFICATION_RECEIPT.json')
 bindings={n:sha(BASE/n) for n in names+['ARCHIVE_VERIFICATION_RECEIPT.json','MECHANISM_DECISION.json']}
 scope='complete single authorized64 historical-parent controlled receiving-LN-state intervention/all192 new full fits/24 actual qualification/all independent root/two PDFs three actual pages/full anonymous outcomes/public/archive; future method/main protocol not completed'
 write(BASE/'ROOT_CLOSING.json',dict(status='complete',scope=scope,bindings=bindings,parents=64,formal_fits=192,actual_qualification_fits=24,PDFs=2,pages=3,CPU_only=True,GT_read=True,time=time.time()))
 write(BASE/'FINAL_COMPLETION.json',dict(status='complete',scope=scope,root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),actual_root=True,actual_PDF_pixel_view=True,
  actual_remote_publication=True,actual_archive=True,complete_current_intervention=True,future_method_and_main_protocol_complete=False,
  new_gate_implemented=False,production_promoted=False,remaining_formal_baseline_cancelled=True,CPU_only=True,GT_read=True,time=time.time()))
 status(BASE/'STATUS.json',dict(status='complete_current_intervention_future_main_protocol_pending',done=192,total=192,CPU_only=True,GT_read=True,time=time.time()))
if __name__=='__main__':run()
