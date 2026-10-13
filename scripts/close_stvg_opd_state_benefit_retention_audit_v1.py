"""Actual finite retention audit closure only after root/view/remote/archive."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_state_benefit_retention_common_v1 import *
from scripts.close_stvg_opd_p7_phase_v1 import archive
def run():
 verify();names=['ROOT_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json','ROOT_VISUAL_REVIEW.json','PUBLIC_SCALAR_AUDIT.json','FINAL_GITHUB_RECEIPT.json']
 assert all(read(BASE/name)['status']=='pass' for name in names)
 view=read(BASE/'ROOT_VISUAL_REVIEW.json');public=read(BASE/'FINAL_GITHUB_RECEIPT.json')
 assert view['actual_pixel_view'] and public['actual_remote_bytes_SHA256_Git_blob_tree_verified']
 for rel,b in view['PDF_files'].items():assert sha(ROOT/rel)==b['sha256'] and (ROOT/rel).stat().st_size==b['bytes']
 root=read(BASE/'ROOT_READBACK.json')
 note='**2026-10-13｜State-Benefit Retention Audit实际完整有限完成。** 64历史P9/P0 previous-P6 parent/192既有sealed fits/4800 saved rounds/4992 arm boundaries/'+str(root['frame_round_rows'])+'匿名frame-boundaries，0新GPU/model/forward/backward/optimizer/DINO。完整独立source/state/input/noise/math/128原fit/逐实际帧与全dense/parent10000 bootstrap、2PDF4页实际pixel view及全部匿名正负neutralunknown/21空证据真实齐全。HC两继承臂输出RMS收缩区间均低于0，初始正确且优势>=1pp U层平均定位下降9.438/9.338pp，末1/4持续受损175/288与204/315帧，joint agreement↑/优势↓/持续损害各20parent；Vid收缩CI跨0，同层均值增益CI跨0，持续受损78/530与96/455帧，各7joint parent，去最大3个变化parent后优势侵蚀CI跨0。HC收益侵蚀证据较强、Vid机制混合；参数距离平均变大，不称参数趋同或catastrophic forgetting，headroom回归只关联/存在机械耦合，未测旧query保留。'+str(public['file_count'])+'公开文件/'+str(public['bytes'])+'bytes在main/research共同'+public['commit']+'逐远端bytes/SHA256/Gitblob/mode/tree实际pass。旧11scope FINAL保持/S原13747prefix同PID/start/optimizer/ordinarylease T standby取消不恢复/P7-A冻结；无新gate/loss/teacher/调参/论文/CURRENT/main自动启动。当前第12有限scope FINAL，未来方法及main协议仍未闭合，同一30min Luna ACTIVE。见[完整审计报告](</home/wwww/visual grounding/results/stvg_opd_state_benefit_retention_audit/2026-10-13/REPORT.md>)与[完整独立根](</home/wwww/visual grounding/artifacts/stvg_opd_state_benefit_retention_audit_v1/ROOT_READBACK.json>)。\n\n'
 ledger=ROOT/'docs/RESEARCH_HISTORY.md';text=ledger.read_text()
 for anchor in ['## 1. 当前状态：先读这一节\n\n','## 7. 候选路线总登记与重开条件\n\n','### 9.4 更新日志（新记录加在最上面）\n\n']:
  assert text.count(anchor)==1;text=text.replace(anchor,anchor+note,1)
 ledger.write_text(text);archive(BASE/'archive_closing','ARCHIVE_VERIFICATION_RECEIPT.json')
 bindings={name:sha(BASE/name) for name in names+['ARCHIVE_VERIFICATION_RECEIPT.json','MECHANISM_DECISION.json']}
 scope='complete one authorized cache-only192 sealed-fit64 historical-parent retention audit/all4992 boundaries/all genuine frames/math/state/source/noise/dense/independent parent statistics/two PDFs four actual pages/all anonymous outcomes/public/archive; future method/main protocol pending'
 write(BASE/'ROOT_CLOSING.json',dict(status='complete',scope=scope,bindings=bindings,parents=64,existing_fits=192,arm_boundaries=4992,
  frame_round_rows=root['frame_round_rows'],PDFs=2,pages=4,new_model_calls=0,new_optimizer_steps=0,CPU_only=True,GT_read=True,time=time.time()))
 write(BASE/'FINAL_COMPLETION.json',dict(status='complete',scope=scope,root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),actual_root=True,
  actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,complete_current_retention_audit=True,
  future_method_and_main_protocol_complete=False,catastrophic_forgetting_established=False,new_gate_or_loss_implemented=False,
  production_promoted=False,remaining_formal_baseline_cancelled=True,CPU_only=True,GT_read=True,time=time.time()))
 status(BASE/'STATUS.json',dict(status='complete_current_retention_audit_future_main_protocol_pending',done=192,total=192,CPU_only=True,GT_read=True,time=time.time()))
if __name__=='__main__':run()
