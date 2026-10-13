"""Finite current-scope close only after actual independent roots/view/public/archive."""
from scripts.stvg_opd_admission_association_common_v1 import *
from scripts.close_stvg_opd_p7_phase_v1 import archive

def close():
 verify();names=['ROOT_QUALIFICATION.json','ROOT_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_ORACLE_SELECTION_READBACK.json','ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json','ROOT_VISUAL_REVIEW.json','PUBLIC_SCALAR_AUDIT.json','FINAL_GITHUB_RECEIPT.json']
 for n in names:assert read(BASE/n)['status']=='pass'
 view=read(BASE/'ROOT_VISUAL_REVIEW.json');pub=read(BASE/'FINAL_GITHUB_RECEIPT.json');root=read(BASE/'ROOT_READBACK.json')
 assert view['actual_pixel_view'] and view['pages']>0 and pub['actual_remote_bytes_SHA256_Git_blob_tree_verified']
 for n,b in view['PDF_files'].items():assert sha(ROOT/n)==b['sha256'] and (ROOT/n).stat().st_size==b['bytes']
 assert root['counts']['logical_fits']==384 and root['counts']['exact_original_H_fits']==64 and root['Oracle_counts']['logical_fits']==64
 s=read(OUT/'SUMMARY.json');effects=[]
 for ds,v in s['datasets'].items():
  e=v['I1_dense_v_after']['contrasts']['system']['estimate'];effects.append(f"{ds} R-Path−H-Top1 I1 mvIoU {100*e['mean']:+.4f}pp [{100*e['ci95'][0]:+.4f},{100*e['ci95'][1]:+.4f}]")
 b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');o=read(BASE/'ORACLE_PREDICTION_BARRIER.json');q=read(BASE/'QUALIFICATION_BARRIER.json')
 note='**2026-10-13｜准入×跨帧关联匹配实验真正有限FINAL。** 固定原64历史P9/P0 previous_P6 parent、Inherited-order1/QR0/1792与原OPD两域配置，H-Top1/H-Path/R-Top1/R-Path/R-Add/R-Shuffle共384逻辑完整fits/9600轮，按完全相同监督明确复用，实际'+str(b['actual_complete_GPU_fits'])+'完整物理拟合/'+str(b['actual_complete_rounds'])+'轮/64frozen capture。另60逻辑资格/1320轮、'+str(q['actual_complete_GPU_fits'])+'物理fit/10capture，64正式H完整原fit精确复现与全部资格repeat/formal对照实际通过。全B sealed后才GT，原R候选池Oracle64逻辑完整/1600轮中另新'+str(o['actual_complete_GPU_fits'])+'物理fit，GT-assisted单独诊断、不进主方法统计、不添加GT框。主P7-A I1与次I0、同真实common U排除所有DINO访问含拒绝、teacher质量/有效反馈/保护纠错/负尾/独立parent bootstrap/source-state-input-noise-action-likelihood-Adam/全匿名正负未知root实际pass；'+ '；'.join(effects)+'。报告保留五组contrast、Add/Shuffle退化和Oracle解释边界，不把几何连续性当referent identity、不把root pass当科学正结果。'+str(view['pages'])+'页PDF实际像素view、'+str(pub['file_count'])+'文件/'+str(pub['bytes'])+'bytes在main/research共同'+pub['commit']+'、tree'+pub['tree_sha']+'逐远端exact验证，档案check-snapshot-check实际pass。旧MAP/功能路线负证据保留，当前历史panel不代表独立未见确认或future method/main闭合；0新DINO/teacher/参数搜索/论文/CURRENT晋升，取消baseline同身份T standby无信号，巡检保持删除。见[完整报告](</home/wwww/visual grounding/results/stvg_opd_admission_association/2026-10-13/REPORT.md>)。\n\n'
 p=ROOT/'docs/RESEARCH_HISTORY.md';text=p.read_text()
 for anchor in ['## 1. 当前状态：先读这一节\n\n','### 9.4 更新日志（新记录加在最上面）\n\n']:
  assert text.count(anchor)==1;text=text.replace(anchor,anchor+note,1)
 p.write_text(text);archive(BASE/'archive_closing','ARCHIVE_VERIFICATION_RECEIPT.json')
 write(BASE/'ROOT_CLOSING.json',dict(status='complete',scope='authorized fixed64 admission x geometric association matched complete task and separate original-pool GT-assisted Oracle',
  bindings={n:sha(BASE/n) for n in names+['ARCHIVE_VERIFICATION_RECEIPT.json','MECHANISM_DECISION.json']},complete_current_admission_association=True,CPU_only=True,GT_read=True,time=time.time()))
 write(BASE/'FINAL_COMPLETION.json',dict(status='complete',scope='current finite admission-association matched experiment',root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),
  complete_current_admission_association=True,actual_root=True,actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,
  production_promoted=False,future_method_and_main_protocol_complete=False,monitor_recreated=False,time=time.time()))
 status(BASE/'STATUS.json',dict(status='complete_current_bounded_admission_association',done=448,total=448,GT_read=True,monitor_recreated=False,time=time.time()))
 print('ADMISSION_ASSOCIATION_TRUE_FINITE_FINAL',pub['commit'])
if __name__=='__main__':close()
