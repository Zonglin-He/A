"""True bounded-task closure after independent roots, actual visual review and remote byte readback."""
from scripts.stvg_opd_real_forward_task_common_v1 import *
from scripts.close_stvg_opd_p7_phase_v1 import archive

def close():
 verify()
 names=['ROOT_QUALIFICATION.json','ROOT_READBACK.json','ROOT_INDEPENDENT_POLICY_APPLICATION_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json','ROOT_VISUAL_REVIEW.json','PUBLIC_SCALAR_AUDIT.json','FINAL_GITHUB_RECEIPT.json']
 for n in names:assert read(BASE/n)['status']=='pass'
 view=read(BASE/'ROOT_VISUAL_REVIEW.json');pub=read(BASE/'FINAL_GITHUB_RECEIPT.json')
 assert view['actual_pixel_view'] and view['pages']==3 and pub['actual_remote_bytes_SHA256_Git_blob_tree_verified']
 for n,b in view['PDF_files'].items():assert sha(ROOT/n)==b['sha256'] and (ROOT/n).stat().st_size==b['bytes']
 assert read(BASE/'ROOT_READBACK.json')['counts']['fits']==256 and read(BASE/'ROOT_READBACK.json')['counts']['exact_original_A_fits']==64
 s=read(OUT/'SUMMARY.json');effects=[]
 for ds,v in s['datasets'].items():
  e=v['dense_v_delta']['contrasts']['SEL_minus_A']['estimate'];effects.append(f"{ds} SEL-A mvIoU {100*e['mean']:+.4f}pp [{100*e['ci95'][0]:+.4f},{100*e['ci95'][1]:+.4f}]")
 note='**2026-10-13｜Real-Forward固定64历史query完整任务验证真正有限FINAL。** 四臂256完整fits（64原A精确重放、192新SEL/HALF/LN），6400保存轮；另32完整资格/800轮/8fresh capture，16whole repeat与16qualification-formal完整一致，0新DINO。全global后原GT评分，所有source-state-pixel-frame-input/noise/action/likelihood/Adam/proposal mask/实际candidate完整forward与独立NumPy应用、真实帧保护-纠错/完整dense/parent bootstrap根及全部匿名正负neutralunknown读回实际pass；'+ '；'.join(effects)+'。这些效果约束当前固定配方与历史panel，简单HALF/LN对照、观察学习、负尾与不确定性均保留，不把root pass当科学收益或部署支持。1PDF3页actual pixel view通过，'+str(pub['file_count'])+'公开files/'+str(pub['bytes'])+'bytes在main/research共同'+pub['commit']+'、tree'+pub['tree_sha']+'逐远端bytes/SHA256/Gitblob/mode/tree实际pass。旧第14的4/9<50%保持失败、第13数值资格负结论不重开；R/Q0/P9/负证据/P7-B结论层级修正已留原证据，未并行重跑。当前有限探索完成不代表独立复制/未来method/main协议闭合，不自动扩大/新teacher/gate/loss/调参/论文/CURRENT晋升，取消基线同身份T standby，巡检按人类已关闭且未重建。见 [完整报告](</home/wwww/visual grounding/results/stvg_opd_real_forward_task_validation/2026-10-13/REPORT.md>)。\n\n'
 p=ROOT/'docs/RESEARCH_HISTORY.md';text=p.read_text()
 for anchor in ['## 1. 当前状态：先读这一节\n\n','### 9.4 更新日志（新记录加在最上面）\n\n']:
  assert text.count(anchor)==1;text=text.replace(anchor,anchor+note,1)
 p.write_text(text);archive(BASE/'archive_closing','ARCHIVE_VERIFICATION_RECEIPT.json')
 bindings={n:sha(BASE/n) for n in names+['ARCHIVE_VERIFICATION_RECEIPT.json','MECHANISM_DECISION.json']}
 write(BASE/'ROOT_CLOSING.json',dict(status='complete',scope='one authorized fixed64 historical query four-arm complete live trajectory task test; all negative/control/protection/correction/unknown findings; no universal route or production claim',bindings=bindings,complete_current_task_validation=True,CPU_only=True,GT_read=True,time=time.time()))
 write(BASE/'FINAL_COMPLETION.json',dict(status='complete',scope='current finite Real-Forward complete-task validation',root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),complete_current_task_validation=True,actual_root=True,actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,production_promoted=False,future_method_and_main_protocol_complete=False,monitor_recreated=False,time=time.time()))
 status(BASE/'STATUS.json',dict(status='complete_current_bounded_task_validation',done=256,total=256,GT_read=True,monitor_recreated=False,time=time.time()))
 print('REAL_FORWARD_TASK_TRUE_FINITE_FINAL',pub['commit'])
if __name__=='__main__':close()
