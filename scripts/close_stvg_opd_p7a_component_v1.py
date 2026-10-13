"""Finite development close only after actual root/view/remote/archive checks."""
from scripts.stvg_opd_p7a_component_common_v1 import *
from scripts.close_stvg_opd_p7_phase_v1 import archive

def close():
 verify();standby()
 names=['ROOT_QUALIFICATION.json','ROOT_DEPLOYMENT_READBACK.json','ROOT_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json',
  'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json','ROOT_VISUAL_REVIEW.json','PUBLIC_SCALAR_AUDIT.json','FINAL_GITHUB_RECEIPT.json']
 for n in names:assert read(BASE/n)['status']=='pass'
 view=read(BASE/'ROOT_VISUAL_REVIEW.json');pub=read(BASE/'FINAL_GITHUB_RECEIPT.json')
 assert view['actual_pixel_view'] and view['pages']>0 and pub['actual_remote_bytes_SHA256_Git_blob_tree_verified']
 for n,b in view['PDF_files'].items():assert sha(ROOT/n)==b['sha256'] and (ROOT/n).stat().st_size==b['bytes']
 q=read(BASE/'QUALIFICATION_BARRIER.json');b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');r=read(BASE/'ROOT_DEPLOYMENT_READBACK.json')
 assert q['actual_complete_fits']==64 and b['actual_complete_fits']==512 and b['actual_complete_rounds']==12800
 assert r['counts']['old_complete_controls']==128 and r['counts']['qualified_formal_pairs']==32
 s=read(OUT/'SUMMARY.json');effects=[]
 for ds,v in s['datasets'].items():
  c=v['paired_both_orders']['fields']['I1_After_v']['contrasts']
  for name in ['remove_QR_carry','remove_carry_joint','remove_both']:
   e=c[name];effects.append(f"{ds}/{name} {e['mean_pp']:+.4f}pp [{e['ci95_pp'][0]:+.4f},{e['ci95_pp'][1]:+.4f}]")
 note='**2026-10-13｜OPD内部QR×LN写回开发四臂实际有限FINAL。** 原两域各32开发parent、两order，joint_carry/LN_carry/joint_reset/LN_reset各走自己的完整轨迹；64实际完整资格/16capture/1600轮及512实际完整正式/128capture/12800轮，128原joint_carry完整复现、32whole-repeat与32qualified-formal完整一致，scope/reset/source/math/noise/scoped Adam和全global后I1主/I0次完整独立dense/parent bootstrap根实际pass。'+ '；'.join(effects)+'。全部正负neutralunknown/各order/实际instrumented成本、'+str(view['pages'])+'页PDF实际像素view与匿名公开'+str(pub['file_count'])+'files/'+str(pub['bytes'])+'bytes在main/research共同'+pub['commit']+'、tree'+pub['tree_sha']+'逐远端exact通过；档案check-snapshot-check实际通过。原OPD固定，无另loss、新DINO/teacher/调参/论文main/生产晋升；既有开发暴露，CI跨零不作等效或删除证明。此有限组件实验不替最终方法冻结、救参或论文主表。CURRENT/取消baseline同身份T standby与巡检关闭保持。见[完整报告](</home/wwww/visual grounding/results/stvg_opd_p7a_component_simplification/2026-10-13/REPORT.md>)。\n\n'
 p=ROOT/'docs/RESEARCH_HISTORY.md';text=p.read_text()
 for anchor in ['## 1. 当前状态：先读这一节\n\n','### 9.4 更新日志（新记录加在最上面）\n\n']:
  assert text.count(anchor)==1;text=text.replace(anchor,anchor+note,1)
 p.write_text(text);archive(BASE/'archive_closing','ARCHIVE_VERIFICATION_RECEIPT.json')
 write(BASE/'ROOT_CLOSING.json',dict(status='complete',scope='finite original64 development parents/four internal OPD arms/two complete own orders',
  bindings={n:sha(BASE/n) for n in names+['ARCHIVE_VERIFICATION_RECEIPT.json','MECHANISM_DECISION.json']},
  complete_current_component_experiment=True,CPU_only=True,GT_read=True,time=time.time()))
 write(BASE/'FINAL_COMPLETION.json',dict(status='complete',scope='finite current development internal OPD component experiment',
  root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),complete_current_component_experiment=True,actual_root=True,
  actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,production_promoted=False,
  final_method_frozen=False,paper_main_experiments_complete=False,future_method_and_main_protocol_complete=False,monitor_recreated=False,time=time.time()))
 status(BASE/'STATUS.json',dict(status='complete_current_finite_component_experiment',done=512,total=512,GT_read=True,monitor_recreated=False,time=time.time()))
 print('OPD_COMPONENT_TRUE_FINITE_FINAL',pub['commit'])

if __name__=='__main__':close()
