"""Sanitized actual results, explicit GT-control and engineering provenance."""
from pathlib import Path
import sys,json,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_e_retaining1792_route32_v1 import D,P,ARMS,STAGES,verify,read,put,load,sha

def main():
 p=verify(True);x=read(P/'RESULTS_PUBLIC.json');audit=read(P/'ROOT_INDEPENDENT_AUDIT.json');assert audit['status']=='pass';cost=read(P/'COST_BREAKDOWN.json');support=read(P/'ORACLE_SUPPORT_PUBLIC.json');codepins=dict(read(D/'RUNTIME_LOCK.json')['pins']);rev=[]
 for f in sorted((D/'amendments').glob('*.json')):
  a=read(f);codepins.update(a['replacement_pins']);rev.append(dict(reason=a['reason'],replacement_pins=a['replacement_pins'],failed_allocation_seconds=a['failed_allocation_seconds'],GT_access=a['GT_access']))
 stages={};rows=[]
 for st in STAGES:
  b=read(D/(st.upper()+'_PREDICTION_BARRIER.json'));stages[st]={k:v for k,v in b.items()if k!='files'};stages[st]['barrier_sha256']=sha(D/(st.upper()+'_PREDICTION_BARRIER.json'))
  for f in b['files']:
   z=load(ROOT/f['path']);rows.append(dict(alias=z['alias'],stage=st,prediction_sha256=f['sha256'],model_sha256=z['model_hash'],zero_exact_E=z['zero_exact_E'],GT_used=z['GT_used'],GT_free_support=z['GT_free_support'],initial_state_counts={a:sum(v.numel()for v in q['initial'].values())for a,q in z['fits'].items()},full_actual_frames=len(z['frame_ids'])))
 put(P/'SOURCE_EVIDENCE_PUBLIC.json',dict(count=32,stages=stages,code_registry_and_config_pins={k:v for k,v in codepins.items()if k.startswith(('scripts/','vg_tta/','methods/','external/','protocols/'))},rows=rows,real_seal_before_oracle_GT_inputs=True,raw_coordinates_states_references_labels_questions_media_weights_and_private_parents_excluded=True));put(P/'ENGINEERING_REVISIONS_PUBLIC.json',dict(revisions=rev,updates_and_outputs_not_changed_by_numeric_audit_fix=True,old_six_sealed_queries_reused=True,all_failed_GPU_charged=True,maximum_mean_gradient_errors=audit['maximum_errors']))
 def f(z):return f"{z['mean']:+.4f} [{z['CI95'][0]:+.4f}, {z['CI95'][1]:+.4f}]"
 lines=['# E 保留的原1792参数入口：Direct / Fixed / Live，一轮路线决策', '', '状态：completed / measurement valid。NH16/NV16已曝光开发，非独立确认；真实DINO与GT诊断分别封存和报告。', '', '## 本轮实际做了什么', '', '- 不继续what提示词、观测位置/数量或局部Fine-Center参数。没有新增DINO、专家、Source前向、定位头或新模型训练。复用同一TA-STVG、原E/finalMBR和原四计划节点。', '- 适应变量为原query residual256与最后空间层norm1/norm3/norm4 affine1536，共1792；每query、每分支重置和全新Adam，最终步读出，不继承/不写回生产。原checkpoint/模型本体冻结，临时副本的这1536个LN参数明确参与适应，不能称为全部decoder参数都冻结。', '- HC40步lr.01 sigma.025 tau.05；Vid10步lr.03 sigma.1 tau.25，M32反向成对，Adam(.9,.999) eps1e-8。无最佳GT步/损失选步、无lr/温度/尺度扫描。平奖励原样跳Adam，实际更新数全部报告。', '- 原E部分cxcywh超出1，不能裁剪以套入sigmoid chart。本轮保留native-logit探索，用固定E中心差与尺寸比映射全部候选/最终框后再计算DINO IoU；Direct也在同映射输出上监督。零态严格等于原E。它是接到E上的明确策略扩展，不伪称与旧Source起点实验完全相同。', '- 原second spatial decoder输入缓存位于全部被适应的计算之前；重跑六层和所有actual帧。QR改变六层，最后三LN真实反传；原时间decoder、门控、视频/文本memory不更新。两offset完整空间输出，不把仅观察框/插值当全轨迹解码。', '- Direct保持原(5L1+2(1-GIoU))sum/4；Fixed冻结首次采样/反馈但持续更新当前学生likelihood；Live每轮从更新后的native均值重新采样，再经E映射取得当前候选IoU反馈。两者首轮候选/反馈/更新相同，原control-variate与实际Adam链均保存。', '', '## 真实证据与GT边界', '', '- 真实分支只用原E-selected raw DINO参考，三分支相同，无新what输出、GT门控或参考选择。32×3真实预测先全封存，之后才准备GT正控。', '- GT正控仅把同一参考位置中精确GT已知的框替换；未知位置只在GT正控省略。三个GT分支同支持，不补造标签、不插值GT。该正控不是可部署结果，而且GT−真实差异包含未知位置省略，不能一概归为纯坐标质量因果。', '- 64组query-stage/192适应输出全部封存后，才做本轮完整评价。全部父视频、无参考、零时间交集与无更新样本均保留。正式FH/FV64未参与本轮选参或新评价。', '', '| 域 | 真实参考数 / 有支持query | GT已知参考数 / 有支持query |', '|---|---:|---:|']
 for ds in ['hc2','vidstg']:
  a,b=x['groups']['real'][ds],x['groups']['oracle'][ds];lines.append(f"| {ds} | {a['references']} / {a['support_queries']} | {b['references']} / {b['support_queries']} |")
 for st,label in [('real','真实DINO，主实验'),('oracle','正确GT参考，单独诊断')]:
  lines += ['', '## '+label, '', '下表为相对同一原E的增量pp；括号为父视频配对bootstrap95%区间。每域16父，不把帧当独立样本。', '', '| 分支−E | HC完整vIoU | Vid完整vIoU | HC未观测actual sIoU | Vid未观测actual sIoU |', '|---|---:|---:|---:|---:|']
  for arm in ARMS:
   h=x['groups'][st]['hc2']['paired'][arm+'_minus_E_MBR'];v=x['groups'][st]['vidstg']['paired'][arm+'_minus_E_MBR'];lines.append(f"| {arm} | {f(h['v'])} | {f(v['v'])} | {f(h['unobserved_actual_s'])} | {f(v['unobserved_actual_s'])} |")
  lines += ['', '| 刷新增量 | HC完整vIoU | Vid完整vIoU | HC未观测actual sIoU | Vid未观测actual sIoU |', '|---|---:|---:|---:|---:|']
  for base in ['Direct','Fixed']:
   h=x['groups'][st]['hc2']['paired']['Live_minus_'+base];v=x['groups'][st]['vidstg']['paired']['Live_minus_'+base];lines.append(f"| Live−{base} | {f(h['v'])} | {f(v['v'])} | {f(h['unobserved_actual_s'])} | {f(v['unobserved_actual_s'])} |")
 lines += ['', 'tIoU四组全不变，未观测actual指标不包括插值；完整v及未观测actual共同判断。bootstrap10000、seed20261018，unknown框不填零，实际有效父数列于JSON。全部signed逐query、正负/严重尾部、留一敏感性见RESULTS_PUBLIC.json。', '', '## 实际更新与锚点拟合', '', '| 证据/域/分支 | 真实梯度步数 | 初始→最终平均锚点loss（有支持query） | 平奖励零IoU轮数 |', '|---|---:|---:|---:|']
 for st in STAGES:
  for ds in ['hc2','vidstg']:
   rr=[r for r in x['rows']if r['stage']==st and r['dataset']==ds]
   for arm in ARMS:
    dd=[r['fit_diagnostics'][arm]for r in rr if r['references']];initial=sum(q['anchor_loss_initial']for q in dd)/len(dd)if dd else 0;final=sum(q['anchor_loss_final']for q in dd)/len(dd)if dd else 0;steps=sum(r['fit_diagnostics'][arm]['gradient_calls']for r in rr);flat=sum(r['fit_diagnostics'][arm]['all_zero_reward_rounds']for r in rr);lines.append(f'| {st}/{ds}/{arm} | {steps} | {initial:.5f}→{final:.5f} | {flat} |')
 lines += ['', '锚点loss下降只证优化发生，不能替代未观察帧和完整tube收益。原平IoU奖励仍可能既阻止错误更新，也漏学正确参考；零步不包装成语义拒绝。', '', '## 成本与工程记录', '', f"GPU持有合计 {cost['GPU_allocation_seconds']:.3f}s（包含失败重试），封存完成产物中的轨迹decoder {cost['decoder_trajectory_calls']} 次／PosDecoder {cost['PosDecoder_calls']} 次／状态反向 {cost['backward_calls']} 次；Source捕获、DINO调用、新模型训练均0。每query每分支秒数在逐条结果中，缓存构造历史成本不计入本轮，不能把1792参数少写成计算免费。失败未完成query的额外decoder/反向计数未落盘，不冒充完整调用总数；GPU持有已包含该失败全部工作。", '', '第一次真实worker在6条已封存后，某Fixed mean-gradient继承绝对2e-5断言停止；失败持有53.275785s计入总成本，没有GT读入。修订仅增加float32幅度误差容差，超过旧阈值必须通过同一损失双精度独立梯度复核。实际loss/梯度/Adam/采样/配置不改，6条已封存精确复用；原代码/日志/修订保留。NH006已回读绝对误差2.39268e-5、梯度幅度189.425，双精度误差2.30973e-5，约float32舍入尺度，未靠放松检查改变方法。', '', '根独立CPU核对E映射、1792状态/零态、真实与GT屏障、Fixed重复反馈/Live当前均值、原IoU奖励和softmax、梯度、实际Adam链及全部均值/区间；旧E指标精确复现。原生产/正式结果与所有输入hash保持。审计不新增模型/GPU或实验。', '', '## 决策与范围', '', read(D/'ROOT_DECISION.json')['text'], '', '本轮按预定决策收束，不自动追加参数扫描、观测、提示词、学生入口或新实验。旧matched32的学习能力证据保留；本轮不能把固定预算失败写成学生普遍没有能力，也不能把GT正控当可部署成绩。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'ROUND_REVIEW.md').write_text('# Root route decision\n\n'+read(D/'ROOT_DECISION.json')['text']+'\n\nReal and GT controls separate; differing unknown support disclosed. All outcomes including severe negatives retained. No automatic follow-up, checkpoint or production promotion.\n')
 files=list(['vg_tta/stvg_e_retaining1792_v1.py','protocols/stvg_e_retaining1792_route32_v1.md'])+[f'scripts/{s}_stvg_e_retaining1792_route32_v1.py'for s in ['run','audit','report']]+['scripts/verify_stvg_e_retaining1792_geometry32_v1.py']+[str(f.relative_to(ROOT))for f in sorted(P.iterdir())if f.is_file()and f.name!='ROUND_REVIEW.md'];put(D/'PUBLIC_FILES_FINAL.json',dict(files=files));print('report/public files ready',len(files),flush=True)
if __name__=='__main__':main()
