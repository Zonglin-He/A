"""Render all signed development outcomes after independent audit."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_coordinate_spatial_opd64_v1 import D,P,ARMS,ENDPOINTS,read,put,sha,verify

def main():
 p=verify(True);r=read(P/'RESULTS_PUBLIC.json');me=read(P/'MECHANISM_PUBLIC.json');audit=read(P/'ROOT_INDEPENDENT_AUDIT.json');assert audit['status']=='pass';cost=read(P/'COST_BREAKDOWN.json');summary={}
 lines=['# 坐标候选 KL 与空间软标签 KL：同条件 OPD64 对照','',
 '运行 completed；测量 valid；范围为 NH/NV、CH/CV 已曝光开发来源，各域32父视频，一源一query。不是新来源确认，不使用 FH/FV 选配置。',
 '', 'A 使用原 Fine-Center 坐标候选 KL；B 将同轮候选概率通过面积核推送到空间后做 KL。两条路线都运行两轮 Fixed/Live，最终按学生中心状态读出；没有硬选框门槛、额外视觉证据、网络训练、新专家或 DINO 调用。',
 '', '冻结配置：HC step1、Vid step0.5；温度0.1、scale0.25、2×17反向成对候选；宽高、E、MBR、四个计划节点和原插值范围全部固定。B 未校准步长，不声称相同步长等于相同优化强度。',
 '', '## 完整执行状态','', '| 输出 | fit数 | 两轮slot数 | 来源 |','|---|---:|---:|---|','| E |64|0|原缓存精确保留|','| A-Fixed |64|128|48缓存；16原NH/HC1 Fixed按原实现补算|','| A-Live |64|128|64精确缓存复用|','| B-Fixed |64|128|64真实新fit|','| B-Live |64|128|64真实新fit|','',
 'slot包含严格无反馈noop，不等同于非零梯度节点。B每轮的q、K和候选停止梯度，只有当前学生概率p保留梯度；Fixed第二轮仍使用当前学生p。每臂每样本从零重置；Live第二轮确实读取其第一轮更新后的状态。相同输入的教师缓存复用不被重复计作新专家计算。',
 '', '## 合并开发来源：完整主矩阵','', '下表为相对 E 的百分点变化。未观察实际节点指标保留全部原已知GT节点分母；不靠缩短时间区间删难帧。',
 '', '| 方法−E |HC完整vIoU|Vid完整vIoU|HC未观察实际节点sIoU|Vid未观察实际节点sIoU|','|---|---:|---:|---:|---:|']
 g=r['groups']['combined']
 for a in ARMS[1:]:
  v=[g[ds]['paired'][a+'_minus_E_MBR'][ep]['mean']for ds,ep in [('hc2','v'),('vidstg','v'),('hc2','unobserved_actual_s'),('vidstg','unobserved_actual_s')]];lines.append('|'+a+'|'+'|'.join(f'{z:+.6f}'for z in v)+'|')
 lines+=['','## 关键配对比较与区间','','父视频配对bootstrap10,000次，seed20261018；开发探索性区间，未作多重比较选择校正。','', '| 对比 |域|完整vIoU pp [95%CI]|未观察实际sIoU pp [95%CI]|','|---|---|---|---|']
 def cell(z):return f"{z['mean']:+.6f} [{z['CI95'][0]:+.6f}, {z['CI95'][1]:+.6f}]"
 for c in ['B_Live_minus_A_Live','B_Live_minus_B_Fixed','A_Live_minus_A_Fixed','B_Live_minus_E_MBR']:
  for ds in ['hc2','vidstg']:lines.append(f"|{c}|{ds}|{cell(g[ds]['paired'][c]['v'])}|{cell(g[ds]['paired'][c]['unobserved_actual_s'])}|")
 lines+=['','## 原开发与后续选参面板分别保留','','| 面板 |域|A-Live−E vIoU|B-Live−E vIoU|B-Live−A-Live vIoU|','|---|---|---|---|---|']
 for pan in ['original','selection']:
  for ds,gg in r['groups'][pan].items():lines.append('|'+pan+'|'+ds+'|'+'|'.join(cell(gg['paired'][c]['v'])for c in ['A_Live_minus_E_MBR','B_Live_minus_E_MBR','B_Live_minus_A_Live'])+'|')
 lines+=['','## 实际更新与表示诊断','', '| 域/方法 |变化query数|两轮有效节点slot|梯度范数之和|实际归一化状态步长范数之和|','|---|---:|---:|---:|---:|']
 for ds in ['hc2','vidstg']:
  rr=[z for z in me['rows']if z['dataset']==ds];dd={}
  for a in ARMS[1:]:
   ts=[t for z in rr for t in z['arms'][a]['rounds_summary']];v=dict(changed_queries=sum(z['arms'][a]['changed_actual_nodes']>0 for z in rr),eligible_node_round_slots=sum(t['eligible_nodes']for t in ts),gradient_norm_sum=sum(t['gradient_norm_sum']for t in ts),eta_step_norm_sum=sum(t['eta_step_norm_sum']for t in ts),degenerate_active_nodes=sum(t.get('identical_all_candidate_kernels_active_nodes',0)for t in ts),outside_valid_candidates=sum(t.get('outside_valid_candidates')or 0 for t in ts),outside_active_candidates=sum(t.get('outside_active_candidates')or 0 for t in ts));dd[a]=v;lines.append(f"|{ds}/{a}|{v['changed_queries']}|{v['eligible_node_round_slots']}|{v['gradient_norm_sum']:.6f}|{v['eta_step_norm_sum']:.6f}|")
  summary[ds]=dd
 lines+=['', '空间网格为预锁原图64×64，完全不包含feature padding；面积交集使小于cell的框仍有归一化质量。用户在预测前明确授权另加一个图像外状态，仅零原图交集候选使用，不改变A、q或有效性规则。B Fixed和Live各触发1次原ROI有效的图像外候选，均为NV012首轮范围外且无信息节点；实际更新节点触发数为0。不是新的图像证据或实例分割标签。',
 '', 'B的有效更新节点没有候选全部映成同一核的退化。两臂均真实改变HC31/32、Vid20/32 query的实际框；无参考样本保留不动。B当前梯度范数之和约为A的HC41.6%、Vid36.9%，并非相同步长下相同更新强度。该事实不能唯一归因为收益差异；本轮没有干预校准尺度或扫描网格。',
 '', '同轮同候选的空间KL满足几何推送的数据处理收缩；损失更小不是更好的定位证据。两种目标和梯度在重叠例子中不同，B没有对候选坐标/教师权重反传，最终也没有改用教师框或热图argmax。',
 '', '## 正负个例与尾部','', '| 对比 |域|指标|正/负/不变|最差单query pp|最佳单query pp|低于−5pp个数|','|---|---|---|---|---:|---:|---:|']
 for c in ['A_Live_minus_E_MBR','B_Live_minus_E_MBR','B_Live_minus_A_Live']:
  for ds in g:
   for ep in ['v','unobserved_actual_s']:
    z=g[ds]['paired'][c][ep];v=z['per_parent_pp'];lines.append(f"|{c}|{ds}|{ep}|{z['positive']}/{z['negative']}/{z['neutral']}|{min(v):+.6f}|{max(v):+.6f}|{z['severe_loss_below_minus5pp']}|")
 lines+=['','全部逐query的带符号/不变/未知结果见 RESULTS_PUBLIC.json，逐臂两轮实际节点更新数见 MECHANISM_PUBLIC.json。无DINO支持、零时间交集和无更新样本均未从主评价移除，缺GT保持未知。',
 '', '## 独立核验与成本','', f"根独立审计通过：{audit['checks']}。最大误差：{audit['max_errors']}。原A-Live/E逐query结果与旧缓存匹配；所有方法同时间，生产注册和正式64文件hash未变。",'',
 f"纯缓存测量耗时{cost['CPU_seconds']:.3f}秒，其中B fit合计{cost['B_fit_seconds']:.3f}秒；新A fit {cost['A_new_fit_seconds']:.3f}秒。B 256轮反馈请求中缓存命中{cost['teacher_cache_hits']}，新ROI反馈调用{cost['teacher_new_calls']}次、候选评估{cost['teacher_new_candidate_evaluations']}次，用时{cost['teacher_seconds']:.3f}秒。另新增A16fit包含32次ROI调用；独立审计另重放64次原ROI反馈，详见独立审计收据。",'',
 f"评分{read(D/'SCORE_COMPLETION.json')['CPU_seconds']:.3f}秒，独立审计{audit['CPU_seconds']:.3f}秒；两者与测量分开。GPU/Source前向/DINO编码/新模型训练全部为0；不包含既有feature和DINO缓存生成成本，不能称整个教师免费。合计512臂轮slot中224来自A缓存、288本轮新执行；B为256新轮slot。",'',
 '一次更新前工程失败已保留：旧tensor-list参考框需要显式stack，第一次入口尚无预测/GT；独立运行锁修订只作格式转换，未改变数值或科学配置。独立审计首次失败及后续定位见 ENGINEERING_RECOVERY_PUBLIC.json；禁止以无说明放宽审计来掩盖不一致。',
 '', '## 当前结论与范围','',
 '在本冻结配置下，B没有超过A：HC完整vIoU和未观察实际节点sIoU的B−A配对区间均为负；Vid B−A区间跨零。B在Vid相对E有很小的开发正增量，但不能把它写成超过A或独立泛化。B-Live−B-Fixed在两域主指标区间均跨零，未建立刷新增益。',
 '', '保留A Fine-Center为研究参考；本轮不替换为B、不自动扩大数据或调参，不晋升CURRENT/MAIN/PAPER。结论限于原图64×64面积核、用户批准图像外状态、两轮和相同步长起点；没有证明所有空间标签损失或经公平开发校准后的B都无效。下一项尺度/表示对照须另有明确协议，不从本轮结果自动展开。',
 '', '实现与复现：run_stvg_coordinate_spatial_opd64_v1.py prepare/measure/score；audit_stvg_coordinate_spatial_opd64_v1.py；report_stvg_coordinate_spatial_opd64_v1.py。私有完整候选/概率/因子化K/Q/P/梯度/状态/最终框均已封存；公开导出不含原始坐标、图像、标注、权重或父视频身份。']
 put(P/'EXECUTION_SUMMARY_PUBLIC.json',dict(status='complete',count=64,arms=ARMS,configuration=p['configuration'],summary=summary,all_outputs_before_GT=True,formal64_not_used_for_selection=True,development_not_confirmation=True))
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print('report written')
if __name__=='__main__':main()
