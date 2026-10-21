"""Render already scored aggregate evidence; no cache, label or model access."""
from pathlib import Path
import json,os
os.environ['MPLCONFIGDIR']='/tmp/stvg-temporal-audit-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'results/temporal_signal_audit_v1';O=ROOT/'outputs/temporal_signal_audit_v1'
r=json.loads((P/'report.json').read_text());s=json.loads((P/'sensitivity.json').read_text())
def fmt(x):return f"{x['mean']:.4f} [{x['CI95'][0]:.4f}, {x['CI95'][1]:.4f}]; n={x['queries']}"if x['mean']is not None else'n=0'
def at(c,a,k):return r['summary'][c][a][k]['auc']
lines=['# 空间、注意力与时间边界：封存缓存离线审计','',
'结论：冻结模型中存在与事件和边界相关的信息；没有发现“目标区域注意力在起止点普遍更大”。HC 的目标位置注意力对比具有事件相关性，Vid 未复现。hidden 变化在三组均靠近边界，但位置/锚点/面积控制后变弱，加入现有时间头控制后未建立独立增量。因此本轮不注册新方法、不调整当前全量参数。','',
'本轮由用户明确授权探索时空关联。主分析先锁定八项问题与全部128条无标签信号，再独立读取标签；后续位置/裁剪/原时间头敏感性检查明确属于看到主结果后的探索性检查，并再次先封存曲线后读标签。','',
'## 数据与核验','',
'- HC64、与其父来源隔离的 HC32、Vid32，共128 query / 10,017采样节点。各组每父/视频一条；全部历史曝光，不能称未见测试确认。事件判别要求未观测帧内外各至少2点，合格数64/32/29；双端边界判别合格数64/32/24，其他行保留各自n。',
'- 使用原224两个offset的真实TimeDecoder输入及对应model_ema权重，只在CPU重放时间子图。256次子图调用，1,536层hidden检查，最大绝对误差3.33786e-6，小于事前1e-4阈值；不是CPU/GPU逐位等价。',
'- 0新RGB、0视觉encoder、0专家、0 GPU调用、0参数更新。无标签提取22.3407秒/128条是诊断计算成本，不是方法部署速度。',
'- 分别保存时间self-attention及读取同帧[text,motion]的cross-attention的聚合量；没有用ROI特征余弦冒充attention。八head及六层平均为主，各层只作次级探索。',
'- 原生缓存已经包含原模型的目标条件信息。本轮没有关闭它，也没有为当前BoxFixed新增时间反馈，所以只能测关联，不能把已有骨干信号归为新增空间模块的因果收益。','',
'## 信号定义','',
'令 A 为真实cross-attention，m为合法motion视觉组质量，p=A_visual/m。M为独立于GT事件的全时间BoxFixed辅助轨迹的1.5倍连续网格覆盖；P为每帧合法视觉格上的固定半圈循环位移，逐帧保留M面积与数值集合。',
'- roi_mass = C = sum(p*M)，是视觉组内目标范围占比；未去除框面积影响。',
'- roi_enrichment = C - mean_legal(M)。',
'- attention_placement = sum(p*M) - sum(p*P)，同面积真实位置对置换位置的注意力对比。',
'- temporal_key_salience 为跨帧self-attention对某帧key的接收量，均值归一到1，不同于同帧ROI占比。',
'- hidden_change = 1-cos(h[t-1],h[t+1])，只比较同offset邻接输出，避免交错offset本身造成伪变化；它是跨固定采样步的变化，不是物理速度。',
'- confidence 是原空间对应可靠性，不是事件概率；eref_support/aux_support 是原固定u；box_speed为中心与log尺度的秒导数范数。',
'- native_event_mass与native_boundary_mass来自已经封存的六层原生物理区间后验，仅作已有时间信息正控。','',
'## 事件内外（主表仅未观测帧）','',
'AUC=0.5为无排序能力，1为所有事件内帧高于外帧；不是定位准确率或vIoU。每query先算AUC，再宏平均与10,000次query bootstrap，表内为均值[95%CI]及合格n。','',
'| 信号 | HC64 | HC32 | Vid32 |','|---|---|---|---|']
for k,title in [('confidence','空间对应可靠性'),('eref_support','ERef支持u'),('aux_support','完整轨迹支持u'),('attention_placement','真实−置换位置注意力'),('roi_enrichment','面积校正目标注意力'),('native_event_mass','已有时间后验事件质量（正控）')]:lines.append('| '+title+' | '+' | '.join(fmt(at(c,'event_unobserved',k))for c in ['HC64','HC32','Vid32'])+' |')
lines+=['','真实−置换注意力在HC两组优于其逐query时间循环错位对照，八项主问题BH校正q为.0004/.0064；Vid q=.9406。HC效应不能推广为双域通用事件奖励。原空间r/u没有稳定事件判别力。',
'对时间位置及平方、最近观察接近度、log框面积作同视频无标签线性残差后，真实−置换事件AUC依次为'+ ' / '.join(fmt(at(c,'event_unobserved','attention_placement_residual'))for c in ['HC64','HC32','Vid32'])+'。残差CI均包含.5，未建立这些共变量之外的独立价值；这不证明原相关性全是伪影，线性残差也会去掉共同的真实信息。','',
'## 边界附近的变化','',
'正例为距任一起止端点<=1秒，负例为>2秒，中间带排除；边界必须在捕获时间轴内，主表排除八观察节点。两端及全部节点结果也完整公开。','',
'| 信号 | HC64 | HC32 | Vid32 |','|---|---|---|---|']
for k,title in [('link_break','轨迹连接断裂'),('box_speed','框运动/尺度速度'),('attention_placement_change','注意力位置对比变化率'),('hidden_change','时间hidden变化'),('temporal_key_salience','时间self-attention接收量（次级）'),('native_boundary_mass','已有起止后验质量（正控）')]:lines.append('| '+title+' | '+' | '.join(fmt(at(c,'boundary_either_unobserved',k))for c in ['HC64','HC32','Vid32'])+' |')
lines+=['','hidden变化相对循环错位在三组均为正，八主问题BH校正q=.0004/.0008/.0120。box_speed和注意力对比变化在HC两组为正，Vid未建立；link_break未建立稳定性。','',
'### “起止点附近的目标注意力更多”并不成立为普遍规律','',
'比较事件内距起点<=1秒与距两端均>2秒的内部帧，HC64面积校正目标注意力差为−.01664，95%CI[−.02778,−.00588]，55条合格query；起点附近反而更低。终点差−.01061、CI跨零。HC32两端和Vid大多数该项区间也跨零，Vid内部对照只有8/6条，不能用少数方向翻转作通用规律。',
'时间self-attention的key接收量可在边界相对全视频远端更高，但相对事件内部的多数起/终点比较不稳定。不能把“边界vs全视频”与“边界vs事件内部”混写。','',
'## 明确标注的后续敏感性分析','',
'看到上述主结果后，只补充读取已有query_time/query_tgt，不再调用decoder。对新曲线先封存，再用同一GT离线评价。移除项为同视频无标签OLS的解释部分：时间位置/平方、锚点接近度、log框面积、位置编码与初始query变化；再单独加入已有boundary后验质量。该分析不构建新预测器，不是严格条件互信息估计。','',
'| 边界信号 | HC64 | HC32 | Vid32 |','|---|---|---|---|']
for k,title in [('positional_change','仅位置编码变化'),('hidden_change','原hidden变化'),('hidden_without_position_nuisance','扣除位置/锚点/面积等'),('hidden_without_position_and_head','再扣除原时间头质量')]:lines.append('| '+title+' | '+' | '.join(fmt(s['summary'][c]['all_captured_boundaries'][k])for c in ['HC64','HC32','Vid32'])+' |')
lines+=['','额外只看两个GT边界距输入轴两端均>2秒的query，35/20/14条中原hidden变化AUC为.6145/.6302/.5960，说明不是纯输入裁剪边缘峰；但共同控制后的残差仍未建立独立关联。残差阴性不能证明没有任何非线性可利用信息，也不能当作空间时间联合不可行的定论。','',
'## 对算法的含义','',
'1. 原骨干已有事件/边界信息，本轮数据不支持“时间分支完全没有可用信息”。',
'2. 值得关注的是内容/状态的变化与已有时间分布的关系，而非预设“目标范围注意力越大越接近起止点”。',
'3. 空间可靠性与固定u不能直接改名为事件概率；真实轨迹对比在HC有信号，但Vid及共变量控制未建立增量。',
'4. 目前已有时间头的正控强于大多数裸空间信号。不能由本轮AUC直接推出换读出会涨tIoU/vIoU；尚无据此采用新权重或修改当前全量的依据。',
'5. 若后续开发新联合评分，首先需要证明它能在原时间后验含混/错误时提供额外区分，而不是重复已有边界峰。这是下一实验的必要判据，本轮未启动新适应、调参或全量重跑。','',
'## 复现与完整证据','',
'从仓库根目录、已具备原私有合法缓存及权重的环境，显式运行：','',
'```bash',
"CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B -m unittest scripts.temporal_signal_audit.test_audit",
"CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/temporal_signal_audit/extract.py prepare",
"CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/temporal_signal_audit/extract.py extract",
"CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/temporal_signal_audit/evaluate.py",
"CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/temporal_signal_audit/followup.py",
"CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/temporal_signal_audit/verify.py",'```','',
'原输出不允许覆盖；prepare/extract/evaluate针对这次有输入绑定的离线队列，不支持静默混入其他缓存。verify另核正式任务源哈希与进程，是此工作站本次并行不干扰证明。',
'协议在protocol.json，全部主/次级层、起/终点与支持范围在report.json，逐query正负关联在signed_queries.tsv；sensitivity.json标记事后检查，verification.json为独立rank-sum及434冻结源核验。原始特征、attention、框和GT坐标不公开。JSON中的positive_queries/negative_queries针对所列数值相对0；对AUC应看chance_delta或alignment_delta，不能把AUC>0的计数当成优于随机。','',
'![Association audit](association.svg)','',
'图：A为事件判别，B为边界判别及明确事后残差控制。点/线是query宏平均/95%CI；灰虚线.5，三组各自有效n见表，不把帧当独立样本。']
(P/'report.md').write_text('\n'.join(lines)+'\n')
plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
colors=['#2460a8','#df8b27','#36845c'];coh=['HC64','HC32','Vid32']
left=[('attention_placement','Real minus permuted ROI attention'),('aux_support','Spatial support (auxiliary tube)'),('native_event_mass','Existing temporal event mass')]
right=[('hidden_change','Temporal hidden change'),('hidden_without_position_nuisance','After positional / nuisance control'),('hidden_without_position_and_head','After temporal-head control'),('head_boundary_mass','Existing boundary posterior')]
for ax,items,kind in [(axes[0],left,'event'),(axes[1],right,'boundary')]:
 for j,c in enumerate(coh):
  y=[];means=[];lo=[];hi=[]
  for i,(key,title)in enumerate(items):
   v=at(c,'event_unobserved',key)if kind=='event'else s['summary'][c]['all_captured_boundaries'][key]
   means.append(v['mean']);lo.append(v['mean']-v['CI95'][0]);hi.append(v['CI95'][1]-v['mean']);y.append(i+(j-1)*.17)
  ax.errorbar(means,y,xerr=[lo,hi],fmt='o',capsize=2,color=colors[j],label=c,markersize=4)
 ax.set_yticks(range(len(items)),[t for _,t in items]);ax.invert_yaxis();ax.axvline(.5,color='gray',linestyle='--',linewidth=1);ax.set_xlim(.3,1.);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Macro AUC on unobserved frames');ax.set_title('A. Event association'if kind=='event'else'B. Boundary association')
axes[0].legend(loc='lower left');fig.suptitle('Sealed-cache audit: associations do not establish incremental temporal gains',fontsize=12)
fig.savefig(P/'association.svg');fig.savefig(O/'association.png',dpi=150);plt.close(fig)
print('Rendered report and figure')
