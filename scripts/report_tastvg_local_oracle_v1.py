"""Render source-backed tables and exportable scientific figures."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'results/tastvg_temporal_local_oracle/2026-10-03'

def read(p):return json.loads(Path(p).read_text())
def value(z,percent=False):
    if z['mean'] is None:return 'undefined'
    scale=100 if percent else 1
    a,b=z['ci95'];return f"{z['mean']*scale:.2f} [{a*scale:.2f}, {b*scale:.2f}]"

def main():
    panels={f'{ds}/{sp}':read(F/sp/ds/'SUMMARY.json') for sp in ['search','confirm'] for ds in ['vidstg','hc2']}
    decision=read(F/'DECISION.json');audit=read(F/'VALIDATION.json');resources=read(F/'RESOURCES.json')
    lines=['# A8 → O32：局部 oracle 与边界位移诊断','',decision['root_conclusion'],'',
        '这是一轮 CPU post-hoc 诊断，输入为 4ecf366 已封存、独立核验的匿名逐候选指标。'
        '没有新增模型、专家、候选、decoder replay 或 GT 标注读取；A 的空间框、Uniform '
        'Rank-RKL 持续状态与原 temporal readout 均未变。已有指标是 GT-derived，不能把本分析称为无标签实验。','',
        '两集各 32 开发＋16 确认来源，一 query/source、两序、clean＋五类 5% corruption、'
        '25% 专家，共 1,152 到达；288 专家有完整候选（240 corrupt、48 clean）。864 非专家'
        '没有完整候选，仅核对覆盖，不制造其 oracle。来源均有历史曝光，确认面板不是 fresh test。','',
        '## 新增机会来自哪里','',
        '以下是 corrupt expert；vIoU 与增量为 pp，区间为 10,000 次同来源配对 bootstrap。', '',
        '| 面板 | 专家来源 / 到达 | 正增来源 / 到达 | A8 | O8 | O32 | O32−O8 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p,s in panels.items():
        z=s['corruption'];m=z['metrics']
        lines.append(f"| {p} | {z['sources']} / {z['cells']} | {z['positive_gain_sources']} / {z['positive_gain_cells']} | "
            f"{100*m['A8_v']['mean']:.2f} | {100*m['O8_v']['mean']:.2f} | {100*m['O32_v']['mean']:.2f} | {value(m['capacity_gain'],True)} |")
    lines += ['', '## 固定局部半径能收回多少新增容量','',
        '半径 r 使用端点 L1 位移除以 **A8 区间长度**。r=.5 表示起点和终点位移的绝对值之和'
        '不超过 A8 时长的一半，不是每个边界都允许移动一半。所有半径在本次计算前固定；'
        '这是 oracle 可达性曲线，不是学习得到的接受阈值。','',
        '`accessible_extra(r)=max(0,L32(r)−O8)`，上界是 `O32−O8`。下表的占比使用来源宏平均'
        '分子除以来源宏平均总增量；不会平均逐样本不稳定的比值。分子与分母共享 bootstrap 来源抽样。','',
        '| 面板 | r=.25 可收回比例 | r=.5 可收回比例 | r=1 可收回比例 | r=2 可收回比例 |',
        '|---|---:|---:|---:|---:|']
    for p,s in panels.items():
        q=s['corruption']['capacity_gain_shares']
        lines.append('| '+p+' | '+' | '.join(value(q['anchor_'+r+'_accessible_extra'],True) for r in ['0p25','0p5','1p0','2p0'])+' |')
    lines += ['', '为防“完整 O32 最优离得远，但稍近的候选也足够好”被漏掉，同时提供两条曲线：'
        '上述局部可收回收益，以及以完整 O32 最优位置为定义的收益加权 CDF。后者对并列最优分别用最近和最远距离。', '',
        '| 面板 | r=.5 完整最优：最近 / 最远占比 | r=1 完整最优：最近 / 最远占比 | r=.5 L32−L8 | r=.5 L32−A8 |',
        '|---|---:|---:|---:|---:|']
    for p,s in panels.items():
        z=s['corruption'];q=z['capacity_gain_shares'];m=z['metrics']
        def pair(r):return f"{100*q['anchor_'+r+'_near_gain_mass']['mean']:.2f}% / {100*q['anchor_'+r+'_far_gain_mass']['mean']:.2f}%"
        lines.append(f"| {p} | {pair('0p5')} | {pair('1p0')} | {value(m['anchor_0p5_local_extra'],True)} | {value(m['anchor_0p5_L32_gain'],True)} |")
    lines += ['', 'L32−L8 限制了两个池；它和“超过完整 O8 的新增机会”不同，不能互换。'
        '完整窗口归一化曲线、所有预锁半径、候选计数、逐来源与两序结果均在 SUMMARY/ROWS。', '',
        '## 起点、终点、中心和长度','',
        '起止距离是 A8 到 **候选 vIoU oracle** 的位移，不是到真实 GT 端点的误差。'
        '秒数按缓存的观测窗口时长还原；该窗口不保证覆盖完整原视频。以下位移均值包含零增量到达，'
        '只作尺度参考；判断新增收益的位置以收益加权曲线和分解为主。','',
        '| 面板 | 最近最优：起点 / 终点秒数 | 端点 L1 秒数 | L1 / A8时长 | A8 与最优 tIoU |',
        '|---|---:|---:|---:|---:|---:|']
    for p,s in panels.items():
        m=s['corruption']['metrics']
        lines.append(f"| {p} | {m['start_seconds']['mean']:.2f} / {m['end_seconds']['mean']:.2f} | "
            f"{m['nearest_distance_seconds']['mean']:.2f} | {m['nearest_distance_anchor']['mean']:.2f} | {m['overlap_tIoU']['mean']:.3f} |")
    lines += ['', '设 ds、de 为有符号端点变化，中心位移 c=(ds+de)/2，长度变化 l=de−ds，则', '',
        '`|ds|+|de| = max(2|c|, |l|)`。', '',
        '中心和长度不是两项可相加的独立误差。下表将新增容量按最近最优候选的几何类型归属；'
        '这是描述性分解，centre 不等于事件 identity 错误，extent 不等于已经获得可实现的边界优化器。','',
        '| 面板 | 中心主导的收益占比 | 长度主导 | 等量 / 单边 | 单独移动 start | 单独移动 end | 扩张 | 收缩 | 同向移动 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p,s in panels.items():
        q=s['corruption']['capacity_gain_shares']
        fields=['gain_dominance_centre','gain_dominance_extent','gain_dominance_equal','gain_start_only','gain_end_only','gain_expansion','gain_contraction','gain_co_directional']
        lines.append('| '+p+' | '+' | '.join(f"{100*q[f]['mean']:.2f}%" for f in fields)+' |')
    balance=read(F/'ENDPOINT_BALANCE.json')
    lines += ['', '**严格主导分类需要连续分解校正。** 两端同向时 centre 项略大，也可能是'
        '一端大幅收缩、另一端只移动一个网格步。因此初次结果回读后增加了连续 post-hoc 诊断，'
        '没有新增阈值或改任何原始结果：`balance=2min(|ds|,|de|)/(|ds|+|de|)`，'
        '0 表示只动一端，1 表示两端等量；另报 centre/L1、extent/L1，两项不能相加。', '',
        '| 面板 | 收益加权端点 balance | 起点位移占比 | 终点位移占比 | centre 项 / L1 | extent 项 / L1 |',
        '|---|---:|---:|---:|---:|---:|']
    for p in panels:
        q=balance['summaries'][p]['corruption']['capacity_gain_shares']
        lines.append('| '+p+' | '+' | '.join(value(q['gain_'+f],True) for f in ['balance','start_fraction','end_fraction','centre_component','extent_component'])+' |')
    lines += ['', '该补充是在首批几何结果回读后提出，不能称为未看结果预注册。'
        '规则与独立验证见 [连续分解补充](../../../protocols/tastvg_temporal_local_oracle_balance_addendum_v1.md)。', '']
    lines += ['', '保持 A8 的某一边界不动、只在现有 32 候选里选另一边界的条件 oracle：', '',
        '| 面板 | 固定 start：收益 / 平均候选数 | 固定 end：收益 / 平均候选数 |',
        '|---|---:|---:|']
    for p,s in panels.items():
        m=s['corruption']['metrics']
        lines.append(f"| {p} | {value(m['fixed_start_gain'],True)} / {m['fixed_start_count']['mean']:.2f} | "
            f"{value(m['fixed_end_gain'],True)} / {m['fixed_end_count']['mean']:.2f} |")
    lines += ['', '这一限制不新增任何区间。候选数量少时，不能由零收益推断连续的单边界 refinement 无效。', '',
        '## 并列、集中度与 clean 对照','',
        '| 面板 | oracle 并列到达 | 并列几何类别不同 | bootstrap 零总增量抽样 | clean O32−O8 | clean r=.5 可收回占比 |',
        '|---|---:|---:|---:|---:|---:|']
    for p,s in panels.items():
        z=s['corruption'];m=z['metrics'];q=z['capacity_gain_shares']['anchor_0p5_accessible_extra'];cm=s['clean']['metrics'];cq=s['clean']['capacity_gain_shares']['anchor_0p5_accessible_extra']
        lines.append(f"| {p} | {m['oracle_tie']['cell_mean']*z['cells']:.0f} / {z['cells']} | "
            f"{m['tie_geometric_ambiguity']['cell_mean']*z['cells']:.0f} | {q['bootstrap_zero_denominator_draws']} / 10000 | "
            f"{value(cm['capacity_gain'],True)} | {value(cq,True)} |")
    lines += ['', '如果某次 bootstrap 的总 O32−O8 为零，占比没有定义，该抽样被计数而非强行置零。'
        '占比 CI 因此条件于该次总增量为正；原始 pp 的 CI 仍使用全部 10,000 抽样。HC 确认只有 '
        '2 个正增来源，必须结合逐来源和 leave-one-out，不能把容量或局部比例宣称为广泛稳定规律。','',
        '## 决策与可解释范围','']
    lines += [x+'\n' for x in decision['interpretation']]
    lines += ['A8→O32 位移不是 GT 边界错误；大幅长度修正和事件平移均可能产生大端点距离。'
        '当前数据不包含用于本诊断的真实 GT 端点，因此不能把几何距离直接命名为“选错事件”。', '',
        '此前 8→32 的 pairwise accuracy 比较改变了候选对总体。它与 top-1 同时改善/恶化并不独立证明 '
        'score calibration 或 multiple-hypothesis extreme-tail 因果机制；本次没有检验该假说。', '',
        '局部 oracle 只回答已有支持中的上限，不保证任一无标签 boundary refiner 能实现它。'
        '本次没有选择方法、修改在线阈值、晋升 CURRENT 或启动下一 GPU 实验。', '',
        '## 复现、审核和完整结果','',
        f"独立审核 `{audit['status']}`：{audit['checks']['expert_rows']} 个完整 expert record、"
        f"{audit['checks']['local_support_checks']} 个局部支持核验、{audit['checks']['summary_groups']} 个统计分组、"
        f"{audit['checks']['scalar_checks']} 标量，最大误差 {audit['max_abs_numeric_error']:.3g}。"
        '输入与代码 SHA256 绑定，候选/指标/状态 hash 保持 predecessor。', '',
        f"CPU 分解耗时 {resources['CPU_wall_seconds']:.2f} 秒（不含审计、制图、公开同步；不是 GPU kernel 时间）。"
        '新模型/专家/decoder/backward/prediction 全部为 0。', '',
        '来源宏平均先聚合 source/order/condition，再对来源平均；所有 bootstrap 和比值使用相同来源抽样。'
        '匿名 ROWS 保留每个 expert 的完整 32 区间/缓存指标、全部最优并列、最近最远位移、每个半径结果；'
        'CASES 同时保留大增量、近处正增与远处正增。原始媒体、query 文本、GT 坐标、权重和 raw cache 不公开。', '',
        '[规则](../../../protocols/tastvg_temporal_local_oracle_v1.md) · '
        '[计算](../../../scripts/run_tastvg_local_oracle_v1.py) · '
        '[独立审核](../../../scripts/audit_tastvg_local_oracle_public_v1.py)', '',
        '![Local oracle capacity](LOCAL_ORACLE_CAPACITY.png)', '',
        '![Endpoint displacement decomposition](ENDPOINT_DECOMPOSITION.png)','']
    text='\n'.join(lines)
    (F/'REVIEW.md').write_text(text);(ROOT/'docs/TA_TEMPORAL_LOCAL_ORACLE_REVIEW.md').write_text(
        text.replace('(LOCAL_ORACLE_CAPACITY.png)','(../results/tastvg_temporal_local_oracle/2026-10-03/LOCAL_ORACLE_CAPACITY.png)')
            .replace('(ENDPOINT_DECOMPOSITION.png)','(../results/tastvg_temporal_local_oracle/2026-10-03/ENDPOINT_DECOMPOSITION.png)')
            .replace('../../../protocols/','../protocols/').replace('../../../scripts/','../scripts/'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.hashsalt':'tastvg_local_oracle_v1'})
    radii=[0.,.1,.25,.5,.75,1.,1.5,2.,3.,4.,8.]
    fig,axes=plt.subplots(2,2,figsize=(11.4,7.4),sharex=True,sharey=True)
    for ax,(p,s) in zip(axes.flat,panels.items()):
        z=s['corruption'];q=z['capacity_gain_shares'];prefix=['anchor_'+str(r).replace('.','p') for r in radii]
        for field,label,col,style in [('accessible_extra','Extra gain locally accessible','#176b99','-'),
                ('near_gain_mass','Exact O32 optimum: nearest tie','#cc7b25','--'),
                ('far_gain_mass','Exact O32 optimum: farthest tie','#8d8e90',':')]:
            y=[q[a+'_'+field]['mean'] for a in prefix];ax.plot(radii,np.array(y)*100,style,color=col,label=label,lw=2)
            if field=='accessible_extra':
                c=np.array([q[a+'_'+field]['ci95'] for a in prefix]);ax.fill_between(radii,c[:,0]*100,c[:,1]*100,color=col,alpha=.13)
        ax.set_xscale('symlog',linthresh=.25);ax.set_ylim(-2,102);ax.set_xlim(0,8)
        ax.set_xticks([0,.25,.5,1,2,4,8]);ax.set_xticklabels(['0','.25','.5','1','2','4','8'])
        ax.set_title(p.replace('vidstg','VidSTG').replace('hc2','HC-STVG-v2').replace('search','development').replace('confirm','confirmation'))
        ax.grid(axis='y',alpha=.2);ax.axvline(.5,color='#999',lw=.7,alpha=.5)
    axes[0,0].legend(frameon=False,fontsize=8,loc='lower right')
    for ax in axes[1,:]:ax.set_xlabel('Endpoint L1 distance / A8 interval duration')
    for ax in axes[:,0]:ax.set_ylabel('Share of added O32 − O8 capacity (%)')
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(F/f'LOCAL_ORACLE_CAPACITY.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.2))
    labels=[];bottom=np.zeros(4);x=np.arange(4)
    for i,(p,s) in enumerate(panels.items()):labels.append(p.replace('vidstg','Vid').replace('hc2','HC').replace('search','dev').replace('confirm','confirm').replace('/','\n'))
    for f,label,col in [('gain_dominance_centre','Centre-dominant','#6b92ad'),('gain_dominance_extent','Extent-dominant','#d59e64'),('gain_dominance_equal','Equal / one boundary','#b5b5b5')]:
        y=np.array([s['corruption']['capacity_gain_shares'][f]['mean']*100 for s in panels.values()])
        axes[0].bar(x,y,bottom=bottom,color=col,label=label,width=.62);bottom+=y
    axes[0].set_xticks(x);axes[0].set_xticklabels(labels);axes[0].set_ylabel('Share of added capacity (%)');axes[0].set_ylim(0,102)
    axes[0].legend(frameon=False,fontsize=9,loc='lower right');axes[0].set_title('Displacement geometry of nearest tied optimum')
    for i,(p,s) in enumerate(panels.items()):
        rr=[r for r in read(F/p.split('/')[1]/p.split('/')[0]/'ROWS.json') if r['condition']!='clean' and r['capacity_gain']>1e-12]
        a=np.array([r['nearest']['doubled_centre_anchor'] for r in rr]);b=np.array([r['nearest']['length_anchor'] for r in rr])
        axes[1].scatter(a,b,s=np.array([r['capacity_gain'] for r in rr])*160+15,alpha=.65,label=labels[i].replace('\n',' '))
    axes[1].plot([0,8],[0,8],color='#777',lw=.8);axes[1].set_xscale('symlog',linthresh=.2);axes[1].set_yscale('symlog',linthresh=.2)
    axes[1].set_xlabel('2 × centre shift / A8 duration');axes[1].set_ylabel('Absolute length change / A8 duration')
    axes[1].set_title('Positive-gain cells (size: raw added gain)');axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(F/f'ENDPOINT_DECOMPOSITION.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.0))
    for j,(field,title) in enumerate([('gain_balance','Continuous endpoint balance'),('gain_extent_component','Extent component / endpoint L1')]):
        vals=[balance['summaries'][p]['corruption']['capacity_gain_shares'][field] for p in panels]
        y=np.array([z['mean']*100 for z in vals]);ci=np.array([z['ci95'] for z in vals])*100
        axes[j].bar(x,y,color=['#7ca6c2','#cead87','#397b9e','#a97540'],width=.62)
        axes[j].errorbar(x,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='none',color='#333',capsize=3,lw=1)
        axes[j].set_xticks(x);axes[j].set_xticklabels(labels);axes[j].set_ylim(0,103)
        axes[j].set_title(title);axes[j].set_ylabel('Added-capacity-weighted quantity (%)');axes[j].grid(axis='y',alpha=.15)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(F/f'CONTINUOUS_ENDPOINT_BALANCE.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print('LOCAL_ORACLE_REPORT_AND_FIGURES_COMPLETE')

if __name__=='__main__':main()
