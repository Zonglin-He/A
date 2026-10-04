"""Inspect-able R1 report and publication plots; no new model execution."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_optimizer_posterior_common_v1 import *
from vg_tta.decota_optimizer_posterior_r1_v1 import ARMS
import numpy as np

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    s=read(PUB/'SPATIAL_SUMMARY.json');t=read(PUB/'TEMPORAL_SUMMARY.json');d=read(PUB/'DECISION.json');cal=read(PUB/'CALIBRATION.json')
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,ds in zip(axes,DATASETS):
        means=[];errors=[]
        for a in ARMS:
            m=s[ds]['confirm'][a]['corruption']['metrics']['vs_before_v'];means.append(m['mean']*100)
            errors.append([(m['mean']-m['ci95'][0])*100,(m['ci95'][1]-m['mean'])*100])
        ax.errorbar(means,np.arange(len(ARMS)),xerr=np.array(errors).T,fmt='o',color='#267f99',capsize=3)
        ax.axvline(0,color='#888',ls='--',lw=.8);ax.set_yticks(np.arange(len(ARMS)),ARMS);ax.set_title(ds);ax.set_xlabel('Current correction vs inherited before (vIoU pp)');ax.invert_yaxis()
    fig.tight_layout();fig.savefig(PUB/'spatial_optimizer.png',dpi=220);fig.savefig(PUB/'spatial_optimizer.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),sharey=True)
    for ax,ds in zip(axes,DATASETS):
        for i,a in enumerate(['hard','full','extent','pm']):
            m=t[ds]['confirm'][a]['corruption']['metrics']['vs_native_v'];ax.errorbar(i,m['mean']*100,yerr=np.array([[m['mean']-m['ci95'][0]],[m['ci95'][1]-m['mean']]])*100,fmt='o',capsize=4,color='#b46c54')
        ax.axhline(0,color='#888',ls='--',lw=.8);ax.set_xticks(range(4),['Hard','Full','Extent','PM']);ax.set_title(ds);ax.set_ylabel('vIoU vs Native (pp)')
    fig.tight_layout();fig.savefig(PUB/'temporal_posterior.png',dpi=220);fig.savefig(PUB/'temporal_posterior.pdf');plt.close(fig)
    di=read(PUB/'SPATIAL_DIAGNOSTICS.json');case=[x for x in di if x['dataset']=='vidstg' and x['split']=='confirm' and x['source_id']==35 and x['condition']=='exposure_5' and x['order']=='order1']
    fig,axes=plt.subplots(1,3,figsize=(12,3.1))
    for x in case:
        if x['arm'] not in ['direct_adam','all_adam','all_sgd','admit_adam','top1_adam','all_post_authority']:continue
        line=axes[0].plot(x['posthoc_GT_vIoU'],label=x['arm'])[0];axes[1].plot(x['functional_displacements']);axes[2].plot(x['step_parameter_displacements'])
        r=next(r for r in read(PUB/'SPATIAL_ROWS.json') if r['dataset']=='vidstg' and r['split']=='confirm' and r['source_id']==35 and r['condition']=='exposure_5' and r['order']=='order1' and r['arm']==x['arm'])
        axes[0].scatter(x['selected_step'],r['v'],marker='D',s=30,color=line.get_color(),zorder=5)
        axes[1].scatter(x['selected_step'],x['used_functional_displacement'],marker='D',s=30,color=line.get_color(),zorder=5)
    axes[0].scatter([],[],marker='D',s=30,color='#333',label='Actual used output')
    for ax,title in zip(axes,['GT path + actual used output','Box displacement + actual use','Full proposal step norm']):ax.set_title(title);ax.set_xlabel('Optimization step')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(PUB/'source35_actuation.png',dpi=220);fig.savefig(PUB/'source35_actuation.pdf');plt.close(fig)
    text=['# DeCoTA optimizer/admission 与 temporal posterior：R1完整对照','',
        '本轮使用两集原32开发＋16确认来源、一query、双序、clean＋五类5%，均有历史曝光。空间共1152到达×10臂；时间同1152读出×5臂。',
        '空间同P1到达前状态固定，不是各优化器自己继承的长流；每query四观察，不是25%专家流。零新DINO与完整backbone，全部预测封存后GT。','',
        '| 数据集／确认corrupt | Spatial arm | 当前纠正 ΔvIoU pp [配对95%CI] | >20pp harm |','|---|---|---:|---:|']
    for ds in DATASETS:
        for a in ARMS:
            ss=s[ds]['confirm'][a]['corruption'];m=ss['metrics']['vs_before_v'];text.append(f"| {ds} | {a} | {m['mean']*100:+.4f} [{m['ci95'][0]*100:+.4f},{m['ci95'][1]*100:+.4f}] | {ss['tails']['harm_gt20pp']} |")
    text+=['','| 数据集／确认corrupt | Temporal | ΔvIoU vs Native pp [95%CI] |','|---|---|---:|']
    for ds in DATASETS:
        for a in ['hard','full','extent','pm']:
            m=t[ds]['confirm'][a]['corruption']['metrics']['vs_native_v'];text.append(f"| {ds} | {a} | {m['mean']*100:+.4f} [{m['ci95'][0]*100:+.4f},{m['ci95'][1]*100:+.4f}] |")
    text+=['','SGD学习率只按32开发clean的第一步功能位移匹配，不使用GT。每种目标单独校准，避免用Direct的梯度单位替代critic单位。','',
        '| 数据集 | 目标 | SGD lr | 第一功能位移相对误差 |','|---|---|---:|---:|']
    for ds in DATASETS:
        for a,v in cal[ds]['objectives'].items():text.append(f"| {ds} | {a} | {v['lr']:.8g} | {v['relative_error']:.5%} |")
    text+=['',f"开发Pareto选择的优化器臂：`{d['spatial_optimizer_search_winner']}`。未按确认重选；如果没有保持两集mean且不增加严重尾部的优化器，保留All-Adam参照，不宣称优化器问题已被解决。",
        f"Temporal posterior资格：`{d['temporal_posterior_qualification']}`；T1状态：`{d['T1']}`。",
        'Extent保留精确物理中心的概率边际；最终MAP中心可能变化。两个原offset各自读出再取原生envelope；没有把合并网格分布冒充原生策略。',
        'Authority是专家候选浓度，不是正确概率；post authority是最终参数位移插值，尚无functional或GT安全保证。',
        'R1结束不代表六轮路线完成。后续track、参数scope、真实online和预算/跨域阶段须按冻结门接续或明确记为条件未满足。',
        '', '![Spatial](../results/decota_optimizer_posterior/2026-10-04/spatial_optimizer.png)',
        '![Temporal](../results/decota_optimizer_posterior/2026-10-04/temporal_posterior.png)',
        '![Actuation](../results/decota_optimizer_posterior/2026-10-04/source35_actuation.png)']
    (ROOT/'docs/TA_DECOTA_OPTIMIZER_POSTERIOR_REVIEW.md').write_text('\n'.join(text)+'\n')
    write(PUB/'REPORT_BINDING.json',dict(report='docs/TA_DECOTA_OPTIMIZER_POSTERIOR_REVIEW.md',sha256=sha(ROOT/'docs/TA_DECOTA_OPTIMIZER_POSTERIOR_REVIEW.md')))
    print('REPORT_R1_WRITTEN',flush=True)

if __name__=='__main__':run()
