"""Publish complete positive/negative R1 evidence with source concentration."""
import os
os.environ['MPLBACKEND']='Agg'
import sys,json,csv,collections,time,hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
PUB=ROOT/'results/tastvg_dta_oracle_r1/2026-10-04'
BASE=ROOT/'artifacts/tastvg_dta_oracle_r1_v1'

def main():
    cfg=read(PUB/'CONFIG.json');s=read(PUB/'SUMMARY.json');rr=read(PUB/'ROWS.json');tr=read(PUB/'ADAPTATION_TRACES_SCORED.json')
    root=read(PUB/'ROOT_AUDIT.json');public=read(PUB/'PUBLIC_AUDIT.json');sel=read(PUB/'SOURCE_SELECTION.json');res=read(PUB/'RESOURCES.json')
    assert root['status']==public['status']=='pass'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42})
    colors={'vidstg':'#4979A7','hc2':'#D48451'}
    figs=[]
    fig,ax=plt.subplots(1,2,figsize=(10.4,3.25),layout='constrained')
    labels=['Vid search','Vid confirm','HC search','HC confirm']
    data=[('vidstg','search'),('vidstg','confirm'),('hc2','search'),('hc2','confirm')]
    for a,metric,title in zip(ax,['t','v'],['Temporal IoU','Video IoU (fixed spatial A)']):
        for j,(ds,sp) in enumerate(data):
            z=s[ds][sp]['expert_corrupt']['metrics']['R1_minus_N_'+metric];mu=100*z['mean'];lo,hi=100*np.array(z['ci95'])
            a.errorbar(mu,j,xerr=[[mu-lo],[hi-mu]],fmt='o',color=colors[ds],capsize=3,ms=6)
            a.text(hi+.35,j,f'{mu:+.2f}',va='center',fontsize=9)
        a.set_yticks(range(4),labels);a.invert_yaxis();a.axvline(0,color='.6',lw=.8)
        a.set_xlabel('Gain over zero-update native head (pp)');a.set_title(title);a.grid(axis='x',alpha=.15);a.set_xlim(-.9,25)
    for ext in ['png','pdf']:fig.savefig(PUB/f'r1_gradient_channel.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig);figs.append('r1_gradient_channel')
    fig,ax=plt.subplots(1,2,figsize=(10.4,3.1),layout='constrained')
    concentration={}
    for a,ds,label in zip(ax,['vidstg','hc2'],['Vid confirmation','HC confirmation']):
        v=s[ds]['confirm']['expert_corrupt']['metrics']['R1_minus_N_v']['source_values'];ids=list(v);vals=np.array(list(v.values()))*100
        a.bar(range(len(ids)),vals,color=colors[ds],width=.65);a.set_xticks(range(len(ids)),ids);a.set_xlabel('Anonymous source ID')
        a.set_ylabel('Mean vIoU gain over native (pp)');a.set_title(label);a.grid(axis='y',alpha=.15)
        a.axhline(vals.mean(),color='.3',ls='--',lw=1,label=f'Source mean {vals.mean():.2f} pp');a.legend(frameon=False,fontsize=9)
        positive=np.maximum(vals,0);total=positive.sum();top=np.argsort(-positive);loo=(vals.sum()-vals)/(len(vals)-1)
        concentration[ds]=dict(positive_sources=int((vals>1e-10).sum()),total_sources=len(vals),
            largest_positive_source=ids[int(top[0])],largest_positive_share=float(positive[top[0]]/total) if total else None,
            top_two_positive_share=float(positive[top[:2]].sum()/total) if total else None,
            leave_one_source_out_gain_range_pp=[float(loo.min()),float(loo.max())])
    for ext in ['png','pdf']:fig.savefig(PUB/f'r1_confirmation_sources.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig);figs.append('r1_confirmation_sources')
    fig,ax=plt.subplots(1,2,figsize=(9.6,3.1),layout='constrained')
    for a,ds,label in zip(ax,['vidstg','hc2'],['Vid: 31 source validation queries','HC: 16 source validation queries']):
        path=sel[ds]['path'];a.semilogx([p['lr'] for p in path],[100*(p['mean_after_tIoU']-p['mean_before_tIoU']) for p in path],marker='o',color=colors[ds])
        a.set_xlabel('Learning rate (source validation only)');a.set_ylabel('tIoU gain after 3 steps (pp)');a.set_title(label);a.grid(alpha=.2)
    for ext in ['png','pdf']:fig.savefig(PUB/f'r1_source_lr.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig);figs.append('r1_source_lr')

    cols=['dataset','split','source_id','condition','order','arrival','expert_scheduled','GT_supervised',
          'N_v','N_t','A8_v','A8_t','R1_v','R1_t','R1_minus_N_v','R1_minus_N_t','R1_minus_A8_v','R1_minus_A8_t',
          'teacher_MAP_v','teacher_MAP_t','GT_time_v','GT_time_t']
    with (PUB/'ROWS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r[k] for k in cols} for r in rr)
    with (PUB/'SOURCE_VALIDATION.csv').open('w',newline='') as f:
        z=read(PUB/'SOURCE_VALIDATION_ROWS.json');cols2=['dataset','source_id','lr','before_t','after_t','teacher_MAP_t','objective_before','objective_after']
        w=csv.DictWriter(f,fieldnames=cols2);w.writeheader();w.writerows({k:r[k] for k in cols2} for r in z)
    write(PUB/'SOURCE_CONCENTRATION.json',concentration)
    cases=[]
    for ds in s:
        for sp in ['search','confirm']:
            q=[r for r in rr if r['dataset']==ds and r['split']==sp and r['expert_scheduled'] and r['condition']!='clean']
            for baseline in ['N','A8']:
                for sign,cell in [('largest_gain',max(q,key=lambda r:r['R1_minus_'+baseline+'_v'])),('largest_harm',min(q,key=lambda r:r['R1_minus_'+baseline+'_v']))]:
                    cases.append(dict(dataset=ds,split=sp,baseline=baseline,selection=sign,cell=cell,
                        trace=next(t['trace'] for t in tr if t['cell_key']==cell['cell_key'])))
    write(PUB/'CASES.json',cases)
    changes={}
    for ds in s:
        changes[ds]={}
        for sp in ['search','confirm']:
            q=[r for r in rr if r['dataset']==ds and r['split']==sp and r['expert_scheduled'] and r['condition']!='clean']
            changes[ds][sp]=dict(cells=len(q),native_interval_changed=sum(r['native_interval_changed'] for r in q),
                loss_decreased=sum(r['loss_after']<r['loss_before'] for r in q),
                source_gains=s[ds][sp]['expert_corrupt']['metrics']['R1_minus_N_v']['source_values'])
    decision=dict(R1_gradient_channel='supported_in_locked_supervised_scope_with_source_concentration',
        deployable_TTA_claim=False,expert_teacher_tested=False,persistence_tested=False,
        R1_target_lr_tuning=False,source_lr_selected_at_upper_grid_boundary=True,
        A8_replacement='Vid confirmation positive; HC uncertain and >5pp harms persist',
        next_conditional_question='Can deployable expert supervision replace GT in this same head/steps/learning-rate interface?',
        R2_started=False,R3_started=False,head_LN_started=False,production_promoted=False,
        source_concentration=concentration,native_interval_changes=changes)
    write(PUB/'DECISION.json',decision)
    def m(ds,sp,field,panel='expert_corrupt'):
        z=s[ds][sp][panel]['metrics'][field];return f'{100*z["mean"]:+.4f} [{100*z["ci95"][0]:+.4f}, {100*z["ci95"][1]:+.4f}]'
    text=['# Oracle Distributional Temporal Adaptation: native head feasibility', '',
      '**R1 已完成：正确的软时间监督可以通过原生末层梯度更新改善两个数据集的当前-query定位。** 这是一轮显式GT监督的容量/执行诊断，不是无标签方法成绩。确认专家corruption上相对zero-update native，Vid ΔtIoU +8.0634pp、ΔvIoU +5.8219pp；HC +5.2840/+2.6037pp，四个源配对区间均高于零。保留source集中性：Vid确认收益83.9%来自一个source；相对现有A8，HC尚未建立可靠增量且有严重负例。未晋升方法，未启动R2/R3。','',
      '## 实际设置与旧实验的区别','',
      'TA-STVG官方同域EMA checkpoint，Vid/HC分别使用原Vid-trained/HC2-trained模型；setting是原Paper48 transient deployment corruption，**不是cross-domain**。空间A的1792参数轨迹、框、原始两offset采样、输入像素和25%专家到达位置全部复用。生产注册仍为DeCoTA，与本研究A不同。','',
      '原设计每集32开发+16确认来源、每源一query、双序、clean+drop/freeze/blur/occlusion/exposure各5%，总1152到达；仅288个已缓存专家位置实际进行R1（Vid16开发/8确认独立专家源，HC14/7）。其余864到达固定缓存A，完整流读出是缓存模拟，不是新跑一条online stream。来源均历史曝光，confirmation指本轮来源划分，不是fresh test。','',
      '只适应原`temp_embed.layers.1`：2×256 weight+2 bias共514名义参数。cached hidden先经冻结MLP首层及ReLU，eval dropout关闭；未把decoder hidden直接当原末层输入。bias整体平移被概率归一化消掉，实际可辨识更新方向至多512。无backbone/suffix/expert新增调用、无decoder/LN/routing/spatial更新。','',
      '完整合法i<j joint span分布按两个offset分别归一化，损失平均；保持原生FP32 MAP及两offset物理envelope。没有切换成merged-grid argmax，故收益来自同一个解码接口的参数更新。所有47源验证输入和288目标输入no-update区间精确匹配旧native，目标CUDA/CPU logits最大误差见ROOT_AUDIT。','',
      'q_GT为物理边界二维Gaussian，各offset sigma为该offset中位相邻帧距（一原网格cell）；不裁剪GT。使用forward KL(q_GT||pθ)+KL(p0||pθ)，p0每query冻结，普通SGD三步、总是最后一步、无step选择/裁剪/EMA/weight decay。prior KL是软惩罚，不是有严格半径的trust region。GT Gaussian监督正确边界，但不等于固定空间下完美vIoU teacher。','',
      '旧native-head实验只有Old8配对受限概率、rank reverse-KL、一次有界更新、未来query收益近零；本轮完整支持/GT forward-KL/3步/当前query/reset是实质不同的可行性诊断。','',
      '## 源验证选择：目标上不调参','',
      '复用官方train来源派生的31 Vid+16 HC验证查询及hidden（历史曝光），与目标source/media互斥。仅测试预锁五lr，其余beta1/K3/sigma固定；每query复位，按最终source平均tIoU选择，exact tie取较小lr。原95/48 fitting来源未用于训练任何readout。两个lr选择均先seal，随后才开始本轮目标GT teacher。','',
      '| 数据集 | 源验证query | 选择lr | native tIoU % | 3步后 tIoU % |','|---|---:|---:|---:|---:|']
    for ds,label in [('vidstg','Vid'),('hc2','HC')]:
        p=sel[ds]['path'][sel[ds]['selected_index']]
        text.append(f'| {label} | {p["queries"]} | {p["lr"]} | {100*p["mean_before_tIoU"]:.4f} | {100*p["mean_after_tIoU"]:.4f} |')
    text +=['','两集均选.01，即这次小网格的上边界。只能称网格内最佳，不称全局最优；本轮没有扩大范围或在目标上再选。完整235个query-lr结果公开。R1直接使用GT Gaussian；附件lambda=.5的PoE构造属于未运行的R2，此轮不用。','',
       '## 参数通道主对照：专家corruption，R1 − Native','',
       'N是原checkpoint head在同A空间输入上的zero-update native输出；A8是Uniform空间持续学习+原UVTG Fast时间选管。先比较R1−N，才能把原专家readout差异和梯度执行分开。单位pp；CI为10000次配对source-bootstrap。','',
       '| 面板 | ΔtIoU | ΔvIoU | v增益/损害cells | >5pp损害 |','|---|---|---|---:|---:|']
    for ds,sp in [('vidstg','search'),('hc2','search'),('vidstg','confirm'),('hc2','confirm')]:
        z=s[ds][sp]['expert_corrupt'];n=z['negative_tails']['N']
        text.append(f'| {ds}/{sp} ({z["sources"]}源/{z["cells"]}cells) | {m(ds,sp,"R1_minus_N_t")} | {m(ds,sp,"R1_minus_N_v")} | {n["gain"]}/{n["harm"]} | {n["severe_harm"]} |')
    text +=['','确认Vid有4/8源正增益、HC6/7；Vid source33占正增益约83.9%，前两源合计99.4%；HC最大source占约52.8%，前两源合计76.4%。两序均正，幅度不同；删除source33后Vid仍正均值，但本轮没有为该删源均值建立独立确认。不能用一个小面板宣布普遍稳定适应。','',
       '所有240个corruption专家cell的loss下降；Vid开发52/80、确认11/40区间改变，HC54/80、21/40。三步小更新仍常未跨越离散MAP决策边界。HC开发有一例loss下降而vIoU下降0.5268pp，证明正确边界监督也不保证每条固定空间tube的最终vIoU改善。','',
       '## 实际分支参照：专家corruption，R1 − A8','',
       '| 面板 | ΔtIoU | ΔvIoU | >5pp损害 |','|---|---|---|---:|']
    for ds,sp in [('vidstg','search'),('hc2','search'),('vidstg','confirm'),('hc2','confirm')]:
        z=s[ds][sp]['expert_corrupt'];text.append(f'| {ds}/{sp} | {m(ds,sp,"R1_minus_A8_t")} | {m(ds,sp,"R1_minus_A8_v")} | {z["negative_tails"]["A8"]["severe_harm"]} |')
    text +=['','**GT参数学习有收益，并不等于当前设置可安全替换A8。** Vid开发均值仍略低于A8；HC确认均值正但区间跨零，6/40个专家corruption到达相对A8下降超过5pp。Native通道的零严重损害不能抹掉这个实际参照上的负尾。','',
       '## 完整缓存流与clean','',
       '| 确认完整corrupt流 | R1−A8 ΔvIoU | R1−A8 ΔtIoU |','|---|---|---|',
       f'| Vid 16源/160到达 | {m("vidstg","confirm","R1_minus_A8_v","flow_corrupt")} | {m("vidstg","confirm","R1_minus_A8_t","flow_corrupt")} |',
       f'| HC 16源/160到达 | {m("hc2","confirm","R1_minus_A8_v","flow_corrupt")} | {m("hc2","confirm","R1_minus_A8_t","flow_corrupt")} |','',
       '非专家输出逐值保留A，未测试时间feedback向未来位置迁移。上表由真实source聚合得出，不用专家增量×25%代替。Full-flow N只在专家位置取消Fast，其余仍A；teacher_MAP也仅在专家位置干预。GT-time控制覆盖全部到达。','',
       f'确认clean专家R1−N ΔvIoU：Vid {m("vidstg","confirm","R1_minus_N_v","expert_clean")}，HC {m("hc2","confirm","R1_minus_N_v","expert_clean")}。Clean收益说明GT监督并非corruption-specific机制，不单开generic refinement研究线。所有clean、order、nonexpert表在SUMMARY/ROWS完整保存。','',
       '## 正控、剩余上限与正负例','',
       '| 确认专家corruption | Native vIoU % | R1 % | GT Gaussian MAP % | GT-time % |','|---|---:|---:|---:|---:|']
    for ds in ['vidstg','hc2']:
        z=s[ds]['confirm']['expert_corrupt']['metrics'];text.append('| '+ds+' | '+' | '.join(f'{100*z[k]["mean"]:.4f}' for k in ['N_v','R1_v','teacher_MAP_v','GT_time_v'])+' |')
    text +=['','teacher_MAP是无学习地把GT Gaussian直接送进同一个native支持/解码接口的诊断正控，不是head TTA成绩。它与GT-time差距包含现有采样支持/高斯/双offsetenvelope误差；R1仍远低于teacher_MAP，不能宣布已穷尽末层容量。','',
       '- Vid确认source33 occlusion/order2：Native 4.2980%→R1 43.3957%，+39.0977pp；该源驱动大量平均收益。','- HC确认source47 exposure/order1：Native 22.3256%→R1 43.1444%，+20.8189pp。','- HC开发source23 occlusion/order2：Native 11.1294%→R1 10.6025%，−0.5268pp，loss仍下降。','- 相对A8的严重损害：Vid开发source19 frame_drop/order2 A8 67.6521%→R1 18.6517%；HC确认source41 exposure/order2 A8 63.8068%→R1 46.1603%。这些都是已有A8更好的情况，并未因为GT teacher而自动被保留。','',
       'CASES保留各数据集/面板相对N和A8的最大收益与最大损害，以及三步loss/位移/指标。零损害面板的最小差值也照实保存，不伪造失败案例。','',
       '## 核验、成本与结论范围','',
       f'6项CPU接口测试通过。根审计独立从joint-marginal交叉熵代数重算523个query-configuration（235source+288target）的全部1569次梯度，并核验每步SGD/参数复位/logits/native解码；1152个A状态/框绑定及所有official dense指标复核通过。根检查数{sum(root["checks"].values()):,}，公开标量{public["checks"]["numeric_scalars"]:,}+tail计数{public["checks"]["tail_counts"]}。公开审计只能复核匿名scalar/选择/汇总，不能凭公开副本重建被排除的private hidden/head weights。','',
       f'模型通道source计算{res["source_CPU_wall_seconds"]:.3f}s、target adaptation {res["target_adaptation_CPU_wall_seconds"]:.3f}s、dense评分{res["dense_score_CPU_wall_seconds"]:.3f}s；这些阶段内时间不含先行metadata/hash验证、checkpoint加载、开发/审计/报告。根独立核验{root["CPU_wall_seconds"]:.3f}s，公开审计{public["CPU_wall_seconds"]:.3f}s另计。705 source+864 target backward，共1569次；0GPU初始化、0backbone、0suffix、0专家新调用、0空间更新或跨query时间写入。Private临时优化轨迹{res["private_run_bytes"]:,}bytes，未公开权重/hidden/GT坐标/媒体。','',
       '启动时旧PyTorch2.0.1不支持mmap，在零预测/零目标GT阶段退出；失败原件已保存，改用已有PyTorch2.7.0的CPU运行，科学配置没有改变。首次合成测试的offset等距tie期望end=9修正为原生早索引end=8，科学数据尚未运行。','',
       '**本轮裁决：保留head-only distributional adaptation作为有正证据的梯度接口。** 以GT Gaussian为teacher时能修正原生时间输出，因此不需要先换backbone或扩到decoder/LN。R2的下一问题是同接口上可部署专家teacher能否保留收益；尚未测试。若换成PoE expert teacher失败，不能唯一归因为专家错误，还包含teacher强度/多峰到单个factorized head的投影差异。R1不建立无标签性能、跨源普遍稳定性或online persistence，不恢复旧队列，不改变A/CURRENT。','',
       '## 可复现材料','',
       '- 协议：`protocols/tastvg_dta_oracle_r1_v1.md`；执行：`docs/tastvg_dta_oracle_r1_v1/EXECUTION.md`。','- 输入/代码锁与private预测barrier：`artifacts/tastvg_dta_oracle_r1_v1`。','- 全部匿名source路径、1152行指标、288条三步轨迹、CI/负尾/cases/三图：`results/tastvg_dta_oracle_r1/2026-10-04`。','- 六测试：`.conda/tubedetr/bin/python -B scripts/test_tastvg_dta_oracle_r1_v1.py`。','- 根：`.conda/tubedetr/bin/python -B scripts/audit_tastvg_dta_oracle_r1_v1.py root`；公开：同脚本加公开结果目录。','']
    (ROOT/'docs/TA_DTA_ORACLE_R1_REVIEW.md').write_text('\n'.join(text))
    own=['vg_tta/tastvg_dta_oracle_v1.py','scripts/run_tastvg_dta_oracle_r1_v1.py','scripts/test_tastvg_dta_oracle_r1_v1.py',
      'scripts/audit_tastvg_dta_oracle_r1_v1.py','scripts/report_tastvg_dta_oracle_r1_v1.py',
      'protocols/tastvg_dta_oracle_r1_v1.md','docs/tastvg_dta_oracle_r1_v1/EXECUTION.md','docs/TA_DTA_ORACLE_R1_REVIEW.md']
    deps=['methods/decota_final_simplified_v1/objectives.py','scripts/decota_matrix_common_v1.py',
      'scripts/tastvg_correction_views_common_v1.py','vg_tta/tastvg_oracle_event5_v1.py',
      'vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py']
    write(PUB/'CODE_BINDING.json',dict(files={f:sha(ROOT/f) for f in own+deps},runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json')))
    write(PUB/'FIGURE_MANIFEST.json',dict(files={str((PUB/f'{f}.{ext}').relative_to(ROOT)):sha(PUB/f'{f}.{ext}') for f in figs for ext in ['png','pdf']},
        source_summary_sha256=sha(PUB/'SUMMARY.json'),source_selection_sha256=sha(PUB/'SOURCE_SELECTION.json')))
    print('REPORT_AND_THREE_FIGURES_WRITTEN',flush=True)

if __name__=='__main__':main()
