"""Anonymous CPU report for matched isolated negative writes and reset-u control."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,csv,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_negative_evidence_common_v1 import BASE,PUB,ARMS,DATASETS,read,write,sha

PANELS=[('vidstg','search'),('vidstg','confirm'),('hc2','search'),('hc2','confirm')]
COLORS={'rank_cross':'#8B76A4','negative_global':'#CA815C','negative_local':'#448D86','reset_u':'#6489B0'}


def concentration(z):
    vals=np.array(list(z['source_values'].values()),float);pos=np.maximum(vals,0);neg=np.maximum(-vals,0)
    leave=[float(np.delete(vals,j).mean()*100) for j in range(len(vals))] if len(vals)>1 else []
    return dict(sources=len(vals),positive_sources=int((vals>1e-12).sum()),negative_sources=int((vals< -1e-12).sum()),
        largest_positive_share=float(pos.max()/pos.sum()) if pos.sum()>0 else None,
        top_two_negative_share=float(np.sort(neg)[-2:].sum()/neg.sum()) if neg.sum()>0 else None,
        leave_one_source_out_mean_range_pp=[min(leave),max(leave)] if leave else None,
        source_values=z['source_values'])


def diagnostics(traces,rows):
    """Explicit cell/step/entry denominators; refreshed targets are not one loss."""
    out={}
    for ds,sp in PANELS:
        out.setdefault(ds,{})[sp]={}
        for arm in ARMS:
            tt=[r for r in traces if r['dataset']==ds and r['split']==sp and r['arm']==arm and r['condition']!='clean']
            rr=[r for r in rows if r['dataset']==ds and r['split']==sp and r['condition']!='clean']
            known=sum(t['known_negative_entries'] for t in tt);mass=sum(t['known_negative_evidence_mass'] for t in tt)
            errors=sum(t['negative_GT_better_entries'] for t in tt);error_mass=sum(t['negative_GT_better_evidence_mass'] for t in tt)
            gknown=sum(t['global_negative_candidates'] for t in tt);gmass=sum(t['global_negative_evidence_mass'] for t in tt)
            gerrors=sum(t['global_negative_GT_better_candidates'] for t in tt);gerror_mass=sum(t['global_negative_GT_better_evidence_mass'] for t in tt)
            x=dict(cells=len(rr),steps=len(tt),candidate_noncenter_observation_entries=sum(8*t['valid_observations'] for t in tt),
                known_negative_entries=known,negative_GT_better_entries=errors,
                observed_negative_misfire_entry_fraction=errors/known if known else None,
                known_negative_evidence_mass=mass,negative_GT_better_evidence_mass=error_mass,
                observed_negative_misfire_weight_fraction=error_mass/mass if mass else None,
                global_negative_candidates=gknown,global_negative_GT_better_candidates=gerrors,
                global_negative_GT_better_candidate_fraction=gerrors/gknown if gknown else None,
                global_negative_evidence_mass=gmass,global_negative_GT_better_evidence_mass=gerror_mass,
                global_negative_GT_better_weight_fraction=gerror_mass/gmass if gmass else None,
                unknown_or_outside_event_observations=sum(t['unknown_or_outside_event_observations'] for t in tt),
                observed_GT_frames=sum(t['observed_GT_frames'] for t in tt),
                steps_with_no_GT_supported_observation=sum(t['observed_GT_frames']==0 for t in tt),
                frozen_step_loss_decreased=sum(t['loss_after']<t['loss']-1e-12 for t in tt),
                frozen_step_loss_increased=sum(t['loss_after']>t['loss']+1e-12 for t in tt),
                frozen_step_loss_equal=sum(abs(t['loss_after']-t['loss'])<=1e-12 for t in tt),
                frozen_step_loss_down_task_down=sum(t['loss_after']<t['loss']-1e-12 and t['post_prediction_v']<t['pre_prediction_v']-1e-12 for t in tt),
                mean_eta_gradient_norm=float(np.mean([t['eta_gradient_norm'] for t in tt])) if tt else None,
                mean_step_parameter_displacement=float(np.mean([t['parameter_displacement'] for t in tt])) if tt else None,
                mean_arrival_parameter_displacement=float(np.mean([r[arm+'_parameter_displacement'] for r in rr])) if rr else None,
                mean_output_gradient_observed_norm=float(np.mean([t['output_gradient_observed_norm'] for t in tt])) if tt else None,
                mean_output_gradient_unobserved_norm=float(np.mean([t['output_gradient_unobserved_norm'] for t in tt])) if tt else None,
                mean_output_movement_observed_norm=float(np.mean([t['output_movement_observed_norm'] for t in tt])) if tt else None,
                mean_output_movement_unobserved_norm=float(np.mean([t['output_movement_unobserved_norm'] for t in tt])) if tt else None,
                target_undecided_log_odds_max_error=max([t['target_undecided_log_odds_max_error'] for t in tt if t['target_undecided_log_odds_max_error'] is not None],default=None),
                actual_undecided_log_odds_max_change=max([t['actual_undecided_log_odds_max_change'] for t in tt if t['actual_undecided_log_odds_max_change'] is not None],default=None),
                native_interval_changed_steps=sum(t['native_interval_changed'] for t in tt))
            changes=[t['GT_better_probability_change'] for t in tt if t['GT_better_probability_change'] is not None]
            x['mean_GT_better_probability_change']=float(np.mean(changes)) if changes else None
            x['probability_diagnosis_steps']=len(changes)
            source={}
            for sid in sorted({t['source_id'] for t in tt}):
                st=[t for t in tt if t['source_id']==sid];m=sum(t['known_negative_evidence_mass'] for t in st)
                source[str(sid)]=dict(steps=len(st),known_evidence_mass=m,
                    misfire_weight_fraction=sum(t['negative_GT_better_evidence_mass'] for t in st)/m if m else None)
            x['per_source']=source;out[ds][sp][arm]=x
    return out


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tick=time.monotonic();cfg=read(PUB/'CONFIG.json');s=read(PUB/'SUMMARY.json')
    root=read(PUB/'ROOT_AUDIT.json');public=read(PUB/'PUBLIC_AUDIT.json');assert root['status']==public['status']=='pass'
    current=read(PUB/'ROWS.json');future=read(PUB/'FUTURE_ROWS.json');reset=read(PUB/'RESET_U_ROWS.json');traces=read(PUB/'STEP_DIAGNOSTICS.json')
    diag=diagnostics(traces,current);write(PUB/'EXECUTION_DIAGNOSTICS.json',diag)
    c={};cases=[]
    for scope,rows in [('current',current),('future',future),('reset_u',reset)]:
        c[scope]={}
        for ds,sp in PANELS:
            c[scope].setdefault(ds,{})[sp]={}
            contrasts=[('reset_u','A'),('reset_u_fixedA','A')] if scope=='reset_u' else [(a,'rank_native') for a in ARMS[1:]]+[('negative_local','negative_global')]
            q=[r for r in rows if r['dataset']==ds and r['split']==sp and r['condition']!='clean']
            for a,b in contrasts:
                metric=a+'_minus_'+b+'_v'
                c[scope][ds][sp][a+'_minus_'+b]=concentration(s[scope][ds][sp]['corrupt']['metrics'][metric])
                for name,choose in [('largest_gain',max),('largest_harm',min)]:
                    r=choose(q,key=lambda r:r[metric]);case=dict(scope=scope,dataset=ds,split=sp,arm=a,baseline=b,selection=name,row=r)
                    if scope=='current':case['scalar_steps']=[t for t in traces if t['cell_key']==r['cell_key'] and t['arm'] in [a,b]]
                    cases.append(case)
    write(PUB/'SOURCE_CONCENTRATION.json',c);write(PUB/'CASES.json',cases)
    confirm={a:all(s['current'][ds]['confirm']['corrupt']['metrics'][a+'_minus_rank_native_v']['ci95'][0]>0 for ds in DATASETS) for a in ARMS[1:]}
    decision=dict(status='completed_scoped_comparison',confirmation_positive_lower_bound_vs_rank_native=confirm,
        interpretation='Per-arm paired confirmation evidence; historical exposure and severe tails remain visible. No amplitude-matched or fully online negative-objective control was run.',
        no_evidence_is_correctness=False,negative_semantics_alone_causally_established=False,
        full_online_negative_loss_established=False,parameter_bundle_components_isolated=False,
        single_write_transfer_is_full_persistence=False,GT_used_for_optimization=False,
        reset_u_is_separate_control=True,production_promoted=False,DTA_promoted=False,
        additional_arms_started=False,lambda_retuned=False,time=time.time())
    write(PUB/'DECISION.json',decision)
    # Rectangular anonymous CSVs preserve all scalar cells; nested probability arrays remain JSON.
    for name,rows in [('ROWS',current),('FUTURE_ROWS',future),('RESET_U_ROWS',reset)]:
        columns=list(rows[0])
        assert all(set(r)==set(columns) for r in rows)
        p=PUB/(name+'.csv');assert not p.exists()
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42})
    figs=[]
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(PUB/(name+'.'+ext),dpi=220,bbox_inches='tight')
        plt.close(fig);figs.append(name)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    labels=['Vid development','Vid confirmation','HC development','HC confirmation']
    for ax,scope,title in zip(axes,['current','future','reset_u'],['Current spatial write: fixed A interval','Next nonexpert: one isolated write','Separate true online reset-u stream']):
        arms=['reset_u'] if scope=='reset_u' else ARMS[1:]
        baseline='A' if scope=='reset_u' else 'rank_native'
        for j,(ds,sp) in enumerate(PANELS):
            for k,a in enumerate(arms):
                z=s[scope][ds][sp]['corrupt']['metrics'][a+'_minus_'+baseline+'_v'];mu=100*z['mean'];lo,hi=np.asarray(z['ci95'])*100
                off=(k-(len(arms)-1)/2)*.16
                ax.errorbar(mu,j+off,xerr=[[max(mu-lo,0)],[max(hi-mu,0)]],fmt='o',color=COLORS[a],capsize=3,label=a if j==0 else None)
        ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.axvline(0,color='.6',lw=.8);ax.grid(axis='x',alpha=.15)
        ax.set_title(title,fontsize=11);ax.set_xlabel('Paired source-macro vIoU gain (pp)');ax.legend(frameon=False,fontsize=8)
    save(fig,'negative_evidence_paired_contrasts')
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    for ax,ds in zip(axes,DATASETS):
        metrics=s['current'][ds]['confirm']['corrupt']['metrics'];ids=list(metrics['negative_local_minus_rank_native_v']['source_values']);x=np.arange(len(ids))
        for a,offset in [('negative_global',-.18),('negative_local',.18)]:
            v=metrics[a+'_minus_rank_native_v']['source_values'];ax.bar(x+offset,[100*v[i] for i in ids],width=.33,color=COLORS[a],label=a)
        ax.set_xticks(x,ids);ax.axhline(0,color='.6',lw=.8);ax.set_xlabel('Anonymous write source');ax.set_ylabel('Current vIoU gain over Rank native (pp)')
        ax.set_title(ds+' historically exposed confirmation');ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=8)
    save(fig,'negative_evidence_confirmation_sources')
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    for ax,ds in zip(axes,DATASETS):
        q=[r for r in current if r['dataset']==ds and r['condition']!='clean']
        for a in ARMS[1:]:
            ax.scatter([r[a+'_parameter_displacement'] for r in q],[100*r[a+'_minus_rank_native_v'] for r in q],s=13,alpha=.6,color=COLORS[a],label=a)
        ax.set_xscale('symlog',linthresh=1e-6);ax.axhline(0,color='.6',lw=.8);ax.set_xlabel('Actual arrival parameter displacement (1792 coordinates)')
        ax.set_ylabel('Current vIoU gain over Rank native (pp)');ax.set_title(ds+' corruption, development + confirmation');ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.12)
    save(fig,'negative_evidence_update_scale')
    def value(scope,ds,sp,k,panel='corrupt'):
        z=s[scope][ds][sp][panel]['metrics'][k];return f'{100*z["mean"]:+.4f} [{100*z["ci95"][0]:+.4f}, {100*z["ci95"][1]:+.4f}]'
    lines=['# Negative evidence: matched isolated spatial writes and reset-u control','',
        '本轮四臂同 A 更新前状态的专家位置对照、下一 query 单次迁移，以及独立 reset-u 在线对照已完成封存、后验评分和独立审计。裁决见 DECISION.json；所有区间、负尾、参数变化强度和相反案例一并保留。当前空间反事实读出不等于已部署的 prequential 输出，单次未来迁移也不等于负证据完整在线流。','',
        '每集32开发源、16源互斥确认、每源一query、双序、clean及五种5% corruption。总1152到达，288专家位置含240corrupt/48clean；确认均有历史曝光。原官方同域EMA checkpoint、Paper48输入像素、缓存H与Uniform Sa2VA、source半径rho .05/D4的九probe、1792空间参数保持。Vid包 lr .033761698432507946 / teacher T .34902548789596055 / K1，HC包 lr .006097133675874025 / T1 / K8。rank_cross仅交换lr/T/K整包，不能确定是哪一个超参数造成差异；本数据集L1/GIoU系数Vid5/3、HC5/4保持。','',
        '当前/未来/每步空间GT诊断都固定该query的原封存A final interval。GT事件内有密集标注的实际观察帧另报 spatial IoU；无交集标为未知。完整事件sIoU仍保留原HC最后一标注帧inclusive，与时间span/scorer半开端点分开。正确性阈值按官方vIoU严格大于.3/.5。','',
        '## 当前专家位置：同前态、固定A时间','',
        '下表为corruption；source宏平均、10000次配对bootstrap seed20261004，单位pp。原Rank必须复现旧A的梯度/状态；新臂更新后当前框是隔离空间反事实读出。','',
        '| 面板 | arm−Rank native | ΔvIoU [95%CI] | >5pp / >20pp损害 | gross gain / loss pp |','|---|---|---:|---:|---:|']
    for ds,sp in PANELS:
        z=s['current'][ds][sp]['corrupt']
        for a in ARMS[1:]:
            tail=z['negative_tails'][a+'_minus_rank_native'];met=z['metrics']
            lines.append(f'| {ds}/{sp} ({z["sources"]}源/{z["cells"]}cells) | {a} | {value("current",ds,sp,a+"_minus_rank_native_v")} | {tail["severe_harm_gt5pp"]}/{tail["severe_harm_gt20pp"]} | {100*met[a+"_gross_gain_rank_native"]["mean"]:.4f}/{100*met[a+"_gross_loss_rank_native"]["mean"]:.4f} |')
    lines+=['','negative_local−negative_global直接比较同帧分布与全clip分布；两者lambda1、native lr/K相同。它改变损失空间支持，不是仅在同一候选上排序；不同arm后续K步状态与支持会自然分叉。负证据与Rank还同时改变teacher形式和有效梯度强度，损害减少不能唯一归因负证据语义。没有幅度匹配臂，也没有lambda扫参。','',
        '| 面板 | local−global ΔvIoU | global−更新前 ΔvIoU | local−更新前 ΔvIoU |','|---|---:|---:|---:|']
    for ds,sp in PANELS:lines.append(f'| {ds}/{sp} | {value("current",ds,sp,"negative_local_minus_negative_global_v")} | {value("current",ds,sp,"negative_global_minus_pre_v")} | {value("current",ds,sp,"negative_local_minus_pre_v")} |')
    lines+=['','## 负证据、teacher和实际梯度','',
        '逐有效观察帧e_jk=max(IoU(center_j,E_j)−IoU(candidate_kj,E_j),0)。global先逐帧取正部再平均，local每帧建立分布；不把正部平均改为平均后正部。q∝p0 exp(−e)，lambda固定1；KL(p||q)=KL(p||p0)+E_p[e]+logZ，最后项在单步中固定。e=0不是correctness标签；没有观察与全零证据显式零梯度。','',
        'teacher中相同e（尤其未决e=0）的候选odds保持，但有限SGD及共享参数可改变实际odds和未观察帧。center的e恒为0，归一化可能保留错误center，不能用该构造宣称GT正确性。误罚仅在GT事件内且有标注的观察帧判断；事件外或缺标注的证据不补成错误标签。full-event候选质量是另一个诊断，不能混作逐帧标注。','',
        '| 面板 / arm | steps | observed误罚条目/已知负条目 | observed误罚权重比例 | loss↓ / ↑ / = | mean实际step displacement |','|---|---:|---:|---:|---:|---:|']
    for ds,sp in PANELS:
        for a in ARMS:
            d=diag[ds][sp][a];ratio='undefined' if d['observed_negative_misfire_weight_fraction'] is None else f'{d["observed_negative_misfire_weight_fraction"]:.4f}'
            lines.append(f'| {ds}/{sp}/{a} | {d["steps"]} | {d["negative_GT_better_entries"]}/{d["known_negative_entries"]} | {ratio} | {d["frozen_step_loss_decreased"]}/{d["frozen_step_loss_increased"]}/{d["frozen_step_loss_equal"]} | {d["mean_step_parameter_displacement"]:.6g} |')
    lines+=['','这些是重复step/candidate/观察的诊断分母，不能当独立来源样本量。每步用同一冻结q评估before/after；下一步刷新target，不能把第一步loss与最后一步loss视为同一目标的优化曲线。目标上升保留，loss下降但GT下降另计，均不作为选择器或GT门控。EXECUTION_DIAGNOSTICS保留实际参数位移、eta×梯度范数、观察/未观察输出梯度与框移动、GT较好候选概率变化及未决odds的target误差/实际变化。','',
        'query residual256与norm1/norm3/norm4共1536 LN参数的四个block-only反事实，在其余块固定pre的条件下重放。块效应不是可加的Jacobian分解；框变化可能经过共享decoder/空间读出，不能按参数名称直接归因为局部/全局。SUMMARY和ROWS保留全部块效果。','',
        '## 下一非专家query：隔离的一次写迁移','',
        '全部288专家位置的下一到达存在且非专家。无写baseline把当前专家write的共同A prestate带到未来query；rank_native写后状态恰为旧A未来query prestate。每arm只带入当前一次写后的状态；每个probe结束后丢弃，未形成新的长期轨迹。主CI按write-source配对，另给target-source敏感性；write与target复用可能留下两种单向bootstrap都未完全表达的依赖。','',
        '| 面板 | arm−Rank native | ΔvIoU write-source CI | target-source CI |','|---|---|---:|---:|']
    for ds,sp in PANELS:
        for a in ARMS[1:]:lines.append(f'| {ds}/{sp} | {a} | {value("future",ds,sp,a+"_minus_rank_native_v")} | {value("future_target_source_sensitivity",ds,sp,a+"_minus_rank_native_v")} |')
    lines+=['','## reset-u：另一个完整在线删除对照','',
        '每query进入前只把256维query residual归零，三组LN继承；每condition/order/split回source状态。原本数据集Rank lr/T/K与Fast读出保持，当前输出先于空间write封存。这个1152到达流只回答query residual生命周期删除的效果，没有捆绑负证据loss。true输出与固定旧A interval空间对照分开。','',
        '| 面板 | true reset-u−原A ΔvIoU | fixed-A interval ΔvIoU | >5pp / >20pp true损害 |','|---|---:|---:|---:|']
    for ds,sp in PANELS:
        z=s['reset_u'][ds][sp]['corrupt']['negative_tails']['reset_u_minus_A']
        lines.append(f'| {ds}/{sp} | {value("reset_u",ds,sp,"reset_u_minus_A_v")} | {value("reset_u",ds,sp,"reset_u_fixedA_minus_A_v")} | {z["severe_harm_gt5pp"]}/{z["severe_harm_gt20pp"]} |')
    lines+=['','## clean、顺序、来源集中与反例','',
        'SUMMARY保留clean/all的CI和尾部，以及两个order各自CI与gross均值；reset另分expert/nonexpert。SOURCE_CONCENTRATION提供逐源值、最大正源份额、前两负源份额、leave-one-source-out范围。CASES在每面板每对照保留最大正例和负例；没有删源追分、事后改lambda、挑GT步数或另加幅度对照。','',
        '| 确认clean | global−Rank vIoU | local−Rank vIoU | future local−Rank vIoU | reset-u−A vIoU |','|---|---:|---:|---:|---:|']
    for ds in DATASETS:lines.append(f'| {ds} | {value("current",ds,"confirm","negative_global_minus_rank_native_v","clean")} | {value("current",ds,"confirm","negative_local_minus_rank_native_v","clean")} | {value("future",ds,"confirm","negative_local_minus_rank_native_v","clean")} | {value("reset_u",ds,"confirm","reset_u_minus_A_v","clean")} |')
    lines+=['','## 与旧N1和原论文的边界','',
        '旧N1在16个历史曝光Vid源上用GT事先筛Useful/Noisy正负信号，目标为方向pairwise softplus，且主正负臂更新次数与梯度量不同；其负结果不能直接否定这里GT-free有限负teacher/同帧loss。这里重开的是不同机制和same-prestate匹配问题，未抹去N1负结果。','',
        '[NLNL原论文](https://arxiv.org/pdf/1908.07387)的negative learning用互补类别−log(1−p_c)，并配有选择NL/PL阶段；互补类别仍可能碰到真实类。[U2PL原论文](https://arxiv.org/pdf/2203.03884)把不可靠像素用于类别相对的对比负样本队列，同时保留监督与可靠正伪标签，并非低置信度即错误。[GKD原论文](https://arxiv.org/pdf/2306.13649)在学生生成prefix上用完整teacher token分布蒸馏，不穿过采样反传；这里九个几何probe分布不是原生自回归tube policy，不能写成对GKD完整复现。','',
        '已存DTA合法i<j的factorized start/end logit家族里，forward KL joint teacher的梯度只依赖其start/end边缘；相同边缘但不同联合相关性会给相同梯度。这个投影限制不是所有joint/mixture机制无用的证明，本轮也没有运行或晋升DTA。','',
        '## 核验与资源','',
        f'全局seal先于GT评分；独立根审计{sum(root["checks"].values()):,}个数值/链/哈希项，公开审计{sum(public["checks"].values()):,}项。ROOT_AUDIT单独声明：未加载模型，因此不能独立复现完整参数Jacobian或重新执行block-only输出；可独立复算输出空间导数、SGD坐标、状态继承、目标、IoU证据、密集指标和误罚统计。','',
        'RESOURCES分别保留两集local/reset-u的suffix/backward/block/future重放、wall/VRAM；缓存backbone和专家的新调用为0。CPU评分/审计/报告时间独立，不把此前专家成本、开发时间或缓存生成含入本轮worker时间。','',
        'CONFIRMATION_POSITIVE_LOWER_BOUND只记录两集corrupt确认主CI是否均>0，不是生产晋升门。历史曝光、小专家独立源、有效步长变化、严重尾部以及单次迁移边界仍限制结论；CURRENT_METHOD未改变。','',
        '公开结果仅匿名标量、teacher概率/证据标量、聚合、CI、正负例和图；raw expert boxes、学生框轨迹、H/features、参数状态、caption、媒体、标注、权重不公开。','']
    p=PUB/'REPORT.md';assert not p.exists();p.write_text('\n'.join(lines))
    own=['scripts/score_tastvg_negative_evidence_v1.py','scripts/audit_tastvg_negative_evidence_v1.py','scripts/report_tastvg_negative_evidence_v1.py',
        'scripts/run_tastvg_negative_evidence_v1.py','scripts/tastvg_negative_evidence_common_v1.py','vg_tta/tastvg_negative_evidence_v1.py','protocols/tastvg_negative_evidence_v1.md',
        'scripts/continue_tastvg_negative_evidence_v1.py','scripts/prepare_tastvg_negative_evidence_v1.py',
        'scripts/finalize_tastvg_negative_evidence_cpu_v1.py','tests/test_tastvg_negative_evidence_v1.py']
    write(PUB/'CODE_BINDING.json',dict(files={f:sha(ROOT/f) for f in own},runtime_lock_sha256=cfg['runtime_lock_sha256']))
    write(PUB/'FIGURE_MANIFEST.json',dict(files={name+'.'+ext:sha(PUB/(name+'.'+ext)) for name in figs for ext in ('png','pdf')},
        summary_sha256=sha(PUB/'SUMMARY.json'),execution_diagnostics_sha256=sha(PUB/'EXECUTION_DIAGNOSTICS.json')))
    write(PUB/'REPORT_MANIFEST.json',dict(time=time.time(),report_CPU_wall_seconds=time.monotonic()-tick,
        files={n:sha(PUB/n) for n in ['REPORT.md','ROWS.csv','FUTURE_ROWS.csv','RESET_U_ROWS.csv','EXECUTION_DIAGNOSTICS.json','SOURCE_CONCENTRATION.json','CASES.json','DECISION.json']}))
    print('NEGATIVE_EVIDENCE_REPORT_AND_THREE_FIGURES_WRITTEN',flush=True)


if __name__=='__main__':
    assert len(sys.argv)==1 or sys.argv[1]=='report'
    report()
