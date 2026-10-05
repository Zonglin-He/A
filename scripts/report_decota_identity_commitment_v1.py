"""Generate a durable report and inspectable research figures from all rows."""
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_identity_common_v1 import *
from scripts.decota_public_result_io_v1 import read
from scripts.score_decota_optimizer_posterior_v1 import stats
import numpy as np

def fmt(m):return f"{100*m['mean']:+.4f} [{100*m['ci95'][0]:+.4f}, {100*m['ci95'][1]:+.4f}]"
def draw_error(ax,x,m,color,label=None):
    ax.errorbar(x,m['mean']*100,yerr=np.array([[m['mean']-m['ci95'][0]],[m['ci95'][1]-m['mean']]])*100,fmt='o',capsize=4,color=color,label=label)
def run():
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    matched=read(PUB/'matched/SUMMARY.json');choice=read(BASE/'SEARCH_SELECTION.json');online=read(PUB/'online/SUMMARY.json');rob=read(PUB/'online/SCHEDULE_ROBUSTNESS.json');duration=read(PUB/'duration/SUMMARY.json')
    text=['# DeCoTA hard latent-path commitment, real online trajectories and duration bias','',
        f"本轮研究接口选择为 **{choice['arm']}**，只依据预锁开发规则；确认结果不用于重选。", 
        '固定各集32开发＋16历史曝光确认来源、一query、双序、clean＋五种5%corruption。不是全部官方query，也不是fresh test。CURRENT保持原decota_refine_uniform_v1。',
        'S-next在同一个封存P1 prestate比较当前纠正；后面的十二条独立流从源状态重新演化LN。两者的统计含义不同。','',
        '主裁决：MAP降低了开发集的部分损害，但没有通过预锁零严重尾部规则，因此回退共享Top1；这不是MAP在所有面板都弱于Top1的结论。Top1的确认当前纠正与实际online结果为正，但开发Vid仍有负均值和严重损害，不能称普遍安全。',
        '100%online确认Before−Frozen在两集为正；但After−匹配episodic尚未双集成立，Vid全部预算净继承区间跨零，HC仅25/50%有明确正区间。当前纠正有效，不自动等于slow LN对最终输出具有双集稳定增量。',
        '同域官方checkpoint为TASTVG_VidSTG.pth与TASTVG_HCSTVG2.pth，文件及模型state哈希见CONFIGURATION。继承PLAN中的anchor_params、temporal_lr/steps等只是旧实验元数据，本轮实际空间Adam.03/10步、Native时间、零temporal更新，以本轮协议与运行锁为准。','',
        '## Matched current correction','',
        'Top1不是每帧无条件raw top1：先保留旧admission接受的观察帧，再取各帧最高DINO分数。Track-Marginal和MAP使用全部有效旧观察。旧四位置是在原Native I0内按物理时间均匀选取，已具有预测事件scope；本轮复用旧位置，不重采样或使用GT事件范围。先前口头称缓存没有event-scoped选帧不准确，已核对原observations接口修正。',
        'MAP E-step固定score＝DINO分数＋prestate IoU＋相邻物理时刻IoU，最多81完整路径；M-step使用tau1的mean-frame GIoU contrastive loss。三个方案同Adam.03、joint1792、10步、自身loss首个最小step；没有GT选step。',
        '旧Track-Marginal的实际目标是−log∑z P(z)exp(∑frame IoU)，而不是线性∑z P(z)IoU。posterior expected GT IoU是事后证据诊断，不是该critic的训练目标；其数值下降不能单独证明线性identity averaging是损害根因。',
        'MAP相对控制同时改变hard path与loss，Top1还保留旧admission；因此三臂不是对identity单因素的证明。完整笛卡尔contrastive归一化可分解为逐帧项，不是新训练的tracking模型；singleton无竞争时严格no-op。E/M目标不同，不声称EM似然单调性。','',
        '| Dataset/panel | Arm | ΔBefore vIoU pp [paired95%] | ΔTop1 pp | >20pp harm |',
        '|---|---|---:|---:|---:|']
    arms=['top1','marginal','map_contrastive'];colors=['#237d9b','#99938b','#c05b47'];labels=['Frame Top1','Track marginal','Track MAP']
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),sharey=True)
    for j,ds in enumerate(DATASETS):
        for i,sp in enumerate(['search','confirm']):
            ax=axes[i,j]
            for k,a in enumerate(arms):
                z=matched[ds][sp][a]['corruption'];m=z['metrics']['vs_before_v'];draw_error(ax,k,m,colors[k]);text.append(f"| {ds}/{sp} | {a} | {fmt(m)} | {fmt(z['metrics']['vs_top1_v'])} | {z['tails']['harm_gt20pp']} |")
            ax.axhline(0,color='#666',ls='--',lw=.8);ax.set_xticks(range(3),labels);ax.set_title(ds+' / '+sp);ax.set_ylabel('Current gain (vIoU pp)')
    fig.tight_layout();fig.savefig(PUB/'matched_gain.png',dpi=220);fig.savefig(PUB/'matched_gain.pdf');plt.close(fig)
    text+=['',f"开发MAP资格：`{choice['search_gate']}`；规则为两集均不降低Top1均值、配对lower95≥−0.5pp且无>20pp当前损害。失败即共享Top1回退，不按确认切换数据集专属方法。",'']
    obs=read(PUB/'matched/OBSERVATION_QUALITY.json');text+=['| Confirm corruption observations | Best-path IoU% | MAP-path IoU% | Posterior expected IoU% | No GT-scored observations |','|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        rr=[r for r in obs if r['dataset']==ds and r['split']=='confirm' and r['condition']!='clean'];good=[r for r in rr if r['GT_scored_observations']]
        z=stats(good,['best_observation_IoU','MAP_observation_IoU','posterior_expected_observation_IoU'])['metrics'];text.append(f"| {ds} | {z['best_observation_IoU']['mean']*100:.2f} | {z['MAP_observation_IoU']['mean']*100:.2f} | {z['posterior_expected_observation_IoU']['mean']*100:.2f} | {len(rr)-len(good)} |")
    text+=['','这只是具备GT计分观察帧子集的空间IoU；不是identity accuracy，更不是全tube vIoU。最佳path允许逐帧GT诊断，不把事件外没有标注的帧虚构为错误。',
        '', '### Same-source success and failure readback', '',
        'source35/exposure的固定MAP path在四观察上的GT IoU86.28%，恰为该支持最优；更新将当前vIoU45.06%→70.05%。但frame-freeze的MAP path在三有GT观察上的IoU只有5.48%，而同支持GT最优91.25%；loss4.1465→3.9139，vIoU31.14%→4.46%，同prestate Top1为53.55%。该坏例在更新前的路径推断已选错，不能全部归因optimizer或LN；hard commitment也不能自动识别正确referent。',
        'frame-freeze坏例的MAP与Top1第一步梯度cosine为−0.9965，几乎反向；两者实际Adam参数位移范数都约1.2699。该案区别首先在目标方向而非位移幅度，不能据此唯一分离hard identity和contrastive loss的作用。数值保留于CASE_ACTUATION_READBACK。',
        '这两个condition各自按同prestate比较，但corruption和各自P1历史都不同；它们说明具体成功/失败链，不单独证明corruption因果。观察/未观察GT变化与匿名path索引保留于CASE_MECHANISM_READBACK。',
        '## Actual independent LN streams','',
        'Native时间始终固定；query残差和Adam每次重置，LN proposal位移×1/16写回。按dataset/split/condition/order/stream重置源状态。',
        '100%与episodic各一条；25/50%各五个预锁hash schedules，25⊂50，每组roster跨condition/order固定。没有按GT强制纳入或排除source35。',
        '| Confirm corruption | Stream | After−Frozen pp [paired95%] | Before−Frozen pp | After−episodic pp | >20pp harm |','|---|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        for stream in ['episodic','online100']+[f'seed{s}_budget{r}' for r in [25,50] for s in range(5)]:
            z=online[ds]['confirm'][stream]['corruption'];m=z['metrics'];text.append(f"| {ds} | {stream} | {fmt(m['vs_frozen_v'])} | {fmt(m['before_vs_frozen_v'])} | {fmt(m['vs_episodic_v'])} | {z['tails']['harm_gt20pp']} |")
    text+=['','| Dataset/panel | Expert budget | 5-schedule mean ΔFrozen pp | Schedule range pp | Source CI after within-source schedule averaging |','|---|---:|---:|---:|---:|']
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
    for ax,ds in zip(axes,DATASETS):
        for sp in ['search','confirm']:
            for rate in [25,50]:
                z=rob[ds][sp][str(rate)];m=z['source_bootstrap_conditional_on_five_schedules']['metrics']['vs_frozen_v'];text.append(f"| {ds}/{sp} | {rate}% | {z['mean']*100:+.4f} | [{z['range'][0]*100:+.4f},{z['range'][1]*100:+.4f}] | {fmt(m)} |")
        for rate in [25,50]:
            z=rob[ds]['confirm'][str(rate)];m=z['source_bootstrap_conditional_on_five_schedules']['metrics']['vs_frozen_v'];draw_error(ax,rate,m,'#237d9b')
            ax.scatter([rate]*5,np.array(z['schedule_means'])*100,color='#237d9b',alpha=.35,s=16)
        draw_error(ax,100,online[ds]['confirm']['online100']['corruption']['metrics']['vs_frozen_v'],'#237d9b')
        ax.axhline(0,color='#777',lw=.8,ls='--');ax.set_xticks([25,50,100]);ax.set_title(ds+' / confirm');ax.set_xlabel('Expert budget (%)');ax.set_ylabel('Gain over Frozen (vIoU pp)')
    fig.tight_layout();fig.savefig(PUB/'online_schedule.png',dpi=220);fig.savefig(PUB/'online_schedule.pdf');plt.close(fig)
    text+=['','上述source bootstrap先在每个来源内平均五个schedule；CI条件于这五个预定schedule，五份重复输入不是五倍独立样本。预算全流、专家/非专家、clean、每个corruption与两顺序的完整分解保留于SUMMARY。','',
        '| Vid confirm/source35 | 25% expert schedules | 50% expert schedules |','|---|---|---|']
    roster=read(BASE/'ONLINE_LOCK.json')['schedules']['vidstg']['confirm'];text.append('| source35 | '+str([s for s in range(5) if 35 in roster[f'seed{s}_budget25']])+' | '+str([s for s in range(5) if 35 in roster[f'seed{s}_budget50']])+' |')
    persistence=read(PUB/'online/MATCHED_PERSISTENCE_SUMMARY.json')
    text+=['','### Matched-roster persistence','',
        '预算流相对100%episodic的差值包含专家可用性，不是纯继承。另以该roster专家位置的episodic预测、非专家位置Frozen构造严格匹配读出；全部来自已封存缓存，无新推理。',
        '| Confirm corruption | Budget | Online−matched episodic pp [source paired95%] |','|---|---:|---:|']
    for ds in DATASETS:
        for rate in [25,50]:text.append(f"| {ds} | {rate}% five-schedule mean | {fmt(persistence[ds]['confirm'][f'five_schedule_mean_{rate}']['metrics']['vs_matched_episodic_v'])} |")
        text.append(f"| {ds} | 100% | {fmt(persistence[ds]['confirm']['online100']['corruption']['metrics']['vs_matched_episodic_v'])} |")
    text+=['','## Positive and negative tails','', '| Setting | Dataset/source | Condition/order | ΔBefore pp | ΔFrozen pp |','|---|---|---|---:|---:|']
    mr=read(PUB/'matched/ROWS.json')
    for ds in DATASETS:
        rr=[r for r in mr if r['dataset']==ds and r['split']=='confirm' and r['condition']!='clean' and r['arm']=='map_contrastive']
        for r in sorted(rr,key=lambda r:r['vs_before_v'])[:3]+sorted(rr,key=lambda r:r['vs_before_v'],reverse=True)[:3]:text.append(f"| MAP matched | {ds}/{r['source_id']} | {r['condition']}/{r['order']} | {r['vs_before_v']*100:+.4f} | {r['vs_frozen_v']*100:+.4f} |")
    rr=read(PUB/'online/ROWS.json')
    for ds in DATASETS:
        q=[r for r in rr if r['dataset']==ds and r['split']=='confirm' and r['condition']!='clean' and r['stream']=='online100']
        for r in sorted(q,key=lambda r:r['vs_frozen_v'])[:3]+sorted(q,key=lambda r:r['vs_frozen_v'],reverse=True)[:3]:text.append(f"| online100 | {ds}/{r['source_id']} | {r['condition']}/{r['order']} | {r['vs_before_v']*100:+.4f} | {r['vs_frozen_v']*100:+.4f} |")
    text+=['','## CPU duration-bias diagnostic','',
        '全部有效raw UVTG proposals的log-duration中位数减Native log-duration。重复proposal保留，不用confidence。物理半开frame区间；不重复计两个顺序。相同pixels/frameids/query下分别用同域与既有对侧源checkpoint的Native；仅有缓存的输入计分。',
        '无标签统计先封存，GT随后只计分。270/576个source-condition格可用（Vid24、HC21来源），306格缺UVTG缓存；其中250个source/pixels不同，Vid20格旧corruption pixels重复仍按预锁condition权重保留。unique_inputs旧字段指cohort格，不是像素去重数。缺失不补推理，覆盖子集不能代表完整stream。',
        '| Cross checkpoint panel | Sources | Median rExpert [95%] | Median rGT [95%] | Source correlation [95%] | Expert-vs-GT logduration corr |','|---|---:|---:|---:|---:|---:|']
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for ax,ds in zip(axes,DATASETS):
        for sp in ['search','confirm']:
            z=duration[ds][sp]['cross']['corruption'];text.append(f"| {ds}/{sp} | {z['sources']} | {z['source_averaged_rE_median']:+.4f} {z['rE_median_ci95']} | {z['source_averaged_rGT_median']:+.4f} {z['rGT_median_ci95']} | {z['source_r_correlation']} {z['source_r_correlation_ci95']} | {z['source_expert_GT_log_duration_correlation']} |")
        z=duration[ds]['confirm']['cross']['corruption']
        for k,(fld,ci,color) in enumerate([('source_averaged_rE_median','rE_median_ci95','#237d9b'),('source_averaged_rGT_median','rGT_median_ci95','#c05b47')]):
            val=z[fld];lo,hi=z[ci];ax.errorbar(k,val,yerr=np.array([[val-lo],[hi-val]]),fmt='o',capsize=5,color=color)
        ax.axhline(0,color='#777',lw=.8,ls='--');ax.set_xticks([0,1],['UVTG median','GT diagnostic']);ax.set_ylabel('Log-duration ratio versus Native');ax.set_title(ds+' / cross confirm')
    fig.tight_layout();fig.savefig(PUB/'duration_bias.png',dpi=220);fig.savefig(PUB/'duration_bias.pdf');plt.close(fig)
    dec=read(PUB/'duration/DECISION.json');text+=['',f"跨域确认诊断资格：`{dec['cross_confirmation_diagnostic_positive']}`。没有启动scalar slow temporal state。",
        'rExpert与rGT共同减Native log-duration，能制造相关性；独立log-duration相关和来源内condition去均值相关均另列。HC确认点估计UVTG偏缩短、GT偏扩长，符号相反；Vid方向点估计一致但GT中位数区间跨零、相关区间宽。不能从少数覆盖来源推断全target domain应系统扩/缩。',
        '本次没有支持新增temporal slow state；结论限于该expert median统计与缓存面板，不是对所有时间校准的永久否定。',
        '## Verification, computation and scope','',
        '无GT smoke在各集前2旧clean输入复现Top1/Marginal全步loss、梯度、状态与tube逐值一致；MAP已有独立NumPy公式、有限差分梯度与single/empty/duplicate合同。1152新MAP全封存后计3456三臂匿名指标；控制来自已审计收据。',
        '独立根审计重新计算所有loss/E-step/Adam参数算术、选择step、状态生命周期、LN链及official dense一致性；没有额外重跑每个decoder Jacobian，报告该边界。最终13824逻辑在线预测包含episodic同输入顺序收据复用，真实独立online链不复用。',
        'source-macro/配对10000-source bootstrap；正负例与clean均保留，不按GT改名单、tau、学习率、目标或写入。观察GT IoU与tube任务指标分开，不把目标降低等同正确纠正。',
        '全部新完整backbone/expert/temporal参数更新为0；执行cached decoder/backward。模型checkpoint加载与I/O、等待和CPU审计分开，不把worker wall称为纯GPU-kernel时间。','']
    resource={'new_expert':0,'new_full_backbone':0,'temporal_updates':0,'matched_logical_arrivals':1152,'matched_rows':3456,'online_logical_arrivals':13824,'stages':{}}
    for stage,bname in [('matched','MATCHED_BARRIER.json'),('online','ONLINE_BARRIER.json')]:
        resource['stages'][stage]={}
        for ds in DATASETS:resource['stages'][stage][ds]={k:v for k,v in read(BASE/ds/bname).items() if k!='files'}
    resource['online_actual_cached_backward_calls']=sum(d['actual_cached_backward_calls'] for d in read(PUB/'online/DIAGNOSTICS.json'))
    text.append(f"Online实际cached backward：{resource['online_actual_cached_backward_calls']}；stage资源完整见RESOURCE_RECEIPT。")
    write(PUB/'RESOURCE_RECEIPT.json',resource)
    for name in ['matched_gain','online_schedule','duration_bias']:text+=['',f'![{name}](../results/decota_identity_commitment/2026-10-05/{name}.png)']
    path=ROOT/'docs/TA_DECOTA_IDENTITY_COMMITMENT_REVIEW.md';path.write_text('\n'.join(text)+'\n');write(PUB/'REPORT_BINDING.json',dict(path=str(path.relative_to(ROOT)),sha256=sha(path),time=time.time()))
    write(PUB/'CONFIGURATION.json',dict(version='decota_identity_commitment_v1',datasets=DATASETS,search_sources_each=32,confirm_sources_each=16,query_each_source=1,
        checkpoints={ds:{'file':read(BASE/ds/'PLAN.json')['configuration']['checkpoint'],'file_sha256':read(BASE/ds/'PLAN.json')['configuration']['checkpoint_sha256'],'model_state_sha256':read(BASE/ds/'PLAN.json')['checkpoint_state_sha256']} for ds in DATASETS},
        historical_exposure=True,conditions=read(BASE/'vidstg/PLAN.json')['conditions'],orders=2,parameters=1792,Adam_lr=.03,steps=10,tau=1,
        native_temporal=True,old_admission_preserved_top1=True,old_four_positions_preserved=True,old_four_scope='physical-time uniform within preexisting Native I0',selection=choice,
        production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),new_expert=0,new_backbone=0,whole_official_dataset=False))
if __name__=='__main__':run()
