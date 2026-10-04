"""Report all executed or explicitly unqualified stages, including negative tails."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_actuation_scope_common_v1 import *

def formatted(m):return f"{m['mean']*100:+.4f} [{m['ci95'][0]*100:+.4f},{m['ci95'][1]*100:+.4f}]"

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    text=['# DeCoTA track evidence, actuation scope and real online qualification','',
        '本轮接续已公开R1，保持各数据集32开发＋16历史曝光确认来源、一query、双序、clean＋五5%。不是全部官方query。',
        'R3/R4在封存的P1同到达前状态上比较当前纠正；R5和预算流独立演化LN，不能把两类结果合并称为持续收益。',
        '优化器、evidence、scope按预锁规则只在开发面板选，确认不重选。CURRENT与旧C1/P0/P1完整保留。','']
    for stage in ['R3','R3G','R4']:
        if not (PUB/stage/'SUMMARY.json').exists():continue
        ss=read(PUB/stage/'SUMMARY.json');arms=list(ss['vidstg']['confirm']);fig,axes=plt.subplots(1,2,figsize=(10,3.5))
        text+=[f'## {stage}: matched current-query interventions','', '| Confirm corruption | Arm | Δbefore vIoU pp [paired95%] | >20pp harm |','|---|---|---:|---:|']
        for ax,ds in zip(axes,DATASETS):
            for i,a in enumerate(arms):
                z=ss[ds]['confirm'][a]['corruption'];m=z['metrics']['vs_before_v'];ax.errorbar(i,m['mean']*100,yerr=np.array([[m['mean']-m['ci95'][0]],[m['ci95'][1]-m['mean']]])*100,fmt='o',capsize=4,color='#286a83')
                text.append(f"| {ds} | {a} | {formatted(m)} | {z['tails']['harm_gt20pp']} |")
            ax.axhline(0,color='#777',lw=.8,ls='--');ax.set_xticks(range(len(arms)),arms);ax.set_ylabel('Current correction (vIoU pp)');ax.set_title(ds)
        fig.tight_layout();fig.savefig(PUB/(stage+'.png'),dpi=220);fig.savefig(PUB/(stage+'.pdf'));plt.close(fig)
        text+=['',f"开发决定：`{read(PUB/stage/'DECISION.json').get('winner',read(PUB/stage/'DECISION.json').get('scope'))}`。非GT最优step；当前输出与最终写入proposal分开保存。",'']
    ss=read(PUB/'online'/'SUMMARY.json');fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    text+=['## Actual independent LN streams','', '| Confirm corruption | Stream | After−Frozen pp [paired95%] | Before−Frozen pp | After−episodic pp | >20pp harm |','|---|---|---:|---:|---:|---:|']
    for ax,ds in zip(axes,DATASETS):
        for stream in ['episodic','budget_0.25','budget_0.5','budget_1.0']:
            z=ss[ds]['confirm'][stream]['corruption'];text.append(f"| {ds} | {stream} | {formatted(z['metrics']['vs_frozen_v'])} | {formatted(z['metrics']['before_vs_frozen_v'])} | {formatted(z['metrics']['vs_episodic_v'])} | {z['tails']['harm_gt20pp']} |")
        for fld,color,label in [('vs_frozen_v','#286a83','After'),('before_vs_frozen_v','#b47456','Before')]:
            mm=[ss[ds]['confirm'][f'budget_{rate}']['corruption']['metrics'][fld] for rate in ['0.25','0.5','1.0']]
            ax.errorbar([25,50,100],[m['mean']*100 for m in mm],yerr=np.array([[m['mean']-m['ci95'][0] for m in mm],[m['ci95'][1]-m['mean'] for m in mm]])*100,fmt='o-',color=color,label=label,capsize=3)
        ax.axhline(0,color='#777',lw=.8,ls='--');ax.set_xlabel('Query-level expert budget (%)');ax.set_ylabel('vIoU gain over Frozen (pp)');ax.set_title(ds);ax.legend()
    fig.tight_layout();fig.savefig(PUB/'online_budget.png',dpi=220);fig.savefig(PUB/'online_budget.pdf');plt.close(fig)
    config=read(PUB/'online'/'CONFIGURATION.json')
    text+=['',f"最终空间scope：`{read(BASE/'R4_SELECTION.json')['scope']}`；时间读出：`{config['temporal']}`。", 
        '先前agent增加的同域资源门槛保留于R6_SCOPE_CORRECTION.json，但它不属于用户要求的停止条件。R6已按原全路线授权做新的跨域测量；冻结方法未按确认结果重选。','']
    cross=read(PUB/'cross_domain'/'SUMMARY.json');fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    text+=['## R6: newly executed cross-domain qualification','',
        'Vid目标采用官方HC-STVG-v2源checkpoint；HC2目标采用官方VidSTG源checkpoint。重新捕获576个匹配输入及对应四帧DINO观察，不复用同域H；两组episodic/online100真实链共2304读出，全封存后GT评分。','',
        '| Confirm corruption | Stream | After−Frozen pp [paired95%] | Before−Frozen pp | After−episodic pp | >20pp harm |',
        '|---|---|---:|---:|---:|---:|']
    for ax,ds in zip(axes,DATASETS):
        for i,stream in enumerate(['episodic','online_100']):
            z=cross[ds]['confirm'][stream]['corruption'];text.append(f"| {ds} | {stream} | {formatted(z['metrics']['vs_frozen_v'])} | {formatted(z['metrics']['before_vs_frozen_v'])} | {formatted(z['metrics']['vs_episodic_v'])} | {z['tails']['harm_gt20pp']} |")
            m=z['metrics']['vs_frozen_v'];ax.errorbar(i,m['mean']*100,yerr=np.array([[m['mean']-m['ci95'][0]],[m['ci95'][1]-m['mean']]])*100,fmt='o',capsize=4,color='#286a83')
        ax.axhline(0,color='#777',lw=.8,ls='--');ax.set_xticks([0,1],['Episodic','Online 100%']);ax.set_title(ds);ax.set_ylabel('Cross-domain gain over Frozen (vIoU pp)')
    fig.tight_layout();fig.savefig(PUB/'cross_domain.png',dpi=220);fig.savefig(PUB/'cross_domain.pdf');plt.close(fig)
    text+=['',f"双数据集跨域确认lower95都正：`{read(PUB/'cross_domain'/'DECISION.json')['positive_lower95_both_confirmation']}`；此处是实际资格测量，不是方法晋升。",'']
    text+=['| Cross-domain panel (online100) | After−Frozen pp [paired95%] | Before−Frozen pp | After−episodic pp | >20pp harm |',
        '|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        for split,group in [('search','corruption'),('confirm','clean')]:
            z=cross[ds][split]['online_100'][group]
            text.append(f"| {ds}/{split}/{group} | {formatted(z['metrics']['vs_frozen_v'])} | {formatted(z['metrics']['before_vs_frozen_v'])} | {formatted(z['metrics']['vs_episodic_v'])} | {z['tails']['harm_gt20pp']} |")
    text+=['','跨域Vid开发corruption总收益区间跨零且有17次>20pp损害，HC开发有2次；确认两方向总收益为正仍不建立普遍安全。两确认方向online−episodic的区间都跨零；Before−Frozen为正不等于After已可靠超过episodic。','']
    text+=['| Cross-domain confirm tail | Dataset/source | Condition/order | After−Frozen pp | Before−Frozen pp | Current spatial pp |',
        '|---|---|---|---:|---:|---:|']
    for ds in DATASETS:
        rr=[r for r in read(PUB/'cross_domain'/'ROWS.json') if r['dataset']==ds and r['stream']=='online_100' and r['split']=='confirm' and r['condition']!='clean']
        for label,seq in [('positive',sorted(rr,key=lambda r:r['vs_frozen_v'],reverse=True)[:3]),('negative',sorted(rr,key=lambda r:r['vs_frozen_v'])[:3])]:
            for r in seq:text.append(f"| {label} | {ds}/{r['source_id']} | {r['condition']}/{r['order']} | {r['vs_frozen_v']*100:+.4f} | {r['before_vs_frozen_v']*100:+.4f} | {r['current_spatial_gain']*100:+.4f} |")
    text+=['','跨域的逐到达observed/unobserved GT损害、own-energy改善但GT受损，以及correctness阈值变化保留在DIAGNOSTICS/SUMMARY；正负例都不作为重选规则。','']
    text+=['## Development selection and same-state contrasts','',
        'R3必须同时保住两个开发面板的原anchor和FrameSum均值，且不增加>20pp受损到达；R4须同时保住joint均值及严重负尾。资格不等于确认已证明普遍收益。',
        '| Stage/panel | Arm | Δbefore vIoU pp [paired95%] | >20pp harm |','|---|---|---:|---:|']
    for stage in ['R3','R3G','R4']:
        if not (PUB/stage/'SUMMARY.json').exists():continue
        su=read(PUB/stage/'SUMMARY.json')
        for ds in DATASETS:
            for arm,z in su[ds]['search'].items():
                z=z['corruption'];text.append(f"| {stage}/{ds} search | {arm} | {formatted(z['metrics']['vs_before_v'])} | {z['tails']['harm_gt20pp']} |")
    contrasts=read(PUB/'MECHANISM_CONTRASTS.json')
    text+=['','| Confirm corruption | Contrast | ΔvIoU pp [paired95%] |','|---|---|---:|']
    for stage,dd in contrasts.items():
        for ds,sp in dd.items():
            for name,z in sp['confirm'].items():text.append(f"| {stage}/{ds} | {name} | {formatted(z['metrics']['delta_v'])} |")
    text+=['','Track−FrameSum对比轨迹权重与独立帧同尺度控制，TrackAuthority−Track改变实际位移权限；两者不单独证明身份被识别。u-only对照也改变当前梯度参数维数和额外慢步计算，不能只归因reset寿命。',
        '## Source35 matched current correction','',
        '该到达从同一封存P1状态出发。Native时间固定，GT只计分；不把该单例推广成全部来源规律。',
        '| Stage | Arm | vIoU% | Δbefore pp |','|---|---|---:|---:|']
    for stage,rr in read(PUB/'SOURCE35_MATCHED_CASES.json').items():
        for r in rr:text.append(f"| {stage} | {r['arm']} | {r['v']*100:.4f} | {r['vs_before_v']*100:+.4f} |")
    text+=['','## Evidence support and update execution','',
        '| Stage/dataset | Arm | Observed GT IoU Δpp | Unobserved Δpp | Proxy improved / GT harmed count |','|---|---|---:|---:|---:|']
    for stage,dd in read(PUB/'MECHANISM_DIAGNOSTIC_SUMMARY.json').items():
        for ds,aa in dd.items():
            for arm,z in aa.items():
                obs=z['observed_GT_delta']['metrics'];unobs=z['unobserved_GT_delta']['metrics']
                a=formatted(obs['value']) if obs else 'no scored observed frame';b=formatted(unobs['value']) if unobs else 'no scored unobserved frame'
                text.append(f"| {stage}/{ds} | {arm} | {a} | {b} | {z['used_proxy_improved_GT_harmed']} |")
    text+=['','Observed为四个实际观察位置落在GT计分帧上的IoU变化，unobserved为其余GT计分帧；不是新的监督或正式GT-selected策略。',
        '固定track posterior的GT诊断只使用已有带GT观察帧：最佳path、MAP path与posterior均值分别保留。事件外/无GT观察不构造虚假的correctness标签；路径重叠仍不保证identity。','']
    text+=['| Confirm corruption observed support | GT-best path IoU% | MAP path IoU% | Posterior expected IoU% | No scored observation arrivals |',
        '|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        z=read(PUB/'MECHANISM_DIAGNOSTIC_SUMMARY.json')['R3'][ds]['track']
        m=lambda k:z[k]['metrics']['value']['mean']*100
        text.append(f"| {ds} | {m('path_best_observed_GT_IoU'):.2f} | {m('path_MAP_observed_GT_IoU'):.2f} | {m('path_posterior_mean_observed_GT_IoU'):.2f} | {z['no_observed_GT_support']} |")
    text+=['','这些是具备GT计分观察位置子集上的source-macro空间IoU，不是全tube vIoU，也不是teacher正确概率。无计分观察到达仍保留在正式全流指标中。','']
    rr=read(PUB/'online'/'ROWS.json');bad=sorted([r for r in rr if r['split']=='confirm' and r['condition']!='clean' and r['stream']=='budget_1.0'],key=lambda r:r['vs_frozen_v'])[:8]
    good=sorted([r for r in rr if r['split']=='confirm' and r['condition']!='clean' and r['stream']=='budget_1.0'],key=lambda r:r['vs_frozen_v'],reverse=True)[:8]
    text+=['## Positive and negative cases','', '| Tail | Dataset/source | Condition/order | After−Frozen pp | Before−Frozen pp | Current spatial pp |','|---|---|---|---:|---:|---:|']
    for label,seq in [('positive',good),('negative',bad)]:
        for r in seq:text.append(f"| {label} | {r['dataset']}/{r['source_id']} | {r['condition']}/{r['order']} | {r['vs_frozen_v']*100:+.4f} | {r['before_vs_frozen_v']*100:+.4f} | {r['current_spatial_gain']*100:+.4f} |")
    dd=read(PUB/'online'/'DIAGNOSTICS.json');text+=['','## Actual resources and limits','',
        f"R5/预算合计{len(rr)}个逻辑到达；实际新cached backward {sum(d['actual_cached_backward_calls'] for d in dd)}，复用完成的旧流到达{sum(d['reused_saved_stream'] for d in dd)}。新DINO/完整backbone均0；每个专家query四观察。", 
        '25/50/100位置按同一来源hash形成嵌套集合，跨顺序和corruption一致；全流、专家、非专家、clean、各corruption、顺序均单列source-macro/10000配对bootstrap。',
        '具体地，Vid确认的25%/50%固定专家来源集合均不包含source35，100%才对它执行当前fit。因此低预算未出现该严重尾部不能证明同一异常更新被稳定修好；未按GT排除source35，也未按确认重新选择预算。',
        'Track使用缓存目标兼容度＋native IoU＋连续IoU，所有系数1、最多81条exact paths。重叠连续性不保证同一identity；posterior浓度不是正确概率。',
        'u-only的当前输出只吃256维残差；随后一新1536维LN步只服务未来1/16写入。Small-LN的当前LN位移额外乘预锁浓度。没有按GT挑step、阈值或样本。',
        'Wall time包括模型加载、I/O和cached decoder/backward，不是纯GPU-kernel时间。完整raw预测/媒体/标注/权重保持私有，公开匿名逐到达指标和数学审计。','']
    text+=['| Stage/dataset | Logical arrivals | Actual cached backwards | Worker wall seconds |','|---|---:|---:|---:|']
    for stage,dd in read(PUB/'RESOURCE_RECEIPT.json')['stages'].items():
        for ds,z in dd.items():text.append(f"| {stage}/{ds} | {z['arrivals']} | {z.get('backward_calls',0)} | {z['seconds']:.2f} |")
    cr=read(PUB/'cross_domain'/'RESOURCE_RECEIPT.json')
    text+=['',f"跨域另有{cr['capture_inputs']}个新two-offset capture，实际DINO {cr['actual_DINO']}次；{cr['readouts']}读出。smoke与正式捕获共享已匹配输入，观察预算没有重复计算。", 
        f"另有smoke完整模型回插{cr['smoke_extra_full_reinsertion_two_offset']}个two-offset forward；合计{cr['actual_full_backbone_single_offset_calls']}次单offset完整backbone调用。raw/normalized匹配smoke另有{cr['smoke_cached_backward_calls']}次cached backward，未冒充正式读出成本。",'']
    for ds,z in cr['datasets'].items():
        text.append(f"跨域{ds}捕获wall {z['seconds']:.2f}s、在线/episodic worker wall {z['prediction']['seconds']:.2f}s，实际cached backwards {z['prediction']['backward_calls']}；smoke总wall另列，不是纯kernel延迟。")
    text+=['','旧缓存/轨迹复用与实际新计算由独立收据区分。保留无GT smoke访问guard失败原件，以及预测前R4最终配置指向修订；都未按GT改变方法。','',
        '## Route decision and remaining limits','',
        'R1完整factorial与CPU posterior已公开独立审计。R2依原开发规则保留All Adam；SGD与Adam均能造成同一严重损害，不能把Adam写成唯一故障。',
        'Full/Extent posterior没有建立两数据集共同收益，按附件停止条件不执行514-head T1或occupancy选帧。R3 track未保住两开发面板，因此不接入track、不运行仅track有效才允许的GIoU变体。',
        'R4 u-only减少Vid当前严重尾部，但未保住两开发均值；small-LN也未达规则，因此实际在线配置保持joint。不存在成功的新authority/track/scope组合，不包装成新方法已成立。',
        '同域预算和新的跨域均报告Frozen、episodic、实际在线Before/After；小面板历史曝光、单一hash预算和有限两顺序不等于全官方query或未见测试。',
        '确认与诊断GT只用于事后指标和资格报告，未重新选择optimizer/参数/预算/状态。部署登记保持原方法。','']
    route=read(BASE/'ROUTE_DECISION.json') if (BASE/'ROUTE_DECISION.json').exists() else dict(status='conditional_continuation_pending',full_route_complete=False)
    if (PUB/'ROUTE_DECISION.json').exists():
        assert read(PUB/'ROUTE_DECISION.json')==route,'Published route changed during report generation'
    else:
        write(PUB/'ROUTE_DECISION.json',route)
    for fig in ['R3','R3G','R4','online_budget','cross_domain']:
        if (PUB/(fig+'.png')).exists():text+=['',f'![{fig}](../results/decota_actuation_scope/2026-10-05/{fig}.png)']
    f=ROOT/'docs/TA_DECOTA_ACTUATION_SCOPE_REVIEW.md';f.write_text('\n'.join(text)+'\n');write(PUB/'REPORT_BINDING.json',dict(report=str(f.relative_to(ROOT)),sha256=sha(f),time=time.time()))

if __name__=='__main__':run()
