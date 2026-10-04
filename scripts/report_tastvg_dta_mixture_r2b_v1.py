"""Audited R2b result, source concentration, positive/negative cases and figures."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,csv,collections,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write as locked_write,sha
PUB=ROOT/'results/tastvg_dta_mixture_r2b/2026-10-04';BASE=ROOT/'artifacts/tastvg_dta_mixture_r2b_v1'
R2PUB=ROOT/'results/tastvg_dta_expert_r2/2026-10-04'
PANELS=[('vidstg','search'),('hc2','search'),('vidstg','confirm'),('hc2','confirm')]

def emit(p,z):
    if not p.exists():return locked_write(p,z)
    old=read(p)
    if isinstance(old,dict) and isinstance(z,dict):assert {k:v for k,v in old.items() if k!='time'}=={k:v for k,v in z.items() if k!='time'},p
    else:assert old==z,p

def source_average(q,extract):
    ss=collections.defaultdict(list)
    for r in q:ss[r['source_id']].append(extract(r))
    return np.asarray([np.mean(v,axis=0) for v in ss.values()]).mean(axis=0)

def main():
    s=read(PUB/'SUMMARY.json');rows=read(PUB/'ROWS.json');cfg=read(PUB/'CONFIG.json');res=read(PUB/'RESOURCES.json')
    root=read(PUB/'ROOT_AUDIT.json');public=read(PUB/'PUBLIC_AUDIT.json');assert root['status']==public['status']=='pass'
    oldrows={r['cell_key']:r for r in read(R2PUB/'ROWS.json')};traces={r['cell_key']:r for r in read(PUB/'MIX_TRACES_SCORED.json')};td=read(PUB/'TEACHER_DIAGNOSTICS.json')
    concentration={};aggreg={};changes={};cases=[];interpret=[]
    for ds,sp in PANELS:
        q=[r for r in rows if r['dataset']==ds and r['split']==sp and r['expert_scheduled'] and r['condition']!='clean'];pa=s[ds][sp]['expert_corrupt']
        concentration.setdefault(ds,{})[sp]={};aggreg.setdefault(ds,{})[sp]={};changes.setdefault(ds,{})[sp]={}
        for b in ['N','A8','EDeploy']:
            v=pa['metrics']['EMix_minus_'+b+'_v']['source_values'];ids=list(v);x=np.asarray(list(v.values()));pos=np.maximum(x,0);neg=np.maximum(-x,0)
            pi=np.argsort(-pos);ni=np.argsort(-neg);loo=(x.sum()-x)/(len(x)-1)
            concentration[ds][sp][b]=dict(sources=len(x),positive_sources=int((x>1e-12).sum()),negative_sources=int((x<-1e-12).sum()),
                largest_positive_source=ids[int(pi[0])],largest_positive_share=float(pos[pi[0]]/pos.sum()) if pos.sum() else None,
                top_two_positive_share=float(pos[pi[:2]].sum()/pos.sum()) if pos.sum() else None,
                largest_negative_source=ids[int(ni[0])],largest_negative_share=float(neg[ni[0]]/neg.sum()) if neg.sum() else None,
                top_two_negative_share=float(neg[ni[:2]].sum()/neg.sum()) if neg.sum() else None,
                leave_one_source_out_net_range_pp=[float(100*loo.min()),float(100*loo.max())])
            for label,r in [('largest_gain',max(q,key=lambda r:r['EMix_minus_'+b+'_v'])),('largest_harm',min(q,key=lambda r:r['EMix_minus_'+b+'_v']))]:
                cases.append(dict(dataset=ds,split=sp,baseline=b,selection=label,cell=r,trace=traces[r['cell_key']]['trace']))
        tq=[r for r in td if r['dataset']==ds and r['split']==sp and r['condition']!='clean']
        am=source_average(tq,lambda r:[r['raw_mean_tIoU'],r['raw_max_tIoU'],r['raw_fraction_tIoU_ge_05'],
            np.mean([v['joint_entropy'] for v in r['offsets']]),np.mean([v['expected_joint_tIoU'] for v in r['offsets']])])
        aggreg[ds][sp]=dict(zip(['raw_mean_tIoU','raw_max_tIoU','raw_fraction_tIoU_ge_05','joint_entropy','expected_offset_joint_tIoU'],am.tolist()))
        changes[ds][sp]=dict(cells=len(q),native_interval_changed=sum(r['EMix_interval_changed'] for r in q),
            loss_decreased=sum(r['EMix_loss_after']<r['EMix_loss_before'] for r in q),loss_down_native_v_down=pa['loss_down_task_down'],
            interval_unchanged_but_loss_decreased=sum(not r['EMix_interval_changed'] and r['EMix_loss_after']<r['EMix_loss_before'] for r in q))
        mixN=pa['metrics']['EMix_minus_N_v'];mixD=pa['metrics']['EMix_minus_EDeploy_v']
        interpret.append(dict(dataset=ds,split=sp,mix_over_native_mean=mixN['mean'],mix_over_native_ci95=mixN['ci95'],
            mix_over_deploy_mean=mixD['mean'],mix_over_deploy_ci95=mixD['ci95'],native_gain_established=mixN['ci95'][0]>0,
            deploy_improvement_established=mixD['ci95'][0]>0))
    emit(PUB/'SOURCE_CONCENTRATION.json',concentration);emit(PUB/'TEACHER_AGGREGATION.json',aggreg);emit(PUB/'EXECUTION_DIAGNOSTICS.json',changes);emit(PUB/'CASES.json',cases)
    decision=dict(EMix='NO_GO for this raw equal-weight teacher with frozen R1/R2 dynamics',panels=interpret,
        confirmation_native_gain_established=False,confirmation_deploy_improvement_established=False,
        uncertainty_preservation_solves_bottleneck_proven=False,bad_raw_mass_is_unique_failure_cause_proven=False,
        measured='All four Native vIoU means negative; Vid confirmation CI below zero. Only HC development demonstrates paired gain over failed EDeploy.',
        competing_explanations=['equal raw proposal mass is not localization-quality mass','joint multimodal teacher projected into additive start/end student','diffuse target changes gradient and MAP execution'],
        next_question='If reopened, distinguish teacher support purification from diffuse joint-target projection; no extra job launched.',
        target_retuning=False,source_retuning=False,production_promoted=False,A8_changed=False,R3_started=False,purification_started=False,weighting_started=False,PoE_started=False,time=time.time())
    emit(PUB/'DECISION.json',decision)
    cols=['cell_key','dataset','split','source_id','condition','order','arrival','expert_scheduled','EMix_GT_supervised']
    cols += [a+'_'+m for a in ['N','A8','R1','EDeploy','EOracle','EMix','EMix_map'] for m in ['v','t']]
    cols += ['EMix_minus_'+b+'_'+m for b in ['N','A8','R1','EDeploy','EOracle'] for m in ['v','t']]
    with (PUB/'ROWS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r[k] for k in cols} for r in rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42})
    plotpanels=[('vidstg','search'),('vidstg','confirm'),('hc2','search'),('hc2','confirm')];figs=[]
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
    for ax,b,title,color in zip(axes,['N','EDeploy'],['Primary: E-Mix vs Native','Matched teacher: E-Mix vs E-Deploy'],['#416C91','#9B6B40']):
        for j,(ds,sp) in enumerate(plotpanels):
            z=s[ds][sp]['expert_corrupt']['metrics']['EMix_minus_'+b+'_v'];mu=100*z['mean'];lo,hi=np.asarray(z['ci95'])*100
            ax.errorbar(mu,j,xerr=[[mu-lo],[hi-mu]],fmt='o',color=color,capsize=3,ms=5)
        ax.set_yticks(range(4),['Vid development','Vid confirmation','HC development','HC confirmation']);ax.invert_yaxis();ax.axvline(0,color='.55',lw=.8)
        ax.set_title(title);ax.set_xlabel('Paired source-macro vIoU gain (pp)');ax.grid(axis='x',alpha=.15)
    def savefig(name):
        fig.savefig(PUB/f'{name}.png',dpi=220,bbox_inches='tight');fig.savefig(PUB/f'{name}.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
        plt.close(fig);figs.append(name)
    savefig('r2b_equal_mixture_primary')
    colors=['#89949D','#BC7357','#456F94','#78A09E','#8B93B1'];labels=['Native','Deploy','Equal mix','Support oracle','GT R1']
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.5),layout='constrained')
    for ax,ds,name in zip(axes,['vidstg','hc2'],['Vid confirmation','HC confirmation']):
        q=s[ds]['confirm']['expert_corrupt']['metrics'];vals=[100*q[a+'_v']['mean'] for a in ['N','EDeploy','EMix','EOracle','R1']]
        ax.bar(range(5),vals,color=colors,width=.67)
        for j,v in enumerate(vals):ax.text(j,v+.5,f'{v:.2f}',ha='center',fontsize=9)
        ax.set_xticks(range(5),labels,rotation=15,ha='right',fontsize=9);ax.set_ylim(0,max(vals)*1.20);ax.set_title(name)
        ax.set_ylabel('Expert-corrupt vIoU (%)');ax.grid(axis='y',alpha=.15)
    savefig('r2b_confirmation_levels')
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.5),layout='constrained')
    for ax,ds,name in zip(axes,['vidstg','hc2'],['Vid confirmation','HC confirmation']):
        q=s[ds]['confirm']['expert_corrupt']['metrics'];sv=q['EMix_minus_N_v']['source_values'];ids=list(sv);x=np.arange(len(ids))
        deploy=[100*np.mean([oldrows[r['cell_key']]['EDeploy_v']-r['N_v'] for r in rows if r['dataset']==ds and r['split']=='confirm' and r['expert_scheduled'] and r['condition']!='clean' and str(r['source_id'])==i]) for i in ids]
        ax.bar(x-.17,deploy,width=.32,color=colors[1],label='E-Deploy vs Native');ax.bar(x+.17,[100*sv[i] for i in ids],width=.32,color=colors[2],label='E-Mix vs Native')
        ax.axhline(0,color='.5',lw=.8);ax.set_xticks(x,ids);ax.set_title(name);ax.set_xlabel('Anonymous source ID');ax.set_ylabel('Source mean vIoU gain (pp)');ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=8)
    savefig('r2b_confirmation_sources')
    def m(ds,sp,k,panel='expert_corrupt'):
        z=s[ds][sp][panel]['metrics'][k];return f'{100*z["mean"]:+.4f} [{100*z["ci95"][0]:+.4f}, {100*z["ci95"][1]:+.4f}]'
    text=['# R2b: Equal-Weight Multi-Hypothesis Temporal Teacher','',
        '**E-Mix 已实际完成，裁决 NO-GO：全部 raw proposals 等权混合没有建立相对 Native 的正向参数适应收益。** 四个专家 corruption 面板的 vIoU 均值均负；Vid 确认配对95%区间在零以下。HC 开发相对失败 E-Deploy 有明确恢复，两个确认相对 E-Deploy 的区间均跨零，不能把“比失败 baseline 少坏一点”写成有效部署 TTA。原 A、生产 CURRENT 及旧结果保持；未启动 weighting、PoE、purification 或 R3。','',
        '## 这次只改变 teacher 聚合','',
        '同一 R2 raw UniversalVTG 支持，所有有效 proposal 的端点、重复和顺序原样保留，每条权重 1/M。没有 confidence weighting、top-K、NMS/dedup、gate、PoE、新视图、sigma 调整。新增仅 E-Mix；E-Deploy/E-Oracle/R1/Native/A8 均读取上轮已公开并核验的封存结果，不重新拟合、选参数或挑样本。','',
        '保持原 TA-STVG 同域 EMA、原两 offset 采样与整数物理端点读出、514 名义 temporal head 末层参数（两个 bias 在归一化 span 分布下不可辨识）、K3、SGD lr=.01、sigma 一原 offset cell、beta1、KL(q||pθ)+KL(p0||pθ)。每query reset 后适应3步、当前query输出后丢弃；空间 A 完整框/参数轨迹、1792空间参数、Vid K1/HC K8、Uniform专家位置及25%到达率全部固定。研究 A 不等于生产 CURRENT。','',
        '等权的精确定义：对每个 offset 和每条 proposal，先用 R1/R2 相同 Gaussian 在原严格 i<j 合法网格独立归一化，再算 q_mix=(1/M)Σq_m；FP64 logsumexp，保留联合分布，**没有把混合的 start/end 边缘再相乘**。先混合未截断密度再整体归一化会按合法网格质量重加权分量，不是本轮等权实验。M=1 测试逐参数、逐梯度复现 R1。','',
        'Native student 仍是 start/end additive logits 加严格 i<j 归一化，两个 offset 损失平均，解码沿用 FP32 logsoftmax 首argmax后 envelope。Mixed teacher MAP 单独用联合 q 的 FP64 argmax 再 envelope，仅作离线读出诊断，不替换 student。保持完整 joint teacher 不等于原 student 可表达任意多峰关联，结果不唯一归因 uncertainty 保留或原proposal错质量。','',
        f'每集32开发+16确认、一query/源、两序、clean+五种5%瞬态corruption；共1152到达，288新query适应/864 backward，864非专家保持缓存 A。专家独立来源开发 Vid16/HC14、确认 Vid8/HC7，均历史曝光，不能称 fresh test。复用{cfg["unique_expert_cache_files"]}个独立expert缓存、每cell {cfg["proposal_count_range"][0]}–{cfg["proposal_count_range"][1]}条原有效proposal；没有新GPU/backbone/专家调用。','',
        'prepare 从已核验的R2 export manifest绑定旧指标hash而不读取数值；E-Mix独立worker带GT/scored-result读入guard，288预测全部封存后另一个进程才读取GT dense评分。全目标配置/teacher/预测先seal；GT仅用于离线指标与teacher质量解释，不参与mixture、步骤、参数、选择或筛样本。','',
        '## 主结果：E-Mix − zero-update Native','',
        '专家 corruption，单位pp；同来源配对、等来源宏平均、10000 source-bootstrap，固定seed20261004。Native为相同A空间输入的原head zero-update readout；不是原始Frozen空间模型。','',
        '| 面板 | 独立源 / cells | ΔtIoU [95% CI] | ΔvIoU [95% CI] | >5pp v损害 |','|---|---:|---|---|---:|']
    for ds,sp in PANELS:
        q=s[ds][sp]['expert_corrupt'];text.append(f'| {ds}/{sp} | {q["sources"]} / {q["cells"]} | {m(ds,sp,"EMix_minus_N_t")} | {m(ds,sp,"EMix_minus_N_v")} | {q["negative_tails"]["N"]["severe_harm"]} |')
    text +=['','Vid确认没有任何corruption专家cell相对Native正增益，20/40条降低vIoU，9条超过5pp；负向source见图及集中性记录。HC确认14/40条有正增益，但17条负、5条严重负尾，净来源宏平均仍负且区间跨零；不能把局部好例抹掉，也不能称两个数据集均统计确定地下降。','',
        '## 匹配 teacher 对照及 A8 参照','',
        '| 面板 | E-Mix − E-Deploy Δv | E-Mix − A8 Δv | E-Mix − E-Oracle Δv |','|---|---|---|---|']
    for ds,sp in PANELS:text.append(f'| {ds}/{sp} | {m(ds,sp,"EMix_minus_EDeploy_v")} | {m(ds,sp,"EMix_minus_A8_v")} | {m(ds,sp,"EMix_minus_EOracle_v")} |')
    text +=['','只有HC开发 E-Mix−E-Deploy 的配对v区间明确高于零；Vid开发均值反而低于Deploy、区间跨零。两个确认面板相对Deploy有正均值但CI跨零；都未相对Native建立正向。因此不能说本轮确认了“hard top1毁掉uncertainty，而mixture解决了它”。这一版等权目标仅在部分面板/坏例缓解top1错误，没有通过部署收益检验。','',
        '| 确认专家corrupt | Native vIoU % | E-Deploy % | E-Mix % | E-Oracle % | R1 GT % |','|---|---:|---:|---:|---:|---:|']
    for ds in ['vidstg','hc2']:
        q=s[ds]['confirm']['expert_corrupt']['metrics'];text.append('| '+ds+' | '+' | '.join(f'{100*q[a+"_v"]["mean"]:.4f}' for a in ['N','EDeploy','EMix','EOracle','R1'])+' |')
    text +=['','GT R1和support Oracle是旧监督诊断，复用正控而非新方法；原R2的Vid Oracle确认Native参照CI跨零且存在严重负尾仍保留。高support oracle不意味着all-proposal平均质量高，也不意味着当前factorized head可以安全利用整个joint mixture。','',
        '## 原支持的平均质量与上限分开','',
        '以下只在全部E-Mix预测封存后用GT计算，不参与训练或决定。raw指标为每cell全部原proposal的连续物理tIoU，再来源宏平均；offset expectation为各联合网格teacher的期望，不能当成最终envelope tIoU。','',
        '| 面板 | raw mean tIoU % | raw best % | raw比例 tIoU≥.5 % | mean teacher entropy (nats) | mean offset expected tIoU % |','|---|---:|---:|---:|---:|---:|']
    for ds,sp in PANELS:
        q=aggreg[ds][sp];text.append(f'| {ds}/{sp} | {100*q["raw_mean_tIoU"]:.4f} | {100*q["raw_max_tIoU"]:.4f} | {100*q["raw_fraction_tIoU_ge_05"]:.4f} | {q["joint_entropy"]:.4f} | {100*q["expected_offset_joint_tIoU"]:.4f} |')
    text +=['','确认集原raw支持平均tIoU约21.01%/22.25%，支持最高约82.27%/91.66%，达到tIoU≥.5的raw条目仅约15.35%/15.56%。这些测量直接支持“支持里有好条目，但等权经验mass整体质量低”；.5阈值只为报告质量分布，未用于teacher或gate。不是证明重复、diffusion或student投影分别导致损害。','',
        'Forward KL到原additive endpoint student，其logit梯度由teacher的start/end边缘决定；多峰joint相关性不能被这个固定student任意表达。本轮没有改变模型族或 loss，也没做joint-vs-marginal新消融。因此“坏mass”“分布过宽”和“teacher投影”仍是竞争解释，不能仅凭本轮唯一定位其中一个。','',
        '## clean、顺序、全流和来源集中性','',
        '| 确认面板 | Δv vs Native | Δv vs A8 |','|---|---|---|']
    for ds in ['vidstg','hc2']:
        for name in ['expert_clean','flow_corrupt']:text.append(f'| {ds}/{name} | {m(ds,"confirm","EMix_minus_N_v",name)} | {m(ds,"confirm","EMix_minus_A8_v",name)} |')
    text +=['','Vid clean专家相对Native也有明确负向区间，不能把风险仅归因于corruption。全流是固定状态读出模拟，非专家864行保持A逐值相同；Native全流同样仅在专家位置取消Fast。仅25%位置改变，真实全流source-macro配对重算，不能把expert收益直接当全流收益，更不能宣称未来query迁移。','',
        '| 确认 / Native参照 | 正源/总源 | 前两负源份额 | leave-one-source-out净值范围 pp |','|---|---|---:|---|']
    for ds in ['vidstg','hc2']:
        q=concentration[ds]['confirm']['N'];nf='—' if q['top_two_negative_share'] is None else f'{100*q["top_two_negative_share"]:.2f}%'
        text.append(f'| {ds} | {q["positive_sources"]}/{q["sources"]} | {nf} | {q["leave_one_source_out_net_range_pp"]} |')
    text +=['','SUMMARY完整保留两序独立配对CI、每源数值、gross gain/loss、.3基线正确→错误/纠正及>5pp严重损害；没有按source、corruption或结果丢样本，也没有按确认结果重新选参数。独立source仅8/7，集中性与宽区间需一起读。','',
        '## 正负例与执行兑现','',
        '- Vid确认source36 exposure/order1：原Deploy 6.0260%，E-Mix恢复至48.4319%，但Native仍为59.1701%；mix loss 7.3403→3.2025。它缓解了一个top1灾难，但没有超过无更新输入。',
        '- HC开发source8 occlusion/order2：原Deploy 5.3568%，E-Mix回到Native 72.0670%；这是明确保留的恢复例，不外推为确认通用收益。',
        '- Vid确认source37 exposure/order2：Native43.2520%→Mix24.4508%，loss8.9280→2.8769；目标被执行但GT受损。',
        '- HC确认source41 frame_drop/order2：Deploy46.2299%、Native44.5656%→Mix14.5194%，loss7.2210→3.2286；等权聚合也能破坏原本好的top1。',
        '- HC确认source42 frame_freeze/order1：Native14.5381%→Mix17.9085%，为保留的真实局部正例；全组净效应依然负。','',
        '四个corruption专家面板全部loss下降；其中loss下降但Native vIoU变坏的条目分别Vid开发11/确认20，HC开发35/确认17。冻结K3而不是按GT挑步；每步loss、梯度、位移、native读出与dense指标均公开。CASES对每面板、Native/A8/Deploy三个参照保留最大正例和最大负例。','',
        '## 审计、成本与裁决','',
        f'独立NumPy逐分量Gaussian归一化/等权joint混合、完整288 fits的864次joint-marginal梯度和SGD参数算术、head reset、logits、原生MAP、teacher joint MAP、旧raw支持及复用R2指标、1152个A状态绑定和official dense全部核验通过；根检查{sum(root["checks"].values()):,}。公开独立bootstrap/trace/paired算术核验{public["checks"]["numeric_scalars"]:,}标量和{public["checks"]["tail_counts"]}tail计数通过。8项CPU单元检查覆盖M1逐值parity、重复质量、多峰joint非边缘乘积、guard、归一化等。','',
        f'本轮新288适应/864 backward，fit阶段CPU {res["mix_CPU_wall_seconds"]:.3f}s、dense阶段{res["dense_score_CPU_wall_seconds"]:.3f}s；不含phase前hash/加载、开发、审计、绘图。独立根审计{root["CPU_wall_seconds"]:.3f}s另计。新GPU/backbone/suffix/expert调用、空间更新、时间跨query写入均0；private状态轨迹{res["private_run_bytes"]:,}bytes不公开。','',
        '**最终保留 A，停止将本轮 raw equal-mixture teacher 作为可部署改进。** 本轮没有证明不确定性保留解决单中心瓶颈；也没有否定 temporal gradient channel、所有mixture或经过净化的expert证据。优先问题若继续，应是“如何获得可信监督mass，并区分student投影代价”；不是在这轮失败后擅自调LR/K、加confidence weighting/PoE/gate或进入R3。没有运行任何后续分支。','',
        '## 复现材料','',
        '- `protocols/tastvg_dta_mixture_r2b_v1.md`；`docs/tastvg_dta_mixture_r2b_v1/EXECUTION.md`。',
        '- `vg_tta/tastvg_dta_mixture_r2b_v1.py`；有限runner `scripts/run_tastvg_dta_mixture_r2b_v1.py`。',
        '- `scripts/test_tastvg_dta_mixture_r2b_v1.py`；独立审计 `scripts/audit_tastvg_dta_mixture_r2b_v1.py root`或公开结果目录；报告与图生成 `scripts/report_tastvg_dta_mixture_r2b_v1.py`。',
        '- `results/tastvg_dta_mixture_r2b/2026-10-04`：全1152匿名指标行、288三步轨迹、GT后验teacher质量、完整source CI/tails/cases及三PNG/PDF。',
        '- 旧R2对照来自commit324b3357dbd67b435d1fb507ae516a44ecf53961；R1正控来自885762286dd75b806fae34b0bb712583d5088d84；未重跑它们。Private GT、原proposal坐标、hidden/weights、媒体与当前生产注册不公开。','']
    (ROOT/'docs/TA_DTA_MIXTURE_R2B_REVIEW.md').write_text('\n'.join(text))
    own=['vg_tta/tastvg_dta_mixture_r2b_v1.py','scripts/run_tastvg_dta_mixture_r2b_v1.py','scripts/test_tastvg_dta_mixture_r2b_v1.py',
        'scripts/audit_tastvg_dta_mixture_r2b_v1.py','scripts/report_tastvg_dta_mixture_r2b_v1.py',
        'protocols/tastvg_dta_mixture_r2b_v1.md','docs/tastvg_dta_mixture_r2b_v1/EXECUTION.md','docs/TA_DTA_MIXTURE_R2B_REVIEW.md']
    deps=['vg_tta/tastvg_dta_oracle_v1.py','vg_tta/tastvg_dta_expert_r2_v1.py','scripts/audit_tastvg_dta_oracle_r1_v1.py',
        'scripts/decota_matrix_common_v1.py','scripts/tastvg_correction_views_common_v1.py','vg_tta/tastvg_oracle_event5_v1.py',
        'vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py']
    emit(PUB/'CODE_BINDING.json',dict(files={f:sha(ROOT/f) for f in own+deps},runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json')))
    emit(PUB/'FIGURE_MANIFEST.json',dict(files={str((PUB/f'{f}.{ext}').relative_to(ROOT)):sha(PUB/f'{f}.{ext}') for f in figs for ext in ['png','pdf']},
        source_summary_sha256=sha(PUB/'SUMMARY.json')))
    print('R2B_REPORT_AND_THREE_FIGURES_WRITTEN',flush=True)

if __name__=='__main__':main()
