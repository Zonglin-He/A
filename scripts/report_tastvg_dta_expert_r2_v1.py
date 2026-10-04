"""Matched R2 controls, concentration, negative cases and scientific figures."""
import os
os.environ['MPLBACKEND']='Agg'
import sys,time,csv,collections,json,shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write as locked_write,sha
PUB=ROOT/'results/tastvg_dta_expert_r2/2026-10-04';BASE=ROOT/'artifacts/tastvg_dta_expert_r2_v1'
ARMS=['EDeploy','EOracle'];PANELS=[('vidstg','search'),('hc2','search'),('vidstg','confirm'),('hc2','confirm')]

def write(p,z):
    """Keep derived evidence immutable; preserve prior export bindings on repair."""
    if not p.exists():return locked_write(p,z)
    old=read(p)
    if old==z:return
    if isinstance(old,dict) and isinstance(z,dict) and {k:v for k,v in old.items() if k!='time'}=={k:v for k,v in z.items() if k!='time'}:return
    assert p.name in ['CODE_BINDING.json','FIGURE_MANIFEST.json'],p
    recovery=BASE/'recovery/report_export_001';recovery.mkdir(parents=True,exist_ok=True)
    target=recovery/(p.name+'.'+sha(p))
    if not target.exists():shutil.copy2(p,target)
    p.write_text(json.dumps(z,indent=2)+'\n')

def main():
    s=read(PUB/'SUMMARY.json');rows=read(PUB/'ROWS.json');cfg=read(PUB/'CONFIG.json');res=read(PUB/'RESOURCES.json')
    root=read(PUB/'ROOT_AUDIT.json');public=read(PUB/'PUBLIC_AUDIT.json');assert root['status']==public['status']=='pass'
    traces={a:{r['cell_key']:r for r in read(PUB/(a+'_TRACES_SCORED.json'))} for a in ARMS};sels=read(PUB/'TEACHER_SELECTION_SCORED.json')
    concentration={};teacher={};changes={};cases=[]
    for ds,sp in PANELS:
        q=[r for r in rows if r['dataset']==ds and r['split']==sp and r['expert_scheduled'] and r['condition']!='clean']
        concentration.setdefault(ds,{})[sp]={};teacher.setdefault(ds,{})[sp]={};changes.setdefault(ds,{})[sp]={}
        for a in ARMS:
            concentration[ds][sp][a]={}
            for b in ['N','A8']:
                v=s[ds][sp]['expert_corrupt']['metrics'][a+'_minus_'+b+'_v']['source_values'];ids=list(v);x=np.array(list(v.values()));pos=np.maximum(x,0);neg=np.maximum(-x,0);loo=(x.sum()-x)/(len(x)-1)
                pi=np.argsort(-pos);ni=np.argsort(-neg)
                concentration[ds][sp][a][b]=dict(sources=len(x),positive_sources=int((x>1e-12).sum()),negative_sources=int((x<-1e-12).sum()),
                    largest_positive_source=ids[int(pi[0])],largest_positive_share=float(pos[pi[0]]/pos.sum()) if pos.sum() else None,
                    top_two_positive_share=float(pos[pi[:2]].sum()/pos.sum()) if pos.sum() else None,
                    largest_negative_source=ids[int(ni[0])],largest_negative_share=float(neg[ni[0]]/neg.sum()) if neg.sum() else None,
                    top_two_negative_share=float(neg[ni[:2]].sum()/neg.sum()) if neg.sum() else None,
                    leave_one_source_out_net_range_pp=[float(100*loo.min()),float(100*loo.max())])
                for sign,r in [('largest_gain',max(q,key=lambda r:r[a+'_minus_'+b+'_v'])),('largest_harm',min(q,key=lambda r:r[a+'_minus_'+b+'_v']))]:
                    cases.append(dict(dataset=ds,split=sp,arm=a,baseline=b,selection=sign,cell=r,trace=traces[a][r['cell_key']]['trace']))
            chosen={r['cell_key']:r for r in sels if r['dataset']==ds and r['split']==sp and r['condition']!='clean' and r['arm']==a}
            bysource=collections.defaultdict(list)
            for r in q:bysource[r['source_id']].append(chosen[r['cell_key']]['continuous_teacher_tIoU'])
            teacher[ds][sp][a]=dict(source_macro_continuous_teacher_tIoU=float(np.mean([np.mean(x) for x in bysource.values()])),
                source_macro_selected_confidence=float(np.mean([np.mean([chosen[r['cell_key']]['confidence'] for r in q if r['source_id']==i]) for i in bysource])),
                teacher_direct_v=s[ds][sp]['expert_corrupt']['metrics'][a+'_direct_v']['mean'],
                Gaussian_MAP_v=s[ds][sp]['expert_corrupt']['metrics'][a+'_map_v']['mean'])
            changes[ds][sp][a]=dict(cells=len(q),native_interval_changed=sum(r[a+'_interval_changed'] for r in q),
                loss_decreased=sum(r[a+'_loss_after']<r[a+'_loss_before'] for r in q),
                interval_unchanged_but_loss_decreased=sum(not r[a+'_interval_changed'] and r[a+'_loss_after']<r[a+'_loss_before'] for r in q),
                loss_down_native_v_down=sum(r[a+'_loss_after']<r[a+'_loss_before'] and r[a+'_minus_N_v']<-1e-12 for r in q))
        sd={r['cell_key']:r['selected_index'] for r in sels if r['dataset']==ds and r['split']==sp and r['condition']!='clean' and r['arm']=='EDeploy'}
        so={r['cell_key']:r['selected_index'] for r in sels if r['dataset']==ds and r['split']==sp and r['condition']!='clean' and r['arm']=='EOracle'}
        teacher[ds][sp]['same_center_index_cells']=sum(sd[k]==so[k] for k in sd);teacher[ds][sp]['expert_corrupt_cells']=len(sd)
    write(PUB/'SOURCE_CONCENTRATION.json',concentration);write(PUB/'TEACHER_DIAGNOSTICS.json',teacher);write(PUB/'EXECUTION_DIAGNOSTICS.json',changes);write(PUB/'CASES.json',cases)
    decision=dict(EDeploy='NO_GO for this raw-confidence single-center teacher',
        EOracle='Support can produce near-R1 gradient gains; Vid confirmation versus Native uncertain and severe harm present',
        teacher_interval_is_only_scientific_variable=True,
        measured_support_selection_gap=True,hard_selection_destroys_uncertainty_proven=False,
        expert_support_globally_inadequate_proven=False,online_persistence_ready=False,
        next_conditional_question='Can deployment evidence aggregation retain good support without GT? Equal mixture remains an untested candidate, not a completed result.',
        R2b_started=False,R3_started=False,production_promoted=False,A8_changed=False,target_retuning=False,
        interpretation='Raw top confidence is insufficient here. Positive oracle means imperfect support plus selection deserve separation; oracle GT choice cannot deploy.',time=time.time())
    write(PUB/'DECISION.json',decision)
    cols=['cell_key','dataset','split','source_id','condition','order','arrival','expert_scheduled','EDeploy_GT_supervised','EOracle_GT_supervised']
    cols += [a+'_'+m for a in ['N','A8','R1','EDeploy','EOracle','EDeploy_direct','EOracle_direct','EDeploy_map','EOracle_map'] for m in ['v','t']]
    cols += [a+'_minus_'+b+'_'+m for a in ARMS for b in ['N','A8','R1'] for m in ['v','t']]+['EOracle_minus_EDeploy_v','EOracle_minus_EDeploy_t']
    with (PUB/'ROWS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r[k] for k in cols} for r in rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42})
    colors={'EDeploy':'#BD654A','EOracle':'#477CA8'};figs=[]
    fig,ax=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    plotpanels=[('vidstg','search'),('vidstg','confirm'),('hc2','search'),('hc2','confirm')]
    for a,metric,title in zip(ax,['t','v'],['Temporal IoU','Video IoU (fixed spatial A)']):
        for j,(ds,sp) in enumerate(plotpanels):
            for arm,offset in [('EDeploy',-.12),('EOracle',.12)]:
                z=s[ds][sp]['expert_corrupt']['metrics'][arm+'_minus_N_'+metric];mu=100*z['mean'];lo,hi=np.array(z['ci95'])*100
                a.errorbar(mu,j+offset,xerr=[[mu-lo],[hi-mu]],fmt='o',color=colors[arm],capsize=3,ms=5,label=arm if j==0 else None)
        a.set_yticks(range(4),['Vid development','Vid confirmation','HC development','HC confirmation']);a.invert_yaxis();a.axvline(0,color='.6',lw=.8)
        a.set_title(title);a.set_xlabel('Gain over zero-update Native (pp)');a.grid(axis='x',alpha=.15);a.legend(frameon=False,fontsize=9)
    def savefig(name):
        for ext in ['png','pdf']:fig.savefig(PUB/f'{name}.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig);figs.append(name)
    savefig('r2_teacher_gradient_channel')
    fig,ax=plt.subplots(1,2,figsize=(10.5,3.4),layout='constrained')
    for a,ds,label in zip(ax,['vidstg','hc2'],['Vid confirmation','HC confirmation']):
        z=s[ds]['confirm']['expert_corrupt']['metrics'];x=np.arange(3)
        for arm,offset in [('EDeploy',-.17),('EOracle',.17)]:
            a.bar(x+offset,[100*z[k]['mean'] for k in [arm+'_direct_v',arm+'_map_v',arm+'_v']],width=.32,color=colors[arm],label=arm)
        a.axhline(100*z['N_v']['mean'],color='.4',ls='--',lw=1,label='Native')
        a.axhline(100*z['R1_v']['mean'],color='#6B9A67',ls=':',lw=1.5,label='R1 GT adaptation')
        a.set_xticks(x,['Direct teacher interval','Teacher Gaussian MAP','3-step head adaptation']);a.tick_params(axis='x',labelsize=8)
        a.set_ylabel('Expert-corrupt vIoU (%)');a.set_title(label);a.legend(frameon=False,fontsize=8);a.grid(axis='y',alpha=.15)
    savefig('r2_teacher_and_execution')
    fig,ax=plt.subplots(1,2,figsize=(10.5,3.4),layout='constrained')
    for a,ds,label in zip(ax,['vidstg','hc2'],['Vid confirmation','HC confirmation']):
        ids=list(s[ds]['confirm']['expert_corrupt']['metrics']['EDeploy_minus_N_v']['source_values']);x=np.arange(len(ids))
        for arm,offset in [('EDeploy',-.17),('EOracle',.17)]:
            v=s[ds]['confirm']['expert_corrupt']['metrics'][arm+'_minus_N_v']['source_values'];a.bar(x+offset,[100*v[i] for i in ids],width=.32,color=colors[arm],label=arm)
        a.axhline(0,color='.5',lw=.8);a.set_xticks(x,ids);a.set_xlabel('Anonymous source ID');a.set_ylabel('Mean vIoU gain over Native (pp)');a.set_title(label);a.grid(axis='y',alpha=.15);a.legend(frameon=False,fontsize=9)
    savefig('r2_confirmation_sources')
    def m(ds,sp,k,panel='expert_corrupt'):
        z=s[ds][sp][panel]['metrics'][k];return f'{100*z["mean"]:+.4f} [{100*z["ci95"][0]:+.4f}, {100*z["ci95"][1]:+.4f}]'
    t=['# R2: Expert-Teacher Distributional Temporal Adaptation','',
       '**本轮已完成两臂实际适应与独立核验：原始专家最高置信度单区间 teacher 未通过，不能进入 R3 持久学习。** 同一专家支持中用GT选中心的E-Oracle可产生接近R1的梯度收益，但Vid确认相对Native的配对区间跨零且有5个严重损害，不把它写成两个确认面板均稳定通过。裁决仅针对预锁单中心转换；没有否定专家支持或native temporal adaptation路线。','',
       '## 唯一变量：teacher interval 来源','',
       '原TA-STVG同域EMA checkpoint、原32开发+16确认来源/每源一query/双序/clean与五种5%瞬态corruption/25%专家位置、原始输入像素和两offset采样均保持。空间A（1792参数、Vid K1/HC K8 Uniform Rank-RKL）状态与完整框轨迹只读复用；生产CURRENT仍为DeCoTA，研究A不等于生产方法。','',
       '本轮沿用R1独立源验证选出的lr=.01，两个数据集相同；只原native `temp_embed.layers.1`的2×256 weight+2 bias，514名义参数、bias在joint归一化后不可辨识。缓存final hidden先经过冻结MLP首层/ReLU。K3普通SGD，sigma一原offset中位相邻网格cell，beta1，forward KL(q||pθ)+KL(p0||pθ)。每query回到原head，最终第3步当前query读出后丢弃；不是未来query更新或online persistence。完全复用R1的fit_query，未调LR/K/subset/先验/解码。','',
       '| Arm | Gaussian中心 | GT参与teacher |','|---|---|---|',
       '| R1（复用正控） | 原GT时间 | 是；不重跑 |',
       '| E-Oracle | 现有UniversalVTG raw proposal列表中连续物理tIoU最高的区间 | 是，明确监督支持诊断 |',
       '| E-Deploy | 同一列表中proposal_confidence最高的区间 | 否，部署可获得的raw top1 |','',
       '保留原列表全部有效proposal、重复、次序和小数端点；exact tie取首索引。没有NMS/unifier/新增候选/mixture/PoE/score weighting/gate/memory。E-Deploy是缓存raw专家top1，**不是**A8在Old8学生区间上max(confidence×tIoU)的最终读出，也不是新实现另一个expert postprocessor。两臂Gaussian形状和所有优化机制相同。','',
       f'原288个专家位置，每臂288次，合计576 query-arm适应/1728 backward；专家支持含{cfg["proposal_count_range"][0]}–{cfg["proposal_count_range"][1]}个原proposal，{cfg["unique_expert_cache_files"]}个独立缓存文件。其他864到达保持缓存A；总1152是匹配读出模拟，未重跑backbone、suffix、空间专家或完整在线流。确认实际专家独立源Vid8/HC7，开发16/14；均有历史曝光，不能称fresh test。','',
       'Deploy在独立worker装入GT-reading guard，所有288预测封存后，另一个Oracle worker才读取目标GT选择中心；其288预测全部封存后才单独dense评分。部署worker禁止GT/Oracle/scored results读入；6项测试验证guard和selector并列等规则。Oracle选择按教师区间tIoU，不按更新后vIoU挑中心/步数。','',
       '## 主对照：E-Deploy / E-Oracle − zero-update Native','',
       '专家corruption；单位pp，等来源宏平均、10000次配对source-bootstrap。Native是在同A空间输入上的原head输出；优先回答teacher→gradient是否产生定位收益。','',
       '| 面板 | Teacher | ΔtIoU [95% CI] | ΔvIoU [95% CI] | >5pp v损害 |','|---|---|---|---|---:|']
    for ds,sp in PANELS:
        for arm in ARMS:
            z=s[ds][sp]['expert_corrupt'];t.append(f'| {ds}/{sp} ({z["sources"]}源/{z["cells"]}cells) | {arm} | {m(ds,sp,arm+"_minus_N_t")} | {m(ds,sp,arm+"_minus_N_v")} | {z["negative_tails"][arm]["N"]["severe_harm"]} |')
    t +=['','E-Deploy四个面板vIoU均值均负；HC开发t/v两个区间明确为负。确认两集相对Native的v区间跨零，不能把负均值包装成统计确定的普遍下降；但严重负尾与未建立正增量已足以停止此部署teacher版本。','',
       'E-Oracle在三个面板相对Native的t/v区间为正；Vid确认均值正但两区间均跨零，5/40条相对Native下降超过5pp，集中在source37。全部面板E-Oracle−R1区间跨零：不能把两者均值接近称为等效性检验通过。其结果仍说明该专家support可以给已验证的参数通道提供有用中心，不能直接认定support完全不足。','',
       '## 第二层参照：当前A8与R1','',
       '| 面板 | Teacher | ΔvIoU vs A8 | ΔvIoU vs R1 | >5pp损害 vs A8 |','|---|---|---|---|---:|']
    for ds,sp in PANELS:
        for arm in ARMS:
            z=s[ds][sp]['expert_corrupt'];t.append(f'| {ds}/{sp} | {arm} | {m(ds,sp,arm+"_minus_A8_v")} | {m(ds,sp,arm+"_minus_R1_v")} | {z["negative_tails"][arm]["A8"]["severe_harm"]} |')
    t +=['','E-Deploy相对A8的HC开发/确认v区间均低于零；Vid两面板跨零且均值负。E-Oracle确认Vid相对A8正，HC仍不确定且有7个严重损害；它使用目标GT，本来也不能直接部署替换A8。','',
       '## Teacher选择、网格投影与执行分开看','',
       '| 面板 | Teacher | 连续teacher tIoU % | 直接teacher vIoU % | Gaussian MAP vIoU % | 3步后 vIoU % |','|---|---|---:|---:|---:|---:|']
    for ds,sp in PANELS:
        z=s[ds][sp]['expert_corrupt']['metrics']
        for a in ARMS:
            x=teacher[ds][sp][a];t.append(f'| {ds}/{sp} | {a} | {100*x["source_macro_continuous_teacher_tIoU"]:.4f} | {100*x["teacher_direct_v"]:.4f} | {100*x["Gaussian_MAP_v"]:.4f} | {100*z[a+"_v"]["mean"]:.4f} |')
    t +=['','连续teacher tIoU用于proposal选中质量诊断；直接teacher/全部预测的official dense scorer按现有代码对端点整数截断，Gaussian中心仍保留小数。Gaussian MAP沿原两offset严格i<j支持和envelope，无GT直接送进decoder。直接teacher与MAP仅为诊断，不能冒充梯度TTA。','',
       'E-Oracle直接区间和Gaussian MAP均明显高于E-Deploy；这条差距在优化之前已存在。E-Deploy跟随的高置信度中心可能远离正确事件，三步KL确实下降，却沿错误目标损害输出。此次没有改变representation、梯度实现或学习率来补偿。','',
       'Oracle选中也不是GT中心：Vid确认source37 exposure/order2，Native 43.2520%→E-Oracle 37.5886%，直接teacher 38.7454%、teacher MAP 34.4994%，R1保持Native。存在较好的support不等于选中的近似边界在固定空间下必然好，更不等于每一步参数优化保证vIoU。','',
       '## 集中性、顺序、clean与完整缓存流','',
       '| 确认集 / teacher | 正源/总源 | 最大正源份额 | 前两负源份额 | leave-one-source-out净增量范围 pp |','|---|---|---:|---:|---|']
    for ds in ['vidstg','hc2']:
        for arm in ARMS:
            x=concentration[ds]['confirm'][arm]['N'];pf='—' if x['largest_positive_share'] is None else f'{100*x["largest_positive_share"]:.2f}%';nf='—' if x['top_two_negative_share'] is None else f'{100*x["top_two_negative_share"]:.2f}%'
            t.append(f'| {ds}/{arm} | {x["positive_sources"]}/{x["sources"]} | {pf} | {nf} | {x["leave_one_source_out_net_range_pp"]} |')
    t +=['','保留所有来源和leave-out变化，不删除source33/36/37/43追分。Vid Oracle正收益仍高度集中，Native参照负尾也集中；HC确认专家源仅7个。图中公开每源两臂净变化，SUMMARY保留每个顺序独立CI和gross gain/loss，不能用cells数替代独立source数。','',
       '| 确认全corrupt缓存流 | Teacher | ΔvIoU vs A8 | ΔtIoU vs A8 |','|---|---|---|---|']
    for ds in ['vidstg','hc2']:
        for arm in ARMS:t.append(f'| {ds} (16源/160cells) | {arm} | {m(ds,"confirm",arm+"_minus_A8_v","flow_corrupt")} | {m(ds,"confirm",arm+"_minus_A8_t","flow_corrupt")} |')
    t +=['','这些为全流来源配对重算；仅专家位置变，其余864行逐值A，不能称时间监督向未来非专家位置迁移。Full-flow Native也仅在专家位置取消Fast，其他位置仍A。','',
       '| 确认clean专家 | Teacher | ΔvIoU vs Native | ΔvIoU vs A8 |','|---|---|---|---|']
    for ds in ['vidstg','hc2']:
        for arm in ARMS:t.append(f'| {ds} | {arm} | {m(ds,"confirm",arm+"_minus_N_v","expert_clean")} | {m(ds,"confirm",arm+"_minus_A8_v","expert_clean")} |')
    t +=['','## 具体正负例与loss','',
       '- Vid确认source36 exposure/order1：Native 59.1701%→E-Deploy 6.0260%，直接teacher和Gaussian MAP为0；loss 13.1850→4.9672。错误center被梯度执行，不是loss不下降。',
       '- HC开发source8 occlusion/order2：Native 72.0670%→E-Deploy 5.3568%，loss 16.6071→4.6945；同support Oracle保留有用中心。',
       '- HC确认source43 exposure/order2：Native 31.3400%→E-Deploy 0%，loss 19.5209→4.4598，baseline-good被破坏。',
       '- Vid确认source33及HC source47的Oracle正例保留在CASES；每个面板/每臂相对Native和A8均公开最大收益及最大损害，不能只挑恢复案例。','',
       '所有corruption专家fit的loss下降；loss下降但Native vIoU下降，Deploy开发/确认Vid12/11、HC45/13；Oracle Vid5/5、HC3/1。非负loss不是任务correctness。EXECUTION_DIAGNOSTICS给出所有MAP是否改变及每步轨迹，而非用最终KL替代定位指标。','',
       '## 核验、成本与当前决定','',
       f'根审计独立NumPy重算576个query-arm fit全部1728次joint-marginal梯度、SGD、head reset、logits和原生MAP；两臂teacher首argmax与全部teacher Gaussian/MAP、R1输入/输出保留、1152个A绑定及official dense指标全部通过。根检查{sum(root["checks"].values()):,}；公开标量{public["checks"]["numeric_scalars"]:,}、tail计数{public["checks"]["tail_counts"]}，六CPU测试通过。','',
       f'阶段内CPU时间Deploy {res["deploy_CPU_wall_seconds"]:.3f}s、Oracle {res["oracle_CPU_wall_seconds"]:.3f}s、dense {res["dense_score_CPU_wall_seconds"]:.3f}s；不含phase前hash/加载、开发、审计、报告。根独立核验{root["CPU_wall_seconds"]:.3f}s另计。新增GPU/backbone/suffix/expert均0；空间更新和时间跨query写入0，private运行轨迹{res["private_run_bytes"]:,}bytes不公开。','',
       '**结论：本轮E-Deploy NO-GO，暂不进入R3；保留A与native-head梯度路线。** 结果更接近“support提供有用中心，部署top-confidence选择不可靠”，但Vid Oracle确认未稳定通过Native参照，不能硬套成两个数据集完美的E-Oracle✅/E-Deploy❌。','',
       '本轮没有验证“hard selection损失有效uncertainty”的因果说法：Oracle−Deploy同时改变了中心correctness；只有另一次matched aggregation实验才能验证多hypothesis保留是否有用。Equal mixture/weighting/PoE/persistence均未运行，不把它们作为已获收益。下一条件问题可以是同机制下无GT证据聚合；需先固定teacher与合法span投影，不能凭本轮就扩大K/LR/参数、换专家或重新扫描gate。','',
       '## 可复现文件','',
       '- `protocols/tastvg_dta_expert_r2_v1.md`；`docs/tastvg_dta_expert_r2_v1/EXECUTION.md`。',
       '- 有限runner：`scripts/run_tastvg_dta_expert_r2_v1.py`；共用R1 fit：`vg_tta/tastvg_dta_oracle_v1.py`。',
       '- 测试：`scripts/test_tastvg_dta_expert_r2_v1.py`；根/公开审计：`scripts/audit_tastvg_dta_expert_r2_v1.py root`或结果目录。',
       '- 匿名1152行指标、576条三步轨迹、全部teacher诊断/CI/负尾/cases/PNG-PDF：`results/tastvg_dta_expert_r2/2026-10-04`。',
       '- Private support/GT坐标/hidden/head weights/媒体均不公开；公开标量审核无法重建这些被排除资产。','']
    (ROOT/'docs/TA_DTA_EXPERT_R2_REVIEW.md').write_text('\n'.join(t))
    own=['vg_tta/tastvg_dta_expert_r2_v1.py','scripts/run_tastvg_dta_expert_r2_v1.py','scripts/test_tastvg_dta_expert_r2_v1.py',
        'scripts/audit_tastvg_dta_expert_r2_v1.py','scripts/report_tastvg_dta_expert_r2_v1.py',
        'protocols/tastvg_dta_expert_r2_v1.md','docs/tastvg_dta_expert_r2_v1/EXECUTION.md','docs/TA_DTA_EXPERT_R2_REVIEW.md']
    deps=['vg_tta/tastvg_dta_oracle_v1.py','scripts/audit_tastvg_dta_oracle_r1_v1.py','methods/decota_final_simplified_v1/objectives.py',
        'scripts/decota_matrix_common_v1.py','scripts/tastvg_correction_views_common_v1.py','vg_tta/tastvg_oracle_event5_v1.py',
        'vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py']
    write(PUB/'CODE_BINDING.json',dict(files={f:sha(ROOT/f) for f in own+deps},runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json')))
    write(PUB/'FIGURE_MANIFEST.json',dict(files={str((PUB/f'{f}.{ext}').relative_to(ROOT)):sha(PUB/f'{f}.{ext}') for f in figs for ext in ['png','pdf']},
        source_summary_sha256=sha(PUB/'SUMMARY.json'),teacher_diagnostics_sha256=sha(PUB/'TEACHER_DIAGNOSTICS.json')))
    print('R2_REPORT_AND_THREE_FIGURES_WRITTEN',flush=True)

if __name__=='__main__':main()
