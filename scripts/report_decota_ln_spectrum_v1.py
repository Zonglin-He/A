"""Complete positive/negative CPU evidence report; no candidate promotion."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_decota_ln_spectrum_v1 import BASE,PUB,archive
from scripts.decota_matrix_common_v1 import write,status
from scripts.decota_public_result_io_v1 import read
from scripts.decota_ln_spectrum_math_v1 import plain,RANKS
import numpy as np

def number(x,scale=1):return 'NA' if x is None else f'{x*scale:.3f}'
def estimate(q,scale=1):
    if not q or q.get('mean') is None:return 'NA'
    ci=q.get('ci95');return number(q['mean'],scale)+(f" [{number(ci[0],scale)}, {number(ci[1],scale)}]" if ci is not None else '')

def run():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert read(PUB/'ROOT_AUDIT.json')['status']==read(PUB/'PUBLIC_AUDIT.json')['status']=='pass'
    g=read(PUB/'GEOMETRY_SUMMARY.json');u=read(PUB/'UTILITY_SUMMARY.json');pairs=read(PUB/'PAIR_ROWS.json');x=read(PUB/'EXTRACTION_AUDIT.json')
    decisions={}
    for ds in ['vidstg','hc2']:
        z=next(s['summary'] for s in u if s['dataset']==ds and s['split']=='confirm' and s['group']=='corruption');checks={}
        for k in ['correlation','positive_minus_negative']:
            ci=z.get('metrics',{}).get(k,{}).get('ci95');checks[k]=ci is not None and ci[0]>0
        decisions[ds]=dict(cosine_utility_evidence=all(checks.values()),criteria=checks,pairs=z['pairs'],donors=z.get('donors',0),recipients=z.get('recipients',0))
    supported=all(d['cosine_utility_evidence'] for d in decisions.values())
    decision=dict(status='connection_supported_in_limited_prefix' if supported else 'shared_geometry_not_validated_as_transferable_memory',datasets=decisions,
        evidence_rule='both confirmation corrupt CI lower >0 for cosine-utility correlation and positive-minus-negative utility',
        projected_memory_executed=False,projected_method_validated=False,GT_selected_rank=False,production_promoted=False,new_GPU_queue_started=False,
        next_scope='A separately locked matched projection experiment is motivated' if supported else 'Do not convert low-rank energy into a shared-memory claim or launch the conditional method',time=time.time())
    write(PUB/'DECISION.json',decision)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.bbox':'tight'})
    fig,ax=plt.subplots(2,2,figsize=(10,6),sharex=True,sharey=True)
    for row,ds in enumerate(['vidstg','hc2']):
        for col,sp in enumerate(['search','confirm']):
            a=ax[row,col]
            for stream,ls in [('episodic','-'),('online100','--')]:
                z=next(s for s in g['spectra'] if s['dataset']==ds and s['stream']==stream and s['split']==sp and s['group']=='corruption')
                for mode,color in [('raw','#2563a6'),('unit','#c77c34'),('centered','#638357')]:
                    a.plot(RANKS,[z['spectra'][mode]['energy'][str(r)]*100 for r in RANKS],marker='o',color=color,ls=ls,label=f'{stream}: {mode}')
            a.set_xscale('log',base=2);a.set_xticks(RANKS,labels=RANKS);a.set_ylim(0,102);a.set_title(f'{ds} / {sp} / corruption');a.grid(alpha=.15);a.set_ylabel('Retained parameter energy (%)')
            if row==1:a.set_xlabel('Rank')
    ax[0,0].legend(fontsize=7,ncol=2);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(PUB/f'spectrum.{ext}')
    plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.7))
    for a,ds in zip(ax,['vidstg','hc2']):
        h=next(s for s in g['heldout'] if s['dataset']==ds and s['stream']=='episodic' and s['group']=='corruption' and s['test']=='development_to_confirmation')
        a.plot(RANKS,[h['stats']['metrics'][f'E{r}']['mean']*100 for r in RANKS],'-o',label='Development basis → confirmation')
        for budget,color in [('25','#c77c34'),('50','#638357'),('100','#6d55a0')]:
            z=next((s for s in g['strict_prefix'] if s['dataset']==ds and s['split']=='confirm' and s['group']=='corruption' and s['budget']==budget and s['minimum_previous_nonzero_writes']==8),None)
            if z:a.plot(RANKS,[z['stats']['metrics'][f'E{r}']['mean']*100 for r in RANKS],'-s',label=f'Prior-only basis, {budget}% budget',color=color)
        a.set_xscale('log',base=2);a.set_xticks(RANKS,labels=RANKS);a.set_ylim(0,102);a.set_title(ds+' confirmation');a.set_xlabel('Rank');a.set_ylabel('Unseen correction energy (%)');a.grid(alpha=.15);a.legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(PUB/f'heldout_prefix.{ext}')
    plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.7))
    for a,ds in zip(ax,['vidstg','hc2']):
        rr=[r for r in pairs if r['dataset']==ds and r['split']=='confirm' and r['condition']!='clean' and r['cosine'] is not None]
        for donor in sorted({r['donor'] for r in rr}):
            q=[r for r in rr if r['donor']==donor];a.scatter([r['cosine'] for r in q],[r['utility_v']*100 for r in q],s=14,alpha=.55,label=f'donor {donor}')
        a.axhline(0,color='k',lw=.7);a.axvline(0,color='k',lw=.7);a.set_title(ds+' confirmation, exact first-write prefix');a.set_xlabel('Donor / recipient episodic LN cosine');a.set_ylabel('Single write ΔvIoU (pp)');a.grid(alpha=.15);a.legend(fontsize=6,ncol=3)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(PUB/f'cosine_utility.{ext}')
    plt.close(fig)
    lines=['# Single-expert DeCoTA: LN correction geometry and first-write transfer','',
        '本轮实际完成纯 CPU 诊断。固定 Native WHEN、一个 frozen Grounding DINO、原四个预测事件内观察、admitted Top1 critic、Adam .03 / joint1792 / 10-step own-loss selection。没有模型前向、decoder重放、反向、新专家或新 GT 计分。没有执行低秩写入；生产 CURRENT 未改。','',
        f"已核验 {x['payloads']} 个旧封存状态；两集各32开发＋16来源互斥但历史曝光确认，一query/source、双序、clean＋五类5%、12独立流。独立episode二序同输入去重后576 source-condition corrections（每集288）。1536维向量取 selected fit.state − initial，不是末步或已缩小1/16的committed状态。零更新保留。",'',
        '**本轮裁决：** '+decision['status']+'. '+decision['next_scope']+'.','',
        '## Correction spectrum','',
        '| Dataset / panel / stream | N / sources / zero | E1 | E2 | E4 | E8 | E16 | E32 | unit E8 | centered E8 | effective rank |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in g['spectra']:
        if s['group']!='corruption':continue
        z=s['spectra']['raw'];e=z['energy'];lines.append(f"| {s['dataset']} / {s['split']} / {s['stream']} | {z['cells']} / {z['sources']} / {z['zero_cells']} | "+' | '.join(number(e[str(r)],100) for r in RANKS)+f" | {number(s['spectra']['unit']['energy']['8'],100)} | {number(s['spectra']['centered']['energy']['8'],100)} | {number(z['effective_rank'])} |")
    lines+=['','百分数是参数平方范数能量，不是任务收益。raw谱不减均值；unit control消除大范数样本优势；centered control去掉共用平均方向。三者分别分析，不把中心化PCA说成原始proposal的实际投影。不同checkpoint的1536维空间不合并。','',
        '## Unseen-source and chronological evidence','',
        '| Dataset / stream / group | Held-out test | sources | E8 (%) with source CI | E16 (%) |','|---|---|---:|---:|---:|']
    for s in g['heldout']:
        if s['group']!='corruption':continue
        z=s['stats'];lines.append(f"| {s['dataset']} / {s['stream']} / {s['group']} | {s['test']} | {z['sources']} | {estimate(z['metrics'].get('E8'),100)} | {estimate(z['metrics'].get('E16'),100)} |")
    lines+=['','Development两hash folds以source互斥，所有corruption和重复order同时holdout。confirmation只由development basis投影；标签不选rank。谱能量低秩即使跨source保留，仍不等于这些方向有正向任务效用。','',
        '| Dataset / confirmation budget | prior nonzero writes ≥ | evaluated sources / cells | prior-only E8 (%) |','|---|---:|---:|---:|']
    for s in g['strict_prefix']:
        if s['split']!='confirm' or s['group']!='corruption':continue
        z=s['stats'];lines.append(f"| {s['dataset']} / {s['budget']}% | {s['minimum_previous_nonzero_writes']} | {z['sources']} / {z['cells']} | {estimate(z['metrics'].get('E8'),100)} |")
    lines+=['','每个prefix basis只使用本stream同condition/order的先前非零write，当前与未来向量不加入。第一次写为cold start，不能报告有历史共享率；小预算确认流可能根本没有8个历史写。其他rank和clean全部保存在匿名JSON。没有把全流SVD basis回灌在线推理。','',
        '## Cosine and exact isolated first-write utility','',
        f"从已有在线轨迹提取{x['pairs']}个去重donor-recipient-condition pair，原逻辑membership为{x['logical_pair_memberships']}。仅第一次非零commit后、第二次commit前的Before；受体source模型、query=0、输入像素和native时间严格匹配。第二个write到达的Before仍包含，仅其写后退出单donor范围。效用为 U^(1/16)=Before_v−Frozen_v。所有后续累计Before−Frozen均不冒充此量。",'',
        '| Dataset / panel / subset | pairs / defined cos | donors / recipients | corr(c,U) | U(c>0), pp | U(c<0), pp | positive − negative, pp |','|---|---:|---:|---:|---:|---:|---:|']
    for s in u:
        z=s['summary'];mm=z.get('metrics',{});lines.append(f"| {s['dataset']} / {s['split']} / {s['group']} | {z['pairs']} / {z['defined_cosine_pairs']} | {z.get('donors',0)} / {z.get('recipients',0)} | {estimate(mm.get('correlation'))} | {estimate(mm.get('positive_cosine_utility'),100)} | {estimate(mm.get('negative_cosine_utility'),100)} | {estimate(mm.get('positive_minus_negative'),100)} |")
    lines+=['','CI采用10000次source-node bootstrap，同一个source作为donor和recipient时共享重采样计数；每个recipient来源基础等权，条件/order/schedule不视为独立n。受体零proposal有utility但cosine不可定义，未删出coverage。相关性与正负cos分组是预锁诊断，不把受体事后episodic correction用于线上选择。区间条件于当前prefix覆盖与已封存向量，未重拟合basis。','',
        '这些对照只覆盖早期source-prestate donor和其prefix受体，不覆盖任意i→j、晚期共享前态或完整未缩小DeltaLN。单写是实际缓存的因果参数干预，cosine与效用的相关性本身仍不是投影方向的因果消融。','',
        '## Positive and negative cases','',
        '| Dataset / panel | donor → recipient | corruption | cosine | single-write Δv, pp |','|---|---|---|---:|---:|']
    cases=[]
    for ds in ['vidstg','hc2']:
        for sp in ['search','confirm']:
            q=[r for r in pairs if r['dataset']==ds and r['split']==sp and r['condition']!='clean']
            for label,ranked in [('gain',sorted(q,key=lambda r:r['utility_v'],reverse=True)),('harm',sorted(q,key=lambda r:r['utility_v']))]:
                for r in ranked[:3]:
                    cases.append({**r,'tail':label});lines.append(f"| {ds} / {sp} | {r['donor']} → {r['recipient']} | {r['condition']} | {number(r['cosine'])} | {number(r['utility_v'],100)} |")
    write(PUB/'CASES.json',cases)
    lines+=['','所有pair、正负cosine组、固定bins、各donor移除后的影响和clean均完整公开，不只展示极值正例。极值是事后诊断，不作优化目标或路由阈值。原online matched-persistence全数表另附，未重新计分或把本轮几何归因成其原有收益。','',
        '## Decision and reproducibility','',
        '目前当前查询仍使用完整Top1纠正；持续写入保留旧1/16规则。本轮没有证明“完整1/16不是最好”的最优性命题，也没有验证shared/private语义。只有关联和实际投影online matched对照都成立后，才有依据谈selective parameter consolidation。不同optimizer的低秩形状、来源集共同偏差与非零norm都仍可能解释几何。','',
        '本轮读取可信旧torch tensor caches（CPU map），不读取checkpoint、视频、原标注、候选框做新评分；拟合历史已接触的GT-derived匿名metric只作事后join。新参数向量不公开，公开其匿名Gram/范数/谱/投影/配对指标，供独立复算。主对照仍是同域TA-STVG，非新增跨域结果。','',
        f"根审计{read(PUB/'ROOT_AUDIT.json')['checks']}检查，独立rectangular SVD与全部prefix-r8 direct SVD通过；最大投影误差{read(PUB/'ROOT_AUDIT.json')['max_projection_error']:.3g}。公开审计{read(PUB/'PUBLIC_AUDIT.json')['anonymous_Gram_projection_pair_and_bootstrap_checks']}项通过。10个数学/因果前缀合同测试。成本为CPU wall，new model/expert/backward/GT scoring均0。",'',
        '![spectrum](../results/decota_ln_spectrum/2026-10-05/spectrum.png)','',
        '![heldout](../results/decota_ln_spectrum/2026-10-05/heldout_prefix.png)','',
        '![utility](../results/decota_ln_spectrum/2026-10-05/cosine_utility.png)','']
    text='\n'.join(lines);(ROOT/'docs/TA_DECOTA_LN_SPECTRUM_REVIEW.md').write_text(text)
    status(BASE/'STATUS.json',dict(status='completed_CPU_pending_visual_publication',time=time.time(),projected_memory_executed=False))
    archive('完整CPU谱/来源互斥投影/严格前缀及1/16首write效用、独立根/公开审计和报告图完成；'+decision['status']+'，待目检与GitHub核验')
    print('LN_REPORT',decision['status'],decisions,flush=True)

if __name__=='__main__':run()
