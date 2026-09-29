"""Report J0 integration and freeze the user-selected research recipe if supported."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,status
from scripts.audit_tastvg_joint_j0_public_v1 import run as public_audit
OUT=ROOT/'artifacts/tastvg_joint_j0_v1';METHOD=ROOT/'methods/tastvg_dual_evidence_j0_v1'


def run():
    s=read(OUT/'SUMMARY.json');r=read(OUT/'RESOURCES.json');a=read(OUT/'AUDIT.json');p=read(OUT/'LOCK.json');pub=public_audit(OUT);write(OUT/'PUBLIC_AUDIT.json',pub)
    def metric(g,sub,k):return s[g][sub]['metrics'][k]
    def fmt(g,sub,k,ci=True):
        x=metric(g,sub,k);t=f"{100*x['mean']:+.6f}";return t+(f" [{100*x['ci95'][0]:+.6f}, {100*x['ci95'][1]:+.6f}]" if ci else '')
    positive=metric('corruption','nonexpert','final_minus_fast_v')['mean']>0 and metric('corruption','nonexpert','final_minus_fast_s')['mean']>0 and metric('corruption','all','final_minus_frozen_v')['mean']>0
    decision=dict(status='completed',measurement='valid',research_recipe_frozen=positive,criterion='user requested positive future Final-Fast and positive whole-stream Final-Frozen, no synergy or new magnitude threshold',mechanism_development='stopped',selected_temporal='native current-policy candidates + UniversalVTG rerank at scheduled arrivals',selected_spatial='Rank-RKL + one plain SGD.005, pre-update spatial output, persistent1792D parameters',closed_routes=['normalized-step','temperature/LR sweep','extra steps','momentum','margin gate','temporal OPD','parameter-space OPD','support expansion','new expert','corruption change'],new_fresh_or_large_evaluation=False,production_CURRENT_METHOD_changed=False,integration_not_independent_transfer_replication=True,time=time.time());write(OUT/'DECISION.json',decision)
    dep=dict(p['pins'])
    if (OUT/'IMPLEMENTATION_REVISION.json').exists():
        rev=read(OUT/'IMPLEMENTATION_REVISION.json');dep[rev['path']]=rev['revised_sha256']
    for f in ['vg_tta/tastvg_evidence_capture_v1.py','vg_tta/tastvg_causal_round2_v1.py','vg_tta/tastvg_spatial_critic_s06_v1.py','methods/decota_final_simplified_v1/backbone.py','methods/decota_final_simplified_v1/objectives.py','methods/decota_final_simplified_v1/tensors.py']:dep[f]=sha(ROOT/f)
    prov=dict(backbone='TA-STVG',source_checkpoint_sha256=read(METHOD/'config.json')['source_checkpoint_sha256'],source_dataset='VidSTG',target='VidSTG original exposed16 parent sources, onequery each',sources=16,queries=16,conditions=6,arrivals_per_arm=96,arms=4,lock_sha256=sha(OUT/'LOCK.json'),effective_dependency_hashes=dep,input_barrier_hashes=p['input_hashes'],prediction_barrier_sha256=sha(OUT/'PREDICTION_BARRIER.json'),revision_sha256=sha(OUT/'IMPLEMENTATION_REVISION.json') if (OUT/'IMPLEMENTATION_REVISION.json').exists() else None,expert_cache_reuse=True,production_unchanged=True);write(OUT/'PROVENANCE.json',prov)
    write(METHOD/'RECIPE_LOCK.json',dict(status='components_and_implementation_locked',J0_whole_stream_criterion_met=positive,validated_final_method=positive,config_sha256=sha(METHOD/'config.json'),dependency_hashes=dep,decision_sha256=sha(OUT/'DECISION.json'),method_optimization_stopped=True,production_registration_changed=False,time=time.time()))
    if positive:
        write(METHOD/'FREEZE.json',dict(status='research_recipe_frozen',version='tastvg_dual_evidence_j0_v1',entrypoint='methods.tastvg_dual_evidence_j0_v1.OnlineMethod',config_sha256=sha(METHOD/'config.json'),dependencies=dep,decision_sha256=sha(OUT/'DECISION.json'),summary_sha256=sha(OUT/'SUMMARY.json'),protocol_sha256=sha(ROOT/'protocols/tastvg_joint_j0_v1.md'),scope='Recipe freeze after exposed-cohort J0 integration. Not fresh-evaluation validation or historical production promotion.',next_phase='Evaluate fixed method on larger/fresh cohorts and streams when separately scoped; no component optimization.',time=time.time()))
    lines=['# J0 — Final Integrated Online Method','',f"Final-method success criterion met: **{positive}**. Component choices and implementation are locked; no further mechanism rescue. Fixed Fast temporal critic reranking + Slow persistent spatial Rank-RKL. Four arms, same25%availability positions; no added module or tuning.",'','## Whole-stream four-arm comparison','', 'Metrics in percent. For corruption, average five conditions within each parent then macro over16sources/80cells. Clean16cells shown separately.','', '| Arm | Corruption sIoU | Corruption tIoU | Corruption vIoU | Clean sIoU | Clean tIoU | Clean vIoU |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,name in [('frozen','Frozen'),('fast','Fast-only'),('slow','Slow-only'),('final','Final')]:lines.append('| '+name+' | '+' | '.join(f"{100*metric(g,'all',arm+'_'+m)['mean']:.6f}" for g in ['corruption','clean'] for m in ['s','t','v'])+' |')
    lines+=['','## Prespecified comparisons','','All differences in percentage points with source-bootstrap95% CI,10000resamples seed20260929. Intervals conditional on this short fixed stream and development cohort.','', '| Comparison / subset | Corruption ΔsIoU | Corruption ΔtIoU | Corruption ΔvIoU |','|---|---:|---:|---:|']
    for key,sub in [('fast_minus_frozen','expert'),('final_minus_fast','nonexpert'),('final_minus_frozen','all'),('final_minus_fast','all'),('final_minus_slow','expert'),('interaction','all')]:lines.append('| '+key+' / '+sub+' | '+' | '.join(fmt('corruption',sub,key+'_'+m) for m in ['s','t','v'])+' |')
    lines+=['','Expert subset is20cells/4sources; nonexpert60cells/12sources; whole stream80cells/16sources. The three initial nonexpert arrivals before first nonempty spatial evidence remain included. Small expert source count makes immediate-benefit intervals especially limited.','',
    '**Integration identity:** Final and Slow-only share identical spatial state and pre-update central boxes. Temporal reranking is readout-only and only active on scheduled arrivals. Consequently, Final−Fast at nonexpert arrivals is exactly the previously measured S1.1 Rank−Frozen effect. J0 verifies correct composition and whole-stream utility; it is not independent new evidence of transfer, nor a demonstration of positive interaction/synergy.','',
    '## Clean control and per-corruption future transfer','', '| Condition | Final−Fast nonexpert ΔsIoU | Final−Fast nonexpert ΔvIoU | Final−Frozen all ΔvIoU |','|---|---:|---:|---:|']
    for g in p['conditions']:lines.append('| '+g+' | '+fmt(g,'nonexpert','final_minus_fast_s')+' | '+fmt(g,'nonexpert','final_minus_fast_v')+' | '+fmt(g,'all','final_minus_frozen_v')+' |')
    lines+=['','Clean control shows whether benefits are general refinement; the measured gains do not establish corruption-specific recovery. All tested conditions use the original fixed seed0 burst pixels, not a stronger benchmark.','',
    '## Current output and state invariants','',
    'At every arrival, run the current spatial policy. At expert arrivals, derive up to8native temporal candidates under those same current parameters, use cached frozen UniversalVTG proposals as scalar evidence, and output one student interval. Preserve current pre-update spatial boxes. Separately regenerate9spatial probes, use Sa2VA rewards only to rank them, and apply one SGD.005 reverse-KL update to1792parameters. Carry the update to future arrivals. At nonexpert arrivals, neither expert is read and no update occurs.',
    '',f"Final execution is exact to the sealed Slow-only control for all192pre/post states and {a['Final_Slow_exact_gradient_coordinates']}gradient coordinates. All96reported spatial outputs match pre-update central boxes; {a['updated_post_boxes_differ_from_reported_pre_boxes']}/18updated post boxes differ and are excluded from current arm outputs. {a['current_temporal_support_changed_cells']}/24scheduled current temporal supports differ from frozen supports; {a['current_temporal_selected_interval_changed_cells']}/24selected current intervals differ from Fast-only. This is measured, not an assumption that spatial learning cannot affect temporal candidates.",
    '',f"Independent audit: {a['independent_SGD_coordinates']}SGD coordinates, {a['independent_KL_updates']}KL reconstructions, {a['independent_temporal_scores']}temporal candidate scores, {a['dual_metric_calls']}dual-metric calls,4learned full-output reinsertions and2learned all-six-layer temporal reinsertions.3CPU contracts and{pub['scalar_checks']}public scalar checks passed. All four-arm predictions sealed before reading the same16previously exposed GT keys.",
    '', '## Specialist budget and resources','', '| Arm | Temporal logical calls | Spatial logical calls | New expert inference in this cached run |','|---|---:|---:|---:|']
    for arm,counts in r['logical_specialist_calls'].items():lines.append(f"| {arm} | {counts['temporal']} | {counts['spatial']} | 0 |")
    lines+=['','The25%availability schedule is shared. Total calls differ: Final uses both experts on24arrivals, single-branch arms one expert. Final vs Fast has the same temporal budget but additional past spatial evidence. Cache reuse saves execution cost; it does not imply equal deployment latency/cost.',
    '',f"New GPU process time {r['new_GPU_process_seconds']:.6f}s, including {r['GPU_failed_attempts']}retained failed attempt. New96Final arrivals,216spatial probes,18backward steps; Frozen/Fast/Slow288control arrivals reuse compatible sealed predictions or frozen candidates. No new encoder capture, expert inference or model download. Initial attempt failed at first CPU-vs-CUDA baseline comparison before any saved arrival or effective update. Validation now compares on CPU; original runner/log/lock retained with append-only IMPLEMENTATION_REVISION. No change to method or evaluator.",
    '', '## Decision and scope','',
    'Freeze the research recipe if the prespecified signs above hold; no giant spatial gain or synergy required. Close Norm, temporal OPD, parameter-space OPD, extra support, gates and tuning. Preserve all historical positive/negative results. Recipe configuration and implementation hashes are frozen separately from the historical CURRENT_METHOD production registration. Larger/fresh same-domain corruption evaluation, clean control, secondary cross-domain and alternate streams are the next evaluation phase, not executed by this bounded J0 experiment.',
    '', 'Data and uncertainty: same16repeatedly exposed VidSTG parent sources, onequery each, original Vid-source TA checkpoint5ab12c86, fixed six short streams; main future12sources, immediate expert4sources. This result supports the selected asymmetric recipe on the measured development setting. It does not establish a universal temporal-selection/spatial-adaptation dichotomy or general necessity of constraining specialists to student support.','']
    diagnosis=read(OUT/'TEMPORAL_DIAGNOSIS.json');cg=diagnosis['groups']['corruption']
    lines+=['## Why the sparse Fast arm differs from the prior positive temporal result','',
    'Read-only partition of already scored C3 results on exactly the same16sources and five5%corruptions. This is a historical full-availability reference, not an added J0 arm or a new experiment.','',
    '| Historical temporal rerank subset | Sources | ΔtIoU (pp) | ΔvIoU (pp) |','|---|---:|---:|---:|']
    for subset in ['all','scheduled','unscheduled']:
        z=cg[subset];lines.append(f"| {subset} | {z['sources']} | {100*z['metrics']['historical_full_rerank_gain_t']['mean']:+.6f} | {100*z['metrics']['historical_full_rerank_gain_v']['mean']:+.6f} |")
    lines+=['', 'The locked quarter of arrivals has negative vIoU reranking gain even though the historical full16source reference is positive. In corruption source means: Q01 gains7.965456pp tIoU but zero vIoU; Q05 loses.828373pp vIoU; Q09 loses.327039pp; Q13 gains.003339pp. The unavailable12sources contain the large historical temporal gains, but J0 correctly does not read their specialists. Candidate vIoU oracle headroom on scheduled sources remains+3.585205pp; this is offline GT diagnosis, not deployable benefit. The result does not uniquely establish a bad random schedule or a universal critic failure. Schedule, expert and candidate family remain unchanged.',
    '', '**Decision:** the positive future-transfer comparison passes, but whole-stream Final−Frozen does not meet the requested positive-sign criterion. Do not label J0 as a successful final-method freeze. Preserve the specified component recipe and stop mechanism optimization; do not choose a favorable schedule with these labels or silently switch to100%expert coverage.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    zh=f'''# J0整合结果：Fast temporal correction + Slow spatial Rank-OPD

本轮最终方法通过并冻结的条件满足：{positive}。组件选择与实现已锁定，但该状态不等于全流效用通过。四臂采用原16曝光来源、六条件、同25%到达位置。时间固定native候选＋UniversalVTG rerank；空间固定1792维持久参数＋9个antithetic probes＋rank teacher＋reverse-KL一步SGD.005。当前空间输出始终为更新前central policy，更新只用于之后样本。

## 四臂全流主表

单位%，corruption先五条件源内平均再16source macro。

| Arm | sIoU | tIoU | vIoU |
|---|---:|---:|---:|
'''
    for arm,name in [('frozen','Frozen'),('fast','Fast-only'),('slow','Slow-only'),('final','Final')]:zh+='| '+name+' | '+' | '.join(f"{100*metric('corruption','all',arm+'_'+m)['mean']:.6f}" for m in ['s','t','v'])+' |\n'
    zh+=f'''
## 两个主要比较

- 当前expert的Fast-only−Frozen：ΔtIoU {fmt('corruption','expert','fast_minus_frozen_t')}pp；ΔvIoU {fmt('corruption','expert','fast_minus_frozen_v')}pp。专家子集只有4来源/20corruption cell。
- 未来nonexpert的Final−Fast-only：ΔsIoU {fmt('corruption','nonexpert','final_minus_fast_s')}pp；ΔvIoU {fmt('corruption','nonexpert','final_minus_fast_v')}pp。12来源/60corruption cell。
- 全流Final−Frozen：ΔvIoU {fmt('corruption','all','final_minus_frozen_v')}pp。16来源/80corruption cell。
- 全流Final−Fast-only：ΔvIoU {fmt('corruption','all','final_minus_fast_v')}pp。
- Clean future Final−Fast：Δs {fmt('clean','nonexpert','final_minus_fast_s')}pp、Δv {fmt('clean','nonexpert','final_minus_fast_v')}pp。

区间为固定轨迹下的source bootstrap95% CI；不同corruption不作为独立来源。Clean收益相近，仍属于一般修正证据，不能说已独立验证corruption专项鲁棒性。

## 集成确实按online语义工作

新Final完整跑96到达/18更新。192个pre/post状态、32256梯度坐标与旧Slow-only逐项精确一致；当前temporal候选从已适应参数产生，{a['current_temporal_support_changed_cells']}/24组与Frozen候选不同，最终选择区间{a['current_temporal_selected_interval_changed_cells']}/24与Fast-only不同。所有96个当前空间输出均使用pre-update，18个更新后的空间预测保留作审计、不进入当前Final成绩。4次学习后完整回插＋2次全部六层temporal回插、384双metric、3CPU及{pub['scalar_checks']}公开scalar检查通过。

这里有一个必须明确的比较关系：Fast只在expert到达修改输出，不改后续参数，所以nonexpert上Final等于Slow-only，Fast-only等于Frozen。J0的future增量因此精确复现S1.1；这轮新增的是组合实现正确性和全流效果，不是独立新来源上的再次确认，也不是强协同效应。

## 调用预算与资源

同25%availability：Frozen 0；Fast-only 24次时间expert；Slow-only 24次空间expert；Final 24次时间＋24次空间。总specialist调用数不相等。Final−Fast隔离的是额外历史空间反馈经参数传递的效益，不是相同总专家成本下的比较。本轮全部复用像素/query一致的冻结expert缓存，新expert调用0。

GPU进程累计{r['new_GPU_process_seconds']:.6f}秒，包含首个工程校验失败尝试；216新spatial候选/18反向、0新encoder capture。首次因CPU/GPU预测直接比较而报错，0到达保存/0有效更新；原日志与runner保留，仅修校验设备并追加版本记录，方法和评分不变。

## 决定

按本轮预定结果分支，研究方法冻结状态为{positive}。保留非对称设计：**Select temporally, probe spatially, and distill ranked on-policy evidence.** 不再做Norm、temporal OPD、parameter-space OPD、LR/temperature/multiple-step或candidate扩展。正式研究recipe单独保存配置和代码hash，历史生产CURRENT_METHOD不被静默替换。

之后工作进入固定方法评价阶段：更大/未曝光来源、same-domain corruption主setting、clean对照、cross-domain辅助及online stream/order稳定性。本轮没有启动这些额外评估，也没有把这批已曝光短流当作fresh结果。历史O2等也有弱online信号，不能把J0写成所有online实验的首次正迁移。
'''
    zh+='''
## 为什么旧时间正结果没有出现在J0全流

只读回查同16来源、同五种5%corruption的旧C3缓存成绩：若每次到达都允许时间专家，Rerank−Frozen为Δt+6.130562/Δv+3.156181pp。按J0固定schedule分组，实际可用的4来源为Δt+1.334720/Δv−0.288018pp；不可用的12来源旧参考为Δt+7.729176/Δv+4.304248pp。四来源的负均值乘25%得到J0 Fast全流Δv−0.072005pp。它解释了与历史全专家结果的差异，不能据此重选更有利的schedule。

具体保留四个来源：Q01的Δt+7.965456pp但Δv0；Q05 Δv−0.828373pp；Q09 −0.327039pp；Q13 +0.003339pp。可用来源的candidate vIoU oracle仍有+3.585205pp，但这是离线GT上限，不能拿来替代实际critic选择。当前负结果不能唯一归因schedule运气，也不能普遍否定Temporal critic。

**本轮结论：future transfer仍为正，然而corruption全流Final−Frozen为−0.036318pp，CI[−0.164640,+0.056689]，没有达到用户预设的正向条件。** 不能宣布J0成功或最终方法效果已验证。按用户意图停止机制救援，封存当前非对称组件与整合代码；没有扩大专家覆盖率、按GT更换schedule、删除负例或追加模块。是否在更大/不同独立流上确认稀疏Fast效应属于之后的固定方法评价范围，本轮未自行启动。
'''
    (OUT/'RESEARCH_UPDATE.md').write_text(zh)
    (OUT/'REPRODUCE.md').write_text('''# Reproduction

Run from the repository root with the authorized original private models/caches and existing environment. Public artifacts omit media/query IDs/GT boxes/H/masks/prediction tubes/gradients/weights.

```bash
.conda/tubedetr/bin/python -B -m pytest -q tests/test_tastvg_joint_j0_v1.py
.conda/tubedetr/bin/python -B scripts/run_tastvg_joint_j0_v1.py prepare
bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/run_tastvg_joint_j0_v1.py run
.conda/tubedetr/bin/python -B scripts/score_tastvg_joint_j0_v1.py
.conda/tubedetr/bin/python -B scripts/report_tastvg_joint_j0_v1.py
```

Fresh runs lock the corrected runner directly. Existing original LOCK and initial failure are retained with IMPLEMENTATION_REVISION. Artifacts are write-once; use a new run directory for independent repetition. Do not overwrite saved state/labels/results. CPU public-scalar verification, no private models or data:

```bash
python scripts/audit_tastvg_joint_j0_public_v1.py results/tastvg_joint_j0/2026-09-29
```

The package OnlineMethod accepts native post-encoder cached data, fixed probe deltas and lazy temporal/spatial evidence providers; use reset only between independent streams and close to restore source parameters. H caching is valid because encoder parameters never change. Freeze covers recipe, not arbitrary untested runtime/cohort compatibility.
''')
    write(OUT/'COMPLETION.json',dict(status='completed',research_recipe_frozen=positive,prediction_cells=96,arm_predictions=384,updates=18,public_scalar_checks=pub['scalar_checks'],publication='pending',time=time.time()))
    print('RECIPE_FROZEN',positive)

if __name__=='__main__':run()
