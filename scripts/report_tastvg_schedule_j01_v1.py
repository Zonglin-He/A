"""Report every prelocked order and freeze only the existing recipe, without selecting an order."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.audit_tastvg_schedule_j01_public_v1 import run as audit
OUT=ROOT/'artifacts/tastvg_schedule_j01_v1';METHOD=ROOT/'methods/tastvg_dual_evidence_j0_v1'

def run():
    p=read(OUT/'LOCK.json');s=read(OUT/'SUMMARY.json');a=read(OUT/'ACROSS_ORDERS.json');old=read(OUT/'J0_REFERENCE.json');orders=read(OUT/'ORDERS.json');tier=read(OUT/'TIER0.json');res=read(OUT/'RESOURCES.json');check=read(OUT/'AUDIT.json')
    pub=audit(OUT);write(OUT/'PUBLIC_AUDIT.json',pub)
    fast=a['corruption']['all']['fast_minus_frozen_v'];final=a['corruption']['all']['final_minus_frozen_v'];future=a['corruption']['nonexpert']['final_minus_fast_v'];future_s=a['corruption']['nonexpert']['final_minus_fast_s']
    freeze=fast['positive']>=3 and final['mean']>0
    decision=dict(status='completed',branch='A_schedule_dependence' if freeze else 'B_sparse_temporal_robustness_unresolved',research_recipe_frozen=freeze,positive_Fast_orders=fast['positive'],negative_Fast_orders=fast['negative'],mean_whole_Final_gain=final['mean'],positive_future_spatial_orders=future['positive'],negative_future_spatial_orders=future['negative'],mean_future_spatial_gain=future['mean'],spatial_robustness_over_tested_orders='all_positive' if future['positive']==5 else 'mixed_or_nonpositive',best_order_selected=False,method_code_or_hyperparameter_changed=False,new_gate=False,new_large_or_fresh_evaluation=False,production_changed=False,scope='Five fixed hash orders on the same16exposed sources. Recipe freeze is a resource/implementation decision, not proof of population generalization or universal future transfer.',time=time.time());write(OUT/'DECISION.json',decision)
    prov=dict(backbone='TA-STVG',source_checkpoint_sha256=read(METHOD/'config.json')['source_checkpoint_sha256'],sources=16,queries=16,new_orders=5,conditions=6,stream_length=16,cells=480,prior_J0_reference_separate=True,lock_sha256=sha(OUT/'LOCK.json'),order_manifest_sha256=sha(OUT/'ORDERS.json'),method_pins_unchanged=p['pins'],input_barrier_hashes=p['inputs'],prediction_barrier_sha256=sha(OUT/'PREDICTION_BARRIER.json'));write(OUT/'PROVENANCE.json',prov)
    if freeze:
        write(METHOD/'FREEZE_J01.json',dict(status='research_recipe_frozen_for_subsequent_evaluation',version='tastvg_dual_evidence_j0_v1',method_code_sha256=sha(METHOD/'method.py'),config_sha256=sha(METHOD/'config.json'),original_recipe_lock_sha256=sha(METHOD/'RECIPE_LOCK.json'),decision_sha256=sha(OUT/'DECISION.json'),order_manifest_sha256=sha(OUT/'ORDERS.json'),best_order_selected=False,spatial_future_positive_orders=future['positive'],spatial_future_mean=future['mean'],scope='Keep exact J0 recipe; no gate/new mechanism. Five-order exposed-cohort schedule evidence is not fresh efficacy confirmation; retain all positive and negative component outcomes.',production_changed=False,time=time.time()))
    def val(order,sub,key,group='corruption'):return s[order][group][sub]['metrics'][key]['mean']*100
    def describe(st):return f"{st['mean']*100:+.6f} ± {st['sample_std']*100:.6f}"
    lines=['# J0.1 — Online Schedule Robustness','',f"Five source-hash orders locked before execution. Exact J0 method code and hyperparameters unchanged. Fast positive orders: {fast['positive']}/5. Research recipe freeze for subsequent evaluation: **{freeze}**.",'', '## Primary five-order comparison','', 'Differences in percentage points. Corruption: average five conditions within source, then16source whole-stream or12source nonexpert macro. Across-order row uses equal order weights and sample SD(ddof1), not a confidence interval.','', '| Order | Whole Fast−Frozen vIoU | Whole Final−Fast vIoU | Whole Final−Frozen vIoU | Future Final−Fast sIoU | Future Final−Fast vIoU |','|---|---:|---:|---:|---:|---:|']
    for o in p['orders']:lines.append('| '+o+' | '+' | '.join(f"{val(o,sub,k):+.6f}" for sub,k in [('all','fast_minus_frozen_v'),('all','final_minus_fast_v'),('all','final_minus_frozen_v'),('nonexpert','final_minus_fast_s'),('nonexpert','final_minus_fast_v')])+' |')
    lines.append('| Five new orders: mean ± SD | '+' | '.join(describe(a['corruption'][sub][k]) for sub,k in [('all','fast_minus_frozen_v'),('all','final_minus_fast_v'),('all','final_minus_frozen_v'),('nonexpert','final_minus_fast_s'),('nonexpert','final_minus_fast_v')])+' |')
    lines.append('| Original J0, separate reference | '+' | '.join(f"{old['corruption'][sub]['metrics'][k]['mean']*100:+.6f}" for sub,k in [('all','fast_minus_frozen_v'),('all','final_minus_fast_v'),('all','final_minus_frozen_v'),('nonexpert','final_minus_fast_s'),('nonexpert','final_minus_fast_v')])+' |')
    lines+=['',f"Fast whole-stream sign counts: {fast['positive']}positive/{fast['negative']}negative/{fast['zero']}zero. Final whole-stream: {final['positive']}positive/{final['negative']}negative. Spatial future vIoU: {future['positive']}positive/{future['negative']}negative; mean{future['mean']*100:+.6f}pp, SD{future['sample_std']*100:.6f}pp. Spatial future sIoU: {future_s['positive']}positive/{future_s['negative']}negative. Do not infer stable spatial transfer from positive combined Fast-dominated performance.",'',
    'The original order is not included in the primary five-order mean. INCLUDING_J0.json separately reports all six including the earlier unfavorable stream; no result was discarded, no best-order selection. Orders share the same16sources and expert caches, so these are descriptive order variations, not five independent cohorts. Per-order conditional source-bootstrap intervals are in SUMMARY.json.','',
    '## Locked specialist availability','', '| Order | Expert source aliases, in arrival order |','|---|---|']
    for o,x in orders['orders'].items():lines.append('| '+o+' | '+', '.join(x['expert_sources'])+' |')
    lines+=['','Hash rule: sort original source IDs by SHA256(UTF8("J01-source-order-20260929|j|source")),then source, j=1..5. Same order used across all six conditions, expert positions0/4/8/12. No rejection/resampling based on data or scores. Anonymous complete sequences are in ORDERS.json; reproducing original source-ID hashes requires authorized private roster.','', '## Clean control','', '| Order | Whole Fast−Frozen vIoU | Whole Final−Frozen vIoU | Future Final−Fast sIoU | Future Final−Fast vIoU |','|---|---:|---:|---:|---:|']
    for o in p['orders']:lines.append('| '+o+' | '+' | '.join(f"{val(o,sub,k,'clean'):+.6f}" for sub,k in [('all','fast_minus_frozen_v'),('all','final_minus_frozen_v'),('nonexpert','final_minus_fast_s'),('nonexpert','final_minus_fast_v')])+' |')
    subset=tier['subset_distribution'];margin=tier['margin_descriptive']
    lines+=['','## Tier0: exact finite-roster subset and margin audit','',
    f"All1820four-of16historical corruption source subsets enumerated. Original J0 selected-subset gain{subset['current_subset_v_gain']*100:+.6f}pp; inclusive percentile{subset['current_percentile_inclusive']:.6f}%,strict percentile{subset['current_percentile_strict']:.6f}%. Negative subsets{subset['negative_count']}/1820={subset['negative_fraction']*100:.6f}%.",
    '', 'Selected-subset mean gain quantiles (pp): '+', '.join(f"{float(q)*100:g}%={v*100:+.6f}" for q,v in subset['quantiles'].items())+'.',
    '', 'These are four-selected-source means; whole16arrival Fast gain is one quarter of each. The1820overlapping subsets do not estimate a population deployment failure rate or repeat persistent adaptation. The five actual online orders above are separately executed.',
    '', f"Pooled-cell margin/gain Pearson: all96={margin['all96']['pearson']:.6f},scheduled24={margin['scheduled24']['pearson']:.6f}. Scheduled upper-quartile margin subset gain={margin['scheduled24']['upper_quartile_v_gain']*100:+.6f}pp ({margin['scheduled24']['upper_quartile_cells']}cells). Purely descriptive, repeated source/conditions; no margin/entropy/confidence threshold deployed.",
    '', '## Execution, validity and costs','',
    f"480new Final arrivals across30source-reset streams; {check['independent_KL_updates']}actual spatial updates and{res['main_spatial_candidates']}current-policy spatial rollouts. Encoder/experts reused, new expert inferences0. Persistent states recomputed for every order; old S1/J0 learned states were not reused. Independent order1/clean Slow-only replay matches all32pre/post states and outputs, with{res['validation_backward_steps']}extra validation backward steps.",
    '',f"State links480/resets30; {check['independent_SGD_coordinates']}independent SGD coordinates, {check['independent_KL_updates']}rankKL reconstructions, {check['independent_temporal_scores']}independent temporal scores, {check['dual_metric_calls']}dual metrics;4full spatial and2all-six-layer temporal native reinsertions;3order contracts;{pub['scalar_checks']}public scalar checks. Current temporal support changed in{check['current_temporal_support_changed_cells']}/120expert cells; selected interval changed in{check['current_temporal_selected_interval_changed_cells']}/120. Nonexpert callbacks never read specialists; current spatial output always pre-update.",
    '',f"GPU process total{res['new_GPU_process_seconds']:.6f}s, failed attempts{res['GPU_failed_attempts']}. No downloads, new expert inference or encoder capture. Per order including clean: Fast24T,Slow24S,Final24T+24S logical calls; all share25%availability, not equal total specialist cost. Cached runtime is not uncached deployment latency.",
    '', '## Decision boundaries','',
    f"User branch outcome: {decision['branch']}. Method recipe freeze={freeze}; no order selected for deployment. Preserve the original J0 negative and all new order results. If Fast is mostly positive, the evidence supports schedule-dependent temporal performance on this roster, not universal sparse-temporal failure. It does not imply each schedule is safe. Spatial future gains are separately reported; any mixed/negative order results prevent calling online transfer uniformly stable.",
    '', 'Do not add a gate, rerun favorable schedules, change Spatial or restart temporal/parameter-space OPD. Larger/fresh/order stability evaluation remains a subsequent scoped phase; historical production CURRENT_METHOD unchanged. Repeated exposure and the same source corpus limit generalization.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    zh=['# J0.1：固定方法的五顺序在线鲁棒性','',f"5个source-hash顺序全部预锁并实际执行。方法代码/超参数未改。Fast正向{fast['positive']}/5，Final全流正向{final['positive']}/5；按用户分支，研究配方冻结状态为{freeze}。",'', '## 五个新顺序的corruption结果','', '单位pp；全流每序16来源，future nonexpert每序12来源。均值/SD对五个预锁顺序等权，不挑最佳序。','', '| Order | 全流Fast−Frozen v | 全流Final−Fast v | 全流Final−Frozen v | Future Δs | Future Δv |','|---|---:|---:|---:|---:|---:|']
    for o in p['orders']:zh.append('| '+o+' | '+' | '.join(f"{val(o,sub,k):+.6f}" for sub,k in [('all','fast_minus_frozen_v'),('all','final_minus_fast_v'),('all','final_minus_frozen_v'),('nonexpert','final_minus_fast_s'),('nonexpert','final_minus_fast_v')])+' |')
    zh+=['| 均值±SD | '+' | '.join(describe(a['corruption'][sub][k]) for sub,k in [('all','fast_minus_frozen_v'),('all','final_minus_fast_v'),('all','final_minus_frozen_v'),('nonexpert','final_minus_fast_s'),('nonexpert','final_minus_fast_v')])+' |','',
    f"原J0全流Fast−Frozen−0.072005pp、Final−Frozen−0.036318pp继续保留，单列历史参照；主均值不把它混成新预锁序，六序合并另外保存INCLUDING_J0。五序复用同16曝光来源，不能说5个独立数据集或480个独立来源。",
    '',f"空间未来Δv正{future['positive']}序/负{future['negative']}序，均值{future['mean']*100:+.6f}pp、SD{future['sample_std']*100:.6f}pp；Δs正{future_s['positive']}序/负{future_s['negative']}序。Fast主导的全流正不能替代空间迁移稳定性的证据，不因先前J0/S1.1正值而省略新的负序。",
    '', '## 附件Tier0数字独立复算','',
    f"全部1820个4/16来源子集：原J0−0.288018pp，百分位{ subset['current_percentile_inclusive']:.4f}%（<=口径）；负子集{subset['negative_count']}/1820，即{subset['negative_fraction']*100:.4f}%。四来源均值的5/25/50/75/95分位为"+' / '.join(f"{x*100:+.4f}" for x in subset['quantiles'].values())+'pp。要转成16到达全流收益需再乘25%。这不是1820次独立实验，也不是总体部署失败概率。',
    '',f"margin与v gain相关：全96cell r={margin['all96']['pearson']:.4f}，原scheduled24 r={margin['scheduled24']['pearson']:.4f}；scheduled上四分位margin的均值Δv={margin['scheduled24']['upper_quartile_v_gain']*100:+.4f}pp。只做缓存描述，不加gate。",
    '', '## 执行与结论范围','',
    f"30条流/480新到达/{check['independent_KL_updates']}实际空间写入；{res['main_spatial_candidates']}新current-policy空间候选，0新expert/encoder采集。GPU进程{res['new_GPU_process_seconds']:.3f}秒，另含固定order1/clean独立Slow复验的16到达/{res['validation_backward_steps']}反向。480状态链、{check['independent_SGD_coordinates']}SGD坐标、{check['dual_metric_calls']}双metrics、4完整空间/2六层temporal回插、3CPU和{pub['scalar_checks']}公开scalar通过。J0方法/config与依赖hash全程保持，未复制旧learned states冒充新order更新。",
    '',f"本轮分支：{decision['branch']}。记录并保留五序均值与方差，不挑winner、不调schedule，不新建margin/reliability gate，不改变Spatial或OPD。冻结的是现有研究配方供之后固定方法评价；不是宣称所有顺序都安全或在未曝光数据上已验证。下一larger/fresh评估本轮未启动，生产CURRENT_METHOD不改。"]
    (OUT/'RESEARCH_UPDATE.md').write_text('\n'.join(zh)+'\n')
    (OUT/'REPRODUCE.md').write_text('''# Reproduction

Use authorized original models, source roster and cached H/expert evidence. Existing method code must match frozen J0 hashes.

1. Run tests/test_tastvg_schedule_j01_v1.py with pytest.
2. Run scripts/run_tastvg_schedule_j01_v1.py prepare to seal source-hash orders.
3. Run the same runner with run via bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B.
4. After all predictions seal, run scripts/score_tastvg_schedule_j01_v1.py and scripts/tier0_tastvg_schedule_j01_v1.py.
5. Run scripts/report_tastvg_schedule_j01_v1.py.

Write-once artifacts: preserve old runs and use a new destination for independent repeats. Never resample orders according to metrics. Public scalar-only audit needs Python/NumPy:

    python scripts/audit_tastvg_schedule_j01_public_v1.py results/tastvg_schedule_j01/2026-09-29

Public roster uses Q aliases; original source-hash regeneration requires the authorized private roster. Scalar aggregation and order membership checks need no weights, labels, media or raw states.
''')
    write(OUT/'COMPLETION.json',dict(status='completed',orders=5,cells=480,research_recipe_frozen=freeze,public_checks=pub['scalar_checks'],publication='pending',time=time.time()))
    print(decision)

if __name__=='__main__':run()
