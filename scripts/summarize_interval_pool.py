"""Consolidate sealed public scores; no inference, labels, or parameter search."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results/interval_pool'


def read(path):
    return json.loads(path.read_text())


def main():
    reports = {name: read(DEST / name / 'report.json') for name in
               ('intervention_hc64', 'recheck_hc2', 'recheck_vidstg')}
    for report in reports.values():
        assert report['status'] == 'completed'
        assert report['proof']['all_predictions_sealed_before_GT']
        assert report['proof']['official_dense_max_absolute_error'] < 2e-10
    gates = {}
    for ds in ('hc2', 'vidstg'):
        r = reports['recheck_' + ds]
        pair = r['paired']['Mix_MBR-Hull_MBR']
        fixed = r['fixed_spatial_paired']['Mix_MBR-Hull_MBR']
        gates[ds] = dict(tIoU_increased=pair['t']['mean'] > 0,
                         full_vIoU_not_decreased=pair['v']['mean'] >= 0,
                         fixed_space_vIoU_not_decreased=fixed['v']['mean'] >= 0)
    passed = all(all(x.values()) for x in gates.values())
    result = dict(
        status='completed', selected_research_temporal='Mix_MBR' if passed else 'Global6_Hull_MBR',
        mix_passed_preregistered_two_direction_gate=passed, direction_gates=gates,
        production_default='BoxFixed+Strong unchanged',
        comparison='Source is overall baseline; Hull_MBR is temporal component control; Strong historical reference only',
        scope='Development evidence only; historical full-domain exposure prevents independent-confirmation claims',
        decision='Adopt mixture only if both frozen direction point-estimate gates pass; otherwise retain old Global6 and stop, no third weight search',
        extra_networks_for_mixture=0, additional_448_or_conditioned_decoder_calls=0,
        engineering_tests=105,
        score_checks=sum(r['proof']['official_dense_checks'] for r in reports.values()),
        score_max_error=max(r['proof']['official_dense_max_absolute_error'] for r in reports.values()),
        precision_cost_goal='Joint benefit and cost; cheaper posterior construction alone is not an acceptable method gain',
        reports={name: {k: r[k] for k in ('queries', 'parents', 'proof', 'means', 'paired', 'fixed_spatial_paired')}
                 for name, r in reports.items()})
    (DEST / 'decision.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# 两轮时间改进：完整结果与定版', '',
             '**共同目标：轻量且涨点。** Source为整体基线，Global6旧hull+MBR为时间控制，Strong只作历史精度/成本参考。', '',
             f'研究决定：**{result["selected_research_temporal"]}**。生产默认BoxFixed+Strong保持；没有把研究读出静默替换为默认。', '',
             '## 固定实验结果', '',
             '| 面板 | 方法 | 完整vIoU % | tIoU % | 边界MAE 秒 |',
             '|---|---|---:|---:|---:|']
    for name, r in reports.items():
        for method, m in r['means'].items():
            lines.append(f'| {name} | {method} | {100*m["m_vIoU"]:.4f} | {100*m["m_tIoU"]:.4f} | {m["boundary_MAE_seconds"]:.4f} |')
    lines += ['', '## 主比较：Mix_MBR − 原Hull_MBR', '',
              '| 面板 | 完整vIoU pp [95%父配对CI] | tIoU pp [95%父配对CI] | 固定空间vIoU pp | v上升/下降/相同 | >5pp下降 |',
              '|---|---:|---:|---:|---:|---:|']
    for name, r in reports.items():
        p = r['paired']['Mix_MBR-Hull_MBR']; v, t = p['v'], p['t']
        fmt = lambda m: f'{m["mean"]:+.4f} [{m["CI95"][0]:.4f},{m["CI95"][1]:.4f}]'
        lines.append(f'| {name} | {fmt(v)} | {fmt(t)} | {r["fixed_spatial_paired"]["Mix_MBR-Hull_MBR"]["v"]["mean"]:+.4f} | {v["improved"]}/{v["harmed"]}/{v["unchanged"]} | {v["declined_over5pp"]} |')
    lines += ['', '所有匿名逐查询正负、严重退步、阈值失守/恢复、新增漏检和固定空间结果保留在各目录JSON及signed_changes.tsv。主判据是双方向tIoU上升、固定空间和完整v不降；不是挑选某个阈值或只看缓存速度。', '',
              '## 第一轮解释与反证', '',
              '外包期望长度增加的恒等式成立（实测最大误差1.95e−14秒）；但其有效事件质量相对原view平均在tIoU≥.5/.7时分别提高19.8471/9.5888pp，均95%区间高于0。因此，“外包是HC64主要损伤来源”未获支持。不能从长度膨胀直接推出应缩短所有事件。短事件过覆盖更大，长事件欠覆盖更大；终点有符号均值接近0是两类错误抵消。', '',
              '13个预封存读出的离线oracle比Global6有+7.6795pp tIoU余量；这是GT选择的诊断上界，不是可部署成绩，不用于选层、改温度或再扫融合权重。真实混合干预及双向复核决定是否保留方法。完整448后验归因未测，Strong差距不能全部归因448。', '',
              '详见[第一轮](development/error_attribution.md)、[HC64 2×2](intervention_hc64/report.md)、[HC2复核](recheck_hc2/report.md)、[Vid复核](recheck_vidstg/report.md)。', '',
              '## 实际成本与轻量边界', '',
              '| 32查询面板 | RGB均秒 | 共享capture均秒 | Source224均秒 | 专家当前wall均秒 | 原hull构造ms | mix构造ms | 原MBR ms | mix MBR ms |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in ('hc2', 'vidstg'):
        c = reports['recheck_' + ds]['costs']; top, cap, post, br = c['top'], c['capture_cost'], c['posterior_cost'], c['branches']
        lines.append(f'| {ds} | {top["RGB_decode_seconds"]["mean"]:.4f} | {cap["capture_wall_seconds"]["mean"]:.4f} | {cap["Source224_seconds"]["mean"]:.4f} | {cap["expert_current_wall_seconds"]["mean"]:.4f} | {1000*post["hull_construct_seconds"]["mean"]:.4f} | {1000*post["mix_construct_seconds"]["mean"]:.4f} | {1000*br["Hull_MBR"]["MBR_seconds"]["mean"]:.4f} | {1000*br["Mix_MBR"]["MBR_seconds"]["mean"]:.4f} |')
    lines += ['',
              'capture内含Source、ERef、专家及证据构建等，分项不能再次相加。每query仅一次共享224/两个offset、同一ReferDINO八帧一次前向，三臂共用；无448、条件decoder、反向或参数更新。小头投影、搬运、后验、DP、输出IO及Source进程GPU峰值另见完整JSON。首条包含Source/专家延迟加载，均值也包含它；Python进程启动、import、资产哈希及setup在计时外。首条/余下31条成本见engineering.json，均不声称应用完整冷启动。专家子进程峰值显存没有同口径实测，不声称完整峰值显存。', '',
              '本轮64条缺必要缓存，实际新建共享捕获；HC64干预则完全复用缓存。缓存后毫秒、共享推理秒、历史448单例耗时不能混成端到端倍率。混合即使减少后验构造耗时，也必须通过精度判据。', '',
              '## 暴露、锁定及范围', '',
              '32×2按parent/query固定排序，每父一条，先排除当前128面板parent和video SHA，方法/名单先于新GT封存。历史完整划分已有开发曝光，两域可证从未曝光父均为0，故严格称“冻结父隔离开发复核”，不称独立确认。名单规则和匿名证明见[cohort_provenance.json](cohort_provenance.json)。', '',
              f'工程测试105项通过；本轮干预及两域评分共{result["score_checks"]}项官方/dense独立核验，最大差{result["score_max_error"]:.3g}。另第一轮1024项核验见诊断。全部预测先seal后GT，末层logits逐位恢复，主配置及正式暂停状态未改。', '',
              '时间问题的严重性、源码证据、已证/待验机制与低成本改进边界已整理为[完整审计](../../docs/temporal_audit.md)和[原论文核查](../../docs/temporal_sources.md)。固定反馈方向及本轮混合结束后不再扫描；正式Vid8405/10303和CG保持暂停，不覆盖旧源码、预测或结果。']
    (DEST / 'ANALYSIS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'selected_research_temporal', 'mix_passed_preregistered_two_direction_gate', 'direction_gates')}))


if __name__ == '__main__':
    main()
