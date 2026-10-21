"""Summarize already-sealed tuning/recheck results; no model or label access."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'results/boxfixed_tuning_v1'

def main():
    read=lambda p:json.loads(p.read_text())
    selected=read(DEST/'selected.json');profile=read(ROOT/'configs/tuned_boxfixed_v1.json');full=read(ROOT/'outputs/boxfixed_full_v1/plan.json')
    trials={ds:[read(p)for p in sorted((DEST/ds).glob('*.json'))]for ds in ('hc2','vidstg')}
    confirmations={ds:read(DEST/f'confirmation_{ds}.json')for ds in trials}
    assert all(len(rows)<=18 and all(r['proof']['predictions_sealed_before_GT']for r in rows)for rows in trials.values())
    full_complete=all((ROOT/'outputs/boxfixed_full_v1'/ds/'score_receipt.json').exists()for ds in trials)
    full_state=read(ROOT/'outputs/boxfixed_full_v1/state.json')['status']
    summary=dict(status='tuning_and_frozen_recheck_completed',full_status_at_export=full_state,full_scored=full_complete,queries_per_development_direction=64,
        actual_trials={ds:len(rows)for ds,rows in trials.items()},selected_development=selected,final_profile=profile,
        engineering_tests=113,cache_regression=read(DEST/'cache_regression.json'),
        full_queue=dict(fingerprint=full['fingerprint'],jobs=full['jobs'],method='BoxFixed+Global6+old hull+original MBR; original parameters after rejected transfer',
            no_training_or_OPD=True,no448_or_conditional_decoder=True,old_formal_and_CG_queues_untouched=True),
        trials={ds:[{k:r[k]for k in ('name','params','means','paired','eligible','process_means','proof','seconds')}for r in rows]for ds,rows in trials.items()})
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 双向有限调参、机制解释与新全量队列','',
        '**调参已完成，两个开发最佳均未通过冻结复核，按事前规则回退原参数后启动全量。没有把开发最好点注册为稳定收益。**', '',
        '对象为当前完整框传播＋Global6旧hull＋原精确物理MBR。不是旧OPD；学习率/更新步数不参与本轮。Source权重、同一ReferDINO八处观察、初始gamma、候选/匹配/置信、输入网格/精度/标签和MBR均不改。', '',
        '## 实际调参过程','',
        'HC64/Vid64各64独立父来源，每方向最多18配置；实际14/17。先逐项粗搜，再做机制驱动邻近值、一次温度与证据比例交互、撤回对照。根逐阶段读取结果并决定后续，记录在[decisions.jsonl](decisions.jsonl)。所有64输出先封存，再另进程读开发GT。不是全笛卡尔、随机seed择优或反复查看全量。', '',
        '| 开发面板 | 原完整vIoU % | 所选完整vIoU % | Δv pp | Δt pp | Δ未观测s pp | 开发所选τ / obs / guard / link |',
        '|---|---:|---:|---:|---:|---:|---|']
    for ds,items in trials.items():
        r=next(x for x in items if x['name']==selected[ds]['name']);d=r['paired']['Candidate-Baseline'];p=r['params']
        lines.append(f'| {ds} | {100*r["means"]["Baseline"]["m_vIoU"]:.4f} | {100*r["means"]["Candidate"]["m_vIoU"]:.4f} | {d["v"]["mean"]:+.4f} | {d["t"]["mean"]:+.4f} | {d["unobserved_actual_sIoU"]["mean"]:+.4f} | {p["temperature"]}/{p["observed_weight"]}/{p["guard_weight"]}/{p["link_weight"]} |')
    lines+=['',
        '- **温度语义：**仅在最终读出中将每view合法区间的端点联合分数除τ；各层/offset独立归一化后仍用旧hull及MBR。不是OPD温度，也不改原生观察位置。τ降低使这些开发输出更尖/更短，两个域均下降；τ升高扩展不确定性，开发长事件受益，但不能据此统一提高整个域的温度。',
        '- **证据比例：**score中的观测项为β mean_obs(u)，未观测项为(1−β) mean_unobs(cu)，原β=.5。提高β降低了不可靠传播unary对未观测动作的直接推动。Vid β1为自然边界：未观测直接unary为0，传播候选和连接仍保留；实际未观测替换降到均5.1094节点。不能把这种开发收益表述为更强未观测监督，也未验证身份错误被解决。',
        '- **几何保护：**λG=.2在当前背景比.05/.1更好，减少无充分证据的轨迹替换；HC提高到.3没有继续改善。不是保护越强越好。',
        '- **连接项：**关闭连接会下降，增强到.5未获得足够增量；保留.25。撤回对照分别在选定背景下检查贡献，不能当各参数无交互的普遍结论。', '',
        '## 一次冻结复核与回退','',
        '所选参数文件在预测32×2之前锁定，两域全部64输出再次seal后才读取复核GT。选择标准：完整v不降、t和未观测s下降不超过.1pp、相对Source新增>5pp严重损伤至多2条；失败即完整退回原配方，不据复核挑第二个配置。', '',
        '| 32父复核 | Δ完整v pp | Δt pp | Δ未观测s pp | 候选/原配置相对Source严重损伤数 | 决定 |',
        '|---|---:|---:|---:|---:|---|']
    for ds,r in confirmations.items():
        d=r['paired']['Candidate-Baseline'];s=r['paired'];accepted=r['decision']['accepted']
        lines.append(f'| {ds} | {d["v"]["mean"]:+.4f} | {d["t"]["mean"]:+.4f} | {d["unobserved_actual_sIoU"]["mean"]:+.4f} | {s["Candidate-Source"]["v"]["declined_over5pp"]}/{s["Baseline-Source"]["v"]["declined_over5pp"]} | {"采用"if accepted else"回退原参数"} |')
    lines+=['',
        '**两方向最终参数均为τ1、β.5、λG.1、λL.25。** 开发平均涨点不代表复核可迁移，更不保证全量涨点；本轮调参没有建立可采用的新参数优势，负结果全部保留。历史完整划分已曝光，开发与复核都不冒充未曝光独立测试。', '',
        '观察到明显时长分布差别：Vid开发GT事件中位6.6400秒，复核2.5192秒；<5秒分别23/64与25/32。HC为6.6525与5.1339秒。升温展宽在更长事件开发上改善、复核短事件受损，与此相符，但不能单独认定时长是全部因果。未从复核或GT时长构造在线门控，未据此继续扫参数。', '',
        '## 成本、缓存与全量执行','',
        'HC64完整缓存复用；Vid64各一次共享224与同专家八帧捕获，后续31个参数trial不重跑网络。另两域各同一已捕获query做miss/hit缓存工程回归，共4次原生捕获，专家前向0；Source/E/观察位置/六层logits/31几何张量逐位一致。113工程测试通过，含默认DP回归、保护极限、物理温度、私有RGB副本、内容指纹、PID防重启及query加权parent bootstrap。', '',
        '新全量是独立队列：HC官方1902与Vid官方10303，Source与Final配对；当前Final与Untuned完全相同，明确作为alias复用，不重复DP/后验/相同评分，不宣称两个独立运行。生产Strong参考默认未动，旧formal Vid8405与CG仍暂停，不混用旧OPD预测。', '',
        '启用512MiB按视频内容/精确网格绑定的RGB LRU，4GiB按实际归一化输入张量/掩码/duration/autocast绑定的视觉编码缓存；不跨query缓存文本融合或decoder。完整capture索引192条另按全部metadata（除路径）及runtime绑定，每文件校验SHA；命中原成本不当当前速度。记录实际视觉/RGB命中、分项、worker身份与近100条吞吐，不能把4次工程回归或缓存阶段时延当全量速度。', '',
        f'全量不可变指纹：`{full["fingerprint"]}`。入口为`scripts/boxfixed_tuning/full.py --run`，故障只在原自有进程退出并核同plan后显式`--resume`；全量评价不再选参。每域完整seal后独立CPU官方/dense核验及parent聚类区间，匿名全部signed分片公开。', '',
        '根负责故障核证与修复；唯一Luna max复用既有ta_luna_watch，半小时heartbeat巡检，正常静默、实质异常/完成通知，不创建其他队列或恢复CG。首次后台启动未存活、无state/worker/预测，空日志保留，核无自有进程后同plan独立session重启；没有修改冻结方法。', '',
        ('全量已完成评分，另见results/boxfixed_full_v1。'if full_complete else'当前导出时全量运行中，尚无本轮完整划分成绩。')+'开发全部trial原始匿名JSON、参数/过程统计、全部正负、严重损伤和复核结果保留本目录；[实际全量profile](../../configs/tuned_boxfixed_v1.json)记录回退。']
    (DEST/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    columns=['dataset','trial','query','v_delta_pp','t_delta_pp','observed_s_delta_pp','unobserved_s_delta_pp','v_vs_Source_pp','severe_vs_baseline','severe_vs_Source'];signed=['\t'.join(columns)]
    for ds,items in trials.items():
        for r in items:
            for q in r['anonymous_queries']:
                c,b,s=[q['values'][m]for m in ('Candidate','Baseline','Source')]
                delta=lambda f:None if c[f]is None or b[f]is None else 100*(c[f]-b[f])
                dv=delta('v');vs=100*(c['v']-s['v']);values=[ds,r['name'],q['query'],dv,delta('t'),delta('observed_actual_sIoU'),delta('unobserved_actual_sIoU'),vs,dv < -5,vs < -5]
                signed.append('\t'.join(''if x is None else str(x)for x in values))
    (DEST/'signed_trials.tsv').write_text('\n'.join(signed)+'\n')
    print(json.dumps(dict(status='summarized',actual_trials=summary['actual_trials'],final_profile=profile['datasets'])))

if __name__=='__main__':main()
