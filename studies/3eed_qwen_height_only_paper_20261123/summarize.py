"""Write the paper evidence tables and the study receipt from verified outputs."""

import csv
import json
from pathlib import Path

from audit_paper import COHORTS, OUT, PLATFORMS, digest, load


HERE = Path(__file__).resolve().parent
NAMES = dict(waymo="Waymo", drone="Drone", quad="Quad", macro="三平台等权")
SPLITS = dict(validation="普通 Validation", extension="录制隔离复测")
MAIN = "height_only"


def read(file):
    return json.loads((OUT / file).read_text())


def fixed_table(headers, rows):
    lines = [headers, ["---"] * len(headers), *rows]
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in lines)


def fmt(x, decimals=2):
    return f"{float(x):.{decimals}f}"


def signed(x, decimals=2):
    return f"{float(x):+.{decimals}f}"


def score_row(row):
    return " / ".join(fmt(row[k]) for k in ("acc25", "acc50", "miou"))


def ci(interval):
    return f"[{signed(interval[0])}, {signed(interval[1])}]"


def count(row, key):
    return f"{int(row[key]):,}"


def sequence(arms, metrics):
    for label, arm in arms:
        yield (label, *(score_row(metrics[c]["macro", arm]) for c in COHORTS))


def report():
    audits = read("independent_recount.json")
    assert set(audits["cohorts"]) == set(COHORTS) and not audits["FINAL_access"]
    metrics = {c: {(r["platform"], r["arm"]): r for r in load(c, "metrics")} for c in COHORTS}
    pairs = {c: {(r["platform"], r["arm"], r["reference"]): r for r in load(c, "pairs")} for c in COHORTS}
    ray = {c: {r["arm"]: r for r in load(c, "geometry") if r["platform"] == "pooled"} for c in COHORTS}
    evals = {c: read(f"{c}_evaluation.json") for c in COHORTS}
    k = {c: pairs[c]["macro", MAIN, "baseline"] for c in COHORTS}
    mech = {r["cohort"]: r for r in csv.DictReader((OUT / "mechanism.csv").open()) if r["platform"] == "pooled"}
    constant = {r["cohort"]: r for r in csv.DictReader((OUT / "constant_prior_metrics.csv").open())
                if r["platform"] == "macro"}
    versus_constant = [float(metrics[c]["macro", MAIN]["acc50"]) - float(constant[c]["acc50"]) for c in COHORTS]
    constant_seal = read("constant_prior_seal.json")
    timing = read("timing.json")
    extension = read("extension_qwen_seal.json")
    disjoint = read("extension_input_seal.json")
    replay = read("extension_live_replay.json")
    assert disjoint["disjoint_recordings_rgb_lidar"] and not extension["target_gt_opened"]
    assert replay["matched_descriptions"] == 1536 and replay["absolute_box_error_m"] == 0.
    readers = ("方法", "普通 Validation", "录制隔离复测")
    platform_rows = []
    for c in COHORTS:
        for p in (*PLATFORMS, "macro"):
            r = pairs[c][p, MAIN, "baseline"]
            bounds = evals[c]["recording_bootstrap_95pp"][f"{p}/{MAIN}-baseline"]["acc50"]
            platform_rows.append((SPLITS[c], NAMES[p], count(r, "n"), count(r, "recordings"),
                fmt(metrics[c][p, "baseline"]["acc50"]), score_row(metrics[c][p, MAIN]),
                signed(r["delta_acc50_pp"]), f"{r['rescue50']} / {r['harm50']}", ci(bounds)))
    main_names = (("原生 Qwen，统一成直立框", "native"), ("Qwen＋窗口中位深度", "A"),
        ("单尺度＋补框", "L7"), ("H，四尺度＋补框", "H"), ("固定 PCA 姿态及同选区", "C_pca"),
        ("全类别双姿态射线选择", "C_ray"), ("冻结 C_ray_simple", "baseline"),
        ("三轴收缩", "ray_equivalent"), ("**只缩顶部**", MAIN))
    contrast = (("只用原生高度", "prior_only"), ("原生高度＋框内实测点", "points_only"),
        ("原生高度＋原首交点", "rays_only"), ("**完整只缩顶部**", MAIN), ("仅取可行区间下端", "no_prior"),
        ("同平台打乱原生高度", "swapped_prior"), ("同输入软射线搜索", "soft_search"),
        ("固定 25% Selection 射线", "rays25pct"), ("固定 50% Selection 射线", "rays50pct"))
    ray_rows = []
    for c in COHORTS:
        for label, a in contrast:
            r = ray[c][a]
            ray_rows.append((SPLITS[c], label, count(r, "updated"),
                f"{count(r, 'selection_changed')} / {count(r, 'affected_selection')}",
                f"{count(r, 'audit_changed')} / {count(r, 'affected_audit')}", count(r, "lost_measured")))

    oracle_rows = []
    for c in COHORTS:
        mn, ora = [metrics[c]["macro", a] for a in (MAIN, "oracle")]
        r = pairs[c]["macro", "oracle", MAIN]
        m = mech[c]
        oracle_rows.append((SPLITS[c], count(m, "branch"), fmt(m["baseline_top_mae_m"], 3),
            fmt(m["main_top_mae_m"], 3), fmt(m["oracle_top_mae_m"], 3), fmt(mn["acc50"]), fmt(ora["acc50"]),
            signed(r["delta_acc50_pp"]), r["rescue50"]))
    above = {r["cohort"]: r for r in csv.DictReader((OUT / "oracle_regimes.csv").open())
             if r["platform"] == "pooled" and r["annotated_top"] == "annotated_top_above_incumbent"}
    old = {r["arm"]: r for r in csv.DictReader((OUT / "historical_3eed.csv").open()) if r["target"] == "macro"}
    old_table = (("3EED 官方 BBF＋原几何，使用类别跨度", "official_bbf_axis"),
                 ("3EED 原始纯文本读出", "native_text_axis"),
                 ("3EED 官方 BBF＋PCA，使用类别跨度", "official_bbf_pca"),
                 ("3EED 原始纯文本读出＋PCA", "native_text_pca"),
                 ("3EED 四尺度 v3", "four_scale_v3"))
    aux = fixed_table(("历史方法", "方向记录数", "六方向等权 Acc@25 / Acc@50 / mIoU（%）"),
                      [(label, count(old[a], "n"), score_row(old[a])) for label, a in old_table])
    categories = []
    for c in COHORTS:
        g = {(r["value"], r["arm"]): r for r in load(c, "strata") if r["platform"] == "pooled" and r["stratum"] == "category"}
        for name in ("car", "pedestrian", "cyclist"):
            b, a = g.get((name, "baseline")), g.get((name, MAIN))
            if a is not None:
                categories.append((SPLITS[c], name, count(a, "n"), fmt(b["acc50"]), fmt(a["acc50"]),
                                   signed(float(a["acc50"]) - float(b["acc50"]))))
    leave_out = {}
    for c in COHORTS:
        cross = [float(r["macro_delta_acc50_pp"]) for r in load(c, "leave_one_out") if r["reference"] == "baseline"]
        leave_out[c] = ci((min(cross), max(cross)))
    pair_interval = [evals[c]["recording_bootstrap_95pp"][f"macro/{MAIN}-baseline"]["acc50"] for c in COHORTS]
    median_cpu = timing["comparisons"]["all_96"]
    margin = timing["comparisons"]["branch_72"]["only_extra_ms"]

    text = rf"""# 只缩顶部 · 单帧 3D 指代论文实验

在 9,310 条普通 Validation 上，**只缩顶部的 Acc@50 为 {fmt(metrics['validation']['macro', MAIN]['acc50'])}%**，较冻结 `C_ray_simple` 高 **{signed(k['validation']['delta_acc50_pp'])} pp**，救回 {k['validation']['rescue50']} 条、误改 {k['validation']['harm50']} 条。在录制、RGB 和 LiDAR 都与该清单分离的 1,536 条复测上，Acc@50 为 **{fmt(metrics['extension']['macro', MAIN]['acc50'])}%**，差值 **{signed(k['extension']['delta_acc50_pp'])} pp**，救回 {k['extension']['rescue50']} 条、误改 {k['extension']['harm50']} 条。两个队列均按三个目标平台等权。 [完整实验图](results/paper_figure.pdf) · [固定协议](PROTOCOL.json) · [独立核验](results/independent_recount.json)

## 主比较

同一帧 RGB、LiDAR、标定和指代表达使用同一套冻结 Qwen3-VL-4B-Instruct 回答。原生 Qwen 框按统一规则转成 LiDAR 坐标系的直立框。表中各格为 **Acc@25 / Acc@50 / mIoU（%）**，阈值采用严格 `IoU > 0.25` 和 `IoU > 0.50`。回退、格式失败与无框结果均计入分母。 [Validation](results/validation_metrics.csv) · [录制隔离复测](results/extension_metrics.csv)

{fixed_table(readers, sequence(main_names, metrics))}

普通 Validation 有 98 段录制和 7,221 个物理时刻，沿用此前允许访问的 9,310 条描述。复测覆盖固定 Ordinary TRAIN 清单的 1,536 条描述和 98 段录制，其录制、RGB 文件和 LiDAR 文件均与此 Validation 清单分离。两者都曾用于项目研发。复测仅用于冻结评估，新增模型训练与目标端权重更新均为 0。 [Validation 清单回执](../3eed_qwen_full_ordinary_validation_20261121/results/input_seal.json) · [复测清单回执](results/extension_input_seal.json) · [历史录制曝光审计](../3eed_pose_recording_exposure_audit_20261121/REPORT_ZH.md)

## 平台、录制与类别

三平台各自先按全部描述计分，再平均。救回和误改是实际配对计数。95% 区间使用每个平台内 4,000 次配对录制重抽样。Drone 与 Quad 的录制数较少，区间反映现有录制的变化。逐个留掉一段录制时，主方法相对冻结版的三平台 Acc@50 差值范围为 {leave_out['validation']} pp 和 {leave_out['extension']} pp。 [Validation 逐录制配对](results/validation_recording_pairs.csv) · [复测逐录制配对](results/extension_recording_pairs.csv) · [Validation 留一录制](results/validation_leave_one_out.csv) · [复测留一录制](results/extension_leave_one_out.csv)

{fixed_table(('队列', '目标平台', '描述数', '录制数', '冻结 Acc@50', '主方法 25 / 50 / mIoU', 'Δ50 (pp)', '救回 / 误改', 'Δ50 95% 区间 (pp)'), platform_rows)}

标注类别仅用于事后分层。此表逐描述加权。距离、点数、相机是否高于原顶面和可行区间宽度在附表分列。 [Validation 分层](results/validation_strata.csv) · [复测分层](results/extension_strata.csv)

{fixed_table(('队列', '类别', '描述数', '冻结 Acc@50', '主方法 Acc@50', 'Δ50 (pp)'), categories)}

## 高度消融与同输入对照

固定 `C_ray_simple` 给出的框底 $b$、水平足迹、朝向和选框射线。令 $P$ 为原框内实测支持点，$E$ 为原 Selection 射线命中的首交点。设原顶面为 $t_0$，可行区间的下端为 $m=\max_{{w\in P\cup E}} w_z$。只在水平足迹与底面固定的嵌套框族中移动顶面，$t\in[m,t_0]$ 恰好描述同时保留这些点和原射线首交的条件。收缩使框成为原框的子集，原本未命中的射线无法增加交点。对原本命中的射线，只要保留原首交点，新框的首交仍发生在同一位置。若顶面低于 $m$，至少移除一个必须保留的点。

设原生框轴方向为 $R$、尺寸为 $d_1,d_2,d_3$，外包直立化后的高度是 $\sum_k |R_{{zk}}|d_k$。物体长轴倾斜时也可能贡献到这项外包高度。读出改用最接近竖直的原生轴的尺寸 $h_Q$ 作为先验，选取 $t=\min(t_0,\max(b+h_Q,m+\varepsilon))$，其中 $\varepsilon=10^{{-8}}\,\mathrm{{m}}$。代码另有空首交和相机处于原框内部时的原框回退。完整约束的度量意义限定在实际用于选框的一组相机代理射线，Audit 留出射线另行检验。 [现有入口](../3eed_qwen_extent_height_only_20261122/live.py) · [公式实现](../3eed_qwen_extent_height_only_20261122/extent_clip.py) · [两队列实现](paper_eval.py)

{fixed_table(readers, [*sequence(contrast, metrics), ("对侧队列中位数常量高度（追加对照）", *[score_row(constant[c]) for c in COHORTS])])}

软搜索控制保持相同候选框底、姿态、测量和原生高度区间，搜索 17 个等距高度及三个受约束高度。目标是命中射线的三维测距误差均值加 10 倍未包容点比例。这一控制移植了 OVM3D-Det 的搜索目标。原论文还有候选构建、伪 LiDAR 和后续自训练。打乱高度保持平台内的高度分布，切断物体与高度的配对。追加常量控制用另一队列冻结输出的原生高度中位数，分别为 {fmt(constant_seal['priors']['validation']['height_m'], 3)} m（用于 Validation）和 {fmt(constant_seal['priors']['extension']['height_m'], 3)} m（用于复测）。 [OVM3D-Det §3.2 和附录 E](https://arxiv.org/html/2411.15657v1#S3.SS2) · [常量对照封存](results/constant_prior_seal.json)

Validation 从冻结版到仅用原生高度已获得 **{signed(pairs['validation']['macro', 'prior_only', 'baseline']['delta_acc50_pp'])} pp**。完整约束相对保留实测点而去掉原首交的版本只增加 **{signed(pairs['validation']['macro', MAIN, 'points_only']['delta_acc50_pp'], 3)} pp**，同时将 Selection 射线变化由 **{count(ray['validation']['points_only'], 'selection_changed')}** 降至 **0**。逐物体高度相对平台内打乱高度的 Validation mIoU 差值为 **{signed(pairs['validation']['macro', MAIN, 'swapped_prior']['delta_miou_pp'], 3)} pp**，Acc@50 差值为 **{signed(pairs['validation']['macro', MAIN, 'swapped_prior']['delta_acc50_pp'], 3)} pp**。与常量高度相比，两个队列的 Acc@50 只多 **{signed(versus_constant[0], 3)} 和 {signed(versus_constant[1], 3)} pp**。结果支持尺寸先验提供绝大部分准确率增量，首交约束提供观测不变性。 [Validation 配对](results/validation_pairs.csv) · [复测配对](results/extension_pairs.csv) · [常量配对](results/constant_prior_pairs.csv)

## 保留了哪些几何证据

计数只汇总对应方案确实改框的记录。一个测量出现在多个描述时计作多次使用。完整方法在 Validation 上有 {count(mech['validation'], 'ray_boundary_raises')} 条收紧了仅实测点版本的高度下界，其中 {count(mech['validation'], 'ray_veto')} 条被约束到原顶面。复测分别为 {count(mech['extension'], 'ray_boundary_raises')} 和 {count(mech['extension'], 'ray_veto')} 条。降低选中射线数的两个控制完整保留各自所用子集，表中 Selection 变化全部来自未用于修正的原 Selection 射线。 [机制统计](results/mechanism.csv) · [Validation 射线审计](results/validation_geometry.csv) · [复测射线审计](results/extension_geometry.csv)

{fixed_table(('队列', '只缩顶部的变体', '改框数', 'Selection 变化 / 射线数', 'Audit 变化 / 射线数', '丢失实测点'), ray_rows)}

## 用真值计算顶面上限

设水平相交面积为 $A_\cap$，预测足迹面积为 $A$，标注底面、顶面和体积为 $g_b,g_t,V_g$。交叠体积为 $I(t)=A_\cap[\min(t,g_t)-\max(b,g_b)]_+$，IoU 为 $I(t)/(A(t-b)+V_g-I(t))$。存在正交叠时，IoU 在 $g_t$ 前非减，在其后非增。因此将标注顶面投影到实际的可行区间即可得到最优 IoU。对 {evals['validation']['oracle_grid_checks'] + evals['extension']['oracle_grid_checks']} 条固定样本又用 33 个高度网格逐条验证最优性。真值在预测封存后才参与这项诊断。 [上限和失败类型](results/oracle_regimes.csv) · [成功、误改、约束锁定和失败示例](results/example_cases.csv)

{fixed_table(('队列', '可收缩分支', '原顶面 MAE (m)', '主方法 MAE (m)', '上限 MAE (m)', '主方法 Acc@50', '上限 Acc@50', 'Δ50 (pp)', '可再救回'), oracle_rows)}

Validation 可收缩分支有 {count(above['validation'], 'n')} 条的标注顶面高于原顶面，其中 {above['validation']['harm50']} 条发生阈值误改。原顶面构成收缩的上界。可行区间内的最优高度也继承原水平足迹和底面的误差。 [上限失败分解](results/oracle_regimes.csv)

## 3EED 已发表实现的历史参考

三份原单源模型各跨到另外两个平台。历史复跑与主表使用相同普通目标清单规模和有向 IoU 口径，分别汇总 18,620 个方向预测与 9,310 个单模型描述预测。官方 BBF 使用评测类名跨度，纯文本列只使用原句。原论文用单平台训练、跨平台测试构成迁移，本文用一个冻结通用 VLM 覆盖三个目标平台。历史表作为不同输入和训练条件的参考。 [3EED §4.1 与附录 B](https://arxiv.org/html/2511.01755v1#S4.SS1) · [已发布本地六方向复跑](https://github.com/Zonglin-He/A/blob/0c68f2c94c0d00cf91305f16df4e0884c97300e0/studies/3eed_official_cross_platform_validation_ablation_20261118/REPORT_ZH.md) · [方向原表](results/historical_3eed.csv)

{aux}

WeakM3D 和 OVM3D-Det 通过测量点到首交点的残差来优化框。OpenBox 的高度细化估计地面和底面。Segment, Lift and Fit 与 AmodalBox 则引入完整形状先验来求框。这里研究固定姿态与底面后的一维几何读出，原首交位置直接确定可行区间，随后才用高度先验选值。它允许报告每一帧的可收缩量和固定足迹下的精度上限。 [WeakM3D §3.3](https://arxiv.org/pdf/2203.08332) · [OpenBox 附录 C.2](https://arxiv.org/html/2512.01352v2#A3.SS2) · [SLF §3.2](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11297.pdf) · [AmodalBox §3.2](https://arxiv.org/html/2611.10821v1#S3.SS2)

## 运行成本与复验材料

单线程 {timing['cpu']}，96 条描述各做三次交叉配对，读取已经预热。完整几何中位时间从 {fmt(median_cpu['frozen_ms']['median_ms'])} ms 到 {fmt(median_cpu['height_only_ms']['median_ms'])} ms，单独的高度操作为 **{fmt(margin['median_ms'], 3)} ms**，第 90 百分位为 {fmt(margin['p90_ms'], 3)} ms。Qwen 两个回答与冻结对照共用。普通 Validation 复用了封存回答。录制复测新增 **{extension['new_outputs']:,}** 个回答且复用 **{extension['cached_prior_replies']:,}** 个，主要运行秒数为 {fmt(extension['seconds'], 1)}，{extension['gpu']} 峰值张量分配 {extension['memory_peak_bytes'] / 2**30:.2f} GiB。此处 CPU 几何时延和 GPU 批量生成吞吐量分别测量。 [CPU 计时](results/timing.json) · [Qwen 复测回执](results/extension_qwen_seal.json)

在有锁定队列和只读模型权重的研究环境中运行以下命令。预测阶段写入私有载荷并封存，评分阶段核对散列后使用目标标注。两个队列的匿名数值、每段录制统计和两套 IoU 评分检查随结果发布。三种正式几何入口在 1,536 条复测上逐框复现。FINAL 访问次数为 0。 [正式入口复放](results/extension_live_replay.json)

```bash
study=studies/3eed_qwen_height_only_paper_20261123
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
crossplatform/env/bin/python "$study/transfer.py" register
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python "$study/transfer.py" infer --model "$QWEN_MODEL_DIR"
for cohort in validation extension; do
  crossplatform/env/bin/python "$study/paper_eval.py" propose "$cohort" --workers 8
  crossplatform/env/bin/python "$study/paper_eval.py" evaluate "$cohort" --workers 8
done
python "$study/audit_paper.py"
crossplatform/env/bin/python "$study/constant_prior.py" --workers 8
crossplatform/env/bin/python "$study/benchmark_cpu.py"
crossplatform/env/bin/python "$study/historical.py" --history "$VERIFIED_GIT_CHECKOUT"
crossplatform/env/bin/python "$study/replay_extension.py" --workers 8
crossplatform/env/bin/python "$study/make_figures.py"
python "$study/summarize.py"
```
"""
    (HERE / "REPORT_ZH.md").write_text(text)
    v, x = [metrics[c]["macro", MAIN] for c in COHORTS]
    decision = dict(status="paper_experiments_completed", method="ray_equivalent_height_only",
        main_lock=read("validation_prediction_seal.json")["files"]["METHOD_LOCK.json"],
        cohorts={c: dict(n=int(m["n"]), macro={a: float(m[a]) for a in ("acc25", "acc50", "miou")},
                        delta_acc50_pp=float(k[c]["delta_acc50_pp"]), rescue50=int(k[c]["rescue50"]),
                        harm50=int(k[c]["harm50"]), recording_bootstrap_95pp=pair_interval[i],
                        prior_project_exposure=True) for i, (c, m) in enumerate(zip(COHORTS, (v, x), strict=True))},
        additional_evaluation=["within-dataset recording-disjoint replication", "same-input module and height ablation",
            "matched soft search", "shuffled and cross-cohort shared height", "ray subsampling", "exact feasible roof oracle",
            "paired category and recording breakdown", "frozen 3EED historical context", "paired CPU timing"],
        evaluation_scope="Existing single-target ordinary 3EED rosters and a single frozen Qwen checkpoint",
        verification=digest(OUT / "independent_recount.json"), report_sha256=digest(HERE / "REPORT_ZH.md"),
        target_updates=0, source_training=0, FINAL_access=False)
    (HERE / "DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")


if __name__ == "__main__":
    report()
