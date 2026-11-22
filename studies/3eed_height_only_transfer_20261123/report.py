"""Assemble the matched controls, provider transfer, and external pilot."""

import csv

from common import HERE, OUT, PROTOCOL, prior, roster


ANCHORS = ("frozen", "symmetric", "cut_bottom", "closest_feasible_symmetric", "roof")
ANCHOR_NAMES = ("冻结 C_ray_simple", "对称收缩", "只缩底部", "最近的证据等价对称位置", "**只缩顶部**")
PROVIDERS = ("qwen_frozen", "qwen_roof", "wild_native_rgb", "wild_native_rgb_lidar", "wild_frozen", "wild_roof")
PROVIDER_NAMES = ("Qwen 三维输出＋冻结读出", "**Qwen 三维输出＋只缩顶部**", "WildDet3D 原生 RGB", "WildDet3D 原生 RGB＋LiDAR",
                  "WildDet3D 三维输出＋冻结读出", "WildDet3D 三维输出＋只缩顶部")
EXTERNAL = ("native", "window", "frozen", "roof")
EXTERNAL_NAMES = ("原生 Qwen 直立框", "Qwen＋窗口中位深度", "冻结 C_ray_simple", "**只缩顶部**")


def json_result(name):
    return roster.read(OUT / name)


def rows(name):
    with (OUT / f"{name}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def columns(name, *by):
    return {tuple(r[k] for k in by): r for r in rows(name)}


def fmt(x):
    return f"{float(x):.2f}"


def plus(x):
    return f"{float(x):+.2f}"


def tr(metrics):
    return " / ".join(fmt(metrics[k]) for k in ("acc25", "acc50", "miou"))


def interval(values):
    return f"[{plus(values[0])}, {plus(values[1])}]"


def table(header, values):
    names = [header, ["---"]*len(header), *values]
    return "\n".join("| " + " | ".join(map(str, v)) + " |" for v in names)


def check(name, field):
    rec = json_result(name)
    assert all(roster.sha(OUT / k) == v for k, v in rec[field].items())
    assert not rec["FINAL_access"]
    return rec


def render():
    a_checks = [check(f"{k}_anchor_evaluation.json", "output_sha256") for k in ("validation", "extension")]
    diag = [check(f"{k}_fixed_height_oracle_evaluation.json", "files") for k in ("validation", "extension")]
    w_checks = check("wilddet3d_evaluation.json", "files")
    n_checks = check("talk2car_evaluation.json", "output_sha256")
    geometry = [columns(f"{k}_anchor_geometry", "platform", "arm") for k in ("validation", "extension")]
    anchor = [columns(f"{k}_anchor_metrics", "platform", "arm") for k in ("validation", "extension")]
    a_pair = [columns(f"{k}_anchor_pairs", "platform", "arm", "reference") for k in ("validation", "extension")]
    a_oracle = [columns(f"{k}_fixed_height_oracle_pairs", "platform", "arm", "reference")["macro", "oracle_feasible_center", "roof"]
                for k in ("validation", "extension")]
    w = columns("wilddet3d_metrics", "platform", "arm")
    w_pair = columns("wilddet3d_pairs", "platform", "arm", "reference")
    n = columns("talk2car_metrics", "platform", "arm")
    talk = json_result("talk2car_input_seal.json")
    n_qwen = json_result("talk2car_qwen_seal.json")
    n_front = json_result("talk2car_frontend_diagnostics.json")
    assert n_front["qwen"] == roster.sha(OUT / "talk2car_qwen_seal.json")
    assert n_front["labels"] == roster.sha(OUT / "talk2car_evaluation.json")
    w_seal = json_result("wilddet3d_box_provider_seal.json")
    axis_note = ""
    if "axis_replay" in w_seal:
        assert w_seal["axis_replay"] == roster.sha(OUT / "wilddet3d_axis_replay.json")
        replay = json_result("wilddet3d_axis_replay.json")
        assert replay["corrected_prediction_payload"] == w_seal["payload"] and replay["new_model_calls"] == 0
        axis_note = (
            f"首轮联接沿用了接口文档的尺寸顺序。核对正式评分器后，按它的轴定义重新解释了 "
            f"**{replay['reinterpreted_native_boxes']:,}** 个封存框的尺寸。原来的中心、旋转、窗口和模型权重均用于重放。"
            f" [轴重放回执](results/wilddet3d_axis_replay.json) · [作者评分器]({replay['evaluator']})\n\n"
        )
    timing_path = OUT / "wilddet3d_runtime_segments.json"
    earlier_segments = roster.read(timing_path)["prior_elapsed_seconds"] if timing_path.exists() else 0
    w_seconds = w_seal["seconds"] + earlier_segments
    n_pair = columns("talk2car_pairs", "platform", "arm", "reference")["pooled", "roof", "frozen"]
    n_timing = columns("talk2car_camera_timing", "camera_keyframe", "arm")

    arms_table = [(label, *(tr(t["macro", a]) for t in anchor)) for a, label in zip(ANCHORS, ANCHOR_NAMES, strict=True)]
    a_geometry = []
    for name, geo, pair in zip(("Validation", "录制隔离复测"), geometry, a_pair, strict=True):
        for a, label in zip(ANCHORS[1:], ANCHOR_NAMES[1:], strict=True):
            g = geo["pooled", a]
            p = pair["macro", a, "frozen"]
            a_geometry.append((name, label, f"{int(g['changed_center']):,}", f"{int(g['selection_changed']):,}",
                f"{int(g['measured_lost']):,}", plus(p["delta_acc50_pp"]), f"{p['rescue50']} / {p['harm50']}"))

    w_table = [(label, *(tr(w[p, a]) for p in (*prior.PLATFORMS, "macro"))) for label, a in zip(PROVIDER_NAMES, PROVIDERS, strict=True)]
    n_table = [(label, tr(n["pooled", a]), tr(n_timing["True", a]), tr(n_timing["False", a]))
               for a, label in zip(EXTERNAL, EXTERNAL_NAMES, strict=True)]
    w_main = w_pair["macro", "wild_roof", "wild_frozen"]
    w_vs_sensor = w_pair["macro", "qwen_roof", "wild_native_rgb_lidar"]
    w_checks_ray = w_checks["counts"]
    n_ci = n_checks["recording_bootstrap_95pp"]["pooled/roof-frozen"]["acc50"]
    w_ci = w_checks["recording_bootstrap_95pp"]["macro/wild_roof-wild_frozen"]["acc50"]
    a_ci = a_checks[0]["recording_bootstrap_95pp"]["macro/roof-closest_feasible_symmetric"]["acc50"]
    n_2d = f"在有二维框标注的 {n_checks['qwen_2d_annotated']} 条上，Qwen 的二维 Acc@50 为 **{fmt(n_checks['qwen_2d_acc50_percent'])}%**。" if n_checks["qwen_2d_annotated"] else "该交集的二维标注数为 0。"

    result = rf"""# 只缩顶部 · 同尺寸消融、冻结三维前端与外部场景

本轮延续原论文主方法与同一有向 IoU，增加同尺寸位置对照、冻结 WildDet3D 三维前端和 Talk2Car 外部自然语言场景。普通 Validation 与录制隔离复测仍分别为 **9,310** 和 **1,536** 条。完整主表与已有高度先验、软搜索、真值顶面上限和效率实验见第一轮报告。 [前一轮完整实验](../3eed_qwen_height_only_paper_20261123/REPORT_ZH.md) · [本轮协议](PROTOCOL.json) · [三维前端划分补充](WILDDET_SPLIT_ADDENDUM.json) · [外部校验补充](EXTERNAL_BRIDGE_ADDENDUM.json) · [补充图](results/transfer_figure.pdf)

固定最终框高和保留的几何证据后，Validation 上保留原底面比尽量对称收缩高 **{fmt(a_pair[0]['macro','roof','closest_feasible_symmetric']['delta_acc50_pp'])} pp Acc@50**。换成冻结 WildDet3D 三维输出后，顶缩的配对增量为 **{plus(w_main['delta_acc50_pp'])} pp**。Talk2Car–mini 的配对变化为 **{plus(n_pair['delta_acc50_pp'])} pp**。

## 同样的框高，应该从哪一端收缩

原有选区、两个姿态候选、最终获选姿态及目标尺寸完全配对。设原底面为 $b$、原框高为 $h_0$、主方法得到的框高为 $h$。三个收缩方案共用同一个 $h$，底面分别上移 0、$(h_0-h)/2$ 和 $h_0-h$。第四项将对称上移投影到仍保留全部原框内实测点与原首交点的范围。

令 $W$ 为这些保护点。固定 $h$ 和水平足迹时，可行上移区间为

$$
0\le \delta \le \min\left\{{h_0-h,\min_{{w\in W}} w_z-b\right\}}.
$$

主方法的顶面已经容纳 $W$，因此再上移只会因底面略过最低保护点而失去证据。实现保留 $10^{{-8}}$ 米外向余量。各格为 **Acc@25 / Acc@50 / mIoU（%）**，两组均三平台等权。 [实现](anchor_control.py) · [逐录制对照](results/validation_anchor_recordings.csv)

{table(('位置对照', '普通 Validation', '录制隔离复测'), arms_table)}

{table(('队列', '位置对照', '相对只缩顶部的中心移动数', 'Selection 射线变化', '失去原框内实测点', '相对冻结 ΔAcc@50 (pp)', '救回 / 误改'), a_geometry)}

Validation 上，保留原底面的 Acc@50 比同高对称收缩高 **{fmt(a_pair[0]['macro','roof','symmetric']['delta_acc50_pp'])} pp**，比保留同样观测的对称位置高 **{fmt(a_pair[0]['macro','roof','closest_feasible_symmetric']['delta_acc50_pp'])} pp**，后一差值的录制重抽样区间为 **{interval(a_ci)} pp**。已缩框样本中，固定底面的下表面 MAE 为 **{diag[0]['mae_bottom_m']['roof']:.3f} m**，证据等价的对称位置为 **{diag[0]['mae_bottom_m']['mid']:.3f} m**。保护点约束限制可行变动，原下表面提供定位锚点。 [Validation 核验](results/validation_anchor_evaluation.json) · [复测核验](results/extension_anchor_evaluation.json)

固定体积后，IoU 随竖直重叠长度单调增加。将标注中心投影到可行上移区间，求得同高上移上限。{sum(v['grid_checks'] for v in diag)} 条固定样本以 21 个网格点复验。相对原底面，上限在 Validation 只多救回 **{a_oracle[0]['rescue50']}** 条和 **{plus(a_oracle[0]['delta_acc50_pp'])} pp**，在复测多 **{a_oracle[1]['rescue50']}** 条和 **{plus(a_oracle[1]['delta_acc50_pp'])} pp**。该上限只使用已封存后的标注。 [真值位置上限](ground_attribution.py) · [Validation 上限表](results/validation_fixed_height_oracle_pairs.csv) · [复测上限表](results/extension_fixed_height_oracle_pairs.csv)

## 同一指代对象，改用冻结 WildDet3D 的三维输出

在 **{w_seal['cases']:,}** 条普通 Validation 上复用 Qwen 的 2D 窗口和类别路由。WildDet3D 的已发布训练配置使用 3EED 的检测和语言 TRAIN。在读取模型权重前，已按作者划分将本项比较从 Ordinary TRAIN 复测改到完整普通 Validation。 [作者训练数据](https://github.com/allenai/WildDet3D/blob/1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a/docs/TRAINING_DATA.md#3eed) · [划分补充](WILDDET_SPLIT_ADDENDUM.json)

作者的 `geometric` 模式在同一图像上按窗输出单个三维框。比较该模型的 RGB、RGB 加最近回波深度，以及从其纯 RGB 输出接入同一选区、双姿态和顶面修正。RGB 与深度共用相机内参，存在畸变时将图像和窗同时校正。光学深度来自该帧 LiDAR 投影，空像素保留为空。四元数框与 Qwen 框按同一规则转成直立框，以相同有向 IoU 计分。 [作者几何窗与深度接口](https://github.com/allenai/WildDet3D/blob/1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a/docs/INFERENCE.md) · [公开权重](https://huggingface.co/allenai/WildDet3D) · [推理与封存](wilddet_eval.py)

{axis_note}{table(('三维提供者与读出', 'Waymo', 'Drone', 'Quad', '三平台等权'), w_table)}

在 WildDet3D 三维输出上，只缩顶部对同前端冻结读出的 Acc@50 变化为 **{plus(w_main['delta_acc50_pp'])} pp**，录制区间 **{interval(w_ci)} pp**，救回／误改为 **{w_main['rescue50']}／{w_main['harm50']}**。Qwen 主方法相对 WildDet3D RGB＋LiDAR 的直接输出为 **{plus(w_vs_sensor['delta_acc50_pp'])} pp**。各组共用 Qwen 的二维定位。

使用 WildDet3D 纯 RGB 三维输出，再接入物理选区和双姿态读出的组合，Acc@50 已达到 **{fmt(w['macro','wild_frozen']['acc50'])}%**，加入顶缩后为 **{fmt(w['macro','wild_roof']['acc50'])}%**。两种三维前端的系统差距同时包含训练数据与预测差异。顶缩的增量由各前端内部的开关对照确定。

WildDet3D 三维输出有 **{w_checks_ray['active']:,}** 条启用双姿态，**{w_checks_ray['changed']:,}** 个框实际收缩。已改框样本中，Selection 首交变化 **{w_checks_ray['selection_changed']:,}／{w_checks_ray['selection']:,}**，额外 Audit 首交变化 **{w_checks_ray['audit_changed']:,}／{w_checks_ray['audit']:,}**。 [逐条度量](results/wilddet3d_rows.csv.gz)

共获得 RGB 的 **{w_seal['filled_rgb']}** 个及带深度的 **{w_seal['filled_rgb_lidar']}** 个有效三维框。作者输出存储尺寸的顺序为 W、L、H，OpenCV 局部轴对应 L、H、W。角点与作者评分器的最大差为 {w_seal['max_camera_corner_difference_m']:.3g} m。包含加载和封存的模型推理阶段累计耗时 {w_seconds/60:.1f} 分钟，{w_seal['gpu']} 在主推理进程的峰值张量分配为 {w_seal['peak_bytes']/2**30:.2f} GiB。独立重算两阈值的配对数目。 [逐平台对照](results/wilddet3d_pairs.csv) · [权重和推理回执](results/wilddet3d_box_provider_seal.json) · [度量核验](results/wilddet3d_evaluation.json)

## 外部数据上的单帧语言复测

用官方 Talk2Car 命令与公开 nuScenes mini 的 CAM_FRONT 元数据精确匹配。作者加载器以相机 `sample_data` 读取该字段。每条命令只取一张原图、同录制内时间最近的一次 LIDAR_TOP，按各自时间的标定和车辆位姿配准。共 **{talk['descriptions']} 条指令**、**{talk['physical_camera_records']} 个相机记录**、**{talk['scenes']} 段录制**、**{talk['camera_keyframe_descriptions']} 条相机关键帧指令**。最大时差 **{talk['max_camera_lidar_time_offset_ms']:.2f} ms**。交集来自原 Talk2Car Train 划分，本实验只做冻结评估，交集内 Val 指令为 0。 [Talk2Car 作者加载器](https://github.com/talk2car/Talk2Car/blob/3fde52aee3341312196ab9b2f999337eff0693a0/talk2car.py) · [nuScenes 坐标与数据元](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html) · [样本封存](results/talk2car_input_seal.json)

{table(('冻结读出', '全集', '关键帧子集', '扫帧子集'), n_table)}

只缩顶部相对冻结读出的 Acc@50 变化为 **{plus(n_pair['delta_acc50_pp'])} pp**，救回／误改 **{n_pair['rescue50']}／{n_pair['harm50']}**，对应录制重抽样区间 **{interval(n_ci)} pp**。这项外部测量使用作者发布的相机系三维标注并转到 LiDAR 系。{n_2d}几何评分输入完整保留无框和回退。冻结 Qwen 生成 {n_qwen['queries']} 个回答，推理阶段耗时 {n_qwen['seconds']/60:.1f} 分钟。外部 3D 结果采用交集协议，Talk2Car 的官方排行榜按其二维任务计分。 [配对结果](results/talk2car_pairs.csv) · [外部核验](results/talk2car_evaluation.json) · [官方二维评价定义](https://github.com/talk2car/Talk2Car#evaluation)

锁定解析器获得 **{n_front['reply_statuses']['3d'].get('ok', 0)} / {n_front['descriptions']}** 个可用三维框。几何读出有 **{n_checks['changed']['active']}** 条进入双姿态分支，只有 **{n_checks['changed']['changed']}** 个框收缩。外部 mIoU 相对冻结版变化 **{plus(n_pair['delta_miou_pp'])} pp**。顶缩作用取决于前端能否提供有效的三维框和物理支撑。9 条相机关键帧的三维标注转换与 nuScenes 原标注逐条比对。 [逐级覆盖](results/talk2car_frontend_diagnostics.json) · [标注坐标核验](results/talk2car_evaluation.json)

## 复验

以下命令在匹配的研究环境中重放实验。目录和权重散列见协议，冻结 Qwen checkpoint 与上一轮相同。模型生成、几何候选和评分各有单独封存。发布数据只包含匿名数字。新增源训练、目标端更新和 FINAL 访问均为 0。

```bash
study=studies/3eed_height_only_transfer_20261123
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for cohort in validation extension; do
  crossplatform/env/bin/python "$study/anchor_control.py" infer "$cohort"
  crossplatform/env/bin/python "$study/anchor_control.py" evaluate "$cohort"
  crossplatform/env/bin/python "$study/ground_attribution.py" seal "$cohort"
  crossplatform/env/bin/python "$study/ground_attribution.py" evaluate "$cohort"
done
python "$study/fetch_assets.py" wilddet3d "$WILDDET_WEIGHTS_DIRECTORY"
python "$study/lingbot_config.py" "$study/results/lingbot_config.json"
python "$study/wilddet_eval.py" infer --repository "$WILDDET_CHECKOUT" --weights "$WILDDET_WEIGHTS_FILE"
crossplatform/env/bin/python "$study/wilddet_eval.py" generate
crossplatform/env/bin/python "$study/wilddet_eval.py" evaluate
python "$study/fetch_assets.py" nuscenes "$NUSCENES_CACHE"
python "$study/fetch_assets.py" talk2car "$TALK2CAR_COMMANDS"
crossplatform/env/bin/python "$study/nuscenes_data.py" "$NUSCENES_ARCHIVE" "$TALK2CAR_COMMANDS"
python "$study/nuscenes_eval.py" infer --model "$QWEN_MODEL_DIR"
crossplatform/env/bin/python "$study/nuscenes_eval.py" generate
crossplatform/env/bin/python "$study/nuscenes_eval.py" evaluate --archive "$NUSCENES_ARCHIVE" --commands "$TALK2CAR_COMMANDS"
crossplatform/env/bin/python "$study/external_diagnostics.py"
crossplatform/env/bin/python "$study/report.py"
```
"""
    (HERE / "REPORT_ZH.md").write_text(result)
    decision = dict(status="complete", method="ray_equivalent_height_only", protocol=roster.sha(PROTOCOL),
        anchor_cases=[x["independent"]["rows"] for x in a_checks],
        transfer=dict(descriptions=w_checks["independent"]["rows"], delta50_pp=float(w_main["delta_acc50_pp"]),
                      rescues=int(w_main["rescue50"]), harms=int(w_main["harm50"])),
        external=dict(descriptions=talk["descriptions"], scenes=talk["scenes"], camera_keyframes=talk["camera_keyframe_descriptions"],
                      delta50_pp=float(n_pair["delta_acc50_pp"]), rescues=int(n_pair["rescue50"]), harms=int(n_pair["harm50"])),
        target_training=0, source_training=0, FINAL_access=False, report=roster.sha(HERE / "REPORT_ZH.md"))
    roster.write(HERE / "DECISION.json", decision)
    plot(anchor, w, n)


def plot(anchor, w, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    colors = ("#295D92", "#B96639")
    panel, axs = plt.subplots(1, 3, figsize=(11.7, 4.2), layout="constrained")
    labels = ("Frozen", "Symmetric", "Bottom", "Feasible\nmidpoint", "Roof")
    x = np.arange(len(ANCHORS), dtype=float)
    for i, data in enumerate(anchor):
        values = [float(data["macro", a]["acc50"]) for a in ANCHORS]
        axs[0].bar(x+(i-.5)*.32, values, .30, color=colors[i], label=("Validation", "Recording-disjoint")[i])
    axs[0].set_xticks(x, labels, fontsize=8)
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title("A  Same-height placement", loc="left", fontweight="bold")
    variants = ("wild_native_rgb", "wild_native_rgb_lidar", "wild_frozen", "wild_roof", "qwen_roof")
    labels = ("RGB", "RGB +\nLiDAR", "WildDet\nfrozen", "WildDet\nroof", "Qwen\nroof")
    axs[1].bar(labels, [float(w["macro", a]["acc50"]) for a in variants], color=[colors[0]]*4+[colors[1]])
    axs[1].tick_params(axis="x", labelsize=8)
    axs[1].set_title("B  Frozen 3D provider", loc="left", fontweight="bold")
    axs[2].bar(("Native", "Window\nmedian", "Frozen", "Roof"), [float(n["pooled", a]["acc50"]) for a in EXTERNAL],
               color=[colors[0]]*3+[colors[1]])
    axs[2].tick_params(axis="x", labelsize=8)
    axs[2].set_title("C  Talk2Car–mini", loc="left", fontweight="bold")
    for ax in axs:
        ax.set_ylabel("Acc@50 (%)")
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
    panel.savefig(OUT / "transfer_figure.pdf")
    panel.savefig(OUT / "transfer_figure.png", dpi=240)


if __name__ == "__main__":
    render()
