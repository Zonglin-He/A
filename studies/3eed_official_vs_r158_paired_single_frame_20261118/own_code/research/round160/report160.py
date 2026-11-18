"""Write the paired metrics and their interpretation after the independent checks."""
import csv
import json
from pathlib import Path
from common160 import HERE, ROOT, EXTRA, DEV, REC, ORDER, read


def rows(name):
    with (HERE/'results'/name).open(newline='') as f:return list(csv.DictReader(f))


def field(r,k):
    return float(r[k])


def duo(r):
    return f"{field(r,'Acc25'):.2f} / {field(r,'Acc50'):.2f}"


def delta(r):
    return f"{field(r,'Acc25'):+.2f} / {field(r,'Acc50'):+.2f}"


def main():
    audit=read(HERE/'AUDIT.json');seal=read(HERE/'PREDICTION_SEAL.json');ev=read(HERE/'EVALUATION.json')
    assert audit['status']=='PASS' and ev['n']==seal['n']==audit['paired_rows']
    dirs={(r['cohort'],r['source'],r['target'],r['arm']):r for r in rows('directions.csv')}
    avg={(r['cohort'],r['arm']):r for r in rows('overall.csv')}
    pairs={(r['cohort'],r['source'],r['target'],r['before'],r['after']):r for r in rows('paired.csv')}
    crossings={(r['cohort'],r['source'],r['target'],float(r['threshold'])):r for r in rows('threshold_crossings.csv')}
    labels={EXTRA:'普通单帧评价',DEV:'选型开发记录',REC:'另批采集序列'}
    arms={'official_span':'官方 3EED，标注类别辅助词跨度','native_text_only':'3EED，仅文本主体跨度',
          'prepared_axis':'源域准备的 RGB 竞争读出，原轴向','prepared_pca':'相同 RGB 竞争读出，加 PCA',
          'r158_full':'完整 R158 单帧模型，加 PCA'}
    macro=pairs[EXTRA,'six_equal','all','official_span','r158_full']
    a=avg[EXTRA,'official_span'];b=avg[EXTRA,'r158_full']
    txt=['# R160　官方 3EED 与单帧 R158 的同样本对照','',
         f"主比较包括普通清单全部 {b['n']} 条方向记录。官方 3EED 的 Acc@25 / Acc@50 为 **{duo(a)}%**，完整 R158 为 **{duo(b)}%**。同描述、同点云采样、同源 checkpoint 的配对差值为 **{delta(macro)} 个百分点**。最终层赢家的 mIoU 从 {field(a,'mIoU'):.4f}% 到 {field(b,'mIoU'):.4f}%。六个方向等权，所有输入和退化均计入分母。[配对统计](results/paired.csv) · [锁定方案](experiment/PROTOCOL.json)",'',
         '## 两套方法各自看到了什么','',
         '官方网络接收当前帧 LiDAR 的 XYZ 和完整英文表达。作者的单目标评估程序还读取目标标注类别，在原句中按该类别的同义词列表顺序确定词跨度，读取最后一层对比头 BBF 的 Top-1。原模型输出中心和尺寸，评估时尺寸下限取 10⁻⁶，模型坐标系内的预测 yaw 为零。另列的仅文本版本复用同一组原生 logits，以已登记的首个词表名词产生跨度，未命中时使用有效文本 token。它用于衡量部署时只有文字而没有额外目标类别的参照水平。[官方数据加载器](https://github.com/worldbench/3EED/blob/843a901f5367a6be5cc484b97c6ecad462961dfa/src/joint_det_dataset.py#L345) · [官方最后层读出](https://github.com/worldbench/3EED/blob/843a901f5367a6be5cc484b97c6ecad462961dfa/src/grounding_evaluator.py#L209) · [官方尺寸处理](https://github.com/worldbench/3EED/blob/843a901f5367a6be5cc484b97c6ecad462961dfa/main_utils.py#L535)','',
         '完整 R158 另用同一目标帧的 RGB 和相机标定，加载已在该源平台准备的融合、语言匹配和实例竞争模块。R158 在同一组 256 个框上校正重复投票，用期望重叠选出一个原始框，然后用同一份采样 LiDAR 修正朝向。目标类别和目标框均未参与这条预测路径。两臂的基础 3EED 相同，Waymo、Drone、Quad 各使用项目自己的单源 epoch100。本轮源域训练与目标端参数更新次数均为零。[R159 部署实现](https://github.com/Zonglin-He/A/tree/4a83f8efefbf8df67b232da2dda9d47e308b6b13/studies/3eed_single_frame_crossplatform_grounding_20261118) · [单源权重来历](https://github.com/Zonglin-He/A/blob/11686f2b12d924fdb5692f9b0e8707650c9d1cd1/studies/3eed_three_source_six_transfer_baselines_20261114/REPORT.md)','',
         '## 六方向配对结果','',
         '每格依次为 Acc@25 / Acc@50（%），差值为 R158 减官方 3EED，单位为百分点。开发集沿用原分层抽样权重，其余两个范围的方向内记录等权。不同范围分别汇总。[全部方向三项指标](results/directions.csv) · [全部阈值的救回与损伤数](results/threshold_crossings.csv)','']
    for cohort in (EXTRA,DEV,REC):
        txt.extend([f"### {labels[cohort]}（{ev['cohorts'][cohort]:,} 条）",'',
            '| 源 → 目标 | 记录 | 官方 3EED | 仅文本跨度 3EED | 完整 R158 | 配对差值 |',
            '|---|---:|---:|---:|---:|---:|'])
        for s,t in ORDER:
            off=dirs[cohort,s,t,'official_span'];native=dirs[cohort,s,t,'native_text_only'];full=dirs[cohort,s,t,'r158_full']
            p=pairs[cohort,s,t,'official_span','r158_full']
            txt.append(f"| {s.title()} → {t.title()} | {off['n']} | {duo(off)} | {duo(native)} | {duo(full)} | {delta(p)} |")
        p=pairs[cohort,'six_equal','all','official_span','r158_full']
        txt.append(f"| **六方向等权** | **{p['n']}** | **{duo(avg[cohort,'official_span'])}** | **{duo(avg[cohort,'native_text_only'])}** | **{duo(avg[cohort,'r158_full'])}** | **{delta(p)}** |")
        txt.append('')
    txt+=['## 哪个环节贡献了差值','',
          '以下为普通 5,942 条上的六方向平均。各行共用物理输入及评分。由官方到源准备的 RGB 行同时改变了视觉输入、源准备模块和跨度处理，这个差值只能归于整段流程。保持同一读出换用 PCA，以及保持同一 PCA 换用 R158，可以分别衡量几何和决策的增量。','','| 读出 | mIoU % | Acc@25 % | Acc@50 % |','|---|---:|---:|---:|']
    for arm,label in arms.items():
        r=avg[EXTRA,arm];txt.append(f"| {label} | {field(r,'mIoU'):.4f} | {field(r,'Acc25'):.4f} | {field(r,'Acc50'):.4f} |")
    def increase(old,new,metric):
        return field(avg[EXTRA,new],metric)-field(avg[EXTRA,old],metric)
    pca50=increase('prepared_axis','prepared_pca','Acc50')
    rule25=increase('prepared_pca','r158_full','Acc25')
    rule50=increase('prepared_pca','r158_full','Acc50')
    adverse=pairs[EXTRA,'quad','waymo','official_span','r158_full']
    txt+=['',f"在同一套 RGB 匹配模块上，PCA 单独给 Acc@50 带来 {pca50:+.2f} 个百分点。朝向处理一致时，R158 选框再给 Acc@25 / Acc@50 带来 {rule25:+.2f} / {rule50:+.2f} 个百分点。四足 → 车载是主比较中唯一 Acc@50 下降的方向（{field(adverse,'Acc50'):+.2f} 个百分点），其中 {adverse['Acc50_up']} 条跨过阈值、{adverse['Acc50_down']} 条跌破阈值。"]
    txt+=['',f"按全部记录对照官方赢家，Acc@25 有 {macro['Acc25_up']} 条跨过阈值、{macro['Acc25_down']} 条跌破阈值。Acc@50 的对应数量为 {macro['Acc50_up']} 和 {macro['Acc50_down']}。段内计数未按方向重新加权，百分点差值采用六方向等权。IoU 变化可来自换实例、同实例内换框和朝向变化。[所有匿名逐条 IoU](results/all_records.csv) · [逐采集序列结果](results/scenes.csv)",'','### 两个迁移到无人机的方向','']
    for s in ('waymo','quad'):
        parts=[]
        for cohort in (EXTRA,DEV,REC):
            p=pairs[cohort,s,'drone','official_span','r158_full'];cr=crossings[cohort,s,'drone',.5]
            base=dirs[cohort,s,'drone','official_span'];full=dirs[cohort,s,'drone','r158_full']
            parts.append(f"{labels[cohort]} {field(base,'Acc50'):.2f}% → {field(full,'Acc50'):.2f}%（{field(p,'Acc50'):+.2f} pp，救回 {cr['R158_only']}、损伤 {cr['official_only']}）")
        txt.append('**'+s.title()+' → Drone 的 Acc@50**。'+'。'.join(parts)+'。');txt.append('')
    native_d=audit['original_float32_threshold_differences']
    counts=audit['recording_count'][EXTRA]
    txt+=['## 实现核对及解释范围','',
          f"共 {seal['n']:,} 条方向记录，每条使用 16,384 个相同采样点、同一个原始表达和相同标点分词。先封存无标签原生最后层和 R158 预测，再仅为官方读出生成类别辅助跨度。{seal['all_official_masks_match_native_registry']:,} 份正跨度都逐字节复现原数据加载器的已登记 mask。用上游原评估函数读取 Top-1，所得框在全部记录上与独立实现完全一致。此时仅隔离选框之后的 IoU 回调，真实任务评分由随后公共评分器完成。原生分数并列的输入有 {seal['official_top_score_ties']} 条，保留 PyTorch 官方设备排序。官方跨度与纯文字跨度选出不同 query 的输入共 {seal['annotated_span_changed_winner_from_proxy']} 条。[预测封存](PREDICTION_SEAL.json) · [上游读出复核程序](own_code/research/round160/decode160.py)",'',
          f"普通与开发范围合计 7,740 条全部核对原始 A256、采样索引与实际文本哈希。另批采集序列采用部署种子 0。此前这批 Waymo 的 509 条原句保留了原始逗号空格，而官方输入会规范化。对其两个源方向共 {ev['freshly_matched_canonical_descriptions_directed']:,} 条重新用相同规范化文本运行完整 R158，并从同一次主干前向取得官方输出。其余 {ev['legacy_R158_three_values_reproduced']:,} 条逐条复现原 R158 和两个冻结对照的 IoU。[原生前向封存](FORWARD_SEAL.json) · [历史兼容性](EVALUATION.json)",'',
          f"公共指标采用原生单目标加载器的七维 GT、三维有向 IoU，并严格比较 IoU > 0.25 / 0.50。另在原始 LiDAR 坐标中对全部 {audit['all_arm_sensor_frame_IoU_checks']:,} 个框用独立三维相交体积重算，最大 IoU 误差 {audit['maximum_independent_geometry_abs_error']:.2g}。再用论文实现的 float32 几何复核官方与 R158 两个输出，其 Acc@25 / Acc@50 与主表的逐例阈值不一致数分别为官方 {native_d['official_span']['0.25']} / {native_d['official_span']['0.5']}，R158 {native_d['r158_full']['0.25']} / {native_d['r158_full']['0.5']}。[独立审计](AUDIT.json) · [原版浮点几何逐方向复核](results/native_geometry_check.csv)",'',
          f"两套系统在相同样本上的差值才用于方法比较。论文表 2 的六方向 24.26% / 8.16% 对应论文自身的数据与权重。普通清单中 Waymo、Drone、Quad 分别有 {counts['waymo']}、{counts['drone']}、{counts['quad']} 个采集序列，附加评价曾用于此前研究，1,798 条还参与过 R158 选型。3,072 条方向记录来自未用于 R158 选型的 98 个采集目录，目标平台的对应 TRAIN 记录在更早研究中出现过。无人机和四足的独立采集序列很少。本轮 FINAL 访问次数为零。[3EED 论文表 2](https://arxiv.org/html/2511.01755v1#S5.T2) · [结果汇总](results/overall.csv)",'']
    report='\n'.join(txt)
    (HERE/'REPORT_ZH.md').write_text(report.replace('(own_code/research/round160/decode160.py)','(decode160.py)'))
    result=dict(primary=EXTRA,n=int(macro['n']),direction_macro_difference_pp={k:field(macro,k) for k in ('mIoU','Acc25','Acc50')},
          cohorts={c:{'official':duo(avg[c,'official_span']),'text_only':duo(avg[c,'native_text_only']),'R158':duo(avg[c,'r158_full']),
                    'delta':delta(pairs[c,'six_equal','all','official_span','r158_full'])} for c in (EXTRA,DEV,REC)},
          paired_negative_rows_kept=True,working_default='R159 frozen single-frame R158 + PCA',FINAL_access=False)
    (HERE/'DECISION.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
