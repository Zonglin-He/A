"""Render the paired ablation and the published 3EED Table 2 comparison."""
from score165 import METRICS, RESULTS, table
from scope165 import HERE, ORDER, atomic_json, read, sha

LABELS = {'waymo':'车载', 'drone':'无人机', 'quad':'四足'}
ARM_LABELS = (
    ('official_bbf_axis','Frozen 3EED，官方类别跨度'),
    ('official_bbf_pca','Frozen 3EED，官方类别跨度＋PCA'),
    ('native_text_axis','Frozen 3EED，纯文本读出'),
    ('native_text_pca','Frozen 3EED，纯文本读出＋PCA'),
    ('r76_lidar_axis','R76，关闭 RGB'),
    ('r76_lidar_pca','R76，关闭 RGB＋PCA'),
    ('r76_rgb_axis','R76，双向 RGB–LiDAR'),
    ('r76_rgb_pca','R76，双向 RGB–LiDAR＋PCA'),
    ('r76_overlap_axis','R76＋重叠去重'),
    ('r76_overlap_pca','R76＋重叠去重＋PCA'),
    ('r91_axis','R76＋R91，原选框'),
    ('r91_pca','R76＋R91，原选框＋PCA'),
    ('r158_axis','R76＋R91＋重叠去重'),
    ('r158_pca','完整单帧 R158＋PCA'),
)
CONTROLLED = (
    ('R76 源域准备，关闭 RGB', 'r76_lidar_axis','native_text_axis'),
    ('R76 的 RGB 路径，统一 PCA', 'r76_rgb_pca','r76_lidar_pca'),
    ('R91 竞争接口，统一原选框和 PCA', 'r91_pca','r76_rgb_pca'),
    ('R91 竞争接口，统一去重和 PCA', 'r158_pca','r76_overlap_pca'),
    ('重叠去重，冻结 R91 和 PCA', 'r158_pca','r91_pca'),
    ('PCA，固定完整方法的候选、中心和尺寸', 'r158_pca','r158_axis'),
    ('PCA，直接加在官方 Frozen 上', 'official_bbf_pca','official_bbf_axis'),
    ('完整方法相对同样带 PCA 的 Frozen', 'r158_pca','official_bbf_pca'),
    ('完整方法相对原始 Frozen', 'r158_pca','official_bbf_axis'),
)


def metrics(r, names=METRICS, prec=3):
    return ' / '.join(f'{float(r[k]):.{prec}f}' for k in names)


def deltas(r, names=METRICS, prec=3):
    return ' / '.join(f'{float(r[k+"_pp"]):+.{prec}f}' for k in names)


def signs(stats, after, before, metric):
    d=[float(stats[f'{s}_to_{t}',after,before][metric+'_pp']) for s,t in ORDER]
    return sum(x>1e-12 for x in d),sum(x< -1e-12 for x in d)


def describe(label, after, before, stats, metric):
    r=stats['six_equal',after,before]
    value=float(r[metric+'_pp'])
    p,n=signs(stats,after,before,metric)
    return f'{label} {value:+.3f} pp（{p} 个方向提高，{n} 个方向下降）'


def main():
    audit=read(HERE/'GEOMETRY_AUDIT.json')
    total=audit['official_direction_records']
    full=audit['cohort']=='full'
    intermediate=read(HERE/'experiment/PROTOCOL.json')['official_population']
    counts=intermediate['expected_eligible' if full else 'initial_eligible']
    m={(r['group'],r['arm']):r for r in table(RESULTS/'metrics.csv')}
    paired={(r['group'],r['after'],r['before']):r for r in table(RESULTS/'paired.csv')}
    pub={(r['direction'],r['method']):r for r in table(RESULTS/'paper_bridge.csv')}
    literal={(r['direction'],r['arm']):r for r in table(RESULTS/'literal_upstream_geometry.csv')}
    total_gain=paired['six_equal','r158_pca','official_bbf_axis']
    paper=pub['six_equal','3EED_published']
    current=m['six_equal','r158_pca']
    dpaper={k:float(current[k])-float(paper[k]) for k in ('Acc25','Acc50')}
    check=[]
    for label,a,b in CONTROLLED:
        r=paired['six_equal',a,b]
        check.append(dict(component=label,after=a,before=b,deltas_pp={k:float(r[k+'_pp']) for k in METRICS},
                          directions={k:dict(zip(('higher','lower'),signs(paired,a,b,k))) for k in METRICS}))
    finding=(f'完整单帧方法相对逐条配对的官方 Frozen，六方向等权 mIoU / Acc@25 / Acc@50 变化为 '
             f'{deltas(total_gain)} pp。')
    pca=m['six_equal','official_bbf_pca']
    gap=paired['six_equal','r158_pca','official_bbf_pca']
    strict_pca=paired['six_equal','official_bbf_pca','official_bbf_axis']
    verdicts={
        'PCA': '在官方 Frozen 与完整方法上均改善六方向的 mIoU、Acc@25 和 Acc@50，是已验证的主要贡献。',
        'overlap_readout': '固定 R91 与 PCA 后仍有总体净收益，Acc@50 在五个方向提高。',
        'R91': '统一去重和 PCA 后有小幅净增量，三个方向的 Acc@50 提升，两个持平，一个下降。',
        'R76_RGB': '对同一套 R76 权重，打开完整 RGB 路径使六个方向的 Acc@50 都下降。',
    }
    scope=('覆盖官方 loader 可评的全部 19,388 条方向记录。' if full else
           '覆盖 18,620 / 19,388 条官方方向记录。既有 FINAL384 对应 768 条方向记录未进入本轮。')
    title='# R165 · 3EED 六方向跨平台 Validation 与严格消融'
    lines=[title,'',finding,scope,'',
           f'**主要收益来自 PCA 朝向。** 只给官方 Frozen 加 PCA，Acc@50 从 {float(m["six_equal","official_bbf_axis"]["Acc50"]):.2f}% '
           f'提高到 {float(pca["Acc50"]):.2f}%，贡献 {float(strict_pca["Acc50_pp"]):+.2f} pp。'
           f'完整方案相对同样带 PCA 的 Frozen 另增加 {float(gap["Acc50_pp"]):+.2f} pp，在六个方向中三升三降。','',
           'Waymo、Drone、Quad 各自使用已存在的单源 epoch100。每个目标表达分别由另外两个来源评估。',
           '全部配置共用一帧目标观测、16,384 个相同采样点、同一描述和源模型的原始 A256。',
           '官方 Frozen 使用发布代码的末层 BBF，类别跨度来自评测注释。R76 和 R91 使用同步 RGB，选框路径只读点云、RGB、标定及原句。',
           '这批 Validation 此前用于开发及评估。R76／R91 是已有的源域准备状态，本轮进行了冻结推理。',
           '[原论文表 2](https://arxiv.org/html/2511.01755v1#S5.T2) · [官方发布代码](https://github.com/worldbench/3EED) · '
           '[官方 Validation 样本数](https://arxiv.org/html/2511.01755v1#A1.T7) · [实验配方](experiment/PROTOCOL.json)','',
           '## 1. 同清单、同 checkpoint 配对','',
           '每个格子为 **mIoU / Acc@25 / Acc@50（%）**，差值单位为百分点。', '',
           '| 单源 → 目标 | n | 官方 Frozen | 完整 R158＋PCA | 配对变化 |',
           '|---|---:|---:|---:|---:|']
    for s,t in ORDER:
        group=f'{s}_to_{t}'
        lines.append(f'| {LABELS[s]} → {LABELS[t]} | {counts[t]:,} | {metrics(m[group,"official_bbf_axis"])} | '
                     f'{metrics(m[group,"r158_pca"])} | {deltas(paired[group,"r158_pca","official_bbf_axis"])} |')
    lines.append(f'| **六方向等权** | **{total:,}** | **{metrics(m["six_equal","official_bbf_axis"])}** | '
                 f'**{metrics(current)}** | **{deltas(total_gain)}** |')
    lines+=['','[全部分方向、分目标平台和总体](results/metrics.csv) · [全部有符号配对差值](results/paired.csv)','',
            '## 2. 组件是否有净贡献','',
            '每行只改变列出的因素。`上升／下降方向` 以 Acc@50 计，`救回／损伤` 是逐条跨过同一阈值的数量。','',
            '| 改动 | ΔmIoU / ΔAcc@25 / ΔAcc@50（pp） | Acc@50 上升／下降方向 | Acc@50 救回／损伤 |',
            '|---|---:|---:|---:|']
    for lbl,a,b in CONTROLLED:
        p=paired['six_equal',a,b]
        plus,minus=signs(paired,a,b,'Acc50')
        lines.append(f'| {lbl} | {deltas(p)} | {plus} / {minus} | {p["Acc50_up"]} / {p["Acc50_down"]} |')
    lines+=['',
        'PCA 与重叠去重有范围明确的任务证据。R91 目前仅支持小幅增量。R76 的 RGB 开关没有提供净收益，且 Acc@50 六方向均下降。',
        '- PCA 路径以原始点云估计朝向，严格固定原候选、中心和尺寸。',
        '- R76 的 RGB 消融在相同训练权重下关闭早期双向 RGB 读取和晚期图像残差。开关整体衡量同步 RGB 路径的增量。'
        +describe('统一 PCA 后的 mIoU', 'r76_rgb_pca','r76_lidar_pca',paired,'mIoU')+'。',
        '- R91 与 R76 共用原模型和同一套融合参数。对比时只启用竞争接口。'
        +describe('固定去重和 PCA 后的 mIoU', 'r158_pca','r76_overlap_pca',paired,'mIoU')+'。',
        '- R158 的重叠去重允许改选候选，PCA 仍在各自最终框上用同一规则。'
        +describe('固定 R91 和 PCA 后的 mIoU', 'r158_pca','r91_pca',paired,'mIoU')+'。','',
        '<details><summary>14 项消融的完整六方向等权指标</summary>','',
        '| 读出 | mIoU / Acc@25 / Acc@50（%） |','|---|---:|']
    for arm,label in ARM_LABELS:
        lines.append(f'| {label} | {metrics(m["six_equal",arm])} |')
    lines+=['','</details>','',
        '## 3. 与论文表 2 的相对水平','',
        '这里各格为 **Acc@25 / Acc@50（%）**。论文和本轮共享跨平台任务定义。发表行采用原作者的数值。',
        '已核对官方发布包的唯一 checkpoint 为 Union 训练。本地三套单源权重独立训练，点云固定种子也与历史批量评测不同。',
        '已发表的每一个方法列为外部参照，逐条配对增益以 §1 的本地复跑 Frozen 为准。'
        ' 表中论文宏平均由六个论文方向等权计算。','',
        '| 单源 → 目标 | 论文 BUTD-DETR | 论文 EDA | 论文 WildRefer | 论文 3EED | 本地 Frozen | 本轮完整方法 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for group,label in [*( (f'{s}_to_{t}',f'{LABELS[s]} → {LABELS[t]}') for s,t in ORDER),('six_equal','**六方向等权**')]:
        items=[metrics(pub[group,name],('Acc25','Acc50'),2) for name in ('BUTD-DETR','EDA','WildRefer','3EED_published')]
        items.extend(metrics(m[group,arm],('Acc25','Acc50'),2) for arm in ('official_bbf_axis','r158_pca'))
        lines.append('| '+' | '.join([label,*items])+' |')
    lines+=['',f'本轮完整方法与论文 3EED 的宏平均数值差为 **{dpaper["Acc25"]:+.2f} / {dpaper["Acc50"]:+.2f} pp**。',
            '[论文表 2](https://arxiv.org/html/2511.01755v1#S5.T2) · '
            '[发布权重包](https://huggingface.co/datasets/RRRong/3EED/blob/main/baseline_ckpt_pred.zip) · [发表数值与本地批量采样复核](results/paper_bridge.csv)']
    if not full:
        bounds={(r['direction'],r['method']):r for r in table(RESULTS/'full_cohort_bounds.csv')}
        q=bounds['six_equal','r158_pca']
        intervals=' / '.join(f'{float(q[a+"_lower"]):.2f}–{float(q[a+"_upper"]):.2f}' for a in ('Acc25','Acc50'))
        lines+=['',f'只依据未评部分的数学上下界，官方完整清单上该配置的 Acc@25 / Acc@50 可能落在 **{intervals}%**。',
                '即使未评记录全部失败，这两项数值仍高于论文等权均值。它们来自现有本地权重与固定采样。'
                ' [范围计算](results/full_cohort_bounds.csv)']
    fp32=literal['six_equal','official_bbf_axis']
    count=sum(sum(a.values()) for a in audit['literal_upstream_vs_primary_threshold_disagreements'].values())
    lines+=['','## 4. 验证边界','',
        '- 作者表 7 列出 9,706 条原始描述。发布加载器按类别同义词筛除 12 条 Waymo 描述，留下 9,694 条可评描述。所有臂使用相同的最小框尺寸 `1e-6`。',
        f'- 使用严格 `IoU > 0.25 / 0.50`。为了让朝向消融共享一个稳定的 IoU 实现，独立传感器坐标凸包重复核验了 '
        f'{audit["independent_sensor_convex_geometry_endpoint_checks"]:,} 个输出。最大绝对 IoU 差 '
        f'{audit["max_independent_sensor_abs_iou_error"]:.3g}。',
        f'- 上游原生 FP32 几何保留了完整平行评分。官方 Frozen 的 Acc@25 / Acc@50 为 '
        f'{metrics(fp32,("Acc25","Acc50"))}%，两种实现合计出现 {count} 个逐臂阈值判定差异。'
        '[评分器更正记录](experiment/SCORER_ADDENDUM.json) · [原生 FP32 的全部指标](results/literal_upstream_geometry.csv)',
        '- 每个录制场景独立出表。同一帧的多个描述与两条来源预测共享匿名标识。'
        '无人机和四足分别只有两个和三个录制，方向均值不能外推为跨新场景显著性。'
        ' [分场景指标](results/recordings.csv) · [逐条指标与负例](results/all_rows)',
        '']
    (HERE/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    (HERE/'REPORT.md').write_text('# Official 3EED cross-platform evaluation and matched ablations\n\n'
       '[Chinese main report](REPORT_ZH.md). For the local paired model and published values, use the main report and '
       '[paper bridge](results/paper_bridge.csv). [Matched contrasts](results/paired.csv) include both positive and negative '
       'outcomes. Each affected direction is measured with the same local source-only checkpoint, aligned raw samples and A256. '
       'PCA modifies yaw on a fixed selected candidate. The native-code geometry bridge provides the literal FP32 outcome for '
       'each arm. Published accuracy is a separate benchmark reference. Replaying the model requires the official inputs '
       'and local source weights, source-prepared states, and audited R97 artifacts.\n',encoding='utf-8')
    atomic_json(HERE/'DECISION.json',dict(finding=finding,paired_scope=audit['cohort'],n=total,official_full=full,
        final_authorization_needed_to_complete_scope=not full,FINAL_access=audit['FINAL_access'],
        task_reports_target_GT_selection=False,source_training_runs=0,target_parameter_updates=0,
        published_reference_numerical_gaps_pp=dpaper,paired_component_findings=check,component_evidence=verdicts,
        evaluation_geometry='common yaw-aware OBB IoU with exact released evaluation bridge',
        geometry_audit_sha256=sha(HERE/'GEOMETRY_AUDIT.json')))
    print('R165 report generated',total,flush=True)


if __name__=='__main__':
    main()
