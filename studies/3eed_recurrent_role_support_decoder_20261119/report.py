"""Build evidence-linked aggregate tables and the predefined study decision."""
import csv

from common import ARMS, HERE, ORDER, PLATFORMS, PRIVATE, RESULTS, json_atomic, read, sha

BASE = 'r158_pca'
ALIASES = {'r158_pca':'R158＋PCA', 'joint':'完整联合', 'without_feedback':'关闭角色交互与几何反馈',
           'without_rgb':'关闭新增 RGB', 'fixed_proposals':'固定候选＋PCA'}
DIRECTIONS = [f'{s}_to_{t}' for s,t in ORDER]
SCOPES = {'all':'全量', 'reference_mention':'解析出参照实体', 'direct_left_right':'直接左右关系'}


def table(filename):
    with (RESULTS/filename).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def metrics(row):
    return ' / '.join(f'{float(row[key]):.3f}' for key in ('Acc25', 'Acc50', 'mIoU'))


def diff(row):
    return ' / '.join(f'{float(row[key]):+.3f}' for key in ('Acc25_pp', 'Acc50_pp', 'mIoU_pp'))


def assemble():
    rows = table('endpoints.csv')
    delta = table('paired_deltas.csv')
    support = table('point_support.csv')
    review = table('prior_verified_identity_panel.csv')
    audit = read(RESULTS/'independent_audit.json')
    roster = {s:read(PRIVATE/s/'ROSTER_SEAL.json')['overview'] for s in PLATFORMS}
    timing = {}
    wire = PRIVATE/'predictions/PREDICTION_SEAL.json'
    assert read(wire)['n'] == audit['n'] == 18620
    for d in DIRECTIONS:
        s = read(PRIVATE/'predictions'/d/'SEAL.json')
        timing[d] = [1000.*t/s['n'] for t in s['seconds']]

    def compare(scope, group, after, before):
        return next(r for r in delta if r['scope']==scope and r['cohort']==group
                    and r['after']==after and r['before']==before)

    def found(scope, group, arm):
        return next(r for r in rows if r['scope']==scope and r['cohort']==group and r['arm']==arm)

    b = compare('six_equal', 'all', 'joint', BASE)
    r = compare('six_equal', 'all', 'joint', 'without_feedback')
    dims = ('Acc25_pp', 'Acc50_pp')
    baseline_success = all(float(b[k])>0 for k in dims)
    feedback_success = all(float(r[k])>0 for k in dims)
    text = ['# 3EED 语言—实例联合解码与连续三维框',
            '',
            '完整联合版本与三项消融完成了三源训练以及普通 Validation 的六方向评价。',
            '',
            f'完整联合相对 R158＋PCA 的 Acc@25、Acc@50 和 mIoU 差值为 **{diff(b)} 个百分点**。'
            f'相同源监督与输入下，完整联合相对关闭反馈的差值为 **{diff(r)} 个百分点**。',
            '',
            '## 方法与比较条件',
            '',
            '每个输入由现有冻结 3EED 和源平台的冻结图像编码器提供 1,024 个点 seed、'
            '绑定到可见 seed 的图像描述子及原文本 token。主体和参照各有四个槽，每轮读取全部 seed。'
            '同一组参数执行三轮，先更新点支持，再预测中心、尺寸和朝向。角色之间使用支持位置及'
            '成对 3D 几何与投影特征交换信息。预测框参与下一轮支持的几何读取，最终主体框由联合解码器直接产生。',
            '',
            '新增解码器含 310,414 个参数。四个臂均使用相同源标注、三轮共享更新及固定 12 个训练轮次。'
            '自然表达监督主体框，参照作为潜变量。合成样本由源帧已标注的可见对象构造，'
            '保留同类干扰并要求根据投影水平轴可唯一判定。合成标签对已有标注库存有效。',
            '固定候选臂和连续臂的训练权重逐张量相同，前者的最终读出是原候选中心、尺寸和既有 PCA 朝向。',
            '关闭 RGB 臂保留相机标定和投影几何。关闭反馈臂同时移除角色间交换与框反馈。',
            '',
            '| 源平台 | 训练自然描述 | 训练合成关系 | 留出自然描述 | 留出合成关系 | 训练录制 / 留出录制 |',
            '|---|---:|---:|---:|---:|---:|']
    for s in PLATFORMS:
        a, m = roster[s]['train'], roster[s]['monitor']
        text.append(f"| {s} | {a['natural']} | {a['relations']} | {m['natural']} | {m['relations']} | "
                    f"{a['recordings']} / {m['recordings']} |")
    text += ['', '## 六方向结果', '',
             '数字为 Acc@25 / Acc@50 / mIoU（%）。按六方向等权汇总，采用严格大于 0.25 / 0.50 的判定。'
             '每个目标观测使用与 R165 相同的描述、LiDAR 采样和 RGB，R158＋PCA 端点逐条复现。',
             '',
             '| 方向 | 记录 | R158＋PCA | 完整联合 | 关闭反馈 | 关闭 RGB | 固定候选＋PCA |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for direction in (*DIRECTIONS, 'six_equal'):
        result = [found(direction, 'all', k) for k in (BASE, *ARMS)]
        text.append(f"| {direction.replace('_to_', '→') if direction != 'six_equal' else '六方向等权'} "
                    f"| {result[0]['n']} | " + ' | '.join(metrics(k) for k in result) + ' |')
    text += ['', '## 配对消融与语言关系', '',
             '每格为完整联合相对列内对照的 Acc@25 / Acc@50 / mIoU 差值（百分点）。'
             '参照与左右子集由输入文本解析确定，包含未纠正和仍无法判断身份的样本。',
             '',
             '| 范围 | 方向记录 | R158＋PCA | 关闭反馈 | 关闭 RGB | 固定候选＋PCA |',
             '|---|---:|---:|---:|---:|---:|']
    for key, group in SCOPES.items():
        rows_g = [compare('six_equal', key, 'joint', arm) for arm in
                  (BASE, 'without_feedback', 'without_rgb', 'fixed_proposals')]
        text.append(f"| {group} | {rows_g[0]['n']} | "+' | '.join(diff(d) for d in rows_g)+' |')
    text += ['', '| 完整联合的对照 | IoU 改善 / 恶化 / 持平 | Acc@25 救回 / 损失 | Acc@50 救回 / 损失 |',
             '|---|---:|---:|---:|']
    for arm in (BASE, 'without_feedback', 'without_rgb', 'fixed_proposals'):
        v = compare('six_equal', 'all', 'joint', arm)
        text.append(f"| {ALIASES[arm]} | {v['mIoU_gain']} / {v['mIoU_loss']} / {v['mIoU_tie']} | "
                    f"{v['Acc25_gain']} / {v['Acc25_loss']} | {v['Acc50_gain']} / {v['Acc50_loss']} |")
    ceiling = next(c for c in table('slot_ceiling.csv') if c['scope']=='six_equal'
                   and c['cohort']=='all' and c['arm']=='joint')
    text += ['', '完整联合臂的四个最终主体槽，事后用真值取最优，'
                f'Acc@25 / Acc@50 上限为 {float(ceiling["oracle_Acc25"]):.3f}% / '
                f'{float(ceiling["oracle_Acc50"]):.3f}%。在阈值 0.25 / 0.50 下，分别有 '
                f'{ceiling["missed_within_slots_at25"]} / {ceiling["missed_within_slots_at50"]} 条看得到达标槽却没有选中。'
                '此分解用于区分现有槽覆盖与最后选框。预测在读真值之前已封存。',
             '', '## 主体与参照支持', '',
             '| 版本 | 全量主体框归属正确 / 未知 | 主体真值内的平均支持权重 | 已核验参照正确 | 双端正确 |',
             '|---|---:|---:|---:|---:|']
    for rv in review:
        arm = rv['arm']
        rr = found('six_equal','all',arm)
        sup = next((f"{float(s['mean_GT_support_mass']):.4f}" for s in support
                    if s['scope']=='six_equal' and s['cohort']=='all' and s['arm']==arm), '—')
        ref = (f"{rv['reference_correct']}/{rv['paired_review_records']}" if arm != BASE else '—')
        dual = (f"{rv['both_correct']}/{rv['paired_review_records']}" if arm != BASE else '—')
        text.append(f"| {ALIASES[arm]} | {rr['subject_owners_correct']} / {rr['subject_owners_unknown']} | "
                    f"{sup} | {ref} | {dual} |")
    n_physical = review[0]['unique_frames']
    text += ['', '主体框归属按候选内部唯一归属点的多数及间隔计算，空框和混杂保留未知。'
             f'参照指标只来自 R163 原图核验后具有 3D 真值的 {review[0]["paired_review_records"]} 条方向记录，'
             f'共 {n_physical} 个物理帧。该面板历史上按失败选取，描述机制用。',
             '源留出数据的参照与双端 Acc 另外按合成标签计。Quad 的留出录制无满足构造条件的合成关系。',
             '留出录制用于检查本轮解码器，三份冻结源骨干曾使用源平台训练集。',
             '',
             '## 数值审计与边界', '',
             '全 18,620 条预测先封存，再读取真值。五个臂都由独立的传感器坐标三维凸体求交复核。'
             f'最大 IoU 差为 {max(x["max_abs_iou_error"] for x in audit["directions"].values()):.2g}，'
             '阈值判定分歧为 0。六方向重复出现同一目标物理帧，按录制报告差值方向。'
             '普通 Validation 具有历史开发曝光。目标域和源骨干参数更新次数均为 0。FINAL 未访问。',
             '',
             '首版合成规则未在 Drone 和 Quad 产生有效双端例。源域改用分离的序位参照和唯一水平最近者后，'
             '关系训练覆盖三个来源。首个训练尝试在 Drone 的弧度输出处出现非有限梯度。'
             '复现后把弧度转换移到输出路径，双角向量继续提供朝向监督和关系特征。所有最终状态来自同一修正版。',
             '',
             '历史 B 小头继续作为参照物排序对照。其 144 帧评测没有对应的六方向主体端点。'
             '本轮的同监督独立对照是关闭角色交换和几何反馈。'
             ' [历史 B 的固定面板](https://github.com/Zonglin-He/A/blob/da78a7e9dcb5ea8ba2de497e60f5f3944d2119ca/studies/3eed_verified_competitor_supervision_20261119/REPORT_ZH.md)',
             '',
             '## 裁决', '']
    feedback_dir = {k:{m:float(compare(k,'all','joint','without_feedback')[m+'_pp']) for m in ('Acc25', 'Acc50')}
                    for k in DIRECTIONS}
    if baseline_success:
        text.append('完整联合在本次普通评价中同时超过 R158＋PCA 的两项主指标。保留为开发候选。')
    else:
        text.append('完整联合在本次普通评价中未同时超过 R158＋PCA 的两项主指标。继续使用 R158＋PCA。')
    if feedback_success:
        text.append('在相同输入和源监督下，完整反馈对总体两项 Acc 均有正增量。'
                    '该对照同时改变角色交换与框反馈，支持组合效应。')
    else:
        text.append('相同输入和源监督下，完整反馈未同时改善两项总体 Acc。'
                    '联合推理作为架构已经执行，现有结果对它的定位贡献没有形成稳健证据。')
    text.extend(('', '[完整指标](results/endpoints.csv) · [配对差值与全部负例计数](results/paired_deltas.csv)'
                 ' · [录制范围](results/recording_signs.csv) · [身份面板](results/prior_verified_identity_panel.csv)'
                 ' · [源留出结果](results/source_diagnostics.csv) · [输出槽诊断](results/slot_ceiling.csv)'
                 ' · [独立审计](results/independent_audit.json)'
                 ' · [运行方式](IMPLEMENTATION.md)', ''))
    (HERE/'REPORT_ZH.md').write_text('\n'.join(text), encoding='utf-8')
    decision = dict(state='completed', n=18620, target_parameter_updates=0, source_backbone_updates=0,
                    final_access=False, joint_above_R158_both_accuracy_thresholds=baseline_success,
                    retained_as_development_candidate=baseline_success, feedback_improved_both_accuracy_thresholds=feedback_success,
                    production_default='R158_PCA', feedback_by_direction_pp=feedback_dir,
                    main_delta_pp={k:float(b[k]) for k in (*dims, 'mIoU_pp')},
                    feedback_delta_pp={k:float(r[k]) for k in (*dims, 'mIoU_pp')},
                    runtime_milliseconds_per_row=dict(by_direction=timing,
                        columns=['frozen_evidence_excluding_archived_proposal_forward', *ARMS],
                        target_time_measured_on_RTX_5090=True),
                    prediction_sha256=sha(wire), audit_sha256=sha(RESULTS/'independent_audit.json'))
    json_atomic(HERE/'DECISION.json', decision)


if __name__ == '__main__':
    assemble()
