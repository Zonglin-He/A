"""Render the audited factorial effects and reference-error denominators."""
import csv

from study import ARMS, CELLS, HERE, NEW, ORDER, OLD_PRIVATE, PLATFORMS, PRIVATE, RESULTS, json_atomic, read, sha

NAMES = dict(r158_pca='R158＋PCA', off_off='全关', role_only='仅角色交互',
             geometry_only='仅框反馈', joint='全开')
PAIR = [('role_at_geometry_off','框反馈关时开角色交互'),
        ('role_at_geometry_on','框反馈开时开角色交互'),
        ('geometry_at_role_off','角色交互关时开框反馈'),
        ('geometry_at_role_on','角色交互开时开框反馈'),
        ('role_main','角色交互平均主效应'),
        ('geometry_main','框反馈平均主效应'),
        ('interaction','两项交互效应'),
        ('joint_minus_off_off','全开相对全关'),
        ('joint_minus_R158_PCA','全开相对 R158＋PCA')]
METRIC = ('Acc25','Acc50','mIoU')


def table(name):
    with (RESULTS/name).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def match(rows, **keys):
    return next(r for r in rows if all(r[k]==v for k,v in keys.items()))


def triplet(r, delta=False):
    if delta:
        return ' / '.join(f"{float(r[k+'_pp']):+.3f}" for k in METRIC)
    return ' / '.join(f'{float(r[k]):.3f}' for k in METRIC)


def write_public_training():
    data = []
    for source in PLATFORMS:
        for cell in CELLS:
            original = cell in ('off_off','joint')
            name = 'without_feedback' if cell=='off_off' else 'joint'
            r = read(OLD_PRIVATE/source/f'{name}_TRAINING.json' if original
                     else PRIVATE/source/f'{cell}_TRAINING.json')
            data.append(dict(source=source,cell=cell,reused=int(original),train_n=r['n_train'],
                             training_updates=r['budget_updates'] if original else r['updates'],
                             parameters=r['parameters'],curve=r['train'] if original else r['curve']))
    json_atomic(RESULTS/'training_receipt.json',data)


def main():
    endpoints = table('endpoints.csv')
    contrasts = table('factorial_contrasts.csv')
    ref = [r for r in table('reference_diagnostics.csv') if r['cohort']=='all']
    identity_coverage = read(RESULTS/'identity_coverage.json')
    blind = read(RESULTS/'blind_switchable_review.json')
    audit = read(RESULTS/'independent_audit.json')
    assert audit['n']==18620 and not audit['FINAL_access']
    summary = [r for r in contrasts if r['scope']=='six_equal']
    with (RESULTS/'factorial_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,list(summary[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)

    def cell(scope,method):
        return match(endpoints,scope=scope,arm=method,cohort='all')

    def effect(name,cohort='all'):
        return match(contrasts,scope='six_equal',cohort=cohort,contrast=name)

    a,b = ref
    text = [
        '# 角色交互与预测框反馈的 2×2 消融', '', '## 设置', '',
        '角色交互开关控制原有成对打分和双向消息。成对打分继续使用原有相对几何特征。'
        '框反馈开关控制上一轮预测框的中心与尺寸对下一轮点支持的直接空间偏置。',
        '四个版本的主体和参照均使用四个槽与三轮共享更新。原有 RGB、点云、文字、初始化、监督、'
        '训练顺序、优化器和最后一轮选择固定。两组单开各训练三个源，原有两组及 R158＋PCA 成绩和端点逐条复核后复用。',
        '目标域使用同一 18,620 条普通 Validation 方向记录，零参数更新。FINAL 未访问。',
        '', '## 定位成绩', '',
        '每格为 Acc@25 / Acc@50 / mIoU（%）。六方向等权，阈值采用严格大于。',
        '', '| 范围 | 记录 | R158＋PCA | 全关 | 仅角色交互 | 仅框反馈 | 全开 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for scope in [*(f'{s}_to_{t}' for s,t in ORDER),'six_equal']:
        data = [cell(scope,k) for k in ARMS]
        text.append(f"| {scope.replace('_to_','→') if scope!='six_equal' else '六方向等权'} "
                    f"| {data[0]['n']} | "+' | '.join(triplet(r) for r in data)+' |')
    text += ['', '## 主效应和交互', '',
             '单位为百分点。平均主效应在另一项关闭和开启的条件下等权平均。'
             '交互效应为全开减去两个单开，再加回全关。四个版本均使用自己的源域训练状态。',
             '', '| 对照 | 全量 Acc@25 / Acc@50 / mIoU | 直接左右子集 Acc@25 / Acc@50 / mIoU |',
             '|---|---:|---:|']
    for name,title in PAIR:
        text.append(f"| {title} | {triplet(effect(name),delta=True)} | {triplet(effect(name,'left_right'),delta=True)} |")
    text += ['', '## 已有达标槽但未选中', '',
             '固定完整模型第三轮产生的主体、参照槽和分数。逐列固定参照，按原成对分数重新选主体。'
             '第二个干预去掉末轮成对分数，直接按同一状态的主体一元分数选槽。',
             '', '| 完整联合的诊断 | IoU > 0.25 | IoU > 0.50 |','|---|---:|---:|']
    track = [
        ('已有达标主体槽却未选中','missed'),
        ('其中有解析出的参照','missed_with_reference'),
        ('其中换现有参照槽可救回','switch_column_rescues'),
        ('可借由不同、可辨认的标注实体救回','switch_to_known_other_owner_rescues'),
        ('有同一已知实体的另一个槽可救回','switch_to_same_known_owner_rescues'),
        ('无现有参照列可救回','missed_without_existing_column_rescue'),
        ('关闭末轮成对分数的全量救回','without_pair_readout_rescues'),
        ('关闭末轮成对分数的全量损失','without_pair_readout_harms'),
        ('漏选中有目视核验参照','visual_known_missed'),
        ('目视核验范围中实际参照指向另一个已标注实体','visual_wrong_ref_missed'),
        ('改选回正确参照后救回','visual_wrong_ref_correct_slot_rescued'),
        ('漏选中有标注代理参照','proxy_known_missed'),
        ('标注代理范围中实际参照指向另一个已标注实体','proxy_wrong_ref_missed'),
        ('改选回代理参照后救回','proxy_wrong_ref_correct_slot_rescued'),
        ('漏选中无核验或代理参照标签','missed_with_no_verified_or_proxy_reference'),
    ]
    for label,key in track:
        text.append(f'| {label} | {a[key]} | {b[key]} |')
    text += [
        '', '可切换参照列的救回数给出固定最终状态与已有参照候选时的上界。换列后主体达标只表示读出依赖该列。'
        '正确参照的受控读出只保留能归属到已核验参照物的列，选择其中原分数最高的完整配对。',
        f'已有目视核验面板共关联 {identity_coverage["panel_rows"]} 条方向记录，'
        f'新解码器实际解析到相同参照短语且可核验的有 {identity_coverage["language_role_matched"]} 条。'
        '有明确身份的错误和无法归属的预测分别记录。代理组沿用 R166 的单一可匹配类别、可见和投影关系检查，'
        '其估计依赖已有物体标注范围。全量数据无法确定的语义身份继续为未知。',
        '两个干预都在最后一次支持更新完成后执行，早期错误的参照消息及其对候选生成的影响不在该计数范围。'
        '两个阈值的漏选可重叠。相同表达在两个来源方向出现，不能作为两份独立人工标注。',
        '', '## 换参照可救回记录的补充盲核', '',
        f'逐条查看所有可换列救回的 {blind["n"]} 个物理帧及描述。阅片只看原图、表达、实际解析的参照短语和已有标注，'
        '隐藏模型输出、排名、阈值和比较结果。锁定单阅片结果后，再核对参照拥有的点及换列决策。',
        '', '| 反事实人群 | 可换列 | 唯一标注参照 | 实际绑定另一个已标注实体 | 参照无法归属 | 正确参照有槽 | 换回正确参照后确认救回 |',
        '|---|---:|---:|---:|---:|---:|---:|'
    ]
    for cut in (0.25,0.50):
        r = blind['thresholds'][str(cut)]
        columns = ('reviewed_switchable','identity_verified','wrong_native_owner','unknown_native_owner',
                   'correct_ref_available','wrong_and_correct_ref_rescued')
        text.append(f'| IoU > {cut:.2f} | '+' | '.join(str(r[k]) for k in columns)+' |')
    reasons = blind['by_resolution_status']
    text += ['', f'31 个帧中，{reasons["non_instance"]} 条把图像边缘或场景区域解析成参照，'
              f'{reasons["subject_or_role_ambiguous"]} 条读到了主体自指，'
              f'{reasons["no_annotated_reference"]} 条缺参照框标注，'
              f'{reasons["ambiguous"]} 条身份无法唯一确定。现有的可切换数已经覆盖所有固定末轮列。'
              '有参照错绑、但切换到正确身份后仍未达标的输入保留。两批身份审核样本可能重叠。'
              '补充核验未用于训练或配置选择。',
        '', '## 审计与裁决', '',
        f'全开的连续框、主体槽和选择在 {audit["n"]} 条重放中与历史档案逐条匹配。'
        '两组新预测和反事实分数先封存后评分。五个系统端点和所有固定主体槽采用独立传感器坐标三维凸体求交复核，'
        '阈值分歧为零。结果来自一个一致的训练种子和已有开发曝光的普通 Validation。'
    ]
    decision = dict(n=18620, seed=174, FINAL_access=False, target_updates=0,
                    production_default='R158_PCA',
                    primary_effects_pp={k:{m:float(effect(k)[m+'_pp']) for m in METRIC} for k,_ in PAIR},
                    reachable_miss_diagnostics=[{k:v for k,v in r.items() if k not in ('scope','cohort')} for r in ref],
                    scoring_audit_sha256=sha(RESULTS/'independent_audit.json'))
    decision['additional_blind_reference_review'] = blind['thresholds']
    decision['retain_geometry_only_as_internal_ablation'] = True
    json_atomic(HERE/'DECISION.json',decision)
    text += [
        '', '**框反馈解释了主要的 Acc@25 和 mIoU 增量。** 仅框反馈相对全关提高 1.095 / 0.727 / 0.566 个百分点，'
        '六方向的 mIoU 均提高。在框反馈已经开启时，加入角色交互让三项总体指标分别下降 0.322 / 0.081 / 0.173 个百分点。'
        '角色交互单开在左右子集对 Acc@50 仍有 1.514 个百分点的正信号。两项合用的 Acc@50 交互为 −0.768 个百分点，'
        '直接左右子集为 −1.670 个百分点。四个版本中仅框反馈的总体成绩最高，R158＋PCA 继续作为默认方法。',
        '',
        '完整模型在固定末轮状态下，靠改选现有参照最多处理 0.84% 的 Acc@25 漏选和 0.37% 的 Acc@50 漏选。'
        '在这批可改选的记录里，观察到真实参照错绑，确认换回正确实体并救回主体的数量为零。'
        '主体本身的分数和成对项对各参照列的共同压制，都仍可能导致遗漏。'
        '取消末轮成对项同时带来大量救回与损失，现有数据没有支持把局部切换规则当成通用修复。',
        '', '[六方向端点](results/endpoints.csv) · [因子效应](results/factorial_summary.csv)'
                 ' · [全部反事实计数](results/reference_diagnostics.csv) · [逐方向反事实](results/reference_by_direction.csv)'
                 ' · [独立审计](results/independent_audit.json)', '']
    (HERE/'REPORT_ZH.md').write_text('\n'.join(text),encoding='utf-8')
    write_public_training()


if __name__=='__main__':
    main()
