"""Render the locked comparison from independently verified scalar results."""
from collections import defaultdict
import csv

import numpy as np

from scope174 import CELLS, HERE, NAMES, ORDER, PLATFORMS, PRIVATE, RESULTS, json_atomic, read, sha, state
from score174 import write_csv

TITLE = {'r158_pca':'R158＋PCA', 'unary':'仅学习候选分数', 'self_only':'单体几何反馈',
         'cross_only':'跨角色交互', 'full':'完整联合反馈'}
METRIC = ('Acc25','Acc50','mIoU')


def table(filename):
    with (RESULTS/filename).open(encoding='utf-8',newline='') as p:
        return list(csv.DictReader(p))


def mt(row):
    return ' / '.join(f'{float(row[k]):.3f}' for k in METRIC)


def signed(row):
    return ' / '.join(f'{float(row[k]):+.3f}' for k in METRIC)


def model_receipts():
    rows=[]
    for source in PLATFORMS:
        for cell in CELLS:
            row=read(PRIVATE/source/f'{cell}_TRAINING.json')
            monitor=row['source_monitor']
            rows.append(dict(source=source,arm=cell,n_train=row['n_train'],n_monitor=int(monitor['n']),
                             monitor_known_ref=int(monitor['ref_n']),parameters=row['parameters'],
                             steps=row['steps'],train_final_loss=row['history'][-1]['train_loss'],
                             seconds=row['seconds'],
                             monitor_subject25=100*monitor['subject25']/monitor['n'],
                             monitor_subject50=100*monitor['subject50']/monitor['n'],
                             monitor_mIoU=100*monitor['subject_iou_sum']/monitor['n'],
                             anchor_monitor_mIoU=100*monitor['anchor_iou_sum']/monitor['n'],
                             monitor_winner_changes=int(monitor['winner_changes']),
                             monitor_reference25=int(monitor['ref25']),
                             monitor_pair25=int(monitor['pair25']),monitor_pair50=int(monitor['pair50'])))
    write_csv(RESULTS/'source_training.csv',rows)
    return rows


def verify_target_intervention_ious():
    data=read(PRIVATE/'target_identity_PRIVATE.json')
    slots={}
    used=0
    specific={25:0,50:0}
    for case in data['rows']:
        if case['status']!='compared':
            continue
        route=f'{case["source"]}_to_{case["target"]}'
        if route not in slots:
            with np.load(PRIVATE/'score'/f'{route}.npz',allow_pickle=False) as z:
                slots[route]=z['slots']
        q=slots[route][case['ordinal']]
        np.testing.assert_allclose(q,case['quality'],atol=1e-10,rtol=0.)
        for replay in case['endpoint'].values():
            choices=np.asarray(replay['slot'])[:,0]
            np.testing.assert_allclose(q[choices],replay['iou'],atol=1e-10,rtol=0.)
        full=case['endpoint']['full']
        for level in specific:
            t=level/100
            specific[level]+=int(full['native_reference_identity']=='wrong'
                                 and case['natural_replay_matches']['full']
                                 and case['sealed_natural_iou']['full']<=t
                                 and full['iou'][1]>t and full['iou'][3]<=t)
        used+=1
    return used,specific


def render():
    points={(r['scope'],r['cohort'],r['arm']):r for r in table('endpoints.csv')}
    contrasts={(r['scope'],r['cohort'],r['contrast']):r for r in table('factorial_effects.csv')}
    counts={r['contrast']:r for r in table('paired_deltas.csv')
            if r['scope']=='six_equal' and r['cohort']=='all'}
    eligible={(r['cohort'],r['arm']):r for r in table('candidate_coverage.csv') if r['scope']=='six_equal'}
    identity={s:{(r['target'],r['arm']):r for r in table(f'{s}_reference_intervention.csv')}
              for s in ('source','target')}
    refs={s:read(RESULTS/f'{s}_reference_intervention_receipt.json') for s in identity}
    audit=read(RESULTS/'independent_audit.json')
    checked_intervention,specific=verify_target_intervention_ious()
    intervals=read(RESULTS/'paired_recording_intervals.json')
    training=model_receipts()
    assert audit['n']==18620
    a=points['six_equal','all','r158_pca']
    b=points['six_equal','all','full']
    diff=contrasts['six_equal','all','full_minus_R158']
    point_positive=all(float(diff[k])>0 for k in METRIC)
    # Deployment assessment from the observed uncertainty and six-direction spread.
    # The measured contrasts and original evaluation protocol are retained separately.
    lower=intervals.get('interval95_pp',{}).get('full_minus_R158',{})
    eligible_to_replace=point_positive and all(lower.get(k,[0.])[0]>0 for k in METRIC)
    decision='保留 R158＋PCA 为主方法' if not eligible_to_replace else '完整 R174 达到本批定位指标的稳定增量'
    lines=[
        '# R174 强锚点联合实例支持校正','',
        f'**{decision}。** 六个迁移方向等权，完整 R174 的 Acc@25 / Acc@50 / mIoU 为 '
        f'**{mt(b)}%**，相对 R158＋PCA 的 **{mt(a)}%** 改变 **{signed(diff)} pp**。全量保留 18,620 条配对记录。',
        '',
        '冻结三个单平台 3EED checkpoint 和原 R158 多模态匹配。每条描述保留 R158 原赢家和三个按 R158 分数筛选的原生候选。'
        '新增头使用单体分数和角色语言读取四个主体、四个参照假设的点云支持。共享三轮更新。'
        '连续中心、尺寸和朝向预测只用于内部几何。五种输出均使用原候选的位置、尺寸及同帧采样点的历史 PCA。',
        '',
        '主体分数以 R158 对数候选分数为起点，原第一名对应零点。单体残差通过 tanh 限幅到 ±0.75。'
        '开启跨角色交互时，主体还加入按参照概率边缘化的成对增量，单对增量限幅到 ±0.50。'
        '初始候选不依赖 GT。四个学习版本共享短名单和单体残差上界。','',
        'R158 视觉路径提供了 RGB 证据。新增头使用点和文本特征及相机可见性，去掉了上一轮新增的 RGB seed 描述子。'
        '四个学习版本共享源域清单、初始参数、三轮计算、监督和每平台 12 轮 AdamW 预算，使用末轮权重。'
        '源域合成关系沿用原双端清单，用三种自然短句模板表达原关系。主体框、参照框及左右约定相同。自然描述只监督主体。'
        '目标域推理冻结，标注在所有预测封存后用于评分和干预诊断，FINAL 未访问。'
        '本批普通 Validation 已用于此前方法的比较，结果衡量该固定样本范围内的配对差异。','',
        '## 1. 共同候选上的整体定位','',
        '| 方法 | Acc@25 (%) | Acc@50 (%) | mIoU (%) | 相对 R158 换候选 |',
        '|---|---:|---:|---:|---:|']
    for arm in NAMES:
        r=points['six_equal','all',arm]
        lines.append(f'| {TITLE[arm]} | {float(r["Acc25"]):.3f} | {float(r["Acc50"]):.3f} | '
                     f'{float(r["mIoU"]):.3f} | {r["changed_from_R158"]} |')
    d=counts['full_minus_R158']
    lines += ['',f'完整版本相对 R158 的 IoU 改善／恶化／持平为 **{d["mIoU_gain"]}／{d["mIoU_loss"]}／{d["iou_same"]}**。'
              f'Acc@25 救回／伤害 {d["Acc25_gain"]}／{d["Acc25_loss"]}，'
              f'Acc@50 救回／伤害 {d["Acc50_gain"]}／{d["Acc50_loss"]}。'
              '均为原分母的配对计数，六方向等权不由这些合计直接相除得出。','',
              '| 训练 → 测试 | n | R158＋PCA | 仅学习分数 | 单体反馈 | 角色交互 | 完整联合 | 完整−R158 (pp) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s,t in ORDER:
        direction=f'{s}_to_{t}'
        v=[points[direction,'all',arm] for arm in NAMES]
        numbers=' | '.join(mt(q) for q in v)
        lines.append(f'| {s.title()} → {t.title()} | {v[0]["n"]} | {numbers} | '
                     f'{signed(contrasts[direction,"all","full_minus_R158"])} |')
    lines += ['', '每组三元数依次为 Acc@25、Acc@50、mIoU。', '',
              '## 2. 角色交互与几何自反馈的增量','',
              '| 比较 (pp) | 全量 Acc@25 / 50 / mIoU | 解析出参照 | 直接左右关系 |',
              '|---|---:|---:|---:|']
    for k,label in (('self_given_no_cross','仅单体反馈 − 仅学习分数'),
                    ('self_given_cross','完整 − 仅角色交互'),
                    ('cross_given_no_self','仅角色交互 − 仅学习分数'),
                    ('cross_given_self','完整 − 仅单体反馈'),
                    ('self_main','单体反馈平均效应'),('cross_main','角色交互平均效应'),
                    ('interaction','两项交互效应'),
                    ('full_minus_R158','完整 − R158＋PCA')):
        seq=[contrasts.get(('six_equal',part,k)) for part in ('all','has_reference','direct_left_right')]
        lines.append('| '+label+' | '+' | '.join(signed(x) if x else '—' for x in seq)+' |')
    n_r=points['six_equal','has_reference','full']['n']
    n_lr=points['six_equal','direct_left_right','full']['n']
    c=eligible['all','full']
    lines += ['',f'解析出参照的样本共 {n_r} 条，直接左右关系 {n_lr} 条。'
              f'四个候选中事后用 GT 取优的 Acc@25 / Acc@50 为 {float(c["oracle_Acc25"]):.3f}% / '
              f'{float(c["oracle_Acc50"]):.3f}%。完整版本有 {c["missed_if_slot_available_at25"]} / '
              f'{c["missed_if_slot_available_at50"]} 条已有达标主体候选仍未选中。'
              f'其中 {c["known_other_subject_within_slots_at25"]} / {c["known_other_subject_within_slots_at50"]} 条'
              '被点归属判为选中另一个标注对象。归属含混记录保留为 UNKNOWN。'
              f'有 {c["full_bound_blocks_all_good_slots_at25"]} / {c["full_bound_blocks_all_good_slots_at50"]} 条'
              '的全部达标候选都落在完整模型残差的理论可翻转范围之外。该范围只由固定的模型分数上界和事后 GT 核验计算。','']
    if intervals['n_boot']:
        ints=intervals['interval95_pp']['full_minus_R158']
        nums='，'.join(f'{k} [{ints[k][0]:+.3f}, {ints[k][1]:+.3f}]' for k in METRIC)
        ns='，'.join(f'{p.title()} {n}' for p,n in intervals['recordings_by_target'].items())
        lines += [f'按目标平台对录制重采样，两条源方向共用抽样权重。现有录制数为 {ns}。'
                  f'完整减 R158 的 95% 描述性区间为 {nums} pp。录制较少的平台对区间影响较大。','']
    lines += ['## 3. 给对参照是否真的帮助主体','',
              '在同一场景中，用正确参照与有独占点支持的同类干扰物做配对。'
              '第一种干预保留框、投影和支持质量统计的显式几何，替换来自该区域的点特征。该点特征本身也可能携带隐式几何。'
              '第二种同时替换参照框与参照点支持。两种干预均固定主体候选、文字及基线分数。'
              '单体版本在所有干预下的主体选择保持数值核验一致。','',
              '| 范围 | 角色条件 | 可比较／全部 | 正确−错配点特征 改善／恶化 | 正确−错配物理参照 改善／恶化 |',
              '|---|---|---:|---:|---:|']
    for split in identity:
        for cell in ('self_only','cross_only','full'):
            r=identity[split]['pooled',cell]
            name='源域留出' if split=='source' else '目标域已核验'
            lines.append(f'| {name} | {TITLE[cell]} | {r["qualified"]}/{r["n"]} | '
                         f'{r["descriptor_correct_improves"]}/{r["descriptor_correct_worsens"]} | '
                         f'{r["physical_correct_improves"]}/{r["physical_correct_worsens"]} |')
    m=identity['target']['pooled','full']
    lines += ['',f'目标核验中，完整版本有 {m["wrong_native_reference_and_missed_at25"]} / '
              f'{m["wrong_native_reference_and_missed_at50"]} 条满足“选到已知错误的参照，且主体池已有达标候选却选错”，'
              f'给正确参照后救回 **{m["correct_binding_recovers_missed_at25"]} / '
              f'{m["correct_binding_recovers_missed_at50"]}** 条。要求正确参照救回且匹配错参照未救回时，'
              f'分别为 **{specific[25]} / {specific[50]}** 条。上述计数是匹配干预子集的观测，'
              '其分母取决于双端核验和同类干扰的可见点支持。目标诊断样本此前已用于机制核验，全部训练和预测早于本次干预。'
              '未能匹配角色、支持不足和无可用错配均在统计分母中保留，单列原因。','',
              '## 4. 训练、封存及独立核验','',
              '| 源平台 | 每个版本训练数 | 留出诊断数 | 留出双端已知 | 每版本梯度步数 |',
              '|---|---:|---:|---:|---:|']
    for cell in training:
        if cell['arm']=='full':
            lines.append(f'| {cell["source"].title()} | {cell["n_train"]} | {cell["n_monitor"]} | '
                         f'{cell["monitor_known_ref"]} | {cell["steps"]} |')
    max_err=max(r['max_abs_error'] for r in audit['directions'].values())
    lines += ['',f'独立 CPU 审计在传感器坐标中逐行重算四个候选的有向 IoU，最大绝对误差 {max_err:.3g}，'
              '两个阈值均无判定分歧。按方向和关系子集从公开匿名逐行结果重算了端点，全部损伤记录保留。'
              f'目标参照干预的 {checked_intervention} 条可比记录也逐条与相同的四候选评分对账。','',
              '配对成绩和所有负例见 [全量端点](results/endpoints.csv)、[配对差值](results/paired_deltas.csv)、'
              '[候选覆盖](results/candidate_coverage.csv) 及 [独立审计](results/independent_audit.json)。']
    (HERE/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    out=dict(active_main='r174_full_candidate' if eligible_to_replace else 'r158_pca',
             eligible_for_replacement=eligible_to_replace,ordinary_rows=18620,
             all_point_estimates_improved=point_positive,
             recorded_uncertainty_all_positive=eligible_to_replace,
             full_deltas_pp={m:float(diff[m]) for m in METRIC},
             target_right_binding_only_rescues_at25_50=[specific[25],specific[50]],
             source_refs=refs['source']['status'],target_refs=refs['target']['status'],
             model_and_threshold_changes_after_target_scores=False,FINAL_access=False,
             audit=sha(RESULTS/'independent_audit.json'))
    json_atomic(RESULTS/'DECISION.json',out)
    state('publication_pending','Complete R174, all matched endpoints and independent audit persisted.')
    return out


if __name__=='__main__':
    print(render())
