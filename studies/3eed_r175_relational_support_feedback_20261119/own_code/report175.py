"""Summarize evidence without selecting models on target results."""
import csv

from scope175 import (CELLS,HERE,N,NAMES,ORDER,PLATFORMS,PRIVATE,RESULTS,
                      json_atomic,read,sha,state)
from score175 import write_csv

LABEL={
    'official_bbf_axis':'3EED 官方 BBF（标注词）','official_bbf_pca':'官方 BBF＋PCA（标注词）',
    'native_text_pca':'3EED 文本读出＋PCA','r158_pca':'R158＋PCA（内部对照）',
    'full':'R175 完整','self_geometry':'单体几何反馈','pair_only':'配对打分',
    'without_dual':'完整架构，去掉双端监督','full_off':'完整权重，关闭点反馈',
    'full_shuffled':'完整权重，配对与点场错配'}
MEASURES=('Acc25','Acc50','mIoU')
CORE=('self_geometry','pair_only','without_dual','full_off','full_shuffled')


def csv_read(name):
    with (RESULTS/name).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def fmt(row,signed=False):
    pattern='{:+.3f}' if signed else '{:.3f}'
    return ' / '.join(pattern.format(float(row[x])) for x in MEASURES)


def training_reports():
    ledgers=[]
    exports={}
    for s in PLATFORMS:
        p=PRIVATE/s
        roster=read(p/'ROSTER_SEAL.json')
        features=read(p/'FEATURE_SEAL.json')
        exports[s]=dict(overview=roster['counts'],qualified=features['supervision'],
                        recording_disjoint=roster['recording_disjoint'])
        for c in CELLS:
            r=read(p/f'{c}_TRAINING.json')
            h=r['source_monitor']
            row=dict(source=s,arm=c,n_train=r['n_train'],steps=r['steps'],epoch=r['selected_epoch'],
                     hours=r['wall_seconds']/3600,parameters=r['params'],
                     natural_monitor=int(h['natural_n']),known_monitor_ref=int(h['n_explicit_reference']),
                     monitor_branches_covered=int(h['n_both_covered']),
                     source_natural_mIoU=100*h['natural_iou']/h['natural_n'],
                     source_natural_Acc25=100*h['natural_acc25']/h['natural_n'],
                     source_natural_Acc50=100*h['natural_acc50']/h['natural_n'],
                     source_monitor_ref_hit25=int(h['ref25']),source_monitor_pair_hit25=int(h['pair25']))
            ledgers.append(row)
    json_atomic(RESULTS/'source_supervision.json',exports)
    write_csv(RESULTS/'source_training.csv',ledgers)
    return exports,ledgers


def render():
    audit=read(RESULTS/'independent_audit.json')
    mech=read(RESULTS/'mechanism_status.json')
    assert audit['n']==N and audit['prediction']==sha(PRIVATE/'predictions/PREDICTION_SEAL.json')
    assert mech['diagnostic_summary_hashed']==sha(RESULTS/'mechanism_summary.csv')
    for d,v in audit['directions'].items():
        assert v['threshold_disagreements']==[0,0]
    sources,training=training_reports()
    rows={(r['scope'],r['cohort'],r['arm']):r for r in csv_read('endpoints.csv')}
    pairs={(r['scope'],r['cohort'],r['arm'],r['comparison']):r for r in csv_read('paired_deltas.csv')}
    strengths={(r['scope'],r['cohort'],r['arm']):r for r in csv_read('field_activity.csv')}
    support={(r['scope'],r['cohort'],r['arm']):r for r in csv_read('support_and_coverage.csv')}
    stamps={(r['split'],r['platform']):r for r in csv_read('mechanism_summary.csv') if r['cohort']=='all'}
    subdomain={a:rows['six_equal','all',a] for a in NAMES}
    delta={a:{m:float(subdomain['full'][m])-float(subdomain[a][m]) for m in MEASURES}
           for a in NAMES if a!='full'}
    ci=read(RESULTS/'recording_intervals.json')
    limits=ci['descriptive_intervals_pp']
    better=all(delta[c][m]>0. for c in ('self_geometry','pair_only') for m in ('Acc25','Acc50'))
    confirmed=better and all(limits[f'full_minus_{c}'][m][0]>0. for c in ('self_geometry','pair_only') for m in ('Acc25','Acc50'))
    deploy=confirmed and all(limits['full_minus_r158_pca'][m][0]>0. for m in MEASURES)
    if not better:
        key='本配置未同时超过两项关键对照，关系点反馈的跨平台收益尚未建立'
    elif not confirmed:
        key='完整版本出现正向点估计，录制间稳定性仍不足'
    else:
        key='本批固定评估支持关系点反馈的定位增量'
    recs=ci['n_recordings']
    lines=[
        '# R175 源域双端监督与关系点支持反馈',
        '',
        f'**{key}。** 完整版本的六方向 Acc@25／Acc@50／mIoU 为 **{fmt(subdomain["full"])}%**。'
        f'相对单体几何反馈变化 **{fmt(delta["self_geometry"],True)} pp**，'
        f'相对仅配对打分变化 **{fmt(delta["pair_only"],True)} pp**。',
        '',
        '## 1. 配方与边界','',
        '三个单平台 3EED epoch100 权重与已有同源 RGB 点投影特征均冻结。'
        '用原生语言匹配从 A256 初始化四个主体和四个参照槽，同样用原始点生成 PCA 朝向。'
        'R158 分数和候选选择只用于表中的内部对照。三个循环共享参数。'
        '所有学习版本保留预测框引导的单体几何反馈，完整版把配对条件概率加权的参照空间场加入下一轮点分数。'
        '参照角色使用反向关系。置信度、投影深度和可见性约束点场，未知关系走单体路径。'
        '输出框来自原生候选，连续回归只提供中间几何状态。',
        '',
        'CORE-3DVG 通过目标与上下文对象的关系匹配辅助定位，EG-3DVG 通过几何掩码控制注意力，'
        'EDA 对语言成分与点云表示进行细粒度对齐。R175 的受检验差异是已监督参照实例的条件空间场'
        '是否在后续点注意力中提供超出同信息配对打分的帮助。'
        ' [CORE-3DVG](https://proceedings.neurips.cc/paper_files/paper/2023/file/9b91ee0da3bcd61905fcd89e770168fc-Paper-Conference.pdf)'
        ' · [EG-3DVG](https://openaccess.thecvf.com/content/CVPR2026/html/Park_EG-3DVG_Expression_and_Geometry_Aware_Grounding_Decoder_for_3D_Visual_CVPR_2026_paper.html)'
        ' · [EDA](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_EDA_Explicit_Text-Decoupling_and_Dense_Alignment_for_3D_Visual_Grounding_CVPR_2023_paper.html)',
        '',
        '## 2. 源域与 checkpoint','',
        '| 源平台 | 自然主体 | 自然参照解析 | 合成双端 | 合成物理帧／录制 | 双端初始候选覆盖 | 训练／留出录制 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def size(source,split,key,field='n'):
        return sources[source]['overview'][split]['origins'].get(key,{}).get(field,0)
    total_synthetic=0
    total_natural=0
    for s in PLATFORMS:
        nat=size(s,'train','natural_unique_inventory')
        gen=size(s,'train','synthetic_labelled_pair')
        fr=size(s,'train','synthetic_labelled_pair','frames')
        rec=size(s,'train','synthetic_labelled_pair','recordings')
        total_natural+=nat
        total_synthetic+=gen
        recs_train=sources[s]['overview']['train']['recordings']
        recs_held=sources[s]['overview']['monitor']['recordings']
        cover=sources[s]['qualified']['train']['both_covered']
        lines.append(f'| {s.title()} | {nat+size(s,"train","natural_subject_only")} | {nat} | {gen} | {fr}／{rec} |'
                     f' {cover}/{nat+gen} | {recs_train}／{recs_held} |')
    lines += ['',f'自然参照新增 **{total_natural} 条**，来自对原描述的依存解析、类别和唯一几何匹配，'
              f'属于源域库存推断标签。合成增加 **{total_synthetic} 条**，每个物理帧最多一条。'
              '源域模糊或不可见的参照保留为未知。双端覆盖指每端均有 IoU > 0.10 的候选，'
              '同类干扰进入正负竞争，只有支持和标签均可用的源样本参与相应损失。'
              '带双端标注训练的三个版本使用相同的支持置信度校准项。'
              '完整、单体、配对和去双端均训练 12 轮，各自从第 4／8／12 轮按源域留出录制的自然主体 mIoU 选点。'
              'Drone 和 Quad 各只有一段源域留出录制。','',
              '| 源平台 | 完整选中 epoch | 单体 | 配对 | 去双端 |',
              '|---|---:|---:|---:|---:|']
    for s in PLATFORMS:
        c={r['arm']:r['epoch'] for r in training if r['source']==s}
        lines.append('| '+s.title()+' | '+' | '.join(str(c[a]) for a in CELLS)+' |')
    lines += ['','## 3. 普通 Validation 六方向配对成绩','',
              '| 方法 | Acc@25 (%) | Acc@50 (%) | mIoU (%) |',
              '|---|---:|---:|---:|']
    for a in NAMES:
        row=subdomain[a]
        lines.append('| '+LABEL[a]+' | '+' | '.join(f'{float(row[m]):.3f}' for m in MEASURES)+' |')
    for a in ('self_geometry','pair_only','without_dual','full_off','full_shuffled','r158_pca'):
        v=pairs['six_equal','all','full',a]
        lines.append(f'\n相对{LABEL[a]}，完整版本 IoU 改善／恶化／持平为 '
                     f'{v["mIoU_gain"]}／{v["mIoU_loss"]}／{v["iou_equal"]}。')
    if all(delta['pair_only'][m]<0. for m in MEASURES):
        lines += ['', '配对打分在三个全量指标上均领先完整版本。'
                    '这里的比较包含训练过程中采用哪条反馈路径的差异。'
                    '同一权重关闭或错配点场的对照进一步测量了推理时注入的贡献。']
    lines += ['', '官方 BBF 复用同 checkpoint、同帧和相同方向中已审计的官方 readout。该 readout 使用标注词位，'
              '同时列出对等目标无标注的 3EED 文本读出＋PCA。三元组顺序为 Acc@25／Acc@50／mIoU。'
              '主表以六个迁移方向等权汇总原普通 Validation 的 18,620 条记录。'
              ' [3EED](https://arxiv.org/abs/2511.01755) · [官方实现](https://github.com/worldbench/3EED)',
              '', '| 方向 | n | 官方 BBF | 单体几何 | 配对打分 | 完整 | 完整−单体 (pp) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for s,t in ORDER:
        a=f'{s}_to_{t}'
        raw=[rows[a,'all',k] for k in ('official_bbf_axis','self_geometry','pair_only','full')]
        d={m:float(raw[-1][m])-float(raw[1][m]) for m in MEASURES}
        lines.append(f'| {s.title()} → {t.title()} | {raw[0]["n"]} | '+' | '.join(fmt(x) for x in raw)+f' | {fmt(d,True)} |')
    lines += ['','## 4. 关系点反馈和真实实例绑定','']
    active=int(strengths['six_equal','all','full']['n_nonzero'])
    cohort=rows.get(('six_equal','has_relation','full'))
    if cohort:
        contrasts=[]
        for c in ('self_geometry','pair_only','full_off','full_shuffled'):
            d={m:float(cohort[m])-float(rows['six_equal','has_relation',c][m]) for m in MEASURES}
            contrasts.append(f'相对{LABEL[c]} {fmt(d,True)} pp')
        lines += [f'**{cohort["n"]} 条**记录解析出支持的关系，实际点反馈非零 **{active} 条**。'
                  '关系子集上的配对差异为'+ '，'.join(contrasts)+'。','']
    lines += [f'同一 checkpoint 保留点场，相对关闭的全量增量为 {fmt(delta["full_off"],True)} pp，'
              f'相对配对与点场错配的增量为 {fmt(delta["full_shuffled"],True)} pp。'
              '关系证据能改变评分前的点注意力。参照身份的可靠性还需要几何与身份匹配的干预证据。','']
    line=support['six_equal','all','full']
    lines += [f'完整模型有 **{line["missed25"]}／{line["missed50"]} 条**已有达标主体候选却未选中。'
              '选择后框内支持比例和各关系子集见统计文件。','',
              '| 物理参照干预 | 可比／全部 | 正确参照的框内支持质量 | 错配的框内支持质量 |'
              ' 正确参照相对错配的 IoU 改善／恶化 |',
              '|---|---:|---:|---:|---:|']
    for split,name in (('source','源域留出'),('target','目标已核验')):
        r=stamps[split,'pooled']
        mass=lambda k:f'{100*float(r[k]):.3f}%' if r[k] else '—'
        lines.append(f'| {name} | {r["paired_n"]}/{r["n"]} | {mass("correct_mass")} | {mass("wrong_mass")} | '
                     f'{r["correct_ref_iou_gain"]}／{r["correct_ref_iou_loss"]} |')
    lines += ['', '干预只替换主体点反馈中读取的参照槽，保留文本、主体槽与同一权重。'
              '框内质量在同一个候选槽和同一批点上计算，表示框内点的注意力质量。'
              '身份含混、支持不足和候选缺失均保留原因和计数。'
              '这些条件性结果在主预测封存后获得，仅用于解释。','']
    for c in ('self_geometry','pair_only','full_off','full_shuffled','r158_pca'):
        iv=limits['full_minus_'+c]
        f='，'.join(f'{m} [{iv[m][0]:+.3f}, {iv[m][1]:+.3f}]' for m in MEASURES)
        lines += [f'完整减{LABEL[c]}的 95% 录制重采样描述性区间为 {f} pp。']
    err=max(v['max_abs_error'] for v in audit['directions'].values())
    lines += ['',f'按目标平台在录制内分组，两条源方向共享重采样。目标录制数为 '
              f'Waymo {recs["waymo"]}、Drone {recs["drone"]}、Quad {recs["quad"]}。'
              f'独立 CPU IoU 核验了 {N:,} 条，最大误差 {err:.3g}，两个阈值零分歧。'
              '所有结果含全部正负变化。目标端参数更新次数和 FINAL 访问次数都为零。','',
              '全部结果见 [六方向与子集](results/endpoints.csv)、[配对差值](results/paired_deltas.csv)、'
              '[源域监督统计](results/source_supervision.json)、[物理参照干预](results/mechanism_summary.csv) 和 '
              '[独立审计](results/independent_audit.json)。']
    (HERE/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    result=dict(n=N,target_model_updates=0,FINAL_access=False,
                relation_feedback_matched_threshold_gains=better,
                relation_feedback_recording_evidence=confirmed,
                deployment_candidate=deploy,mean_deltas_pp=delta,source_natural_dual=total_natural,
                source_synthetic_dual=total_synthetic,mechanism=mech['denominators'],
                audit_sha256=sha(RESULTS/'independent_audit.json'))
    json_atomic(RESULTS/'DECISION.json',result)
    state('publication_pending','R175 measurements and paired mechanism probes audited; publishing.')
    return result


if __name__=='__main__':
    print(render())
