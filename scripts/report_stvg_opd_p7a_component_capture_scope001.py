"""Capture-scope revision001 of the root-gated anonymous OPD component report.

The preserved raw actual_frozen_capture_forwards key counts full capture
function invocations; each has two source offset model forwards. Extra decoder
replay calls are outside this counter. No scientific witness or pin is changed.

report() writes the actual deliverables only after both real root receipts pass.
render_pdf() accepts already anonymous arithmetic and also permits a synthetic
layout fixture.  This authoring module is separate from frozen science code.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_p7a_component_common_v1 import (
    BASE, OUT, ARMS, BOOT, SEED, read, write, sha, verify)
from scripts.score_stvg_opd_p7a_component_v1 import (
    FIELDS, DATASETS, ORDERS, STATES, READOUTS, CONTRASTS, PUBLIC_KEYS,
    rows_read, public_rows_check, parent_rows, summary, near)

LABELS = {'hc2': 'HC-STVG-v2', 'vidstg': 'VidSTG'}
ARM_LABELS = {'joint_carry': 'Joint + carry', 'LN_carry': 'LN + carry',
              'joint_reset': 'Joint + reset', 'LN_reset': 'LN + reset'}
SHORT = {'remove_QR_carry': 'Remove QR / carry',
         'remove_carry_joint': 'Remove carry / joint',
         'remove_carry_LN': 'Remove carry / LN',
         'remove_QR_reset': 'Remove QR / reset',
         'remove_both': 'Remove both', 'interaction': '2 x 2 interaction'}


def number(x, places=3):
    return 'unknown' if x is None else f'{x:.{places}f}'


def interval(e, signed=True):
    if e['mean'] is None:
        return 'unknown'
    sign = '+' if signed else ''
    lo, hi = e['ci95']
    return f"{100*e['mean']:{sign}.3f} [{100*lo:{sign}.3f}, {100*hi:{sign}.3f}]"


def evidence_label(e):
    if e['mean'] is None:
        return '未定义'
    lo, hi = e['ci95']
    if lo > 0:
        return '此开发条件下正向，探索性 CI 全正'
    if hi < 0:
        return '此开发条件下负向，探索性 CI 全负'
    return '探索性 CI 跨零；未建立优势，也未建立等效'


def validate_inputs():
    """Root receipts and all pinned output bytes precede any anonymous read."""
    verify()
    bp = BASE/'GLOBAL_PREDICTION_BARRIER.json'
    barrier = read(bp)
    assert barrier['status'] == 'sealed' and barrier['GT_read'] is False
    assert barrier['actual_complete_fits'] == 512 and barrier['actual_complete_rounds'] == 12800
    assert barrier['actual_frozen_captures'] == 128 and barrier['actual_new_DINO_calls'] == 0
    root = read(BASE/'ROOT_READBACK.json')
    scalar = read(BASE/'ROOT_ANONYMOUS_SCALAR_READBACK.json')
    complete = read(BASE/'CPU_COMPLETION.json')
    for receipt in (root, scalar):
        assert receipt['status'] == 'pass' and receipt['barrier_sha256'] == sha(bp)
        assert receipt['rows'] == 512 and receipt['parent_arm_rows'] == 256
        assert receipt['time'] >= barrier['time']
        assert receipt['scoring_completion_sha256'] == sha(BASE/'CPU_COMPLETION.json')
        assert receipt['deployment_readback_sha256'] == sha(BASE/'ROOT_DEPLOYMENT_READBACK.json')
        assert receipt['files'] == complete['files']
    assert complete['barrier_sha256'] == sha(bp)
    for rel, digest in complete['files'].items():
        assert sha(ROOT/rel) == digest, rel
    return barrier, root, scalar


def tables(s):
    metrics = []
    contrasts = []
    costs = []
    for target, blocks in [('pooled', {'paired_both_orders': s['pooled'], 'orders': {}}),
                           *s['datasets'].items()]:
        for order, block in [('paired_both_orders', blocks['paired_both_orders']),
                             *blocks['orders'].items()]:
            for field, values in block['fields'].items():
                for arm, e in values['arms'].items():
                    metrics.append(dict(target=target, order=order, arm=arm, field=field,
                        **{k: v for k, v in e.items() if k not in ('parent_values', 'ci95', 'ci95_pp')},
                        ci95_low=e['ci95'][0] if e['ci95'] else None,
                        ci95_high=e['ci95'][1] if e['ci95'] else None))
                for name, e in values['contrasts'].items():
                    contrasts.append(dict(target=target, order=order, contrast=name, field=field,
                        **{k: v for k, v in e.items() if k not in ('parent_values', 'ci95', 'ci95_pp')},
                        ci95_low=e['ci95'][0] if e['ci95'] else None,
                        ci95_high=e['ci95'][1] if e['ci95'] else None))
            for arm, c in block['compute'].items():
                for key, value in c['instrumented'].items():
                    additive = not ('peak' in key.lower() or 'parameters' in key.lower())
                    published = dict(value)
                    if not additive:
                        published['total'] = None
                    units = ('full_capture_function_invocations' if key == 'actual_frozen_capture_forwards' else
                             'bytes_per_fit_peak' if 'peak' in key.lower() else
                             'coordinates_per_fit' if 'parameters' in key.lower() else
                             'seconds' if 'seconds' in key.lower() else 'logged_count')
                    costs.append(dict(target=target, order=order, arm=arm, measure=key,
                        **published, additive_total=additive, units=units,
                        actual_complete_fits=c['actual_complete_fits'],
                        actual_complete_rounds=c['actual_complete_rounds'],
                        actual_backwards=c['actual_backwards'], empty_fits=c['empty_fits'],
                        no_information_rounds=c['no_information_rounds'],
                        active_parameters=';'.join(str(x) for x in c['active_parameters'])))
    return metrics, contrasts, costs


def write_csv(path, rows):
    path = Path(path)
    assert not path.exists(), 'Preserve previous report output'
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x', newline='', encoding='utf-8') as f:
        out = csv.DictWriter(f, fieldnames=keys)
        out.writeheader()
        out.writerows(rows)


def flatten_row(r):
    out = {k: v for k, v in r.items() if k not in ('compute', 'orders')}
    if 'orders' in r:
        out['orders'] = ';'.join(r['orders'])
    out.update({'compute_'+k: v for k, v in r['compute'].items()})
    return out


def markdown(s, barrier):
    lines = ['# P7-A / OPD 内部组件：QR × LN 历史写回', '',
        '本报告覆盖原 HC-STVG-v2 与 VidSTG 开发名单，每域 32 个 parent、原两序、四臂共 512 个实际完整 fit、12800 条更新轮记录。四臂是 joint_carry、LN_carry、joint_reset、LN_reset；OPD 目标固定。原开发集已经历史 GT 暴露，本轮是匹配内部组件实验，不是独立确认结果。', '',
        'I1 为 source-frozen actionness projection 主读出，I0 为同一空间 boxes 的 Native 次读出。使用原 Native Uniform4 admitted Top1 缓存证据；专家、像素、decoder grid、key/round 噪声一致，各臂保留自己的完整状态链。Query 与 Adam 每次 arrival 重置；LN-only 优化 1536 个坐标且 Query 固定零，joint 优化 1792 个。reset 从真实 source LN 重启；carry 使用 HC 1/16、Vid 1/8 的衰减写回。', '',
        '参数固定：HC lr=.01 / sigma=.025 / tau=.05 / K=40；Vid .03/.1/.25/K=10。没有新 DINO、teacher、参数搜索或生产晋升。资格阶段另计 64 完整 fit、16 full capture function invocations、1600 rounds；正式阶段 128 full source-frozen capture function invocations，四臂共用每次完整函数捕获。全部正式预测及独立部署数学读回通过后才进行后验 GT 评分。', '',
        '聚合先在每个 parent 内平均原两序，再等 parent paired bootstrap 10000 次，seed=20261013。以下绝对值为百分数；差值为 percentage points（pp）。R30/R50 严格使用单 query vIoU > .3/.5，边界等于阈值算未通过。m_sIoU 是固定 GT 支持上的 dense 空间均值。', '',
        '## 四臂完整读出', '']
    for iv in READOUTS:
        lines += [f'### {iv}（'+('主读出' if iv == 'I1' else '次读出')+'）', '',
            '|目标|臂|Frozen mvIoU|Before mvIoU|After mvIoU [95% CI]|Frozen/Before/After R30|Frozen/Before/After R50|After mtIoU|After msIoU|',
            '|---|---|---:|---:|---|---|---|---:|---:|']
        for ds in DATASETS:
            block = s['datasets'][ds]['paired_both_orders']['fields']
            for arm in ARMS:
                mean = lambda st, met: 100*block[f'{iv}_{st}_{met}']['arms'][arm]['mean']
                r30 = '/'.join(number(mean(st, 'R30')) for st in STATES)
                r50 = '/'.join(number(mean(st, 'R50')) for st in STATES)
                lines.append(f"|{LABELS[ds]}|{arm}|{number(mean('Frozen','v'))}|{number(mean('Before','v'))}|"
                    f"{interval(block[f'{iv}_After_v']['arms'][arm],False)}|{r30}|{r50}|"
                    f"{number(mean('After','t'))}|{number(mean('After','s'))}|")
        lines.append('')
    lines += ['## 六个预先固定的结构对比', '',
        'remove_QR_carry=LN_carry−joint_carry；remove_carry_joint=joint_reset−joint_carry；remove_carry_LN=LN_reset−LN_carry；remove_QR_reset=LN_reset−joint_reset；remove_both=LN_reset−joint_carry；interaction=(LN_reset−joint_reset)−(LN_carry−joint_carry)。正值表示该明确对比下的前者较高，不预设删除组件成功。', '',
        '|目标|对比|I1 After mvIoU 差值 pp [95% CI]|I0 差值 pp [95% CI]|I1 正/负/中性/未知 parent|I1 >5/>20pp 损害 parent|原 order1/order2 I1 差值 pp|',
        '|---|---|---|---|---|---|---|']
    for ds in DATASETS:
        v = s['datasets'][ds]
        for key in CONTRASTS:
            e = v['paired_both_orders']['fields']['I1_After_v']['contrasts'][key]
            e0 = v['paired_both_orders']['fields']['I0_After_v']['contrasts'][key]
            signs = '/'.join(str(e[k]) for k in ('positive','negative','neutral','unknown'))
            orders = '/'.join(number(100*v['orders'][o]['fields']['I1_After_v']['contrasts'][key]['mean']) for o in ORDERS)
            lines.append(f"|{LABELS[ds]}|{key}|{interval(e)}|{interval(e0)}|{signs}|{e['harm_gt5pp']}/{e['harm_gt20pp']}|{orders}|")
    lines += ['', '## 每序与全部 parent', '',
        '|目标|order|臂|I1 After mvIoU [95% CI]|I0 After mvIoU [95% CI]|I1 total/current/inherited 增量 pp|',
        '|---|---|---|---|---|---|']
    for ds in DATASETS:
        for o in ORDERS:
            b = s['datasets'][ds]['orders'][o]['fields']
            for a in ARMS:
                delta = '/'.join(number(100*b[f'I1_delta_{x}_v']['arms'][a]['mean']) for x in ('total','current','inherited'))
                lines.append(f"|{LABELS[ds]}|{o}|{a}|{interval(b['I1_After_v']['arms'][a],False)}|{interval(b['I0_After_v']['arms'][a],False)}|{delta}|")
    lines += ['', 'FIT_ROWS.csv 保留全部 512 匿名 arrival；PARENT_MEANS.csv 保留全部 256 parent×arm 两序均值。METRICS.csv/CONTRASTS.csv 覆盖所有 v/t/s/R30/R50、total/current/inherited 增量、严格阈值变化、各域/合并/每序，且保留所有正、负、中性和未知计数。没有根据 GT 删除 parent 或选择最佳 step。', '',
        '## 实际 instrumented 成本', '',
        '计量 scope 修订 capture_scope001：原始字段 actual_frozen_capture_forwards=1 保持不变，语义是一次 frozen_forward 完整 capture 函数调用（invocation），不是单次 model(...)。每次 capture 包含 offset 0/1 两个 source model forwards，records 长度为 2。因此资格 16 + 正式 128 次完整 capture 函数调用对应 32 + 256 次 source-offset model forwards。额外 decoder replay 调用不由该字段计数，不能把 32+256 称为全部 model/module calls。COSTS.csv 对原字段的 units 明确为 full_capture_function_invocations。', '',
        '|目标|臂|实际 fits/rounds/backwards|空监督 fit / no-information rounds|active 坐标|instrumented fit 秒/fit|allocated capture 总秒/full function invocations|decoder cache reuse capture|',
        '|---|---|---|---|---|---:|---|---|']
    for ds in DATASETS:
        for a,c in s['datasets'][ds]['paired_both_orders']['compute'].items():
            ins = c['instrumented']
            get = lambda k,x: ins.get(k,{}).get(x)
            lines.append(f"|{LABELS[ds]}|{a}|{c['actual_complete_fits']}/{c['actual_complete_rounds']}/{c['actual_backwards']}|"
                f"{c['empty_fits']}/{c['no_information_rounds']}|{','.join(map(str,c['active_parameters']))}|"
                f"{number(get('instrumented_fit_seconds','mean'),6)}|{number(get('frozen_capture_seconds','total'),6)}/"
                f"{number(get('actual_frozen_capture_forwards','total'),0)}|{number(get('decoded_frame_reuse','total'),0)}|")
    lines += ['',
        f"正式 controller 从开始到全球 seal 的 elapsed_seconds={number(barrier['elapsed_seconds'],6)}；这个整体计时与各 instrumented 子项有不同范围，不能相加或替代对方。", '',
        'instrumented_fit_seconds 包括原 OPD fit 及实际记录/算术检查的开销，不是未插桩部署速度。共享 frozen capture 的时间、完整函数调用计数和 decode cache reuse 仅按 capture_owner 分配一次；0 分配给其他臂表示同次捕获共用，不表示这些臂独立部署可以免费读取视频。缓存专家来自已验证旧输入，actual_new_DINO_calls=0 不计入旧专家生成成本，也不证明新视频专家成本为零。CSV 保留计量缺失的 unknown，禁止把缺失计量填零。', '',
        'CUDA_peak_allocated 是每个 fit 的峰值字节数；COSTS.csv 保留每 fit 峰值的均值但不公开可加总 total。active 坐标同样是优化范围的每 fit 计量，不累加为资源成本。逐 fit 原标量保留在 FIT_ROWS.csv；缓存复用、同步与 trace/audit 插桩属于本次实际测量范围，不能由这些秒数证明最终端到端 deployment speed。', '',
        '## 证据与决策边界', '']
    for key in CONTRASTS:
        e=s['pooled']['fields']['I1_After_v']['contrasts'][key]
        lines.append(f"- 合并 64 parent，{key}：{interval(e)} pp；{evidence_label(e)}。")
    lines += ['',
        '本轮保留 OPD 核心，结构决策限定在 QR×carry 四臂。CI 跨零不等效，不足以证明组件可删；要据成本决定精简，仍需预先指定可接受任务损失的非劣界。即使某域探索性 CI 全正，也不能按测试域逐域选择结构赢家。两序仅提供顺序敏感性，不代表完整流顺序分布。正式保留/删除和后续有限调参须以统一主方法目标作出；本报告不自动删除 QR、carry 或晋升任何臂。', '',
        'Before 是该臂自身状态链的进入读出，After−Before 是当前 query 写入效果，Before−Frozen 是继承影响，After−Frozen 是总影响；这些增量代数相加，不能用当前 query 改善替代累计表现。I1/I0 使用同一 boxes 且都是固定 source 时间读出，空间组件不会在本轮重新训练时间模块。GT 只用于封存后的评估，不用于监督、准入、排序、时间选择或 checkpoint/step 选择。', '',
        '该开发名单有历史 GT 暴露；本轮没有使用独立确认 cohort，因此不能宣称独立泛化或等效。本轮 6 对比及大量次指标的 CI 未做多重比较校正，属于探索性区间。匿名算术读回精确复现是可复现性检查，不是新增实验或新增独立数据。公开完整匿名结果保留负结果、中性和未知；真实媒体、GT 坐标、完整数学向量、模型状态和 query 文本保持私有。', '',
        '来源：protocols/stvg_opd_p7a_component_simplification_v1.md；scripts/score_stvg_opd_p7a_component_v1.py；同目录 SUMMARY.json、ROWS.jsonl.gz、PARENT_ROWS.jsonl.gz。实际部署数学和后验独立 dense/匿名统计读回分开完成。']
    return '\n'.join(lines)+'\n'


def _table(ax, data, cols, title, *, fontsize=8.1, widths=None):
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold', pad=8)
    table=ax.table(cellText=data,colLabels=cols,loc='upper center',cellLoc='center',
                   colWidths=widths,bbox=[0,0,1,.95])
    table.auto_set_font_size(False);table.set_fontsize(fontsize)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('#d5dce2');cell.set_linewidth(.45)
        if i==0:cell.set_facecolor('#1e3a4e');cell.get_text().set_color('white');cell.get_text().set_weight('bold')
        else:cell.set_facecolor('#f0f4f7' if i%2==0 else 'white')
    return table


def render_pdf(s, pp, path):
    """Three fixed English pages; anonymous source matrices only."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    path=Path(path);assert not path.exists()
    path.parent.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    size=(11.69,8.27)
    def page(title, subtitle, n):
        fig=plt.figure(figsize=size,facecolor='white')
        fig.text(.04,.958,title,fontsize=17,fontweight='bold',color='#17384d')
        fig.text(.04,.924,subtitle,fontsize=9,color='#40576b')
        fig.text(.04,.025,'P7-A OPD internal factorial | 64 historically exposed development parents | No automatic structural deletion',fontsize=8,color='#40576b')
        fig.text(.96,.025,f'{n} / 3',ha='right',fontsize=8,color='#40576b')
        return fig
    with PdfPages(path) as pdf:
        fig=page('OPD internal components: Query Residual x LN history',
            '512 actual complete fits; 12800 recorded rounds; four matched arms; both original development orders',1)
        for y,iv,title in [(.515,'I1','Primary: source-frozen actionness projection I1'),(.17,'I0','Secondary: Native I0, evaluated on the same spatial boxes')]:
            ax=fig.add_axes([.04,y,.92,.29]);data=[]
            for ds in DATASETS:
                b=s['datasets'][ds]['paired_both_orders']['fields']
                for a in ARMS:
                    vals=[f"{100*b[f'{iv}_{st}_{m}']['arms'][a]['mean']:.3f}" for m in ('v','R30','R50') for st in STATES]
                    data.append([LABELS[ds]+' / '+ARM_LABELS[a],*vals])
            _table(ax,data,['Target / arm','F v','B v','A v','F R30','B R30','A R30','F R50','B R50','A R50'],title,widths=[.25]+[.08333]*9)
        fig.text(.04,.855,'F / B / A = Frozen / Before / After. Absolute values in percent. R30/R50 use strict per-query vIoU > 0.3 / 0.5.',fontsize=9)
        fig.text(.04,.10,'QR and Adam reset at every query; LN-only freezes QR at zero (1536 optimizer coordinates); joint uses 1792.',fontsize=9)
        fig.text(.04,.075,'HC: lr .01, sigma .025, tau .05, K40, carry 1/16. Vid: .03 / .1 / .25 / K10, carry 1/8. No parameter search.',fontsize=9)
        fig.text(.04,.05,'Native Uniform4 admitted Top1 cached evidence and original key/round noise; each arm maintains its own state chain.',fontsize=9)
        pdf.savefig(fig);plt.close(fig)
        fig=page('Six fixed contrasts: paired task effects and every tail',
            'Difference is the specified first structure minus reference; 10000 parent bootstrap resamples, seed 20261013',2)
        data=[]
        for ds in DATASETS:
            b=s['datasets'][ds]
            for name in CONTRASTS:
                e=b['paired_both_orders']['fields']['I1_After_v']['contrasts'][name]
                e0=b['paired_both_orders']['fields']['I0_After_v']['contrasts'][name]
                signs='/'.join(str(e[k]) for k in ('positive','negative','neutral','unknown'))
                ov='/'.join(f"{100*b['orders'][o]['fields']['I1_After_v']['contrasts'][name]['mean']:+.3f}" for o in ORDERS)
                data.append([LABELS[ds],SHORT[name],interval(e),interval(e0),signs,
                             f"{e['harm_gt5pp']}/{e['harm_gt20pp']}",ov])
        _table(fig.add_axes([.04,.32,.92,.52]),data,
            ['Target','Contrast','I1 delta pp [95% CI]','I0 delta pp [95% CI]','+ / - / 0 / ?','Harm >5/>20 pp','Order1 / order2 pp'],
            'After mvIoU; both orders averaged within each of 32 parents per target',fontsize=7.7,
            widths=[.10,.19,.20,.20,.08,.10,.13])
        fig.text(.04,.25,'Remove QR / carry: LN carry - joint carry; remove carry / joint: joint reset - joint carry.',fontsize=9)
        fig.text(.04,.225,'Remove carry / LN: LN reset - LN carry; remove QR / reset: LN reset - joint reset; remove both: LN reset - joint carry.',fontsize=9)
        fig.text(.04,.20,'Interaction = (LN reset - joint reset) - (LN carry - joint carry). No contrast presumes that deletion succeeds.',fontsize=9)
        fig.text(.04,.14,'All signs and unknowns are retained. Severe harms use strict delta < -0.05 / -0.20, i.e. more than 5 / 20 pp.',fontsize=9)
        fig.text(.04,.115,'These unadjusted exploratory intervals are conditional on historically exposed development sources.',fontsize=9)
        fig.text(.04,.09,'A CI crossing zero is not equivalence. Cost-based simplification needs a predeclared noninferiority margin.',fontsize=9,fontweight='bold')
        fig.text(.04,.065,'Do not select a different structure for each target domain. The OPD core remains fixed; no production promotion.',fontsize=9)
        pdf.savefig(fig);plt.close(fig)
        fig=page('All anonymous parent effects and actual instrumented cost',
            'No parent removed; color shows I1 After-mvIoU contrast in percentage points; full parent and order data accompany the PDF',3)
        matrices=[]
        for ds in DATASETS:
            table={(r['parent_id'],r['arm']):r for r in pp if r['dataset']==ds}
            ids=sorted({p for p,a in table})
            matrices.append(np.array([[100*math.fsum(c*table[p,a]['I1_After_v'] for a,c in coeff.items()) for p in ids] for coeff in CONTRASTS.values()]))
        limit=max(1e-8,max(float(np.max(np.abs(x))) for x in matrices))
        for i,ds in enumerate(DATASETS):
            ax=fig.add_axes([.20,.67-i*.24,.74,.18]);im=ax.imshow(matrices[i],aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
            ax.set_yticks(range(6),[SHORT[k] for k in CONTRASTS]);ax.tick_params(axis='y',labelsize=8)
            ax.set_xticks([0,7,15,23,31],[1,8,16,24,32]);ax.set_xlabel('Anonymous parent order within fixed target roster',fontsize=8)
            ax.set_title(LABELS[ds],loc='left',fontsize=10,fontweight='bold')
        cax=fig.add_axes([.95,.44,.012,.39]);fig.colorbar(im,cax=cax)
        cax.set_title('pp',fontsize=8,pad=5);cax.tick_params(labelsize=8)
        data=[]
        for ds in DATASETS:
            for arm,c in s['datasets'][ds]['paired_both_orders']['compute'].items():
                ins=c['instrumented'];get=lambda k,x:ins.get(k,{}).get(x)
                data.append([LABELS[ds]+' / '+ARM_LABELS[arm],str(c['actual_complete_fits']),str(c['actual_complete_rounds']),str(c['actual_backwards']),str(c['empty_fits']),
                    '/'.join(map(str,c['active_parameters'])),number(get('instrumented_fit_seconds','mean'),4),number(get('frozen_capture_seconds','total'),3),number(get('actual_frozen_capture_forwards','total'),0)])
        cost_table=_table(fig.add_axes([.04,.14,.92,.215]),data,['Target / arm','Fits','Rounds','Backwards','Empty','Active','Fit sec / fit','Capture total sec','Full capture invocations'],
            'Actual logged cost; full capture function invocations allocated once by capture_owner',fontsize=7.8,widths=[.25,.05,.065,.08,.055,.065,.11,.19,.135])
        cost_table[(0,8)].get_text().set_fontsize(7.0)
        fig.text(.04,.11,'Fit seconds include trace/audit/synchronization and cache reuse; cached teacher generation and serialization are outside this sub-timing.',fontsize=8.2)
        fig.text(.04,.089,'16 qualification + 128 formal full capture invocations; two source offset forwards each = 32 + 256 source-offset model forwards.',fontsize=8.2)
        fig.text(.04,.068,'Extra decoder replay calls are outside this counter. Other arms share owner captures; zero allocation does not imply free video access.',fontsize=8.2)
        fig.text(.04,.047,'Before is each arm own incoming state. After-Before: current effect; Before-Frozen: inherited effect; After-Frozen: total.',fontsize=8.2)
        pdf.savefig(fig);plt.close(fig)
    return path


def report():
    barrier, root, scalar=validate_inputs()
    rows=rows_read(OUT/'ROWS.jsonl.gz');assert public_rows_check(rows)==512
    actual=read(OUT/'SUMMARY.json');calculated=summary(rows);near(actual,calculated)
    pp=parent_rows(rows);near(pp,rows_read(OUT/'PARENT_ROWS.jsonl.gz'));assert len(pp)==256
    outputs=[OUT/'REPORT.md',OUT/'METRICS.csv',OUT/'CONTRASTS.csv',OUT/'COSTS.csv',
             OUT/'FIT_ROWS.csv',OUT/'PARENT_MEANS.csv',OUT/'figures/P7A_COMPONENT_FACTORIAL.pdf']
    assert not any(p.exists() for p in outputs), 'Existing report outputs are preserved; an explicit revision is required'
    for p in outputs:p.parent.mkdir(parents=True,exist_ok=True)
    metric,contrast,cost=tables(actual)
    write_csv(outputs[1],metric);write_csv(outputs[2],contrast);write_csv(outputs[3],cost)
    write_csv(outputs[4],[flatten_row(r) for r in rows]);write_csv(outputs[5],[flatten_row(r) for r in pp])
    outputs[0].write_text(markdown(actual,barrier),encoding='utf-8')
    render_pdf(actual,pp,outputs[6])
    receipt=dict(status='authored_pending_actual_all_page_pixel_review',PDF_pages=3,
        capture_scope_revision='capture_scope001',
        full_capture_function_invocations=dict(qualification=16,formal=128),
        source_offset_model_forwards=dict(qualification=32,formal=256),
        extra_decoder_replay_calls_outside_capture_counter=True,
        root_readback_sha256=sha(BASE/'ROOT_READBACK.json'),
        anonymous_scalar_readback_sha256=sha(BASE/'ROOT_ANONYMOUS_SCALAR_READBACK.json'),
        csv_rows=dict(METRICS=len(metric),CONTRASTS=len(contrast),COSTS=len(cost),FIT_ROWS=512,PARENT_MEANS=256),
        files={str(p.relative_to(ROOT)):sha(p) for p in outputs},
        anonymous_only=True,GT_read=False,media_read=False,fit_read=False,model_calls=0,
        automatic_structural_deletion=False,production_promoted=False,time=time.time())
    write(BASE/'REPORT_EXPORT.json',receipt)
    print('P7A_COMPONENT_REPORT_THREE_PAGES_PENDING_PIXEL_REVIEW',flush=True)
    return receipt


if __name__=='__main__':
    assert len(sys.argv)==1 or sys.argv[1]=='report'
    report()
