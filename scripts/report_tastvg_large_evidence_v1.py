"""Source-backed report and scientific figures; no inference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'results/tastvg_large_correction_evidence/2026-10-03'

def read(p):return json.loads(Path(p).read_text())
def fmt(z,scale=100):
    if z['mean'] is None:return 'undefined'
    a,b=z['ci95'];return f"{scale*z['mean']:.2f} [{scale*a:.2f}, {scale*b:.2f}]"

def main():
    panels={f'{ds}/{sp}':(read(F/sp/ds/'SUMMARY.json'),read(F/sp/ds/'DISCRIMINATION.json'))
        for sp in ['search','confirm'] for ds in ['vidstg','hc2']}
    decision=read(F/'DECISION.json');audit=read(F/'VALIDATION.json');seal=read(F/'SIGNAL_SEAL.json')
    lines=['# 大幅 temporal correction：边界证据审计','',decision['conclusion'],'',
        '固定前一轮 A8 / Expanded32、A 空间轨迹、Uniform 持久更新和专家到达位置。'
        '两集各32开发＋16确认来源，一query/source、两序、clean＋五类5% corruption、25%专家；'
        '1,152到达、288专家（240corrupt、48clean），864非专家仍是原A。来源全部历史曝光，'
        '本轮是机制开发诊断，不是fresh test或新方法全量评估。','',
        '## 实际信号与证据边界','',
        '- **N：冻结 TA 原生起止先验。** 同输入的source-checkpoint两offset各自softmax后等质量交织，'
        '候选分数为起止log概率之和。在线日志没有完整A-current logits；N不能冒充它。'
        'cross-offset边界配对也不是TA原生envelope的联合概率。',
        '- **U：UVTG proposal边界密度。** 全proposal等质量Gaussian起止边界密度，固定带宽0.5秒；'
        '无confidence、无interval-IoU配对、无PE语义曲线。边缘分布仍可能组合不兼容的两端。',
        '- **S：两视图边界密度稳健性。** 用已缓存phase0/.25秒真实采样，取同一A8相对总分增量的'
        '两视图最小值。它是同模型的稳健性诊断，不是独立专家或定位正确保证；旧T使用confidence×IoU，'
        '此处只使用边界密度。','',
        '所有分数、几何分组和选择先seal，才关联已公开的逐候选dense GT指标；没有新增GT标注读取、'
        '模型调用、专家调用、backbone或decoder replay。确认结果不参与选择规则。', '',
        f"冻结源native与A的pre-Fast native一致 {seal['checks']['source_native_agrees_A_native']}/288，"
        f"与最终A8一致 {seal['checks']['source_native_agrees_A8']}/288；"
        f"真实采样位置发生变化 {seal['checks']['real_two_view_changed']}/288。", '',
        '三类信号在相同支持上用同一预锁读出：只有唯一最高分且严格优于原A8才替换；任何top tie或'
        '无增量都保留A8。不按GT调阈值、带宽或组合权重。', '',
        '## 相同支持上的最终读出','',
        '下表是corrupt专家子集；vIoU和增量均为pp，配对95%区间使用10,000次source bootstrap。'
        '括号中的regret是同支持oracle减实际选择。', '',
        '| 面板 | A8 | O32 | N8−A8 | N32−A8 | U8−A8 | U32−A8 | S8−A8 | S32−A8 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p,(z,_) in panels.items():
        m=z['corruption']['expert']['metrics'];lines.append('| '+p+f" | {100*m['A8_v']['mean']:.2f} | {100*m['O32_v']['mean']:.2f} | "+
            ' | '.join(fmt(m[a+'_gain']) for a in ['N8','N32','U8','U32','S8','S32'])+' |')
    lines += ['', '全corruption流含75%不变的非专家；以下保留全流稀释、tIoU、gross gain/loss和负尾。','',
        '| 面板 / 信号32 | 全流vIoU增量 | 专家tIoU增量 | 专家regret | 专家gross gain / loss | 改善 / 受损 | >5pp损害 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p,(z,_) in panels.items():
        for s in ['N','U','S']:
            arm=s+'32';e=z['corruption']['expert'];m=e['metrics'];tr=e['transitions'][arm]
            lines.append(f"| {p} / {s} | {fmt(z['corruption']['all']['metrics'][arm+'_gain'])} | "
                f"{fmt(m[arm+'_t_gain'])} | {100*m[arm+'_regret']['mean']:.2f} | "
                f"{100*m[arm+'_gross_gain']['mean']:.2f} / {100*m[arm+'_gross_loss']['mean']:.2f} | "
                f"{tr['improved']} / {tr['harmed']} | {tr['severe_harm_gt5pp']} |")
    lines += ['', '## 是否识别有益的大幅修改','',
        '大幅修改固定为端点L1位移/A8时长≥.5。b=2min(|ds|,|de|)/L1≤.25归为单端主导，'
        '主导起点向内为trim-start、主导终点向内为trim-end，向外为expand；其余同向变化为shift。'
        '双端向内而平衡者保留trim-both，same另列，避免强行四分。连续ds/de/b/r均已发布。'
        '这些是描述性分组，不是线上接受gate。','',
        '二元标签是候选vIoU相对**同一A8**改善或损害，零差值不作为正/负类。正证据是信号分数'
        '严格高于A8；先在到达内平均候选计数比例，再按条件、顺序、来源汇总。precision/recall'
        '用共享source抽样的分子/分母；AUC只在同时有正、负候选的到达上定义，缺类不补.5。', '',
        '| 面板 / 信号 | large：AUC / 有效到达 | 接受的benefit precision | beneficial recall | harmful acceptance |',
        '|---|---:|---:|---:|---:|']
    for p,(_,d) in panels.items():
        for s in ['N','U','S']:
            z=d['corruption']['large'][s];am=z['auc']['metrics'].get('auc');rt=z['ratios']
            lines.append(f"| {p} / {s} | {fmt(am,1) if am else 'undefined'} / {z['eligible_auc_cells']} | "+
                ' | '.join(fmt(rt[k]) if k in rt else 'undefined' for k in ['precision','benefit_recall','harm_acceptance'])+' |')
    lines += ['', '大幅裁剪单列（每个信号候选群完全相同；多个候选不会膨胀成独立样本数）：','',
        '| 面板 / 分组 / 信号 | 非零候选 / 正 / 负 | AUC有效到达 / 来源 | AUC | 有益precision | 有害acceptance |',
        '|---|---:|---:|---:|---:|---:|']
    for p,(_,d) in panels.items():
        for group in ['large_trim_start','large_trim_end']:
            for s in ['N','U','S']:
                z=d['corruption'][group][s];q=z['raw_counts'];a=z['auc'];am=a['metrics'].get('auc');rt=z['ratios']
                lines.append(f"| {p} / {group} / {s} | {q['candidates']} / {q['positives']} / {q['negatives']} | "
                    f"{a['cells']} / {a['sources']} | {fmt(am,1) if am else 'undefined'} | "
                    f"{fmt(rt['precision']) if 'precision' in rt else 'undefined'} | "
                    f"{fmt(rt['harm_acceptance']) if 'harm_acceptance' in rt else 'undefined'} |")
    lines += ['', '所有分组含expand/shift/trim-both、原始TP/FP/FN/TN、缺类/零分母、两序和'
        'leave-one-source-out结果见DISCRIMINATION/SUMMARY。AUC是同到达的benefit-vs-harm判别，'
        '不是跨支持pairwise提升，不证明校准或extreme-tail原因。','',
        '## Clean与正确结果保护','',
        '| 面板 / 信号32 | clean全流vIoU增量 | corrupt专家.3错误→正确 / 正确→错误 | .5错误→正确 / 正确→错误 | 原有Fast正收益被破坏 |',
        '|---|---:|---:|---:|---:|']
    for p,(z,_) in panels.items():
        for s in ['N','U','S']:
            arm=s+'32';tr=z['corruption']['expert']['transitions'][arm];c=tr['correctness'];m=z['corruption']['expert']['metrics']
            destroy=m[arm+'_destroyed_old_fast']['cell_mean']*z['corruption']['expert']['cells']
            lines.append(f"| {p} / {s} | {fmt(z['clean']['all']['metrics'][arm+'_gain'])} | "
                f"{c['0.3']['rescued']} / {c['0.3']['destroyed']} | {c['0.5']['rescued']} / {c['0.5']['destroyed']} | {destroy:.0f} |")
    lines += ['', '## 代表性正负例','',
        '以下例子按预定最大增益/最大损害读出；区间是观测窗口归一化候选，不公开GT坐标。','',
        '| 面板 / 信号 / source / corruption | A8 → selected | 类别 / r | vIoU变化pp |',
        '|---|---|---:|---:|']
    ev={e['cell_key']:e for e in read(F/'EVIDENCE_ROWS.json')}
    for ds in ['vidstg','hc2']:
        cases=read(F/'confirm'/ds/'CASES.json')
        for s in ['N','U','S']:
            for sign in ['positive','negative']:
                r=cases[s][sign][0];e=ev[r['evidence_key']];a=e['anchor_index'];i=e['choices'][s+'32'];g=e['geometry'][i]
                aa=e['intervals'][a];bb=e['intervals'][i]
                lines.append(f"| {ds}/confirm / {s} / {r['source_id']} / {r['condition']} | "
                    f"[{aa[0]:.3f},{aa[1]:.3f}] → [{bb[0]:.3f},{bb[1]:.3f}] | "
                    f"{g['kind']} / {g['radius']:.2f} | {100*r[s+'32_gain']:+.2f} |")
    lines += ['', '## 决策与限制','']+decision['interpretation']+['',
        '本轮三信号是预锁的探索性机制测试，未作多重比较校正；来源少、共享专家错误、候选重复、'
        '缺少current-state logits和边缘密度可能配出不存在的span，均限制外推。相关性或某个均值正向'
        '不等于一个已验证的安全selector。本轮不晋升A、不启动额外expert/局部refiner/历史队列。', '',
        f"独立公开审计通过：{audit['checks']['numeric_checks']}标量检查、"
        f"{audit['checks']['decisions']}选择、{audit['checks']['candidate_geometry']}候选几何；"
        f"最大数值差 {audit['max_numeric_error']:.3g}。7个CPU边界测试通过。", '',
        '[执行协议](../protocols/tastvg_large_correction_evidence_v1.md) · '
        '[匿名结果](../results/tastvg_large_correction_evidence/2026-10-03)']
    (ROOT/'docs/TA_LARGE_CORRECTION_EVIDENCE_REVIEW.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(1,2,figsize=(10,3.4),sharey=True)
    colors=['#4069A6','#CF8947','#5C967C']
    for j,ds in enumerate(['vidstg','hc2']):
        z=panels[ds+'/confirm'][0]['corruption']['all']['metrics']
        for i,s in enumerate(['N','U','S']):
            m=z[s+'32_gain'];y=100*m['mean'];lo,hi=np.array(m['ci95'])*100
            ax[j].errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='o',capsize=4,color=colors[i],markersize=7)
        ax[j].axhline(0,color='#888',lw=.8);ax[j].set_xticks(range(3),['Frozen native','UVTG boundaries','Two views'])
        ax[j].set_title({'vidstg':'VidSTG','hc2':'HC-STVG-v2'}[ds]);ax[j].spines[['top','right']].set_visible(False)
        ax[j].grid(axis='y',alpha=.2)
    ax[0].set_ylabel('Corrupt full-flow delta vIoU (pp)');fig.tight_layout()
    fig.savefig(F/'readout_confirmation.png',dpi=230);fig.savefig(F/'readout_confirmation.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.4),sharey=True)
    for j,ds in enumerate(['vidstg','hc2']):
        d=panels[ds+'/confirm'][1]['corruption']['large']
        for i,s in enumerate(['N','U','S']):
            m=d[s]['auc']['metrics'].get('auc')
            if m:
                y=m['mean'];lo,hi=m['ci95'];ax[j].errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='o',capsize=4,color=colors[i],markersize=7)
        ax[j].axhline(.5,color='#888',lw=.8,ls='--');ax[j].set_ylim(0,1);ax[j].set_xticks(range(3),['Frozen native','UVTG boundaries','Two views'])
        ax[j].set_title({'vidstg':'VidSTG','hc2':'HC-STVG-v2'}[ds]);ax[j].spines[['top','right']].set_visible(False);ax[j].grid(axis='y',alpha=.2)
    ax[0].set_ylabel('Large-change benefit / harm AUC');fig.tight_layout()
    fig.savefig(F/'large_change_discrimination.png',dpi=230);fig.savefig(F/'large_change_discrimination.pdf');plt.close(fig)
    print('LARGE_EVIDENCE_REPORT_AND_FIGURES_COMPLETE')

if __name__=='__main__':main()
