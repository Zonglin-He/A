"""CPU-only post-hoc diagnosis of the complete frozen VidSTG ablations."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write


def describe(values):
    a = np.asarray(values, float)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p10=float(np.quantile(a,.1)), p90=float(np.quantile(a,.9)))


def diagnose():
    root = ROOT/'outputs/tubedetr/hc2_to_vid'
    seal = json.loads((root/'sealed.json').read_text())
    metric_path = ROOT/'outputs/tubedetr_ablation/hc2_to_vid/metrics_five_rows.json'
    receipt = json.loads(metric_path.with_name('receipt.json').read_text())
    if (seal['queries'] != 10303 or receipt['prediction_seal_sha256'] != sha(root/'sealed.json')
            or receipt['metrics_sha256'] != sha(metric_path)):
        raise ValueError('Only complete, immutable scored populations can be diagnosed')
    metrics = json.loads(metric_path.read_text())
    rows = metrics['rows']
    if len(rows) != 10303 or {r['id'] for r in rows} != set(seal['files']):
        raise ValueError('Diagnostic rows differ from complete seal')
    summaries, detailed = {}, []
    for row in rows:
        entry = seal['files'][row['id']]
        prediction, cost = root/entry['path'], root/entry['receipt_path']
        if sha(prediction) != entry['sha256'] or sha(cost) != entry['receipt_sha256']:
            raise ValueError('Frozen prediction or receipt changed')
        saved = json.loads(cost.read_text())
        init = saved['initialization']
        with np.load(prediction, allow_pickle=False) as z:
            source, e, live = [z[k].astype(float) for k in ['Source','E_Ref','Live']]
            observed=np.zeros(len(source),dtype=bool);observed[z['positions8'].astype(int)]=True
            centers=np.linalg.norm(live[:,:2]-e[:,:2],axis=1)
            drift = dict(init_center=float(np.linalg.norm(e[:,:2]-source[:,:2],axis=1).mean()),
                opd_center=float(np.linalg.norm(live[:,:2]-e[:,:2],axis=1).mean()),
                opd_observed_center=float(centers[observed].mean()),
                opd_unobserved_center=float(centers[~observed].mean()) if (~observed).any() else None,
                init_logsize=float(np.abs(np.log(e[:,2:]/source[:,2:])).mean()),
                opd_logsize=float(np.abs(np.log(live[:,2:]/e[:,2:])).mean()))
        v = row['values']
        def delta(a,b,k):
            return None if v[a][k] is None or v[b][k] is None else 100*(v[a][k]-v[b][k])
        detailed.append(dict(id=row['id'], parent=row['parent'], support=init['support_frames'],
            delta=init['delta'], updates=saved['cost']['actual_updates'], **drift,
            source_sIoU=v['Source']['m_sIoU'],
            initialization_v_pp=delta('WithoutOPD','Temporal','m_vIoU'),
            opd_v_pp=delta('Full','WithoutOPD','m_vIoU'),
            initialization_s_pp=delta('WithoutOPD','Temporal','m_sIoU'),
            opd_s_pp=delta('Full','WithoutOPD','m_sIoU'),
            initialization_observed_pp=delta('WithoutOPD','Temporal','observed_actual_sIoU'),
            initialization_unobserved_pp=delta('WithoutOPD','Temporal','unobserved_actual_sIoU')))
    def group(sub):
        fields = ['initialization_v_pp','opd_v_pp','initialization_s_pp','opd_s_pp',
                  'initialization_observed_pp','initialization_unobserved_pp']
        return dict(queries=len(sub), **{k:describe([r[k] for r in sub if r[k] is not None])
            for k in fields if any(r[k] is not None for r in sub)},
            initialization_loss_over5_pp=sum(r['initialization_v_pp'] < -5 for r in sub),
            opd_loss_over5_pp=sum(r['opd_v_pp'] < -5 for r in sub))
    for support in sorted({r['support'] for r in detailed}):
        summaries[str(support)] = group([r for r in detailed if r['support']==support])
    result = dict(status='complete_frozen_VidSTG_posthoc_diagnosis', queries=10303, parents=732,
        prediction_seal_sha256=sha(root/'sealed.json'), metrics_sha256=sha(metric_path),
        no_new_model_or_expert_forward=True, no_parameter_candidate_evaluated=True,
        existing_five_row_metrics=metrics['datasets']['vidstg']['query_means'],
        paired=metrics['datasets']['vidstg']['paired'], overall=group(detailed),
        by_consensus_support=summaries,
        by_source_spatial_quality={str((lo,hi)):group([r for r in detailed if lo<=r['source_sIoU']<hi])
            for lo,hi in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.000001)]},
        coordinate_drift={k:describe([r[k] for r in detailed])
            for k in ['init_center','opd_center','init_logsize','opd_logsize']},
        opd_observation_propagation={k:describe([r[k] for r in detailed if r[k] is not None])
            for k in ['opd_observed_center','opd_unobserved_center']},
        update_counts={str(n):sum(r['updates']==n for r in detailed) for n in sorted({r['updates']for r in detailed})},
        observed_improves_unobserved_harms=sum(r['initialization_observed_pp'] is not None
            and r['initialization_unobserved_pp'] is not None and r['initialization_observed_pp']>0
            and r['initialization_unobserved_pp']<0 for r in detailed),
        interpretation=dict(identified='At gamma1 the global consensus initialization accounts for the spatial loss; OPD adds a small positive increment.',
            hypotheses=['gamma1 may transfer too much global correction from eight observations',
                'TA-STVG source calibration may mismatch TubeDETR geometry; not established causally',
                'Small OPD effect does not identify whether lr, tau, steps or decoder sensitivity is responsible'],
            next_checks=['after all original experiments finish: choose gamma on fixed128 development inputs',
                'inspect pseudo-target loss, entropy/ESS, gradients and displacement during real128 replay',
                'sequential lr, then tau, then steps; reason and full signed outcomes recorded at each stage']),
        full_target_results_seen_for_hypothesis_generation=True,
        tuning_and_all_changed_parameter_evaluations_deferred_until_original_main_and_ablation_seals=True)
    # Renderer selection was fixed after full scoring; this is an illustrative
    # worst-case, not a frequency estimate or a development-panel selection.
    case=next(r for r in rows if r['id']=='vidstg-02513-q-00')
    case_saved=json.loads((root/seal['files'][case['id']]['receipt_path']).read_text())
    result['illustrative_failure']=dict(id_sha256=__import__('hashlib').sha256(case['id'].encode()).hexdigest(),
        selection='previous full-population renderer maximum vIoU loss; illustrative posthoc case',
        Source_vIoU=case['values']['Source']['m_vIoU'],
        initialization_Strong_vIoU=case['values']['WithoutOPD']['m_vIoU'],
        Full_vIoU=case['values']['Full']['m_vIoU'],
        consensus_support=case_saved['initialization']['support_frames'],
        consensus_delta=case_saved['initialization']['delta'],
        visual_readout='Source aligns with GT person; Ours moves to the other person hand/instrument area and shrinks. Six supporting observations do not establish correct target identity.',
        no_new_parameter_selection_from_this_case=True)
    write(ROOT/'results/tubedetr/vid_diagnosis.json',result)
    write(ROOT/'outputs/tubedetr_vid_diagnosis/query_diagnostics.json',detailed)
    lines=['# TubeDETR VidSTG 冻结结果离线诊断','',
        '完整10303 query／732父视频；只读取已封存预测和已完成五行指标，没有新模型、专家或参数候选运行。','',
        '| 比较 | m_vIoU差值 pp | 父视频聚类95%区间 pp |','|---|---:|---|']
    for name,value in result['paired'].items():
        lines.append(f"| {name} | {value['query_mean_pp']:.6f} | {value['query_parent_cluster_CI95_pp']} |")
    lines += ['', 'gamma=1的初始化在Strong时间相同条件下造成主要空间损失；现有OPD小幅改善自身初始化，不能把初始化损失归咎于学习率过大。',
        '初始化对所有帧施加同一中心/对数尺寸修正；同八帧共识的支持数、Source质量和框位移分组是事后诊断，不能替代128上单参数干预。',
        '既有最大损失对齐图中，Source已对准GT人物，初始化后转向另一人的手/乐器区域并缩框；六个共识支持帧仍可一致地支持错误目标。该例说明仅提高共识支持数不必然解决身份错误，不能当作错误发生率。',
        '目前没有证据区分学习率、温度、步数和decoder敏感性。待原两方向主实验及五行消融全完成后，先调初始化强度，再顺序验证学习率、温度和步数。',
        '原冻结主表保留，新128开发曝光、候选、选择轨迹及最终全量结果单独保存；不能将已看过的完整目标划分称为未曝光确认集。','']
    (ROOT/'results/tubedetr/vid_diagnosis.md').write_text('\n'.join(lines))
    print(json.dumps({k:result[k] for k in ['overall','coordinate_drift','update_counts','by_consensus_support']},indent=2))


if __name__=='__main__':
    diagnose()
