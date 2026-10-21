"""Independent CPU analysis only after the complete fixed128 ROI pilot seals.

No models, feature extraction, adaptation, hyperparameter selection or GPU jobs.
All GT-dependent strata are descriptive and never used by inference.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write, digest
from tools.layout_trial.diagnostics import paired_ranking, mean

PUBLIC = ROOT / 'results/tubedetr_opd_direct_diagnosis'
OUT = ROOT / 'outputs/tubedetr_opd_direct_diagnosis'
PLAN = PUBLIC / 'plan.json'
PILOT = ROOT / 'results/tubedetr_layout_trial'
PRIVATE = ROOT / 'outputs/tubedetr_layout_trial'
ARMS = ['GeometryOPD', 'Layout1OPD', 'Layout3OPD', 'Layout3Direct', 'GeometryDirect']
METRICS = ['m_vIoU', 'm_sIoU', 'observed_actual_sIoU', 'unobserved_actual_sIoU',
           'vIoU@0.3', 'vIoU@0.5', 'm_tIoU']


def verify_final(path):
    if not path.exists():
        return False
    seal = json.loads(path.read_text())
    if seal['status'] != 'complete':
        raise ValueError('Non-complete terminal seal: ' + str(path))
    for name, expected in seal['files'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Terminal artifact changed: ' + name)
    return True


def cluster_difference(values, parents, repeats=5000):
    """Equal-query estimand, resample whole parents; missing values are excluded."""
    valid = [(float(v), str(p)) for v, p in zip(values, parents)
             if v is not None and np.isfinite(v)]
    if not valid:
        return dict(queries=0, parents=0, mean_pp=None, CI95_pp=None)
    groups = sorted({p for _, p in valid})
    sums = np.array([sum(v for v, q in valid if q == p) for p in groups])
    counts = np.array([sum(q == p for _, q in valid) for p in groups])
    rng = np.random.default_rng(41)
    draws = rng.integers(0, len(groups), (repeats, len(groups)))
    boot = 100 * sums[draws].sum(1) / counts[draws].sum(1)
    v = np.array([v for v, _ in valid])
    return dict(queries=len(v), parents=len(groups), mean_pp=float(100*v.mean()),
                CI95_pp=np.quantile(boot, [.025, .975]).tolist(),
                equal_parent_mean_pp=float(100*np.mean(sums/counts)),
                wins=int((v > 1e-12).sum()), losses=int((v < -1e-12).sum()),
                ties=int((abs(v) <= 1e-12).sum()), below_minus5=int((v < -.05).sum()))


def difference(rows, a, b, metric):
    return [None if r['values'][a][metric] is None or r['values'][b][metric] is None
            else r['values'][a][metric]-r['values'][b][metric] for r in rows]


def needs_research(paired, minimum_pp):
    """A triage rule, not an equivalence test or a significance declaration."""
    v = paired['m_vIoU']
    reasons = []
    if v['mean_pp'] is None or v['mean_pp'] < minimum_pp:
        reasons.append('m_vIoU improvement below pre-outcome practical triage threshold')
    if v['CI95_pp'] is None or v['CI95_pp'][0] <= 0:
        reasons.append('paired parent bootstrap interval does not exclude zero')
    for key in ['m_sIoU', 'unobserved_actual_sIoU']:
        if paired[key]['mean_pp'] is None or paired[key]['mean_pp'] <= 0:
            reasons.append('no positive '+key+' difference')
    return dict(required=bool(reasons), reasons=reasons,
                no_equivalence_or_independent_confirmation_claim=True)


def verified_predictions(arm, ids):
    root = PRIVATE / 'predictions' / arm
    seal = json.loads((root / 'sealed.json').read_text())
    if seal['queries'] != 128 or set(seal['files']) != set(ids):
        raise ValueError('Require exactly the same complete128: '+arm)
    for entry in seal['files'].values():
        for path, field in [('path', 'sha256'), ('receipt_path', 'receipt_sha256')]:
            if sha(root / entry[path]) != entry[field]:
                raise ValueError('Prediction/receipt changed: '+arm)
    return root, seal


def analyze(plan):
    pilot = json.loads((PILOT/'plan.json').read_text())
    if sha(PILOT/'plan.json') != plan['pilot_plan_sha256']:
        raise ValueError('Pilot plan identity changed')
    if sha(ROOT/pilot['manifest']) != pilot['manifest_sha256'] or sha(ROOT/pilot['labels']) != pilot['labels_sha256']:
        raise ValueError('Pilot population/labels changed')
    requests = [json.loads(s) for s in (ROOT/pilot['manifest']).read_text().splitlines()]
    ids = [r['id'] for r in requests]
    if len(ids) != 128 or len(set(ids)) != 128:
        raise ValueError('Only complete128 supported')
    # All predictions, receipts and public metrics are verified before GT is opened.
    predictions = {arm: verified_predictions(arm, ids) for arm in ARMS}
    metrics = {arm: json.loads((PILOT/'metrics'/f'{arm}.json').read_text()) for arm in ARMS}
    maps = {arm: {r['id']: r for r in m['rows']} for arm, m in metrics.items()}
    if any(set(m) != set(ids) for m in maps.values()):
        raise ValueError('Unpaired metric population')
    diagnostic_seal = json.loads((PRIVATE/'diagnostic_seal.json').read_text())
    if diagnostic_seal['queries'] != 128 or set(diagnostic_seal['files']) != set(ids):
        raise ValueError('Unpaired diagnostic bank')
    for entry in diagnostic_seal['files'].values():
        if sha(PRIVATE/entry['path']) != entry['sha256']:
            raise ValueError('Diagnostic bank changed')
    labels = {r['id']: r for r in map(json.loads, (ROOT/pilot['labels']).read_text().splitlines())}
    if set(labels) != set(ids) or len({r['parent'] for r in labels.values()}) != 128:
        raise ValueError('Expected128 unique parents')
    rows, features, private_details, cost_records = [], [], [], {a: [] for a in ARMS}
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    cal = json.loads((PRIVATE/'temperature_calibration.json').read_text())
    candidate_details = []
    for request in requests:
        query = request['id']; meta = request['metadata']; truth = labels[query]
        for arm in ARMS[1:]:
            for control in ['Source', 'Temporal', 'WithoutOPD']:
                if maps[arm][query]['values'][control] != maps[ARMS[0]][query]['values'][control]:
                    raise ValueError('Shared source/time/initialization scores differ')
        row = dict(id=query, parent=truth['parent'], values={a: maps[a][query]['values']['Full'] for a in ARMS})
        rows.append(row)
        boxes = np.array(list(truth['boxes'].values()), dtype=float)
        fids = np.array(list(map(int, truth['boxes'])), dtype=int); order = np.argsort(fids)
        boxes, fids = boxes[order], fids[order]
        area = np.maximum(boxes[:, 2:]-boxes[:, :2], 0).prod(1)/(meta['width']*meta['height'])
        centers = (boxes[:, :2]+boxes[:, 2:])/2/[meta['width'], meta['height']]
        speed = np.linalg.norm(np.diff(centers, axis=0), axis=1)/np.maximum(np.diff(fids)/meta['fps'], 1e-12)
        frame_stats, ranks = [], []
        with np.load(PRIVATE/diagnostic_seal['files'][query]['path'], allow_pickle=False) as bank:
            kinds = bank['candidate_kinds'].tolist(); ti = kinds.index('teacher'); si = kinds.index('source')
            for i, fid in enumerate(bank['observed_frame_ids']):
                if str(int(fid)) not in truth['boxes']:
                    continue
                iou = box_iou(xyxy(bank['boxes'][i], meta['width'], meta['height']), truth['boxes'][str(int(fid))])
                frame_stats.append(dict(teacher_iou=float(iou[ti]), source_iou=float(iou[si]),
                                        oracle_policy_gain=float(iou[:32].max()-iou[si])))
                per = {}
                for reward, tau in [('layout3', cal['layout3']['tau']), ('layout1', cal['layout1']['tau']), ('geometry', pilot['configuration']['tau'])]:
                    scores = bank[reward+'_reward'][i, :32]
                    weights = np.exp((scores-scores.max())/tau); weights /= weights.sum()
                    per[reward] = {**paired_ranking(scores, iou[:32]),
                        'top1_IoU_regret': float(iou[:32].max()-iou[int(scores.argmax())]),
                        'weighted_IoU_gain_over_Source': float(weights@iou[:32]-iou[si]),
                        'normalized_entropy': float(-(weights*np.log(np.maximum(weights, 1e-300))).sum()/np.log(32)),
                        'ESS': float(1/(weights@weights)), 'reward_std': float(scores.std())}
                ranks.append(per)
        scalar = dict(query_kind=str(meta.get('kind', 'unavailable')), observations_with_GT=len(frame_stats),
            mean_target_area_fraction=float(area.mean()), event_seconds=float((truth['interval'][1]-truth['interval'][0])/meta['fps']),
            event_fraction=float((truth['interval'][1]-truth['interval'][0])/(meta['end_frame']-meta['start_frame']+1)),
            normalized_center_speed_per_second=float(speed.mean()) if len(speed) else None,
            teacher_minus_Source_observed_IoU=mean([f['teacher_iou']-f['source_iou'] for f in frame_stats]),
            oracle_policy_gain_over_Source=mean([f['oracle_policy_gain'] for f in frame_stats]))
        features.append(scalar)
        candidate_details.append({reward: {k: mean([r[reward][k] for r in ranks]) for k in
            ['pair_accuracy','spearman','top1_IoU_regret','weighted_IoU_gain_over_Source','normalized_entropy','ESS','reward_std']}
            for reward in ['layout3','layout1','geometry']})
        for arm, (root, seal) in predictions.items():
            cost = json.loads((root/seal['files'][query]['receipt_path']).read_text())['cost']
            cost_records[arm].append(cost)
        private_details.append(dict(id=query, parent=truth['parent'], attributes=scalar, values=row['values']))
    parents = [r['parent'] for r in rows]
    pairs = [('Layout3OPD','Layout3Direct'), ('GeometryOPD','GeometryDirect'),
             ('Layout3OPD','GeometryOPD'), ('Layout3OPD','Layout1OPD')]
    paired = {a+'-'+b: {k: cluster_difference(difference(rows,a,b,k),parents) for k in METRICS} for a,b in pairs}
    groups = dict(all=list(range(128)), no_GT_in_eight=[i for i,f in enumerate(features) if f['observations_with_GT']==0],
                  teacher_better=[i for i,f in enumerate(features) if f['teacher_minus_Source_observed_IoU'] is not None and f['teacher_minus_Source_observed_IoU']>0],
                  teacher_not_better=[i for i,f in enumerate(features) if f['teacher_minus_Source_observed_IoU'] is not None and f['teacher_minus_Source_observed_IoU']<=0])
    for kind in sorted({f['query_kind'] for f in features}):
        groups['query_kind_'+kind] = [i for i,f in enumerate(features) if f['query_kind']==kind]
    for name in ['mean_target_area_fraction','event_fraction','normalized_center_speed_per_second']:
        values = [f[name] for f in features if f[name] is not None]
        cut = float(np.median(values)) if values else None
        if cut is not None:
            groups[name+'_below_median'] = [i for i,f in enumerate(features) if f[name] is not None and f[name]<=cut]
            groups[name+'_above_median'] = [i for i,f in enumerate(features) if f[name] is not None and f[name]>cut]
    grouped = {name: dict(queries=len(indices), paired={a+'-'+b: {k: cluster_difference(
        [difference(rows,a,b,k)[i] for i in indices], [parents[i] for i in indices], repeats=2000)
        for k in ['m_vIoU','m_sIoU','unobserved_actual_sIoU']} for a,b in pairs[:2]}) for name,indices in groups.items()}
    optimization = {a: {key: mean([c.get(key) for c in vals]) for key in
        ['adaptation_seconds','actual_updates','mean_state_gradient_norm','nonzero_gradient_steps',
         'pseudo_loss_before','pseudo_loss_after','peak_allocated_bytes']} for a,vals in cost_records.items()}
    ranking = {reward: {k: mean([r[reward][k] for r in candidate_details]) for k in candidate_details[0][reward]}
               for reward in ['layout3','layout1','geometry']}
    triage = needs_research(paired['Layout3OPD-Layout3Direct'], plan['practical_triage_margin_pp'])
    result = dict(status='complete_offline_diagnosis', queries=128, parents=128, paired=paired,
        descriptive_subgroups=grouped, candidate_policy32_ranking=ranking, optimization=optimization,
        further_research=triage, direct_fair_tuning_not_yet_done=True, GT_used_only_after_all_seals=True,
        strata_never_used_as_adaptation_gates=True, subgroup_multiple_testing_no_confirmatory_claim=True,
        reward_bank_only_initial_not_full_trajectory=True, gradient_scale_not_comparable_across_losses=True,
        unresolved=['decoder state-to-observed/unobserved Jacobian and conditioning require a separately frozen bounded probe',
                    'language/identity/occlusion causality cannot be inferred from box annotations alone',
                    'eight-frame target absence in GT is not an occlusion label',
                    'same-budget Direct is not independently fair tuned; no OPD necessity claim',
                    'correlations and oracle policy coverage diagnose hypotheses, not causal proof'])
    write(OUT/'per_query_private.json', private_details)
    write(PUBLIC/'summary.json', result)
    lines = ['# TubeDETR OPD / Direct 完整128封存后诊断','',
        '同奖励比较是主对照；固定预算Direct尚未独立公平调参，128及原目标全量已曝光。',
        '所有差值为OPD减对应Direct，单位pp；置信区间按父视频配对bootstrap，仅开发诊断。','',
        '| 比较 | m_vIoU差值 [95%CI] | dense sIoU差值 | 未观测sIoU差值 |','|---|---:|---:|---:|']
    for name in paired:
        m=paired[name];v=m['m_vIoU'];lines.append(f"| {name} | {v['mean_pp']:.5f} {v['CI95_pp']} | {m['m_sIoU']['mean_pp']:.5f} | {m['unobserved_actual_sIoU']['mean_pp']} |")
    lines += ['', '后续机制研究触发：'+json.dumps(triage,ensure_ascii=False),
        '0.5pp为结果前登记的工程分流阈值，不是等效界限、临床式显著性标准或128统计功效保证。',
        '目标尺寸/运动/事件占比的中位数分组仅描述；无GT观测不等于遮挡。结果不回流当前试验。',
        '教师质量、候选覆盖、排序、初始熵、梯度及局部到未观测传播须结合分析，不能仅凭伪损失下降归因。',
        '后续主线程须先以本证据提出可证伪假设，再检索原论文/官方代码、冻结有限验证协议；禁止只为胜Direct而削弱其对照。','']
    (PUBLIC/'report.md').write_text('\n'.join(lines))
    write(PUBLIC/'final_seal.json',dict(status='complete',queries=128,pilot_final_seal_sha256=sha(PILOT/'final_seal.json'),
        files={str(p.relative_to(ROOT)):sha(p) for p in [PLAN,PUBLIC/'summary.json',PUBLIC/'report.md']},
        publication_pending=True,further_research_required=triage['required']))
    write(OUT/'status.json',dict(status='complete',further_research_required=triage['required'],new_GPU_work_started=False))


def main(wait):
    plan=json.loads(PLAN.read_text())
    if sha(Path(__file__)) != plan['analysis_code_sha256']:
        raise ValueError('Frozen analysis code changed')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if verify_final(PUBLIC/'final_seal.json'):
            return
        while not verify_final(PILOT/'final_seal.json'):
            if (PRIVATE/'failure.json').exists():
                raise RuntimeError('ROI pilot failed; do not diagnose partial arms')
            write(OUT/'status.json',dict(status='waiting_for_complete_ROI_pilot_seal',
                  GT_read=False,new_GPU_work_started=False,as_of=time.time()))
            if not wait:
                return
            time.sleep(30)
        analyze(plan)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');a=parser.parse_args()
    try:
        main(a.wait)
    except Exception as error:
        failure=dict(status='failed',error_type=type(error).__name__,message=str(error),as_of=time.time())
        write(OUT/'failure.json',failure);write(OUT/'status.json',failure);raise
