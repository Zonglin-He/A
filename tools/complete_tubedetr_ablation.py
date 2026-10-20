"""Offline five-row TubeDETR ablations from the existing complete paired runs.

No additional model forward, expert call or adaptation episode is launched.
Each direction must be fully sealed before its independent label-reading score.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import implementation_hash, population_hash, write
from stvg.evaluation_v2 import ARMS

ORDER = ['Source', 'Spatial', 'Temporal', 'WithoutOPD', 'Full']
NAMES = ['Source', '仅空间分支', '仅时间分支', '去掉空间OPD', '完整方法']
MAIN = ['m_vIoU', 'vIoU@0.3', 'vIoU@0.5', 'm_tIoU', 'm_sIoU']
SUPPLEMENT = ['tIoU_ge_0_5', 'tIoU_zero', 'boundary_MAE_seconds', 'observed_actual_sIoU']
COMPONENTS = {'初始化': ('WithoutOPD','Temporal'), '空间OPD': ('Full','WithoutOPD'),
              '时间分支': ('Full','Spatial'), '整个空间分支': ('Full','Temporal'), '整体收益': ('Full','Source')}
DIRECTIONS = [('hc2_to_vid','vidstg',10303,732), ('vid_to_hc2','hc2',1902,227)]
PLAN = ROOT/'results/tubedetr/ablation_plan.json'


def prepare():
    base = json.loads((ROOT/'configs/default.json').read_text())
    adapter = json.loads((ROOT/'configs/tubedetr.json').read_text())
    datasets = {}
    for direction, ds, count, parents in DIRECTIONS:
        path = ROOT/f'data/evaluation/{ds}.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        if len(rows) != count or len({r['id'] for r in rows}) != count or {r['dataset']for r in rows} != {ds}:
            raise ValueError('Complete canonical input population required')
        conf = {**base['datasets'][ds], **adapter['common'], **adapter['datasets'][ds]}
        datasets[direction] = dict(dataset=ds, queries=count, parents=parents,
            population_sha256=population_hash(rows), manifest_sha256=sha(path),
            source_checkpoint=conf['source_checkpoint'], source_checkpoint_sha256=conf['source_checkpoint_sha256'],
            configuration=conf)
    value = dict(status='frozen_offline_ablation_plan', backbone='TubeDETR', seed=41,
        implementation_sha256=implementation_hash(), configuration_sha256=sha(ROOT/'configs/default.json'),
        adapter_configuration_sha256=sha(ROOT/'configs/tubedetr.json'),
        qualification_sha256=sha(ROOT/'results/tubedetr/qualification.json'), directions=datasets,
        arms={k:dict(box_array=ARMS[k][0], interval_array=ARMS[k][1]+'_interval')for k in ORDER},
        main_metric_fields=MAIN, supplementary_metric_fields=SUPPLEMENT,
        additional_diagnostics=['unobserved_actual_sIoU','observed_GT_nodes','unobserved_GT_nodes'],
        metric_count_note='The explicit main table has5 columns and supplement has4; unobserved spatial IoU is retained separately.',
        comparisons={k:list(v)for k,v in COMPONENTS.items()},
        main_estimand='equal queries', supplementary_estimands=['equal parent','parent-cluster bootstrap'],
        same_complete_predictions_as_main_experiment=True, extra_GPU_episodes=0,
        cache_equivalence='Eight positions and initialization depend on native Source and same ReferDINO evidence; spatial OPD does not read Strong, so five rows are exact readouts.',
        labels_read_only_after_complete_prediction_seal=True,
        old_results_modified=False)
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != value:
            raise ValueError('Existing frozen ablation plan changed')
    else:
        write(PLAN, value)
    return value


def score_direction(direction, expected, plan):
    if (implementation_hash() != plan['implementation_sha256']
        or sha(ROOT/'configs/default.json') != plan['configuration_sha256']
        or sha(ROOT/'configs/tubedetr.json') != plan['adapter_configuration_sha256']
        or sha(ROOT/'results/tubedetr/qualification.json') != plan['qualification_sha256']):
        raise ValueError('Code, configuration or qualification changed after the ablation plan was frozen')
    runroot = ROOT/'outputs/tubedetr'/direction
    seal_path = runroot/'sealed.json'
    seal = json.loads(seal_path.read_text())
    run = json.loads((runroot/'run.json').read_text())
    if (seal['queries'] != expected['queries'] or seal['manifest_sha256'] != expected['manifest_sha256']
        or run['population_sha256'] != expected['population_sha256']
        or run['configuration'].get('backbone') != 'TubeDETR'
        or run['configuration']['source_checkpoint_sha256'] != expected['source_checkpoint_sha256']
        or run['configuration'] != expected['configuration']
        or run['implementation_sha256'] != plan['implementation_sha256']
        or run['qualification_sha256'] != plan['qualification_sha256']
        or run['manifest_sha256'] != expected['manifest_sha256']
        or seal['identity'] != run):
        raise ValueError('Ablation cache population, backbone or checkpoint mismatch')
    target = ROOT/'outputs/tubedetr_ablation'/direction
    target.mkdir(parents=True, exist_ok=True)
    receipt_path, metrics_path = target/'receipt.json', target/'metrics_five_rows.json'
    identity = dict(plan_sha256=sha(PLAN), prediction_seal_sha256=sha(seal_path),
                    manifest_sha256=expected['manifest_sha256'])
    if metrics_path.exists() or receipt_path.exists():
        if not metrics_path.exists() or not receipt_path.exists():
            raise ValueError('Incomplete offline-score write; preserve and inspect before resuming')
        receipt = json.loads(receipt_path.read_text())
        if any(receipt.get(k) != v for k,v in identity.items()) or sha(metrics_path) != receipt['metrics_sha256']:
            raise ValueError('Existing ablation receipt mismatch')
    else:
        ds = expected['dataset']
        # This child alone opens labels, after the complete prediction seal.
        subprocess.run([sys.executable,'-B','-m','stvg','score','--aggregation','query',
            '--arms',','.join(ORDER),'--manifest',str(ROOT/f'data/evaluation/{ds}.jsonl'),
            '--labels',str(ROOT/f'data/evaluation/{ds}_labels.jsonl'),
            '--predictions',str(runroot),'--output',str(metrics_path)], cwd=ROOT, check=True)
        write(receipt_path, {**identity,'metrics_sha256':sha(metrics_path),'extra_GPU_episodes':0})
    return json.loads(metrics_path.read_text()), dict(identity, metrics_sha256=sha(metrics_path))


def publish_local(completed):
    summaries, file_seals, text = {}, {}, ['# TubeDETR 两方向五行组件消融','',
        '完整主实验封存预测的离线读出；query平均。所有IoU与比例显示为百分数，边界误差单位为秒。','']
    for direction, ds, _, _ in DIRECTIONS:
        result, evidence = completed[direction]
        group = result['datasets'][ds]
        changes = {name:{field:(group['query_means'][a][field]-group['query_means'][b][field])
                      if group['query_means'][a][field] is not None and group['query_means'][b][field] is not None else None
                      for field in MAIN+SUPPLEMENT}
                   for name,(a,b) in COMPONENTS.items()}
        summaries[direction] = dict(queries=group['queries'],parents=group['parents'],
            query_means=group['query_means'], query_denominators=group['query_denominators'],
            equal_parent_means=group['equal_parent_means'], paired=group['paired'], bootstrap=group['bootstrap'],
            component_differences=changes, evidence=evidence)
        public = json.loads(json.dumps(result))
        for g in public['datasets'].values():
            for cluster in g['clusters']:
                cluster['parent'] = hashlib.sha256(str(cluster['parent']).encode()).hexdigest()
        for row in public['rows']:
            row['parent'] = hashlib.sha256(str(row['parent']).encode()).hexdigest()
        dest = ROOT/f'results/tubedetr/ablation/{direction}.json'
        write(dest, public); file_seals[str(dest.relative_to(ROOT))] = sha(dest)
        text += ['## '+direction,'',f"完整query：{group['queries']}；父来源：{group['parents']}。",'']
        for fields, title in [(MAIN,'主消融表'), (SUPPLEMENT,'补充分析表')]:
            text += ['### '+title,'','| 设置 | '+' | '.join(fields)+' |', '|---|'+'---:|'*len(fields)]
            for arm,name in zip(ORDER,NAMES):
                values = group['query_means'][arm]
                cells = ['—' if values[f] is None else f'{values[f]*(1 if f=="boundary_MAE_seconds" else 100):.4f}'for f in fields]
                text.append('| '+name+' | '+' | '.join(cells)+' |')
            text.append('')
        text += ['未观测实际节点sIoU、观测/未观测GT节点分母、父来源平均和聚类统计保存在对应JSON中。','']
    write(ROOT/'results/tubedetr/ablation_summary.json',dict(status='complete_sealed_offline_ablation',directions=summaries,
        backbone='TubeDETR',rows_per_direction=5,extra_GPU_episodes=0))
    (ROOT/'results/tubedetr/ablation_report.md').write_text('\n'.join(text)+'\n')
    for rel in ['results/tubedetr/ablation_summary.json','results/tubedetr/ablation_report.md','results/tubedetr/ablation_plan.json']:
        file_seals[rel] = sha(ROOT/rel)
    write(ROOT/'results/tubedetr/ablation_seal.json',dict(status='complete',files=file_seals,
        prediction_seals={d:e['prediction_seal_sha256']for d,(_,e)in completed.items()},publication_pending=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--wait',action='store_true')
    args = parser.parse_args()
    plan = prepare()
    if args.prepare_only:
        print('Frozen TubeDETR five-row offline ablation plan; no labels or GPU used',flush=True)
        return
    lock_path = ROOT/'runtime/tubedetr_ablation.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        completed = {}
        status = ROOT/'outputs/tubedetr_ablation/status.json'
        while len(completed) != 2:
            for direction, expected in plan['directions'].items():
                if direction not in completed and (ROOT/'outputs/tubedetr'/direction/'sealed.json').exists():
                    completed[direction] = score_direction(direction,expected,plan)
            waiting = [d for d in plan['directions']if d not in completed]
            write(status,dict(status='waiting_for_complete_prediction_seals'if waiting else'complete',
                scored_directions=list(completed),waiting_directions=waiting,extra_GPU_episodes=0,rows_per_direction=5))
            if not waiting:
                break
            if (ROOT/'outputs/tubedetr/MAIN_QUEUE_FAILURE.json').exists():
                raise RuntimeError('Main queue failed before required full seals; never score partial populations')
            if not args.wait:
                return
            time.sleep(30)
        publish_local(completed)
        for action in ['check','snapshot','check']:
            subprocess.run([sys.executable,'-B','scripts/research_archive.py',action],cwd=ROOT,check=True)
        print('COMPLETE TubeDETR five-row ablations for both full populations',flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'outputs/tubedetr_ablation/failure.json',dict(status='failed',error_type=type(error).__name__,
            message=str(error),extra_GPU_episodes=0))
        path = ROOT/'outputs/tubedetr_ablation/status.json'
        status = json.loads(path.read_text()) if path.exists() else {}
        write(path, {**status, 'status':'failed', 'error_type':type(error).__name__,
                     'message':str(error), 'extra_GPU_episodes':0})
        raise
