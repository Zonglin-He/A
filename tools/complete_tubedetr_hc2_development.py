"""Wait for both original directions/ablations, then finite HC128 tuning/full."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import tubedetr_hc2_development as dev
from stvg.runtime import sha
from stvg.tube_runner import digest,write,save_prediction


def require_original_completion():
    paths=[ROOT/'results/tubedetr/final_seal.json',ROOT/'results/tubedetr/ablation_seal.json',
           ROOT/'results/tubedetr_vid_development/final_seal.json']
    if not all(p.exists()for p in paths):return False
    for path in paths:
        seal=json.loads(path.read_text())
        if seal['status']!='complete':return False
        for name,value in seal['files'].items():
            if sha(ROOT/name)!=value:raise ValueError('Original final artifact differs from its seal')
    registry=json.loads((ROOT/'runtime/tubedetr_active_jobs.json').read_text())
    for name in ['finite_queue','ablation_offline','vid_development_controller','vid_development_gpu']:
        job=registry['jobs'].get(name)
        if job is None:continue
        try:
            argv=(Path('/proc')/str(job['pid'])/'cmdline').read_bytes().split(b'\0')
            if [a.decode() for a in argv if a]==shlex.split(job['command']):return False
        except FileNotFoundError:pass
    return True


def bind_seeded_view(manifest,output,params,plan,source=None,scope='dev128_actual_adaptation',
                     identity_manifest=None,seal_when_complete=True):
    records=[json.loads(s)for s in Path(manifest).read_text().splitlines()]
    oldseal=json.loads((dev.OLD/'sealed.json').read_text())
    ident=dev.identity(identity_manifest or manifest,params,scope,plan)
    for row in records:
        if dev.saved(output,row,ident):continue
        if source is None:
            value,r=dev.from_original(row,oldseal)
        else:
            path=source/'predictions'/f"{row['id']}.npz"
            r=json.loads(path.with_suffix('.json').read_text())
            if sha(path)!=r['prediction_sha256']:raise ValueError('Selected candidate cache changed')
            with np.load(path,allow_pickle=False)as z:value={k:z[k].copy()for k in z.files}
        value.update(cost={**r['cost'],'exact_prediction_cache_reuse':True,
                           'not_fresh_end_to_end_timing':True},
                     checkpoint=r['checkpoint'],initialization=r['initialization'])
        save_prediction(output/'predictions'/f"{row['id']}.npz",value,
            dict(identity_sha256=digest(ident),request_sha256=digest(row)))
    if len(records)==128 and seal_when_complete:dev.seal(output,records,ident)


def score(manifest,labels,out,arms):
    path=out/'metrics.json';receipt=out/'score_receipt.json'
    ident=dict(seal_sha256=sha(out/'sealed.json'),manifest_sha256=sha(manifest),
               labels_sha256=sha(labels),arms=arms)
    if path.exists() or receipt.exists():
        if not path.exists() or not receipt.exists():raise ValueError('Incomplete candidate score write')
        old=json.loads(receipt.read_text())
        if any(old.get(k)!=v for k,v in ident.items()) or old['metrics_sha256']!=sha(path):
            raise ValueError('Cached development score identity changed')
    else:
        env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
        subprocess.run([sys.executable,'-B','-m','stvg','score','--aggregation','query','--arms',','.join(arms),
            '--manifest',str(manifest),'--labels',str(labels),'--predictions',str(out),'--output',str(path)],
            cwd=ROOT,check=True,env=env,stdout=subprocess.DEVNULL)
        write(receipt,{**ident,'metrics_sha256':sha(path)})
    return json.loads(path.read_text())


def launch_profiles(params,manifest,plan,stage,output_base=None,scope='dev128_actual_adaptation'):
    base=output_base or dev.OUT/'candidates'
    pending=[p for p in params if not (base/dev.pid(p)/'sealed.json').exists()]
    if not pending:return
    request=dev.OUT/'worker_requests'/f'{stage}.json'
    write(request,dict(manifest=str(manifest),profiles=pending,output_base=str(base),scope=scope,
                       receipt=str(dev.OUT/'worker_receipts'/f'{stage}.json')))
    log=ROOT/'logs'/f'tubedetr_hc2_development_{stage}.log'
    command=[sys.executable,'-B','-u',str(ROOT/'tools/tubedetr_hc2_development.py'),'--worker-request',str(request)]
    write(dev.OUT/'status.json',dict(status=stage,new_GPU_work_started=True,as_of=dev.now()))
    with log.open('a')as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        registry=ROOT/'runtime/tubedetr_active_jobs.json';value=json.loads(registry.read_text())
        value['jobs']['hc2_development_gpu']=dict(pid=process.pid,command=shlex.join(command),log=str(log.relative_to(ROOT)))
        write(registry,value)
        if process.wait():raise RuntimeError('Development worker failed; inspect '+str(log))


def main(wait):
    plan=dev.prepare();dev.OUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'runtime/tubedetr_hc2_development.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    completed=dev.PUBLIC/'final_seal.json'
    if completed.exists():
        receipt=json.loads(completed.read_text())
        if receipt['status']!='complete' or any(sha(ROOT/p)!=h for p,h in receipt['files'].items()):
            raise ValueError('Completed development seal changed')
        print('Already complete; immutable development results reused',flush=True);return
    while not require_original_completion():
        for path in [ROOT/'outputs/tubedetr/MAIN_QUEUE_FAILURE.json',ROOT/'outputs/tubedetr_ablation/failure.json',
                     ROOT/'outputs/tubedetr_vid_development/failure.json']:
            if path.exists():raise RuntimeError('Upstream study failed; HC tuning waits for original study and selected Vid full terminal seals')
        write(dev.OUT/'status.json',dict(status='waiting_for_Vid_selected_full_and_original_terminal_seals',
            tuning_candidates_evaluated=0,new_GPU_work_started=False,as_of=dev.now()))
        if not wait:return
        time.sleep(30)
    dev.guard_plan(plan)
    manifest=ROOT/'data/evaluation/tubedetr_hc2_dev128.jsonl'
    labels=ROOT/'data/evaluation/tubedetr_hc2_dev128_labels.jsonl'
    decisions_path=dev.PUBLIC/'decisions.json';decisions=[]
    def note(stage,reason,data):
        decisions.append(dict(stage=stage,reason=reason,outcome=copy.deepcopy(data),as_of=dev.now()))
        write(decisions_path,dict(status='development_in_progress',decisions=decisions,
            not_independent_confirmation=True,full_results_not_used_to_rank_profiles=True))
        write(dev.OUT/'status.json',dict(status=stage,as_of=dev.now(),new_GPU_work_started=stage!='initialization'))
    dev.initialization_candidates(manifest,plan)
    initial={}
    for gamma in dev.GAMMAS:
        params=dev.parameters(gamma=gamma)
        result=score(manifest,labels,dev.OUT/'initialization'/dev.pid(params),['Source','Temporal','WithoutOPD'])
        initial[str(gamma)]=result['datasets']['hc2']['query_means']['WithoutOPD']['m_vIoU']
    gamma=max([g for g in dev.GAMMAS if g>0],key=lambda g:(initial[str(g)],g==dev.BASE['initialization_gamma'],g))
    note('initialization','Test weaker global correction first; gamma0 is a temporal-only control, not eligible Full.',
         dict(scores=initial,selected_nonzero_gamma=gamma,zero_control_better=initial['0.0']>initial[str(gamma)]))
    baseline=dev.OUT/'candidates'/dev.pid(dev.BASE)
    bind_seeded_view(manifest,baseline,dev.BASE,plan)
    values={}
    def collect(params):
        for p in params:
            root=dev.OUT/'candidates'/dev.pid(p)
            result=score(manifest,labels,root,['Source','Spatial','Temporal','WithoutOPD','Full'])
            costs=[json.loads(p.read_text())['cost']for p in (root/'predictions').glob('*.json')]
            fields=['mean_weight_entropy','mean_ESS','mean_state_gradient_norm','pseudo_loss_before','pseudo_loss_after']
            diagnostics={field:float(np.mean([c[field]for c in costs if field in c]))
                         if any(field in c for c in costs) else None for field in fields}
            values[dev.pid(p)]=dict(params=p,score=result['datasets']['hc2']['query_means']['Full']['m_vIoU'],
                metric_means=result['datasets']['hc2']['query_means'],diagnostics=diagnostics,
                metrics_sha256=sha(root/'metrics.json'))
        subprocess.run([sys.executable,'-B','tools/analyze_tubedetr_spatial.py','--dataset','hc2'],
                       cwd=ROOT,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=subprocess.DEVNULL)
        return values[dev.pick(values)]['params']
    collect([dev.BASE])
    rates=[dev.parameters(gamma=gamma,lr=lr)for lr in [.0035,.007,.014,.028]]
    launch_profiles(rates,manifest,plan,'learning_rate');best=collect(rates)
    note('learning_rate','Small original OPD effect does not tell the direction of LR change; compare both directions.',
         dict(tested=rates,selected=best,all_outcomes=values,
              next_reason='LR winner is global128 m_vIoU maximum; next distinguish soft candidate weights from insufficient update size using tau and measured entropy/ESS.'))
    entropy=values[dev.pid(best)]['diagnostics']['mean_weight_entropy']
    order=[.07,.14,.28] if entropy is None or entropy>.9 else [.28,.14,.07]
    temperatures=[{**best,'tau':tau}for tau in order]
    launch_profiles(temperatures,manifest,plan,'temperature');best=collect(temperatures)
    note('temperature','Check sharpening/flattening and recorded entropy/ESS and pseudo-loss; choose by128 localization.',
         dict(tested=temperatures,selected=best,all_outcomes=values,
              measured_prior_entropy=entropy,
              next_reason='High normalized entropy motivates testing sharper feedback first; both directions are retained. Next compare saved steps to distinguish under-updating from pseudo-label overfitting.'))
    steps=[{**best,'steps':n}for n in [20,40,80]]
    launch_profiles(steps,manifest,plan,'steps');best=collect(steps)
    note('steps','Check whether additional updates help or damage;80 prefixes must reproduce any saved40 result.',
         dict(tested=steps,selected=best,all_outcomes=values,
              next_reason='Compare localization and pseudo-loss at20/40/80; a lower pseudo-loss with lower localization is a development sign of misaligned teacher fitting, not a reason to increase the budget. Recheck nearby gamma at the best optimizer.'))
    allowed=[g for g in dev.GAMMAS if g>0];i=allowed.index(best['initialization_gamma'])
    neighbours=[{**best,'initialization_gamma':allowed[j]}for j in [i-1,i+1]if 0<=j<len(allowed)]
    launch_profiles(neighbours,manifest,plan,'gamma_neighbours');best=collect(neighbours)
    note('gamma_neighbours','Recheck initialization/optimizer coupling with at most two adjacent gamma values.',
         dict(tested=neighbours,selected=best,all_outcomes=values))
    selected=dict(status='locked_after_dev128_before_full_evaluation',parameters=best,
        profile=dev.pid(best),selection_objective=plan['selection_objective'],all_dev_outcomes=values,
        plan_sha256=sha(dev.PLAN),development_manifest_sha256=sha(manifest),
        full_target_previously_seen=True,original_results_preserved=True,
        no_retuning_after_new_full_results=True,tuning_diagnostics_include=['pseudo_loss','entropy','ESS','gradient_norm'])
    chosen=values[dev.pid(best)]['metric_means']
    selected['development_component_differences_pp']=dict(
        initialization=100*(chosen['WithoutOPD']['m_vIoU']-chosen['Temporal']['m_vIoU']),
        OPD=100*(chosen['Full']['m_vIoU']-chosen['WithoutOPD']['m_vIoU']),
        complete_spatial=100*(chosen['Full']['m_vIoU']-chosen['Temporal']['m_vIoU']))
    selected_path=dev.PUBLIC/'selected_profile.json'
    if selected_path.exists() and json.loads(selected_path.read_text())!=selected:
        raise ValueError('Selected profile already frozen; do not change it after full evaluation')
    write(selected_path,selected) if not selected_path.exists() else None
    write(decisions_path,dict(status='selection_locked',decisions=decisions,not_independent_confirmation=True))
    if best==dev.BASE:
        full_root=dev.OLD;reused=True
        full=json.loads((ROOT/'outputs/tubedetr_ablation/vid_to_hc2/metrics_five_rows.json').read_text())
    else:
        full_manifest=ROOT/'data/evaluation/hc2.jsonl'
        full_base=dev.OUT/'full_selected';full_root=full_base/dev.pid(best)
        # Exact same128 requests/seed/checkpoint/profile can be reused in full.
        bind_seeded_view(manifest,full_root,best,plan,source=dev.OUT/'candidates'/dev.pid(best),
                         scope='full_selected_profile',identity_manifest=full_manifest,seal_when_complete=False)
        note('full_selected','One selected global profile on all1902; exact128 caches reused, Strong time outputs reused.',best)
        launch_profiles([best],full_manifest,plan,'full_selected',output_base=full_base,scope='full_selected_profile')
        full=score(full_manifest,ROOT/'data/evaluation/hc2_labels.jsonl',full_root,
                   ['Source','Spatial','Temporal','WithoutOPD','Full']);reused=False
    public=copy.deepcopy(full)
    for group in public['datasets'].values():
        for c in group['clusters']:c['parent']=digest(c['parent'])
    for row in public['rows']:row['parent']=digest(row['parent'])
    write(dev.PUBLIC/'full_metrics.json',public)
    group=full['datasets']['hc2'];report=['# TubeDETR HC128开发后全量结果','',
        '原冻结两方向结果保留。本轮仅HC2目标域数值开发；全量包含128开发样本，且原全量结果已用于假设诊断，不宣称未曝光确认。','',
        '选中参数：'+json.dumps(best,ensure_ascii=False),'',
        '| 设置 | m_vIoU | vIoU@0.3 | vIoU@0.5 | m_tIoU | m_sIoU |','|---|---:|---:|---:|---:|---:|']
    for arm in ['Source','Spatial','Temporal','WithoutOPD','Full']:
        report.append('| '+arm+' | '+' | '.join(f"{100*group['query_means'][arm][k]:.4f}"for k in
            ['m_vIoU','vIoU@0.3','vIoU@0.5','m_tIoU','m_sIoU'])+' |')
    report += ['', '所有候选、负结果、推理检查和每阶段选择理由见plan及decisions。',
        '若原配置获胜，完整旧预测精确复用；否则新全量保留Strong缓存并重新执行所选空间适应，缓存执行时间不能当作全新端到端部署时间。','']
    (dev.PUBLIC/'report.md').write_text('\n'.join(report))
    files=[dev.PLAN,decisions_path,selected_path,dev.PUBLIC/'full_metrics.json',dev.PUBLIC/'report.md']
    write(dev.PUBLIC/'final_seal.json',dict(status='complete',queries=1902,development_queries=128,
        selected_profile=dev.pid(best),original_full_reused=reused,
        prediction_seal_sha256=sha(full_root/'sealed.json'),
        files={str(p.relative_to(ROOT)):sha(p)for p in files},publication_pending=True))
    write(dev.OUT/'status.json',dict(status='complete',queries=1902,selected=best,publication_pending=True))
    print('COMPLETE original study followed by bounded HC128 development and full selected evaluation',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    try:main(args.wait)
    except BaseException as error:
        value=dict(status='failed',error_type=type(error).__name__,message=str(error),as_of=dev.now())
        write(dev.OUT/'failure.json',value);write(dev.OUT/'status.json',value);raise
