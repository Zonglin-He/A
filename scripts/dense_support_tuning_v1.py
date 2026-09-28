"""F40 bounded aggressive coordinate HPO; immutable F39/A4, no schedules.

prepare -> capture-dev -> search -> capture-eval -> evaluate -> summarize.
Only search/summarize read labels. All fitting calls take frozen label-free
head inputs/teacher, with F39's loss-only best-state rule. Evaluation parameters
are committed before the evaluation stage; no evaluation-driven second search.
"""
import argparse
import copy
import fcntl
import gc
import hashlib
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, load, save, sha

OUT = ROOT/'artifacts/decota_dense_support_tuning_v1'
PARENT = ROOT/'artifacts/decota_dense_support_v1'
OWN = ['scripts/dense_support_tuning_v1.py', 'vg_tta/dense_support_tuning_v1.py']
METRICS = ['vIoU_corrected', 'sIoU', 'tIoU', 'temporal_recall', 'temporal_precision', 'span']
STEPS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
LIB = ROOT/'external/runtime_libs/nvidia_580_159_03/extracted/usr/lib/x86_64-linux-gnu'


def prepare():
    parent = read(PARENT/'LOCK.json')
    completion = read(PARENT/'COMPLETION.json')
    protected = {**parent['protected_pins'], **completion['files'], **completion['code_pins']}
    protected[str((PARENT/'COMPLETION.json').relative_to(ROOT))] = sha(PARENT/'COMPLETION.json')
    for f, h in protected.items():
        assert sha(ROOT/f) == h, f
    for cohort, rr in parent['rows'].items():
        assert len(rr) == 24
        assert len([r for r in rr if r['f37_role'] == 'development']) == 8
        assert len({r['group'] for r in rr}) == 24
    plan = dict(name='F40_A4_frozen_dense_time_aggressive_HPO', created=time.time(),
        authority='User asks broad aggressive tuning on these datasets to try to outperform native time.',
        rows=parent['rows'], protected_pins=protected, code_pins={f:sha(ROOT/f) for f in OWN},
        parent_lock=sha(PARENT/'LOCK.json'), spatial=parent['spatial'],
        fixed=dict(teacher=parent['temporal']['teacher'], loss=parent['temporal']['loss'],
            implementation='vg_tta.dense_support_temporal_v1.fit (unchanged)',
            trainable='actual temp_embed, 66306 network parameters',
            optimizer='AdamW betas(.9,.999), eps1e-4, weight_decay0',
            backtracking=[1., .5, .25, .125], output='native legal per-offset argmax then envelope',
            initial='source head, no warm start; exact A4 frozen student boxes',
            new_spatial_fits=0, new_DINO=0, GT_online=False),
        search=dict(backend='Optuna 4.5.0 TPESampler seed20260914/20260915',
            separate_per_direction=True, default=dict(lr=.01,beta=1.,steps=5),
            lr_range=[1e-6,1e2], beta_range=[1e-6,1e4], include_beta_zero=True, steps=STEPS,
            stages=[dict(name='lr_wide',coordinate='lr',trials=36),
                    dict(name='beta_wide',coordinate='beta',trials=28),
                    dict(name='steps',coordinate='steps',trials=12),
                    dict(name='lr_refine',coordinate='lr',trials=20),
                    dict(name='beta_refine',coordinate='beta',trials=20),
                    dict(name='steps_recheck',coordinate='steps',trials=12)],
            refine='positive coordinate within one log10 decade of incumbent, clipped to wide range; beta0 uses wide beta range',
            criterion='maximum source-macro corrected vIoU delta over same A4+native time on 8 development sources only',
            tie='within1e-12 raw units, fewer steps; then closer log10(lr/.01); then smaller beta',
            pruning=False, exclude_bad_cases=False, cache_identical_config=True,
            evaluation='once after both choices committed; T0, selected T1, same-config T2/T3/T4, original F39 T1 retained',
            no_eval_retuning=True, no_per_sample_hyperparameters=True),
        caps=dict(network_or_output_fits=2400, backward=240000, max_wall_seconds=7200, new_DINO=0),
        historical_exposure=True, untouched=False,
        provenance_note='Prior evaluation results already seen. This is locked-config re-evaluation, not fresh confirmation.',
        environment=dict(system_kernel_driver='580.159.03', system_userspace='580.173.02',
            initial_cuda_error='804 forward compatibility; NVML driver/library mismatch',
            process_only_library=str(LIB), system_driver_changed=False, reboot=False,
            official_package_url='https://archive.ubuntu.com/ubuntu/pool/restricted/n/nvidia-graphics-drivers-580-server/libnvidia-compute-580-server_580.159.03-0ubuntu0.25.10.1_amd64.deb',
            package_sha256='12fc2c18183751c9c841d77bf4b231f6669a20f39c237296317da72881eff481',
            libcuda_sha256=sha(LIB/'libcuda.so.580.159.03')),
        registry_change=False, automation=False, new_backbone=False, corruption=False)
    write(OUT/'LOCK.json', plan)
    write(OUT/'STATUS.json',dict(stage='prepared',finished=False,new_fits=0))
    print('F40 locked: 16 development / 32 evaluation, coordinate trials <=256, no A4 changes',flush=True)


def plan(check_parents=False):
    p = read(OUT/'LOCK.json')
    for f, h in p['code_pins'].items():
        assert sha(ROOT/f) == h, f
    if check_parents:
        for f, h in p['protected_pins'].items():
            assert sha(ROOT/f) == h, f
    return p


def keyfile(key):
    return key.replace(':','_')+'.pt'


def receipt(path):
    return path.with_suffix('.json')


def recorded(path):
    if not path.exists():
        return False
    rr = read(receipt(path))
    assert sha(path) == rr['sha256'], path
    return True


def record(path, value):
    save(path, value)
    write(receipt(path),dict(sha256=sha(path),key=value['key'],completed=time.time(),
        code_pins={f:sha(ROOT/f) for f in OWN}))


def capture(role):
    import torch
    from scripts.run_closure_v1 import model_for
    from scripts.run_spatial10_components_v1 import capture_timed
    from vg_tta.parametric_observation_v1 import ObservationReplay
    from vg_tta.simplification_partial_v1 import FullInputHeadReplay
    from vg_tta.dense_support_temporal_v1 import teacher_from_logits, fit
    from vg_tta.dense_support_tuning_v1 import HeadReplay, move, compact_fit
    from vg_tta.foreground_runtime import state_digest
    p = plan(True)
    if role == 'evaluation':
        assert (OUT/'SELECTION.json').is_file(), 'evaluation capture before committed selection'
    for cohort, rr in p['rows'].items():
        rows = [r for r in rr if r['f37_role'] == role]
        pending = [r for r in rows if not recorded(OUT/'cache'/keyfile(r['key']))]
        if not pending:
            continue
        model = model_for(cohort)
        digest = state_digest(model)
        for r in pending:
            frames, _, records, views, qa, cost = capture_timed(model,r)
            it = FullInputHeadReplay(ObservationReplay(model,views,len(r['input']['frame_ids']),'head'))
            z = it.zero
            prior = load(PARENT/('dev' if role=='development' else 'eval')/keyfile(r['key']))
            assert qa == prior['qa']
            assert all(torch.equal(a.cpu(), b) for a,b in zip(z['logits'],prior['T0']['logits']))
            teacher = teacher_from_logits(z['actions'],records)
            assert all(torch.equal(a['a'].cpu(),b['a']) for a,b in zip(teacher,prior['teacher']))
            cache = dict(key=r['key'],cohort=cohort,group=r['group'],source=r['source'],role=role,
                frame_ids=r['input']['frame_ids'],records=records,inputs=move(it.inputs,'cpu'),
                head_state=move(it.initial,'cpu'),zero=dict(boxes=prior['T0']['boxes'],logits=prior['T0']['logits']),
                teacher=move(teacher,'cpu'),wrong_teacher=prior['wrong_teacher'],rolled_teacher=prior['rolled_teacher'],
                T0=prior['T0'],qa=qa,capture_seconds=cost,GT_online=False,
                A4_reference=prior['A4_reference'],parent_sha256=sha(PARENT/('dev' if role=='development' else 'eval')/keyfile(r['key'])),
                exact_f39_native_logits=True,exact_f39_teacher=True)
            portable=HeadReplay(cache)
            assert all(torch.equal(v.cpu(),cache['head_state'][n]) for n,v in portable.named)
            if role=='development':
                # Actual trajectory equivalence at the old default, not only zero outputs.
                check=fit(portable,records,cache['frame_ids'],portable.teacher,lr=.01,beta=1.,steps=5)
                old=prior['fits']['T1_lr0.01_b1']
                assert check['best_step']==old['best_step']
                assert all(torch.equal(v,old['state'][n]) for n,v in check['state'].items())
                assert all(a['indices']==b['indices'] and a['loss']==b['loss'] for a,b in zip(check['path'],old['path']))
                record(OUT/'equivalence'/keyfile(r['key']),dict(key=r['key'],fit=compact_fit(check),
                    exact_f39_all_steps=True,zero_and_updated_head_replay=True))
            record(OUT/'cache'/keyfile(r['key']),cache)
            assert state_digest(model)==digest
            del portable,it,views,frames,z
            print('F40 capture',role,r['key'],'exact',flush=True)
            status(OUT/'STATUS.json',dict(stage='capture_'+role,last=r['key'],finished=False))
        del model
        gc.collect();torch.cuda.empty_cache()


def config_id(config):
    import json
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:18]


def run_fit(cache, config, arm='T1'):
    from vg_tta.dense_support_tuning_v1 import HeadReplay, move, compact_fit
    from vg_tta.dense_support_temporal_v1 import fit, OutputReplay
    it=HeadReplay(cache)
    teacher=it.teacher
    if arm=='T2':it=OutputReplay(it.zero)
    if arm=='T3':teacher=move(cache['wrong_teacher'],'cuda')
    if arm=='T4':teacher=move(cache['rolled_teacher'],'cuda')
    result=fit(it,cache['records'],cache['frame_ids'],teacher,**config,output_control=arm=='T2')
    for row in result['path']:row['boxes']=cache['T0']['boxes']
    result['final']=result['path'][result['best_step']]
    return compact_fit(result)


def score_fit(cache, result, gt):
    from scripts.analyze_spatial10_components_v1 import checked_score
    final=result['final']
    metric,_=checked_score(final['boxes'],gt,cache['frame_ids'],final['indices'])
    native,_=checked_score(cache['T0']['boxes'],gt,cache['frame_ids'],cache['T0']['indices'])
    assert metric['sIoU']==native['sIoU']
    return dict(metrics=metric,T0=native,delta_v=metric['vIoU_corrected']-native['vIoU_corrected'],
        parameter_changed=result['state_delta']>0,interval_changed=final['indices']!=cache['T0']['indices'],
        state_delta=result['state_delta'],best_step=result['best_step'],backward=result['backwards'],
        seconds=result['seconds'],final_loss=final['loss'],initial_loss=result['path'][0]['loss'],
        accepted=sum(s.get('accepted',False) for s in result['path']),failure=result['failure'])


def choose(records):
    best=max(r['delta_v'] for r in records)
    tied=[r for r in records if best-r['delta_v']<=1e-12]
    return min(tied,key=lambda r:(r['config']['steps'],abs(math.log10(r['config']['lr']/.01)),r['config']['beta']))


def search():
    import optuna
    import numpy as np
    from scripts.analyze_spatial10_components_v1 import labels_for
    p=plan(True)
    assert not (OUT/'SELECTION.json').exists(), 'selection is immutable; do not retune from evaluation'
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    selections={}; tick=time.time()
    for ci,(cohort,rr) in enumerate(p['rows'].items()):
        rows=[r for r in rr if r['f37_role']=='development']
        assert len(rows)==8
        gt=labels_for(rows)
        caches={r['key']:load(OUT/'cache'/keyfile(r['key'])) for r in rows}
        all_results={}; stages=[]; newfits=0; backwards=0
        def evaluate_config(config):
            nonlocal newfits,backwards
            config=dict(lr=float(config['lr']),beta=float(config['beta']),steps=int(config['steps']))
            cid=config_id(config)
            if cid in all_results:return all_results[cid]
            path=OUT/'search'/cohort/(cid+'.json')
            if path.exists():
                result=read(path);assert result['config']==config
                for r in result['rows']:
                    assert recorded(Path(r['path']))
                all_results[cid]=result
                return result
            scored=[]
            for r in rows:
                dest=OUT/'search_fits'/cohort/cid/keyfile(r['key'])
                if recorded(dest):zz=load(dest)['fit']
                else:
                    assert newfits<1200 and backwards+config['steps']<=120000
                    assert time.time()-tick<p['caps']['max_wall_seconds']
                    zz=run_fit(caches[r['key']],config)
                    record(dest,dict(key=r['key'],config=config,fit=zz,GT_online=False))
                    newfits+=1;backwards+=zz['backwards']
                s=score_fit(caches[r['key']],zz,gt[r['key']])
                scored.append(dict(key=r['key'],group=r['group'],source=r['source'],path=str(dest),**s))
            result=dict(config=config,id=cid,cohort=cohort,rows=scored,
                delta_v=float(np.mean([r['delta_v'] for r in scored])),
                mean_v=float(np.mean([r['metrics']['vIoU_corrected'] for r in scored])),
                native_v=float(np.mean([r['T0']['vIoU_corrected'] for r in scored])),
                GT_use='offline development config selection only',evaluation_read=False)
            write(path,result);all_results[cid]=result
            print('F40 dev',cohort,config,'delta_v_pp',round(100*result['delta_v'],5),
                'updated',sum(r['parameter_changed'] for r in scored),'changed',sum(r['interval_changed'] for r in scored),flush=True)
            status(OUT/'STATUS.json',dict(stage='search',cohort=cohort,last_config=config,
                completed_unique_configs=len(all_results),best_delta_v_pp=100*choose(list(all_results.values()))['delta_v'],
                new_fits_this_cohort=newfits,backwards_this_cohort=backwards,finished=False))
            return result
        incumbent=evaluate_config(p['search']['default'])
        for si, stage in enumerate(p['search']['stages']):
            base=dict(incumbent['config']);coord=stage['coordinate'];refine='refine' in stage['name']
            sampler=optuna.samplers.TPESampler(seed=20260914+ci*100+si,n_startup_trials=8)
            study=optuna.create_study(direction='maximize',sampler=sampler,
                storage='sqlite:///'+str(OUT/'optuna.sqlite3'),study_name=cohort+'__'+stage['name'],load_if_exists=True)
            if coord=='lr':
                lo,hi=-6.,2.
                if refine:lo=max(lo,math.log10(base['lr'])-1);hi=min(hi,math.log10(base['lr'])+1)
                anchors=[base['lr']]+([10.**k*x for k in range(-6,3) for x in (1.,3.) if 1e-6<=10.**k*x<=100] if not refine else [base['lr']*x for x in (.1,.25,.5,.75,1,1.25,1.5,2,4,10)])
                for v in anchors:
                    if lo<=math.log10(v)<=hi:study.enqueue_trial({'log10_lr':math.log10(v)},skip_if_exists=True)
                def objective(trial):
                    conf={**base,'lr':10**trial.suggest_float('log10_lr',lo,hi)}
                    result=evaluate_config(conf);trial.set_user_attr('config_id',result['id']);return result['delta_v']
            elif coord=='beta':
                lo,hi=-6.,4.
                if refine and base['beta']>0:lo=max(lo,math.log10(base['beta'])-1);hi=min(hi,math.log10(base['beta'])+1)
                study.enqueue_trial({'zero_keep':True},skip_if_exists=True)
                anchors=[base['beta']]+([1e-6,1e-4,.001,.01,.03,.1,.3,1.,3.,10.,30.,100.,1000.,1e4] if not refine else [base['beta']*x for x in (.1,.25,.5,.75,1,1.25,1.5,2,4,10)])
                for v in anchors:
                    if v>0 and lo<=math.log10(v)<=hi:study.enqueue_trial({'zero_keep':False,'log10_beta':math.log10(v)},skip_if_exists=True)
                def objective(trial):
                    beta=0. if trial.suggest_categorical('zero_keep',[False,True]) else 10**trial.suggest_float('log10_beta',lo,hi)
                    result=evaluate_config({**base,'beta':beta});trial.set_user_attr('config_id',result['id']);return result['delta_v']
            else:
                for n in STEPS:study.enqueue_trial({'steps':n},skip_if_exists=True)
                def objective(trial):
                    result=evaluate_config({**base,'steps':trial.suggest_categorical('steps',STEPS)})
                    trial.set_user_attr('config_id',result['id']);return result['delta_v']
            completed=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
            study.optimize(objective,n_trials=max(0,stage['trials']-completed),catch=())
            candidates=[all_results[t.user_attrs['config_id']] if t.user_attrs['config_id'] in all_results
                else evaluate_config(read(OUT/'search'/cohort/(t.user_attrs['config_id']+'.json'))['config'])
                for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
            incumbent=choose([incumbent]+candidates)
            stages.append(dict(stage=stage,before=base,chosen=incumbent['config'],delta_v=incumbent['delta_v'],
                complete_trials=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)))
            print('F40 coordinate locked',cohort,stage['name'],incumbent['config'],100*incumbent['delta_v'],flush=True)
        best=choose(list(all_results.values()))
        selection=dict(cohort=cohort,chosen=best['config'],development_delta_v=best['delta_v'],
            chosen_id=best['id'],stages=stages,unique_configs=len(all_results),development_sources=8,
            development_keys=[r['key'] for r in rows],evaluation_used=False,
            all_results={k:str(OUT/'search'/cohort/(k+'.json')) for k in all_results})
        selections[cohort]=selection
    write(OUT/'SELECTION.json',dict(by_cohort=selections,created=time.time(),
        evaluation_used=False,locked_before_new_evaluation=True,historically_exposed=True,
        total_seconds=time.time()-tick,lock_sha256=sha(OUT/'LOCK.json')))
    status(OUT/'STATUS.json',dict(stage='selection_locked',finished=False,chosen={c:s['chosen'] for c,s in selections.items()}))


def evaluate():
    import torch
    from scripts.run_closure_v1 import model_for
    from scripts.run_spatial10_components_v1 import capture_timed
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    from vg_tta.foreground_runtime import state_digest
    p=plan(True);selection=read(OUT/'SELECTION.json')
    for cohort,rr in p['rows'].items():
        rows=[r for r in rr if r['f37_role']=='evaluation']
        config=selection['by_cohort'][cohort]['chosen']
        pending=[r for r in rows if not recorded(OUT/'eval'/keyfile(r['key']))]
        if not pending:continue
        model=model_for(cohort);digest=state_digest(model)
        for r in pending:
            cache=load(OUT/'cache'/keyfile(r['key']));fits={};audits={}
            frames,_,records,views,qa,cost=capture_timed(model,r)
            assert qa==cache['qa'];del views
            a4=load(cache['A4_reference']['path'])['fits']['A4']
            assert sha(cache['A4_reference']['path'])==cache['A4_reference']['sha256']
            audits['T0']=full_prediction(model,frames,cache['frame_ids'],r['input'],r['subject'],a4['state'],cache['T0'])['audit']
            for arm in ('T1','T2','T3','T4'):
                zz=run_fit(cache,config,arm)
                if arm!='T2':
                    audits[arm]=full_prediction(model,frames,cache['frame_ids'],r['input'],r['subject'],
                        {**a4['state'],**zz['state']},zz['final'])['audit']
                fits[arm]=zz
            record(OUT/'eval'/keyfile(r['key']),dict(key=r['key'],cohort=cohort,group=r['group'],source=r['source'],
                config=config,frame_ids=cache['frame_ids'],T0=cache['T0'],fits=fits,audits=audits,
                selection_sha256=sha(OUT/'SELECTION.json'),GT_online=False,source_restored=True,new_DINO=0))
            assert state_digest(model)==digest
            print('F40 eval',r['key'],config,{n:z['final']['physical_interval'] for n,z in fits.items()},flush=True)
            status(OUT/'STATUS.json',dict(stage='evaluation',last=r['key'],finished=False))
        del model;gc.collect();torch.cuda.empty_cache()


def summarize():
    from scripts.analyze_spatial10_components_v1 import labels_for,checked_score,summary
    from scripts.analyze_dense_support_v1 import behavior
    p=plan(True);selection=read(OUT/'SELECTION.json')
    rows=[r for rr in p['rows'].values() for r in rr if r['f37_role']=='evaluation']
    gt=labels_for(rows);old={r['key']:r for r in read(PARENT/'EVAL_RESULTS.json')['rows']}
    records=[];names=['T0','T1','T1_F39','T2','T3','T4']
    for r in rows:
        path=OUT/'eval'/keyfile(r['key']);assert recorded(path)
        x=load(path);cache=load(OUT/'cache'/keyfile(r['key']));arms={};details={}
        arms['T0'],_=checked_score(x['T0']['boxes'],gt[r['key']],x['frame_ids'],x['T0']['indices'])
        arms['T1_F39']=old[r['key']]['arms']['T1']
        for arm,z in x['fits'].items():
            scored=score_fit(cache,z,gt[r['key']]);arms[arm]=scored['metrics']
            details[arm]={k:v for k,v in scored.items() if k not in ('metrics','T0')}
            details[arm].update(interval=z['final']['physical_interval'],
                behavior=behavior(old[r['key']]['native_physical_interval'],z['final']['physical_interval']))
        assert all(arms[n]['sIoU']==arms['T0']['sIoU'] for n in names)
        records.append(dict(key=r['key'],cohort=x['cohort'],group=r['group'],source=r['source'],query=r['input']['caption'],
            arms=arms,details=details,native_interval=old[r['key']]['native_physical_interval'],
            GT_interval=old[r['key']]['GT_interval'],event_length_group=old[r['key']]['event_length_group'],
            path=str(path),sha256=sha(path),audits=x['audits']))
    summaries={}
    for cohort in p['rows']:
        rr=[r for r in records if r['cohort']==cohort]
        s=summary(rr,names,METRICS,[(n,'T0') for n in names if n!='T0']+[('T1',n) for n in ('T1_F39','T2','T3','T4')])
        s['behavior']={arm:{k:sum(r['details'][arm]['behavior']==k for r in rr)
            for k in ('expansion','shrink','shift_or_mixed','no_op')} for arm in ('T1','T2','T3','T4')}
        s['parameter_changes']={arm:sum(r['details'][arm]['parameter_changed'] for r in rr) for arm in ('T1','T2','T3','T4')}
        summaries[cohort]=s
    write(OUT/'EVAL_RESULTS.json',dict(rows=records,summary=summaries,selection=selection,
        historical_exposure=True,untouched=False,GT_online=False))
    lines=['# F40：固定 A4 的时间分支宽范围逐坐标调参','',
        '两方向分别在原8开发来源选参，锁定后各16评价来源一次复验。历史暴露池，不是全量或全新测试。A4/教师/损失形式/时间头/原生解码未改。','',
        '## 锁定配置与结果','']
    for cohort,s in summaries.items():
        pick=selection['by_cohort'][cohort]
        lines += [f'### {cohort}', '',f"配置：{pick['chosen']}；开发 Δv={pick['development_delta_v']*100:+.4f} pp；{pick['unique_configs']} 个不同配置。",'',
            '| 条件 | vIoU % | 固定 GT 帧 sIoU % | tIoU % |','|---|---:|---:|---:|']
        for n in names:
            v=s['arms'][n]
            lines.append(f"| {n} | {v['vIoU_corrected']['mean']*100:.4f} | {v['sIoU']['mean']*100:.4f} | {v['tIoU']['mean']*100:.4f} |")
        lines+=['','| 比较 | Δv pp | 配对95% CI pp |','|---|---:|---:|']
        for pair,cc in s['contrasts'].items():
            v=cc['vIoU_corrected'];a,b=v['ci95']
            lines.append(f"| {pair} | {v['mean']*100:+.4f} | [{a*100:+.4f}, {b*100:+.4f}] |")
        lines+=['',f"T1 参数变化：{s['parameter_changes']['T1']}/16；区间行为：{s['behavior']['T1']}。",'']
    lines+=['## 定义与边界','',
        'T0=A4+原生时间；T1=本轮锁定宽搜时间网络参数TTA；T1_F39=原共同lr.01/beta1/5步。T2同新配置优化输出logits而非网络；T3错query、T4时间循环错位。T2未独立调参，不当最优输出算法或上界。',
        'GT仅离线开发选参与评分；episode内最优实际状态仍由无标签loss选择。没有逐样例GT回退，没有插值/固定跨度替换，也不按评价结果第二次挑配置。',
        '本轮扩大的是可学习时间分支的配置机会；若仍不能胜原生，只限定本目标/接口/搜索预算，不否定所有时间TTA。两个方法注册和A4保持不变。','']
    dest=OUT/'REPORT.md';assert not dest.exists();dest.write_text('\n'.join(lines))
    status(OUT/'STATUS.json',dict(stage='evaluated_pending_final_audit',finished=False,
        results={c:s['contrasts']['T1 - T0']['vIoU_corrected'] for c,s in summaries.items()}))
    print('\n'.join(lines),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','capture-dev','search','capture-eval','evaluate','summarize'])
    args=ap.parse_args()
    if args.stage=='prepare':prepare();return
    if args.stage=='summarize':summarize();return
    from scripts.run_decota_refine_v1 import configure
    configure()
    import torch
    assert torch.cuda.is_available(),'CUDA unavailable; do not silently change precision/device'
    lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if args.stage=='capture-dev':capture('development')
        elif args.stage=='capture-eval':capture('evaluation')
        elif args.stage=='search':search()
        elif args.stage=='evaluate':evaluate()
    finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


if __name__=='__main__':main()
