"""Draft-only report from completed, audited round6 aggregate evidence.
No training/scoring/model/network calls. Main is deliberately gated on actual final
completion and all ten sealed source-VAL scores. Never alters final publications.
"""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, math

B = Path(__file__).resolve().parents[2]
R = B / 'research/round6'
RUN = B / 'crossplatform/round6/waymo_source100'
ARMS = ('official_full_last_bbf', 'text_full_last_bbf')
DRAFT_NAMES = ('REPORT_DRAFT.md', 'DECISION_DRAFT.json', 'ROOT_SUMMARY_DRAFT.json')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def safe_local(relative):
    q = PurePosixPath(relative)
    assert q.parts and not q.is_absolute() and '..' not in q.parts
    p = B / q.as_posix()
    assert p.resolve().is_relative_to(B.resolve())
    return p

def fraction(count, n):
    assert isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= n
    assert isinstance(n, int) and n > 0
    return count / n

def validate_summary(s, expected_arm):
    assert s['arm'] == expected_arm and s['n'] == 3782 and s['sourceVAL'] is True
    assert s['native_yaw'] == 0
    assert s['label_assisted'] == expected_arm.startswith('official_')
    for k in ('mean_iou', 'equal_scene_mean_iou', 'mean_candidate_oracle_iou_GT_assisted'):
        assert math.isfinite(s[k]) and 0 <= s[k] <= 1
    top = {str(k): {str(t): fraction(s['topk'][str(k)]['acc' + str(t)], s['n'])
                          for t in (25, 50)} for k in (1, 5, 10, 256)}
    for t in ('25', '50'):
        assert all(top[str(a)][t] <= top[str(b)][t] for a,b in ((1,5),(5,10),(10,256)))
    for k in top: assert top[k]['50'] <= top[k]['25']
    return {'arm': s['arm'], 'n': s['n'], 'mean_iou': s['mean_iou'],
            'equal_scene_mean_iou': s['equal_scene_mean_iou'],
            'n_scene_clusters': s['n_scene_clusters'],
            'mean_candidate_oracle_iou_GT_assisted': s['mean_candidate_oracle_iou_GT_assisted'],
            'topk': top,
            'topk_counts': {str(k): {str(t): s['topk'][str(k)]['acc'+str(t)] for t in (25,50)} for k in (1,5,10,256)},
            'label_assisted': s['label_assisted'], 'native_yaw': 0}

def fmt_percent(x): return f'{100*x:.4f}%'
def count_percent(count, n): return f'{count}/{n}（{fmt_percent(fraction(count,n))}）'
def duration(seconds): return f'{seconds:.2f}秒（{seconds/60:.2f}分钟）'

def load_completed_evidence():
    # Completion gates precede any output or result-table generation.
    complete_path = RUN / 'TRAINING_COMPLETE.json'
    done = read(complete_path)
    assert done['state'] == 'training_completed_source_only_epoch100'
    assert done['primary_epoch'] == 100 and done['completed_optimizer_steps'] == 15300
    audit_path = R / 'audit/FINAL_TRAINING_INDEPENDENT_AUDIT.json'
    a = read(audit_path)
    assert a['state'] == 'passed_actual_final_epoch100_source_checkpoint_audit'
    assert a['actual_completed_epochs'] == 100 and a['actual_optimizer_updates'] == 15300
    assert a['training_complete_sha256'] == sha(complete_path)
    assert a['auditor_source_sha256'] == sha(R/'audit/audit_final_training.py')
    assert a['formal_plan_sha256'] == done['plan_sha256'] == sha(R/'FORMAL_PLAN_LOCK.json')
    assert a['primary_checkpoint_sha256'] == done['primary_checkpoint_sha256']
    assert a['all_15300_completed_steps_finite_nonzero_grad_and_registered_LR']
    assert a['source_only_provenance_and_fixed_epoch100_no_metric_selection']
    assert a['source_train_val_contracts_bound'] and a['all_checkpoint_VAL_resume_events_ordered']
    for name, digest in a['source_code_hashes'].items():
        assert sha(B/'crossplatform/round6'/name) == digest
    milestones = {x['epoch']: x for x in a['all_10_milestone_checkpoint_and_VAL_prediction_hashes_verified']}
    assert sorted(milestones) == list(range(10,101,10))
    config = read(R/'EXPORT_FINAL_SOURCES.json')
    paths = [safe_local(p) for p in config['source_val_results']]
    assert len(paths) == len(set(paths)) == 10
    rows = []; provenance = []; score_cpu_seconds = 0.
    for p in paths:
        seal = read(p.parent/'SEALED.json'); v = read(p)
        assert seal['state'] == 'sourceVAL_CPU_scoring_complete'
        assert seal['files'][p.name] == sha(p)
        assert seal['scorer_program_lock_sha256'] == sha(R/'source_validation_scoring/SCORER_PROGRAM_LOCK.json')
        ep = v['epoch']; assert ep in milestones and v['primary_epoch'] == 100 and v['n'] == 3782
        assert seal['epoch'] == ep and seal['n'] == 3782
        assert seal['checkpoint_sha256'] == v['checkpoint_sha256'] == milestones[ep]['checkpoint_sha256']
        assert seal['prediction_seal_sha256'] == v['prediction_seal_sha256'] == milestones[ep]['val_seal_sha256']
        assert v['source_val_order_sha256'] == done['source_val_contract']['order_sha256']
        assert v['checkpoint_selection_does_not_change_from_fixed100']
        assert v['actual_batch_size'] == 24 and v['last_batch'] == 14
        assert v['independent_selected_audit']['all_rows_all8readouts_selectedTop1_verified']
        by_name = {x['arm']: x for x in v['summaries']}; assert len(by_name) == 8
        rows.append({'epoch': ep, 'primary': ep==100, 'checkpoint_sha256': v['checkpoint_sha256'],
                     'prediction_seal_sha256': v['prediction_seal_sha256'],
                     'arms': {arm: validate_summary(by_name[arm], arm) for arm in ARMS},
                     'padding_length_min_max': v['padding_length_min_max'],
                     'CPU_scoring_seconds': v['seconds']})
        provenance.append({'epoch': ep, 'results_sha256': sha(p), 'score_seal_sha256': sha(p.parent/'SEALED.json')})
        score_cpu_seconds += v['seconds']
    rows.sort(key=lambda x:x['epoch']); assert [x['epoch'] for x in rows] == list(range(10,101,10))
    attempt_cost = []; peak = 0; all_step_seconds = 0.; all_logged_epoch_seconds = 0.
    for adir in sorted((RUN/'attempts').iterdir()):
        if not adir.is_dir(): continue
        ep = adir/'events.jsonl'; sp = adir/'steps.jsonl'
        assert ep.name in a['step_log_hashes'][adir.name]
        assert sha(ep) == a['step_log_hashes'][adir.name][ep.name]
        events = [json.loads(x) for x in ep.read_text().splitlines() if x.strip()]
        terminal = any(x['event']=='TRAINING_COMPLETE' for x in events)
        elapsed = max(x['elapsed_seconds'] for x in events)
        attempt_cost.append({'attempt': adir.name, 'last_logged_elapsed_seconds': elapsed,
                             'has_TRAINING_COMPLETE_event': terminal,
                             'unlogged_interruption_or_shutdown_tail_not_measured': not terminal})
        all_logged_epoch_seconds += sum(x['train_seconds'] for x in events if x['event']=='epoch_training_complete')
        if sp.exists():
            assert sha(sp) == a['step_log_hashes'][adir.name][sp.name]
            for line in sp.read_text().splitlines():
                x=json.loads(line); peak=max(peak, x['peak_allocated_bytes'])
                all_step_seconds += x['forward_backward_update_seconds']
    assert any(x['has_TRAINING_COMPLETE_event'] for x in attempt_cost)
    timing = read(B/'crossplatform/round6/timing12/TIMING.json')
    assert timing['state']=='engineering12_completed_weights_discarded_not_scientific_checkpoint'
    data = read(R/'source_data_audit/PUBLIC_SCALAR_SUMMARY.json')
    costs = {'canonical_train_including_worker_data_wait_seconds': a['cost_seconds']['canonical_train_total'],
             'scheduled_source_VAL_prediction_and_seal_seconds': a['cost_seconds']['scheduled_source_val_total'],
             'source_VAL_CPU_scoring_seconds_sum_may_overlap_training': score_cpu_seconds,
             'all_attempts': attempt_cost,
             'all_attempts_last_logged_elapsed_seconds_sum': sum(x['last_logged_elapsed_seconds'] for x in attempt_cost),
             'final_attempt_runner_elapsed_seconds': done['elapsed_attempt_seconds'],
             'all_attempts_logged_completed_epoch_train_seconds': all_logged_epoch_seconds,
             'all_attempts_recorded_train_forward_backward_update_seconds': all_step_seconds,
             'training_observed_peak_allocated_bytes': peak,
             'training_observed_peak_allocated_GiB': peak/2**30,
             'peak_scope': 'Training-step allocator observations across all attempts; sourceVAL not separately instrumented; not whole-process reserved/VRAM peak.',
             'timing12_separate': {'n_steps':12, 'warmup_steps':4, 'timed_steps':8,
                                  'steady_end_to_end_step_seconds':timing['timed_mean_seconds'],
                                  'steady_forward_loss_backward_update_seconds':timing['timed_gpu_mean_seconds'],
                                  'peak_allocated_bytes':timing['peak_allocated_bytes'],
                                  'max_recorded_event_elapsed_seconds':max(x['elapsed_seconds'] for x in timing['events'])},
             'sum_scope': 'Attempt elapsed includes startup, worker/data waiting, training, scheduled VAL, saving and sealing. CPU scoring may overlap and must not be added as serial GPU time. Interrupted unlogged tails are unknown.'}
    return done,a,data,rows,costs,provenance

def render(done, audit, data, rows, costs, provenance):
    primary=next(x for x in rows if x['epoch']==100)
    lines=['# 3EED Round6：Waymo-only 监督源模型训练（审核草稿）', '',
           '本次已完成固定100轮、15,300次优化更新的Waymo-only监督训练。主模型固定为epoch100；每十轮源域验证只用于监控，不按验证分数替换主模型。通用RoBERTa预训练允许，视觉与grounding模块按官方构造随机初始化，没有载入三平台联合任务权重。', '',
           '**结论边界：监督训练完成与源域表现分别成立；本轮尚未完成该源模型的目标平台冻结评测，也未执行TTA。源域结果不能证明跨域收益。**', '',
           '## 全十次源域验证', '',
           '所有行均为Waymo VAL的3,782条官方eligible表达，B24动态padding（末批14）。`official_full_last_bbf`使用真实源GT类别的同义词span辅助读出；`text_full_last_bbf`在同一个完整描述forward上使用固定纯文本首名词读出，不是类别词干预forward。下面各TopK指标表示按预测分数前K候选中是否存在IoU严格大于阈值的框；K>1以及all256是使用GT核验候选覆盖的诊断，不能当作可部署Top1定位准确率。未重新计算任何GT oracle。', '']
    for arm in ARMS:
        lines += [f'### {arm}', '', '| Epoch | mIoU | Top1 Acc25 | Top1 Acc50 | Top5覆盖25 / 50 | Top10覆盖25 / 50 | all256覆盖25 / 50 |', '|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            s=row['arms'][arm];t=s['topk']; n=s['n']; c=s['topk_counts']
            pair=lambda k:f"{count_percent(c[k]['25'],n)} / {count_percent(c[k]['50'],n)}"
            lines.append(f"| {row['epoch']}{'（主）' if row['primary'] else ''} | {fmt_percent(s['mean_iou'])} | {count_percent(c['1']['25'],n)} | {count_percent(c['1']['50'],n)} | {pair('5')} | {pair('10')} | {pair('256')} |")
        lines += ['']
    lines += ['本轮采用“保留官方读出协议的float64源VAL评分”：保留原native GPU Top10顺序、GT类别span、动态padding和尺寸clamp协议；实际源GT先按float32预处理，再提升为float64进行几何计算，并用独立一般3D交集实现核验全部八种读出的selected Top1。本轮未逐例执行官方float32 IoU原函数，不声称与该原函数逐位一致。mIoU为按表达平均的Top1 IoU；JSON另保留scene等权mIoU和分母。原生预测为6D中心/尺寸、yaw=0，尺寸按官方规则下限1e-6；本轮没有PCA或几何修正。主表不与Round5的B1联合模型直接计算方法delta，因为模型训练来源与批处理/token padding条件均不同。', '', '## 数据覆盖与排除', '', '| 源split | 官方列出场景 | 实际可用 | 缺失目录 | 帧 | raw对象 | eligible表达 | 无类别同义词span排除 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for part,label in [('source_train','TRAIN'),('source_val','VAL')]:
        d=data[part]; lines.append(f"| {label} | {d['official_listed_scenes']} | {d['present_scenes']} | {d['official_listed_scenes']-d['present_scenes']} | {d['all_actual_frames']} | {d['raw_objects']} | {d['eligible_actual_loader_annos']} | {d['ineligible_no_class_synonym_span']} |")
    lines += ['',f"TRAIN/VAL场景和帧交集均为0。TRAIN每轮153批、drop_last丢10条；VAL158批、末批14且不丢条。训练/验证分别有{data['source_train']['eligible_expressions_sampling_with_replacement']}/{data['source_val']['eligible_expressions_sampling_with_replacement']}条表达的原始点不足16,384，按官方有放回采样。缺失目录及官方过滤不会由合成样本补齐。仅使用XYZ+文本，不使用RGB；本报告没有输出真实场景/帧标识、caption或GT坐标。", '', '## 实际成本与计量范围', '',
           f"- 100轮canonical训练（含worker启动、数据等待）：{duration(costs['canonical_train_including_worker_data_wait_seconds'])}。",
           f"- 十次源域VAL预测、写盘和封存：{duration(costs['scheduled_source_VAL_prediction_and_seal_seconds'])}。",
           f"- 全attempt最后已记录elapsed之和：{duration(costs['all_attempts_last_logged_elapsed_seconds_sum'])}；最终attempt的runner完成计时：{duration(costs['final_attempt_runner_elapsed_seconds'])}。各attempt明细见ROOT_SUMMARY_DRAFT.json；若中断后没有终止事件，未记录尾部时间未知，不把该和声称精确总墙时。",
           f"- 全attempt训练步骤观测峰值allocated：{costs['training_observed_peak_allocated_bytes']:,} bytes（{costs['training_observed_peak_allocated_GiB']:.4f} GiB）。VAL未单独采集峰值；这不是reserved或整机显存峰值。",
           f"- 源VAL CPU评分累计：{duration(costs['source_VAL_CPU_scoring_seconds_sum_may_overlap_training'])}，可能与训练重叠，不作为串行GPU耗时相加。",
           f"- 独立12-step工程计时（4 warmup + 8 timed，权重丢弃）：稳态端到端每步{costs['timing12_separate']['steady_end_to_end_step_seconds']:.6f}秒，forward/loss/backward/update {costs['timing12_separate']['steady_forward_loss_backward_update_seconds']:.6f}秒；峰值allocated {costs['timing12_separate']['peak_allocated_bytes']:,} bytes；全部事件最后记录elapsed {duration(costs['timing12_separate']['max_recorded_event_elapsed_seconds'])}。该工程成本独立于正式训练，不用短计时外推替代实测总成本。", '',
           '## 官方配方及明确适配', '',
           '采用官方FP32 batch24、AdamW三参数组（含空的冻结文本组）、基础/backbone/text LR分别1e-4/1e-3/1e-5、weight decay 0.0005、梯度裁剪0.1。Hungarian matcher为class1/bbox0/GIoU2；loss为8×query objectness与七个预测head的CE+5×bbox+GIoU+contrastive平均，bbox内部center L1+0.2×size L1。query topk4，contrastive温度0.07。视觉BN momentum0.1保持训练；RoBERTa权重冻结但model.train下dropout仍活动。', '',
           'scheduler每次optimizer更新后前进；官方warmup=-1的真实语义使名义25/26里程碑在第3978/4131次更新后生效，即26/27轮末，低LR首次用于27/28轮，没有静默校正。数据使用原始16,384点随机采样与Waymo固定suffix坐标旋转；声明的增强helper未被官方getter实际调用。', '',
           '适配仅包括：显式全局seed0（原入口rng_seed标志未全局生效）、四worker采用spawn避免CUDA/NCCL后的fork问题、真实单rank NCCL/DDP、原始训练配方上的日志/原子checkpoint与可恢复RNG。无AMP、梯度累积、缩小batch、目标平台训练或目标标签选模。原loader未排序目录；登记的数据顺序hash与独立join绑定实际样本顺序，不能仅凭seed声称跨文件系统逐位复现。', '',
           '## 主模型及证据', '',
           f"- 固定epoch100 checkpoint SHA256：`{done['primary_checkpoint_sha256']}`。",
           f"- 最终可恢复状态SHA256：`{done['final_resume_checkpoint_sha256']}`（模型/optimizer/scheduler/RNG张量仅本地保存，不公开）。",
           f"- 正式计划SHA256：`{done['plan_sha256']}`。",
           f"- 完成收据SHA256：`{audit['training_complete_sha256']}`；独立训练审计SHA256：`{sha(R/'audit/FINAL_TRAINING_INDEPENDENT_AUDIT.json')}`。",
           '- 训练完整轨迹已核15,300次有限loss/非零有限梯度及登记LR；十次checkpoint→预测封存→评分事务均核绑定。源VAL预测封存后由独立CPU scorer评分，并以一般3D交集实现核全部八种读出的Top1。', '',
           '本轮决定为保留并验收真正的Waymo-only监督源模型，主模型仍为epoch100。不据源域高低选择目标域方案，也不将监督源训练称作TTA贡献。后续目标域冻结评测或TTA需要独立计划和同条件对照。']
    summary={'state':'root_review_draft_only','primary_epoch':100,'completed_epochs':100,'completed_optimizer_steps':15300,
             'primary_checkpoint_sha256':done['primary_checkpoint_sha256'],'final_resume_sha256':done['final_resume_checkpoint_sha256'],
             'formal_plan_sha256':done['plan_sha256'],'final_training_audit_sha256':sha(R/'audit/FINAL_TRAINING_INDEPENDENT_AUDIT.json'),
             'source_validation_all10_epochs':rows,'costs':costs,
             'source_data_counts':{p:{k:v for k,v in data[p].items() if k in ('official_listed_scenes','present_scenes','all_actual_frames','raw_objects','eligible_actual_loader_annos','ineligible_no_class_synonym_span','eligible_expressions_sampling_with_replacement')} for p in ('source_train','source_val')},
             'source_train_val_scene_overlap':0,'source_train_val_frame_overlap':0,
             'evidence':provenance,'target_frozen_evaluation_completed_by_this_round':False,'TTA_executed_by_this_round':False,
             'no_round5_B1_joint_model_method_delta':True,'no_PCA_or_new_GT_oracle_computation':True}
    decision={'state':'draft_pending_root_review_not_final_adoption','primary_epoch':100,'primary_checkpoint_sha256':done['primary_checkpoint_sha256'],
              'decision':'验收并保留完成独立审计的Waymo-only监督源模型；固定epoch100，不按源VAL或目标指标替换。',
              'supervised_training_completed':True,'source_domain_validation_completed':True,
              'target_domain_frozen_generalization_tested_by_this_round':False,'TTA_benefit_tested_by_this_round':False,
              'scope':'源域表现不能推出跨域收益；未执行TTA，无部署或方法收益声明。',
              'not_a_round5_B1_joint_model_comparable_method_delta':True}
    return '\n'.join(lines)+'\n',decision,summary

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.parse_args()
    assert all(not (R/name).exists() for name in DRAFT_NAMES), 'Never overwrite an existing review draft.'
    evidence=load_completed_evidence()
    report,decision,summary=render(*evidence)
    payload={DRAFT_NAMES[0]:report,DRAFT_NAMES[1]:json.dumps(decision,indent=2,ensure_ascii=False,allow_nan=False)+'\n',DRAFT_NAMES[2]:json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n'}
    for text in payload.values():
        assert '/home/' not in text and 'meta_path' not in text and 'ground_info' not in text
    for name,text in payload.items(): (R/name).write_text(text)
    print(json.dumps({'state':'three_drafts_written_for_root_review_only','files':{name:sha(R/name) for name in DRAFT_NAMES}}))

if __name__=='__main__': main()
