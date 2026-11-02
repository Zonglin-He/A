"""R51 saved-only final report builder; no reads or execution on import.

Root must pin all three completed scoring seals, independent audits, normalized
sealed-cost/history projections and a closed TTA decision before execution.
This program never imports scoring, producers, a model, or a GT evaluator.
"""
from pathlib import Path
import argparse, csv, hashlib, io, json, math

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'studies/3eed_single_grounder_evidence_replacement_20261102'
PLATFORMS = ('drone', 'quad')
FULL_ARMS = ['source', 'external_G', 'replace_all', 'no_relation', 'no_instance_association']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def path(s):
    p = Path(s)
    assert not p.is_absolute() and '..' not in p.parts
    assert 'round46/final' not in s and 'FINAL384' not in s
    return ROOT / p
def dump(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def finite_tree(x):
    if isinstance(x, dict):
        for v in x.values(): finite_tree(v)
    elif isinstance(x, list):
        for v in x: finite_tree(v)
    elif isinstance(x, float): assert math.isfinite(x)
def table(summary, scope, rows):
    s = summary[scope]
    out = ['| 配置 | Drone mIoU | Quad mIoU | 平台等权 mIoU | 相对 source | 相对外挂 G |',
           '|---|---:|---:|---:|---:|---:|']
    avg = s['balanced_mIoU_percent']
    for label, arm in rows:
        v = avg[arm]
        out.append(f'| {label} | {s["drone"][arm]["mIoU_percent"]:.6f} | {s["quad"][arm]["mIoU_percent"]:.6f} | {v:.6f} | {v-avg["source"]:+.6f} | {v-avg["external_G"]:+.6f} |')
    return '\n'.join(out)

def accuracy_table(summary, scope, rows):
    s=summary[scope]
    lines=['| 配置 | Drone Acc@25 | Drone Acc@50 | Quad Acc@25 | Quad Acc@50 |',
           '|---|---:|---:|---:|---:|']
    for label,arm in rows:
        values=[s[p][arm][key] for p in PLATFORMS for key in ('Acc25_percent','Acc50_percent')]
        lines.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    return '\n'.join(lines)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', required=True); ap.add_argument('--authorization', required=True)
    a = ap.parse_args(); plan = read(a.plan); auth = read(a.authorization)
    assert auth['state'] == 'ROOT_AUTHORIZED_R51_SAVED_ONLY_FINAL_REPORT'
    assert auth['program_sha256'] == sha(__file__) and auth['plan_sha256'] == sha(a.plan)
    assert plan['state'] == 'ROOT_LOCKED_R51_SAVED_ONLY_FINAL_REPORT'
    assert plan['prefix'] == PREFIX and plan['no_GT_media_models_or_future_artifacts'] is True
    pins = plan['inputs_sha256']
    def get(rel):
        assert rel in pins and sha(path(rel)) == pins[rel], ('input_pin', rel)
        return read(path(rel))
    runs = {}
    for label, n, arm_count in [('INITIAL_DEV',336,16),('REPAIR_DEV',336,6),('FULL',5912,5)]:
        cfg = plan['score_runs'][label]; base = cfg['score_root']; seal = get(base+'/SEALED.json')
        # All pinned completed scoring members are checked, including every
        # task row and negative, although writing uses saved summaries only.
        for rel,h in seal['files'].items():
            pp = Path(rel); assert not pp.is_absolute() and '..' not in pp.parts
            assert sha(path(base)/pp) == h
        receipt = get(base+'/SCORING_RECEIPT.json')
        assert receipt['state'] == 'PASS_R51_SINGLE_POSTSEAL_CONFIGURED_ORIGINAL_BOX_TASK_SCORE'
        assert receipt['n'] == n and len(receipt['arm_names']) == arm_count
        assert receipt['task_values'] == n*arm_count
        assert receipt['new_GT_scored_task_values']+receipt['reused_saved_task_values'] == n*arm_count
        for k in ('independent_prediction_audit','saved_scalar_audit'):
            evidence = get(cfg[k]); assert evidence['state'].startswith('PASS_')
        runs[label] = {'summary':get(base+'/TASK_SUMMARY.json'),
                       'scenes':get(base+'/ALL_SCENE_SUMMARIES_PRIVATE.json'),
                       'selection':get(base+'/SHARED_CALIBRATION_SELECTION.json'), 'receipt':receipt}
    initial=runs['INITIAL_DEV']; repair=runs['REPAIR_DEV']; full=runs['FULL']
    assert full['receipt']['arm_names'] == FULL_ARMS
    chosen = initial['selection']; assert chosen['selected_shared_factor'] == .5
    dm=repair['summary']['ALL336']['balanced_mIoU_percent']
    assert dm['grouped_replace_all_c05'] <= dm['source']+1e-10
    closure=get(plan['closed_decision'])
    assert closure['TTA_executed'] is False and closure['TTA_gate_passed'] is False
    assert closure['B_capture_executed'] is False and closure['selected_factor'] == .5
    assert closure['selected_method'] == 'grouped_replace_all_c05'
    assert closure['full_mandatory_bounded_evaluation'] is True
    costs=get(plan['normalized_costs']); history=get(plan['normalized_history'])
    assert set(costs)=={'counts','timing_seconds','peak_bytes','flags'}
    assert all(isinstance(d,dict) and all(v is None or type(v) in (bool,int,float) for v in d.values()) for d in costs.values())
    for key in ('full_raw_preparation','full_reference_capture','full_grouped_G','full_noassociation'):
        assert key in costs['timing_seconds'] and type(costs['timing_seconds'][key]) in (int,float)
        assert math.isfinite(costs['timing_seconds'][key]) and costs['timing_seconds'][key]>=0
    for key in ('full_reference_calls','full_subject_calls_reused'):
        assert key in costs['counts'] and type(costs['counts'][key]) is int and costs['counts'][key]>=0
    assert costs['counts']['full_reference_calls']==10465
    assert 'full_reference_model' in costs['peak_bytes'] and type(costs['peak_bytes']['full_reference_model']) in (int,float)
    assert math.isfinite(costs['peak_bytes']['full_reference_model']) and costs['peak_bytes']['full_reference_model']>=0
    assert set(history)=={'events'}
    # Root supplies an explicit closed whitelist projection of real sealed
    # COUNTS/COSTS/attempt receipts; no raw logs, guesses, or last-job scans.
    for role,rel in plan['cost_history_source_artifacts'].items():
        assert role and rel in pins; get(rel)
    fm=full['summary']['ALL5912']['balanced_mIoU_percent']
    delta={ref:fm['replace_all']-fm[ref] for ref in ('source','external_G')}
    decision={'state':'R51_CLOSED_SAVED_RESULT_REPORT', 'selected_method':'grouped_replace_all_c05',
        'selected_factor':.5,'selection_uses_exposed_DEV_labels':True,
        'full_balanced_mIoU_percent':fm,'full_delta_vs_source_and_external_pp':delta,
        'descriptive_full_strict_gain_over_both':all(v>1e-10 for v in delta.values()),
        'automatic_method_adoption_claimed':False,'TTA_gate_passed':False,'TTA_executed':False,
        'B_capture_executed':False,'TTA_parameter_effectiveness_not_tested':True,
        'full_no_relation_not_posthoc_selected':True,'all_unknown_fallback_negative_rows_retained':True,
        'no_new_scoring_or_method_change':True}
    report=['# R51：单 grounding 模型证据替换与候选载体诊断', '',
        f'全量5912条上，锁定 grouped 替换的两平台等权 mIoU 为 **{fm["replace_all"]:.6f}%**；相对 frozen source **{delta["source"]:+.6f} pp**，相对外挂 G **{delta["external_G"]:+.6f} pp**。数值来自已封存、独立审核的一次任务评分。是否显示总体增益由这两个实际差值判断；表中去关系和去关联结果用于消融，不据其分数更高而事后改选主方法。', '',
        '固定源模型为 Waymo-only epoch100，本轮没有新增源域训练。单模型版本保留点云、RGB、完整文字和标定；RGB 在模型外以可见点像素测量参与 G，未训练新的 RGB 编码分支。主体候选与参照短语调用同一份 3EED 权重，输出始终从源模型原始256个框中选择。', '',
        'DEV监督资格门未通过：grouped 修复仍低于 source，因此 B 观测采集和参数 TTA 均未运行。本轮没有检验匹配 MLP 或 cross-attention Q/V LoRA 的参数更新效果，不能把未运行解释为参数适应无效。全量是用户要求的固定方法覆盖评估，并不撤销开发阶段的停止决定。', '',
        '## 开发阶段六组替换（固定 c=.5）', '',
        table(initial['summary'],'ALL336',[
            ('冻结 source','source'),('完整外挂 G','external_G'),('只替换实例路径','replace_instances'),
            ('只替换外观证据','replace_appearance_c05'),('只替换参照定位','replace_references'),('初始全替换','replace_all_c05')]), '',
        '系数只由原三系数全替换的开发平台等权规则选择，并对两个平台共享。原16臂、全部其他系数、UNKNOWN、回退、未更新及所有负比较均保留；此处主表不是删去不利试验后的结果。', '',
        '这些是组件替换敏感性，不能当成可加的独立贡献。Qwen 替换版仍保留 Mask、DINO、位置与关系；缺少其余条件相同的 zero-appearance 对照，故其相对 source 的收益不能归因于颜色本身。实例路径的下降也未被全体独立身份标签拆解，可能同时包含前景/邻物污染、映射与排序变化，以及同一实例不同 query 的框质量变化。', '',
        '## 唯一候选载体修复与消融', '',
        table(repair['summary'],'ALL336',[
            ('冻结 source','source'),('完整外挂 G','external_G'),('初始全替换','initial_replace_all_c05'),
            ('无实例关联','no_instance_association_c05'),('grouped medoid 修复','grouped_replace_all_c05'),('修复去关系','grouped_no_relation_c05')]), '',
        '修复改变候选点支持的载体组织，从而连带改变可见颜色、参照分布和关系可用性。它不新增模型、不修改框，也不提供语义身份真值；不能把其总变化拆成未经单独测量的颜色或关系因果收益。去关系结果未用于替换锁定 grouped 主法。去关联对照保留 RGB/点云/文字，同时移除实例绑定和重复 query 抑制；它是联合结构消融，不是单独去掉 RGB，也未独立隔离重复抑制的贡献。', '',
        '## 全量固定五臂', '',
        table(full['summary'],'ALL5912',[(a,a) for a in FULL_ARMS]), '',
        accuracy_table(full['summary'],'ALL5912',[(a,a) for a in FULL_ARMS]), '',
        'source 与外挂 G 使用同5912人群、同原始256候选框的 R50 已保存标量及端点复核，不重复读取 GT 评分。这两个对照不是本轮重新训练或重跑的独立系统。新方法反馈不读取 Mask/Qwen/DINO 输出；旧外部缓存只用于外挂对照及中间替换诊断。', '',
        '### DEV 与其余样本', '']
    for scope in ('DEV336','NONDEV5576'):
        report += [f'**{scope}**', '',table(full['summary'],scope,[(a,a) for a in FULL_ARMS]),'']
    report += ['DEV336与NONDEV5576是既有全量人群的划分，不能声称NONDEV为整个研究历史从未接触的独立金标或新场景确认。来源相同不自动等于推理上下文相同；跨轮 whole-G 端点是否逐位相同，须以相应源点采样、文本和参照上下文桥接审计为准。', '',
        '### 全量场景结果（匿名场景簇）', '',
        '| 场景簇 | n | source | 外挂 G | grouped | 去关系 | 去关联 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    scene_rows=[r for r in full['scenes'] if r['scope']=='ALL5912']
    assert len(scene_rows)==5
    for pl in PLATFORMS:
        sr=sorted([r for r in scene_rows if r['platform']==pl],key=lambda r:r['scene_PRIVATE'])
        for j,r in enumerate(sr,1):
            report += ['| '+('D' if pl=='drone' else 'Q')+f'_S{j:02d} | {r["n"]} | '+' | '.join(f'{r["mIoU_percent"][arm]:.6f}' for arm in FULL_ARMS)+' |']
    report += ['', '场景簇数量有限，报告逐簇描述，不把样本数当作独立场景数，不构造无法支持的总体场景置信区间。候选 query 可重复覆盖同一物体；IoU 和 query 变化不单独认证实例身份。', '',
        '## 覆盖、负例与评价边界', '',
        'Acc@25/50 使用严格 IoU 阈值 >.25/>.5；完整平台、子集、unknown/fallback/no-update计数、改善/恶化数及场景标量随公开表保留。所有固定 source 框不变，因此本轮讨论的是候选选择和证据传递，未证明框回归修正。', '']
    for label,r in runs.items():
        q=r['receipt'];report += [f'- {label}：{q["task_values"]} 个表格 IoU，{q["new_GT_scored_task_values"]} 个本次新评分、{q["reused_saved_task_values"]} 个旧标量复用；保留全部 {q["ordered_negative_comparisons"]} 条有向负比较。']
    report += ['', '## 实际成本与技术记录', '',
        '| 已完成额外阶段 | 登记 wall（秒） | 平均阶段秒／条 |', '|---|---:|---:|',
        f'| 原始输入与实例属性准备 | {costs["timing_seconds"]["full_raw_preparation"]:.3f} | {costs["timing_seconds"]["full_raw_preparation"]/5912:.6f} |',
        f'| 全量参照 grounding 采集 | {costs["timing_seconds"]["full_reference_capture"]:.3f} | {costs["timing_seconds"]["full_reference_capture"]/5912:.6f} |',
        f'| grouped G 生成 | {costs["timing_seconds"]["full_grouped_G"]:.3f} | {costs["timing_seconds"]["full_grouped_G"]/5912:.6f} |',
        f'| 无实例关联消融 | {costs["timing_seconds"]["full_noassociation"]:.3f} | {costs["timing_seconds"]["full_noassociation"]/5912:.6f} |', '',
        f'全量参照实际调用 **{costs["counts"]["full_reference_calls"]} 次**；参照推理阶段 PyTorch 峰值已分配显存 **{costs["peak_bytes"]["full_reference_model"]/(2**30):.3f} GiB**，不含 CUDA 保留内存与其他进程。主体 source 调用复用 **{costs["counts"]["full_subject_calls_reused"]} 次**。本次以已有 source 缓存为给定输入；表列是额外阶段登记 wall，不能当作五臂各自完整重跑的端到端延迟，参照阶段 peak 也不是整条管线的全局峰值。', '',
        '公开成本 JSON 保留真实新调用、缓存逻辑使用、各阶段 wall 和 peak、失败尝试及复用数量。共享采集 wall 不是每个方法分别重跑的端到端延迟；source 使用缓存时本次新增 forward 为零，不等于 source 系统端到端成本为零。Stanza 是 CPU 神经解析器，解析条数与物理 forward 次数分别表述。', '',
        'Qwen target/reference 的成功新调用及此前失败尝试中的有效调用分别记账，不删除失败前已完成的数值调用；未消费的参照调用也保留成本。技术故障与修复记录不当作任务负例，未改变已封存预测或既有 GT 评分次数。具体计数以同行提供的 sealed COUNTS/COSTS 白名单投影为准，不以 worklist cap 代替实际调用。', '',
        '## 裁决', '',
        f'固定 grouped 路线全量相对 source/外挂 G 是否均严格增加：**{decision["descriptive_full_strict_gain_over_both"]}**。这项描述性比较不改变已经关闭的 DEV TTA 门、不追加规则、损失或系数、不反选消融。本轮可支持受控替换和载体机制诊断；任何方法采用主张须同时满足事前资格、实际总体指标及资源证据，不由工程 PASS 自动推导。', '']
    config={'prefix':PREFIX,'source_boxes_per_observation':256,'full_n':5912,'DEV_n':336,
        'source_platform':'waymo','source_epoch':100,
        'source_checkpoint_sha256':'0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd',
        'appearance_factor':.5,'shared_platform_factor':True,'full_arms':FULL_ARMS,
        'selected_method':'grouped_replace_all_c05','TTA_executed':False,
        'GT_used_for_DEV_factor_selection':True,'new_full_calibration':False,
        'full_existing_baselines_reused_not_rescored':True,'UNKNOWN_neutral_not_negative':True,
        'full_method_external_vision_cache_dependency':False,'parameter_updates':0,
        'frame_selection_no_new_boxes':True,'full_noR_not_selected_after_results':True,
        'new_RGB_encoder_training':False,'original_source_sampling_preserved':True,
        'grouping':{'point_Jaccard_min':.7,'bidirectional_coverage_min':.75,'minimum_points':6,
                    'complete_link':True,'mutual_source_centers_required':True,
                    'box_IoU_only_grouping':False,'ambiguous_membership_UNKNOWN':True,
                    'representative':'maximum_mean_member_point_Jaccard_tie_lowest_original_query',
                    'cross_query_point_union':False,'semantic_instance_identity_certified':False},
        'appearance':{'actual_globally_visible_point_pixels_only':True,
                      'alternative_hypothesis_shared_pixels_retained':True,'minimum_pixels':8,
                      'HSV_Lab_support':True,'pseudocount':.5,'logodds_clip':4.,
                      'reliability':'n_over_n_plus_8','complex_appearance':'UNKNOWN'},
        'references':{'same_grounder_only':True,'pure_text_headspan':True,
                      'source_fixed_top_k':3,'UNKNOWN_logit':0.,'truncated_mass_retained':True,
                      'minimum_support_points':3,'minimum_carrier_coverage':.15,
                      'source_logits_not_calibrated_identity_probabilities':True},
        'fusion_coefficients':{'appearance':2.,'global_position':2.,'reference_relation':2.},
        'relation_denominator':'one_plus_supported_probability_mass',
        'UNKNOWN_cue_exact_zero_after_anchoring':True,
        'measured_surface_not_full_object_boundary':True}
    finite_tree(decision)
    out=path(plan['output_directory']); assert not out.exists();out.mkdir(parents=True)
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    dump(out/'DECISION.json',decision);dump(out/'public_fixed_config.json',config)
    dump(out/'normalized_costs.json',costs);dump(out/'normalized_history.json',history)
    dump(out/'SOURCES.json',{'program_sha256':sha(__file__),'plan_sha256':sha(a.plan),'saved_input_sha256':pins,
         'no_new_GT_models_media_or_scoring_execution':True})
    dump(out/'SEALED.json',{'state':'R51_SAVED_ONLY_FINAL_REPORT_COMPLETE','files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}})

if __name__=='__main__': main()
