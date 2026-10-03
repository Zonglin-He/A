"""Build an evidence-bounded review of the actual frozen linear-probe run."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
P=ROOT/'results/tastvg_temporal_latent_quality/2026-10-03'
B=ROOT/'artifacts/tastvg_temporal_latent_quality_v1'
OLD=ROOT/'results/tastvg_large_correction_evidence/2026-10-03'
def read(p):return json.loads(Path(p).read_text())
def write(p,z):Path(p).write_text(json.dumps(z,indent=2,ensure_ascii=False)+'\n')
def val(z):return f'{100*z["mean"]:.2f}'
def ci(z):return '未定义' if z['mean'] is None else f'{100*z["mean"]:.2f} [{100*z["ci95"][0]:.2f}, {100*z["ci95"][1]:.2f}]'

def report():
    cfg=read(P/'CONFIG.json');fit=read(P/'SOURCE_FIT_SUMMARY.json');root=read(P/'ROOT_READBACK.json');audit=read(P/'VALIDATION.json')
    extra=read(P/'TOP1_DIAGNOSTICS.json');res=read(P/'RESOURCES.json');text=[]
    def add(s):text.append(s)
    add('# 候选条件 temporal latent：源监督线性探针审计\n')
    add('**这版线性探针没有建立可替换 A8 的共同收益。** Vid 确认完整 corruption 流几乎为零，HC 为负；它比部分激进边界信号更少破坏输出，但没有稳定超过原生起止先验。源验证能读出一定区间质量信息，不等于能在当前目标流可靠判断大幅修改。保留 A，本轮不晋升探针，不自动训练 MLP 或增加专家。\n')
    add('## 实际执行与数据边界\n')
    add('目标固定为前轮相同的两集各32开发＋16确认来源，一query/source、两序、clean＋五类5% corruption、25%专家：1152到达中288有完整候选（240corrupt、48clean），864非专家逐值保留A。全部来源有历史曝光；确认面板与本批开发来源互斥，不能称fresh。Old8完整包含于Expanded32，原A空间框、Uniform Rank-RKL的1792参数持续轨迹与Vid K1/HC K8、像素、专家位置、原Paper48采样、官方同域checkpoint全部固定。\n')
    add('源监督使用现有官方训练媒体，Vid126独立源沿用95拟合/31验证；HC64独立源按预定来源hash划48/16。源与目标来源ID及媒体SHA256互斥。Vid保留原5fps/max200采样，HC保留原64-frame索引配方（此面板合并65帧）；不按事件质量筛样、不增加媒体或专家。官方训练GT仅为预先生成的32候选提供physical tIoU标签；源验证只选ridge强度，不并入最终拟合。目标GT不参与拟合、选参、阈值或候选生成。两集所有分数/选择GLOBAL_SCORE_SEAL先于已封存dense GT候选指标关联。\n')
    add('这是一项**额外source-supervised probe**，不是零监督temporal heuristic，也不是latent TTA。源训练derived validation参与ridge选择，不是独立最终检验。训练媒体与GT已有项目曝光；源侧当前有限预算和目标历史曝光限制外推。\n')
    add('## 实际表示和拟合\n')
    add('捕获最终第六层时间decoder输入temp_embed的256维状态；两offset按真实frame ID合并。候选表示是 `[h_s,h_e,mu_inside,mu_left,mu_right,h_s-mu_left,h_e-mu_right]`，共1792维。左右各1秒，只平均实际观测帧；缺外侧上下文填零并计数。没有使用PE语义cosine、空间合理性、额外专家或GT事件边界。区间为 `[frame_i,frame_j+1)`。\n')
    add('FP64岭回归，训练行的均值/标准差标准化，拟合截距；alpha网格`.001,.01,.1,1,10,100,1000`。按源验证top1 tIoU最大、MSE最小、alpha最小依次选择，冻结后目标不再调整。几何对照G只含归一化start/end/length，使用相同标签、划分和alpha规则。L/G都用未裁剪回归值排序；不是已校准的IoU或安全概率。与前轮N/U/S相同，唯一最大分且严格超过A8才替换，最高分并列保留A8。\n')
    add('| 源数据集／探针 | 拟合／验证源 | 维度 | 选定alpha | 验证top1 tIoU | 同面板native | 同面板32候选oracle | 验证MSE |\n|---|---:|---:|---:|---:|---:|---:|---:|')
    for ds in ['vidstg','hc2']:
        for signal in ['L','G']:
            r=fit[ds][signal];z=r['path'][r['selected_index']]
            add(f'| {ds}/{signal} | {r["training_queries"]}/{r["validation_queries"]} | {r["dimensions"]} | {r["selected_alpha"]:g} | {100*z["validation_top1_tIoU"]:.2f} | {100*r["validation_native"]:.2f} | {100*r["validation_oracle"]:.2f} | {z["validation_MSE"]:.5f} |')
    add('\n源验证L相对native为Vid+4.13pp、HC+2.44pp，均为**参与alpha选择的描述性结果**，不能包装成独立统计改善；尚余约20.03/16.81pp候选oracle缺口。源GT训练并没有自动提供可靠目标读出。\n')
    add('![源验证路径](../results/tastvg_temporal_latent_quality/2026-10-03/figures/source_validation.png)\n')
    add('## 固定支持上的最终预测\n')
    add('增量单位pp，95%区间为10000次同来源配对bootstrap。先到达内、条件、顺序，再来源宏平均；HC专家子集在两个顺序的来源覆盖不均，source-macro不能简单等于全流四倍。\n')
    add('| 面板 | A8全corrupt vIoU | L8−A8 | L32−A8 | L32 tIoU增量 | G32−A8 |\n|---|---:|---:|---:|---:|---:|')
    summaries={}
    for sp in ['search','confirm']:
        for ds in ['vidstg','hc2']:
            s=read(P/sp/ds/'SUMMARY.json');summaries[(sp,ds)]=s;m=s['corruption']['all']['metrics']
            add(f'| {ds}/{sp} | {val(m["A8_v"])} | {ci(m["L8_gain"])} | {ci(m["L32_gain"])} | {ci(m["L32_t_gain"])} | {ci(m["G32_gain"])} |')
    add('\n| corrupt专家面板 | A8 vIoU | L32 vIoU | L32 tIoU | O32 vIoU | L32 regret | gross gain / loss | 改善／受损／>5pp损害 |\n|---|---:|---:|---:|---:|---:|---:|---:|')
    for (sp,ds),s in summaries.items():
        z=s['corruption']['expert'];m=z['metrics'];tr=z['transitions']['L32'];old=read(OLD/sp/ds/'SUMMARY.json')['corruption']['expert']['metrics']
        oracle=m['L32_v']['mean']+m['L32_regret']['mean']
        add(f'| {ds}/{sp} | {val(m["A8_v"])} | {val(m["L32_v"])} | {val(m["L32_t"])} | {100*oracle:.2f} | {val(m["L32_regret"])} | {val(m["L32_gross_gain"])}/{val(m["L32_gross_loss"])} | {tr["improved"]}/{tr["harmed"]}/{tr["severe_gt5pp"]} |')
    add('\n| 完整corruption流的配对差距 | L32−N32 | L32−U32 | L32−S32 | L32−G32 |\n|---|---:|---:|---:|---:|')
    for (sp,ds),s in summaries.items():
        m=s['corruption']['all']['metrics'];add(f'| {ds}/{sp} | {ci(m["L32_vs_N32_v"])} | {ci(m["L32_vs_U32_v"])} | {ci(m["L32_vs_S32_v"])} | {ci(m["L32_vs_G32_v"])} |')
    add('\n确认L32没有稳定超过N32；相对U/S正均值也没有正区间的共同证据。“比一个受损的baseline好”不能代替相对A的实际改善。Vid确认相对G32为+1.86pp区间高于零，表明这版latent读出不等同于这版纯位置/时长规则；它仍几乎没有净收益，也不能建立“信息被标量压缩丢失”这一原因。\n')
    add('![目标确认读出](../results/tastvg_temporal_latent_quality/2026-10-03/figures/target_confirmation.png)\n')
    add('## 大幅修改：判别与实际许可分开\n')
    add('沿用端点L1/A8时长≥.5及单端balance≤.25操作分组。候选有益/有害以固定A框的vIoU相对同A8差值定义；零差值排除。precision/recall以同来源抽样的分子/分母比值汇总，非池化TP/(TP+FP)；AUC仅在到达内同时有正负候选时定义，缺类不补.5。\n')
    add('| 面板／L | large条件AUC | 有效到达／来源 | beneficial precision | beneficial recall | harmful acceptance | raw TP/FP/FN/TN |\n|---|---:|---:|---:|---:|---:|---:|')
    for (sp,ds),s in summaries.items():
        z=read(P/sp/ds/'DISCRIMINATION.json')['corruption']['large']['L'];a=z['auc'];counts=z['raw_counts'];au=a['metrics'].get('auc')
        astr='未定义' if au is None else f'{au["mean"]:.3f} [{au["ci95"][0]:.3f}, {au["ci95"][1]:.3f}]'
        add(f'| {ds}/{sp} | {astr} | {z["eligible_auc_cells"]}/{a["sources"]} | {ci(z["ratios"]["precision"])} | {ci(z["ratios"]["benefit_recall"])} | {ci(z["ratios"]["harm_acceptance"])} | {counts["TP"]}/{counts["FP"]}/{counts["FN"]}/{counts["TN"]} |')
    add('\nVid确认L的large precision为43.71%，高于旧U/S的描述性约10%，但置信区间宽、263/10000次抽样无接受分母；raw仍接受12有益/15有害候选。最终top1采用4次大改，4次均有益，只来自3个source、均order1；不能外推安全性，且另外13次小幅修改受损。HC条件AUC=.877，但任何大幅候选都未获相对A8的正分许可：TP=FP=0、precision未定义、recall=0。高条件AUC不能证明可用的top1决策。\n')
    add('| 确认corrupt／L32 | 新增24候选被选中 | 大改改善／损害 | 小改改善／损害 | 原Fast正收益被破坏 | vIoU .3正确→错误／救回 | .5正确→错误／救回 |\n|---|---:|---:|---:|---:|---:|---:|')
    for ds in ['vidstg','hc2']:
        c=extra[f'confirm/{ds}/corruption']['L32']['counts']
        add(f'| {ds} | {c["added_candidate_selected"]}/40 | {c["large_improved"]}/{c["large_harmed"]} | {c["small_improved"]}/{c["small_harmed"]} | {c["old_fast_gain_destroyed"]} | {c["v0.3_correct_destroyed"]}/{c["v0.3_correct_rescued"]} | {c["v0.5_correct_destroyed"]}/{c["v0.5_correct_rescued"]} |')
    add('\nHC在全部144专家到达上L8和L32选择相同，确认没有选择新增候选，也没有任何大幅top1；确认9改善、25损害中13次损害>5pp，保守并不等于安全。Vid确认专家两序L32增量order1 +1.083pp、order2 −.987pp，leave-one-source-out −.462至+1.295pp；HC为−.033/−5.958pp，leave-one-source-out −3.759至−1.919pp。顺序专家来源覆盖不同，这些数值不单独证明参数漂移。\n')
    add('所有large trim-start/end/expand/shift/trim-both、clean、G对照、raw计数、缺类/零分母和same-support pairwise诊断已发布。PAIRWISE_DIAGNOSTICS是分数封存后的描述分析，不参与模型选择；只在相同支持内比较不同signal，不能用Old8与32不同pair population推断极值校准机制。\n')
    add('## Clean和代表性正负例\n')
    add('| 面板 | clean完整流L8−A8 | clean完整流L32−A8 | clean完整流G32−A8 |\n|---|---:|---:|---:|')
    for (sp,ds),s in summaries.items():
        m=s['clean']['all']['metrics'];add(f'| {ds}/{sp} | {ci(m["L8_gain"])} | {ci(m["L32_gain"])} | {ci(m["G32_gain"])} |')
    add('\n| 确认／L32代表例 | A8→L32 vIoU | 增量pp |\n|---|---:|---:|')
    for ds in ['vidstg','hc2']:
        c=read(P/'confirm'/ds/'CASES.json')['L32']
        for direction in ['positive','negative']:
            r=c[direction][0];add(f'| {ds}/{direction}/source{r["source_id"]}/{r["condition"]}/{r["order"]} | {100*r["A8_v"]:.2f}→{100*r["L32_v"]:.2f} | {100*r["L32_gain"]:+.2f} |')
    add('\n## 新测得的接口一致性与核验\n')
    add('本轮实际重放288个A更新前状态，空间输出与封存A框逐值相同；时间两offset的全部start/end logits与冻结source capture逐值相同，提取hidden逐值重算head也一致。**在当前A的空间更新范围和这个面板上，可以排除“旧N使用source先验而不是A-current logits”作为失败原因。** 这不推广到别的更新范围或时间参数适应，也不把边缘配对变成完整原生joint-span likelihood。\n')
    add(f'独立根审计从原hidden重新拼接27,410,432个特征数值、45,888个几何值及上下文计数；190源官方训练label join、source/media互斥、训练标准化、直接线性方程解与全部源验证路径、18,432目标分数、1152决策及1152目标指标绑定均通过。直接ridge解最大差{root["maximum_errors"]["direct_ridge_coefficients"]:.3g}。公开审计独立复算{audit["checks"]["numeric_checks"]}标量及1152选择，最大差{audit["max_numeric_error"]:.3g}；只用匿名导出可核验选择/标签/汇总，不声称公开副本能重建被排除的原latent和probe权重。五项CPU数学测试通过，两图已目检。\n')
    add('保留源摄取两个工程失败（相同caption跨used-segment、错误截取来源ID末11字符），均发生在模型推理前。补充CPU导出首次NumPy int64不能JSON序列化也保留原件，修复仅为Python标量转换；正式分数/候选/指标未改。\n')
    # Clarify scope of the original source counter without rewriting its barrier.
    calls=dict(source_queries=190,source_full_model_offset_forwards=380,
        source_extra_cached_suffix_offset_calls=760,target_expert_arrivals=288,target_cached_suffix_offset_calls=576,
        target_new_backbone_forwards=0,target_new_experts=0,parameter_update_backwards=0,
        direct_head_identity_calls=956,scope='Derived from completed immutable query counts and executed call graph. Each suffix offset call includes native two-stage decoding.',
        original_source_barrier_suffix_counter_scope='native_suffix_calls counted the observer offsets only; source_capture also called combined once per query. Original barriers are preserved.')
    write(P/'CALL_ACCOUNTING.json',calls)
    source_wall=sum(z['SOURCE_FEATURE_BARRIER.json']['worker_wall_seconds'] for z in res['stages'].values())
    target_wall=sum(z['TARGET_FEATURE_BARRIER.json']['worker_wall_seconds'] for z in res['stages'].values())
    fitseal=read(B/'FIT_BARRIER.json')
    add(f'源capture GPU路径worker wall累计{source_wall:.2f}秒，目标缓存重放worker wall累计{target_wall:.2f}秒，CPU拟合{fitseal["CPU_wall_seconds"]:.2f}秒；含模型加载/解码/CPU处理，**不是纯GPU kernel时间**。源完整模型380个offset forward，源额外cached suffix760个offset调用、目标576个offset suffix调用，head直接一致性调用956次；每个offset suffix含两stage decoder。原源barrier的suffix计数仅覆盖observer而未计source_capture内部combined，完整调用口径更正在CALL_ACCOUNTING，原barrier不重写。目标零backbone/新专家/反向更新。峰值显存Vid源约20.61GiB、HC源4.80GiB，目标约1GiB。保存源/目标特征{res["source_feature_bytes"]:,}/{res["target_feature_bytes"]:,}bytes。\n')
    add('## 决策\n')
    add('保留A；L8/L32本轮均不接入正式时间分支。有限source监督线性probe在源验证读出了一定质量信号，Vid确认也有具体成功大改，但当前target transfer和top1行为不足以兑现oracle机会。**失败只限定这组表示、源预算、linear ridge、tIoU目标和冻结读出；不证明所有native latent没有信息，也不证明非线性probe必能成功。** 本轮没有测试MLP/listwise/latent TTA/额外expert，不把“标量压缩”当成测得原因。当前约束不按确认GT改threshold、window、alpha或模型。\n')
    add('[执行协议](../protocols/tastvg_temporal_latent_quality_v1.md) · [完整匿名结果](../results/tastvg_temporal_latent_quality/2026-10-03)\n')
    (ROOT/'docs/TA_TEMPORAL_LATENT_QUALITY_REVIEW.md').write_text('\n'.join(text))
    write(P/'DECISION.json',dict(status='completed_valid_no_promotion',retain='A',latent_linear_probe_promoted=False,
        scope='Source-supervised finite linear accessibility/transfer audit, not all latent information or MLP.',
        observed={ds+'_confirmation_full_corrupt_gain':100*summaries[('confirm',ds)]['corruption']['all']['metrics']['L32_gain']['mean'] for ds in ['vidstg','hc2']},
        unit='vIoU percentage points',target_GT_used_for_selection=False,production_changed=False,new_experts=0,
        next_experiments_started=False,scalar_compression_causality_established=False))
    write(P/'ENGINEERING_RECOVERY.json',dict(preserved=['source_intake_001','source_intake_002','supplement_json_003'],
        inference_failed=False,source_intake_failed_before_any_model_calls=True,
        source_identifier_join='full source/caption/used-segment for Vid; exact original source strings for HC',
        supplementary_export_fix='Convert numpy counter values to Python ints; no score or label changes',
        resource_accounting_clarification='CALL_ACCOUNTING includes source_capture internal cached suffix; original barrier counter covered observer only.'))
    code=['scripts/run_tastvg_temporal_latent_quality_v1.py','scripts/tastvg_latent_quality_math_v1.py',
        'vg_tta/tastvg_temporal_latent_quality_v1.py','scripts/test_tastvg_latent_quality_v1.py',
        'scripts/continue_tastvg_temporal_latent_quality_v1.py','scripts/audit_tastvg_latent_quality_root_v1.py',
        'scripts/audit_tastvg_latent_quality_public_v1.py','scripts/draw_tastvg_latent_quality_v1.py',
        'scripts/summarize_tastvg_latent_quality_v1.py','scripts/report_tastvg_latent_quality_v1.py',
        'protocols/tastvg_temporal_latent_quality_v1.md','docs/tastvg_temporal_latent_quality_v1/EXECUTION.md']
    write(P/'CODE_PINS.json',{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in code})
    print('REPORT_WRITTEN',ROOT/'docs/TA_TEMPORAL_LATENT_QUALITY_REVIEW.md')

if __name__=='__main__':report()
