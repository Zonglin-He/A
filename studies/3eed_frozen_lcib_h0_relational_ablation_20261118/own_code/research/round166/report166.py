"""Report fixed-denominator task and role results for the five frozen h0 arms."""
from scope166 import ARMS,HERE,ORDER,atomic_json,read
from score166 import METRICS,OUT,table

LABEL={
    'r158_pca':'R158＋PCA','h0_full':'完整 LCIB h₀','h0_independent':'去关系与重叠惩罚',
    'h0_no_ref_role':'保留关系，去参照端专属角色证据','h0_original_only':'只用原 A256 假设',
    'r158_r88_reference':'R158＋R88 参照读出（诊断）',
}
TITLES={'all':'全部记录','parsed_direct':'可解析的主体—参照关系',
        'camera_horizontal':'LCIB 支持的相机左右关系','active_factor':'实际有非零联合代价'}
STAGES={'a256':'原 A256 的种子支持','r88':'R88 的三项短名单','h0_pool':'LCIB 初始池','h0_eligible':'LCIB 支持资格','h0_beam':'进入 LCIB 联合比较'}


def vec(r,metrics=METRICS[:4],diff=False):
    return '／'.join(f'{float(r[k+"_pp"]):+.3f}' if diff else f'{float(r[k]):.3f}' for k in metrics)


def classify(contrasts,idn):
    against_baseline=contrasts['six_equal','all','r158_pca']
    overall=contrasts['six_equal','all','h0_independent']
    relevant=contrasts['six_equal','camera_horizontal','h0_independent']
    increments=[float(r[k]) for r in (overall,relevant) for k in ('Acc25_pp','Acc50_pp')]
    both_more=int(idn['all','all_supported','h0_full']['both_correct'])-int(idn['all','all_supported','h0_independent']['both_correct'])
    better_than_base=min(float(against_baseline[k]) for k in ('Acc25_pp','Acc50_pp'))>0
    if better_than_base and min(increments)>0 and both_more>0:
        return ('联合因子在本轮整体与关系子集均有正增量，双端标注归属也有净改善。',
                '本轮支持保留冻结 h₀ 的联合读取，贡献限定于已评场景及可核身份的分母。')
    if better_than_base:
        return ('冻结 h₀ 超过 R158＋PCA，但成对关系贡献未同时通过全量、关系子集和双端归属。',
                '完整版本的任务收益可单列。联合推理的独立贡献需要使用同池去关系结果界定。')
    return ('冻结 h₀ 未在两项定位阈值同时超过 R158＋PCA，现有证据不足以把联合推理作为主要性能贡献。',
            '候选覆盖与已覆盖仍选错的分母分别用于定位检索和分配问题。')


def main():
    audit=read(HERE/'EVALUATION_AUDIT.json')
    photo=read(HERE/'VISUAL_REVIEW_AUDIT.json')
    photo_coverage=read(HERE/'VISUAL_COVERAGE_AUDIT.json')
    metrics={(r['group'],r['subset'],r['arm']):r for r in table(OUT/'metrics.csv')}
    pairs={(r['group'],r['subset'],r['control']):r for r in table(OUT/'paired.csv')}
    ids={(r['group'],r['truth'],r['arm']):r for r in table(OUT/'identity.csv')}
    cover={(r['group'],r['truth'],r['stage']):r for r in table(OUT/'coverage.csv')}
    finding,bound=classify(pairs,ids)
    primary=f"{float(pairs['six_equal','all','r158_pca']['subject_identity_pp']):+.3f}"
    sensitivity=f"{float(pairs['six_equal','all','r158_pca']['subject_box_proxy_pp']):+.3f}"
    n=audit['n']
    truth=audit['edge_truth_counts']
    n_gold=int(ids['all','all_supported','h0_full']['n'])
    old_gold=truth.get('visual_verified',0)
    text=[
        '# R166 · 冻结 LCIB h₀ 的联合关系消融','',f'**{finding}** {bound}','',
        '所有结果使用 R165 的同一批 18,620 条记录和六个单源跨平台方向。五臂共用冻结模型、同步 RGB、描述、点云采样、A256 和最终框的 PCA 规则。',
        '完整臂逐步复现 R162 初始化，候选位置、尺寸和点权重从初始化至最终选择保持固定。48 条盲于标签的预检复现了原始实现的候选选择。新空间假设的初始中心和尺寸也固定。',
        '预测和候选池在任何本轮任务标签读取前封存。已有历史曝光，六方向用同一套评测数据。',
        '[锁定协议](experiment/PROTOCOL.json) · [预测封存](PREDICTION_SEAL.json)','',
        '## 全量及关系样本定位','',
        '每格依次为 **mIoU／Acc@25／Acc@50／主体身份正确率（%）**，在有记录的方向上等权平均。',
        '主体身份由所选框对应的种子点在标注实例中唯一归属确定。另有框匹配归属供敏感性核对。',
        '','| 方法 | 全部 | 可解析的主体—参照关系 | 相机左右关系 | 非零联合代价 |',
        '|---|---:|---:|---:|---:|',
    ]
    scopes=('all','parsed_direct','camera_horizontal','active_factor')
    for a in ARMS:
        text.append('| '+' | '.join([LABEL[a],*(vec(metrics['six_equal',s,a]) for s in scopes)])+' |')
    text+=['','样本数依次为 '+ '／'.join(f'{int(metrics["six_equal",s,"h0_full"]["n"]):,}' for s in scopes)+'。',
        '关系语法子集保留了解析器抽出的直接左右边。LCIB 执行其中可绑定相机坐标的边，具备可比较候选且产生非零代价的记录进入最后一列。',
           '', '| 完整 h₀ 减对照 | 全量变化（pp） | 相机左右关系变化（pp） | Acc@50 全量改善／损伤 |',
           '|---|---:|---:|---:|']
    for a in ('r158_pca','h0_independent','h0_no_ref_role','h0_original_only'):
        p,q=[pairs['six_equal',s,a] for s in ('all','camera_horizontal')]
        text.append(f'| {LABEL[a]} | {vec(p,diff=True)} | {vec(q,diff=True)} | {p["Acc50_up"]}／{p["Acc50_down"]} |')
    text+=['',
        '去关系臂共用完整的候选池、单元分数和比较预算。去参照角色臂固定由原检索得到的空间池，把参照端专属注意力及候选权重替换成共享全句证据。原 A256 臂用旧候选补充新假设的名额。',
        f'完整配置选中新空间主体 {audit["arm_budgets"]["new_subject_selected"]:,} 条。原候选臂在 {audit["arm_budgets"]["any_role_deficit"]:,} 条里有至少一个角色缺少足够互异候选，仍计入共同总分母。',
        f'完整配置对基线的主体归属变化为 {primary} pp。按标注框的独立保守归属复算后为 {sensitivity} pp，归属定义会改变增量方向。',
        '[各平台、各方向及总体指标](results/metrics.csv) · [配对收益与损伤](results/paired.csv)','',
        '## 主体和参照物身份','',
        f'相机左右关系共 {sum(truth.values()):,} 条方向—关系边。其中 {n_gold:,} 条有可用参照身份，'
        f'{truth.get("annotation_proxy",0):,} 条来自唯一类别及左右投影一致的标注代理，{old_gold:,} 条来自已有原图核验。',
        '已知身份的边使用同一个分母；输出 UNKNOWN 计入分母。R158 原生只输出主体，补充的 R88 参照读出与 R158 主体组成独立诊断。',
        '', '| 方法 | 主体正确 | 参照正确 | 双端正确（共同标注分母） | 既有原图核验双端 |',
        '|---|---:|---:|---:|---:|']
    def count(r,role):
        return f'{r[role+"_correct"]}／{r[role+"_n"]}' if r and int(r[role+'_n']) else '—'
    for a in (*ARMS,'r158_r88_reference'):
        r=ids['all','all_supported',a]
        v=ids.get(('all','visual_verified',a))
        text.append(f'| {LABEL[a]} | {count(r,"subject")} | {count(r,"reference")} | {count(r,"both")} | {count(v,"both")} |')
    x=photo['verified_direction_edges']
    text+=['',f'额外遮蔽模型输出后由一名读者复核了 {photo["review_cases"]} 个固定输入。'
           f'确认带独立框的方向—边 {photo["verified_direction_edges"]} 条，'
           f'在 {photo["proxy_eligible"]} 条自动代理覆盖的已核验方向记录中匹配了 {photo["proxy_matches"]} 条。',
           f'另有 {photo["counts"].get("annotated_box_wrong_instance",0)} 个代理选中的物理实例在原图被证实选错，真实参照没有独立框。'
           f'新核验分母下，完整 h₀ 双端正确 {photo["results"]["h0_full"]["correct"]}／{x}，'
           f'去关系双端正确 {photo["results"]["h0_independent"]["correct"]}／{x}。额外核验与全量代理平行报告。',
           '[身份表与未知数](results/identity.csv) · [盲审结果](results/visual_review.csv)','',
           '## 正确候选在哪一步失去','',
           '| 位置 | 覆盖主体 | 覆盖参照 | 双端可选 | 双端可选但完整 h₀ 未选中 |',
           '|---|---:|---:|---:|---:|']
    for step,label in STAGES.items():
        r=cover['all','all_supported',step]
        text.append(f'| {label} | {r["subject"]} | {r["reference"]} | {r["both"]}／{r["n"]} | {r["covered_but_failed_both"]} |')
    more=photo_coverage['coverage']
    old_vision=cover.get(('all','visual_verified','a256'))
    earlier=cover.get(('all','visual_verified','r88'))
    beam=cover['all','all_supported','h0_beam']
    text+=['',f'应优先改 R88 的参照物检索。新核验集从 A256 的 {more["a256"]["both"]}／{x} 双端覆盖降到 R88 的 {more["r88"]["both"]}／{x}；'
           f'既有核验集为 {old_vision["both"]}／{old_vision["n"]} → {earlier["both"]}／{earlier["n"]}。'
           f'进入联合比较且两端皆可选的 {beam["both"]} 条中仍有 {beam["covered_but_failed_both"]} 条发生选择失败，后续可用同一固定候选池量化分配损失。',
           '覆盖按区域对标注实例的唯一种子归属度量。LCIB 初始池、支持资格和联合比较共用同一候选框口径。',
           '[逐方向覆盖](results/coverage.csv) · [逐边证据及未知](results/relation_edges) · [逐表达任务分数](results/all_rows)','',
           '## 验证与范围','',
           f'用独立传感器坐标凸包审计了 {audit["geometry_endpoints"]:,} 个主体框，最大绝对 IoU 差 '
           f'{audit["max_absolute_sensor_iou_error"]:.3g}，两项阈值判定一致。R158 基线与 R165 框逐项一致。',
           '主体标签来自数据集原注释的目标对象。在缺少参照实例标注或无法唯一指认时，保留关系边及其不可判定状态。'
           '双端指标受标注覆盖和复核范围约束。Waymo、Drone、Quad 的可泛化性仍受已有录制构成约束。',
           '[几何与分母审计](EVALUATION_AUDIT.json) · [分场景指标](results/recordings.csv)','']
    (HERE/'REPORT_ZH.md').write_text('\n'.join(text),encoding='utf-8')
    (HERE/'REPORT.md').write_text('# Frozen LCIB h0 relational ablation on the matched 3EED population\n\n'
        '[Full data, identity scopes and measured decisions](REPORT_ZH.md). '
        'Replaying h0 requires the data and existing source weights, plus the frozen R162/R165 implementation and R97 evidence inputs. '
        'The anonymized tables include every contrast and unresolved case.\n',encoding='utf-8')
    atomic_json(HERE/'DECISION.json',dict(finding=finding,decision_scope=bound,n=n,
        overall={a:{k:float(metrics['six_equal','all',a][k]) for k in METRICS} for a in ARMS},
        matched_control_pp={a:{k:float(pairs['six_equal','all',a][k+'_pp']) for k in METRICS}
                            for a in ARMS if a!='h0_full'},referenced_edges_labeled=n_gold,
        candidate_priority='R88 role-conditioned reference retrieval',
        allocation_failures_given_both_in_beam=int(beam['covered_but_failed_both']),
        joint_inference_primary_paper_claim_supported=False,
        current_stage_frozen_h0=True,source_training_runs=0,target_parameter_updates=0,FINAL_access=False))
    print('R166 written',finding,flush=True)


if __name__=='__main__':
    main()
