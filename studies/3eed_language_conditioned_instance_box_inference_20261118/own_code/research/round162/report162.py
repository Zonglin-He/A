"""Render the locked LCIB experiment from verified numeric artifacts."""
import numpy as np

from manifest162 import DEV, EXTRA, REC, HERE, ORDER, atomic_json, read, sha
from score162 import put_table, table

ACTIVE = {
    'official_span': '原版 3EED',
    'prepared_axis': '冻结 R76＋R91，原框',
    'prepared_pca': '冻结 R76＋R91，PCA',
    'r158_full': 'R158＋PCA',
    'pca_obb': '固定赢家＋PCA 分位框',
    'uniform_fixed': '固定赢家＋均匀裁点，固定尺寸',
    'uniform_fit': '固定赢家＋均匀裁点，连续拟框',
    'source_selected_simple': '源域选择的简单拟框',
    'fixed_winner_support': '固定赢家＋实例支持，连续拟框',
    'serial': '同预算串行支持＋拟框',
    'lcib': 'LCIB 交替推断',
}
LABELS = {DEV:'开发范围 1,798 条', EXTRA:'普通范围 5,942 条', REC:'补充采集范围 3,072 条'}


def triple(row, digits=3):
    return ' / '.join(f'{float(row[m]):.{digits}f}' for m in ('mIoU','Acc25','Acc50'))


def delta(row, digits=3):
    return ' / '.join(f'{float(row[m]):+.{digits}f}' for m in ('mIoU_pp','Acc25_pp','Acc50_pp'))


def scope_data(folder):
    if not (folder/'AUDIT.json').exists():
        return {}
    audit=read(folder/'AUDIT.json')
    assert audit['status']=='verified' and sha(folder/'rows.csv')==audit['result_rows_sha256']
    return {t:table(folder/(t+'.csv')) for t in ('overall','directions','paired','oracle','rows','paired_scene_ci')}


def by_arm(data, cohort):
    return {r['arm']:r for r in data['overall'] if r['cohort']==cohort}


def pair(data,cohort,before,after):
    return next(r for r in data['paired'] if r['cohort']==cohort and r['source']=='six_equal'
                and r['before']==before and r['after']==after)


def combined_old_pool(data, cohort):
    outcomes=[]
    for s,t in ORDER:
        selected=[r for r in data['rows'] if (r['cohort'],r['source'],r['target'])==(cohort,s,t)]
        for threshold in (.25,.50):
            missing=[r for r in selected if max(float(r['oracle_axis']), float(r['oracle_pca']))<=threshold]
            cover=[r for r in missing if float(r['oracle_lcib'])>threshold]
            actual=[r for r in missing if float(r['lcib'])>threshold]
            outcomes.append(dict(cohort=cohort,source=s,target=t,n=len(selected),threshold=threshold,
                        old_axis_and_PCA_both_missed=len(missing),generated_covers=len(cover),
                        final_rescues=len(actual),unselected=len(cover)-len(actual),
                        final_rescues_from_scene=sum(int(r['selected_origin'])==-1 for r in actual)))
    return outcomes


def main():
    dev=scope_data(HERE/'results/DEV')
    evaluation=scope_data(HERE/'results/EVAL')
    measured = evaluation or dev
    main_cohort = EXTRA if evaluation else DEV
    if not measured:
        return
    values=by_arm(measured,main_cohort)
    info=values['lcib']
    incumbent=values['r158_full']
    matched=pair(measured,main_cohort,'serial','lcib')
    prev=pair(measured,main_cohort,'r158_full','lcib')
    geometry=pair(measured,main_cohort,'source_selected_simple','lcib')
    win=lambda p: min(float(p['Acc25_pp']),float(p['Acc50_pp']))>0
    primary=win(prev)
    joint=win(matched)
    better_geo=win(geometry)
    out_rows=[r for r in measured['oracle'] if r['cohort']==main_cohort and r['source']=='six_equal']
    oracle=out_rows[0]
    support_path=HERE/'results/DEV/support_curves.csv'
    points={}
    if support_path.exists():
        points={r['arm']:r for r in table(support_path) if r['role']=='subject' and r['quota']=='32'}
    graph_ok=False
    if points and float(points['lcib']['matched_roles'])>0:
        full=float(points['lcib']['gt_box_fraction'])
        graph_ok=all(full > float(points[r]['gt_box_fraction']) for r in ('geometry_component','swap_cross','no_rgb_graph'))
    stricter=combined_old_pool(measured,main_cohort)
    strict_rescues={int(h*100):sum(r['final_rescues'] for r in stricter if r['threshold']==h) for h in (.25,.50)}
    out_of_pool=sum(strict_rescues.values())
    accept=bool(evaluation and primary and joint and better_geo and graph_ok and out_of_pool>0)
    decision=dict(adopted=accept, formal_ranges_complete=bool(evaluation), evaluation_reference=main_cohort,
                  both_accuracy_thresholds_beat_R158=primary, both_beat_same_budget_serial=joint,
                  both_beat_simple_fitting=better_geo, matched_box_support_advantage=graph_ok,
                  effective_beyond_old_PCA_pool={25:int(oracle['beyond_pca_25']),50:int(oracle['beyond_pca_50'])},
                  effective_beyond_both_old_pools=strict_rescues,
                  direction_macro_delta_vs_R158={k:float(prev[k+'_pp']) for k in ('mIoU','Acc25','Acc50')},
                  FINAL_access=False,staged_inference_without_network_updates=True)
    atomic_json(HERE/'DECISION.json',decision)
    disposition = '支持继续验证 LCIB' if accept else '当前配方未达到采纳条件'
    out=['# R162　语言条件的实例支持与框联合推断','',
         f'**{disposition}。** {LABELS[main_cohort]}上，LCIB 为 **{triple(info)}**，R158＋PCA 为 **{triple(incumbent)}**。'
         f'与旧流程的配对差值为 **{delta(prev)} 个百分点**，与同预算串行的配对差值为 **{delta(matched)} 个百分点**。'
         '顺序均为 mIoU / Acc@25 / Acc@50。','']
    for data,cohort,tag in ((dev,DEV,'DEV'),(evaluation,EXTRA,'EVAL'),(evaluation,REC,'EVAL')):
        if not data:
            continue
        v=by_arm(data,cohort)
        out.extend(['## '+LABELS[cohort],'',
             '准确率使用有向 3D IoU 严格大于 0.25 或 0.50。每列均为六方向等权百分数。','',
             '| 方法 | mIoU | Acc@25 | Acc@50 |','|---|---:|---:|---:|'])
        for arm,name in ACTIVE.items():
            out.append('| '+name+' | '+' | '.join(triple(v[arm]).split(' / '))+' |')
        out.extend(['','| 源 → 目标 | 条数 | R158＋PCA | 串行 | LCIB | LCIB 减 R158 |',
                    '|---|---:|---:|---:|---:|---:|'])
        directions={(r['source'],r['target'],r['arm']):r for r in data['directions'] if r['cohort']==cohort}
        changes={(p['source'],p['target']):p for p in data['paired']
                 if p['cohort']==cohort and (p['before'],p['after'])==('r158_full','lcib')}
        for s,t in ORDER:
            old=directions[s,t,'r158_full']
            out.append(f'| {s.title()} → {t.title()} | {old["n"]} | {triple(old)} | '
                        f'{triple(directions[s,t,"serial"])} | {triple(directions[s,t,"lcib"])} | {delta(changes[s,t])} |')
        out.extend(['',f'[逐方向的全部方法](results/{tag}/directions.csv) · [配对改善、恶化与差异](results/{tag}/paired.csv) · '
                       f'[按目标采集序列配对区间](results/{tag}/paired_scene_ci.csv) · [独立审计](results/{tag}/AUDIT.json)',''])
    if dev:
        put_table(HERE/'results/DEV/combined_old_pool.csv',combined_old_pool(dev,DEV))
    if evaluation:
        put_table(HERE/'results/EVAL/combined_old_pool.csv',
                  combined_old_pool(evaluation,EXTRA)+combined_old_pool(evaluation,REC))
    out.extend(['## 支持、框池与联合更新','',
                '| 上界或最终输出 | Acc@25 | Acc@50 |',
                '|---|---:|---:|'])
    for name,key in [('旧 A256 原始轴向','oracle_axis'),('旧 A256 全框采用 R155 PCA','oracle_pca'),
                     ('LCIB 生成框集合','oracle_lcib'),('旧 PCA256 与新集合的并集','oracle_union')]:
        out.append(f'| {name} | {float(oracle[key+"_Acc25"]):.3f} | {float(oracle[key+"_Acc50"]):.3f} |')
    out.extend(['',f'在旧 PCA256 没有达标框的样本上，新集合分别新增 **{oracle["generated_cover_25"]} / {oracle["generated_cover_50"]}** 条可达阈值的记录。'
                    f'实际选中达标新框的有 **{oracle["beyond_pca_25"]} / {oracle["beyond_pca_50"]}** 条，'
                    f'其中从全场峰初始化的有 **{oracle["beyond_pca_scene_25"]} / {oracle["beyond_pca_scene_50"]}** 条。'
                    '数字顺序为 Acc@25 / Acc@50。上界均在预测封存后由目标框计算。','',
                    f'要求轴向和 PCA 两套旧框池都缺少达标框时，最终有效新增数为 '
                    f'**{strict_rescues[25]} / {strict_rescues[50]}**。'
                    f'[两套旧框池的联合覆盖诊断](results/{"EVAL" if evaluation else "DEV"}/combined_old_pool.csv)',''])
    if points:
        out.extend(['### 匹配支持点数的主体诊断','',
                    '| 支持 | 共同可取 32 点 | 32 点中的标注框内比例 | GT 框内采样点覆盖 |',
                    '|---|---:|---:|---:|'])
        for arm,r in points.items():
            if r['gt_box_fraction']:
                out.append(f'| {arm} | {r["matched_roles"]} / {r["panel_roles"]} | '
                           f'{100*float(r["gt_box_fraction"]):.2f}% | {100*float(r["gt_box_point_recall"]):.2f}% |')
        out.extend(['','这些值使用人工标注框的包含关系。逐点实例掩码缺失，主体和参照物的身份代理保留无法唯一归属的记录。'
                    '参考物仅在独立标注列表、参考类别唯一性和相机左右关系同时通过时计作可核验。'
                    '[支持曲线](results/DEV/support_curves.csv) · [身份代理计数](results/DEV/identity_proxy.csv) · '
                    '[anchor 对应](results/DEV/anchor_audit.csv) · [RGB、初始化、关系和遮挡对照](results/DEV/panel_paired.csv)',''])
    mechan=HERE/'results/DEV/MECHANISMS.json'
    if mechan.exists():
        identity=read(mechan)
        out.extend([
            f'192 条诊断中有 **{identity["anchor_outside"]}** 条锚点位于原预测框外。'
            f'锚点与框对应对象都能在独立标注列表中唯一归属的 **{identity["anchor_box_assignment_known"]}** 条里，'
            f'只有 **{identity["anchor_box_assignment_differs"]}** 条落在不同标注对象。'
            f'参照物只有 {identity["relations_with_labelled_reference"]} 次有可核查的独立标签。'
            '本轮证据不支持把锚点系统性错绑作为主要机制。[固定抽样读图记录](VISUAL_REVIEW.md)',''])
    shape_path=HERE/'results/DEV/same_annotated_target_geometry.csv'
    if shape_path.exists():
        pair_geometry(out, shape_path)
    effects=HERE/'results/DEV/panel_paired.csv'
    if effects.exists():
        contrast={(r['before'],r['after']):r for r in table(effects) if r['source']=='six_equal'}
        out.extend(['### 192 条固定诊断的控制差值','',
                    '| 差值，后者减前者 | Δ mIoU | Δ Acc@25 | Δ Acc@50 |',
                    '|---|---:|---:|---:|'])
        for label,keys in [
            ('完整 LCIB 减新增 RGB 图关闭',('no_rgb_graph','lcib')),
            ('完整 LCIB 减场景内错配对应',('swap_cross','lcib')),
            ('完整 LCIB 减全系统无 RGB',('system_no_rgb','lcib')),
            ('完整 LCIB 减错误帧 RGB',('wrong_frame_rgb','lcib')),
            ('去显式关系因子减完整 LCIB',('lcib','no_relation')),
            ('去框对支持反馈减完整 LCIB',('lcib','no_feedback')),
            ('允许旧起点支持外扩减框内限制',('bounded_origins','expanded_origins')),
            ('完整全场初始化减仅旧起点外扩',('expanded_origins','lcib')),
            ('完整尺寸更新减固定尺寸',('fixed_dimensions','lcib')),
            ('弱尺寸先验减仅可见表面拟合',('visible_surface_only','lcib')),
            ('缺失保留未知减缺失视作负证据',('missing_as_negative','lcib'))]:
            out.append('| '+label+' | '+' | '.join(delta(contrast[keys]).split(' / '))+' |')
        out.extend(['','诊断清单按原方法成功与失败配额抽取，以上幅度服务于同输入对照。'
                         '全量指标使用各自原始分母。实际错配成立的范围、自然稀疏和可见性分层单列。'
                         '[诊断配对明细](results/DEV/panel_paired.csv) · [逐轮目标、支持变化与 IoU](results/DEV/panel_trace.csv) · '
                         '[可见性与稀疏度分层](results/DEV/visibility_strata.csv)',''])
    counter=HERE/'results/DEV/COUNTERFACTUAL_RESULT.json'
    if counter.exists():
        c=read(counter)
        out.extend(['### 同帧主体和参照物交换','',
                    f'有 {c["n"]} 组句子分别指向两个具有独立标注、水平投影明确分离的对象。两句均以新全文重跑。',''])
        for arm,row in c['arms'].items():
            out.append(f'- {arm} 有 {row["both_over25"]} 组双句均达 Acc@25，'
                       f'{row["both_over50"]} 组均达 Acc@50，{row["role_switch_correct"]} 组通过身份代理核验。')
        out.extend(['','参照角色与左右词交换发生在不同的已标注类别之间。'
                    '该检查只支持此类形态的互换诊断。[反事实控制](results/DEV/counterfactual.csv)',''])
    trajectories=HERE/'results'/('EVAL' if evaluation else 'DEV')/'all_cycle_metrics.csv'
    if trajectories.exists():
        tr=[r for r in table(trajectories) if r['source']=='six_equal' and r['cohort']==main_cohort]
        out.extend(['### 交替轮次的任务指标','',
                    '| 轮次 | 对照目标值 | mIoU / Acc@25 / Acc@50 |',
                    '|---|---:|---:|'])
        for r in tr:
            out.append(f'| {r["cycle"]} | {float(r["mean_objective"]):.4f} | {triple(r)} |')
        out.extend(['',f'[全部方向的逐轮数据](results/{"EVAL" if evaluation else "DEV"}/all_cycle_metrics.csv)',''])
    out.extend(['## 实现与评价口径','',
        '从冻结原生全文前向提取实体 token 到真实 3D seed 的注意力，组合两级索引回到原采样点。'
        '从冻结 R76 读取双向局部注意力、几何可见性和融合前图像特征。候选起点包括旧框和场景全局 seed 峰。'
        '每个角色保留多个假设及 UNKNOWN，仅对可解析且相机坐标可核验的左右关系加入有界因子。'
        '支持求解采用二次证据保持与图一致性。拟框使用归一化的鲁棒表面代价与弱尺寸先验，三轮更新连续中心、长宽高与轴向 yaw。'
        '串行先在初始框上用完 12 次支持扫掠，再用相同拟合迭代上限。'
        '[冻结接口复核](EVIDENCE_SMOKE.json) · [协议](experiment/PROTOCOL.json) · [源验证参数](experiment/SOURCE_PARAMETERS.json)','',
        '全系统无 RGB 对照从冻结源模型重新前向，跳过双向早期残差并清空 late RGB 的有效邻接。'
        '同场景错配控制交换互不重叠近邻候选区域中的等量点索引。公开表保留每个可用和不可用控制范围。[可见性分层](results/DEV/visibility_strata.csv)','',
        '推理从对应源平台的既有 epoch100 和 R76/R91 权重起步。源域各 64 条验证记录仅用于弱尺寸先验与三种推断设置的固定选择。'
        '常规方向使用原 16,384 个采样点、同步 RGB 和一条完整描述。权重更新次数为零。'
        '所有目标范围均有历史曝光，两个主要评估范围分开计算，当前轮 FINAL 访问次数为零。'
        '按采集序列抽样的区间对同一目标场景的两个源方向共享重采样权重。单场景方向只报告点估计。','',
        '传统模型原生轴向和类别辅助读出的差异以 R160 的同样本表为准。'
        '[R160 同样本报告](https://github.com/Zonglin-He/A/blob/7b44f7f5309e90966c2340564a30ce03814310fd/studies/3eed_official_vs_r158_paired_single_frame_20261118/REPORT_ZH.md)'])
    headings=['## 运行量','',
              '| 评估范围 | 累计有效记录 | 从原始输入到联合输出估算中位 ms | 缓存后联合求解中位 ms | 缓存后串行中位 ms | 平均目标框假设数 | 回退条数 | CUDA 峰值 MiB |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for cohort, data, tag in ((DEV,dev,'DEV'),(EXTRA,evaluation,'EVAL'),(REC,evaluation,'EVAL')):
        if not data:
            continue
        subset=[r for r in data['rows'] if r['cohort']==cohort]
        med=[1000*np.median([float(r[k]) for r in subset]) for k in ('solve_seconds','serial_seconds')]
        total=1000*np.median([float(r['forward_seconds'])+float(r['solve_seconds']) for r in subset])
        n=sum(r['fallback']=='True' for r in subset)
        peaks=read(HERE/(tag+'_SEAL.json'))['peak_MiB']
        mean=np.mean([int(r['retained']) for r in subset])
        headings.append(f'| {LABELS[cohort]} | {len(subset)} | {total:.1f} | {med[0]:.1f} | {med[1]:.1f} | {mean:.2f} | {n} | {peaks:.0f} |')
    out.extend(['',*headings,'',
                '延迟在端到端批量实验中采集，包含语言解析和点像对应构造。每条主臂有一次原模型前向和两次冻结融合读出，'
                '缓存后的求解器不再运行神经网络。CUDA 峰值为所在评估阶段分配量。',''])
    work = [r for r in measured['rows'] if r['cohort']==main_cohort]
    sweeps = float(np.mean([int(r['sweeps']) for r in work]))
    cost_joint,cost_serial = [np.mean([int(r[k]) for r in work])
                             for k in ('fit_calls','serial_fit_calls')]
    out.extend([f'主范围 LCIB 与串行每条平均均做 {sweeps:.1f} 次角色支持扫掠，'
                f'共同的单框拟合迭代上限为 5。'
                f'提前停止和线搜索后，每条平均实际评估拟框目标 {cost_joint:.1f} / {cost_serial:.1f} 次。',''])
    runtime=HERE/'RAW_RUNTIME_AUDIT.json'
    if runtime.exists():
        quick=read(runtime)
        probes=quick['timing_subset']
        med_old=np.median([r['original_ms'] for r in probes])
        med_new=np.median([r['LCIB_ms'] for r in probes])
        added=np.median([r['LCIB_ms']-r['original_ms'] for r in probes])
        out.extend([f'独立加载原始输入的六个复放样本全部复现封存框，模型参数和缓冲区逐字节相同。'
                    f'该小样本中旧入口与 LCIB 的中位延迟为 {med_old:.1f} ms 和 {med_new:.1f} ms，'
                    f'配对新增延迟中位数 {added:.1f} ms。[原始输入复核](RAW_RUNTIME_AUDIT.json)',''])
    (HERE/'REPORT_ZH.md').write_text('\n'.join(out)+'\n')
    print(disposition,triple(info),delta(prev),delta(matched),flush=True)


def pair_geometry(out, location):
    examples={}
    for r in table(location):
        examples.setdefault((r['source'],r['target'],r['ordinal']),{})[r['arm']]=r
    common=[v for v in examples.values() if 'r158_full' in v and 'lcib' in v]
    if not common:
        return
    keys=('center_m','absolute_log_size','axial_yaw_deg')
    nums={}
    for arm in ('r158_full','lcib'):
        nums[arm]=[np.median([float(v[arm][k]) for v in common]) for k in keys]
    out.extend(['### 同一已标注对象的几何变化','',
                f'在 {len(common)} 条两个框均唯一匹配到同一标注目标的记录上，中心距离中位数从 '
                f'{nums["r158_full"][0]:.3f} m 变到 {nums["lcib"][0]:.3f} m，'
                f'尺寸绝对对数误差从 {nums["r158_full"][1]:.3f} 变到 {nums["lcib"][1]:.3f}，'
                f'轴向 yaw 误差从 {nums["r158_full"][2]:.1f}° 变到 {nums["lcib"][2]:.1f}°。'
                '朝向局部改善与中心、尺寸变差可以同时出现。'
                '[同对象测量](results/DEV/same_annotated_target_geometry.csv)',''])


if __name__=='__main__':
    main()
