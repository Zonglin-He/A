"""Post-evaluation root audit and report; never an inference dependency."""
from pathlib import Path
import sys,json,hashlib,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_where_when_event128_v1 import D,P,PRE,FINAL

def main():
    p=read(D/'PLAN_LOCK.json');out=read(P/'RESULTS_PUBLIC.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
    assert len(out['rows'])==len(bar['files'])==128 and all(sha(R/f)==h for f,h in p['pins'].items())
    prev={r['alias']:r['values']for r in read(R/'results/stvg_error_state/2026-10-19/referdino_final/SELECTED_DEVELOPMENT_RESULTS_PUBLIC.json')['rows']}
    nchecks=0;association_checks=0;maxerr=0.;same=0;zero=0
    for r,rr in zip(p['records'],out['rows']):
        a=r['alias'];assert a==rr['alias'];old=prev[a]
        for key,oldkey in [('E_MBR','E_MBR')]+[(f'Live{s}_OriginalTime',f'T20_Live_{s}')for s in p['seeds']]:
            for m in ['v','t','s','unobserved_actual_s']:
                x=rr['values'][key][m];y=old[oldkey][m]
                assert (x is None and y is None)or abs(x-y)<2e-10,(a,key,m,x,y)
                if x is not None:maxerr=max(maxerr,abs(x-y))
                nchecks+=1
        for n,v in rr['values'].items():
            if n.startswith('E_'):
                assert abs(v['s']-rr['values']['E_MBR']['s'])<1e-14
                x,y=v['unobserved_actual_s'],rr['values']['E_MBR']['unobserved_actual_s']
                assert (x is None and y is None)or abs(x-y)<1e-14
        for f,k in [(PRE/'teacher0'/f'{a}.pt','teacher0_sha256'),(PRE/'inputs'/f'{a}.pt','spatial_input_sha256'),(FINAL/'Live/predictions'/f'{a}.pt','live_sha256'),(R/r['subject_path'],'subject_sha256')]:assert sha(f)==r[k]
        z=load(D/'predictions'/f'{a}.pt');sem=load(D/'semantics'/f'{a}.pt');b=sem['all_boxes'].numpy()
        for name,entry in z['association'].items():
            center=np.asarray(z['trajectories'][name])[r['positions8']];scores=[]
            for j in range(b.shape[1]):
                vals=[]
                for s,t in zip(center,b[:,j]):
                    lo=np.maximum(s[:2]-s[2:]/2,t[:2]-t[2:]/2);hi=np.minimum(s[:2]+s[2:]/2,t[:2]+t[2:]/2);area=np.maximum(hi-lo,0).prod();vals.append(area/(s[2:].prod()+t[2:].prod()-area))
                scores.append(float(np.mean(vals)))
            assert entry['query']==int(np.argmax(scores)) and np.max(np.abs(np.asarray(scores)-entry['all_query_mean_IoU']))<1e-6;association_checks+=1;zero+=entry['no_geometric_overlap']
        for s in p['seeds']:
            if z['association'][f'Live{s}']['query']==z['association']['E']['query']:assert z['readouts'][f'EventLive{s}']==z['readouts']['EventE'];same+=1
    assert out['cost']['GPU_seconds']<=p['GPU_seconds_cap'] and out['cost']['new_teacher_video_queries']==128
    for f,h in read(D/'CHECKPOINT_PREFLIGHT.json')['pins'].items():assert sha(R/f)==h
    root=dict(status='pass',old_baseline_metric_checks=nchecks,old_baseline_max_error=maxerr,independent_global_track_association_checks=association_checks,same_track_identical_feedback_checks=same,zero_overlap_associations=zero,all_old_inputs_predictions_subjects_and_teachers_unchanged=True,teacher_checkpoint_hashes_unchanged=True,frozen_spatial_config_unchanged=True,all128_parents_all3seeds_retained=True,prediction_barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json'),separate_fixed_space_temporal_and_joint_metrics=True,new_temporal_OPD=False)
    put(P/'ROOT_AUDIT.json',root)
    put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),configuration=p['configuration'],frozen_spatial_config=p['frozen_spatial_config'],checkpoint_pins=read(D/'CHECKPOINT_PREFLIGHT.json')['pins'],code_pins=p['pins'],cohort='existing exposed RC128; not independent confirmation',all_sealed_before_new_GT=True,new_spatial_updates=0,new_temporal_OPD=False,private_excluded=['RGB','original captions','parent source identifiers','annotations','weights','raw premax logits and features','raw trajectories']))
    def point(a,k):return 100*out['means'][a][k]
    def pp(a,k):
        z=out['paired_pp'][a][k];return f"{z['mean']:+.4f} [{z['CI95'][0]:+.4f}, {z['CI95'][1]:+.4f}]"
    lines=['# ReferDINO对象条件事件反馈：HC128能力检查','',
           '日期：2026-10-19。128个已暴露开发父视频，冻结空间配置、原八帧、原生ReferDINO；没有运行时间OPD、空间重适应或参数搜索。',
           '', '## 固定E空间坐标的时间读出','', '| 读出 | 完整vIoU % | tIoU % |', '|---|---:|---:|']
    for a in ['E_MBR','E_NativeConfidence','E_EventNative','E_EventE','E_EventLive_seedmean']:lines.append(f'| {a} | {point(a,"v"):.4f} | {point(a,"t"):.4f} |')
    lines+=['','全部方法使用同一E空间坐标，因此actual-node sIoU按构造相同；tIoU变化与vIoU变化分别报告。三seed先父内平均指标，不平均框或挑seed。','',
            '| 比较，pp [父视频配对95%区间] | 完整vIoU | tIoU |','|---|---:|---:|']
    for a in ['E_EventE_minus_E_NativeConfidence','E_EventE_minus_E_EventNative','E_EventLive_seedmean_minus_E_EventE','E_EventLive_seedmean_minus_E_MBR','E_EventE_minus_E_MBR','Live_NewTime_seedmean_minus_Live_OriginalTime_seedmean','Live_NewTime_seedmean_minus_E_MBR']:lines.append(f'| {a} | {pp(a,"v")} | {pp(a,"t")} |')
    lines+=['','## 稀疏语义与候选排序','', '| 响应 | 事件内外AUC（父均值） | 区间tIoU成对排序准确率（父均值） |','|---|---:|---:|']
    for a,x in out['event_summary'].items():
        def f(k):
            v=x[k];return 'unknown'if v['mean']is None else f"{100*v['mean']:.3f}% [{100*v['CI95'][0]:.3f}, {100*v['CI95'][1]:.3f}], n={v['parents']}"
        lines.append(f'| {a} | {f("inside_outside_AUC")} | {f("candidate_tIoU_pair_accuracy")} |')
    lines+=['','有效词组数、时间覆盖、对象切换和仅离线候选容量：','', '```json',json.dumps(out['coverage'],indent=2,ensure_ascii=False),'```','',
            '## 负例与成本','', '完整主评价保留全部128。以下是最终联合输出相对E的严重尾部；不是筛样或回退依据。','', '```json',json.dumps(out['paired_pp']['Live_NewTime_seedmean_minus_E_MBR']['tails'],indent=2,ensure_ascii=False),'```','',
            '```json',json.dumps(out['cost'],indent=2,ensure_ascii=False),'```','',
            '成本为新增教师租约，包含一次模型加载及持锁期间解码/写出；没有新Source/DINO-T或空间更新。既有E/空间OPD缓存的历史成本没有计作免费总推理。逐query解码/前向成本另见COST_BREAKDOWN.json。','',
            '## 结论边界','',
            '这项实验只检验固定的词组/全轨迹关联/稀疏时间contrast接口。事件分数不是校准事件概率；全上下文融合响应不是独立帧观测。描述性分词和交互对象名词也可能进入固定词组。','',
            '同一教师track的语义响应不依赖学生框；空间作用若存在，首先归于选择不同track，不能称同一对象的语义被重读。一个track index也不保证物理身份从未漂移。','',
            '时间曲线只来自八个实际位置；Voronoi单元仅定义积分。fullspan没有outside，contrast未知，不能借人为0分宣称能排序全视频。无事件/恒定响应按协议保留MBR。','',
            'HC事件框不提供独立事件外可见性真值。本轮不能完成“目标可见但事件不成立”的可靠负例验证；不拿teacher高分或缺GT框循环定义这类标签。','',
            '候选oracle在全预测封存后离线计算，仅衡量有限支持容量，不是可部署收益。直接读出失败也不等于所有OPD失败；候选排序和任务结果分别判断。没有运行新时间策略采样/蒸馏，不把本轮成绩归为OPD增量。','',
            '现空间版本继续封存，CURRENT/MAIN/PAPER不改；新一轮模型、扫参、扩大观察或OPD均不自动启动。所有query/seed signed结果与unknown见RESULTS_PUBLIC.json；数值与源缓存核验见INDEPENDENT_AUDIT.json、ROOT_AUDIT.json。','']
    (P/'REPORT.md').write_text('\n'.join(lines))
    costs=[]
    for r in p['records']:
        s=load(D/'semantics'/f"{r['alias']}.pt");costs.append(dict(alias=r['alias'],**s['cost']))
    put(P/'COST_BREAKDOWN.json',dict(**out['cost'],rows=costs,means={k:float(np.mean([x[k]for x in costs]))for k in ['expert_seconds','preprocess_seconds','forward_seconds','postprocess_seconds','decode_seconds']},shared_model_load_seconds=sum(read(f).get('model_load_seconds',0)for f in (D/'cost').glob('*.json'))))
    print(json.dumps(root,indent=2))

if __name__=='__main__':main()
