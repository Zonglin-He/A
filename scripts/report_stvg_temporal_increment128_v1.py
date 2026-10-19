"""Independent algebra/input audit and post-screen report. No new model calls."""
from pathlib import Path
import sys,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_increment128_v1 import D,P,EVENT,PRE

def main():
    from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,independent_cdf
    from vg_tta.stvg_where_when_event128_v1 import interval_scores
    from scipy.special import logsumexp
    p=read(D/'PLAN_LOCK.json');results=read(P/'RESULTS_PUBLIC.json');assert len(results['rows'])==128
    checks=dict(prior_CDF=0,raw_feedback_integrals=0,posterior_reweight=0,zero_lambda_exact=0,spatial_arrays_exact=0)
    maxcdf=maxpost=0.
    for r in p['records']:
        a=r['alias'];prior=load(D/'priors'/f'{a}.pt');z=load(D/'predictions'/f'{a}.pt');old=load(EVENT/'predictions'/f'{a}.pt');q=z['q0'];ids=np.asarray(z['frame_ids']);cdf=independent_cdf(probabilities(prior['records'],prior['logits']),ids.tolist());err=float(abs(cdf-q).max());assert err<1e-12;maxcdf=max(maxcdf,err);checks['prior_CDF']+=1
        for space,b in z['trajectories'].items():assert np.array_equal(b,old['trajectories'][space]);checks['spatial_arrays_exact']+=1
        ii,jj=np.triu_indices(len(ids),1);cs=np.stack((ids[ii],ids[jj]+1),1).tolist()
        for space in ['E']+[f'Live{s}'for s in p['seeds']]:
            name='EventE'if space=='E'else'Event'+space;curve=old['readouts'][name]['curve'];raw=interval_scores(curve,np.array(old['edges']),cs);valid=np.array([v is not None for v in raw]);vals=np.array([v for v in raw if v is not None]);tilde=np.zeros(len(ii))
            if curve is not None and len(vals)and vals.std()>1e-12 and np.ptp(curve)>1e-12:tilde[valid]=(vals-vals.mean())/vals.std()
            positive=q[ii,jj]>0
            for lam in p['lambdas']:
                o=z['outputs'][f'{space}_L{lam:g}'];post=o['posterior']
                if lam==0:assert np.array_equal(post,q)and o['interval']==old['original_interval'];checks['zero_lambda_exact']+=1
                expected=np.zeros(len(ii))
                lp=np.log(q[ii,jj][positive])+lam*tilde[positive];expected[positive]=np.exp(lp-logsumexp(lp))
                err=float(abs(post[ii,jj]-expected).max());assert err<1e-11;maxpost=max(maxpost,err);assert np.all(post[q==0]==0);checks['posterior_reweight']+=1
            checks['raw_feedback_integrals']+=len(raw)
        assert sha(EVENT/'predictions'/f'{a}.pt')==r['event_prediction_sha256']
        assert sha(PRE/'inputs'/f'{a}.pt')==r['spatial_input_sha256']
    assert all(sha(R/f)==h for f,h in p['pins'].items())
    audit=dict(status='pass',checks=checks,max_prior_CDF_error=maxcdf,max_posterior_reweight_error=maxpost,all_old_event_and_spatial_inputs_unchanged=True,frozen_source_and_config_pins_unchanged=True,all128_parents_3seeds_retained=True,new_teacher_calls=0,new_temporal_OPD=False)
    put(P/'ROOT_AUDIT.json',audit)
    selected=results['selected_lambda'];means=results['means'];lines=['# 保留原时间分布的事件重加权：HC128筛查','',
        '日期：2026-10-19。已暴露的全部128开发父视频；固定既有Live空间轨迹/种子41、42、43，指标先父内平均。原E空间另列，不用E结果选择lambda。没有新增专家调用、训练或OPD更新。','',
        '## 唯一参数筛查','', '| lambda | 固定Live空间完整vIoU % | tIoU % | 平均边界误差 frames | 完整v增量pp [95%配对区间] |','|---:|---:|---:|---:|---|']
    for lam in p['lambdas']:
        n=f'Live_seedmean_L{lam:g}';v=means[n];delta=results['paired'][n+'_minus_Live_seedmean_L0']['v'];lines.append(f'| {lam:g} | {100*v["v"]:.5f} | {100*v["t"]:.5f} | {v["mean_boundary_error_frames"]:.4f} | {delta["mean"]:+.5f} [{delta["CI95"][0]:+.5f}, {delta["CI95"][1]:+.5f}] |')
    lines+=['',f'预登记完整v均值排序选择 **lambda={selected:g}**，同视频/错误类型/种子不分配不同值。所有结果均为开发结果，选择后区间不是独立确认的统计保证。lambda0是精确恒等基线，不能把它当成新时间收益。','',
        'p0为当前原MBR实际接受的两offset全物理包络分布，不是上一轮约37个事件读出区间。重加权前后支持和原MBR代码相同：65端点/2080个strict-index决策，包括零mass决策；零先验质量不增加。R仍是旧八帧/旧词组/旧关联/旧duration contrast，仅按全legal定义区间做视频内mean/std标准化。全实际视频没有outside，增量0；若不在原端点支持则不补入。','',
        '## 固定E空间的辅助对照','', '| lambda | vIoU % | tIoU % |','|---:|---:|---:|']
    for lam in p['lambdas']:
        v=means[f'E_L{lam:g}'];lines.append(f'| {lam:g} | {100*v["v"]:.5f} | {100*v["t"]:.5f} |')
    lines+=['','## 事后时间错误分解','',
        '所有分组仅在全128输出封存后使用GT解释结果，不进入重加权、参数或回退。原时间损失以相同空间轨迹配GT时间为参考；容量以原MBR的完整决策支持为oracle，非上一轮粗候选容量，不能跨空间基线相减。','',
        '| 原时间错误类型 | 父视频 | 原时间损失pp | 原MBR支持最多恢复pp | 选中lambda实际恢复pp |','|---|---:|---:|---:|---:|']
    for name,v in results['error_decomposition'].items():
        def num(k):return 'unknown'if v[k]is None else f'{v[k]:+.5f}'
        lines.append(f'| {name} | {v["parents"]} | {num("original_temporal_loss_pp")} | {num("candidate_capacity_pp")} | {num("actual_selected_recovery_pp")} |')
    lines+=['','错段=零交集；漏段=预测严格包含于GT；超出=预测严格包含GT；其他=双边错位或精确匹配，精确匹配数另见JSON。每组以组内所有父视频为分母，三seed先父内平均；不是按每类选参数。','',
        '## 旧报告算术核对','', '```json',json.dumps(results['prior_experiment_arithmetic_check'],indent=2),'```','',
        '上表为旧输出的正负贡献分解，分母始终128；不是GT门控的新方法，也不能与另一个空间基线的oracle容量相减。','',
        '## 独立确认状态','',
        '当前验证清单3482query只有237个父视频，扣本轮RC128最多109父。109名单在开发选参前锁定；含RH32/RF32/NH16既往曝光，不能称完全新测试。它低于用户要求至少128，且缺逐token事件缓存。在无新专家调用约束下尚不能完成独立确认；两个范围问题已提出，答复前不新增ReferDINO、不借开发样本冒充确认。','',
        '## 成本与结论边界','', '```json',json.dumps(results['cost'],indent=2),'```','',
        '只补导出了缺失的冻结Source分布；无新ReferDINO/DINO/参数更新。λ=0恢复旧时间和指标精确；独立CDF/直接积分/指数重加权/官方与DenseTube评价核验通过。保留全部父视频、所有lambda和种子、严重退化，不用GT选错误类型。','',
        '本轮不是temporal OPD；重加权胜出也只能先称时间互补性信号，不能据此归于on-policy更新。若开发非零配置均不优于0，则当前反馈的互补性未获支持，不再追加lambda、温度或改反馈救结果；若开发提高，仍需与开发父独立且证据同条件的确认。冻结空间和生产/研究登记不变。','']
    (P/'REPORT.md').write_text('\n'.join(lines))
    put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),selection_sha256=sha(D/'SELECTION_LOCK.json'),code_pins=p['pins'],prior_cached=p['prior_cached'],new_Source_exports=p['new_Source_exports_required'],no_new_expert_calls=True,all_outputs_before_new_GT=True,confirmation_requested=128,confirmation_candidates=109,confirmation_not_completed=True,private_excluded=['media','captions','parent identifiers','weights','GT coordinates','raw logits/posteriors/trajectories']))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
