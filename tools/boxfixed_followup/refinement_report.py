"""Post-seal audit and descriptive analysis; no prediction/selection changes."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write
from tools.tubedetr_opd_direct_diagnosis import cluster_difference
PUBLIC=ROOT/'results/tubedetr_spatial_refinement';OUT=ROOT/'outputs/tubedetr_spatial_refinement'


def main():
    terminal=json.loads((PUBLIC/'final_seal.json').read_text())
    for f,h in terminal['files'].items():assert sha(ROOT/f)==h,f
    plan=json.loads((PUBLIC/'plan.json').read_text())
    for f,h in plan['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert sha(ROOT/'results/tubedetr_boxfixed_development/final_seal.json')==plan['parent_final_seal_sha256']
    analysis={};audit={};rows=[]
    for ds in plan['dataset_order']:
        selection=json.loads((PUBLIC/ds/'selected_profile.json').read_text())
        metrics=json.loads((PUBLIC/ds/'all_metrics.json').read_text())
        assert len(metrics)==selection['profiles']<=16
        incumbent=selection['incumbent']
        from tools.boxfixed_tube.profiles import profile_id
        incroot=OUT/ds/'profiles'/profile_id(incumbent['parameters'])
        incseal=json.loads((incroot/'sealed.json').read_text())
        predictions=0
        for m in metrics:
            target=OUT/ds/'profiles'/profile_id(m['parameters']);s=json.loads((target/'sealed.json').read_text())
            assert sha(target/'sealed.json')==m['prediction_seal_sha256'] and s['queries']==128
            assert s['plan_sha256']==sha(PUBLIC/'plan.json') and s['parameters']==m['parameters']
            for q,e in s['files'].items():
                assert sha(target/e['path'])==e['sha256'] and sha(target/e['receipt_path'])==e['receipt_sha256']
                with np.load(target/e['path'])as z,np.load(incroot/incseal['files'][q]['path'])as a:
                    for k in ['Source','Native_interval','Global6_interval','frame_ids','positions8']:
                        assert np.array_equal(z[k],a[k]),(ds,q,k)
                    assert np.isfinite(z['BoxFixed']).all() and (z['BoxFixed'][:,2:]>0).all()
                predictions+=1
        gs=OUT/ds/'geometry/sealed.json';g=json.loads(gs.read_text())
        for e in g['files'].values():assert sha(Path(e['path']))==e['sha256']
        root=OUT/ds/'profiles'/selection['profile_id'];details=json.loads((root/'details_private.json').read_text())
        old={x['id']:x for x in json.loads((incroot/'details_private.json').read_text())}
        paired={}
        for name in ['Temporal','incumbent']:
            groups={}
            for key in ['v','s']:
                vals=[d['values']['Full'][key]-(d['values']['Temporal'][key]if name=='Temporal'else old[d['id']]['values']['Full'][key])for d in details]
                groups[key]=cluster_difference(vals,[d['parent']for d in details])
            vals=[d['unobserved']['Full']-(d['unobserved']['Temporal']if name=='Temporal'else old[d['id']]['unobserved']['Full'])for d in details]
            groups['unobserved_sIoU']=cluster_difference(vals,[d['parent']for d in details]);paired[name]=groups
        family={}
        for mode in ['original','center_preserve','relative_residual']:
            a=[m for m in metrics if m['parameters']['mode']==mode]
            best=max(a,key=lambda m:m['means']['Full']['s'])
            family[mode]=dict(parameters=best['parameters'],dense=best['means']['Full']['s'],v=best['means']['Full']['v'],unobserved=best['unobserved_sIoU']['Full'],changed_frames=best['mean_changed_from_ERef_frames'])
        result=dict(parameters=selection['parameters'],qualifies=selection['qualifies_for_spatial_promotion'],paired=paired,family_best_by_dense=family,post_selection_intervals_not_independent=True)
        analysis[ds]=result
        audit[ds]=dict(profiles=len(metrics),predictions_and_receipts=predictions,unchanged_Source_time_frames_observations=True,incumbent_arrays_bitexact=selection['incumbent_all_arrays_bitexact'],geometry_seal_sha256=sha(gs),all_geometry_payloads_verified=True)
        m=selection['metrics'];t=m['means']['Temporal'];b=m['means']['Full'];u=m['unobserved_sIoU']
        rows.append(f"| {ds}128 | {len(metrics)} | {t['s']:.6f} | {b['s']:.6f} | {b['s']-t['s']:+.6f} | {u['Full']-u['Temporal']:+.6f} | {b['v']:.6f} | {b['v']-t['v']:+.6f} | {selection['qualifies_for_spatial_promotion']} |")
    write(PUBLIC/'analysis.json',dict(status='complete',datasets=analysis,new_full_predictions=0,negative_profiles_retained=True))
    write(PUBLIC/'integrity.json',dict(status='pass',datasets=audit,immutable_code_files=len(plan['code_sha256']),old_full_seal_unchanged=True,engineering_tests=90,new_teacher_or_neural_calls=0,new_full_predictions=0))
    report='# TubeDETR空间小改动：两域固定128完整结果\n\n本轮为用户授权的两种未观察候选公式小改动及有限标量验证；主干、教师、八观察、时间、原评分与精确DP保持不变。先完整预测封存，再独立CPU评分。数据已多次曝光并用于选择，不能称独立确认。没有新神经网络调用或全量。\n\n| 面板 | 配置数 | Temporal dense | 所选dense | dense增量pp | 未观察增量pp | Full v | v增量pp | 事前护栏通过 |\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n'+'\n'.join(rows)+'\n\n所选参数、正负结果、逐阶段理由见各域selected_profile/all_metrics/decisions。原仅参数最佳是本轮incumbent；没有用旧未执行机制草稿的结果选参。两个候选族分别针对WLS收缩和绝对框迁移风险，不同时改变置信或评分。候选小改动属于输出侧后处理，不能称原学生decoder新预测。\n\n'
    for ds,a in analysis.items():
        report+=f"## {ds}\n\n参数 `{json.dumps(a['parameters'],ensure_ascii=False)}`；事前护栏通过={a['qualifies']}。\n\n"
        for ref,groups in a['paired'].items():
            report+=f"相对{ref}："+'；'.join(f"{k} {d['mean_pp']:+.6f}pp，描述性父bootstrap95%区间[{d['CI95_pp'][0]:.6f},{d['CI95_pp'][1]:.6f}]，胜/负/平={d['wins']}/{d['losses']}/{d['ties']}"for k,d in groups.items())+'。\n\n'
        report+='各族完整结果中dense最高配置：\n\n'
        for mode,m in a['family_best_by_dense'].items():report+=f"- {mode}: dense={m['dense']:.6f}，v={m['v']:.6f}，未观察={m['unobserved']:.6f}，平均改动帧={m['changed_frames']:.4f}，参数{json.dumps(m['parameters'])}。\n"
        report+='\n'
    report+='护栏只保证本开发面板点估计，没有解决多次比较和重复曝光。所有bootstrap均在选择之后描述，不作独立显著性确认。若无可行配置则不晋升；未启动新全量。零初始化、尺寸保留或更保守修改不能自动证明传播可靠性，不能把减少改动当成未观察帧获得真实信息。\n'
    (PUBLIC/'report.md').write_text(report)
    files=[PUBLIC/'final_seal.json',PUBLIC/'analysis.json',PUBLIC/'integrity.json',PUBLIC/'report.md',ROOT/'docs/tubedetr_spatial_refinement.md',Path(__file__)]
    write(PUBLIC/'completion_seal.json',dict(status='complete_fixed128_only',files={str(f.relative_to(ROOT)):sha(f)for f in files},publication_pending=True))
    print(json.dumps(dict(status='complete',datasets={ds:dict(qualifies=a['qualifies'],parameters=a['parameters'])for ds,a in analysis.items()}),indent=2))


if __name__=='__main__':main()
