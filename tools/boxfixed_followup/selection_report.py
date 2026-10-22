"""Post-seal controls and audit; never reselect a trial configuration."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha,setup
from stvg.tube_runner import write
from tools.boxfixed_tube.profiles import profile_id
from tools.boxfixed_tube.spatial import evidence_dp
from tools.tubedetr_opd_direct_diagnosis import cluster_difference
from tools.boxfixed_followup.selection import signals
import tools.tubedetr_boxfixed_development as original
PUBLIC=ROOT/'results/tubedetr_selection';OUT=ROOT/'outputs/tubedetr_selection'
CONTROL=PUBLIC/'control_plan.json';COUT=OUT/'controls'
CP=dict(selection_mode='ERef_only_no_selection')


def check_parent():
    f=PUBLIC/'final_seal.json';s=json.loads(f.read_text())
    for path,h in s['files'].items():assert sha(ROOT/path)==h,path
    return json.loads((PUBLIC/'plan.json').read_text())


def prepare_control():
    assert not CONTROL.exists();p=copy.deepcopy(check_parent())
    p.update(status='posthoc_baseline_only_no_reselection',trial_final_seal_sha256=sha(PUBLIC/'final_seal.json'),
             purpose='Separate candidate-selection gains from fixed ERef initialization; no new hyperparameters or trial selection',
             additional_predictions_per_dataset=128,parameters=CP)
    p['code_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    write(CONTROL,p)


def control_prediction(ds):
    setup();p=json.loads(CONTROL.read_text());torch.set_num_threads(1)
    for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h
    sf=OUT/ds/'banks/sealed.json';assert sha(sf)==p['candidate_banks'][ds]
    s=json.loads(sf.read_text());assert s['queries']==128
    target=COUT/ds/'profiles'/profile_id(CP);target.mkdir(parents=True,exist_ok=True);files={}
    for q,e in s['files'].items():
        assert sha(Path(e['path']))==e['sha256'];c=torch.load(e['path'],weights_only=False,map_location='cpu')
        def stay(b):return dict(final=b.E.clone(),actions=torch.zeros(b.frames,dtype=torch.long),reliability=b.temporal_reliability())
        path=target/(q+'.npz');rp=path.with_suffix('.json');original.save_npz(path,c,stay(c['bank']),stay(c['native_bank']))
        write(rp,dict(plan_sha256=sha(CONTROL),parameters=CP,request_sha256=e['request_sha256'],labels_read=False,admitted_frames=0,changed_from_ERef_frames=0,candidate_bank_sha256=e['sha256']))
        files[q]=dict(path=str(path.relative_to(target)),sha256=sha(path),receipt_path=str(rp.relative_to(target)),receipt_sha256=sha(rp))
    write(target/'sealed.json',dict(status='complete',queries=128,files=files,plan_sha256=sha(CONTROL),parameters=CP,labels_read=False))


def control_score(ds):
    original.PLAN=CONTROL;original.OUT=COUT;original.PUBLIC=PUBLIC
    original.checked();return original.score(ds,CP)


def describe_delta(details,reference,key):
    if key=='unobserved':values=[d['unobserved']['Full']-reference[d['id']]['unobserved']['Full']for d in details]
    else:values=[d['values']['Full'][key]-reference[d['id']]['values']['Full'][key]for d in details]
    parents=[d['parent']for d in details];r=cluster_difference(values,parents)
    groups=sorted(set(parents));sums={p:sum(v for v,a in zip(values,parents)if p==a)for p in groups};counts={p:parents.count(p)for p in groups}
    loo=[100*(sum(values)-sums[p])/(len(values)-counts[p])for p in groups]
    r.update(leave_one_parent_out_min_pp=min(loo),leave_one_parent_out_max_pp=max(loo),selection_after_repeated_development_exposure=True)
    return r


def main():
    p=check_parent();assert sha(PUBLIC/'final_seal.json')==json.loads(CONTROL.read_text())['trial_final_seal_sha256']
    for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h,f
    result={};audits={};table=[]
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    for ds in p['dataset_order']:
        selected=json.loads((PUBLIC/ds/'selected_profile.json').read_text());allm=json.loads((PUBLIC/ds/'all_metrics.json').read_text())
        sf=OUT/ds/'banks/sealed.json';assert sha(sf)==p['candidate_banks'][ds];banks=json.loads(sf.read_text())
        baseline=COUT/ds/'profiles'/profile_id(CP);es=json.loads((baseline/'sealed.json').read_text())
        for e in es['files'].values():assert sha(baseline/e['path'])==e['sha256'] and sha(baseline/e['receipt_path'])==e['receipt_sha256']
        assert es['queries']==128 and es['plan_sha256']==sha(CONTROL)
        metricsE=json.loads((baseline/'metrics.json').read_text());assert metricsE['prediction_seal_sha256']==sha(baseline/'sealed.json')
        metricsRoot=OUT/ds/'profiles'/selected['profile_id'];details=json.loads((metricsRoot/'details_private.json').read_text())
        originalRoot=OUT/ds/'profiles'/profile_id(selected['incumbent']['parameters']);originalDetails={d['id']:d for d in json.loads((originalRoot/'details_private.json').read_text())}
        EDetails={d['id']:d for d in json.loads((baseline/'details_private.json').read_text())}
        temporal={d['id']:dict(values={'Full':d['values']['Temporal']},unobserved={'Full':d['unobserved']['Temporal']})for d in details}
        observedRoot=OUT/ds/'profiles'/next(profile_id(m['parameters'])for m in allm if m['parameters']['selection_mode']=='observe_protection')
        observedDetails={d['id']:d for d in json.loads((observedRoot/'details_private.json').read_text())}
        paired={name:{key:describe_delta(details,ref,key)for key in ['s','unobserved','v']}for name,ref in [('Temporal',temporal),('ERef_only',EDetails),('original_score',originalDetails),('observation_protection',observedDetails)]}
        family={}
        for mode in ['original','observe_protection','cross_score','cross_gate','cross_agreement']:
            a=[m for m in allm if m['parameters']['selection_mode']==mode];best=max(a,key=lambda m:m['means']['Full']['s'])
            family[mode]=dict(parameters=best['parameters'],metrics=best)
        count=0
        for m in allm:
            target=OUT/ds/'profiles'/profile_id(m['parameters']);s=json.loads((target/'sealed.json').read_text())
            assert s['queries']==128 and sha(target/'sealed.json')==m['prediction_seal_sha256'] and s['plan_sha256']==sha(PUBLIC/'plan.json')
            for q,e in s['files'].items():
                assert sha(target/e['path'])==e['sha256'] and sha(target/e['receipt_path'])==e['receipt_sha256'];be=banks['files'][q]
                assert sha(Path(be['path']))==be['sha256'];c=torch.load(be['path'],map_location='cpu',weights_only=False)
                with np.load(target/e['path'])as z:
                    assert np.array_equal(z['Source'],c['Source']) and np.array_equal(z['frame_ids'],c['frame_ids']) and np.array_equal(z['positions8'],c['positions8'])
                    assert np.array_equal(z['Native_interval'],c['Native_interval']) and np.array_equal(z['Global6_interval'],c['Global6_interval'])
                    assert np.array_equal(z['BoxFixed'],c['bank'].boxes[torch.arange(c['bank'].frames),torch.from_numpy(z['actions'])].numpy())
                    assert ((z['BoxFixedNative'][:,None,:]==c['native_bank'].boxes.numpy()).all(-1).any(-1)).all()
                    if m['parameters']['selection_mode']=='original':
                        old=p['candidate_profiles'][ds];oldroot=ROOT/'outputs/tubedetr_spatial_refinement'/ds/'profiles'/old['profile_id'];os_=json.loads((oldroot/'sealed.json').read_text())
                        with np.load(oldroot/os_['files'][q]['path'])as a:assert set(z.files)==set(a.files) and all(np.array_equal(z[k],a[k])for k in z.files)
                count+=1
        # All predictions and baselines are complete/verified before this GT
        # analysis. Physical-node results are reported separately from dense.
        d=p['datasets'][ds];assert sha(ROOT/d['labels'])==d['labels_sha256']
        truth={r['id']:r for r in map(json.loads,(ROOT/d['labels']).read_text().splitlines())};requests={r['id']:r for r in map(json.loads,(ROOT/d['manifest']).read_text().splitlines())}
        node_gains=[];node_queries=[];changed_nodes=observed_changes=admitted=eligible=0
        for q,e in banks['files'].items():
            c=torch.load(e['path'],map_location='cpu',weights_only=False);b=c['bank'];sig=signals(b);valid=b.valid.clone();mode=selected['parameters']['selection_mode']
            if mode!='original':valid[b.observed,1:]=False
            if mode.startswith('cross'):
                ok=sig['independent']&(sig['joint_reliability']>=selected['parameters']['cycle_floor'])
                for k in (1,2):valid[:,k]&=ok&(sig['advantage'][:,k]>selected['parameters']['margin'])
                if mode=='cross_agreement':valid[:,1:]&=(sig['agreement']>=selected['parameters']['agreement_floor'])[:,None]
            admitted+=int((b.active&valid[:,1:].any(-1)).sum());eligible+=int((b.active&b.valid[:,1:].any(-1)).sum())
            ps=json.loads((metricsRoot/'sealed.json').read_text());meta=requests[q]['metadata']
            with np.load(metricsRoot/ps['files'][q]['path'])as z:
                actual=z['BoxFixed'];a=z['actions'];observed_changes+=int(((a!=0)&b.observed.numpy()).sum());changed_nodes+=int(((a!=0)&~b.observed.numpy()).sum())
                ids=np.array(c['frame_ids']);boxes={int(k):v for k,v in truth[q]['boxes'].items()};nodes=[t for t,f in enumerate(ids)if int(f)in boxes and not b.observed[t]]
                if nodes:
                    gt=np.array([boxes[int(ids[t])]for t in nodes]);pred=xyxy(actual[nodes],meta['width'],meta['height']);src=xyxy(np.asarray(c['Source'])[nodes],meta['width'],meta['height']);eref=xyxy(b.E.numpy()[nodes],meta['width'],meta['height'])
                    if ds=='hc2':pred=np.maximum(pred,0);src=np.maximum(src,0);eref=np.maximum(eref,0)
                    change=box_iou(pred,gt)-box_iou(src,gt);node_gains.extend(change.tolist());node_queries.append(float(change.mean()))
        physical=dict(GT_annotated_unobserved_nodes=len(node_gains),pooled_node_IoU_gain_vs_Source_pp=100*float(np.mean(node_gains)),equal_query_node_IoU_gain_vs_Source_pp=100*float(np.mean(node_queries)),improved_nodes=int((np.array(node_gains)>1e-12).sum()),harmed_nodes=int((np.array(node_gains)<-1e-12).sum()),actual_unobserved_actions=changed_nodes,actual_observed_actions=observed_changes,new_admitted_nodes=admitted,original_eligible_nodes=eligible,scope='Physical GT-supported unobserved nodes; not dense interpolated estimand. Source/ERef differences on HC include fixed initialization.')
        point=selected['qualifies'];stable=point and all(paired['Temporal'][k]['CI95_pp'][0]>0 for k in ['s','unobserved'])
        result[ds]=dict(selected_parameters=selected['parameters'],candidate_parameters=selected['candidate_parameters'],point_guardrails_pass=point,practical_target=selected['practical_positive_target'],positive_bootstrap_lower_bounds=stable,paired=paired,family_controls=family,ERef_only_metrics=metricsE,physical_unobserved=physical,not_independent_confirmation=True)
        audits[ds]=dict(profiles=len(allm),trial_predictions_and_receipts=count,control_predictions=128,old_control_arrays_bitexact=True,candidate_coordinates_and_membership_unchanged=True,Source_frames_observations_and_both_time_arrays_bitexact=True,bank_seal_sha256=sha(sf),control_seal_sha256=sha(baseline/'sealed.json'))
        m=selected['metrics'];a=m['means']['Full'];t=m['means']['Temporal'];u=m['unobserved_sIoU']
        table.append(f"| {ds}128 | {len(allm)} | {a['s']:.6f} | {a['s']-t['s']:+.6f} | {u['Full']:.6f} | {u['Full']-u['Temporal']:+.6f} | {a['v']:.6f} | {a['v']-t['v']:+.6f} | {selected['practical_positive_target']} |")
    write(PUBLIC/'analysis.json',dict(status='complete',datasets=result))
    write(PUBLIC/'integrity.json',dict(status='pass',datasets=audits,engineering_tests=94,old_source_code_files_verified=len(p['code_sha256']),new_neural_calls=0,new_full_predictions=0,counts_scope='mean_admitted_frames in trial metrics is original-bank eligibility, not new gate acceptance; actual new counts in analysis.physical_unobserved'))
    text='# 固定候选评分与准入：两域128完整验证\n\n候选坐标、时间分支、Source、教师、八观察和输入不变；只修改评分/准入。全部预测先封存后独立CPU评分，没有新神经网络调用或全量。所有负配置保留。128已重复曝光，以下不是独立泛化确认。\n\n| 面板 | 配置数 | Full dense | dense增量pp | 未观察dense | 未观察增量pp | Full v | v增量pp | 两项空间均达+.1pp |\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n'+'\n'.join(table)+'\n\n增量相对同一Source+Global6（Temporal）；另补同初始化ERef-only控制，控制在原trial终态后单独冻结，不用于重新选择配置。\n\n'
    for ds,v in result.items():
        text+=f"## {ds}\n\n评分参数：`{json.dumps(v['selected_parameters'])}`；候选参数固定：`{json.dumps(v['candidate_parameters'])}`。点估计护栏={v['point_guardrails_pass']}，实际增量目标={v['practical_target']}，两项空间bootstrap下界均正={v['positive_bootstrap_lower_bounds']}。\n\n"
        for ref,groups in v['paired'].items():
            text+=f'相对{ref}：'+ '；'.join(f"{k} {g['mean_pp']:+.6f}pp，95%父bootstrap[{g['CI95_pp'][0]:.6f},{g['CI95_pp'][1]:.6f}]，胜/负/平{g['wins']}/{g['losses']}/{g['ties']}，逐父留一最低{g['leave_one_parent_out_min_pp']:+.6f}pp"for k,g in groups.items())+'。\n\n'
        text+='实际物理节点（独立于dense插值统计）：'+json.dumps(v['physical_unobserved'],ensure_ascii=False)+'\n\n'
    text+='## 边界\n\n不同锚点的交叉证据仍可能同错，不是身份真值。观察保护关闭的是直接覆盖，HC仍含原gamma初始化；ERef控制用于分解这一点。未观察dense会受到观察节点两侧插值影响，必须与实际未观察节点统计并读。小幅正点估计不能冒充稳定收益；bootstrap在选择后描述且未修正多轮开发选择。旧全量没有改写，本轮不提供自动全量入口。\n'
    (PUBLIC/'report.md').write_text(text)
    files=[PUBLIC/f for f in ['final_seal.json','control_plan.json','analysis.json','integrity.json','report.md']]+[ROOT/'docs/tubedetr_selection.md',Path(__file__)]
    write(PUBLIC/'completion_seal.json',dict(status='complete_fixed128_only',files={str(f.relative_to(ROOT)):sha(f)for f in files},publication_pending=True))
    print(json.dumps({ds:dict(point=v['point_guardrails_pass'],practical=v['practical_target'],bootstrap=v['positive_bootstrap_lower_bounds'],physical=v['physical_unobserved'])for ds,v in result.items()},indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['prepare','predict','score','report'],required=True);a.add_argument('--dataset',choices=['vidstg','hc2']);args=a.parse_args()
    if args.stage=='prepare':prepare_control()
    elif args.stage=='predict':control_prediction(args.dataset)
    elif args.stage=='score':control_score(args.dataset)
    else:main()
