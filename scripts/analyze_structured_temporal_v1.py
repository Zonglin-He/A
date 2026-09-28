"""Independent F43 NumPy math, original metrics, real-state and source audit.

Does not choose hyperparameters or alter any prediction. GT is offline only.
"""
import collections
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.special import expit,log_expit,logsumexp
from sklearn.metrics import roc_auc_score
from scripts.decota_matrix_common_v1 import read,write,status,load,sha
from scripts.structured_temporal_v1 import OUT,F41,plan,cache_for,recorded,METRICS
from scripts.analyze_spatial10_components_v1 import labels_for,checked_score,summary,aggregate
from scripts.audit_spatial4_attribute_v1 import aggregate_check
from vg_tta.dense_support_tuning_v1 import state_hash


def independent_teacher(zs,teacher,records,cfg,calibrated=True):
    offsets=[]
    for z,t,record in zip(zs,teacher,records):
        x=t['raw_logits'].numpy().reshape(-1).astype(float)
        center=np.median(x);mad=np.median(abs(x-center))
        zz=(x-center)/(mad+cfg['epsilon'])/cfg['temperature'] if calibrated else x
        la,lb=log_expit(zz),log_expit(-zz)
        f=np.asarray(record['frame_ids'],float)
        edges=np.concatenate((f[:1],(f[:-1]+f[1:])/2,f[-1:]+1))
        w=np.diff(edges);w/=w.sum()
        z=z.numpy().reshape(-1,2).astype(float)
        s,e=np.triu_indices(len(z),1);score=z[s,0]+z[e,1];lp=score-logsumexp(score)
        inside=(np.arange(len(f))[None]>=s[:,None]) & (np.arange(len(f))[None]<=e[:,None])
        cost=-(w[None]*(inside*la[None]+(~inside)*lb[None])).sum(1)
        g=-cost+cfg['prior_weight']*lp
        order=np.argsort(-g,kind='stable');top=int(order[0]);second=int(order[min(1,len(order)-1)])
        offsets.append(dict(x=x,a=expit(zz),center=center,mad=mad,w=w,edges=edges,lp=lp,
            cost=cost,g=g,s=s,e=e,top=top,margin=g[top]-g[second],
            interval=[int(f[s[top]]),int(f[e[top]])+1]))
    a,b=[t['interval'] for t in offsets]
    overlap=max(0,min(a[1],b[1])-max(a[0],b[0]))/(max(a[1],b[1])-min(a[0],b[0]))
    return offsets,float(overlap*expit(offsets[0]['margin'])*expit(offsets[1]['margin']))


def independent_loss(zs,ev,cfg,rho,gamma):
    values=[]
    for z,t in zip(zs,ev):
        a=z.numpy().reshape(-1,2).astype(float);scores=a[t['s'],0]+a[t['e'],1]
        lp=scores-logsumexp(scores);at=t['top'];other=np.delete(lp,at)
        hinge=max(0,cfg['margin']-(lp[at]-other.max())) if len(other) else 0.
        kl=(np.exp(t['lp'])*(t['lp']-lp)).sum()
        values.append((rho*(-lp[at]+gamma*hinge)+cfg['beta']*kl,-lp[at],hinge,kl))
    return np.mean(values,axis=0)


def change(native,new):
    if native==new:return 'no_op'
    if new[0]<=native[0] and new[1]>=native[1]:return 'expansion'
    if new[0]>=native[0] and new[1]<=native[1]:return 'contraction'
    if new[1]-new[0]==native[1]-native[0]:return 'translation_equal_length'
    return 'shift_and_resize'


def main():
    torch.set_num_threads(2)
    p=plan();allrows=[r for rr in p['rows'].values() for r in rr]
    bykey={r['key']:r for r in allrows};gt=labels_for(allrows)
    data=read(OUT/'ALL_SOURCE_RESULTS.json');rows=data['rows'];assert len(rows)==113
    assert len({r['key'] for r in rows})==len({r['group'] for r in rows})==113
    for cohort,rr in p['rows'].items():
        a=[r for r in rr if r['f43_role']=='development'];b=[r for r in rr if r['f43_role']=='second_validation']
        assert len(a)==32 and len(b)==(25 if cohort=='hcstvg1_test' else 24)
        for field in ['group','source']:assert not {r[field] for r in a}&{r[field] for r in b}
        assert not {r['input']['video_sha256'] for r in a}&{r['input']['video_sha256'] for r in b}
    counts=collections.Counter();maxloss=0.;maxproj=0.;fitmeta={};cachemap={};indep={}
    for file in sorted((OUT/'fits').rglob('*.pt')):
        assert recorded(file);tr=load(file);r=bykey[tr['key']]
        if r['key'] not in cachemap:cachemap[r['key']]=cache_for(r)
        cache=cachemap[r['key']];kind=tr['kind']
        teacher=cache['wrong_teacher'] if kind=='wrong' else cache['rolled_teacher'] if kind=='shift' else cache['teacher']
        sig=(r['key'],'wrong' if kind=='wrong' else 'shift' if kind=='shift' else 'correct')
        if sig not in indep:indep[sig]=independent_teacher(cache['zero']['logits'],teacher,cache['records'],p['fixed'])
        iv,rho=indep[sig];ev=tr['evidence'];assert abs(rho-ev['rho'])<1e-12
        for a,t in zip(iv,ev['offsets']):
            error=max(np.max(abs(a['cost']-t['cost'].numpy())),np.max(abs(a['g']-t['score'].numpy())))
            maxproj=max(maxproj,float(error));assert error<1e-8
            assert a['top']==t['target'] and a['interval']==t['interval']
            assert abs(a['mad']-t['mad'])<1e-12
            assert np.max(abs(a['a']-t['a'].numpy()))<1e-12
            assert abs(sum(a['w'])-1)<1e-12
        active_rho=rho if tr['reliability'] else 1.
        for i,state in enumerate(tr['path']):
            lv,nll,hinge,kl=independent_loss(state['logits'],iv,p['fixed'],active_rho,tr['gamma'])
            error=max(abs(lv-state['loss']),abs(nll-state['parts']['nll']),abs(hinge-state['parts']['hinge']),abs(kl-state['parts']['kl']))
            maxloss=max(maxloss,float(error));assert error<1e-8
            g=[];drops=[]
            for pair,a,t in zip(state['raw_offset_indices'],iv,ev['offsets']):
                at=int(np.flatnonzero((a['s']==pair[0]) & (a['e']==pair[1]))[0])
                g.append(a['g'][at]);drops.append(a['lp'][t['native']]-a['lp'][at])
            assert abs(np.mean(g)-state['decision_score'])<1e-8
            assert state['guard']==all(d<=2.+1e-12 for d in drops)
            if i and not tr['path'][i-1]['accepted']:
                assert state['state_sha256']==tr['path'][i-1]['state_sha256']
                assert all(torch.equal(a,b) for a,b in zip(state['logits'],tr['path'][i-1]['logits']))
            counts['loss_states']+=1
        for budget,choices in tr['prefixes'].items():
            for selector,choice in choices.items():
                best=None
                for state in tr['path'][:int(budget)+1]:
                    if selector.startswith('guard') and not state['guard']:continue
                    value=state['decision_score'] if selector.endswith('decision') else -state['loss']
                    if best is None or value>best[0]+1e-12:best=(value,state['step'])
                assert best[1]==choice['best_step']
                assert state_hash(tr['states'][choice['state_sha256']])==choice['state_sha256']
                assert tr['path'][best[1]]['state_sha256']==choice['state_sha256']
                counts['prefix_state_choices']+=1
        assert tr['parameters']==66306 and tr['GT_online'] is False and tr['source_restored']
        counts['structured_fits']+=1;counts['actual_backwards']+=tr['backwards']
        counts['fit_milliseconds']+=round(1000*tr['seconds'])
        fitmeta[str(file)]=dict(path_states=len(tr['path']),accepted=sum(r.get('accepted',False) for r in tr['path']),
            guard_rejected_states=sum(not r['guard'] for r in tr['path']),
            target_matches_native=all(t['target_indices']==t['native_indices'] for t in ev['offsets']),
            reached_target_any=any(all(q==t['target_indices'] for q,t in zip(s['raw_offset_indices'],ev['offsets'])) for s in tr['path']),
            rho=rho,offset_iou=ev['offset_iou'])
    # Recompute all task scores from unchanged A4 boxes and stored physical endpoints.
    for r in rows:
        c=cachemap[r['key']]
        for name,interval in r['intervals'].items():
            indices=[c['frame_ids'].index(interval[0]),c['frame_ids'].index(interval[1]-1)]
            m,_=checked_score(c['T0']['boxes'],gt[r['key']],c['frame_ids'],indices)
            for metric in METRICS:
                a,b=m[metric],r['arms'][name][metric]
                assert (a is None and b is None) or abs(a-b)<1e-12
            counts['independent_task_conditions']+=1
        for name,ref in r['fit_references'].items():
            if ref.get('old'):continue
            tr=load(ref['path']);ch=tr['prefixes'][str(ref['budget'])][ref['selector']]
            assert tr['path'][ch['best_step']]['physical_interval']==r['intervals'][name]
            assert ch['best_step']==ref['best_step']
        assert len({v['sIoU'] for v in r['arms'].values()})==1
    for file in (OUT/'old_objective').rglob('*.pt'):
        assert recorded(file);t=load(file)
        counts['old_objective_fits']+=1;counts['actual_backwards']+=t['backwards']
    for r in [r for r in rows if r['role']=='second_validation']:
        file=OUT/'full_replay'/(r['key'].replace(':','_')+'.json')
        deadline=time.time()+1800
        while not file.exists():
            if time.time()>deadline:raise TimeoutError('Incomplete real model replay: '+str(file))
            time.sleep(5)
        x=read(file)
        assert set(x['audits'])=={'C','B_tuned','wrong_query','time_shift'}
        for a in x['audits'].values():
            assert a['full_model'] and a['source_restored'] and a['box_max_error']==a['logit_max_error']==0
            counts['full_model_replays']+=1
    # The final no-op receipts are written after a cohort's full forwards.
    deadline=time.time()+1800
    while sum(recorded(f) for f in (OUT/'noops').rglob('*.pt'))<4:
        if time.time()>deadline:raise TimeoutError('Incomplete zero-update controls')
        time.sleep(5)
    for f in (OUT/'noops').rglob('*.pt'):
        assert recorded(f);t=load(f);assert all(s['state_delta']==0 for s in t['path'])
        counts['noops']+=1;counts['actual_backwards']+=t['backwards']
    assert counts['noops']==4 and counts['full_model_replays']==196
    assert counts['structured_fits']+counts['old_objective_fits']+4<=p['caps']['trajectories']
    assert counts['actual_backwards']<=p['caps']['backwards']
    # Recheck that selection is exactly the predeclared development-only rule.
    selected=read(OUT/'SELECTION.json')
    for c in p['rows']:
        dev=read(OUT/'development'/f'{c}.json')
        for kind in ['B','R']:
            candidates=[x for x in dev if x['kind']==kind]
            for x in candidates:
                assert len(x['rows'])==32 and {y['key'] for y in x['rows']}=={r['key'] for r in p['rows'][c] if r['f43_role']=='development'}
                assert abs(x['mean_delta_v']-np.mean([y['delta_v'] for y in x['rows']]))<1e-12
            mv=max(x['mean_delta_v'] for x in candidates)
            ties=[x for x in candidates if mv-x['mean_delta_v']<=1e-12]
            pick=min(ties,key=lambda x:(x['config']!=p['default'],x['config']['steps'],abs(math.log10(x['config']['lr']/.001))))
            assert pick['config']==selected['choices'][c][kind]['config']
    names=list(rows[0]['arms']);comparisons=[(n,'T0') for n in names if n!='T0']+[
        ('A','A_raw'),('B_default','A'),('B_default','B_no_margin'),('C','B_match'),
        ('C','A'),('C','wrong_query'),('C','time_shift'),('R_loss','B_match'),
        ('R_guard_loss','R_loss'),('C','R_guard_loss'),('C','R_decision')]
    sums={};behavior={};cases=[];f41=read(F41/'EVAL_RESULTS.json');f41rows={r['key']:r for r in f41['rows']}
    for cohort in p['rows']:
        sums[cohort]={};behavior[cohort]={}
        for role in ['development','second_validation']:
            rr=[r for r in rows if r['cohort']==cohort and r['role']==role]
            s=summary(rr,names,METRICS,comparisons);cells,err=aggregate_check(rr,s)
            counts['statistic_cells']+=cells;sums[cohort][role]=s
            behavior[cohort][role]={}
            for name in names:
                changes=collections.Counter(change(r['intervals']['T0'],r['intervals'][name]) for r in rr)
                behavior[cohort][role][name]=dict(interval_changes=dict(changes),
                    actual_parameter_changed=sum(r['fit_references'].get(name,{}).get('state_delta',0)>0 for r in rr),
                    chosen_initial_state=sum(r['fit_references'].get(name,{}).get('best_step',-1)==0 for r in rr),
                    sources=len(rr))
            correct=[r['projection']['A'] for r in rr]
            behavior[cohort][role]['evidence']=dict(mean_rho=float(np.mean([v['rho'] for v in correct])),
                mean_offset_iou=float(np.mean([v['offset_iou'] for v in correct])),
                both_projected_native=sum(all(t['interval']==t['native_interval'] for t in v['offsets']) for v in correct),
                any_projected_outside_guard=sum(any(t['native_drop']>2 for t in v['offsets']) for v in correct),
                calibrated_changes_vs_raw=sum(r['intervals']['A']!=r['intervals']['A_raw'] for r in rr))
        for r in [x for x in rows if x['cohort']==cohort]:
            c=cachemap[r['key']];event=gt[r['key']]['interval'];aucs=[]
            for t in c['teacher']:
                y=np.array([event[0]<=i<event[1] for i in t['frame_ids']])
                aucs.append(float(roc_auc_score(y,t['raw_logits'])) if len(set(y))==2 else None)
            ref=r['fit_references']['C'];tr=load(ref['path']);path=[];cost_decomposition=[]
            for st in tr['path']:
                m,_=checked_score(c['T0']['boxes'],gt[r['key']],c['frame_ids'],st['indices'])
                path.append(dict(step=st['step'],interval=st['physical_interval'],loss=st['loss'],
                    decision_score=st['decision_score'],native_drops=st['native_drops'],guard=st['guard'],
                    state_delta=st['state_delta'],vIoU=m['vIoU_corrected'],tIoU=m['tIoU'],
                    selected=st['step']==ref['best_step']))
            for t in tr['evidence']['offsets']:
                frame=np.asarray(t['frame_ids']);ij=t['ij'].numpy();native=t['native'];star=t['target']
                # Offline display only: nearest legal physical boundaries to GT.
                distances=abs(frame[ij[0]]-event[0])+abs(frame[ij[1]]+1-event[1])
                near=int(distances.argmin())
                cost_decomposition.append(dict(native=t['native_interval'],projected=t['interval'],
                    nearest_GT_grid_interval=[int(frame[ij[0,near]]),int(frame[ij[1,near]])+1],
                    boundary_mapping_error=int(distances[near]),
                    D_native=float(t['cost'][native]),D_projected=float(t['cost'][star]),D_nearest_GT=float(t['cost'][near]),
                    native_logp_loss_nearest_GT=float(t['logp0'][native]-t['logp0'][near]),
                    G_native=float(t['score'][native]),G_nearest_GT=float(t['score'][near]),
                    teacher_mean_raw=float(expit(t['raw_logits'].numpy()).mean()),teacher_mean_standardized=float(t['a'].mean()),
                    note='offline cost diagnosis, not GT training, correction or online candidate choice'))
            record=dict(key=r['key'],cohort=cohort,role=r['role'],source=r['source'],caption=r['caption'],
                GT_interval=event,event_fraction=(event[1]-event[0])/(r['frame_ids'][-1]+1-r['frame_ids'][0]),
                delta_v=r['arms']['C']['vIoU_corrected']-r['arms']['T0']['vIoU_corrected'],
                intervals=r['intervals'],metrics=r['arms'],actionness_auc_by_offset=aucs,
                evidence=r['projection'],fit_references=r['fit_references'],
                actual_C_path=str(r['fit_references']['C']['path']),trajectory=path,cost_decomposition=cost_decomposition,
                GT_online=False)
            if r['key'] in f41rows:record['F41_hpo_metrics']=f41rows[r['key']]['arms']['T1']
            cases.append(record)
    length_sums={}
    for cohort in p['rows']:
        length_sums[cohort]={}
        def length_group(f):
            return 'short_0_.2' if f<=.2 else 'middle_.2_.6' if f<=.6 else 'long_.6_1' if f<=1 else 'longer_than_observed_grid'
        for label in ['short_0_.2','middle_.2_.6','long_.6_1','longer_than_observed_grid']:
            keys={r['key'] for r in cases if r['cohort']==cohort and r['role']=='second_validation' and length_group(r['event_fraction'])==label}
            rr=[r for r in rows if r['key'] in keys]
            if rr:length_sums[cohort][label]=summary(rr,['T0','A','B_tuned','C'],METRICS,[('C','T0'),('A','T0')])
    write(OUT/'SUMMARY.json',sums);write(OUT/'BEHAVIOR.json',behavior);write(OUT/'LENGTH_GROUPS.json',length_sums)
    write(OUT/'CASE_ANALYSIS.json',dict(cases=sorted(cases,key=lambda r:r['delta_v']),
        selection_rule='all cases kept; descriptive sorting by C delta_v, no online GT selection',GT_online=False))
    write(OUT/'FIT_DIAGNOSTICS.json',fitmeta)
    audit=dict(status='pass',counts=dict(counts),maximum_independent_loss_error=maxloss,maximum_projection_error=maxproj,
        sources=113,development_sources=64,second_validation_sources=49,independent_test=False,
        teacher_frozen=True,GT_online=False,A4_changed=False,new_DINO=0,new_space_fits=0,
        wrong_query_weak_control=True,corrected_evaluator_checked=True,source_split_checked=True,
        protected_files=len(p['protected_pins']),full_model_max_box_error=0,full_model_max_logit_error=0)
    for f,h in p['protected_pins'].items():assert sha(ROOT/f)==h,f
    tests=['tests/test_structured_temporal_v1.py','tests/test_dense_support_temporal_v1.py',
           'tests/test_dense_support_tuning_v1.py','tests/test_dense_support_prefix_v1.py']
    tests=[f for f in tests if (ROOT/f).exists()]
    result=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','-m','pytest','-q',*tests],cwd=ROOT,capture_output=True,text=True)
    write(OUT/'TEST_RECEIPT.json',dict(command=result.args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    assert result.returncode==0,result.stdout+result.stderr
    audit['test_receipt']=str(OUT/'TEST_RECEIPT.json');write(OUT/'AUDIT.json',audit)
    # Durable, fully source-backed concise tables. Interpretation added separately after inspection.
    lines=['# F43：结构化可靠时间适应——已完成有界实验','',
        '全部空间框来自精确A4；仅时间网络改动。32开发来源/方向选参，HC25/Vid24第二级验证；全部历史暴露，不是全量、未见测试或SOTA证明。',
        '','A为无参数投影；B/C为真实时间网络候选。C允许按无标签决策规则保留初始状态，实际更新数另报。', '']
    for c,roles in sums.items():
        lines+=['## '+c,'',f"开发锁定：B `{selected['choices'][c]['B']}`；C `{selected['choices'][c]['R']}`。",'']
        for role,s in roles.items():
            lines += [f"### {role}：{s['sources']}来源/{s['queries']}查询",'',
                '| 条件 | vIoU % | 固定sIoU % | tIoU % | Δv pp | 95%源配对CI pp |','|---|---:|---:|---:|---:|---|']
            for n in names:
                v=[100*s['arms'][n][m]['mean'] for m in METRICS[:3]]
                if n=='T0':dv=0.;ci=[0.,0.]
                else:
                    d=s['contrasts'][n+' - T0']['vIoU_corrected'];dv=100*d['mean'];ci=[100*x for x in d['ci95']]
                lines.append(f"| {n} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} | {dv:+.3f} | [{ci[0]:+.3f},{ci[1]:+.3f}] |")
            lines+=['',f"实际C行为：`{behavior[c][role]['C']}`。",f"投影诊断：`{behavior[c][role]['evidence']}`。",'']
        if c in f41['summary']:
            old=f41['summary'][c]['all']['arms']['T1']['vIoU_corrected']['mean']
            lines += [f'同49来源面板的旧F41充分调参时间参照 vIoU={100*old:.3f}%。该旧参照预算/目标不同，不是本轮单因素消融。','']
    lines += ['## 核验与边界','',f'独立数学、状态、任务评分和来源统计审计：`{audit}`。',
        '', '无新空间训练/专家请求；真实模型重插回不等于重新训练backbone。MAD不是标签校准，offset一致不是语义正确率，native guard不是任务安全保证。',
        '各表CI为预定来源bootstrap，无搜索多重校正；不含独立测试/不同训练seed不确定性。跨历史多轮可见数据只支持开发/机制结论。',
        '原生产与A4工作注册保持不变。本报告不以CI跨零判普遍失败，也不把参数变化或loss下降当效用提升。']
    (OUT/'TEMPORAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'CONFIG.json',dict(fixed=p['fixed'],default=p['default'],selection=selected,spatial=p['spatial'],method=str(OUT/'METHOD.md')))
    status(OUT/'STATUS.json',dict(stage='audited_pending_interpretation',finished=False))
    print(json.dumps({'audit':audit,'validation':{c:{n:100*v['vIoU_corrected']['mean'] for n,v in s['second_validation']['arms'].items()} for c,s in sums.items()}},indent=2),flush=True)


if __name__=='__main__':main()
