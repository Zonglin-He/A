"""Independent all-source greedy-chain/bootstrap audit and actual sensitivity plots."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,csv,gzip,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_hc2_coordinate_common_v1 import *
import numpy as np

def independent(rows):
    assert len(rows)==64 and len({(r['source_id'],r['order']) for r in rows})==64
    sources=sorted({r['source_id'] for r in rows});orders=sorted({r['order'] for r in rows});assert len(sources)==32 and len(orders)==2
    matrix=np.array([[sum(r[f] for r in rows if r['source_id']==s)/2 for f in FIELDS] for s in sources])
    rng=np.random.default_rng(20261006);boots=[]
    for _ in range(200):boots.append(matrix[rng.integers(0,32,(50,32))].mean(axis=1))
    ci=np.quantile(np.concatenate(boots),[.025,.975],axis=0)
    return matrix,{f:dict(mean=float(matrix[:,j].mean()),ci95=ci[:,j].tolist(),
        query_macro=float(sum(r[f] for r in rows)/64),
        harm_gt5pp_sources=int((matrix[:,j]<-.05).sum()) if f.startswith('delta_') else None,
        harm_gt20pp_sources=int((matrix[:,j]<-.2).sum()) if f.startswith('delta_') else None) for j,f in enumerate(FIELDS)}

def run():
    verify();selection=read(BASE/'SELECTION_BARRIER.json');assert selection['status']=='all_five_coordinates_locked'
    if (BASE/'ROOT_AUDIT_COMPLETION.json').exists():return
    PUB.mkdir(parents=True,exist_ok=True);allrows={};audits={};checks=0;reference=None;invalid=[]
    for p in sorted((BASE/'trials').glob('cfg_*/CONFIG.json')):
        uid=p.parent.name
        if (p.parent/'INVALID_DISPOSITION.json').exists():
            invalid.append(dict(trial=uid,**read(p.parent/'INVALID_DISPOSITION.json')));continue
        rc=read(p.parent/'CPU_COMPLETION.json');out=PUB/'trials'/uid
        assert sha(out/'ROWS.jsonl.gz')==rc['rows_sha256'] and sha(out/'SUMMARY.json')==rc['summary_sha256']
        assert rc['GT_exposure_time']>=rc['coordinate_barrier_time']
        with gzip.open(out/'ROWS.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
        allrows[uid]=rows;matrix,stats=independent(rows);summary=read(out/'SUMMARY.json')
        for field in FIELDS:
            a=stats[field];b=summary['statistics']['metrics'][field]
            for key in ['mean','query_macro']:
                assert abs(a[key]-b[key])<3e-12;checks+=1
            assert np.max(np.abs(np.array(a['ci95'])-b['ci95']))<3e-12;checks+=2
            if field.startswith('delta_'):
                assert a['harm_gt5pp_sources']==b['harm_gt5pp_sources'] and a['harm_gt20pp_sources']==b['harm_gt20pp_sources'];checks+=2
        frozen={(r['source_id'],r['order'],r['query_ordinal'],r['arrival']):(r['Frozen_v'],r['Frozen_t'],r['Frozen_s']) for r in rows}
        if reference is None:reference=frozen
        else:assert frozen==reference
        for r in rows:
            assert abs(r['delta_total_v']-r['delta_current_v']-r['delta_inherited_v'])<3e-15
            assert r['Frozen_t']==r['Before_t']==r['After_t'];checks+=2
        tails=[dict(source_id=s,**{field:float(matrix[j,k]) for k,field in enumerate(FIELDS)}) for j,s in enumerate(sorted({r['source_id'] for r in rows}))]
        audits[uid]=dict(config=summary['config'],metrics=stats,parents=tails,
            total_GPU_fit_seconds=summary['GPU_fit_seconds'],exact_history_reused=summary['exact_prediction_history_reused'],
            CPU_math_seconds=sum(r['compute']['CPU_math_seconds'] for r in rows),
            capture_seconds=sum(r['compute']['capture_seconds'] for r in rows),
            new_DINO_calls=sum(r['compute']['new_DINO_calls'] for r in rows),
            backward_steps=sum(r['backward_steps'] for r in rows),
            math_and_state_arrivals=summary['independent_math_state_arrivals'])
    current=START.copy();chain=[];prev=None;logical=0
    for i,k in enumerate(COORDINATES):
        c,barrier=coordinate_barrier(i);dest=BASE/'coordinates'/f'{i:02}_{k}';s=read(dest/'SELECTION.json')
        assert c['incumbent']==s['incumbent']==current and c['previous_selection_sha256']==s['previous_selection_sha256']==prev
        if i:assert c['time']>=read(BASE/'coordinates'/f'{i-1:02}_{COORDINATES[i-1]}'/'SELECTION.json')['time']
        expected=candidate_configs(current,k);assert [v['config'] for v in c['candidates']]==expected
        candidates=[]
        for v in c['candidates']:
            logical+=1
            if v['trial'] not in audits:candidates.append(dict(**v,status='unscored_numerical_invalid'));continue
            summary=read(PUB/'trials'/v['trial']/'SUMMARY.json')
            assert s['score_receipts'][v['trial']]==sha(BASE/'trials'/v['trial']/'CPU_COMPLETION.json')
            candidates.append(dict(**v,status='complete',rank=rank(summary,current,v['ordinal']),
                metrics=audits[v['trial']]['metrics']))
        winner=max([v for v in candidates if v['status']=='complete'],key=lambda v:v['rank'])
        assert winner['trial']==s['trial'] and winner['config']==s['config'];checks+=1
        assert s['time']>=barrier['time'];prev=sha(dest/'SELECTION.json');current=s['config']
        chain.append(dict(coordinate=k,incumbent=c['incumbent'],selected=current,trial=s['trial'],candidates=candidates))
    assert current==selection['config'] and logical==26 and len(audits)+len(invalid)<=22
    final=audits[selection['selected_trial']];baseline=audits[trial_id(START)]
    pairs=[]
    # Same parents/orders, resample paired final-minus-start effects independently.
    keys=lambda rr:{(r['source_id'],r['order']):r for r in rr}
    x=keys(allrows[selection['selected_trial']]);y=keys(allrows[trial_id(START)])
    for key in x:
        pairs.append(dict(source_id=key[0],order=key[1],**{f:x[key][f]-y[key][f] for f in FIELDS}))
    _,vs_start=independent(pairs)
    write(PUB/'ROOT_AUDIT.json',dict(status='pass',independent_statistical_and_chain_checks=checks,
        all_32_sources_included=True,paired_source_bootstrap=10000,coordinate_chain=chain,trials=audits,
        invalid_trials=invalid,selected_config=current,selected_trial=selection['selected_trial'],
        final_minus_original_development_config=vs_start,development_selection=True,
        confirmation_or_P1_scores_used=False,independent_decoder_Jacobian_replay=False,time=time.time()))
    write(PUB/'SELECTED_CONFIG.json',dict(dataset='hc2',config=current,trial=selection['selected_trial'],
        status='selected_on_historically_exposed_32_parent_development_two_orders',VidSTG_unchanged=True,
        independent_confirmation=False,metrics=final['metrics'],time=time.time()))
    write(PUB/'INVALID_TRIALS.json',invalid)
    with (PUB/'ALL_CANDIDATES.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['coordinate','trial','value','status','delta_v_pp','current_pp','inherited_pp','harm5','harm20']);writer.writeheader()
        for c in chain:
            for v in c['candidates']:
                row=dict(coordinate=c['coordinate'],trial=v['trial'],value=v['config'][c['coordinate']],status=v['status'])
                if v['status']=='complete':
                    m=v['metrics'];row.update(delta_v_pp=m['delta_total_v']['mean']*100,current_pp=m['delta_current_v']['mean']*100,
                        inherited_pp=m['delta_inherited_v']['mean']*100,harm5=m['delta_total_v']['harm_gt5pp_sources'],harm20=m['delta_total_v']['harm_gt20pp_sources'])
                writer.writerow(row)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,5,figsize=(17,4.2),constrained_layout=True)
    for ax,c in zip(axes,chain):
        for field,color,label in [('delta_total_v','#2563eb','Total'),('delta_current_v','#d97706','Current'),('delta_inherited_v','#059669','Inherited')]:
            valid=[(j,v) for j,v in enumerate(c['candidates']) if v['status']=='complete'];xx=[j for j,v in valid];yy=[v['metrics'][field]['mean']*100 for j,v in valid]
            ax.plot(xx,yy,'o-',color=color,label=label)
            if field=='delta_total_v':
                low=[v['metrics'][field]['ci95'][0]*100 for j,v in valid];high=[v['metrics'][field]['ci95'][1]*100 for j,v in valid];ax.fill_between(xx,low,high,color=color,alpha=.12)
        selected=[j for j,v in enumerate(c['candidates']) if v['trial']==c['trial']][0];ax.axvline(selected,color='#111827',linestyle=':',alpha=.5)
        for j,v in enumerate(c['candidates']):
            if v['status']!='complete':ax.text(j,0,'invalid',rotation=90,color='#dc2626',ha='center',va='bottom')
        ax.axhline(0,color='#9ca3af',lw=.7);ax.set_xticks(range(len(c['candidates'])),[str(v['config'][c['coordinate']]) for v in c['candidates']],rotation=45)
        ax.set_title(c['coordinate']);ax.set_ylabel('Development delta vIoU (pp)');ax.grid(alpha=.15)
    axes[0].legend(fontsize=8);fig.suptitle('HC2 sequential coordinate search: 32 exposed development parents / two orders')
    for ext in ['png','pdf']:fig.savefig(PUB/('HC2_coordinate_sensitivity.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    names=['Start']+[c['coordinate'] for c in chain];uids=[trial_id(START)]+[c['trial'] for c in chain]
    for field,color,label in [('delta_total_v','#2563eb','Total'),('delta_current_v','#d97706','Current'),('delta_inherited_v','#059669','Inherited')]:
        axes[0].plot(range(6),[audits[u]['metrics'][field]['mean']*100 for u in uids],'o-',label=label,color=color)
    axes[0].set_xticks(range(6),names,rotation=45);axes[0].set_ylabel('Development delta vIoU (pp)');axes[0].legend(fontsize=8);axes[0].axhline(0,color='grey',lw=.7)
    for uid,label,color in [(trial_id(START),'Start','#94a3b8'),(selection['selected_trial'],'Selected','#2563eb')]:
        values=sorted(p['delta_total_v']*100 for p in audits[uid]['parents']);axes[1].plot(range(32),values,label=label,color=color)
    axes[1].axhline(-5,color='#d97706',ls=':');axes[1].axhline(-20,color='#dc2626',ls=':');axes[1].set_ylabel('All parent effects (pp)');axes[1].set_xlabel('Sorted parent rank');axes[1].legend(fontsize=8)
    valid=sorted(audits);seconds=[audits[u]['total_GPU_fit_seconds']/64 for u in valid];effects=[audits[u]['metrics']['delta_total_v']['mean']*100 for u in valid]
    axes[2].scatter(seconds,effects,color='#64748b');axes[2].scatter(final['total_GPU_fit_seconds']/64,final['metrics']['delta_total_v']['mean']*100,color='#dc2626',marker='*',s=140)
    axes[2].set_xlabel('Measured GPU adaptation seconds / arrival');axes[2].set_ylabel('Development delta vIoU (pp)')
    fig.suptitle('Greedy locking, complete negative tail, and real adaptation cost (development selection)')
    for ext in ['png','pdf']:fig.savefig(PUB/('HC2_coordinate_chain_tail_cost.'+ext),dpi=180)
    plt.close(fig)
    m=final['metrics'];a=m['delta_total_v'];b=m['delta_current_v'];old=baseline['metrics']['delta_total_v']
    text=f'''# HC2 顺序坐标调参实际结果\n\n最终配置：`{json.dumps(current)}`。按 lr→sigma→tau→steps→writeback 一轮顺序锁定；26次逻辑候选、{len(audits)}套完整配置、{len(invalid)}套数值无效配置。完整配置共{len(audits)*64}条开发到达，原32父来源×2顺序全部纳入。\n\n选定配置开发 parent-macro ΔvIoU={a['mean']*100:.6f}pp，10000 paired source-bootstrap 95% CI=[{a['ci95'][0]*100:.6f},{a['ci95'][1]*100:.6f}]pp；current={b['mean']*100:.6f}pp，inherited={m['delta_inherited_v']['mean']*100:.6f}pp。原起点开发 ΔvIoU={old['mean']*100:.6f}pp。负尾超过5/20pp的来源为{a['harm_gt5pp_sources']}/{a['harm_gt20pp_sources']}，全部来源和负结果在 ROOT_AUDIT 与 ALL_CANDIDATES 中保留。\n\n独立统计/链路核验{checks}项通过；每条完整预测已复核 Gaussian likelihood、detached reward、Adam 算术、query 重置、1792参数状态链接、LN写回和末轮输出，以及官方和独立 dense v/t 指标。未独立重放解码器 Jacobian。\n\n数据是历史曝光的 HC2 validation 开发来源，所得分数有选参偏差；128来源确认集和P1全量分数未参与此次选择。不能把开发置信区间称为独立确认，也不能以此宣称HC2稳定提高。仅一轮预设网格内条件最优，可能受参数次序与交互影响。\n\n成本分GPU适应、冻结输入capture和CPU数学审计；复用精确旧输入/历史配置的成本标明为原测量，不能声称冷启动端到端速度。VidSTG参数和算法结构保持原样，EATA暂停。旧P1 685条原预测保持；参数变化后在独立修订目录重新锁定从源模型开始的HC2完整流，不拼接不同参数的prefix。\n'''
    (PUB/'ROOT_REVIEW.md').write_text(text)
    write(BASE/'ROOT_AUDIT_COMPLETION.json',dict(status='pending_actual_root_visual_publication_registration',
        statistics_and_chain='pass',independent_checks=checks,complete_configurations=len(audits),
        numerical_invalid_configurations=len(invalid),logical_candidates=logical,
        unique_complete_arrivals=len(audits)*64,plots=[str((PUB/name).relative_to(ROOT)) for name in ['HC2_coordinate_sensitivity.png','HC2_coordinate_chain_tail_cost.png']],
        public_files={str(p.relative_to(ROOT)):sha(p) for p in PUB.rglob('*') if p.is_file()},time=time.time()))
    print('HC2_COORDINATE_ROOT_AUDIT_COMPLETE',current,checks,flush=True)

if __name__=='__main__':run()
