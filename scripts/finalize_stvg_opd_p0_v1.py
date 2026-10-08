"""Independent scalar readback, parent-paired statistics, tails and actual P0 gate."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *
import numpy as np

def rows_for(ds):
    with gzip.open(PUB/('P0_'+ds)/'ROWS.jsonl.gz','rt') as f:
        return [json.loads(line) for line in f]

def independent(rows,fields):
    # A different grouping/reduction implementation from the scorer.
    groups=collections.defaultdict(list)
    for r in rows:groups[r['source_id']].append(r)
    ordered=sorted(groups)
    matrix=np.stack([np.add.reduce(np.array([[r[f] for f in fields] for r in groups[s]],float))/len(groups[s]) for s in ordered])
    assert len(ordered)==128 and all(len(groups[s])==2 for s in ordered)
    rng=np.random.default_rng(20261006)
    samples=np.concatenate([matrix[rng.integers(0,128,size=(50,128))].sum(axis=1)/128 for _ in range(200)])
    ci=np.percentile(samples,[2.5,97.5],axis=0)
    return matrix,dict(zip(fields,[dict(mean=float(matrix[:,j].sum()/128),ci95=ci[:,j].tolist(),
        query_macro=float(sum(r[f] for r in rows)/len(rows)),
        harm_gt5pp_sources=int(sum(x<-.05 for x in matrix[:,j])) if f.startswith('delta_') else None,
        harm_gt20pp_sources=int(sum(x<-.2 for x in matrix[:,j])) if f.startswith('delta_') else None)
        for j,f in enumerate(fields)])),ordered

def plot(allrows,statistics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for ax,ds in zip(axes,DATASETS):
        st=statistics[ds]['on_policy'];names=['delta_total_v','on_policy_minus_shuffled_v','on_policy_minus_frozen_rollout_v']
        for j,name in enumerate(names):
            m=st[name];mean=m['mean']*100;lo,hi=np.array(m['ci95'])*100
            ax.errorbar(j,mean,yerr=np.array([[mean-lo],[hi-mean]]),fmt='o',capsize=5,color=['#1261a0','#8c46a2','#bb7334'][j])
        ax.axhline(0,color='#777',lw=1);ax.set_xticks(range(3),['OPD − Frozen','OPD − Shuffled','OPD − Fixed\nrollout'])
        ax.set_title('VidSTG target' if ds=='vidstg' else 'HC2 validation target');ax.set_ylabel('Paired parent ΔvIoU (pp), 95% CI')
        ax.grid(axis='y',alpha=.2)
    fig.savefig(PUB/'P0_efficacy_mechanisms.png',dpi=180);fig.savefig(PUB/'P0_efficacy_mechanisms.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for col,ds in enumerate(DATASETS):
        rows=[r for r in allrows[ds] if r['arm']=='on_policy'];groups=collections.defaultdict(list)
        for r in rows:groups[r['source_id']].append(r)
        mat=np.array([[np.mean([r[f] for r in groups[s]])*100 for f in ['delta_inherited_v','delta_current_v','delta_total_v']] for s in sorted(groups)])
        axes[0,col].plot(np.arange(128),np.sort(mat[:,2]),color='#1261a0');axes[0,col].axhline(0,color='#777',lw=1)
        axes[0,col].axhline(-5,color='#c06',ls='--',lw=.8);axes[0,col].axhline(-20,color='#c06',ls=':',lw=.8)
        axes[0,col].set_title(('VidSTG' if ds=='vidstg' else 'HC2')+' target: all 128 parent effects');axes[0,col].set_ylabel('OPD − Frozen vIoU (pp)');axes[0,col].set_xlabel('Parents sorted by total effect')
        axes[1,col].scatter(mat[:,0],mat[:,1],s=22,c=mat[:,2],cmap='coolwarm',edgecolor='none',alpha=.8)
        axes[1,col].axhline(0,color='#777',lw=.8);axes[1,col].axvline(0,color='#777',lw=.8)
        axes[1,col].set_xlabel('Inherited gain (pp)');axes[1,col].set_ylabel('Current correction (pp)')
    fig.savefig(PUB/'P0_negative_tail_state.png',dpi=180);fig.savefig(PUB/'P0_negative_tail_state.pdf');plt.close(fig)

def run():
    verify();assert read(BASE/'P0_PREDICTION_BARRIER.json')['status']=='sealed'
    stats={};allrows={};audits={};table=[];gate={};checks=0;cases={}
    for ds in DATASETS:
        completion=read(BASE/'stages'/('P0_'+ds)/'CPU_COMPLETION.json')
        for f,h in completion['files'].items():assert sha(ROOT/f)==h
        summary=read(PUB/('P0_'+ds)/'SUMMARY.json');rows=rows_for(ds);allrows[ds]=rows
        assert len(rows)==768 and len({r['source_id'] for r in rows})==128
        assert {r['order'] for r in rows}=={'order1','order2'}
        stats[ds]={};audits[ds]=dict(status='pass',rows=len(rows),parent_sources=128,checks=0)
        for arm in ARMS:
            rr=[r for r in rows if r['arm']==arm];fields=list(summary['arms'][arm]['metrics'])
            _,m,parents=independent(rr,fields);stats[ds][arm]=m
            for f,v in m.items():
                original=summary['arms'][arm]['metrics'][f]
                assert abs(v['mean']-original['mean'])<2e-12
                assert np.max(np.abs(np.array(v['ci95'])-original['ci95']))<2e-12
                assert abs(v['query_macro']-original['query_macro'])<2e-12
                if f.startswith('delta_'):
                    assert v['harm_gt5pp_sources']==original['harm_gt5pp_sources'] and v['harm_gt20pp_sources']==original['harm_gt20pp_sources']
                checks+=5;audits[ds]['checks']+=5
            for r in rr:
                assert abs(r['delta_total_v']-r['delta_current_v']-r['delta_inherited_v'])<2e-12
                assert r['Frozen_t']==r['Before_t']==r['After_t']
                for t in [.3,.5]:
                    assert r['correct_to_wrong_'+str(t)]==int(r['Frozen_v']>t and r['After_v']<=t)
                    assert r['wrong_to_correct_'+str(t)]==int(r['Frozen_v']<=t and r['After_v']>t)
                checks+=6;audits[ds]['checks']+=6
            cost={key:float(np.mean([r['compute'][key] for r in rr])) for key in ['shared_capture_seconds','fit_GPU_seconds','CPU_math_seconds','new_DINO_calls','CUDA_peak_allocated','backward_steps']}
            audits[ds].setdefault('cost',{})[arm]=cost
            table.append(dict(dataset=ds,arm=arm,config=summary['config'],statistics=m,cost=cost))
        full=[r for r in rows if r['arm']=='on_policy']
        success=max(full,key=lambda r:r['delta_current_v'])
        confused=[r for r in full if r['admitted_expert_GT_IoU'] is not None and r['admitted_expert_GT_IoU']<.3 and r['delta_current_v']<0]
        wrong=min(confused or full,key=lambda r:r['delta_current_v'])
        mismatch=[r for r in full if r['central_expert_reward_round_mean_delta'] is not None and r['central_expert_reward_round_mean_delta']>0 and r['delta_current_v']<=0]
        proxy=min(mismatch or full,key=lambda r:r['delta_current_v'])
        cases[ds]={name:{k:r[k] for k in ['query_ordinal','source_id','order','delta_total_v','delta_current_v','delta_inherited_v','admitted_expert_GT_IoU','central_expert_reward_round_mean_delta','observed_iou_delta','unobserved_iou_delta']} for name,r in [('success',success),('low_expert_quality_failure',wrong),('expert_reward_task_mismatch',proxy)]}
        m=stats[ds]['on_policy'];gate[ds]=dict(delta_v=m['delta_total_v'],current=m['delta_current_v'],inherited=m['delta_inherited_v'],
            feedback_direction=m['on_policy_minus_shuffled_v'],on_policy_refresh=m['on_policy_minus_frozen_rollout_v'],
            stable_efficacy=m['delta_total_v']['ci95'][0]>0)
    advance=all(x['stable_efficacy'] for x in gate.values())
    write(PUB/'P0_ROOT_STATISTICS.json',dict(status='pass',independent_scalar_checks=checks,datasets=stats,source_bootstrap=10000,
        historical_exposure=True,target_development_selected=True,no_confirmation_retuning=True,time=time.time()))
    write(PUB/'P0_ROOT_CASES.json',cases)
    write(PUB/'P0_ROOT_AUDIT.json',dict(status='pass',independent_scalar_checks=checks,datasets=audits,statistics_and_thresholds_and_cost=True,time=time.time()))
    plot(allrows,stats)
    lines=['# Fixed-parameter OPD: locked 128-parent confirmation','',
        'Each target has 128 prelocked parent sources, one query each and two orders. These parents were excluded from the current OPD parameter search but have earlier project exposure. The configurations were selected on exposed target-development GT. This is locked confirmation, not fresh unseen evaluation.','',
        '| Target / arm | Frozen vIoU % | After vIoU % | ΔvIoU pp [95% paired parent CI] | Current pp | Inherited pp | Sources worse >5 / >20 pp |',
        '|---|---:|---:|---|---:|---:|---:|']
    for ds in DATASETS:
        for a in ARMS:
            m=stats[ds][a];v=m['delta_total_v'];lo,hi=np.array(v['ci95'])*100
            lines.append(f"| {ds} / {a} | {m['Frozen_v']['mean']*100:.3f} | {m['After_v']['mean']*100:.3f} | {v['mean']*100:+.3f} [{lo:+.3f}, {hi:+.3f}] | {m['delta_current_v']['mean']*100:+.3f} | {m['delta_inherited_v']['mean']*100:+.3f} | {v['harm_gt5pp_sources']} / {v['harm_gt20pp_sources']} |")
    lines+=['','Efficacy, feedback alignment and rollout refresh are separate comparisons. Native temporal readout is fixed: every Frozen/Before/After tIoU is identical. Source and query macro coincide here because each parent has one query and two complete orders. Full-source and official-query results still require P1.','',
        '| Target | OPD − Shuffled pp [95% CI] | OPD − Fixed rollout pp [95% CI] |','|---|---|---|']
    for ds in DATASETS:
        vals=[]
        for f in ['on_policy_minus_shuffled_v','on_policy_minus_frozen_rollout_v']:
            m=stats[ds]['on_policy'][f];lo,hi=np.array(m['ci95'])*100;vals.append(f"{m['mean']*100:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append('| '+ds+' | '+' | '.join(vals)+' |')
    lines+=['','## Prelocked P0 gate','',
        'Both targets establish positive paired parent efficacy; proceed to P1 without retuning.' if advance else
        'The prelocked both-direction stable-efficacy gate is not satisfied. Root must complete failure attribution before a decision on expensive P1. No configuration or roster changes are authorized by these results.',
        '', '## Audit and cost boundaries','',
        f'All 1536 adaptive fits retain full Gaussian actions/rewards/weights, gradients, Adam raw updates and all 1792 states. The CPU scorer independently recalculates support, reward/likelihood/Adam arithmetic and exact per-arm LN chains, then official and independent dense v/t readout. Root independently verified {checks} scalar/statistical/threshold/decomposition checks, including 10000 paired parent-bootstrap results. The decoder Jacobian is not independently reimplemented; arithmetic checks are not that stronger claim.',
        '', 'Capture is shared across matched arms and cached original DINO evidence is reused after exact input/native/interval verification. Actual new DINO calls during P0 can therefore be zero; the method still uses the nominal Uniform4 observations. Shared capture, decoder GPU fit and CPU audit are reported separately. These reused-input times are not a cold uncached end-to-end latency claim. All anonymous scalar rows and all negative tails are retained.',
        '', 'P0_ROOT_CASES contains deterministic scalar case choices after all outcomes were retained: strongest current correction, low-expert-quality harm, and positive expert-reward/no-current-task-gain mismatch. Identity confusion requires actual visual root inspection; low expert overlap alone does not prove identity confusion.',
        '', 'P1–P6 remain unexecuted at this P0 report. EATA missing direction/media/Fisher remains user paused. Existing baseline evaluation remains separately completed; old IoU-energy Ours is not the new OPD row.']
    (PUB/'P0_REPORT.md').write_text('\n'.join(lines)+'\n')
    write(BASE/'P0_GATE.json',dict(status='pass_to_P1' if advance else 'failure_attribution_before_P1_decision',
        datasets=gate,no_retuning=True,root_visual_inspection_and_publication_pending=True,time=time.time()))
    write(BASE/'P0_CPU_COMPLETION.json',dict(status='pending_actual_root_visual_and_publication',
        P0_postseal_scoring_complete=True,independent_scalar_checks=checks,P1_gate_pass=advance,
        files={str(p.relative_to(ROOT)):sha(p) for p in PUB.glob('P0*') if p.is_file()},time=time.time()))
    status(BASE/'CPU_STAGE.json',dict(status='pending_actual_root_visual_and_publication',P1_gate_pass=advance,time=time.time()))
    print('P0_CPU_ROOT_COMPLETE',advance,checks,flush=True)

if __name__=='__main__':run()
