"""Postseal anonymous sample/teacher/central-task chains for actual P0 cases."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *
import numpy as np

def run():
    assert read(BASE/'P0_CPU_COMPLETION.json')['P0_postseal_scoring_complete']
    assert read(BASE/'P0_PREDICTION_BARRIER.json')['all_deployment_arms_both_directions']
    cases=read(PUB/'P0_ROOT_CASES.json');design=read(BASE/'DESIGN_LOCK.json');output={}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    kinds=['success','low_expert_quality_failure','expert_reward_task_mismatch']
    for col,ds in enumerate(DATASETS):
        with gzip.open(PUB/('P0_'+ds)/'SAMPLE_DIAGNOSIS.jsonl.gz','rt') as f:
            records=[json.loads(line) for line in f]
        output[ds]={}
        for rowidx,kind in enumerate(kinds):
            case=cases[ds][kind];q=case['query_ordinal'];order=case['order']
            selected=[r for r in records if r['query_ordinal']==q and r['order']==order and r['arm']=='on_policy'];assert len(selected)==1
            stage=design['stages']['P0_'+ds];at=stage['orders'][order].index(q)
            payload=BASE/'stages'/('P0_'+ds)/'clean'/order/'on_policy'/f'{at:05}.pt'
            assert sha(payload)==read(payload.with_suffix('.json'))['sha256']
            fit=load(payload)['fit'];details=selected[0]['details'];rounds=[]
            for k,rd in enumerate(fit['rounds']):
                rr=[v for v in details if v['round']==k]
                if not rr:continue
                js=[fit['positions'].index(v['position']) for v in rr]
                rounds.append(dict(round=k+1,GT_evaluable_observed_positions=len(rr),
                    sample_best_GT_IoU=float(np.mean([v['sample_best_gt_iou'] for v in rr])),
                    uniform_sample_GT_IoU=float(np.mean([v['sample_mean_gt_iou'] for v in rr])),
                    teacher_weighted_GT_IoU=float(np.mean([v['teacher_weighted_gt_iou'] for v in rr])),
                    central_before_GT_IoU=float(np.mean([v['central_before_gt_iou'] for v in rr])),
                    central_after_GT_IoU=float(np.mean([v['central_after_gt_iou'] for v in rr])),
                    central_expert_IoU_before=float(rd['central_reward_before'][js].mean()),
                    central_expert_IoU_after=float(rd['central_reward_after'][js].mean()),
                    ESS=float(np.mean([v['ESS'] for v in rr])),
                    reward_variance=float(np.mean([v['reward_variance'] for v in rr]))))
            output[ds][kind]={**case,'rounds':rounds,'all_chains_conditioned_on_GT_evaluable_admitted_positions':True,
                'case_selection':'deterministic_after_full_stage_retention','identity_confusion_not_assumed_from_low_IoU':True}
            ax=axes[rowidx,col]
            if rounds:
                xs=[r['round'] for r in rounds]
                for key,label,color,style in [
                    ('central_after_GT_IoU','Central output vs GT','#b52b3b','-'),
                    ('central_expert_IoU_after','Central output vs expert','#7952b3','-'),
                    ('teacher_weighted_GT_IoU','Teacher-weighted samples vs GT','#168b80','-'),
                    ('uniform_sample_GT_IoU','Uniform samples vs GT','#999999','--'),
                    ('sample_best_GT_IoU','Best sampled box vs GT (offline)','#b78a22',':')]:
                    ax.plot(xs,[r[key] for r in rounds],label=label,color=color,ls=style,lw=1.5)
                ax.set_ylim(0,1)
            else:ax.text(.5,.5,'No GT-evaluable admitted frame',ha='center',transform=ax.transAxes)
            name={'success':'Strongest current correction','low_expert_quality_failure':'Low expert quality + task harm','expert_reward_task_mismatch':'Expert agreement gain + task harm'}[kind]
            ax.set_title(f'{ds}: {name}\ncurrent {case["delta_current_v"]*100:+.2f} pp, inherited {case["delta_inherited_v"]*100:+.2f} pp')
            ax.set_xlabel('Configured update round');ax.set_ylabel('IoU (observed GT-evaluable frames)');ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
    fig.savefig(PUB/'P0_case_signal_chains.png',dpi=180);fig.savefig(PUB/'P0_case_signal_chains.pdf');plt.close(fig)
    write(PUB/'P0_CASE_SIGNAL_CHAINS.json',dict(status='postseal_actual_anonymous_chains',datasets=output,
        raw_actions_boxes_captions_video_exported=False,GT_only_offline_after_global_barrier=True,
        not_an_independent_case_efficacy_estimate=True))
    print('P0_ACTUAL_SIGNAL_CHAIN_RENDERED',sum(len(x) for x in output.values()))

if __name__=='__main__':run()
