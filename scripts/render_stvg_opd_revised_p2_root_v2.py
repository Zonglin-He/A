"""Postseal complete P2 contrasts and six saved feedback/RGB diagnostic cases."""
import collections
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,PUB,read,write,sha
activate()
from scripts.decota_matrix_common_v1 import load,status
from scripts.stvg_opd_paper_later_common_v1 import stage_definition,record_path
from scripts.render_stvg_opd_revised_p1_root_v2 import coordinates,overlap,private_view
DEST=BASE/'P2_actual_root_signal_views';OUT=PUB/'P2'


def signal(fit,row,truth):
    import numpy as np
    from vg_tta.tastvg_oracle_event5_v1 import box_iou
    from vg_tta.decota_spatial_opd_action_readback_precision005 import FIELD
    chain=[];checks=0;action_origins=collections.Counter()
    for k,rd in enumerate(fit['rounds']):
        if FIELD in fit:
            actions=fit[FIELD]['trace'][k]['actions'].numpy();origin='same saved original GPU actions'
        else:
            raw=rd['rollout']['samples'].numpy().astype(np.float64)
            actions=np.empty_like(raw);positive=raw>=0
            actions[positive]=1/(1+np.exp(-raw[positive]));v=np.exp(raw[~positive]);actions[~positive]=v/(1+v)
            origin='offline float64 sigmoid of immutable sample logits; not a new production action'
        action_origins[origin]+=1;frames=[]
        for j,pos in enumerate(fit['positions']):
            fid=row['frame_ids'][pos]
            if fid not in truth:continue
            px=coordinates(actions[j],row['input']['width'],row['input']['height']);quality=overlap(px,truth[fid])
            assert np.max(np.abs(quality-box_iou(px,truth[fid])))<2e-12;checks+=len(quality)
            weights=rd['rollout']['weights'][j].numpy().astype(np.float64)
            frames.append(dict(sample_best_gt_iou=float(quality.max()),sample_mean_gt_iou=float(quality.mean()),
                teacher_weighted_gt_iou=float(weights@quality),teacher_vs_uniform_gt_iou=float(weights@quality-quality.mean()),
                central_before_gt_iou=float(overlap(coordinates(fit['path'][k]['boxes'][pos].numpy(),row['input']['width'],row['input']['height']),truth[fid])),
                central_after_gt_iou=float(overlap(coordinates(fit['path'][k+1]['boxes'][pos].numpy(),row['input']['width'],row['input']['height']),truth[fid])),
                central_expert_IoU_before=float(rd['central_reward_before'][j]),central_expert_IoU_after=float(rd['central_reward_after'][j]),
                ESS=float(rd['ESS'][j]),weight_max=float(rd['weight_max'][j])))
        if frames:chain.append(dict(round=k+1,GT_evaluable_admitted_positions=len(frames),
            **{f:float(np.mean([r[f] for r in frames])) for f in frames[0]}))
    assert len(chain)==fit['selected_step']
    return chain,checks,dict(action_origins)


def run():
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import read_row,unpack_expert
    torch.set_num_threads(2)
    for f,h in read(DEST/'RUNTIME.json')['pins'].items():assert sha(ROOT/f)==h,f
    complete=read(BASE/'P2_actual_root_readback/COMPLETION.json')
    assert complete['actual_arrivals']==7168 and complete['status']=='pending_actual_root_signal_cases_view_publication'
    for f,h in complete['outputs'].items():assert sha(ROOT/f)==h
    assert read(BASE/'P2_PREDICTION_BARRIER.json')['all_deployment_arms_and_directions']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    stats=read(OUT/'ROOT_STATISTICS.json');fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,ds in zip(axes,['hc2','vidstg']):
        y=0;labels=[]
        for name,result in stats['Full_minus_controls_condition_mean'].items():
            if ds not in name:continue
            for arm in ['direct_L1_GIoU','shuffled_feedback','frozen_rollout']:
                m=result[arm]['metrics']['Full_minus_v'];lo,hi=np.asarray(m['ci95'])*100;value=m['mean']*100
                ax.errorbar(value,y,xerr=[[max(0,value-lo)],[max(0,hi-value)]],fmt='o',capsize=3,color='#206c99')
                labels.append(name.replace('P2_'+ds+'_','')+' / '+arm);y+=1
        ax.set_yticks(range(y),labels);ax.axvline(0,color='#777');ax.grid(alpha=.15)
        ax.set_xlabel('Full − matched control vIoU (pp), parent 95% CI');ax.set_title(ds)
    fig.savefig(OUT/'actual_paired_control_contrasts.png',dpi=180);fig.savefig(OUT/'actual_paired_control_contrasts.pdf');plt.close(fig)
    selection=read(OUT/'ACTUAL_ROOT_CASE_SELECTION.json')['records'];assert len(selection)==6
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True);views=[];records=[];checks=0
    labels=[('central_after_gt_iou','Central vs GT','#c54942','-'),('central_expert_IoU_after','Central vs DINO','#7855a3','-'),
        ('teacher_weighted_gt_iou','Teacher-weighted samples vs GT','#14897d','-'),('sample_mean_gt_iou','Uniform samples vs GT','#888','--'),
        ('sample_best_gt_iou','Best sample vs GT (offline)','#b58a24',':')]
    private=DEST/'private_cases';private.mkdir(parents=True,exist_ok=True)
    for col,ds in enumerate(['hc2','vidstg']):
        chosen=[r for r in selection if r['dataset']==ds];stage=stage_definition(chosen[0]['stage'])
        truth,spans,provenance=truths(ds,stage)
        for band,case in enumerate(chosen):
            barrier=read(BASE/'stages'/case['stage']/'PREDICTION_BARRIER.json')
            path,binding=record_path(barrier,case['condition'],case['order'],case['arm'],case['arrival'])
            assert sha(path)==binding['sha256'];z=load(path);fit=z['fit'];inp=load(BASE/z['input']['path'])
            assert sha(BASE/z['input']['path'])==z['input']['sha256'];row=read_row(ds,case['query_ordinal'])
            chain,n,origins=signal(fit,row,truth[case['query_ordinal']]);checks+=n
            records.append(dict(case,rounds=chain,action_representation=origins,
                best_sample_is_offline_not_a_deployment_output=True,conditioning='GT-evaluable admitted observations only'))
            ax=axes[band,col]
            for key,label,color,style in labels:ax.plot([v['round'] for v in chain],[v[key] for v in chain],label=label,color=color,ls=style)
            ax.set_ylim(0,1);ax.grid(alpha=.15);ax.set_xlabel('Configured update round');ax.set_ylabel('Frame IoU')
            ax.set_title(f'{ds}: {case["kind"].replace("_"," ")}\ncurrent {case["delta_current_v"]*100:+.2f} pp; inherited {case["delta_inherited_v"]*100:+.2f} pp')
            views.append(private_view(ds,case,fit,inp,row,truth[case['query_ordinal']],unpack_expert(inp['expert']),private))
            status(DEST/'STATUS.json',dict(status='running',stage='actual_saved_signal_and_RGB_case_render',done=len(views),total=6,
                worker_pid=os.getpid(),CPU_only=True,model_calls=0,time=time.time()))
            print('ROOT_P2_SIGNAL_CASE',ds,case['kind'],len(views),6,flush=True)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,fontsize=8)
    fig.savefig(OUT/'actual_case_signal_chains.png',dpi=180);fig.savefig(OUT/'actual_case_signal_chains.pdf');plt.close(fig)
    write(OUT/'ACTUAL_ROOT_CASE_SIGNAL_CHAINS.json',dict(status='pass',records=records,cases=6,
        independent_sample_GT_IoU_checks=checks,all_cases_posthoc_after_complete_population=True,
        RGB_query_raw_logits_GT_boxes_or_actions_exported=False,model_calls=0,time=time.time()))
    write(DEST/'PRIVATE_CASE_VIEWS.json',dict(status='rendered_pending_actual_root_view',records=views,cases=6,
        private_media_exported=False,GT_after_global_seal=True,time=time.time()))
    write(DEST/'COMPLETION.json',dict(status='pending_actual_root_view_publication',cases=6,model_calls=0,CPU_only=True,
        outputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'ACTUAL_ROOT_CASE_SIGNAL_CHAINS.json',OUT/'actual_case_signal_chains.png',
            OUT/'actual_case_signal_chains.pdf',OUT/'actual_paired_control_contrasts.png',OUT/'actual_paired_control_contrasts.pdf',DEST/'PRIVATE_CASE_VIEWS.json']},time=time.time()))
    status(DEST/'STATUS.json',dict(status='pending_actual_root_view_publication',cases=6,CPU_only=True,paper_suite_complete=False,time=time.time()))


if __name__=='__main__':
    try:run()
    except BaseException:
        p=DEST/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc());raise
