"""Postseal P5 complete budget/paired contrast/decomposition/cost and six saved feedback/RGB diagnostic cases."""
import collections
import json
import os
from pathlib import Path
import sys
import time
import textwrap
import traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,PUB,read,write,sha
activate()
from scripts.decota_matrix_common_v1 import load,status
from scripts.stvg_opd_paper_later_common_v1 import stage_definition,record_path
from scripts.render_stvg_opd_revised_p1_root_v2 import coordinates,overlap
DEST=BASE/'P5_actual_root_signal_views';OUT=PUB/'P5'


from scripts.render_stvg_opd_revised_p2_root_v2 import signal


def private_view(ds, case, fit, inp, row, truth, expert, directory):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from scripts.run_decota_paper_main_v1 import frames_for
    from vg_tta.decota_fixed_full_audit_v1 import top1_support
    frames, ids, _ = frames_for(ds, row, collections.OrderedDict())
    assert ids == row['frame_ids']
    from scripts.stvg_opd_paper_inputs_v1 import observation
    frames,pixel,spec=observation(ds,row,case['condition'],frames)
    assert pixel==inp['pixel_sha256'] and spec==inp.get('corruption_spec'), 'Rendered physical RGB must match original captured input'
    support = dict(top1_support(expert))
    impact = []
    width, height = row['input']['width'], row['input']['height']
    for pos, fid in enumerate(ids):
        if fid not in truth:
            continue
        before = float(overlap(coordinates(fit['before'][pos].numpy(), width, height), truth[fid]))
        after = float(overlap(coordinates(fit['final'][pos].numpy(), width, height), truth[fid]))
        impact.append((after-before, pos, fid))
    selected = []
    for label, observed in [('admitted observation', True), ('unobserved frame', False)]:
        eligible = [v for v in impact if (v[1] in fit['positions']) == observed]
        if eligible:
            chosen = (max if case['kind']=='success' else min)(eligible, key=lambda v: (v[0], -v[2]))
            selected.append((label, chosen))
    assert selected, 'Chosen descriptive case must have a real GT-evaluable frame'
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    canvas = Image.new('RGB', (1800, 115 + len(selected)*515), 'white')
    draw = ImageDraw.Draw(canvas)
    caption = '\n'.join(textwrap.wrap(row['input']['caption'], width=135))
    draw.text((12, 8), f'{ds}: {case["kind"]} / {case["stage"]} | query {case["query_ordinal"]} | {case["order"]} | {case["condition"]}', font=big, fill='black')
    draw.multiline_text((12, 34), caption, font=font, fill='black')
    draw.text((12, 84), f'Current {case["delta_current_v"]*100:+.2f} pp; inherited {case["delta_inherited_v"]*100:+.2f} pp. '
              'Post-hoc illustrative frames; GT green, DINO cyan, student coral.', font=font, fill='black')
    chosen_records = []
    for band, (label, (_, pos, fid)) in enumerate(selected):
        scale = min(590/width, 420/height)
        w, h = round(width*scale), round(height*scale)
        y = 115 + band*515
        for col, (name, boxes) in enumerate([('Frozen', inp['native_boxes']), ('Before', fit['before']), ('After', fit['final'])]):
            x = col*600 + 5
            image = Image.fromarray(frames[pos]).resize((w, h))
            canvas.paste(image, (x, y+46))
            draw.text((x, y), f'{name} | {label} | physical frame {fid}', font=font, fill='black')
            student = coordinates(boxes[pos].numpy(), width, height)
            entries = [('GT', truth[fid], '#159947'), ('Student', student, '#f05245')]
            if pos in support:
                entries.insert(1, ('DINO Top1', coordinates(support[pos], width, height), '#10a8bb'))
            for tag, box, color in entries:
                rect = np.asarray(box, float)*scale
                rect[[0,2]] += x
                rect[[1,3]] += y+46
                draw.rectangle(rect.tolist(), outline='white', width=6)
                draw.rectangle(rect.tolist(), outline=color, width=3)
                draw.text((max(x, min(x+w-90, float(rect[0]))), max(y+46, float(rect[1])-18)), tag, font=font, fill=color,
                          stroke_width=1, stroke_fill='white')
            score = float(overlap(student, truth[fid]))
            draw.text((x, y+46+h+8), f'Student/GT frame IoU {score:.3f}; Native WHEN remains fixed.', font=font, fill='black')
        chosen_records.append(dict(scope=label, frame_id=fid, position=pos))
    file = directory/f'{ds}_{case["kind"]}.png'
    canvas.save(file)
    return dict(dataset=ds, kind=case['kind'], query_ordinal=case['query_ordinal'],
        path=str(file.relative_to(ROOT)), sha256=sha(file), frames=chosen_records,
        condition=case['condition'],actual_corrupted_RGB_sha256=pixel,original_input_pixel_sha256_matched=True,
        caption_for_private_identity_readback=row['input']['caption'], private_media_not_public=True)


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
    complete=read(BASE/'P5_actual_root_readback_revision004/COMPLETION.json')
    assert complete['actual_arrivals']==2560 and complete['status']=='pending_actual_root_signal_cases_view_publication'
    for f,h in complete['outputs'].items():assert sha(ROOT/f)==h
    assert read(BASE/'P5_PREDICTION_BARRIER.json')['all_deployment_arms_and_directions']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    import gzip
    from scripts.audit_stvg_opd_p5_public_v2 import budget_analysis,compare
    data={}
    from scripts.stvg_opd_paper_later_common_v1 import phases
    for name in phases()['P5']:
        with gzip.open(PUB/name/'ROWS.jsonl.gz','rt') as f:data[name]=list(map(json.loads,f))
    analysis=budget_analysis(data);compare(analysis,read(OUT/'ACTUAL_ROOT_BUDGET_CONTRASTS.json'),[0])
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True,sharey=True)
    for ax,ds in zip(axes,['hc2','vidstg']):
        for k in [1,2,4,8]:
            m=analysis['datasets'][ds][str(k)]['all_orders']['metrics']['delta_total_v'];lo,hi=np.asarray(m['ci95'])*100;mu=m['mean']*100
            ax.errorbar(k,mu,yerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt='o',color='#206c99',capsize=4)
        ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8],['1','2','4','8']);ax.axhline(0,color='#777');ax.grid(alpha=.15)
        ax.set_xlabel('Uniform observation budget K');ax.set_title(ds+' / 128 parents, two orders')
    axes[0].set_ylabel('After − Frozen vIoU (pp), paired 95% CI')
    fig.savefig(OUT/'actual_budget_effects.png',dpi=180);fig.savefig(OUT/'actual_budget_effects.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True,sharex=True)
    for ax,ds in zip(axes,['hc2','vidstg']):
        contrasts=analysis['pairwise_budget_contrasts'][ds]
        for y,(label,group) in enumerate(contrasts.items()):
            m=group['all_orders']['metrics']['v'];lo,hi=np.asarray(m['ci95'])*100;mu=m['mean']*100
            ax.errorbar(mu,y,xerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt='o',color='#206c99',capsize=3)
        ax.set_yticks(range(len(contrasts)),[v.replace('_',' ') for v in contrasts]);ax.invert_yaxis();ax.axvline(0,color='#777');ax.grid(alpha=.15)
        ax.set_xlabel('Paired After vIoU difference (pp), 95% CI');ax.set_title(ds)
    fig.savefig(OUT/'actual_budget_pairwise_contrasts.png',dpi=180);fig.savefig(OUT/'actual_budget_pairwise_contrasts.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True,sharex=True,sharey='row')
    for col,ds in enumerate(['hc2','vidstg']):
        for field,color,marker,label in [('delta_current_v','#c97842','s','Current: After − Before'),('delta_inherited_v','#5b86b8','o','Inherited: Before − Frozen')]:
            for k in [1,2,4,8]:
                m=analysis['datasets'][ds][str(k)]['all_orders']['metrics'][field];lo,hi=np.asarray(m['ci95'])*100;mu=m['mean']*100
                x=k*(.96 if marker=='s' else 1.04)
                axes[0,col].errorbar(x,mu,yerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt=marker,color=color,capsize=3,label=label if k==1 else None)
        for field,color,marker,label in [('observed_iou_delta_mean','#c97842','s','Admitted observed frames'),('unobserved_iou_delta_mean','#5b86b8','o','Other GT frames')]:
            values=[analysis['datasets'][ds][str(k)]['all_orders']['frame_populations'][field] for k in [1,2,4,8]]
            axes[1,col].plot([1,2,4,8],[np.nan if v is None else v*100 for v in values],marker=marker,color=color,label=label)
        for band in [0,1]:
            ax=axes[band,col];ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8],['1','2','4','8']);ax.axhline(0,color='#777');ax.grid(alpha=.15);ax.legend(fontsize=8)
        axes[0,col].set_title(ds);axes[1,col].set_xlabel('Uniform observation budget K')
    axes[0,0].set_ylabel('Own-trajectory vIoU components (pp), 95% CI')
    axes[1,0].set_ylabel('Current frame IoU mean difference (pp)')
    fig.savefig(OUT/'actual_budget_mechanism.png',dpi=180);fig.savefig(OUT/'actual_budget_mechanism.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True,sharey=True)
    for ax,ds in zip(axes,['hc2','vidstg']):
        for field,color,marker,label in [('real_fit_seconds_per_arrival','#206c99','o','Fit / synchronous recording'),('shared_capture_seconds_per_arrival','#c97842','s','Recorded shared input capture'),('independent_CPU_math_seconds_per_arrival','#777','^','Recorded CPU math audit')]:
            vals=[analysis['datasets'][ds][str(k)]['all_orders']['actual_cost'][field] for k in [1,2,4,8]]
            ax.plot([1,2,4,8],vals,marker=marker,color=color,label=label)
        ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8],['1','2','4','8']);ax.set_ylim(bottom=0);ax.grid(alpha=.15)
        ax.set_xlabel('Uniform observation budget K');ax.set_title(ds);ax.legend(fontsize=8)
    axes[0].set_ylabel('Recorded seconds / logical arrival')
    fig.suptitle('K4 uses exact original stream timings; capture can reuse cached expert outputs',fontsize=10)
    fig.savefig(OUT/'actual_budget_recorded_cost.png',dpi=180);fig.savefig(OUT/'actual_budget_recorded_cost.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True,sharex=True)
    for ax,ds in zip(axes,['hc2','vidstg']):
        v=analysis['unified_appendix'][ds]
        for y,(label,m) in enumerate([('Unified After − Frozen',v['statistics']['metrics']['delta_total_v']),('Unified − main K4',v['unified_minus_main_K4']['metrics']['v'])]):
            lo,hi=np.asarray(m['ci95'])*100;mu=m['mean']*100
            ax.errorbar(mu,y,xerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt='o',capsize=4,color='#206c99')
        ax.set_yticks([0,1],['Unified After − Frozen','Unified − main K4']);ax.axvline(0,color='#777');ax.grid(alpha=.15);ax.set_title(ds)
        ax.set_xlabel('vIoU difference (pp), paired parent 95% CI')
    fig.suptitle('Pre-GT unified configuration appendix / multiple parameters differ',fontsize=10)
    fig.savefig(OUT/'actual_unified_appendix.png',dpi=180);fig.savefig(OUT/'actual_unified_appendix.pdf');plt.close(fig)
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
            ax.set_title(f'{ds}: {case["kind"].replace("_"," ")} / {case["stage"]}\ncurrent {case["delta_current_v"]*100:+.2f} pp; inherited {case["delta_inherited_v"]*100:+.2f} pp')
            views.append(private_view(ds,case,fit,inp,row,truth[case['query_ordinal']],unpack_expert(inp['expert']),private))
            status(DEST/'STATUS.json',dict(status='running',stage='actual_saved_signal_and_RGB_case_render',done=len(views),total=6,
                worker_pid=os.getpid(),CPU_only=True,model_calls=0,time=time.time()))
            print('ROOT_P5_SIGNAL_CASE',ds,case['kind'],len(views),6,flush=True)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,fontsize=8)
    fig.savefig(OUT/'actual_case_signal_chains.png',dpi=180);fig.savefig(OUT/'actual_case_signal_chains.pdf');plt.close(fig)
    write(OUT/'ACTUAL_ROOT_CASE_SIGNAL_CHAINS.json',dict(status='pass',records=records,cases=6,
        independent_sample_GT_IoU_checks=checks,all_cases_posthoc_after_complete_population=True,
        RGB_query_raw_logits_GT_boxes_or_actions_exported=False,model_calls=0,time=time.time()))
    write(DEST/'PRIVATE_CASE_VIEWS.json',dict(status='rendered_pending_actual_root_view',records=views,cases=6,
        private_media_exported=False,GT_after_global_seal=True,time=time.time()))
    write(DEST/'COMPLETION.json',dict(status='pending_actual_root_view_publication',cases=6,model_calls=0,CPU_only=True,
        outputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'ACTUAL_ROOT_CASE_SIGNAL_CHAINS.json',DEST/'PRIVATE_CASE_VIEWS.json',*OUT.glob('actual_*.png'),*OUT.glob('actual_*.pdf')]},time=time.time()))
    status(DEST/'STATUS.json',dict(status='pending_actual_root_view_publication',cases=6,CPU_only=True,paper_suite_complete=False,time=time.time()))


if __name__=='__main__':
    try:run()
    except BaseException:
        p=DEST/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc());raise
