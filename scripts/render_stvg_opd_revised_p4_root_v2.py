"""Postseal all 16 P4 condition effects/decomposition/cost and six saved feedback/RGB diagnostic cases."""
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
DEST=BASE/'P4_actual_root_signal_views';OUT=PUB/'P4'


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
    assert pixel==inp['pixel_sha256'] and spec==inp['corruption_spec'], 'Rendered physical RGB must match original captured input'
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
    draw.text((12, 8), f'{ds}: {case["kind"]} | query {case["query_ordinal"]} | {case["order"]} | {case["condition"]}', font=big, fill='black')
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
    complete=read(BASE/'P4_actual_root_readback/COMPLETION.json')
    assert complete['actual_arrivals']==15504 and complete['status']=='pending_actual_root_signal_cases_view_publication'
    for f,h in complete['outputs'].items():assert sha(ROOT/f)==h
    assert read(BASE/'P4_PREDICTION_BARRIER.json')['all_deployment_arms_and_directions']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    stats=read(OUT/'ROOT_STATISTICS.json');fig,axes=plt.subplots(1,2,figsize=(13,7),constrained_layout=True)
    for ax,ds in zip(axes,['hc2','vidstg']):
        name='P4_'+ds;labels=[]
        for y,(condition,result) in enumerate(stats['by_stage'][name].items()):
            m=result['on_policy']['metrics']['delta_total_v'];lo,hi=np.asarray(m['ci95'])*100;value=m['mean']*100
            ax.errorbar(value,y,xerr=[[max(0,value-lo)],[max(0,hi-value)]],fmt='o',capsize=3,color='#206c99')
            labels.append(condition.replace('_',' '))
        ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.axvline(0,color='#777');ax.grid(alpha=.15)
        ax.set_xlabel('Full − Frozen vIoU (pp), paired parent 95% CI');ax.set_title(ds)
    fig.savefig(OUT/'actual_all_condition_paired_effects.png',dpi=180);fig.savefig(OUT/'actual_all_condition_paired_effects.pdf');plt.close(fig)
    from scripts.audit_stvg_opd_p4_public_v2 import aggregate
    import gzip
    fig,axes=plt.subplots(1,2,figsize=(13,8),constrained_layout=True);mechanism={}
    for ax,ds in zip(axes,['hc2','vidstg']):
        with gzip.open(PUB/('P4_'+ds)/'ROWS.jsonl.gz','rt') as f:rr=list(map(json.loads,f))
        conditions=list(stats['by_stage']['P4_'+ds]);mechanism[ds]={}
        for y,condition in enumerate(conditions):
            rows=[r for r in rr if r['condition']==condition]
            m=aggregate(rows,['delta_current_v','delta_inherited_v'])[0]['metrics']
            observed=[r['observed_iou_delta'] for r in rows if r['observed_iou_delta'] is not None]
            unobserved=[r['unobserved_iou_delta'] for r in rows if r['unobserved_iou_delta'] is not None]
            mechanism[ds][condition]=dict(current_v=m['delta_current_v'],inherited_v=m['delta_inherited_v'],
                observed_positions_iou_mean=float(np.mean(observed)) if observed else None,
                observed_valid_query_count=len(observed),unobserved_positions_iou_mean=float(np.mean(unobserved)) if unobserved else None,
                unobserved_valid_query_count=len(unobserved),
                actual_fit_wall_seconds_per_arrival=float(np.mean([r['compute']['fit_GPU_seconds'] for r in rows])),
                actual_shared_capture_seconds_per_arrival=float(np.mean([r['compute']['shared_capture_seconds'] for r in rows])))
            for field,offset,color,label in [('delta_current_v',-.13,'#e07a48','Current: After − Before'),('delta_inherited_v',.13,'#5b86b8','Inherited: Before − Frozen')]:
                v=m[field];lo,hi=np.asarray(v['ci95'])*100;mu=v['mean']*100
                ax.errorbar(mu,y+offset,xerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt='o',color=color,capsize=2,label=label if y==0 else None)
        ax.set_yticks(range(len(conditions)),[v.replace('_',' ') for v in conditions]);ax.invert_yaxis();ax.axvline(0,color='#777');ax.grid(alpha=.15)
        ax.set_xlabel('Own-trajectory vIoU decomposition (pp), parent 95% CI');ax.set_title(ds);ax.legend(fontsize=8)
    fig.savefig(OUT/'actual_all_condition_decomposition.png',dpi=180);fig.savefig(OUT/'actual_all_condition_decomposition.pdf');plt.close(fig)
    write(OUT/'ACTUAL_ROOT_ALL_CONDITION_MECHANISM_COST.json',dict(status='pass',datasets=mechanism,
        inherited_component_is_not_an_alpha0_causal_contrast=True,observed_and_unobserved_are_different_frame_populations=True,
        actual_cost_includes_synchronous_recording=True,cold_deployment_latency=False,model_calls=0,time=time.time()))
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
            ax.set_title(f'{ds}: {case["kind"].replace("_"," ")} / {case["condition"]}\ncurrent {case["delta_current_v"]*100:+.2f} pp; inherited {case["delta_inherited_v"]*100:+.2f} pp')
            views.append(private_view(ds,case,fit,inp,row,truth[case['query_ordinal']],unpack_expert(inp['expert']),private))
            status(DEST/'STATUS.json',dict(status='running',stage='actual_saved_signal_and_RGB_case_render',done=len(views),total=6,
                worker_pid=os.getpid(),CPU_only=True,model_calls=0,time=time.time()))
            print('ROOT_P4_SIGNAL_CASE',ds,case['kind'],len(views),6,flush=True)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,fontsize=8)
    fig.savefig(OUT/'actual_case_signal_chains.png',dpi=180);fig.savefig(OUT/'actual_case_signal_chains.pdf');plt.close(fig)
    write(OUT/'ACTUAL_ROOT_CASE_SIGNAL_CHAINS.json',dict(status='pass',records=records,cases=6,
        independent_sample_GT_IoU_checks=checks,all_cases_posthoc_after_complete_population=True,
        RGB_query_raw_logits_GT_boxes_or_actions_exported=False,model_calls=0,time=time.time()))
    write(DEST/'PRIVATE_CASE_VIEWS.json',dict(status='rendered_pending_actual_root_view',records=views,cases=6,
        private_media_exported=False,GT_after_global_seal=True,time=time.time()))
    write(DEST/'COMPLETION.json',dict(status='pending_actual_root_view_publication',cases=6,model_calls=0,CPU_only=True,
        outputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'ACTUAL_ROOT_CASE_SIGNAL_CHAINS.json',OUT/'actual_case_signal_chains.png',
            OUT/'actual_case_signal_chains.pdf',OUT/'actual_all_condition_paired_effects.png',OUT/'actual_all_condition_paired_effects.pdf',
            OUT/'ACTUAL_ROOT_ALL_CONDITION_MECHANISM_COST.json',OUT/'actual_all_condition_decomposition.png',OUT/'actual_all_condition_decomposition.pdf',DEST/'PRIVATE_CASE_VIEWS.json']},time=time.time()))
    status(DEST/'STATUS.json',dict(status='pending_actual_root_view_publication',cases=6,CPU_only=True,paper_suite_complete=False,time=time.time()))


if __name__=='__main__':
    try:run()
    except BaseException:
        p=DEST/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc());raise
