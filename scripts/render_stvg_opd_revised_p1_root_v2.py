"""Actual postseal P1 feedback chains and private RGB cases; no model execution.

Six cases are the already locked deterministic extremes after the full population
readback. They diagnose saved fits, never select an online step or configuration.
Only scalar curves and anonymous IDs are public; RGB/query/box geometry is local.
"""
import collections
import gzip
import json
import os
from pathlib import Path
import sys
import textwrap
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate, BASE, PUB, verify, read, write, sha

DEST = BASE / 'P1_actual_root_signal_views'


def coordinates(boxes, width, height):
    import numpy as np
    a = np.asarray(boxes, dtype=np.float64)
    return np.concatenate([(a[..., :2] - a[..., 2:] / 2) * [width, height],
                           (a[..., :2] + a[..., 2:] / 2) * [width, height]], axis=-1)


def overlap(boxes, target):
    import numpy as np
    a = np.asarray(boxes, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    sizes = np.maximum(np.minimum(a[..., 2:], b[..., 2:]) - np.maximum(a[..., :2], b[..., :2]), 0)
    intersection = sizes[..., 0] * sizes[..., 1]
    ap = np.maximum(a[..., 2:] - a[..., :2], 0)
    bp = np.maximum(b[..., 2:] - b[..., :2], 0)
    union = ap[..., 0] * ap[..., 1] + bp[..., 0] * bp[..., 1] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def signal_chain(fit, row, truth):
    import numpy as np
    from vg_tta.tastvg_oracle_event5_v1 import box_iou
    width, height = row['input']['width'], row['input']['height']
    details = []
    all_details = []
    checks = 0
    for k, rd in enumerate(fit['rounds']):
        frames = []
        a = rd['rollout']['samples'].numpy()
        actions = 1 / (1 + np.exp(-a))  # Original saved diagnostic uses NumPy float32 sigmoid.
        weights = rd['rollout']['weights'].numpy()
        for j, pos in enumerate(fit['positions']):
            fid = row['frame_ids'][pos]
            if fid not in truth:
                continue
            px = coordinates(actions[j], width, height)
            quality = overlap(px, truth[fid])
            independent = box_iou(px, truth[fid])
            assert np.max(np.abs(quality - independent)) < 2e-12
            checks += len(quality)
            before = float(overlap(coordinates(fit['path'][k]['boxes'][pos].numpy(), width, height), truth[fid]))
            after = float(overlap(coordinates(fit['path'][k+1]['boxes'][pos].numpy(), width, height), truth[fid]))
            values = dict(sample_best_gt_iou=float(quality.max()), sample_mean_gt_iou=float(quality.mean()),
                teacher_weighted_gt_iou=float(weights[j] @ quality),
                teacher_vs_uniform_gt_iou=float(weights[j] @ quality - quality.mean()),
                central_before_gt_iou=before, central_after_gt_iou=after, central_gt_delta=after-before,
                central_expert_IoU_before=float(rd['central_reward_before'][j]),
                central_expert_IoU_after=float(rd['central_reward_after'][j]),
                reward_variance=float(rd['reward_variance'][j]), ESS=float(rd['ESS'][j]),
                weight_max=float(rd['weight_max'][j]))
            frames.append(values)
            all_details.append(values)
        if frames:
            details.append(dict(round=k+1, GT_evaluable_admitted_positions=len(frames),
                **{key: float(np.mean([v[key] for v in frames])) for key in frames[0]}))
    return details, all_details, checks


def private_view(ds, case, fit, inp, row, truth, expert, directory):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from scripts.run_decota_paper_main_v1 import frames_for
    from vg_tta.decota_fixed_full_audit_v1 import top1_support
    frames, ids, _ = frames_for(ds, row, collections.OrderedDict())
    assert ids == row['frame_ids']
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
    draw.text((12, 8), f'{ds}: {case["kind"]} | query {case["query_ordinal"]} | {case["order"]}', font=big, fill='black')
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
        caption_for_private_identity_readback=row['input']['caption'], private_media_not_public=True)


def run():
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.decota_matrix_common_v1 import load, status
    activate(); verify(); torch.set_num_threads(2)
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import read_row, unpack_expert
    pin = read(DEST/'RUNTIME.json')
    for f, h in pin['pins'].items(): assert sha(ROOT/f)==h, f
    complete = read(BASE/'P1_actual_root_readback/COMPLETION.json')
    assert complete['actual_arrivals']==41355 and complete['status']=='pending_actual_root_signal_cases_view_publication'
    for f, h in complete['artifacts'].items(): assert sha(ROOT/f)==h
    assert read(BASE/'P1_PREDICTION_BARRIER.json')['all_deployment_OPD_directions']
    design = read(BASE/'DESIGN_LOCK.json')
    selected = read(PUB/'TABLE1_ACTUAL_ROOT_CASE_SELECTION.json')
    assert sha(PUB/'TABLE1_ACTUAL_ROOT_CASE_SELECTION.json')==pin['case_selection_sha256']
    private = DEST/'private_cases'; private.mkdir(parents=True, exist_ok=True)
    views = []; datasets = {}; cases = []; count = 0; checks = 0
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(3, 2, figsize=(12,10), constrained_layout=True)
    labels = [('central_after_gt_iou', 'Central output vs GT', '#c54942', '-'),
        ('central_expert_IoU_after', 'Central output vs expert', '#7855a3', '-'),
        ('teacher_weighted_gt_iou', 'Teacher-weighted samples vs GT', '#14897d', '-'),
        ('sample_mean_gt_iou', 'Uniform samples vs GT', '#888888', '--'),
        ('sample_best_gt_iou', 'Best sample vs GT (offline)', '#b58a24', ':')]
    for col, ds in enumerate(['hc2','vidstg']):
        stage = design['stages']['P1_'+ds]
        truth, spans, provenance = truths(ds, stage)
        with gzip.open(PUB/('P1_'+ds)/'ROWS.jsonl.gz', 'rt') as f:
            scored = {(r['query_ordinal'], r['order']):r for r in map(json.loads, f)}
        datasets[ds] = []
        for band, case in enumerate(selected['datasets'][ds]):
            q, order, at = case['query_ordinal'], case['order'], case['arrival']
            assert stage['orders'][order][at]==q
            file = BASE/'stages'/('P1_'+ds)/'clean'/order/'on_policy'/f'{at:05}.pt'
            assert sha(file)==read(file.with_suffix('.json'))['sha256']
            z = load(file); fit = z['fit']; inp = load(BASE/z['input']['path'])
            assert sha(BASE/z['input']['path'])==z['input']['sha256']
            expert = unpack_expert(inp['expert']); row = read_row(ds,q)
            chain, details, n = signal_chain(fit,row,truth[q]); checks += n
            assert len(chain)==fit['selected_step']
            saved = scored[q,order]
            for key in ['sample_best_gt_iou','teacher_vs_uniform_gt_iou','central_gt_delta','reward_variance','ESS','weight_max']:
                val = float(np.mean([r[key] for r in details]))
                assert abs(val-saved[key])<2e-10, (ds,q,key,val,saved[key])
                checks += 1
            record = {**case, 'rounds':chain, 'GT_evaluable_admitted_positions':chain[0]['GT_evaluable_admitted_positions'],
                'admitted_expert_GT_IoU':saved['admitted_expert_GT_IoU'],
                'conditioning':'GT-evaluable admitted observation frames only',
                'best_sample_is_offline_not_a_deployment_output':True}
            datasets[ds].append(record); cases.append(record)
            ax = axes[band,col]
            for key,label,color,style in labels:
                ax.plot([r['round'] for r in chain],[r[key] for r in chain],label=label,color=color,ls=style,lw=1.5)
            ax.set_ylim(0,1); ax.grid(alpha=.15); ax.set_xlabel('Configured update round'); ax.set_ylabel('Frame IoU')
            ax.set_title(f'{ds}: {case["kind"].replace("_"," ")}\ncurrent {case["delta_current_v"]*100:+.2f} pp; inherited {case["delta_inherited_v"]*100:+.2f} pp')
            views.append(private_view(ds,case,fit,inp,row,truth[q],expert,private)); count += 1
            status(DEST/'STATUS.json',dict(status='running',stage='actual_saved_case_signal_and_RGB_render',
                dataset=ds,case_kind=case['kind'],done=count,total=6,CPU_only=True,worker_pid=os.getpid(),time=time.time()))
            print('ROOT_P1_SIGNAL_CASE',ds,case['kind'],count,6,flush=True)
    handles, names = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc='outside lower center',ncol=3,fontsize=8)
    fig.savefig(PUB/'TABLE1_actual_case_signal_chains.png',dpi=180)
    fig.savefig(PUB/'TABLE1_actual_case_signal_chains.pdf'); plt.close(fig)
    write(PUB/'TABLE1_ACTUAL_ROOT_CASE_SIGNAL_CHAINS.json',dict(status='pass',datasets=datasets,
        cases=6,independent_sample_IoU_and_saved_scalar_checks=checks,case_selection_sha256=sha(PUB/'TABLE1_ACTUAL_ROOT_CASE_SELECTION.json'),
        all_cases_selected_after_complete_population=True,raw_boxes_RGB_query_exported=False,
        GT_only_after_global_seal=True,not_an_independent_efficacy_estimate=True,model_calls=0,time=time.time()))
    write(DEST/'PRIVATE_CASE_VIEWS.json',dict(status='rendered_pending_actual_root_view',records=views,
        actual_case_count=count,private_media_exported=False,GT_after_all_41355_predictions_sealed=True,time=time.time()))
    write(DEST/'COMPLETION.json',dict(status='pending_actual_root_view_publication',scope='six actual saved signal chains and RGB cases',
        cases=count,model_calls=0,CPU_only=True,GT_after_global_seal=True,
        outputs={str(p.relative_to(ROOT)):sha(p) for p in [PUB/'TABLE1_ACTUAL_ROOT_CASE_SIGNAL_CHAINS.json',
            PUB/'TABLE1_actual_case_signal_chains.png', PUB/'TABLE1_actual_case_signal_chains.pdf',DEST/'PRIVATE_CASE_VIEWS.json']},time=time.time()))
    status(DEST/'STATUS.json',dict(status='pending_actual_root_view_publication',scope='render done; actual human-agent view and publication remain',
        cases=count,CPU_only=True,paper_suite_complete=False,time=time.time()))


if __name__=='__main__':
    try: run()
    except BaseException:
        from scripts.decota_matrix_common_v1 import status
        dest=DEST/'failures'/str(time.time_ns()); dest.mkdir(parents=True,exist_ok=True)
        (dest/'traceback.txt').write_text(traceback.format_exc())
        status(DEST/'STATUS.json',dict(status='failed_preserved',scope='CPU report helper only',evidence=str(dest.relative_to(ROOT)),time=time.time()))
        raise
