"""Post-global-seal six historical cases, with fixed old display frames.

Cases and frames are copied from completed P3 before any new GT read. The old
success/harm/mismatch labels describe historical Full, never the new controls.
No new inference, frame selection, parameter choice or online GT action occurs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import gzip
import json
from pathlib import Path
import sys
import textwrap
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_design_ablation_common_v1 import *

DEST = BASE / 'actual_fixed_case_views'
LABELS = {'event_full': 'Event-scoped Full', 'whole_video': 'Whole-video',
          'carry_query': 'Query carry', 'full_LN_carry': 'Full LN carry'}


def prepare():
    verify()
    selection = OLDPUB / 'P3/ACTUAL_ROOT_CASE_SELECTION.json'
    old_views = OLD / 'P3_actual_root_signal_views/PRIVATE_CASE_VIEWS.json'
    known = read(selection)['records']
    frames = read(old_views)['records']
    assert len(known) == len(frames) == 6
    records = []
    for case in known:
        matched = [v for v in frames if all(v[k] == case[k] for k in ['dataset','kind','query_ordinal'])]
        assert len(matched) == 1
        st = case['stage'].replace('P3_', 'A_')
        assert stages()[st]['orders'][case['order']][case['arrival']] == case['query_ordinal']
        records.append({**{k: case[k] for k in ['dataset','query_ordinal','source_id','condition','order','arrival']},
            'stage': st, 'historical_role': case['kind'], 'frames': matched[0]['frames']})
    lock = DEST / 'CASE_DIAGNOSTIC_LOCK.json'
    value = dict(status='fixed_historical_cases_before_new_GT', cases=6, records=records,
        old_case_selection_sha256=sha(selection), old_frame_metadata_sha256=sha(old_views),
        scope='same six completed P3 cases and display frames; no new-score selection',
        new_scores_read=False, new_GT_read=False, no_new_model_calls=True,
        scientific_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'), time=time.time())
    if lock.exists():
        prior = read(lock)
        for key in value:
            if key != 'time': assert prior[key] == value[key], key
    else:
        assert not (BASE/'GT_EXPOSURE.json').exists()
        write(lock, value)
    runtime = dict(status='locked_CPU_only_fixed_case_diagnostic',
        pins={p:sha(ROOT/p) for p in ['scripts/render_stvg_opd_design_ablation_fixed_cases_v1.py',
            'scripts/render_stvg_opd_revised_p1_root_v2.py']},
        case_lock_sha256=sha(lock), scientific_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        scope='postseal saved scalar signals and physical RGB readback only', time=time.time())
    rp = DEST/'RUNTIME.json'
    if rp.exists():
        prior = read(rp)
        for key in runtime:
            if key != 'time': assert prior[key] == runtime[key], key
    else: write(rp, runtime)


def signal(fit, row, gt):
    import numpy as np
    from vg_tta.decota_spatial_opd_action_readback_precision005 import FIELD
    from vg_tta.tastvg_oracle_event5_v1 import box_iou
    from scripts.render_stvg_opd_revised_p1_root_v2 import coordinates, overlap
    chain = []; checks = 0; origins = collections.Counter()
    for k, rd in enumerate(fit['rounds']):
        if FIELD in fit:
            actions = fit[FIELD]['trace'][k]['actions'].numpy()
            origin = 'same saved original GPU actions'
        else:
            raw = rd['rollout']['samples'].numpy().astype(np.float64)
            actions = np.empty_like(raw); pos = raw >= 0
            actions[pos] = 1/(1+np.exp(-raw[pos])); e = np.exp(raw[~pos]); actions[~pos] = e/(1+e)
            origin = 'offline float64 sigmoid of historical immutable samples'
        origins[origin] += 1
        part = []
        for j, p in enumerate(fit['positions']):
            fid = row['frame_ids'][p]
            if fid not in gt: continue
            px = coordinates(actions[j], row['input']['width'], row['input']['height'])
            quality = overlap(px, gt[fid])
            assert np.max(np.abs(quality-box_iou(px,gt[fid]))) < 2e-12
            checks += len(quality)
            weights = rd['rollout']['weights'][j].numpy().astype(np.float64)
            part.append(dict(sample_mean_GT_IoU=float(quality.mean()), sample_best_GT_IoU=float(quality.max()),
                teacher_weighted_GT_IoU=float(weights@quality),
                central_before_GT_IoU=float(overlap(coordinates(fit['path'][k]['boxes'][p].numpy(),row['input']['width'],row['input']['height']),gt[fid])),
                central_after_GT_IoU=float(overlap(coordinates(fit['path'][k+1]['boxes'][p].numpy(),row['input']['width'],row['input']['height']),gt[fid])),
                central_expert_IoU_after=float(rd['central_reward_after'][j])))
        chain.append(dict(round=k+1, GT_evaluable_admitted_positions=len(part),
            **{key:float(np.mean([v[key] for v in part])) if part else None for key in [
                'sample_mean_GT_IoU','sample_best_GT_IoU','teacher_weighted_GT_IoU',
                'central_before_GT_IoU','central_after_GT_IoU','central_expert_IoU_after']}))
    return dict(rounds=chain, empty=bool(fit['empty']), actual_rounds=len(fit['rounds']),
        action_representation=dict(origins), conditional_on_GT_evaluable_admitted_frames=True,
        missing_quality_is_not_zero=True, offline_best_sample_is_not_deployment_output=True), checks


def private_view(case, fits, inputs, row, gt, frames):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from scripts.render_stvg_opd_revised_p1_root_v2 import coordinates, overlap
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from vg_tta.decota_fixed_full_audit_v1 import top1_support
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    columns=[('frozen','Frozen source'), *LABELS.items()]
    cell=440; width,height=row['input']['width'],row['input']['height']
    scale=min((cell-16)/width,340/height); w,h=round(width*scale),round(height*scale)
    canvas=Image.new('RGB',(cell*len(columns),120+(h+85)*len(case['frames'])),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),f"{case['dataset']} | fixed historical {case['historical_role']} | query {case['query_ordinal']} | {case['order']}",font=font,fill='black')
    draw.multiline_text((12,34),'\n'.join(textwrap.wrap(row['input']['caption'],width=165)),font=font,fill='black')
    draw.text((12,94),'Same previously fixed frames; GT green, final prediction coral. Native WHEN is identical in all columns.',font=small,fill='black')
    for band, item in enumerate(case['frames']):
        p,fid=item['position'],item['frame_id']; assert row['frame_ids'][p]==fid and fid in gt
        y=120+band*(h+85)
        for col,(arm,label) in enumerate(columns):
            x=col*cell+8;canvas.paste(Image.fromarray(frames[p]).resize((w,h)),(x,y+32))
            draw.text((x,y),label+f' | frame {fid}',font=small,fill='black')
            boxes=inputs['event_full']['native_boxes'] if arm=='frozen' else fits[arm]['final']
            pixel_box=coordinates(boxes[p].numpy(),width,height)
            for box,color in [(gt[fid],'#159947'),(pixel_box,'#f05245')]:
                z=np.asarray(box,float)*scale;z[[0,2]]+=x;z[[1,3]]+=y+32
                draw.rectangle(z.tolist(),outline='white',width=6);draw.rectangle(z.tolist(),outline=color,width=3)
            draw.text((x,y+34+h),f"Frame IoU {float(overlap(pixel_box,gt[fid])):.3f}",font=small,fill='black')
            if arm!='frozen':
                admitted=dict(top1_support(unpack_expert(inputs[arm]['expert'])))
                draw.text((x,y+54+h),'Admitted evidence at this frame: '+str(p in admitted),font=small,fill='black')
    private=DEST/'private_cases';private.mkdir(parents=True,exist_ok=True)
    path=private/f"{case['dataset']}_{case['historical_role']}.png";canvas.save(path)
    return dict(dataset=case['dataset'],historical_role=case['historical_role'],query_ordinal=case['query_ordinal'],
        path=str(path.relative_to(ROOT)),sha256=sha(path),common_frames=case['frames'],RGB_query_GT_geometry_private=True)


def run():
    verify(); prepare(); activate_old()
    assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['new_formal_fits']==5376
    assert read(BASE/'actual_root/ROOT_ALL_FITS_READBACK.json')['counts']['complete_fits']==5376
    runtime=read(DEST/'RUNTIME.json')
    for rel,digest in runtime['pins'].items(): assert sha(ROOT/rel)==digest
    assert sha(DEST/'CASE_DIAGNOSTIC_LOCK.json')==runtime['case_lock_sha256']
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import read_row, frames_for
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    torch.set_num_threads(2)
    records=[]; views=[]; checks=0;gt_cache={}
    fig,axes=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    colors={'event_full':'#206c99','whole_video':'#dd8c35','carry_query':'#a66390','full_LN_carry':'#14897d'}
    for i,case in enumerate(read(DEST/'CASE_DIAGNOSTIC_LOCK.json')['records']):
        ds=case['dataset'];st=stages()[case['stage']];q=case['query_ordinal']
        if ds not in gt_cache: gt_cache[ds]=truths(ds,st)[0]
        gt=gt_cache[ds][q];row=read_row(ds,q)
        fits={};inputs={};bindings={};first=None
        for arm in ARMS:
            path=BASE/'stages'/case['stage']/case['condition']/case['order']/arm/f"{case['arrival']:05}.pt"
            z=load(path);assert z['query_ordinal']==q and sha(path)==read(path.with_suffix('.json'))['sha256']
            inpfile=ROOT/z['input']['path'];assert sha(inpfile)==z['input']['sha256']
            fits[arm]=z['fit'];inputs[arm]=normalized(load(inpfile));bindings[arm]=dict(fit_sha256=sha(path),input_sha256=sha(inpfile))
            if first is None:first=z
            else:assert z['interval']==first['interval']
        oldfile=ROOT/first['matched_full_payload_path'];assert sha(oldfile)==first['matched_full_payload_sha256']
        old=load(oldfile);fits['event_full']=old['fit'];oldinp=OLD/old['input']['path'];assert sha(oldinp)==old['input']['sha256']
        inputs['event_full']=normalized(load(oldinp));bindings['event_full']=dict(fit_sha256=sha(oldfile),input_sha256=sha(oldinp))
        frames,ids,pixel=frames_for(ds,row,collections.OrderedDict());assert ids==row['frame_ids']
        for inp in inputs.values(): assert pixel==inp['pixel_sha256'] and inp['frame_ids']==ids
        signals={};ax=axes[i%3,i//3]
        for arm in LABELS:
            signals[arm],n=signal(fits[arm],row,gt);checks+=n
            chain=signals[arm]['rounds']
            ax.plot([v['round'] for v in chain],[np.nan if v['central_after_GT_IoU'] is None else v['central_after_GT_IoU'] for v in chain],label=LABELS[arm],color=colors[arm])
        ax.set(title=f"{ds}: historical {case['historical_role'].replace('_',' ')}",xlabel='Configured update round',ylabel='GT IoU on each arm\'s admitted evidence',ylim=(0,1));ax.grid(alpha=.15)
        views.append(private_view(case,fits,inputs,row,gt,frames))
        records.append(dict(dataset=ds,query_ordinal=q,source_id=case['source_id'],historical_role=case['historical_role'],
            stage=case['stage'],condition=case['condition'],order=case['order'],arrival=case['arrival'],
            bindings=bindings,signals=signals,all_four_inputs_same_physical_RGB_SHA=True,pixel_sha256=pixel,
            signal_observation_positions_differ_by_arm=True,no_new_case_or_frame_selection=True))
        status(DEST/'STATUS.json',dict(status='postseal_CPU_saved_signal_and_private_fixed_RGB',done=len(views),total=6,CPU_only=True,model_calls=0,time=time.time()))
        print('DESIGN_FIXED_CASE',ds,case['historical_role'],len(views),6,flush=True)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
    fig.savefig(PUB/'fixed_historical_case_signals.png',dpi=180);fig.savefig(PUB/'fixed_historical_case_signals.pdf');plt.close(fig)
    write(PUB/'FIXED_HISTORICAL_CASE_SIGNALS.json',dict(status='pass',cases=6,records=records,
        complete_fits=24,independent_sample_GT_IoU_checks=checks,
        case_lock_sha256=sha(DEST/'CASE_DIAGNOSTIC_LOCK.json'),private_media_exported=False,
        cases_not_population_estimates=True,GT_only_post_global_seal=True,model_calls=0,time=time.time()))
    write(DEST/'PRIVATE_CASE_VIEWS.json',dict(status='rendered_pending_actual_root_view',cases=6,records=views,GT_only_post_global_seal=True,private_media_not_public=True,time=time.time()))
    write(DEST/'COMPLETION.json',dict(status='pending_actual_root_view',cases=6,complete_fits=24,
        independent_sample_GT_IoU_checks=checks,CPU_only=True,model_calls=0,
        case_lock_sha256=sha(DEST/'CASE_DIAGNOSTIC_LOCK.json'),runtime_sha256=sha(DEST/'RUNTIME.json'),
        outputs={str(p.relative_to(ROOT)):sha(p) for p in [PUB/'FIXED_HISTORICAL_CASE_SIGNALS.json',PUB/'fixed_historical_case_signals.png',PUB/'fixed_historical_case_signals.pdf',DEST/'PRIVATE_CASE_VIEWS.json']},time=time.time()))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare();print('FIXED_CASE_LOCKED',sha(DEST/'CASE_DIAGNOSTIC_LOCK.json'),flush=True)
    else:run()
