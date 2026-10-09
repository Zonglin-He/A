"""Presentation revision: real filmstrips and a transparent conditional view.

Reuses all 512 sealed v6 native scalar rows, without any model or optimizer
call. The conditional endpoint is explicitly post hoc; the original complete
quadrants and their contrary result are preserved and accompany publication.
Dataset pixels, exact query and GT overlays remain local.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OLD = ROOT / 'artifacts/stvg_motivation_cross_domain_v6'
BASE = ROOT / 'artifacts/stvg_motivation_filmstrip_v7'
DIRECTIONS = ['vidstg_to_hc2', 'hc2_to_vidstg']
MODELS = ['tastvg', 'tubedetr']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    assert not path.exists(), ('Preserve existing artifact', str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def conditional(rows, quadrants):
    """Paired parent bootstrap on saved scalars; no fresh GT evaluation."""
    import numpy as np
    assert len(rows) == 512
    results = {}
    comparisons = 0
    for di, direction in enumerate(DIRECTIONS):
        # Same registered seed and draw distribution as v6. A new display,
        # not a replacement for the original registered primary contrast.
        rng = np.random.default_rng(20261009 + di)
        draws = rng.integers(0, 128, size=(10000, 128))
        results[direction] = {}
        for model in MODELS:
            z = sorted([r for r in rows if r['direction'] == direction and r['model'] == model],
                       key=lambda r: r['parent'])
            assert [r['parent'] for r in z] == list(range(128))
            t = np.array([r['tIoU'] > .5 for r in z], dtype=np.int64)
            s = np.array([r['sIoU'] > .5 for r in z], dtype=np.int64)
            counts = [int(np.sum(t*s)), int(np.sum(t*(1-s))),
                      int(np.sum((1-t)*s)), int(np.sum((1-t)*(1-s)))]
            original = quadrants['directions'][direction]['models'][model]
            assert counts == original['counts']
            comparisons += 4
            denominator = int(t.sum()); numerator = counts[1]
            boot_den = t[draws].sum(1)
            assert np.all(boot_den > 0)
            boot_num = (t*(1-s))[draws].sum(1)
            rate = 100*numerator/denominator
            ci = np.quantile(100*boot_num/boot_den, [.025, .975]).tolist()
            results[direction][model] = dict(N_all=128, temporal_correct=denominator,
                temporal_correct_spatial_wrong=numerator, conditional_spatial_failure_percent=rate,
                conditional_95CI_percent=ci, original_quadrant_counts=counts,
                original_TplusSminus_minus_TminusSplus_pp=original['contrast_TplusSminus_minus_TminusSplus_pp']
                if 'contrast_TplusSminus_minus_TminusSplus_pp' in original else
                100*(counts[1]-counts[2])/128)
    return dict(status='pass', scope='post-hoc conditional descriptive view of all sealed native v6 scalars',
        total_native_rows=512, unique_parents=256, sources_per_direction=128,
        conditional_event='S <= .5 conditional on T > .5', temporal_metric='physical-frame tIoU',
        spatial_metric='mean frame IoU on all legal GT frames, uncovered support zero',
        thresholds_changed=False, checkpoint_cohort_outputs_changed=False,
        original_quadrant_comparisons=comparisons, bootstrap_draws=10000,
        bootstrap_seed=20261009, bootstrap_direction_offset=True,
        all_four_original_categories_retained=True,
        original_error_dominance_hypothesis_not_supported=True,
        no_new_model_calls=True, no_new_GT_metric_evaluation=True,
        active_P1_payload_access=False, directions=results)


def prepare():
    from scripts.stvg_motivation_cross_domain_common_v6 import check_global_seal
    assert not (BASE / 'CONDITIONAL_STATISTICS.json').exists()
    check_global_seal()
    assert read(OLD / 'ROOT_AUDIT.json')['status'] == 'pass'
    lock = read(BASE / 'DESIGN_LOCK.json')
    assert lock['pins']['scripts/render_stvg_motivation_filmstrip_v7.py'] == sha(Path(__file__))
    result = conditional(read(OLD / 'SCALAR_ROWS.json'), read(OLD / 'QUADRANT_STATISTICS.json'))
    result['input_sha256'] = {n:sha(OLD/n) for n in ['SCALAR_ROWS.json', 'QUADRANT_STATISTICS.json',
                                                   'ROOT_AUDIT.json', 'GLOBAL_PREDICTION_BARRIER.json']}
    write(BASE / 'CONDITIONAL_STATISTICS.json', result)
    # Existing, explicitly illustrative v6 case. Five uniformly spaced shared
    # event samples, not frames chosen for the largest spatial error.
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from scripts.score_stvg_motivation_cross_domain_v6 import read_truths
    import numpy as np
    from PIL import Image
    old_case = read(OLD / 'PRIVATE_CASE.json')
    direction = old_case['direction']; ordinal = old_case['ordinal']
    roster = read(OLD / direction / 'ROSTER.json'); row = roster['rows'][ordinal]
    pixels, ids = decode(row['input'])
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == old_case['canonical_RGB_sha256']
    predictions = {m:read(OLD/direction/m/'predictions'/f'{ordinal:05}.json') for m in MODELS}
    truth, spans, provenance = read_truths(direction, roster)
    gt = truth[ordinal]; span = spans[ordinal]
    eligible = [fid for fid in ids if fid in gt and span[0] <= fid < span[1]
        and all(z['interval'][0] <= fid < z['interval'][1] for z in predictions.values())]
    indices = np.linspace(0, len(eligible)-1, 5).round().astype(int).tolist()
    chosen = [eligible[j] for j in indices]
    assert len(set(chosen)) == 5
    frames = []; models = {m:dict(interval=z['interval'], boxes=[],
        tIoU=old_case['models'][m]['tIoU'], sIoU=old_case['models'][m]['sIoU']) for m,z in predictions.items()}
    width, height = row['input']['width'], row['input']['height']
    for fid in chosen:
        image = pixels[ids.index(fid)]
        path = BASE / 'private_frames' / f'frame_{fid:05}.png'
        path.parent.mkdir(exist_ok=True); assert not path.exists()
        Image.fromarray(image).save(path)
        frames.append(dict(frame_id=fid, path=str(path.relative_to(BASE)), sha256=sha(path),
            pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(), GT_box=list(map(float,gt[fid]))))
        for m,z in predictions.items():
            boxes = np.asarray(z['boxes'], dtype=float)*[width,height,width,height]
            models[m]['boxes'].append([float(np.interp(fid,z['box_frame_ids'],boxes[:,j])) for j in range(4)])
    endpoints = [*old_case['input_frame_extent'], *span]
    for z in models.values(): endpoints += z['interval']
    write(BASE / 'PRIVATE_CASE.json', dict(direction=direction, ordinal=ordinal,
        query=old_case['query'], width=width, height=height, fps=old_case['fps'],
        GT_interval=span, timeline_extent=[min(endpoints),max(endpoints)],
        frames=frames, models=models, original_case_sha256=sha(OLD/'PRIVATE_CASE.json'),
        input_prediction_sha256={m:sha(OLD/direction/m/'predictions'/f'{ordinal:05}.json') for m in MODELS},
        GT_provenance_sha256=provenance, canonical_RGB_sha256=old_case['canonical_RGB_sha256'],
        case_selection=old_case['selection'], frame_selection='five evenly spaced sampled frames in shared event support',
        new_model_calls=0, actual_RGB_unchanged=True, local_private_media_query_GT_overlay=True,
        active_P1_payload_access=False, time=time.time()))
    print(json.dumps(dict(status='prepared', native_rows=512, actual_frame_ids=chosen)))


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
        'figure.facecolor':'white','savefig.facecolor':'white','axes.facecolor':'white',
        'svg.fonttype':'none','pdf.fonttype':42})
    return plt


def stats(ax, data):
    from matplotlib.patches import Rectangle
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.text(0,.985,'(b) Correct event ≠ correct instance',fontsize=17,weight='semibold',va='top',color='#17191C')
    ax.text(0,.91,'Spatial failures among correct-time predictions',fontsize=10.5,va='top',color='#666A70')
    names = ['VidSTG → HC2','HC2 → VidSTG']
    for di,direction in enumerate(DIRECTIONS):
        top = .79-di*.355
        ax.text(0,top,names[di],fontsize=12.5,weight='semibold',color='#17191C')
        for mi,model in enumerate(MODELS):
            y=top-.09-mi*.125; z=data['directions'][direction][model]
            rate=z['conditional_spatial_failure_percent']; lo,hi=z['conditional_95CI_percent']
            ax.text(0,y,'TA-STVG' if model=='tastvg' else 'TubeDETR',va='center',fontsize=10.5,color='#444950')
            x=.245; w=.545
            ax.add_patch(Rectangle((x,y-.024),w,.048,facecolor='#ECEEF1',edgecolor='none'))
            ax.add_patch(Rectangle((x,y-.024),w*rate/100,.048,facecolor='#E66B58',edgecolor='none'))
            ax.plot([x+w*lo/100,x+w*hi/100],[y,y],color='#6A2F23',lw=1.3)
            for v in [lo,hi]:ax.plot([x+w*v/100]*2,[y-.009,y+.009],color='#6A2F23',lw=1.3)
            ax.text(.835,y,f'{rate:.1f}%',fontsize=16,weight='semibold',va='center',color='#B64A39')
            ax.text(x,y-.050,f"{z['temporal_correct_spatial_wrong']} / {z['temporal_correct']} correct-time queries",
                    fontsize=8.7,color='#666A70',va='top')
    ax.text(.245,.096,'0',fontsize=8.8,color='#8A8E95')
    ax.text(.79,.096,'100%',fontsize=8.8,color='#8A8E95',ha='right')
    ax.text(0,.035,'T: tIoU > .5   ·   S: mean frame IoU > .5',fontsize=9,color='#666A70')
    ax.text(0,0,'128 parents / direction  ·  whiskers: 95% paired bootstrap CI',fontsize=8.5,color='#666A70')


def save(fig, stem, private):
    files={}
    for ext in ['png','pdf','svg']:
        p=BASE/(stem+'.'+ext); assert not p.exists()
        fig.savefig(p,dpi=260,facecolor='white')
        files[p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
    write(BASE/(stem+'_RENDER.json'),dict(status='rendered_pending_actual_view',files=files,
        statistics_sha256=sha(BASE/'CONDITIONAL_STATISTICS.json'), private_media=private,
        original_error_dominance_hypothesis_not_supported=True,
        new_model_calls=0, active_P1_payload_access=False,actual_root_view=False,time=time.time()))


def render(full):
    import numpy as np
    from PIL import Image
    from matplotlib.patches import Rectangle
    data=read(BASE/'CONDITIONAL_STATISTICS.json'); plt=setup()
    if not full:
        fig=plt.figure(figsize=(8,6.3)); stats(fig.add_axes([.05,.05,.90,.90]),data)
        save(fig,'panel_B_conditional_cross_domain',False); plt.close(fig); return
    case=read(BASE/'PRIVATE_CASE.json')
    fig=plt.figure(figsize=(19,6.9))
    whole=fig.add_axes([0,0,1,1]); whole.axis('off'); whole.set_xlim(0,1);whole.set_ylim(0,1)
    whole.text(.028,.966,'(a) Cross-domain instance ambiguity',fontsize=17,weight='semibold',va='top',color='#17191C')
    whole.text(.028,.908,'Query: '+case['query'],fontsize=11.5,color='#444950',va='top')
    # Continuous black film stock with white sprocket perforations. The five
    # RGB frames and physical box coordinates are unmodified and uncropped.
    start=.115; end=.645; fw=(end-start-.020)/5; fh=fw*19*.75/6.9
    rows=[('Ground truth','gt','#22875B'),('TA-STVG','tastvg','#E15449'),('TubeDETR','tubedetr','#E15449')]
    bottoms=[.634,.356,.078]
    for ri,(label,model,color) in enumerate(rows):
        bottom=bottoms[ri]; rail=.011
        whole.add_patch(Rectangle((start-.006,bottom-rail),end-start+.006,fh+2*rail,
                                  facecolor='#18191B',edgecolor='none'))
        for py in [bottom-.0077,bottom+fh+.0030]:
            for px in np.arange(start-.002,end-.007,.0105):
                whole.add_patch(Rectangle((px,py),.0045,.0048,facecolor='white',edgecolor='none'))
        whole.text(.028,bottom+fh*.55,label,fontsize=12.5,weight='semibold',va='center',color=color)
        if model!='gt':
            whole.text(.028,bottom+fh*.36,f"tIoU {case['models'][model]['tIoU']:.2f}",fontsize=9.5,color='#666A70')
        for j,frm in enumerate(case['frames']):
            path=BASE/frm['path']; assert sha(path)==frm['sha256']
            image=np.asarray(Image.open(path).convert('RGB'))
            assert hashlib.sha256(image.tobytes()).hexdigest()==frm['pixel_sha256']
            ax=fig.add_axes([start+j*(fw+.004),bottom,fw,fh]); ax.axis('off');ax.imshow(image)
            box=frm['GT_box'] if model=='gt' else case['models'][model]['boxes'][j]
            x1,y1,x2,y2=box
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,lw=2.0))
        timeline=fig.add_axes([start,bottom-.032,end-start,.016])
        extent=np.asarray(case['timeline_extent'])/case['fps']
        span=np.asarray(case['GT_interval'] if model=='gt' else case['models'][model]['interval'])/case['fps']
        timeline.set_xlim(extent[0]-.3,extent[1]+.3);timeline.set_ylim(0,1);timeline.axis('off')
        timeline.plot(extent,[.5,.5],color='#E3E5E8',lw=2)
        timeline.plot(span,[.5,.5],color=color,lw=4,solid_capstyle='butt')
        for f in case['frames']:timeline.plot(f['frame_id']/case['fps'],.5,'|',color='#18191B',ms=3)
        whole.text(start,bottom-.048,f'{extent[0]:.1f}s',fontsize=8,color='#8A8E95')
        whole.text(end,bottom-.048,f'{extent[1]:.1f}s',fontsize=8,color='#8A8E95',ha='right')
    whole.text(.115,.012,'Same video and query · native outputs · illustrative case',fontsize=9,color='#777C83')
    stats(fig.add_axes([.705,.076,.272,.875]),data)
    save(fig,'figure1_filmstrip_cross_domain',True);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','full','statistics','audit'])
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='audit':
        result=conditional(read(OLD/'SCALAR_ROWS.json'),read(OLD/'QUADRANT_STATISTICS.json'))
        saved=read(BASE/'CONDITIONAL_STATISTICS.json')
        assert result['directions']==saved['directions']
        print(json.dumps(dict(status='pass',native_rows=512,original_quadrant_checks=16,
                             conditional_cells=4,all_original_negative_findings_retained=True)))
    else:render(args.mode=='full')
