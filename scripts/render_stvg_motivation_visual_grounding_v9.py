"""Presentation-only filmstrips: temporal coverage and spatial disagreement.

All native outputs are previously sealed and audited. This file never invokes
models, optimizers, GT scoring, or active P1 payloads. Actual dataset images
and exact query/GT overlays remain local; publication contains only the code,
design/source integrity receipts, and a schematic without dataset pixels.
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
from scripts.render_stvg_motivation_filmstrip_v7 import setup, read, sha

SOURCE = ROOT/'artifacts/stvg_motivation_filmstrip_v7'
NATIVE = ROOT/'artifacts/stvg_motivation_cross_domain_v6'
BASE = ROOT/'artifacts/stvg_motivation_visual_grounding_v9'
COLORS = {'gt':'#22875B', 'tastvg':'#E66B58', 'tubedetr':'#5A8CA8'}


def write(path, data):
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def verify():
    lock = read(BASE/'DESIGN_LOCK.json')
    runtime = BASE/'PRESENTATION_RUNTIME_REVISION001.json'
    if runtime.exists():
        revision = read(runtime)
        assert revision['original_code_sha256'] == lock['code_sha256']
        assert revision['code_sha256'] == sha(Path(__file__))
    else:
        assert lock['code_sha256'] == sha(Path(__file__))
    assert lock['private_case_sha256'] == sha(SOURCE/'PRIVATE_CASE.json')
    assert lock['native_global_seal_sha256'] == sha(NATIVE/'GLOBAL_PREDICTION_BARRIER.json')
    assert read(NATIVE/'ROOT_AUDIT.json')['status'] == 'pass'
    for name, digest in lock['unchanged_statistics'].items():
        assert sha(NATIVE/name) == digest
    return read(SOURCE/'PRIVATE_CASE.json')


def prepare():
    import numpy as np
    from PIL import Image
    from vg_tta.exact_frame_decode_audit_v2 import decode
    case = verify()
    row = read(NATIVE/case['direction']/'ROSTER.json')['rows'][case['ordinal']]
    pixels, ids = decode(row['input'])
    assert ids == row['frame_ids']
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == case['canonical_RGB_sha256']
    indices = np.linspace(0, len(ids)-1, 5).round().astype(int).tolist()
    assert len(set(indices)) == 5
    frames = []
    for index in indices:
        p = BASE/'private_frames'/f'context_{ids[index]:05}.png'
        assert not p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels[index]).save(p)
        frames.append(dict(frame_id=ids[index], path=str(p.relative_to(BASE)),
            sha256=sha(p), pixel_sha256=hashlib.sha256(pixels[index].tobytes()).hexdigest()))
    write(BASE/'PRIVATE_TEMPORAL_FRAMES.json', dict(status='actual_context_RGB_bound',
        frames=frames, selection='five uniformly spaced indices of the original common native input',
        canonical_RGB_sha256=case['canonical_RGB_sha256'], source_roster_sha256=sha(NATIVE/case['direction']/'ROSTER.json'),
        source_case_sha256=sha(SOURCE/'PRIVATE_CASE.json'), new_model_calls=0,
        new_GT_scoring=0, active_P1_payload_access=False, time=time.time()))


def frame_image(base, frame):
    import numpy as np
    from PIL import Image
    path = base/frame['path']
    assert sha(path) == frame['sha256']
    image = np.asarray(Image.open(path).convert('RGB'))
    assert hashlib.sha256(image.tobytes()).hexdigest() == frame['pixel_sha256']
    return image


def strip(fig, page, x, y, width, frames, base, overlay=None, case=None):
    import numpy as np
    from matplotlib.patches import Rectangle
    gap = .0035
    fw = (width-gap*(len(frames)-1))/len(frames)
    fh = fw*fig.get_figwidth()*.75/fig.get_figheight()
    rail = .008
    page.add_patch(Rectangle((x-.004,y-rail),width+.008,fh+2*rail,
                            facecolor='#191A1C',edgecolor='none'))
    for py in [y-.0057,y+fh+.0022]:
        for px in np.arange(x-.001,width+x-.003,.012):
            page.add_patch(Rectangle((px,py),.005,.0038,facecolor='white',edgecolor='none'))
    for j, frame in enumerate(frames):
        ax = fig.add_axes([x+j*(fw+gap),y,fw,fh])
        ax.axis('off'); ax.imshow(frame_image(base,frame))
        if overlay:
            for model in overlay:
                if model == 'gt':box = frame['GT_box']
                else:
                    k = next(i for i,v in enumerate(case['frames']) if v['frame_id']==frame['frame_id'])
                    box = case['models'][model]['boxes'][k]
                x1,y1,x2,y2 = box
                ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,
                    edgecolor=COLORS[model],lw=2.3,linestyle='--' if model=='gt' and len(overlay)>1 else '-'))
    return fh


def interval(page, x, y, width, case, model, label=False, linewidth=4):
    extent = case['timeline_extent']; length = extent[1]-extent[0]
    span = case['GT_interval'] if model=='gt' else case['models'][model]['interval']
    assert extent[0] <= span[0] < span[1] <= extent[1]
    page.plot([x,x+width],[y,y],color='#E6E8EB',lw=2,solid_capstyle='butt')
    page.plot([x+width*(span[0]-extent[0])/length,x+width*(span[1]-extent[0])/length],
              [y,y],color=COLORS[model],lw=linewidth,solid_capstyle='butt')
    if label:page.text(x-.008,y,'GT' if model=='gt' else 'Prediction',ha='right',va='center',
                      fontsize=10,color=COLORS[model])


def legend(page, x, y):
    for k,(name,model) in enumerate([('GT','gt'),('Prediction','tastvg')]):
        bx=x+k*.145
        page.plot([bx,bx+.028],[y,y],color=COLORS[model],lw=2,
                  linestyle='--' if model=='gt' else '-')
        page.text(bx+.035,y,name,fontsize=10.5,va='center',color='#565B61')


def export(fig, stem, private, image_panels, unique_frames):
    # Numeric labels are prohibited by the latest presentation request. Numeric
    # coordinates remain only in evidence metadata and the unchanged caption.
    import re
    texts = [text.get_text() for ax in fig.axes for text in ax.texts]
    assert all(not re.search(r'\d',text) for text in texts),texts
    files = {}
    for ext in ['png','pdf','svg']:
        p = BASE/(stem+'.'+ext); assert not p.exists()
        fig.savefig(p,dpi=260,facecolor='white')
        files[p.name] = dict(sha256=sha(p),bytes=p.stat().st_size)
    write(BASE/(stem+'_RENDER.json'),dict(status='rendered_pending_actual_view',files=files,
        source_case_sha256=sha(SOURCE/'PRIVATE_CASE.json'), design_sha256=sha(BASE/'DESIGN_LOCK.json'),
        no_numeric_text=True,plotted_text=texts,actual_image_panels=image_panels,
        unique_real_frames=unique_frames,private_media=private,
        actual_intervals_and_boxes_unchanged=True,new_model_calls=0,new_GT_scoring=0,
        active_P1_payload_access=False,qualitative_illustration_not_frequency_estimate=True,time=time.time()))


def render(mode):
    plt = setup(); case = verify()
    temporal = read(BASE/'PRIVATE_TEMPORAL_FRAMES.json')['frames']
    if mode == 'panel':
        fig = plt.figure(figsize=(12,7.2)); page = fig.add_axes([0,0,1,1])
        page.axis('off');page.set_xlim(0,1);page.set_ylim(0,1)
        page.text(.075,.96,'Temporal overlap',fontsize=19,weight='semibold',va='top',color='#17191C')
        strip(fig,page,.075,.615,.85,temporal,BASE)
        interval(page,.18,.565,.745,case,'gt',True)
        interval(page,.18,.525,.745,case,'tastvg',True)
        page.text(.075,.44,'Spatial mismatch',fontsize=19,weight='semibold',va='top',color='#17191C')
        strip(fig,page,.075,.145,.85,case['frames'],SOURCE,['gt','tastvg'],case)
        legend(page,.075,.070)
        page.text(.925,.07,'TA-STVG · cross-domain',fontsize=10.5,ha='right',color='#6C7178')
        unique = len({frame['frame_id'] for frame in temporal+case['frames']})
        export(fig,'panel_B_temporal_spatial',True,10,unique);plt.close(fig);return
    if mode == 'schematic':
        # An openly shareable, data-free layout companion. It contains no
        # invented natural-image prediction or numerical experimental value.
        from matplotlib.patches import Rectangle
        fig=plt.figure(figsize=(9,4.5));page=fig.add_axes([0,0,1,1]);page.axis('off')
        page.set_xlim(0,1);page.set_ylim(0,1)
        for title,y in [('Temporal overlap',.68),('Spatial mismatch',.20)]:
            page.text(.06,y+.22,title,fontsize=17,weight='semibold',color='#17191C')
            page.add_patch(Rectangle((.06,y),.88,.19,color='#191A1C',lw=0))
            for i in range(5):
                page.add_patch(Rectangle((.07+i*.175,y+.015),.162,.160,color='#F2F3F5',lw=0))
        page.plot([.095,.885],[.645,.645],color=COLORS['gt'],lw=4)
        page.plot([.26,.84],[.612,.612],color=COLORS['tastvg'],lw=4)
        for i in range(5):
            x=.11+i*.175
            page.add_patch(Rectangle((x,.225),.042,.105,fill=False,edgecolor=COLORS['gt'],lw=2,ls='--'))
            page.add_patch(Rectangle((x+.047,.218),.049,.115,fill=False,edgecolor=COLORS['tastvg'],lw=2))
        legend(page,.06,.06)
        page.text(.94,.06,'Layout schematic',fontsize=10,ha='right',color='#737980')
        export(fig,'PUBLIC_LAYOUT_schematic',False,0,0);plt.close(fig);return
    fig = plt.figure(figsize=(20,7.2)); page=fig.add_axes([0,0,1,1])
    page.axis('off');page.set_xlim(0,1);page.set_ylim(0,1)
    page.text(.025,.967,'(a) Instance ambiguity',fontsize=17,weight='semibold',va='top',color='#17191C')
    page.text(.025,.91,'Query: '+case['query'],fontsize=10.8,va='top',color='#444950')
    start=.11; width=.50
    for (label,model),bottom in zip([('Ground truth','gt'),('TA-STVG','tastvg'),('TubeDETR','tubedetr')],
                                   [.640,.360,.080]):
        fh=strip(fig,page,start,bottom,width,case['frames'],SOURCE,[model],case)
        page.text(.025,bottom+fh*.50,label,fontsize=12.2,weight='semibold',va='center',color=COLORS[model])
        interval(page,start,bottom-.030,width,case,model,False,3.5)
    page.text(.660,.967,'(b) Temporal vs. spatial grounding',fontsize=16.5,weight='semibold',va='top',color='#17191C')
    page.text(.660,.889,'Temporal overlap',fontsize=13.5,weight='semibold',color='#17191C')
    subset = [temporal[i] for i in [0,2,4]]
    strip(fig,page,.660,.642,.313,subset,BASE)
    interval(page,.727,.591,.246,case,'gt',True)
    interval(page,.727,.552,.246,case,'tastvg',True)
    page.text(.660,.470,'Spatial mismatch',fontsize=13.5,weight='semibold',color='#17191C')
    strip(fig,page,.660,.221,.313,[case['frames'][i] for i in [0,2,4]],SOURCE,['gt','tastvg'],case)
    legend(page,.660,.155)
    page.text(.660,.091,'TA-STVG · cross-domain',fontsize=10.5,color='#6C7178')
    export(fig,'figure1_filmstrip_temporal_spatial',True,21,8);plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','full','panel','schematic'])
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:render(args.mode)
