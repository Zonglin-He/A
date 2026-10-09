"""Human sketch: three filmstrip rows, GT/predicted time bars and score bars.

TA-STVG/TubeDETR are source-only cross-domain outputs. PTD/Qwen3-VL is a
separately labeled joint-trained reference, not a third cross-domain model.
Candidate zero is reused from its sealed historical native outputs. Case
selection is post-hoc illustration and does not modify any aggregate result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/stvg_motivation_sketch_v10'
TITLE_BASE=ROOT/'artifacts/stvg_motivation_backbone_names_v11'
CROSS=ROOT/'artifacts/stvg_motivation_cross_domain_v6'
OLD=ROOT/'artifacts/stvg_native_support_fig1_v1/uniform64_v2'
MODELS=['tastvg','tubedetr','ptd']
COLORS={'gt':'#22875B','prediction':'#E66B58','time':'#5A8CA8'}


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,z):
    assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')


def verify():
    lock=read(TITLE_BASE/'DESIGN_LOCK.json')
    revision=TITLE_BASE/'HUMAN_LAYOUT_RUNTIME001.json'
    latest=TITLE_BASE/'HUMAN_GT_OVERLAY_RUNTIME002.json'
    if latest.exists():
        rr=read(latest);previous=read(revision)
        assert previous['original_renderer_sha256']==lock['renderer_sha256']
        assert rr['previous_renderer_sha256']==previous['renderer_sha256']
        assert rr['renderer_sha256']==sha(Path(__file__))
    elif revision.exists():
        rr=read(revision);assert rr['original_renderer_sha256']==lock['renderer_sha256']
        assert rr['renderer_sha256']==sha(Path(__file__))
    else:assert lock['renderer_sha256']==sha(Path(__file__))
    for p,h in lock['source_pins'].items():assert sha(ROOT/p)==h
    assert read(BASE/'ROOT_CLOSING_RECEIPT.json')['status']=='complete'
    return lock



def render():
    import re
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.patheffects as pe
    verify();case=read(BASE/'PRIVATE_CASE.json')
    output=TITLE_BASE/'GT_overlay_revision002' if (TITLE_BASE/'HUMAN_GT_OVERLAY_RUNTIME002.json').exists() else TITLE_BASE/'font_revision001' if (TITLE_BASE/'HUMAN_LAYOUT_RUNTIME001.json').exists() else TITLE_BASE
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(16.5,8));page=fig.add_axes([0,0,1,1]);page.axis('off')
    page.set_xlim(0,1);page.set_ylim(0,1)
    page.text(.12,.966,'Query: '+case['query'],fontsize=18,va='top',color='#3D434A')
    # Each row contains exact GT/native boxes, two physical-time ranges and
    # the two saved native scores. Only their six numeric labels are shown;
    # timelines retain no tick numbers, timestamps, percentages or counts.
    x=.12;width=.62;gap=.004;n=len(case['frames']);fw=(width-(n-1)*gap)/n
    fh=fw*16.5*.75/8
    labels={'tastvg':'TA-STVG','tubedetr':'TubeDETR','ptd':'PTD /\nQwen3-VL'}
    for m,y in zip(MODELS,[.695,.415,.135]):
        page.text(.02,y+fh*.63,labels[m],fontsize=16 if m=='ptd' else 17,weight='semibold',va='center',color='#252930')
        page.add_patch(Rectangle((x-.004,y-.008),width+.008,fh+.016,facecolor='#18191B',edgecolor='none'))
        for py in [y-.0057,y+fh+.0021]:
            for px in np.arange(x-.001,x+width-.003,.011):
                page.add_patch(Rectangle((px,py),.0045,.0038,facecolor='white',edgecolor='none'))
        for j,frm in enumerate(case['frames']):
            p=BASE/frm['path'];assert sha(p)==frm['sha256']
            im=np.asarray(Image.open(p).convert('RGB'));assert hashlib.sha256(im.tobytes()).hexdigest()==frm['pixel_sha256']
            ax=fig.add_axes([x+j*(fw+gap),y,fw,fh]);ax.axis('off');ax.imshow(im)
            for box,color,ls in [(frm['GT_box'],COLORS['gt'],'--'),(frm['predictions'][m],COLORS['prediction'],'-')]:
                x1,y1,x2,y2=box
                is_gt=color==COLORS['gt']
                patch=Rectangle((x1,y1),x2-x1,y2-y1,fill=False,lw=3 if is_gt else 2,edgecolor=color,linestyle=ls)
                if is_gt:patch.set_path_effects([pe.Stroke(linewidth=4.5,foreground='white'),pe.Normal()])
                ax.add_patch(patch)
                if is_gt:
                    ax.text(x2+4,(y1+y2)/2,'GT',fontsize=10,weight='bold',color=color,va='center',ha='left',
                        bbox=dict(facecolor='white',edgecolor=color,linewidth=.8,pad=1.8,alpha=.95))
        a,b=case['timeline_extent']
        for name,span,ty,color in [('Prediction',case['models'][m]['interval'],y-.034,COLORS['prediction']),
                                  ('GT',case['GT_interval'],y-.065,COLORS['gt'])]:
            page.text(x-.011,ty,name,ha='right',va='center',fontsize=12,color=color)
            page.plot([x,x+width],[ty,ty],color='#E4E7EA',lw=2,solid_capstyle='butt')
            page.plot([x+width*(span[0]-a)/(b-a),x+width*(span[1]-a)/(b-a)],[ty,ty],
                      color=color,lw=4,solid_capstyle='butt')
        scoreax=fig.add_axes([.81,y-.004,.165,fh+.014]);scoreax.set_ylim(0,1.15);scoreax.set_xlim(-.6,1.6)
        values=[case['models'][m]['scores'][z] for z in ['tIoU','sIoU']]
        assert all(0<=v<=1 for v in values)
        scoreax.bar([0,1],[1,1],width=.48,color='#F0F2F4',edgecolor='none',zorder=1)
        scoreax.bar([0,1],values,width=.48,color=[COLORS['time'],COLORS['prediction']],edgecolor='none',zorder=2)
        for j,value in enumerate(values):
            scoreax.text(j,value+.036,f'{value:.3f}',ha='center',va='bottom',fontsize=14,
                         color=COLORS['time'] if j==0 else COLORS['prediction'])
        scoreax.set_xticks([0,1],['tIoU','sIoU'],fontsize=14);scoreax.set_yticks([])
        scoreax.tick_params(length=0,pad=8)
        for spine in scoreax.spines.values():spine.set_visible(False)
    texts=[t.get_text() for ax in fig.axes for t in ax.texts]
    number_text=[t for t in texts if re.search(r'\d',t) and t!='PTD /\nQwen3-VL']
    assert len(number_text)==6 and all(re.fullmatch(r'0\.\d{3}',t) for t in number_text),number_text
    files={}
    for ext in ['png','pdf','svg']:
        p=output/('figure1_three_backbone_sketch.'+ext);assert not p.exists()
        fig.savefig(p,dpi=260,facecolor='white');files[p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
    plt.close(fig)
    write(output/'RENDER.json',dict(status='rendered_pending_actual_view',files=files,numeric_annotations=6,
        numeric_annotations_only_saved_native_score_labels=True,score_label_decimal_places=3,
        actual_image_panels=3*n,unique_real_frames=n,explicit_GT_box_labels=3*n,GT_geometry_unchanged=True,MLLM_backbone_named=True,joint_training_scope_in_caption_only=True,
        bars_are_case_scores_not_aggregate_rates=True,bar_heights_not_rescaled_or_clipped=True,
        same_case_same_RGB_same_GT_for_all_rows=True,GT_predictions_and_intervals_unchanged=True,
        private_media_exported=False,new_model_calls=0,new_GT_scoring=0,active_P1_payload_access=False,time=time.time()))


if __name__=='__main__':
    render()
