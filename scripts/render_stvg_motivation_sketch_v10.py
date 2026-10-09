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
    lock=read(BASE/'DESIGN_LOCK.json')
    revision=BASE/'HUMAN_SCORE_LABEL_RUNTIME001.json'
    if revision.exists():
        r=read(revision)
        assert r['original_code_sha256']==lock['code_sha256']
        assert r['code_sha256']==sha(Path(__file__)) and r['human_allows_score_labels']
    else:
        assert lock['code_sha256']==sha(Path(__file__))
    for p,digest in lock['source_pins'].items():assert sha(ROOT/p)==digest
    assert read(CROSS/'ROOT_AUDIT.json')['status']=='pass'
    assert read(OLD/'FINAL_COMPLETION.json')['status']=='completed_verified_and_published'
    assert read(OLD/'SCALAR_AUDIT.json')['rows_sha256']==sha(OLD/'SCALAR_ROWS.json')
    return lock


def prepare():
    import numpy as np
    from PIL import Image
    from scripts.score_stvg_motivation_cross_domain_v6 import read_truths
    from vg_tta.exact_frame_decode_audit_v2 import decode
    lock=verify();direction='hc2_to_vidstg';ordinal=lock['illustration_parent']
    assert ordinal==61 and lock['human_authorized_joint_MLLM_reference']
    cross_roster=read(CROSS/direction/'ROSTER.json');r=cross_roster['rows'][ordinal]
    old_roster=read(OLD/'ROSTER.json');o=old_roster['rows'][ordinal]
    assert r['input']==o['input'] and r['frame_ids']==o['frame_ids'] and r['source']==o['source']
    newscalars={(z['model'],z['parent']):z for z in read(CROSS/'SCALAR_ROWS.json') if z['direction']==direction}
    oldscalar=next(z for z in read(OLD/'SCALAR_ROWS.json') if z['backbone']=='ptd' and z['parent']==ordinal and z['condition']=='clean')
    preds={m:read(CROSS/direction/m/'predictions'/f'{ordinal:05}.json') for m in MODELS[:2]}
    pp=OLD/'ptd/predictions/clean'/f'{ordinal:05}.json';pr=read(pp.with_suffix('.receipt.json'))
    assert sha(pp)==pr['sha256'];native=read(pp)
    assert native['native_parity'] and native['parameter_updates']==0
    assert native['ordinal']==ordinal and native['condition']=='clean'
    barrier=read(OLD/'GLOBAL_PREDICTION_BARRIER.json')
    assert sha(OLD/'ptd/PREDICTION_BARRIER.json')==barrier['barriers']['ptd']
    preds['ptd']=dict(interval=native['intervals'][0],boxes=native['boxes'][0],
        box_frame_ids=native['box_frame_ids'],pixel_sha256=native['pixel_sha256'])
    truth,spans,provenance=read_truths(direction,cross_roster);gt=truth[ordinal];span=spans[ordinal]
    pixels,ids=decode(r['input']);digest=hashlib.sha256(pixels.tobytes()).hexdigest()
    assert ids==r['frame_ids'] and all(z['pixel_sha256']==digest for z in preds.values())
    eligible=[fid for fid in ids if fid in gt and span[0]<=fid<span[1] and all(
        z['interval'][0]<=fid<z['interval'][1] and min(z['box_frame_ids'])<=fid<=max(z['box_frame_ids']) for z in preds.values())]
    assert len(eligible)>=3
    chosen=[eligible[j] for j in np.linspace(0,len(eligible)-1,min(5,len(eligible))).round().astype(int)]
    assert len(set(chosen))==len(chosen)
    w,h=r['input']['width'],r['input']['height'];frames=[]
    for fid in chosen:
        image=pixels[ids.index(fid)];p=BASE/'private_frames'/f'frame_{fid:05}.png'
        p.parent.mkdir(exist_ok=True);assert not p.exists();Image.fromarray(image).save(p)
        pred={}
        for m,z in preds.items():
            boxes=np.asarray(z['boxes'],float)*[w,h,w,h]
            pred[m]=[float(np.interp(fid,z['box_frame_ids'],boxes[:,j])) for j in range(4)]
        frames.append(dict(frame_id=fid,path=str(p.relative_to(BASE)),sha256=sha(p),
            pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),GT_box=list(map(float,gt[fid])),predictions=pred))
    native_scores={m:dict(tIoU=newscalars[m,ordinal]['tIoU'],sIoU=newscalars[m,ordinal]['sIoU']) for m in MODELS[:2]}
    native_scores['ptd']=dict(tIoU=oldscalar['temporal'][0],sIoU=oldscalar['spatial'][0])
    assert all(z['tIoU']>.5 and z['sIoU']<=.5 for z in native_scores.values())
    # Independent spatial/temporal formula already audited for the two native
    # cross-domain outputs; reuse exactly the same saved scalar definition.
    endpoints=[span[0],span[1]]+[v for z in preds.values() for v in z['interval']]
    lo=max(0,min(endpoints)-30);hi=min(r['input']['frame_count'],max(endpoints)+30)
    write(BASE/'PRIVATE_CASE.json',dict(status='three_actual_native_rows_bound',query=r['input']['caption'],
        direction=direction,ordinal=ordinal,frames=frames,GT_interval=span,
        models={m:dict(interval=preds[m]['interval'],scores=native_scores[m],
            training_scope='joint-domain supervised reference' if m=='ptd' else 'HC2-source-only → VidSTG') for m in MODELS},
        timeline_extent=[lo,hi],canonical_RGB_sha256=digest,source_input_sha256=sha(CROSS/direction/'ROSTER.json'),
        source_prediction_sha256={m:sha(pp) if m=='ptd' else sha(CROSS/direction/m/'predictions'/f'{ordinal:05}.json') for m in MODELS},
        GT_overlay_provenance=provenance,candidate_index=0,case_selection=lock['case_selection'],
        selected_after_statistics=True,prevalence_claim=False,MLLM_is_not_cross_domain=True,
        new_model_calls=0,new_GT_scoring=0,active_P1_payload_access=False,time=time.time()))
    # Anonymous exactly saved scalar companion for transparent bar heights.
    write(BASE/'CASE_SCALAR_READBACK.json',dict(status='pass',scope='exact saved native scalar index zero; case-only display',
        models=native_scores,GT_and_prediction_intervals={m:preds[m]['interval'] for m in MODELS},
        independent_cross_domain_root_sha256=sha(CROSS/'ROOT_AUDIT.json'),old_MLLM_scalar_audit_sha256=sha(OLD/'SCALAR_AUDIT.json'),
        joint_MLLM_not_a_cross_domain_cell=True,no_new_scores=True,statistical_inference=False,
        full_original_negative_quadrants_retained=True,chosen_case_has_uniform_shared_frames=len(chosen),time=time.time()))
    print(json.dumps(dict(status='actual_case_prepared',actual_frames=len(chosen),actual_models=3,MLLM_scope='joint reference')))


def render():
    import re
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    verify();case=read(BASE/'PRIVATE_CASE.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(16.5,8));page=fig.add_axes([0,0,1,1]);page.axis('off')
    page.set_xlim(0,1);page.set_ylim(0,1)
    page.text(.12,.966,'Query: '+case['query'],fontsize=12,va='top',color='#3D434A')
    # Each row contains exact GT/native boxes, two physical-time ranges and
    # the two saved native scores. Only their six numeric labels are shown;
    # timelines retain no tick numbers, timestamps, percentages or counts.
    x=.12;width=.62;gap=.004;n=len(case['frames']);fw=(width-(n-1)*gap)/n
    fh=fw*16.5*.75/8
    labels={'tastvg':'TA-STVG','tubedetr':'TubeDETR','ptd':'MLLM'}
    for m,y in zip(MODELS,[.695,.415,.135]):
        page.text(.02,y+fh*.63,labels[m],fontsize=14,weight='semibold',va='center',color='#252930')
        if m=='ptd':
            page.text(.02,y+fh*.34,'PTD / Qwen-VL',fontsize=8.8,color='#68717B')
            page.text(.02,y+fh*.18,'Joint-trained',fontsize=8.8,color='#68717B')
        page.add_patch(Rectangle((x-.004,y-.008),width+.008,fh+.016,facecolor='#18191B',edgecolor='none'))
        for py in [y-.0057,y+fh+.0021]:
            for px in np.arange(x-.001,x+width-.003,.011):
                page.add_patch(Rectangle((px,py),.0045,.0038,facecolor='white',edgecolor='none'))
        for j,frm in enumerate(case['frames']):
            p=BASE/frm['path'];assert sha(p)==frm['sha256']
            im=np.asarray(Image.open(p).convert('RGB'));assert hashlib.sha256(im.tobytes()).hexdigest()==frm['pixel_sha256']
            ax=fig.add_axes([x+j*(fw+gap),y,fw,fh]);ax.axis('off');ax.imshow(im)
            for box,color,ls in [(frm['GT_box'],COLORS['gt'],'--'),(frm['predictions'][m],COLORS['prediction'],'-')]:
                x1,y1,x2,y2=box;ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,lw=2,edgecolor=color,linestyle=ls))
        a,b=case['timeline_extent']
        for name,span,ty,color in [('Prediction',case['models'][m]['interval'],y-.034,COLORS['prediction']),
                                  ('GT',case['GT_interval'],y-.065,COLORS['gt'])]:
            page.text(x-.011,ty,name,ha='right',va='center',fontsize=9,color=color)
            page.plot([x,x+width],[ty,ty],color='#E4E7EA',lw=2,solid_capstyle='butt')
            page.plot([x+width*(span[0]-a)/(b-a),x+width*(span[1]-a)/(b-a)],[ty,ty],
                      color=color,lw=4,solid_capstyle='butt')
        scoreax=fig.add_axes([.81,y-.004,.165,fh+.014]);scoreax.set_ylim(0,1.15);scoreax.set_xlim(-.6,1.6)
        values=[case['models'][m]['scores'][z] for z in ['tIoU','sIoU']]
        assert all(0<=v<=1 for v in values)
        scoreax.bar([0,1],[1,1],width=.48,color='#F0F2F4',edgecolor='none',zorder=1)
        scoreax.bar([0,1],values,width=.48,color=[COLORS['time'],COLORS['prediction']],edgecolor='none',zorder=2)
        for j,value in enumerate(values):
            scoreax.text(j,value+.036,f'{value:.3f}',ha='center',va='bottom',fontsize=11,
                         color=COLORS['time'] if j==0 else COLORS['prediction'])
        scoreax.set_xticks([0,1],['tIoU','sIoU'],fontsize=11);scoreax.set_yticks([])
        scoreax.tick_params(length=0,pad=8)
        for spine in scoreax.spines.values():spine.set_visible(False)
    texts=[t.get_text() for ax in fig.axes for t in ax.texts]
    number_text=[t for t in texts if re.search(r'\d',t)]
    assert len(number_text)==6 and all(re.fullmatch(r'0\.\d{3}',t) for t in number_text),number_text
    files={}
    for ext in ['png','pdf','svg']:
        p=BASE/('figure1_three_backbone_sketch.'+ext);assert not p.exists()
        fig.savefig(p,dpi=260,facecolor='white');files[p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
    plt.close(fig)
    write(BASE/'RENDER.json',dict(status='rendered_pending_actual_view',files=files,numeric_annotations=6,
        numeric_annotations_only_saved_native_score_labels=True,score_label_decimal_places=3,
        actual_image_panels=3*n,unique_real_frames=n,MLLM_marked_joint_trained=True,
        bars_are_case_scores_not_aggregate_rates=True,bar_heights_not_rescaled_or_clipped=True,
        same_case_same_RGB_same_GT_for_all_rows=True,GT_predictions_and_intervals_unchanged=True,
        private_media_exported=False,new_model_calls=0,new_GT_scoring=0,active_P1_payload_access=False,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','render']);a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:render()
