"""Fixed original cases: Native/Before/After/GT and admitted-feedback markers.

Only saved, already sealed P3 and Fig1 evidence is read. No inference, selection
from new scores, or public export of real RGB/query/GT geometry is performed.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import hashlib
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,status,sha,load
BASE=ROOT/'artifacts/stvg_opd_supplement_v1/saved_evidence'
PUB=ROOT/'results/stvg_opd_supplement/2026-10-11/saved_evidence'
OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
FIG=ROOT/'output/pdf/stvg_opd_supplement'

def run():
    import torch
    torch.set_num_threads(2)
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old()
    from scripts.run_decota_paper_main_v1 import read_row,frames_for,unpack_expert
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from scripts.stvg_opd_paper_inputs_v1 import observation
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.decota_fixed_full_audit_v1 import top1_support
    from scripts.render_stvg_opd_revised_p1_root_v2 import coordinates,overlap
    from PIL import Image,ImageDraw,ImageFont
    assert read(OLD/'FINAL_COMPLETION.json')['paper_suite_complete']
    lock=ROOT/'artifacts/stvg_opd_design_ablation_v1/actual_fixed_case_views002/CASE_DIAGNOSTIC_LOCK.json'
    cases=read(lock)['records'];definitions=read(OLD/'LATER_DESIGN_LOCK.json')['stages'];gtcache={};images=[];records=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    for case in cases:
        ds=case['dataset'];name=case['stage'].replace('A_','P3_');q=case['query_ordinal'];stage=definitions[name]
        if ds not in gtcache:gtcache[ds]=truths(ds,stage)[0]
        gt=gtcache[ds][q];row=read_row(ds,q);b=read(OLD/'stages'/name/'PREDICTION_BARRIER.json')
        binding=b['logical_records'][f"{case['condition']}/{case['order']}/on_policy/{case['arrival']:05}"]
        path=OLD/binding['path'];assert sha(path)==binding['sha256'];z=load(path);assert z['query_ordinal']==q
        ip=OLD/z['input']['path'];assert sha(ip)==z['input']['sha256'];inp=normalized(load(ip));fit=z['fit'];support=dict(top1_support(unpack_expert(inp['expert'])))
        frames,ids,_=frames_for(ds,row,collections.OrderedDict());frames,pixel,spec=observation(ds,row,case['condition'],frames)
        assert pixel==inp['pixel_sha256'] and spec==inp['corruption_spec'] and ids==inp['frame_ids']
        w,h=row['input']['width'],row['input']['height'];cell=480;scale=min((cell-24)/w,360/h);iw,ih=round(w*scale),round(h*scale)
        canvas=Image.new('RGB',(4*cell,150+len(case['frames'])*(ih+100)),'white');draw=ImageDraw.Draw(canvas)
        title=f"{ds}: fixed historical {case['historical_role'].replace('_',' ')}";draw.text((16,12),title,font=font,fill='#25343c')
        import textwrap
        draw.multiline_text((16,54),'\n'.join(textwrap.wrap(row['input']['caption'],width=130)),font=small,fill='#3d454b')
        draw.text((16,110),'Green: official GT. Orange: student prediction. Blue marker: admitted expert feedback.',font=small,fill='#455762')
        details=[]
        for band,item in enumerate(case['frames']):
            p,fid=item['position'],item['frame_id'];assert ids[p]==fid and fid in gt;y=150+band*(ih+100)
            reference=p in support
            for col,label in enumerate(['Native','Before','After','GT']):
                x=col*cell+12;draw.text((x,y),label,font=font,fill='#25343c');canvas.paste(Image.fromarray(frames[p]).resize((iw,ih)),(x,y+40))
                if label=='GT':pred=None
                else:
                    boxes=inp['native_boxes'] if label=='Native' else fit['before'] if label=='Before' else fit['final'];pred=coordinates(boxes[p].numpy(),w,h)
                for box,color in [(gt[fid],'#159947')]+([] if pred is None else [(pred,'#e9774d')]):
                    box=np.asarray(box)*scale;box[[0,2]]+=x;box[[1,3]]+=y+40
                    draw.rectangle(box.tolist(),outline='white',width=7);draw.rectangle(box.tolist(),outline=color,width=4)
                footer='Feedback frame' if reference else 'Unobserved frame';draw.text((x,y+ih+47),footer,font=small,fill='#245f86' if reference else '#67737a')
                if reference:draw.ellipse((x+iw-27,y+48,x+iw-9,y+66),fill='#245f86',outline='white',width=2)
                if label!='GT':details.append(dict(label=label,frame_id=fid,IoU=float(overlap(pred,gt[fid])),feedback_admitted=reference))
            assert reference==(item['scope']=='admitted observation')
        out=BASE/'private_qualitative'/f"{ds}_{case['historical_role']}.png";out.parent.mkdir(parents=True,exist_ok=True);canvas.save(out);images.append(canvas)
        records.append(dict(dataset=ds,role=case['historical_role'],query_ordinal=q,source_id=case['source_id'],old_fit_sha256=sha(path),old_input_sha256=sha(ip),original_case_lock_sha256=sha(lock),
            actual_RGB_sha256=pixel,private_image_sha256=sha(out),private_image_path=str(out.relative_to(ROOT)),frame_readback=details,Native_Before_After_GT_visible=True,feedback_labels_match_saved_admission=True,no_new_model_calls=True))
    images[0].save(FIG/'fixed_qualitative_cases_private.pdf',save_all=True,append_images=images[1:],resolution=170.)
    write(BASE/'PRIVATE_QUALITATIVE_READBACK.json',dict(status='rendered_pending_actual_root_view',cases=6,panels=48,records=records,new_model_calls=0,case_or_frame_reselection=False,private_media_not_public=True,time=time.time()))
    # The figure has three backbones; actual saved model metrics/geometry remain local.
    cross=ROOT/'artifacts/stvg_motivation_cross_domain_v6';native=ROOT/'artifacts/stvg_native_support_fig1_v1/uniform64_v2';v10=ROOT/'artifacts/stvg_motivation_sketch_v10';case=read(v10/'PRIVATE_CASE.json');direction=case['direction'];q=case['ordinal'];roster=read(cross/direction/'ROSTER.json');row=roster['rows'][q]
    assert read(cross/'ROOT_CLOSING_RECEIPT.json')['status']=='complete' and read(native/'FINAL_COMPLETION.json')['status']=='completed_verified_and_published'
    from scripts.score_stvg_motivation_cross_domain_v6 import read_truths
    from vg_tta.stvg_motivation_quadrants_v6 import independent_components
    gt,spans,provenance=read_truths(direction,roster); metrics={};geometry=0
    for model in ['tastvg','tubedetr','ptd']:
        p=cross/direction/model/'predictions'/f'{q:05}.json' if model!='ptd' else native/'ptd/predictions/clean'/f'{q:05}.json'
        assert sha(p)==case['source_prediction_sha256'][model];raw=read(p)
        boxes=raw['boxes'] if model!='ptd' else raw['boxes'][0];interval=raw['interval'] if model!='ptd' else raw['intervals'][0];ids=raw['box_frame_ids']
        result=independent_components(boxes,ids,interval,gt[q],spans[q],row['input']['width'],row['input']['height'])
        for k in ['tIoU','sIoU']:assert abs(result[k]-case['models'][model]['scores'][k])<2e-10
        pix=np.asarray(boxes)*[row['input']['width'],row['input']['height'],row['input']['width'],row['input']['height']]
        for frame in case['frames']:
            fid=frame['frame_id'];assert np.array_equal(frame['GT_box'],gt[q][fid])
            predicted=np.array([np.interp(fid,ids,pix[:,j]) for j in range(4)]);assert np.max(np.abs(predicted-frame['predictions'][model]))<2e-10;geometry+=8
        metrics[model]=dict(tIoU=result['tIoU'],sIoU=result['sIoU'],native_prediction_sha256=sha(p),source_only_cross_domain=model!='ptd',joint_training_reference=model=='ptd')
    write(PUB/'FIGURE1_CASE_PROVENANCE.json',dict(status='actual_saved_case_metrics_and_GT_overlay_readback_pass',scalar_comparisons=6,box_coordinate_checks=geometry,models=metrics,actual_case_metadata_sha256=sha(v10/'PRIVATE_CASE.json'),
        original_case_root_sha256=sha(v10/'ROOT_CASE_READBACK.json'),final_v11_root_sha256=sha(ROOT/'artifacts/stvg_motivation_backbone_names_v11/ROOT_CLOSING_RECEIPT.json'),
        case_candidate_index=0,posthoc_illustration_not_prevalence=True,joint_MLLM_excluded_strict_cross_domain_statistics=True,new_model_calls=0,old_saved_GT_only=True,private_RGB_query_GT_not_exported=True,time=time.time()))
    write(BASE/'CASE_COMPLETION.json',dict(status='pending_actual_root_visual_review',cases=6,original_case_lock_sha256=sha(lock),private_PDF_sha256=sha(FIG/'fixed_qualitative_cases_private.pdf'),figure1_provenance_sha256=sha(PUB/'FIGURE1_CASE_PROVENANCE.json'),model_calls=0,time=time.time()))
    print('FIXED_CASES_AND_FIG1_ACTUAL_PROVENANCE_PASS',flush=True)

if __name__=='__main__':run()
