"""Three original fixed cases, four observations and real intermediate frames."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import sys
import textwrap
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,status,sha,load
BASE=ROOT/'artifacts/stvg_opd_claim_strengthening_v1/aligned_cases'
OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
FIG=ROOT/'output/pdf/stvg_opd_claim_strengthening'


def prepare():
    lock=ROOT/'artifacts/stvg_opd_design_ablation_v1/actual_fixed_case_views002/CASE_DIAGNOSTIC_LOCK.json'
    records=read(lock)['records']
    selected=[]
    for ds,role in [('hc2','success'),('vidstg','success'),('vidstg','current_harm')]:
        matching=[r for r in records if r['dataset']==ds and r['historical_role']==role]
        assert len(matching)==1;selected.append(matching[0])
    write(BASE/'CASE_LOCK.json',dict(status='locked_before_rendering',records=selected,
        source_case_lock_sha256=sha(lock),selection='existing HC success, Vid success and Vid harm; no new outcome selection',
        four_observations_and_original_grid_midpoints=True,new_model_calls=0,
        new_comparison_scores_read=False,new_GT_experiment=False,time=time.time()))
    pins={p:sha(ROOT/p) for p in ['scripts/render_stvg_opd_aligned_observation_cases_v1.py',
        'protocols/stvg_opd_claim_strengthening_v1.md','scripts/stvg_opd_legacy_input_bridge003.py',
        'scripts/render_stvg_opd_revised_p1_root_v2.py']}
    write(BASE/'RUNTIME_LOCK.json',dict(pins=pins,case_lock_sha256=sha(BASE/'CASE_LOCK.json'),
        old_completed_suite_sha256=sha(OLD/'FINAL_COMPLETION.json'),new_model_calls=0,time=time.time()))


def run():
    import torch
    torch.set_num_threads(2)
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old()
    from scripts.run_decota_paper_main_v1 import read_row,frames_for
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from scripts.stvg_opd_paper_inputs_v1 import observation
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.render_stvg_opd_revised_p1_root_v2 import coordinates
    from PIL import Image,ImageDraw,ImageFont
    runtime=read(BASE/'RUNTIME_LOCK.json')
    for p,h in runtime['pins'].items():assert sha(ROOT/p)==h
    assert read(OLD/'FINAL_COMPLETION.json')['paper_suite_complete']
    assert sha(BASE/'CASE_LOCK.json')==runtime['case_lock_sha256']
    definitions=read(OLD/'LATER_DESIGN_LOCK.json')['stages'];pages=[];details=[]
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    titlefont=ImageFont.truetype(font,28);labelfont=ImageFont.truetype(font,23);small=ImageFont.truetype(font,18)
    for case in read(BASE/'CASE_LOCK.json')['records']:
        ds=case['dataset'];name=case['stage'].replace('A_','P3_');stage=definitions[name];q=case['query_ordinal']
        barrier=read(OLD/'stages'/name/'PREDICTION_BARRIER.json')
        binding=barrier['logical_records'][f"{case['condition']}/{case['order']}/on_policy/{case['arrival']:05}"]
        p=OLD/binding['path'];assert sha(p)==binding['sha256'];z=load(p)
        ip=OLD/z['input']['path'];assert sha(ip)==z['input']['sha256'];inp=normalized(load(ip))
        row=read_row(ds,q);frames,ids,reused=frames_for(ds,row,collections.OrderedDict())
        frames,pixel,spec=observation(ds,row,case['condition'],frames)
        assert pixel==inp['pixel_sha256'] and spec==inp['corruption_spec'] and ids==inp['frame_ids']
        nominal=inp['expert']['positions4'];assert len(nominal)==4 and nominal==sorted(set(nominal))
        attempted=set(inp['expert']['actual_observation_positions']);admitted=set(z['fit']['positions'])
        mids=[(a+b)//2 for a,b in zip(nominal[:-1],nominal[1:]) if a+1<b]
        positions=sorted(set(nominal+mids));gt=truths(ds,stage)[0][q]
        w,h=row['input']['width'],row['input']['height'];cell=270;left=120
        scale=min((cell-14)/w,190/h);iw,ih=round(w*scale),round(h*scale)
        canvas=Image.new('RGB',(left+cell*len(positions)+10,180+4*(ih+42)+80),'white');draw=ImageDraw.Draw(canvas)
        label=f"{'HC-STVG-v2' if ds=='hc2' else 'VidSTG'} - historical {'improvement' if case['historical_role']=='success' else 'regression'}"
        draw.text((14,12),label,font=titlefont,fill='#26343c')
        draw.multiline_text((14,52),'\n'.join(textwrap.wrap(row['input']['caption'],width=105)),font=labelfont,fill='#26343c',spacing=4)
        readback=[]
        for col,pos in enumerate(positions):
            x=left+col*cell;fid=ids[pos]
            if pos in nominal:
                k=nominal.index(pos)+1
                header=f'E{k}: '+('admitted' if pos in admitted else 'rejected' if pos in attempted else 'not called')
                color='#245f86'
            else:header='Between observations';color='#59636c'
            draw.text((x,137),header,font=small,fill=color)
            for band,kind in enumerate(['Native','Before','After','GT']):
                y=180+band*(ih+42);canvas.paste(Image.fromarray(frames[pos]).resize((iw,ih)),(x,y))
                if col==0:draw.text((14,y+ih//2-14),kind,font=labelfont,fill='#26343c')
                gtbox=gt.get(fid)
                predicted=None if kind=='GT' else coordinates((inp['native_boxes'] if kind=='Native' else z['fit']['before'] if kind=='Before' else z['fit']['final'])[pos].numpy(),w,h)
                for box,color in ([] if gtbox is None else [(gtbox,'#159947')])+([] if predicted is None else [(predicted,'#e9774d')]):
                    scaled=np.asarray(box)*scale;scaled[[0,2]]+=x;scaled[[1,3]]+=y
                    draw.rectangle(scaled.tolist(),outline='white',width=6);draw.rectangle(scaled.tolist(),outline=color,width=3)
                if gtbox is None:draw.text((x+5,y+5),'GT unavailable',font=small,fill='#687078',stroke_width=1,stroke_fill='white')
                if pos in nominal:draw.ellipse((x+iw-23,y+7,x+iw-7,y+23),fill='#245f86',outline='white',width=2)
                readback.append(dict(kind=kind,position=pos,physical_frame_id=fid,
                    nominal_expert_observation=pos in nominal,attempted_expert=pos in attempted,
                    admitted_expert=pos in admitted,official_GT_present=gtbox is not None))
        y=canvas.height-57
        draw.text((14,y),'Green: official GT   Orange: actual saved student box   Blue: original expert observation',font=small,fill='#47515a')
        draw.text((14,y+27),'All columns use real decoded frames. Intermediate boxes are saved native-grid outputs.',font=small,fill='#47515a')
        out=BASE/'private_frames'/f"{ds}_{case['historical_role']}.png";out.parent.mkdir(parents=True,exist_ok=True);canvas.save(out);pages.append(canvas)
        details.append(dict(dataset=ds,historical_role=case['historical_role'],query_ordinal=q,
            original_fit_sha256=sha(p),original_input_sha256=sha(ip),pixel_sha256=pixel,
            frames=readback,nominal_observations=4,intermediate_original_positions=len(mids),
            image_sha256=sha(out),panels=4*len(positions),new_model_calls=0,new_GT_experiment=False))
    FIG.mkdir(parents=True,exist_ok=True);pdf=FIG/'aligned_observations_private.pdf'
    pages[0].save(pdf,save_all=True,append_images=pages[1:],resolution=170.)
    write(BASE/'PRIVATE_READBACK.json',dict(status='rendered_pending_actual_PDF_pixel_view',cases=3,
        pages=3,records=details,panels=sum(r['panels'] for r in details),
        no_reinference=True,GT_read='sealed_historical_offline_only',new_model_calls=0,
        original_RGB_and_GT_geometry_preserved=True,private_media_not_public=True,
        pdf_sha256=sha(pdf),time=time.time()))
    status(BASE/'STATUS.json',dict(status='pending_actual_root_PDF_pixel_view',cases=3,pages=3,
        new_model_calls=0,experiment_complete=False,time=time.time()))
    print('ALIGNED_OBSERVATION_CASES_RENDERED',3,flush=True)


if __name__=='__main__':
    if not (BASE/'RUNTIME_LOCK.json').exists():prepare()
    run()
