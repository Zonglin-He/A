"""Private actual-frame root readback after the complete P0 label barrier."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *
import numpy as np

def run():
    import torch
    from PIL import Image,ImageDraw,ImageFont
    from scripts.score_stvg_opd_paper_v1 import truths
    from scripts.run_decota_paper_main_v1 import read_row,frames_for,unpack_expert
    from vg_tta.decota_fixed_full_audit_v1 import top1_support
    from vg_tta.tastvg_paper48_metrics_v1 import xyxy
    from vg_tta.tastvg_oracle_event5_v1 import box_iou
    assert read(BASE/'P0_CPU_COMPLETION.json')['P0_postseal_scoring_complete']
    cases=read(PUB/'P0_ROOT_CASES.json');design=read(BASE/'DESIGN_LOCK.json');records=[]
    out=BASE/'private_root_case_views';out.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for ds in DATASETS:
        stage=design['stages']['P0_'+ds];dense,spans,_=truths(ds,stage);cache=collections.OrderedDict()
        for kind,case in cases[ds].items():
            q=case['query_ordinal'];order=case['order'];at=stage['orders'][order].index(q)
            z=load(BASE/'stages'/('P0_'+ds)/'clean'/order/'on_policy'/f'{at:05}.pt')
            fit=z['fit'];inp=load(BASE/z['input']['path']);row=read_row(ds,q)
            frames,ids,_=frames_for(ds,row,cache);ex=unpack_expert(inp['expert']);support=top1_support(ex)
            eligible=[(pos,e) for pos,e in support if ids[pos] in dense[q]]
            if not eligible:
                records.append(dict(dataset=ds,kind=kind,query_ordinal=q,status='no_GT_on_admitted_frame'));continue
            scored=[(float(box_iou(xyxy(fit['final'][pos].numpy(),row['input']['width'],row['input']['height']),dense[q][ids[pos]])-box_iou(xyxy(fit['before'][pos].numpy(),row['input']['width'],row['input']['height']),dense[q][ids[pos]])),pos,e) for pos,e in eligible]
            _,pos,e=(max(scored,key=lambda x:x[0]) if kind=='success' else min(scored,key=lambda x:x[0]))
            width,height=row['input']['width'],row['input']['height'];scale=min(600/width,430/height)
            w,h=round(width*scale),round(height*scale)
            panels=[]
            for state,boxes in [('Before',fit['before']),('After',fit['final'])]:
                canvas=Image.new('RGB',(600,510),'white');draw=ImageDraw.Draw(canvas)
                image=Image.fromarray(frames[pos]).resize((w,h));canvas.paste(image,(0,65))
                title=f'{ds} / {kind} / anonymous query {q} / {state}'
                draw.text((6,5),title,fill='black',font=font)
                draw.text((6,27),f"current {case['delta_current_v']*100:+.2f} pp; inherited {case['delta_inherited_v']*100:+.2f} pp",fill='black',font=font)
                for label,box,color in [('GT',dense[q][ids[pos]],'#30d230'),('DINO Top1',xyxy(e,width,height),'#00cce0'),('Student',xyxy(boxes[pos].numpy(),width,height),'#ff483b')]:
                    rect=np.asarray(box,float)*scale;rect[[1,3]]+=65
                    draw.rectangle(rect.tolist(),outline=color,width=3)
                    draw.text((max(0,float(rect[0])),max(65,float(rect[1])-17)),label,fill=color,font=font)
                draw.text((6,65+h+8),f"Physical frame {ids[pos]}, observed by DINO. Native WHEN is fixed.",fill='black',font=font)
                panels.append(canvas)
            combined=Image.new('RGB',(1200,510),'white');combined.paste(panels[0],(0,0));combined.paste(panels[1],(600,0))
            file=out/f'{ds}_{kind}.png';combined.save(file)
            # Caption is retained only in this private root view receipt, never exported.
            records.append(dict(dataset=ds,kind=kind,query_ordinal=q,frame_id=ids[pos],path=str(file.relative_to(ROOT)),
                caption_for_actual_identity_readback=row['input']['caption'],private_media_not_public=True))
    write(BASE/'P0_PRIVATE_CASE_VIEWS.json',dict(status='rendered_pending_actual_root_view',records=records,
        private_media_exported=False,GT_after_P0_seal=True))
    print('PRIVATE_CASE_VIEWS_RENDERED',len(records))

if __name__=='__main__':run()
