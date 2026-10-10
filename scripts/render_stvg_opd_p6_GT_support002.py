"""Display-only posthoc RGB frame revision retains all cases/outputs and adds actual GT support frames."""
import collections,os,sys,textwrap,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p6_existing_common001 import *
def rgb(case,z,orig,inp,row,truth,span,directory):
    from PIL import Image,ImageDraw,ImageFont
    from scripts.run_decota_paper_main_v1 import frames_for
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    from scripts.render_stvg_opd_revised_p1_root_v2 import coordinates
    ds=case['dataset'];frames,ids,reuse=frames_for(ds,row,collections.OrderedDict())
    frames,pixel,spec=observation(row,'clean',frames)
    assert ids==z['frame_ids']==inp['frame_ids'] and pixel==z['pixel_sha256']==inp['pixel_sha256'] and spec is None
    valid=[i for i,fid in enumerate(ids) if fid in truth]
    chosen=[]
    if valid:
        chosen=[valid[0],valid[len(valid)//2],valid[-1]]
    for label in ['Native','TemporalHead']:
        iv=z['predictions'][label]['physical_interval']
        chosen.extend([int(np.argmin(np.abs(np.asarray(ids)-iv[0]))),int(np.argmin(np.abs(np.asarray(ids)-(iv[1]-1))))])
    chosen=list(dict.fromkeys(chosen))[:5]
    for i in np.linspace(0,len(ids)-1,5).round().astype(int):
        if len(chosen)<5 and int(i) not in chosen:chosen.append(int(i))
    chosen=sorted(chosen)
    width,height=row['input']['width'],row['input']['height'];w=310;h=round(height*w/width)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    big=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    canvas=Image.new('RGB',(110+len(chosen)*320,160+3*(h+36)+190),'white');draw=ImageDraw.Draw(canvas)
    draw.text((10,8),f'{ds} {case["kind"]}: existing temporal and original spatial history',font=big,fill='black')
    draw.multiline_text((10,36),'\n'.join(textwrap.wrap(row['input']['caption'],width=140)),font=font,fill='black')
    draw.text((10,112),'Post-hoc illustrative frames; GT green, prediction coral. Original full spatial predecessors are retained.',font=font,fill='black')
    for band,a in enumerate(['Native','TemporalHead','SpatialOPD']):
        y=155+band*(h+36);draw.text((5,y+24),a,font=font,fill='black')
        pr=z['predictions'][a]
        for col,pos in enumerate(chosen):
            x=110+col*320;fid=ids[pos];canvas.paste(Image.fromarray(frames[pos]).resize((w,h)),(x,y+26))
            temporal_active=pr['physical_interval'][0]<=fid<pr['physical_interval'][1]
            draw.text((x,y),f'Frame {fid} | '+('in prediction' if temporal_active else 'outside prediction'),font=font,fill='black')
            boxes=[('Pred',coordinates(pr['boxes'][pos].numpy(),width,height),'#f05245')]
            if fid in truth:boxes.append(('GT',np.asarray(truth[fid]),'#159947'))
            else:draw.text((x+5,y+31),'No spatial GT at this frame',font=font,fill='white',stroke_width=1,stroke_fill='black')
            for tag,box,color in boxes:
                b=np.asarray(box,float)*(w/width);b[[0,2]]+=x;b[[1,3]]+=y+26
                draw.rectangle(b.tolist(),outline='white',width=5);draw.rectangle(b.tolist(),outline=color,width=3)
                draw.text((max(x,min(x+w-45,b[0])),max(y+26,b[1]-18)),tag,font=font,fill=color,stroke_width=1,stroke_fill='white')
    y=155+3*(h+36)+10;left=160;extent=ids[-1]+1-ids[0];scale=(canvas.width-left-20)/extent
    for i,(label,iv,color) in enumerate([('GT',span,'#159947')]+[(a,z['predictions'][a]['physical_interval'],'#f05245') for a in ARMS]):
        yy=y+i*28;draw.text((5,yy),label,font=font,fill='black');draw.line((left,yy+10,canvas.width-20,yy+10),fill='#dddddd',width=2)
        draw.line((left+(iv[0]-ids[0])*scale,yy+10,left+(iv[1]-ids[0])*scale,yy+10),fill=color,width=8)
    p=directory/(ds+'_'+case['kind']+'.png');p.parent.mkdir(parents=True,exist_ok=True);canvas.save(p)
    return dict(dataset=ds,query_ordinal=case['query_ordinal'],kind=case['kind'],path=str(p.relative_to(ROOT)),sha256=sha(p),
        original_pixel_sha256=pixel,physical_original_pixels_verified=True,official_GT_coordinates_drawn=True,private_not_published=True)

def run():
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import read_row
    activate();assert read(BASE/'P6_PREDICTION_BARRIER.json')['status']=='sealed'
    dest=NS/'actual_root_signal_views';cases=read(PUB/'P6/CASE_SELECTION.json')['records'];records=[]
    gt={ds:truths(ds,definition(ds)) for ds in ['hc2','vidstg']}
    for c in cases:
        ds,at,q=c['dataset'],c['arrival'],c['query_ordinal'];z=load(NS/'formal'/ds/f'{at:05}.pt');orig,inp,b=original(ds,q)
        records.append(rgb(c,z,orig,inp,read_row(ds,q),gt[ds][0][q],gt[ds][1][q],dest/'private_RGB_GT_support002'))
    write(dest/'PRIVATE_RGB_GT_SUPPORT_REVISION.json',dict(status='complete_display_frame_support_revision',records=records,cases=6,original_RGB_preserved=True,
        original_case_selection_sha256=sha(PUB/'P6/CASE_SELECTION.json'),all_scores_predictions_GT_geometry_unchanged=True,new_GPU_model_calls=0,GT_after_global_seal=True,time=time.time()))

if __name__=='__main__':run()
