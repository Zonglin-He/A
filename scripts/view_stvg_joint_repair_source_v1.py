"""Private pixel view of the prelocked first source-val case in each domain."""
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_posterior_repair_v1 import BASE,OLD,read,put,load,decode_state

def main():
    from PIL import Image,ImageDraw
    import numpy as np
    from scripts.run_decota_paper_main_v1 import frames_for
    files=[]
    for ds in ['hc2','vidstg']:
        r=next(r for r in read(BASE/'PLAN_LOCK.json')['source']if r['dataset']==ds and r['split']=='source_validation')
        z=load(OLD/'features/source'/(r['alias']+'.pt'));pred=load(BASE/'selected_predictions'/(r['alias']+'.pt'))
        frames,ids,_=frames_for(ds,dict(input=r['input'],frame_ids=r['frame_ids']),collections.OrderedDict());assert ids==z['frame_ids']
        positions=np.linspace(0,len(ids)-1,4).round().astype(int).tolist();arms=[dict(boxes=z['boxes'][0].numpy(),interval=z['native_interval'])]+[decode_state(x,z)for x in pred['samples']]
        canvas=Image.new('RGB',(1120,5*192),(22,22,22));draw=ImageDraw.Draw(canvas)
        for j,a in enumerate(arms):
            for k,pos in enumerate(positions):
                im=Image.fromarray(frames[pos]);im.thumbnail((280,155));ox,oy=k*280,j*192+34;canvas.paste(im,(ox,oy))
                cx,cy,w,h=a['boxes'][pos];box=[ox+(cx-w/2)*im.width,oy+(cy-h/2)*im.height,ox+(cx+w/2)*im.width,oy+(cy+h/2)*im.height]
                draw.rectangle(box,outline=(255,210,40),width=2)
                label='Source'if j==0 else f'sample{j-1}'
                active=a['interval'][0]<=ids[pos]<a['interval'][1]
                draw.text((ox+4,j*192+4),f"{r['alias']} {label} slot{k+1} event={active}",fill='white')
        f=BASE/'root_view'/(ds+'.png');f.parent.mkdir(exist_ok=True);canvas.save(f);files.append(str(f))
    put(BASE/'ROOT_VIEW_PLAN.json',dict(status='rendered_pending_actual_view',files=files,source_only=True,GT_overlays=False,identity_semantic_annotation=False))
    print('\n'.join(files))

if __name__=='__main__':main()
