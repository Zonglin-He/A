"""Private display-only real-pixel QA; no inference, labels or case selection by scores."""
from common import *

def main():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from operators import apply, burst_spec, FAMILIES
    from stvg.decode import decode
    from stvg.runtime import guard
    validate();sys.addaudithook(guard)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    canvas=Image.new('RGB',(6*240,2*210),'white');draw=ImageDraw.Draw(canvas);checks=[]
    for j,ds in enumerate(COUNTS):
        row=records(ds)[0];m=row['metadata'];spec=burst_spec(str(m['source']),m['frame_count'],m['frame_ids'],5.)
        fid=m['frame_ids'][spec['positions'][0]]if spec['positions']else spec['physical_start']
        frames,ids=decode({**m,'frame_ids':[fid]},ds)
        images=[('Original',frames[0])]
        for family in FAMILIES:
            c=dict(id=family+'_p5',kind='burst',family=family,coverage_percent=5.)
            a,meta=apply(frames,ids,m,ds,c);images.append((family,a[0]));checks.append(dict(dataset=ds,family=family,physical_frame=fid,output_SHA=hashlib.sha256(a.tobytes()).hexdigest()))
        for i,(name,rgb)in enumerate(images):
            im=Image.fromarray(rgb);im.thumbnail((232,165));canvas.paste(im,(i*240+4,j*210+39));draw.text((i*240+4,j*210+3),ds+' / '+name,fill='black',font=font)
            draw.text((i*240+4,j*210+21),'5% physical-burst QA',fill='black',font=font)
    out=D/'pixel_QA.png';canvas.save(out)
    atomic(D/'pixel_QA_receipt.json',dict(status='rendered_pending_root_pixel_view',PNG_sha256=sha(out),
        code_sha256=sha(Path(__file__)),checks=checks,new_Source=0,new_expert=0,new_GT_scoring=0,
        scope='One actual physical burst frame from first manifest query/domain, selected without labels or scores. Display-only frame grid; does not force an event/sample hit in the evaluation. Media remains private.'))
    print(out,flush=True)

if __name__=='__main__':main()
