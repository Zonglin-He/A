"""Private full-frame reference/mapping review, no policy changes from review."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round21'))
from common21 import *
from observations import load
from PIL import Image,ImageDraw,ImageFont
def main():
    rows=read(ROWS);aliases=['V_quad_004','W_quad_011','Z_quad_026']+[x['alias'] for x in read(EXP/'RELATION_INTERVENTIONS.json')];aliases=list(dict.fromkeys(aliases));out=EXP/'review_PRIVATE';out.mkdir(exist_ok=False);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for alias in aliases:
        r=next(x for x in rows if x['alias']==alias);frame,scene,binding=load(r);im=Image.fromarray(frame['rgb']);scale=min(1.,1100/im.width);im=im.resize((int(im.width*scale),int(im.height*scale)));draw=ImageDraw.Draw(im);g=read(BASE/'text208_attempt002/cases_PRIVATE'/(r['id']+'.json'));ref=read(BASE/'reference208_attempt002/cases_PRIVATE'/(r['id']+'.json'));prep=old(r,'pairs');lines=[alias]
        for pi,pair in enumerate(prep['pairs']):
            for label,m,color in [('A',pair['mask_indices_PRIVATE'][0],'cyan'),('B'+str(pi),pair['mask_indices_PRIVATE'][1],'orange')]:
                bb=[int(x*scale) for x in scene['objects'][m]['detector_box_xyxy']];draw.rectangle(bb,outline=color,width=3);draw.text((bb[0],max(0,bb[1]-22)),label,fill=color,font=font)
        for rr in g['references']:
            for k,obs in enumerate(ref['references'].get(rr['entity'],[])):
                color='lime' if obs['semantic_accepted'] else 'red';bb=[int(x*scale) for x in obs['box_PRIVATE']];draw.rectangle(bb,outline=color,width=2);label=rr['entity']+'#'+str(k);draw.text((bb[0],max(0,bb[1])),label,fill=color,font=font);lines.append(f"{label} {rr['phrase']} | full={obs['full']['log_odds_yes_no']:.2f} visual={obs['visual_evidence']:.2f} modifier={obs['modifier_increment']:.2f} accept={obs['semantic_accepted']}")
        canvas=Image.new('RGB',(1100,im.height+30*(len(lines)+1)),(245,245,245));canvas.paste(im,(0,0));dd=ImageDraw.Draw(canvas)
        for j,line in enumerate(lines):dd.text((6,im.height+30*j),line,fill='black',font=font)
        canvas.save(out/(alias+'.png'))
    dump(EXP/'REVIEW_MANIFEST.json',{'aliases':aliases,'selection':'Three prechosen failures plus ALL relation-vs-no-reference interventions; no benefit filter.','private_images':True});print('PANELS',len(aliases))
if __name__=='__main__':main()
