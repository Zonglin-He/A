"""Offline descriptive old negative gallery, never used to choose runtime candidates."""
import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round27'))
from common27 import *
from observations27 import load
from joint27 import local_view,marked_image
from PIL import Image,ImageDraw
def main():
 rr=read(EXP/'RESOLUTION_ROWS_PRIVATE.json');lookup={r['alias']:r for r in rows()};chosen=[]
 for p in ['drone','quad']:
  a=[(v['scene_short_px'],r,v) for r in rr if r['platform']==p and r['R26_delta_vs_legacy_pp']< -1e-8 for v in r['instance_sizes'] if v['legacy_winner_mask']];a.sort(key=lambda z:(z[0],z[1]['alias']));chosen.extend([a[0],a[len(a)//2]])
 sheet=Image.new('RGB',(1500,4*560),(245,245,245));draw=ImageDraw.Draw(sheet)
 for i,(short,x,v) in enumerate(chosen):
  r=lookup[x['alias']];frame,scene,binding=load(r);p=read(paths(r)['prep']/'cases_PRIVATE'/(r['id']+'.json'));m=p['mask_pool_PRIVATE'][v['pool_slot']];obj=scene['objects'][m];sceneim=marked_image(frame['rgb'],obj).resize(tuple(v['scene_resized_wh']),Image.Resampling.BICUBIC);crop,_=local_view(frame['rgb'],obj);crop=crop.resize(tuple(v['crop_resized_wh']),Image.Resampling.BICUBIC);sceneim.thumbnail((1080,510));crop.thumbnail((380,510));y=i*560;draw.text((10,y+10),f"{x['alias']} | full-frame subject short side {short:.1f}px | crop {v['crop_short_px']:.1f}px | old R26 vs legacyG {x['R26_delta_vs_legacy_pp']:+.2f}pp",fill=(0,0,0));sheet.paste(sceneim,(10,y+40));sheet.paste(crop,(1110,y+40))
 out=EXP.parent/'private_views';out.mkdir(exist_ok=True);sheet.save(out/'RESOLUTION_DIAGNOSTIC.jpg');dump(out/'SELECTION.json',{'policy':'perplatform smallest and median legacy-selected mask among oldR26harm cases;posthoc diagnosis,no runtime use','aliases':[r['alias'] for _,r,_ in chosen],'raw_media_not_public':True});print(out/'RESOLUTION_DIAGNOSTIC.jpg')
if __name__=='__main__':main()
