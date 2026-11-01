"""GT-free exact pinned resize audit, then descriptive old-result stratification."""
import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round27'))
from common27 import *
from observations27 import load
from joint27 import local_view
from association import DEFAULT_CONFIG
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
def dims(H,W,cap):
 h,w=smart_resize(H,W,factor=28,min_pixels=DEFAULT_CONFIG.min_pixels,max_pixels=cap*784);return w,h
def main():
 old={x['alias']:x for x in read(ROOT/'research/round26/experiment/ALL464_ROWS_PRIVATE.json')};rec=[];summary={}
 for i,r in enumerate(rows('OLD400')):
  prep=read(paths(r)['prep']/'cases_PRIVATE'/(r['id']+'.json'));a=load_npz(paths(r)['prep']/'arrays_PRIVATE'/(r['id']+'.npz'));pool=prep['mask_pool_PRIVATE'];x=old[r['alias']];loss=x['ious']['qwen_full_scene_raw']-x['ious']['legacy_complex'];rep={'alias':r['alias'],'platform':r['platform'],'R26_delta_vs_legacy_pp':100*loss,'pool_n':len(pool),'instance_sizes':[]}
  if pool:
   frame,scene,binding=load(r);H,W=frame['rgb'].shape[:2];rw,rh=dims(H,W,1024);hw,hh=dims(H,W,2048);q=x['q']['legacy_complex'];chosen=int(np.argmax(a['weights'][q])) if a['weights'][q].sum()>0 else None
   for j,m in enumerate(pool):
    obj=scene['objects'][m];crop,_=local_view(frame['rgb'],obj);cw,ch=dims(crop.height,crop.width,1024);b=np.asarray(obj['detector_box_xyxy']);wh=b[2:]-b[:2];base=wh*np.array([rw/W,rh/H]);high=wh*np.array([hw/W,hh/H]);detail=wh*np.array([cw/crop.width,ch/crop.height]);rep['instance_sizes'].append({'pool_slot':j,'legacy_winner_mask':j==chosen,'native_short_px':float(wh.min()),'scene_short_px':float(base.min()),'hires_short_px':float(high.min()),'crop_short_px':float(detail.min()),'scene_native_scale':float(min(rw/W,rh/H)),'scene_resized_wh':[rw,rh],'hires_resized_wh':[hw,hh],'crop_resized_wh':[cw,ch]})
  rec.append(rep)
 for p in ['drone','quad']:
  for group in ['all_pooled_masks','legacy_winner_R26_harm','legacy_winner_R26_gain','legacy_winner_R26_tie']:
   vals=[]
   for r in rec:
    if r['platform']!=p:continue
    for v in r['instance_sizes']:
     eligible=group=='all_pooled_masks' or v['legacy_winner_mask'] and ((group.endswith('harm') and r['R26_delta_vs_legacy_pp']< -1e-8) or (group.endswith('gain') and r['R26_delta_vs_legacy_pp']>1e-8) or (group.endswith('tie') and abs(r['R26_delta_vs_legacy_pp'])<=1e-8))
     if eligible:vals.append(v)
   summary[p+'_'+group]={'n_masks_not_independent_samples':len(vals)}
   for name in ['native_short_px','scene_short_px','hires_short_px','crop_short_px','scene_native_scale']:
    a=np.array([v[name] for v in vals]);summary[p+'_'+group][name]={'median':float(np.median(a)),'q10':float(np.quantile(a,.1)),'q90':float(np.quantile(a,.9))} if len(a) else None
   summary[p+'_'+group]['scene_short_lt14']=sum(v['scene_short_px']<14 for v in vals);summary[p+'_'+group]['scene_short_lt28']=sum(v['scene_short_px']<28 for v in vals)
 dump(EXP/'RESOLUTION_ROWS_PRIVATE.json',rec);dump(EXP/'RESOLUTION_SUMMARY.json',{'scope':'OLD400 fixed input audit; oldR26outcome groups descriptive only,no newlabels/gates','processor':'pinned official Qwen2VL smart_resize patch14 merge2','by_platform_and_group':summary});print('RESOLUTION27',summary)
if __name__=='__main__':main()
