"""Matched source-position control without GT generation or teacher prompt changes."""
import sys,hashlib,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from scripts.stvg_temporal_caption_query_v1 import ROOT,read,write,sha,inference_guard
from scripts.stvg_box_scale_grid_v1 import masked_window_input,marked_window_input,_pixel_hash
B=ROOT/'artifacts/stvg_box_scale_global_evidence_v1';BASE=B/'source_position_control001'
def main():
 sys.addaudithook(inference_guard);BASE.mkdir(exist_ok=True);old=read(B/'TEACHER_JOB_LOCK.json')['records'];grid=read(B/'GRID_INPUT_BARRIER.json');jobs=[];records=[]
 for r in grid['records']:
  cid=r['case_id'];initial=next(j for j in old if j['case_id']==cid and j['kind']=='original');marked=next(j for j in old if j['case_id']==cid and j['kind']=='marked');box=r['grid']['student_baseline']['cxcywh'];cx,cy,w,h=box;window=dict(window_id=0,x0=cx-w/2,y0=cy-h/2,w=w,h=h,cx=cx,cy=cy);image=Image.open(B/'private_inputs'/cid/'source.png').convert('RGB');assert _pixel_hash(image)==r['source_pixel_sha256'];directory=BASE/'private_inputs'/cid;directory.mkdir(parents=True);jobs.append(initial);records.append(dict(case_id=cid,dataset=r['dataset'],source_case_sha256=r['source_case_sha256'],student_source_window=window,original_grid_windows=r['grid']['window_count'],original_anchor_not_changed=True))
  for kind,func,template in [('source_masked',masked_window_input,initial),('source_marked',marked_window_input,marked)]:
   variant=func(image,window);p=directory/f'{kind}.png';variant['image'].save(p);job={**template,'kind':kind,'window_id':0,'input_path':str(p.relative_to(ROOT)),'input_file_sha256':sha(p),'input_pixel_sha256':_pixel_hash(variant['image']),'metadata':variant['metadata']};jobs.append(job)
 write(BASE/'GRID_INPUT_BARRIER.json',dict(status='complete',count=8,records=records,files=[dict(path=j['input_path'],sha256=j['input_file_sha256'])for j in jobs],GT_read=False))
 write(BASE/'TEACHER_JOB_LOCK.json',dict(status='locked',count=24,records=jobs,GT_read=False,source_only_geometric_control=True,original_query_prompts_and_anchor_exactly_reused=True,GT_exposure_historically_known=True,control_added_after_first_P0_readback_disclosed=True))
 print('8 original repeat +8 source blur +8 source mark =24 fixed matched-control calls')
if __name__=='__main__':main()
