"""Fresh canonical decode, exact marked-video rebuild and pinned tracklet replay."""
import sys,json,hashlib,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image
from scripts.stvg_temporal_caption_query_v1 import read,write,sha,catalog,inference_guard
from scripts.stvg_adjacent_affine_tracklet_v1 import track_adjacent
from scripts.prepare_stvg_motion_order_layout001 import _render_order_layout
B=ROOT/'artifacts/stvg_adjacent_motion_opd_v1';I=ROOT/'artifacts/stvg_target_teacher_distillation_v1/canonical_inputs'
def exact(a,b,path='root'):
 if isinstance(a,dict):
  assert isinstance(b,dict) and set(a)==set(b),path
  for k in a:exact(a[k],b[k],path+'.'+k)
 elif isinstance(a,list):
  assert isinstance(b,list) and len(a)==len(b),path
  for n,(x,y) in enumerate(zip(a,b)):exact(x,y,path+f'[{n}]')
 else:assert a==b,(path,a,b)
def main():
 sys.addaudithook(inference_guard);cv2.setNumThreads(2);t0=time.perf_counter();bar=read(B/'CANDIDATE_INPUT_BARRIER.json');jobs=read(B/'TEACHER_JOB_LOCK.json')['records'];roster={r['case_id']:r for r in read(B/'ROSTER_LOCK.json')['records']};assert bar['status']=='complete' and bar['count']==8 and len(jobs)==48
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256'] and (ROOT/f['path']).stat().st_size==f['bytes']
 ngray=nrgb=ntrack=nrender=0;counts=[]
 for r in bar['records']:
  cid=r['case_id'];c=catalog(roster[cid])[0];req=read(I/cid/'REQUEST.json');assert sha(Path(req['video_path']))==req['canonical_video_sha256']==r['canonical_video_sha256']
  ids=r['tracking_physical_frame_ids'];assert ids==list(range(r['physical_frame_ids'][0],r['physical_frame_ids'][-1]+1))
  fm={int(x['physical_frame_id']):int(x['local_frame_index']) for x in req['frame_map']};targets={fm[f]:f for f in ids};imgs={};grays={};cap=cv2.VideoCapture(req['video_path']);i=0
  try:
   while targets:
    ok,bgr=cap.read();assert ok
    if i in targets:
     fid=targets.pop(i);gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);h,w=gray.shape;scale=min(1.,640/max(h,w));gray=cv2.resize(gray,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA) if scale<1 else gray
     assert hashlib.sha256(gray.tobytes()).hexdigest()==r['tracking_gray_sha256_by_frame'][str(fid)];grays[fid]=gray;ngray+=1
     if fid in r['physical_frame_ids']:
      rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert hashlib.sha256(rgb.tobytes()).hexdigest()==c['source_RGB'][fid]['RGB_sha256']==r['source_RGB_sha256_by_frame'][str(fid)];imgs[fid]=Image.fromarray(rgb);nrgb+=1
    i+=1
  finally:cap.release()
  anchor=ids.index(r['physical_frame_ids'][1]);graylist=[grays[f] for f in ids]
  for cand in r['candidate_geometry']:
   tr=cand['tracklet']
   if not tr.get('trajectory') or tr.get('status')=='not_run_invalid_anchor':continue
   anchorbox=tr['trajectory'][anchor]['box_xyxy_norm'];new=track_adjacent(graylist,ids,anchor,anchorbox)
   rho={ids[anchor]:1.};rho.update({t['to_physical_frame_id']:t['rho'] for t in new['transitions'] if t['status']=='accepted'})
   for q in new['trajectory']:q['rho']=rho.get(q['physical_frame_id'],0.)
   exact(new,tr,cid+'.'+str(cand['candidate_id']));ntrack+=1
   assert cand['geometry_boxes_xyxy']==[new['trajectory'][ids.index(f)]['box_xyxy_norm'] for f in r['physical_frame_ids']]
   assert cand['boxes_xyxy']==cand['geometry_boxes_xyxy'] or cand['is_Source']
  base=read(B/'private_inputs'/cid/'base_render.json')
  for job in [j for j in jobs if j['case_id']==cid]:
   kind=job['kind'];meta=read(B/'private_inputs'/cid/(kind+'_render.json'))
   arr,rows=_render_order_layout([imgs[f] for f in r['physical_frame_ids']],r['actual_times_seconds'],base,meta,cid,kind)
   v=job['videos'][0];saved=np.load(ROOT/v['path'],allow_pickle=False);assert np.array_equal(arr,saved) and hashlib.sha256(arr.tobytes()).hexdigest()==v['tensor_sha256'];assert [m['rendered_RGB_sha256'] for m in rows]==[m['rendered_RGB_sha256'] for m in meta['records']];nrender+=1
   assert hashlib.sha256(job['prompt'].encode()).hexdigest()==job['prompt_sha256']
  counts.append(dict(case_id=cid,count=len(r['candidate_geometry']),geometry_support_counts=[sum(b is not None for b in ca['geometry_boxes_xyxy']) for ca in r['candidate_geometry']]))
  del c,imgs,grays,graylist,arr,saved;gc.collect();print(cid,'exact input readback',flush=True)
 write(B/'ROOT_INPUT_READBACK.json',dict(status='pass',scope='fresh source RGB/gray decode, marked array reconstruction and same-pinned-tracker replay',count=8,candidate_count=ntrack,RGB_count=nrgb,gray_count=ngray,render_count=nrender,records=counts,GT_read=False,model_loaded=False,GPU_initialized=False,independent_tracking_algorithm=False,barrier_sha256=sha(B/'CANDIDATE_INPUT_BARRIER.json'),reader_sha256=sha(Path(__file__)),wall_seconds=time.perf_counter()-t0))
if __name__=='__main__':main()
