"""Re-decode cached canonical pixels and verify complete candidate rendering; no GT."""
import sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
from scripts.run_stvg_coordinate_confirmation_v1 import BASE,roster,load_case,selected_rgb
def main():
 import numpy as np,torch
 from PIL import Image
 from scripts import stvg_weightfree_spatial_v1 as w
 from scripts import prepare_stvg_short_window_teacher_v1 as render
 sys.addaudithook(guard);bar=read(BASE/'CANDIDATE_INPUT_BARRIER.json');jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records'];lookup={j['job_id']:j for j in jobs};windows=masks_count=0
 for rec in roster():
  cid=rec['probe_alias'];case,p=load_case(rec);request=read(BASE/'requests'/(cid+'.json'));assert sha(Path(request['video_path']))==request['canonical_video_sha256'] and request['source_video_sha256']==case['input']['video_sha256'];pr=read(BASE/'proposals'/(cid+'.json'));wins=pr['windows'];temporal=torch.load(BASE/'temporal'/(cid+'.pt'),map_location='cpu',weights_only=False)['temporal'];interval=temporal['final_interval'];fm={int(x['physical_frame_id'])for x in request['frame_map']};eligible=[int(f)for f in case['frame_ids']if interval[0]<=int(f)<interval[1] and all(int(f)+k in fm for k in range(-2,3))];expected=[]
  if len(eligible)>=4:
   for i in range(4):expected.append(min((f for f in eligible if f not in expected),key=lambda f:(abs(f-(interval[0]+(i+.5)*(interval[1]-interval[0])/4)),f)))
  assert sorted(expected)==[x['center_physical_frame_id']for x in wins], 'original QA clock mismatch'
  if not wins:continue
  centers=[x['center_physical_frame_id']for x in wins];rgb=selected_rgb(request,{f+k for f in centers for k in range(-2,3)});context,_=render.render_context_frames([Image.fromarray(rgb[f])for f in centers]);context=context.transpose(0,3,1,2).copy();assert np.array_equal(context,np.load(BASE/'teacher_inputs'/cid/'context.npy'))
  for row in wins:
   slot=row['slot'];job=lookup[f'{cid}_diverse8_{slot}'];assert job['candidate_bindings']==row['tracks'];assert row['physical_frame_ids']==list(range(row['center_physical_frame_id']-2,row['center_physical_frame_id']+3));assert list(case['frame_ids'])[row['source_grid_position']]==row['center_physical_frame_id'];mask=np.load(BASE/'teacher_inputs'/cid/f'masks_{slot}.npz')['masks']
   assert len(mask)==len(row['tracks'])-1
   for tr,mm in zip(row['tracks'][1:],mask):
    assert tr['presence']==sum(np.count_nonzero(m)>=32 for m in mm)/5
    for box,m in zip(tr['boxes'],mm):
     yy,xx=np.where(m)
     if len(xx)<32:assert box is None;continue
     h,ww=m.shape;expected=[float(xx.min()/ww),float(yy.min()/h),float((xx.max()+1)/ww),float((yy.max()+1)/h)];assert np.allclose(box,expected,atol=1e-7,rtol=0)
    masks_count+=1
   vid=w.render([rgb[f]for f in row['physical_frame_ids']],row['tracks'],list(mask),job['visible_id_to_candidate']);assert np.array_equal(vid,np.load(ROOT/job['videos'][1]['path']))
   for v,a in zip(job['videos'],[context,vid]):assert sha(ROOT/v['path'])==v['sha256'] and hashlib.sha256(a.tobytes()).hexdigest()==v['tensor_sha256']
   windows+=1
 assert windows==bar['window_count']==len(jobs)
 write(BASE/'ROOT_INPUT_RENDER_READBACK.json',dict(status='pass',case_count=32,window_count=windows,mask_track_count=masks_count,GT_read=False,CPU_only=True,original_QA_clock_checked=True,scope='all canonical video hashes, source frame identities, fresh pixel decode, mask bbox geometry and exact stored R1 rendering; same renderer replay, no independent proposal algorithm'))
 print('input render root',windows,masks_count)
if __name__=='__main__':main()
