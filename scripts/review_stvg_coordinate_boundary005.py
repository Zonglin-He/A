"""Verify preserved CPU candidate prefix and the Source-only visible-ROI adapter."""
import sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
from scripts.run_stvg_coordinate_confirmation_v1 import BASE,roster,load_case
D=BASE/'boundary_adapter005'
def main():
 import numpy as np
 sys.addaudithook(guard)
 for r in read(D/'CAPTURE.json')['files']:assert sha(D/r['path'])==r['sha256']
 n=0
 for p in sorted((D/'proposals').glob('*.json')):
  a=read(p);b=read(BASE/'proposals'/p.name)
  for x in [a,b]:
   for row in x['windows']:row.pop('proposal_seconds')
  assert a==b,p.name;n+=1
 arrays=0
 for p in sorted((D/'teacher_inputs').rglob('*')):
  if not p.is_file():continue
  q=BASE/p.relative_to(D)
  if p.suffix=='.npz':
   with np.load(p)as a,np.load(q)as b:
    assert a.files==b.files and all(np.array_equal(a[k],b[k])for k in a.files),str(p)
  else:assert sha(p)==sha(q),str(p)
  arrays+=1
 rs={r['probe_alias']:r for r in roster()};rows=read(D/'BOUNDARY_READBACK.json')['records']
 for row in rows:
  case,_=load_case(rs[row['case_id']]);position=list(case['frame_ids']).index(row['physical_frame_id']);raw=np.asarray(case['native_boxes_xyxy_normalized'])[position];roi=np.clip(raw,0,1)
  assert raw.tolist()==row['raw_anchor']and roi.tolist()==row['tracker_roi']and not np.array_equal(raw,roi)
  pr=read(BASE/'proposals'/(row['case_id']+'.json'));w=next(w for w in pr['windows']if w['slot']==row['slot']);assert w['tracks'][0]['center_box']==raw.tolist()
 assert read(BASE/'ROOT_INPUT_RENDER_READBACK.json')['status']=='pass'
 write(D/'ROOT_REVISION_READBACK.json',dict(status='pass',preserved_complete_cases=n,preserved_input_files=arrays,boundary_window_count=len(rows),GT_read=False,CPU_only=True,source_predictions_unchanged=True,scope='original prefix exact proposal geometry and rendered bytes/arrays; clipped Source ROI only, original source/scoring coordinates preserved'))
 print('boundary/prefix readback',n,arrays,len(rows))
if __name__=='__main__':main()
