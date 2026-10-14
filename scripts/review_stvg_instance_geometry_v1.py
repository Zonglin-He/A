"""Independent RLE-to-bbox readback of every geometry-only source-grid output."""
import sys,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'external/stvg_runtime')]
from scripts.stvg_instance_identity_common_v1 import BASE,OLD,read,write,sha,guard

def main():
 import numpy as np
 from pycocotools import mask as mu
 sys.addaudithook(guard)
 roster={r['case_id']:r for r in read(OLD/'ROSTER_LOCK.json')['records']}
 refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};count=present=0
 for r in read(BASE/'TEACHER_GEOMETRY_ONLY_READOUT.json')['records']:
  cid=r['case_id'];rec=roster[cid];assert sha(ROOT/rec['request_path'])==rec['request_sha256']
  req=read(ROOT/rec['request_path']);clock={x['physical_frame_id']:x['local_frame_index']for x in req['frame_map']}
  directory=Path(read(OLD/'TRACKING'/cid/'ACCEPTED.json')['mask_data_dir']);files={int(re.search(r'mask_(\d+).json',p.name)[1]):p for p in directory.glob('mask_*.json')}
  for arm in ['raw_decoupled','strict_original']:
   a=r[arm];ident=a['candidate_id'];assert a['interval_frame_bounds_half_open']==rec['qa_interval']
   for i,fid in enumerate(a['frame_ids_physical']):
    d=read(files[clock[fid]]);obj=d['objects'].get(str(ident))if ident is not None else None
    box=None
    if obj is not None:
     encoded=dict(obj['rle']);encoded['counts']=encoded['counts'].encode()if isinstance(encoded['counts'],str)else encoded['counts']
     if int(mu.area(encoded)):
      x,y,w,h=map(float,mu.toBbox(encoded));H,W=encoded['size'];box=[x/W,y/H,(x+w)/W,(y+h)/H]
    assert (box is not None)==a['teacher_mask_present'][i]
    if box is not None:assert np.array_equal(box,a['boxes_xyxy_normalized'][i]);present+=1
    count+=1
 sources=[Path(mu.__file__),ROOT/'external/stvg_runtime/pycocotools/_mask.cpython-311-x86_64-linux-gnu.so']
 sources=[p for p in sources if p.exists()]
 write(BASE/'ROOT_INDEPENDENT_GEOMETRY_READBACK.json',dict(status='pass',count=count,mask_boxes=present,GT_read=False,GPU_used=False,independent_bbox='pycocotools.toBbox run-length bbox versus pixel argwhere writer',decoder_bindings={str(p.relative_to(ROOT)):sha(p)for p in sources},reference_sha256=sha(BASE/'REFERENCE_LOCK.json'),geometry_sha256=sha(BASE/'TEACHER_GEOMETRY_ONLY_READOUT.json')))
 print('pass',count,present)
if __name__=='__main__':main()
