"""One native YOLO/SAM2 case; no historical queue dispatch or annotation loader."""
import sys,os,time,argparse,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import BASE,read,write,sha,verify,guard,row

def inventory(prep,req):
 from scripts.stvg_r1_native_teacher_v1 import _load_mask_inventory
 files,ids=_load_mask_inventory(prep['mask_data_dir'])
 assert sha(prep['marker_video_path'])==prep['marker_video_sha256']
 assert all(0<=i<req['local_frame_count']for i in files)
 entries=[dict(index=i,path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)for i,p in sorted(files.items())]
 return dict(mask_count=len(files),known_ids=sorted(ids),files=entries,GT_read=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args();verify('tracking');sys.addaudithook(guard);r=row(a.case);req=read(ROOT/r['request_path']);assert sha(ROOT/r['request_path'])==r['request_sha256'];assert sha(req['video_path'])==req['canonical_video_sha256']
 dst=BASE/'TRACKING'/a.case;assert not(dst/'ACCEPTED.json').exists();old=ROOT/'artifacts/stvg_target_teacher_distillation_v1/tracking'/a.case/'ACCEPTED.json'
 if old.exists():
  z=read(old);assert z['status']=='pass'and z['GT_read']is False and z['target_mode']=='all'and z['input_request_sha256']==r['request_sha256'];assert z['checkpoint_code_compatibility']['source_sha256']==sha(ROOT/'scripts/stvg_yolov12_qkv_checkpoint_compat_v2.py')
  inv=inventory(z,req);write(dst/'MASK_INVENTORY.json',inv);write(dst/'ACCEPTED.json',dict(z,reused=True,old_receipt_path=str(old.relative_to(ROOT)),old_receipt_sha256=sha(old),mask_inventory_sha256=sha(dst/'MASK_INVENTORY.json')));print('ACCEPTED_REUSE',a.case,inv['mask_count'],flush=True);return
 sys.path.insert(0,str(ROOT/'external/yolov12'));sys.path.insert(0,str(ROOT/'external/Grounded-SAM-2-main'))
 import torch,ultralytics
 from scripts.stvg_sam_bounded_frame_cache010 import bounded_frame_loading
 from scripts.stvg_yolov12_qkv_checkpoint_compat_v2 import install_checkpoint_qkv_aattn_compat_v2
 from scripts.stvg_r1_native_teacher_v1 import make_official_preprocess_runner
 original=ultralytics.YOLO
 def yolo(*args,**kw):
  m=original(*args,**kw);assert install_checkpoint_qkv_aattn_compat_v2(m.model)['aattn_count']==16;return m
 ultralytics.YOLO=yolo;torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 try:
  run=make_official_preprocess_runner(ROOT/'external/stvg-r1',ROOT/'external/Grounded-SAM-2-main',detector_asset=ROOT/'checkpoints/STVG-R1-tracking/yolov12x.pt',sam2_asset=ROOT/'checkpoints/STVG-R1-tracking/sam2.1_hiera_large.pt')
  with bounded_frame_loading():z=run(req,target_mode='all',output_dir=dst/'official')
 finally:ultralytics.YOLO=original
 inv=inventory(z,req);write(dst/'MASK_INVENTORY.json',inv)
 z.update(status='pass',reused=False,input_request_sha256=r['request_sha256'],mask_inventory_sha256=sha(dst/'MASK_INVENTORY.json'),peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),actual_tracking_wall_s=time.perf_counter()-start,GT_read=False,checkpoint_code_compatibility=dict(source_sha256=sha(ROOT/'scripts/stvg_yolov12_qkv_checkpoint_compat_v2.py'),attention_backend='torch_SDPA_half_QKV',bitwise_equivalence_claimed=False,CPU_frame_cache_limit=2,frame_cache_source_sha256=sha(ROOT/'scripts/stvg_sam_bounded_frame_cache010.py')))
 write(dst/'ACCEPTED.json',z);print('ACCEPTED_NEW',a.case,inv['mask_count'],flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(BASE/'TRACKING_FAILURE_CAPTURE.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()),mutable=True);raise
