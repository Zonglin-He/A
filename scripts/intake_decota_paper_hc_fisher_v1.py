"""Finite official HC2 source-train intake for EATA, zero GPU/target labels.

Only selected remote ZIP members are fetched. No full archive downloads,
credentials/tokens in receipts, sample replacement, or training here.
"""
import sys,time,collections,shutil,traceback,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_paper_common_v1 import *
OUT=BASE/'source_Fisher';DATA=ROOT/'data/decota_paper_source_fisher_v1/hc2/video'
def run():
 import zlib
 from remotezip import RemoteZip
 from scripts import download_hcstvg2_subset as official
 from scripts.prepare_stvg_fullscale_v1 import _probe
 from scripts.prepare_official_dense_pool_v2 import frame_ids_from_segment
 OUT.mkdir(parents=True,exist_ok=True);f=(OUT/'CPU_INTAKE.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 lock=read(OUT/'ROSTER_LOCK.json')
 for rel,h in lock['inputs'].items():assert sha(ROOT/rel)==h
 plan=read(OUT/'hc2/SOURCE_ROSTER.json');assert len(plan['rows'])==2000 and plan['target_validation_clip_overlap']==0
 if (OUT/'hc2/MEDIA_BARRIER.json').exists():return
 DATA.mkdir(parents=True,exist_ok=True)
 # The helper uses the password publicly documented in the official release;
 # access tokens are kept in this process only and never written/printed.
 session,drive,token=official.authenticate(official.DEFAULT_PASSWORD)
 items={x['name']:x for x in official.list_children(session,drive,token,'HC-STVG/VIdeo')}
 groups=collections.defaultdict(list)
 for r in plan['rows']:groups[r['archive_part']].append(r)
 outputs={};totalbytes=0;tick=time.time()
 for part,rows in sorted(groups.items(),key=lambda x:int(x[0])):
  item=items[part+'.zip'];url=item.get('@content.downloadUrl');assert url
  with RemoteZip(url,session=session) as archive:
   infos={Path(i.filename).name:i for i in archive.infolist() if not i.is_dir()}
   for r in rows:
    name=r['filename'];info=infos[name];dest=DATA/name;receipt=OUT/'hc2/media'/f'{name}.json'
    if receipt.exists():
     rc=read(receipt);assert rc['CRC']==info.CRC and sha(dest)==rc['sha256'];outputs[name]=sha(receipt);totalbytes+=dest.stat().st_size;continue
    assert totalbytes+info.file_size<=24*2**30,'Source intake finite 24GiB cap; preserve input roster, do not resample'
    assert shutil.disk_usage(ROOT).free-info.file_size>16*2**30,'Source intake reserves 16GiB for active inference'
    tmp=dest.with_suffix(dest.suffix+'.part');crc=0;size=0
    # A failed partial stays under recovery until root chooses exact-input retry.
    assert not tmp.exists() and not dest.exists(),'Preserve unreceipted partial for root engineering recovery'
    with archive.open(info) as stream,tmp.open('xb') as out:
     for block in iter(lambda:stream.read(4<<20),b''):out.write(block);size+=len(block);crc=zlib.crc32(block,crc)
    assert size==info.file_size and (crc&0xffffffff)==info.CRC;tmp.replace(dest)
    m=_probe(dest);assert (m['height'],m['width'])==(r['height'],r['width'])
    assert abs(m['frame_count']-r['annotation_frame_count'])<=2,'Source media frame-count discrepancy; do not change roster'
    ids=frame_ids_from_segment(frame_count=m['frame_count'],fps=m['fps'],start_frame=0,end_frame=m['frame_count'],max_frames=200)
    q=dict(caption=r['caption'],index=len(outputs),source=r['source'],original_video_id=r['original_video_id'],video_path=str(dest),video_sha256=sha(dest),kind='hcstvg',frame_ids=ids,start_frame=0,end_frame=m['frame_count'],**{k:m[k] for k in ['width','height','fps','frame_count','duration']})
    write(receipt,dict(status='official_source_input',input=q,key=r['key'],official_part=part+'.zip',member=info.filename,CRC=info.CRC,bytes=size,sha256=q['video_sha256'],target_GT_read=False,source_boxes_or_intervals_used=False,probe='ffprobe header; exact sampled frames validated by GPU-stage original decoder',time=time.time()))
    outputs[name]=sha(receipt);totalbytes+=size
    status(OUT/'CPU_INTAKE_STATUS.json',dict(status='running',pid=__import__('os').getpid(),done=len(outputs),total=2000,bytes=totalbytes,seconds=time.time()-tick,GPU_seconds=0,GT_read=False,time=time.time()))
    if len(outputs)%20==0:print('SOURCE_INTAKE',len(outputs),2000,totalbytes,flush=True)
 assert len(outputs)==2000
 write(OUT/'hc2/MEDIA_BARRIER.json',dict(status='sealed',source_roster_sha256=sha(OUT/'hc2/SOURCE_ROSTER.json'),files=outputs,queries=2000,bytes=totalbytes,GT_read=False,GPU_seconds=0,time=time.time()))
 status(OUT/'CPU_INTAKE_STATUS.json',dict(status='completed_pending_serial_GPU_Fisher',done=2000,bytes=totalbytes,GT_read=False,time=time.time()))
 print('SOURCE_MEDIA_SEALED',2000,totalbytes,flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  fd=OUT/'intake_failures'/str(time.time_ns());fd.mkdir(parents=True,exist_ok=True)
  # Exception type only: requests exceptions can contain expiring access URLs.
  write(fd/'FAILURE.json',dict(status='failed',type=type(e).__name__,GT_read=False,GPU_seconds=0,time=time.time()))
  status(OUT/'CPU_INTAKE_STATUS.json',dict(status='failed',error_type=type(e).__name__,failure=str(fd),GT_read=False,time=time.time()))
  print('SOURCE_INTAKE_FAILURE',type(e).__name__,str(fd),flush=True);sys.exit(1)
