"""CPU-only source roster for the full EATA-STVG port; no target GT loaded.

Official train files are used only for input/query metadata, never target labels.
The roster is fixed before source/target baseline predictions. HC2 source media
intake is a separate finite official-release step, not an old queue restart.
"""
import sys,time,zipfile,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_paper_common_v1 import *
OUT=BASE/'source_Fisher'
CAT=ROOT/'downloads/dataset_intake_20260910'
def run():
 if (OUT/'ROSTER_LOCK.json').exists():return
 vpath=CAT/'annotations/datasets/VidSTG/annotations/train_annotations.json'
 hpath=CAT/'hcstvg/datasets/HC-STVG2/HC-STVG/anno_v2/train_v2.json'
 parts=ROOT/'data/hcstvg2_official_metadata/video_parts.json'
 targets={ds:{r['source'] for r in read(BASE/ds/'PLAN.json')['rows']} for ds in DATASETS}
 manifest={};inputs={str(vpath.relative_to(ROOT)):sha(vpath),str(hpath.relative_to(ROOT)):sha(hpath),str(parts.relative_to(ROOT)):sha(parts)}
 archives={}
 for p in sorted((CAT/'datasets/VidOR').glob('training-video-part*.zip')):
  with zipfile.ZipFile(p) as z:
   for info in z.infolist():
    if not info.is_dir() and info.filename.endswith('.mp4'):
     key=Path(info.filename).stem;assert key not in archives
     archives[key]=dict(path=str(p),member=info.filename,bytes=info.file_size,crc32=info.CRC,archive_bytes=p.stat().st_size,archive_mtime_ns=p.stat().st_mtime_ns)
 rows=[]
 for i,a in enumerate(read(vpath)):
  source=str(a['vid']);assert source not in targets['vidstg']
  for kind in ['captions','questions']:
   for j,q in enumerate(a[kind]):
    rows.append(dict(key=f'vidtrain:{i}:{kind}:{j}',source=source,original_video_id=source,caption=q['description'],width=int(a['width']),height=int(a['height']),fps=float(a['fps']),frame_count=int(a['frame_count']),start_frame=int(a['used_segment']['begin_fid']),end_frame=min(int(a['frame_count']),int(a['used_segment']['end_fid'])+1),media=archives[source]))
 selected=sorted(rows,key=lambda r:(digest(['PaperEATAFisher2000-20261005','vidstg',r['key'],r['caption']]),r['key']))[:2000]
 manifest['vidstg']=dict(source_checkpoint='vidstg',source_split='official_train',rows=selected,source_count=len({r['source'] for r in selected}),target_same_domain_source_overlap=0,media_available_as_original_ZIP=True)
 partof={name:str(part) for part,files in read(parts).items() for name in files}
 validation_clips={r['input']['original_video_id'] for r in read(BASE/'hc2/PLAN.json')['rows']}
 rows=[];overlap=[]
 for name,a in sorted(read(hpath).items()):
  source=Path(name).stem.split('_',1)[1]
  assert Path(name).stem not in validation_clips,'Official source-train input cannot be a target-validation clip'
  if source in targets['hc2']:overlap.append(name)
  assert name in partof
  rows.append(dict(key='hc2train:'+name,source=source,original_video_id=Path(name).stem,caption=a['English'],width=int(a['img_size'][1]),height=int(a['img_size'][0]),annotation_frame_count=int(a['img_num']),filename=name,archive_part=partof[name]))
 selected=sorted(rows,key=lambda r:(digest(['PaperEATAFisher2000-20261005','hc2',r['key'],r['caption']]),r['key']))[:2000]
 manifest['hc2']=dict(source_checkpoint='hcstvg2',source_split='official_train',rows=selected,source_count=len({r['source'] for r in selected}),target_validation_clip_overlap=0,
  selected_train_parent_overlap_with_same_domain_validation=len({r['source'] for r in selected}&targets['hc2']),official_train_clips_sharing_validation_parent=len(overlap),
  split_caveat='Official clip-disjoint train/validation share parent movies. Source Fisher uses train only; not a parent-disjoint or fresh source claim.',media_available='requires_official_range_intake')
 for ds,p in manifest.items():
  assert len(p['rows'])==2000 and len({r['key'] for r in p['rows']})==2000
  p.update(target_GT_read=False,source_labels_used=False,selection='deterministic hash over all official source-training queries, no outcome filter',historical_exposure=True)
  f=OUT/ds/'SOURCE_ROSTER.json'
  if f.exists():assert read(f)==p,'Retain incompatible partial source roster for root review'
  else:write(f,p)
  inputs[str(f.relative_to(ROOT))]=sha(f)
 write(OUT/'ROSTER_LOCK.json',dict(status='source_input_rosters_sealed',queries_per_checkpoint=2000,inputs=inputs,pins={str(Path(__file__).relative_to(ROOT)):sha(__file__)},target_GT_read=False,source_labels_used=False,time=time.time()))
 status(OUT/'STATUS.json',dict(status='source_rosters_locked_pending_HC_media_and_serial_GPU_Fisher',source_queries=4000,Fisher_model_forwards=0,GT_read=False,time=time.time()))
 print('SOURCE_FISHER_ROSTERS_LOCKED',[(ds,len(p['rows']),p['source_count']) for ds,p in manifest.items()],flush=True)
if __name__=='__main__':run()
