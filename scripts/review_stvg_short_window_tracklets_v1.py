"""Independent custody/frame/anchor audit and exact fixed-helper CPU replay."""
import sys,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np, cv2
from scripts.prepare_stvg_short_event_tracklets_v1 import track_short_window_lk
B=R/'artifacts/stvg_short_event_tracklets_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hb=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
def main():
 cv2.setNumThreads(1);bar=json.loads((B/'GLOBAL_TRACKLET_BARRIER.json').read_text());m=json.loads((B/'PREPARED_MANIFEST.json').read_text());assert bar['status']=='complete'and bar['count']==8 and len(bar['files'])==16 and bar['manifest_sha256']==sha(B/'PREPARED_MANIFEST.json')
 for z in bar['files']:
  p=R/z['path'];assert sha(p)==z['sha256']and p.stat().st_size==z['bytes']
 total=frames=valid=0
 for c in m['cases']:
  cid=c['case_id'];d=json.loads((B/'tracklets'/f'{cid}.json').read_text());a=np.load(B/'private_rgb'/f'{cid}.npz',allow_pickle=False)
  assert d['input_binding']['source_case_sha256']==c['source_case_sha256'];assert d['input_binding']['manifest_sha256']==sha(B/'PREPARED_MANIFEST.json')
  locals_=list(map(int,a['local_frame_indices']));phys=list(map(int,a['physical_frame_ids']));assert len(locals_)==len(set(locals_))==20
  for v in d['frame_readback']['selected_frames']:
   k=locals_.index(v['local_frame_index']);assert phys[k]==v['physical_frame_id'];assert hb(a['rgb_uint8'][k])==v['cached_rgb_sha256'];assert hb(a['gray_uint8'][k])==v['tracking_gray_sha256'];frames+=1
  lookup={t['tracklet_id']:t for t in d['tracklets']};assert len(lookup)==sum(len(x['matched_tracklets'])for x in c['centers'])
  for w in c['centers']:
   for p in w['matched_tracklets']:
    t=lookup[p['tracklet_id']];assert all(t[k]==v for k,v in p.items());total+=1;valid+=sum(x['valid']for x in t['frames'])
    if p['anchor_box_xyxy_norm']is None:assert t['result']is None;continue
    x=track_short_window_lk(gray_frames=[a['gray_uint8'][locals_.index(l)]for l in w['local_frame_indices']],physical_frame_ids=w['physical_frame_ids'],center_physical_frame_id=w['center_physical_frame_id'],anchor_box_xyxy_norm=p['anchor_box_xyxy_norm'],coverage_half_open=c['canonical_physical_coverage_half_open'])
    assert x['tracklet']==t['result'],p['tracklet_id']
 assert(total,frames,valid)==(152,160,427)
 out=dict(status='pass',count=8,tracklets=total,unique_frames=frames,valid_boxes=valid,barrier_sha256=sha(B/'GLOBAL_TRACKLET_BARRIER.json'),manifest_sha256=sha(B/'PREPARED_MANIFEST.json'),same_helper_exact_replay=True,independent_second_tracking_algorithm=False,GT_read=False,model_loaded=False,GPU_used=False,time=time.time());p=B/'ROOT_TRACKLET_READBACK.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
