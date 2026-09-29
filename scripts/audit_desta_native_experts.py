"""Read back provider mapping, raw masks, physical media and all seals on CPU."""
import hashlib
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.desta_native_common import D,read,write,load,sha,verified,check_pins


def main():
    import numpy as np
    start=time.monotonic();rows=read(D/'DEV64.json');files={};media={};counts={'temporal':0,'spatial_frames':0,'total_frames':0,'spatial_queries':0}
    check_pins(read(D/'EXPERT_ASSETS.json')['pins'])
    for i,row in enumerate(rows):
        video=Path(row['input']['video_path'])
        if str(video) not in media:media[str(video)]=sha(video)
        assert media[str(video)]==row['input']['video_sha256']
        te=D/'experts/temporal'/f'{i:02}';se=D/'experts/spatial'/f'{i:02}'
        assert verified(te) and verified(se)
        t=read(te/'EVIDENCE.json');s=read(se/'EVIDENCE.json');ids=row['input']['frame_ids']
        assert t['frame_ids']==s['frame_ids']==ids and t['pixel_sha256']==s['pixel_sha256']
        assert t['key']==s['key']==row['key'] and not t['GT_read'] and not s['GT_read']
        slots=np.asarray(t['slots_physical']);expected=np.abs(np.asarray(ids)[None,:]-slots[:,None]).argmin(1)
        assert np.array_equal(expected,t['nearest_observation_indices'])
        if t['selected'] is not None:
            segments=np.asarray(t['segments_seconds']);scores=np.asarray(t['scores'])
            valid=[j for j,x in enumerate(segments) if np.isfinite(x).all() and x[1]>=x[0] and np.isfinite(scores[j])]
            winner=max(valid,key=lambda j:(float(scores[j]),-j));assert winner==t['selected']
            assert np.allclose(segments[winner]*row['input']['fps']+ids[0],t['interval_physical'],rtol=1e-6,atol=1e-4)
            counts['temporal']+=1
        check_pins({str(se/p):h for p,h in read(se/'FRAME_SEAL.json').items()})
        if (se/'MASKS.pt').exists():
            masks=load(se/'MASKS.pt');assert set(masks)=={str(x) for x in ids}
            for frame,m in masks.items():
                h,w=m['shape'];mask=np.unpackbits(m['packed'])[:h*w].reshape(h,w)
                ys,xs=np.where(mask)
                box=[float(xs.min()/w),float(ys.min()/h),float((xs.max()+1)/w),float((ys.max()+1)/h)] if len(xs) else None
                assert box==m['box']==s['boxes_by_frame'][frame]
                counts['spatial_frames']+=int(box is not None)
            counts['spatial_queries']+=int(any(x is not None for x in s['boxes_by_frame'].values()))
        counts['total_frames']+=len(ids)
        for ep in (te,se):
            for p in ep.rglob('*'):
                if p.is_file():files[str(p.relative_to(D))]=sha(p)
    write(D/'EXPERT_EVIDENCE_SEAL.json',dict(files=files,media_hashes=media,GT_read=False))
    write(D/'ROOT_EXPERT_READBACK.json',dict(status='passed',counts=counts,queries=64,parents=16,
          CPU_seconds=time.monotonic()-start,seal_sha=sha(D/'EXPERT_EVIDENCE_SEAL.json'),
          statement='Coverage is availability, not target correctness. No labels opened.',
          optional_sam2_hole_filling=False))
    print('EXPERT_READBACK',counts,flush=True)


if __name__=='__main__':main()
