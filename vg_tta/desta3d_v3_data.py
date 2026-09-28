"""Full official-source rows, bounded ZIP media, labels separate from inputs."""
from __future__ import annotations
import hashlib,importlib.util,json,math,os,random,shutil,zipfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'downloads/dataset_intake_20260910'
INTAKE=ROOT/'artifacts/desta3d_v3/source_intake_v1'
MEDIA=ROOT/'artifacts/desta3d_v3/source_media_audit_v1'
def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def media_key(r):return 'vid-'+r['parent'] if r['domain']=='Vid' else 'hc1-'+Path(r['annotation_key']).stem
def balanced_epoch(rows,epoch,seed=20260928):
    pools={d:[i for i,r in enumerate(rows) if r['domain']==d] for d in ('Vid','HC1')}
    if not all(pools.values()):raise ValueError('both domains need training rows')
    rng=random.Random(seed+epoch);n=max(map(len,pools.values()));orders={}
    for d,pool in pools.items():
        order=[]
        while len(order)<n:
            block=pool.copy();rng.shuffle(block);order.extend(block)
        orders[d]=order[:n]
    return [i for pair in zip(orders['Vid'],orders['HC1']) for i in pair]
def physical_grid(start,end,fps,max_frames=32):
    """Official 2fps, [start,end) physical range, cap32 for registered memory."""
    if not 0<=start<end or not math.isfinite(fps) or fps<=0:raise ValueError('invalid media grid')
    n=min(max(math.floor((end-start)/fps*2),4),max_frames,end-start)
    ids=torch.linspace(start,end-1,n).round().long().tolist()
    assert ids==sorted(set(ids));return ids
def box_xyxy(b,w,h,xywh=False):
    x,y,a,c=map(float,b);a=x+a if xywh else a;c=y+c if xywh else c
    x,y,a,c=max(0,x),max(0,y),min(w,a),min(h,c)
    if not all(math.isfinite(v) for v in [x,y,a,c]) or not x<a or not y<c:return [0.,0.,0.,0.],False
    return [x/w,y/h,a/w,c/h],True
class SourcePool:
    def __init__(self,cache_limit=2*2**30,load_annotations=True):
        self.rows=read(INTAKE/'MANIFEST.json');self.cache=ROOT/'.cache/desta3d_v3_source_media';self.cache.mkdir(parents=True,exist_ok=True)
        self.cache_limit=cache_limit
        self.vid=read(BASE/'annotations/datasets/VidSTG/annotations/train_annotations.json') if load_annotations else None
        self.hc=read(BASE/'hcstvg/datasets/HC-STVG1/HC-STVG(v1-5660)/train.json') if load_annotations else None
        self.annzip=zipfile.ZipFile(BASE/'datasets/VidOR/training-annotation.zip') if load_annotations else None;self.last_ann=None
        self.repairs={}
        p=ROOT/'artifacts/desta3d_v3/hc_media_repair_v1/COMPLETE.json'
        if p.exists():self.repairs=read(p)['repairs']
        p=ROOT/'external/ParallelTubeDecoding/data/prepare_vidstg.py';spec=importlib.util.spec_from_file_location('desta3d_v3_official_builder',p)
        self.builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.builder)
    def media(self,r,materialize=True):
        key=media_key(r);a=self.repairs.get(key) or read(MEDIA/'media'/(key+'.json'))
        if a['status']!='probe_passed_full_decode_not_yet_checked':raise ValueError('unavailable_media:'+key)
        s=a['source']
        if 'archive' not in s:return Path(s['path']),a
        dest=self.cache/(key+'.mp4')
        if materialize:
            if dest.exists():
                assert digest(dest)==a['sha256'];os.utime(dest,None)
            else:
                size=s['bytes'];files=sorted(self.cache.glob('*.mp4'),key=lambda p:p.stat().st_mtime);used=sum(p.stat().st_size for p in files)
                while files and (used+size>self.cache_limit or shutil.disk_usage(ROOT).free-size<8*2**30):
                    old=files.pop(0);used-=old.stat().st_size;old.unlink() # only this module's disposable original-media extracts
                assert size<=self.cache_limit and shutil.disk_usage(ROOT).free-size>8*2**30
                tmp=dest.with_suffix('.partial')
                with zipfile.ZipFile(s['archive']) as z,z.open(s['member']) as src,tmp.open('xb') as f:
                    for b in iter(lambda:src.read(8<<20),b''):f.write(b)
                assert tmp.stat().st_size==size and digest(tmp)==a['sha256'];tmp.rename(dest)
        return dest,a
    def example(self,r,*,labels=True,materialize=True):
        path,media=self.media(r,materialize);m=media['metadata'];w,h=m['width'],m['height'];n=m['frame_count'];fps=m['fps']
        if r['domain']=='Vid':
            a=self.vid[r['annotation_index']];assert str(a['vid'])==r['parent']
            assert (w,h)==(a['width'],a['height'])
            # Original frame-index labels need an identical frame rate. An
            # unexplained resampling is a data issue, never silently shifted GT.
            if abs(fps/a['fps']-1)>1e-4:raise ValueError('annotation_media_fps_mismatch')
            start=int(a['used_segment']['begin_fid']);end=min(n,int(a['used_segment']['end_fid'])+1)
            interval=[int(a['temporal_gt']['begin_fid']),int(a['temporal_gt']['end_fid'])]
        else:
            a=self.hc[r['annotation_key']];assert (w,h)==(a['width'],a['height'])
            if not 0<=a['img_num']-n<=2:raise ValueError('HC annotation_media_frame_count_mismatch')
            start,end=0,n;interval=[a['st_frame']-1,a['st_frame']-1+len(a['bbox'])]
        if not 0<=interval[0]<interval[1]<=n:raise ValueError('event_outside_actual_media')
        ids=physical_grid(start,end,fps)
        q=dict(caption=r['caption'],source=r['parent'],original_video_id=r['parent'] if r['domain']=='Vid' else Path(r['annotation_key']).stem,
               video_path=str(path),video_sha256=media['sha256'],frame_ids=ids,start_frame=start,end_frame=end,
               kind='vidstg' if r['domain']=='Vid' else 'hcstvg',**m)
        row=dict(key=r['key'],source=r['parent'],split=r['split'],domain=r['domain'],input=q)
        if not labels:return row,None
        if r['domain']=='Vid':
            if self.last_ann is None or self.last_ann[0]!=r['parent']:
                raw=json.loads(self.annzip.read(r['trajectory_member']));assert str(raw['video_id'])==r['parent'];self.last_ann=(r['parent'],raw)
            raw=self.last_ann[1];track={}
            for fid in ids:
                boxes=[b for b in raw['trajectories'][fid] if int(b['tid'])==int(r['target_id'])] if fid<len(raw['trajectories']) else []
                if len(boxes)>1:raise ValueError('duplicate_target_in_frame')
                if boxes:
                    b=boxes[0]['bbox'];track[fid]=[b['xmin'],b['ymin'],b['xmax'],b['ymax']]
        else:track={interval[0]+i:[b[0],b[1],b[0]+b[2],b[1]+b[3]] for i,b in enumerate(a['bbox'])}
        boxes=[];valid=[];active=[];tokens=[]
        for pos,fid in enumerate(ids,1):
            b,ok=box_xyxy(track.get(fid,[0,0,0,0]),w,h);inside=interval[0]<=fid<interval[1]
            boxes.append(b);valid.append(ok);active.append(inside)
            if ok and inside:
                tok=self.builder.normalize_box([b[0]*w,b[1]*h,b[2]*w,b[3]*h],w,h)
                if tok is not None:tokens.append((pos,tok))
        response=self.builder.build_record(str(path),r['caption'].strip(),tokens)
        # Never let a missing event-edge box redefine the GT endpoints.
        event_positions=[i+1 for i,x in enumerate(active) if x]
        if [p for p,b in tokens]!=event_positions:response=None
        record=dict(key=r['key'],source=r['parent'],split=r['split'],frame_ids=ids,fps=fps,width=w,height=h,
            event_interval=dict(begin_fid=interval[0],end_fid=interval[1]),event_active=active,boxes_xyxy=boxes,box_valid=valid,
            response_eligible=response is not None,response=response['conversations'][1]['value'] if response else None,
            response_box_positions_1based=[p for p,b in tokens],source_GT=True,
            missing_box_rule='unknown referent support; retain temporal BCE; no partial-box redefinition of endpoint')
        return row,record
