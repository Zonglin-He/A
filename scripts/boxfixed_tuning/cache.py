"""Content-bound immutable capture reuse and bounded RGB cache for the new full run."""
import hashlib,json,time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from stvg.runtime import sha
from stvg.decode import decode
from stvg.teacher_bridge import ResidentTeacher,evidence_key

def binding(record,runtime):
    m=record['metadata']
    body=dict(dataset=record['dataset'],runtime_fingerprint=runtime,
        input={k:v for k,v in m.items()if k!='video_path'})
    return hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

class RGBCache:
    def __init__(self,budget=512*2**20):
        self.budget=budget;self.entries=OrderedDict();self.bytes=0;self.hits=self.misses=0
    def get(self,metadata,dataset):
        key=(dataset,metadata['video_sha256'],tuple(metadata['frame_ids']),metadata['width'],metadata['height'])
        start=time.perf_counter()
        if key in self.entries:
            assert sha(metadata['video_path'])==metadata['video_sha256']
            a=self.entries.pop(key);self.entries[key]=a;self.hits+=1
            return a.copy(),list(metadata['frame_ids']),dict(hit=True,seconds=time.perf_counter()-start)
        a,ids=decode(metadata,dataset);self.misses+=1
        if a.nbytes<=self.budget:
            while self.entries and self.bytes+a.nbytes>self.budget:
                _,old=self.entries.popitem(last=False);self.bytes-=old.nbytes
            self.entries[key]=a.copy();self.bytes+=a.nbytes
        return a,ids,dict(hit=False,seconds=time.perf_counter()-start)

class ReusingTeacher(ResidentTeacher):
    def __init__(self,cache_dir,fingerprint,previous=()):
        super().__init__(cache_dir,fingerprint);self.previous=[Path(p)for p in previous]
    def __call__(self,frames,caption,*,positions,frame_ids,dataset):
        key,_=evidence_key(frames,caption,positions=positions,frame_ids=frame_ids,fingerprint=self.fingerprint,dataset=dataset)
        original=self.cache_dir
        try:
            if not(original/(key+'.json')).exists():
                for p in self.previous:
                    if(p/(key+'.json')).exists():self.cache_dir=p;break
            return super().__call__(frames,caption,positions=positions,frame_ids=frame_ids,dataset=dataset)
        finally:self.cache_dir=original

def from_native(cap,logits):
    return dict(Source=cap['Source'],E=cap['E'],frame_ids=cap['frame_ids'],positions8=cap['planned_positions'],
        records=cap['records'],geometry=cap['geometry'].state_dict(),logits=logits,
        native_interval=cap['intervals']['Native'],GT_read=False)
