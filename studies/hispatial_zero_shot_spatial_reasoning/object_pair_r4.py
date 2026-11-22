"""Run the answer-free native and visible-geometry stages of the R4 pair audit."""
import argparse
import ast
from collections import Counter
from functools import lru_cache
import hashlib
from io import BytesIO
from pathlib import Path
import re
import shutil
import time

import cv2
import numpy as np
from PIL import Image

from download import ROOT, STUDY, read_json, sha256, write_json
from geometry_cohort import stable
from height_transfer import BGR_ON_RGB, COLORS, INPUT_COLUMNS, REGIONS_CONFIG, RUN as HEIGHT_RUN, SOURCE, source_files
from object_pair_r4_geometry import CATEGORIES, object_points, prepare_scene, readout
from object_region_plan import index_rows
from score import jsonl


CONFIG=STUDY/'OBJECT_PAIR_R4.json'
RUN=ROOT/'outputs'/'hispatial_object_pair_r4_20261122'
MODEL_SEAL='native_seal.json'
GEOMETRY_SEAL='geometry_seal.json'


def question_fields(row):
    qa=ast.literal_eval(row['qa_info'])
    if qa['type']!='quantitative' or qa['category'] not in CATEGORIES:
        return None
    question=row['text_q']
    refs=[f'the {name} (highlighted by a {color} box)' for name,color in zip(qa['class'],COLORS)]
    token=re.compile(r'Region\s*\[(\d+)\]',re.I)
    indices=[int(t) for t in token.findall(question)]
    assert indices==[0,1] and len(refs)>=2,(row['id'],indices)
    boxes=ast.literal_eval(row['bbox'])
    assert len(boxes)>=2 and all(len(b)==4 and all(isinstance(v,int) for v in b) for b in boxes)
    meta=ast.literal_eval(row['image_info'])
    prompt='<image>\n '+token.sub(lambda m:refs[int(m[1])],question)
    return dict(id=row['id'],category=qa['category'],prompt=prompt,objects=refs[:2],boxes=boxes,
                source=meta['dataset'],size=[meta['width'],meta['height']])


def decode_coco(rle):
    """Decode COCO compressed runs without importing a model or score package."""
    h,w=rle['size']
    assert 0<h<=10000 and 0<w<=10000 and h*w<=25000000
    counts=rle['counts']
    if isinstance(counts,str):
        runs=[]
        i=0
        while i<len(counts):
            value=0
            shift=0
            while True:
                code=ord(counts[i])-48
                i+=1
                assert 0<=code<=63 and shift<35
                value|=(code&31)<<shift
                shift+=5
                if not code&32:
                    if code&16:
                        value-=1<<shift
                    break
            runs.append(value+(runs[-2] if len(runs)>2 else 0))
    else:
        runs=counts
    assert min(runs)>=0 and sum(runs)==h*w
    return np.repeat(np.arange(len(runs))%2,runs).astype(np.uint8).reshape((h,w),order='F')


def prepare():
    import pyarrow.parquet as pq

    if (RUN/'input_seal.json').exists():
        raise FileExistsError(RUN/'input_seal.json')
    for name in ('images','benchmark_masks'):
        (RUN/name).mkdir(exist_ok=True,parents=True,mode=0o700)
    meta=read_json(SOURCE/'source.json')
    assert all(meta[k]==v for k,v in read_json(CONFIG)['dataset'].items())
    columns=(*INPUT_COLUMNS,'rle')
    roster=[]
    total=0
    for path in source_files():
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16,columns=columns):
            for source in batch.to_pylist():
                total+=1
                row=question_fields(source)
                if row is None:
                    continue
                data=source['image']['bytes']
                key=hashlib.sha256(data).hexdigest()
                image=RUN/'images'/key
                if not image.exists():
                    image.write_bytes(data)
                assert sha256(image)==key
                stored=None
                if source['rle']:
                    rles=ast.literal_eval(source['rle'])
                    if len(rles)>=2:
                        assert all(r['size']==row['size'][::-1] for r in rles[:2])
                        masks=np.stack([cv2.resize(decode_coco(r),(448,448),interpolation=cv2.INTER_NEAREST)[2::4,2::4]
                                        for r in rles[:2]])
                        data=np.packbits(masks).tobytes()
                        stored=hashlib.sha256(data).hexdigest()
                        (RUN/'benchmark_masks'/stored).write_bytes(data)
                roster.append(dict(**row,image=key,mask=stored))
    roster.sort(key=lambda r:(r['image'],r['boxes'],r['id']))
    counts=Counter(r['category'] for r in roster)
    assert len(roster)==len({r['id'] for r in roster})==483
    assert set(counts)==set(CATEGORIES)
    jsonl(RUN/'inputs.jsonl',roster)
    write_json(RUN/'input_seal.json',dict(inputs=sha256(RUN/'inputs.jsonl'),config=sha256(CONFIG),
        sources=sha256(SOURCE/'source.json'),regions=sha256(REGIONS_CONFIG),columns=columns,labels=False,
        counts=counts,dataset_rows=total,frames=len({r['image'] for r in roster}),
        rows_with_benchmark_masks=sum(r['mask'] is not None for r in roster)))
    print('prepared',dict(counts),'frames',len({r['image'] for r in roster}),flush=True)


def locked_roster():
    seal=read_json(RUN/'input_seal.json')
    assert seal['inputs']==sha256(RUN/'inputs.jsonl') and seal['config']==sha256(CONFIG)
    assert seal['sources']==sha256(SOURCE/'source.json') and seal['regions']==sha256(REGIONS_CONFIG)
    assert not seal['labels'] and 'conversations' not in seal['columns']
    rows=index_rows(RUN/'inputs.jsonl')
    assert len(rows)==sum(seal['counts'].values())
    return rows


def read_rgb(key):
    image=(RUN/'images'/key).read_bytes()
    assert hashlib.sha256(image).hexdigest()==key
    with Image.open(BytesIO(image)) as frame:
        return np.asarray(frame.convert('RGB'))


def check_seal(name):
    seal=read_json(RUN/name)
    assert not seal['labels']
    assert seal['input_seal']==sha256(RUN/'input_seal.json')
    for file,digest in seal['files'].items():
        assert sha256(RUN/file)==digest,file
    return seal


class FrozenRunner:
    def __init__(self,cache):
        import torch
        from predict import BATCH_N,batch_query,cached_single,feature,release

        cv2.setNumThreads(1)
        torch.manual_seed(20261122)
        torch.set_num_threads(3)
        torch.backends.cudnn.benchmark=False
        self.model,self.moge,self.runtime=release(cache)
        self.torch=torch
        self.batch_n=BATCH_N
        self.query=batch_query
        self.single=cached_single
        self.encode=feature
        self.reused=set()
        self.cache_root=RUN/'pointmaps'
        self.prior={}
        if (HEIGHT_RUN/'runtime.json').exists() and (HEIGHT_RUN/'pointmap_manifest.json').exists():
            prior_runtime=read_json(HEIGHT_RUN/'runtime.json')['model']
            if prior_runtime==self.runtime:
                self.prior=read_json(HEIGHT_RUN/'pointmap_manifest.json')
        self.cache_root.mkdir(exist_ok=True,mode=0o700)

    @lru_cache(maxsize=12)
    def frame(self,key):
        rgb=read_rgb(key)
        target=self.cache_root/f'{key}.npy'
        if not target.exists():
            if key in self.prior:
                original=HEIGHT_RUN/'pointmaps'/target.name
                assert sha256(original)==self.prior[key]
                shutil.copyfile(original,target)
                self.reused.add(key)
            else:
                with self.torch.inference_mode():
                    pts=self.moge.apply_transform(rgb)
                partial=target.with_suffix('.part.npy')
                np.save(partial,pts.numpy(),allow_pickle=False)
                partial.replace(target)
        xyz=self.torch.from_numpy(np.load(target,allow_pickle=False))
        assert xyz.shape==(4,448,448)
        return rgb,xyz

    @lru_cache(maxsize=12)
    def prepared(self,key,boxes):
        rgb,xyz=self.frame(key)
        overlay=rgb.copy()
        for i,(x1,y1,x2,y2) in enumerate(boxes):
            color=BGR_ON_RGB[i] if i<len(BGR_ON_RGB) else (255,255,0)
            cv2.rectangle(overlay,(x1,y1),(x2,y2),color,2)
        resized=cv2.resize(overlay,(448,448))
        return resized,xyz,self.encode(self.model,resized,xyz)

    def parity(self,rows):
        ordered=sorted(rows,key=lambda r:stable('object-pair-parity/'+r['id']))[:12]
        equal=0
        for i in range(0,len(ordered),self.batch_n):
            group=ordered[i:i+self.batch_n]
            inputs=[self.prepared(r['image'],tuple(map(tuple,r['boxes']))) for r in group]
            single=[self.single(self.model,v,r['prompt']) for v,r in zip(inputs,group,strict=True)]
            batch=self.query(self.model,inputs,group)
            equal+=sum(x==y for x,y in zip(single,batch,strict=True))
        batch=self.batch_n if equal==len(ordered) else 1
        return batch,equal,len(ordered)

    def infer(self,tasks,batch):
        outputs={}
        start=time.monotonic()
        for i in range(0,len(tasks),batch):
            group=tasks[i:i+batch]
            ready=[self.prepared(r['image'],tuple(map(tuple,r['boxes']))) for r in group]
            texts=self.query(self.model,ready,group) if len(group)>1 else [self.single(self.model,ready[0],group[0]['prompt'])]
            outputs.update({r['id']:text for r,text in zip(group,texts,strict=True)})
            completed=i+len(group)
            if completed%64<len(group) or completed==len(tasks):
                print('queries',completed,'/',len(tasks),round(time.monotonic()-start,1),'s',flush=True)
        return outputs

    def assert_frozen(self):
        assert not any(p.requires_grad for model in (self.model.model,self.moge.model) for p in model.parameters())


def predict(cache):
    if (RUN/MODEL_SEAL).exists():
        raise FileExistsError(RUN/MODEL_SEAL)
    roster=locked_roster()
    started=time.monotonic()
    runner=FrozenRunner(cache)
    rows=list(roster.values())
    size,equal,parity=runner.parity(rows)
    print('batch parity',equal,'/',parity,'batch',size,flush=True)
    outputs=runner.infer(rows,size)
    assert set(outputs)==set(roster)
    jsonl(RUN/'native.jsonl',[dict(id=q,response=outputs[q]) for q in roster])
    keys={r['image'] for r in rows}
    write_json(RUN/'pointmap_manifest.json',{key:sha256(RUN/'pointmaps'/f'{key}.npy') for key in sorted(keys)})
    write_json(RUN/'native_runtime.json',dict(model=runner.runtime,accuracy_parity=equal,parity_count=parity,
        batch_size=size,model_calls=len(rows),pointmap_frames=len(keys),reused_pointmaps=len(runner.reused),
        seconds=time.monotonic()-started,peak_cuda_bytes=runner.torch.cuda.max_memory_allocated()))
    runner.assert_frozen()
    write_json(RUN/MODEL_SEAL,dict(rows=len(rows),labels=False,training_steps=0,input_seal=sha256(RUN/'input_seal.json'),
        files={p:sha256(RUN/p) for p in ('native.jsonl','pointmap_manifest.json','native_runtime.json')}))
    print('sealed native',len(rows),flush=True)


def geometry():
    import importlib.metadata

    if (RUN/GEOMETRY_SEAL).exists():
        raise FileExistsError(RUN/GEOMETRY_SEAL)
    check_seal(MODEL_SEAL)
    roster=locked_roster()
    cfg=read_json(REGIONS_CONFIG)
    files=read_json(RUN/'pointmap_manifest.json')
    started=time.monotonic()

    @lru_cache(maxsize=4)
    def scene(key):
        rgb=read_rgb(key)
        path=RUN/'pointmaps'/f'{key}.npy'
        assert sha256(path)==files[key]
        return prepare_scene(cv2.resize(rgb,(448,448)),np.load(path,allow_pickle=False),cfg)

    @lru_cache(maxsize=10)
    def pair(key,boxes,size,mask):
        roi=np.clip(np.asarray(boxes,dtype=np.float64)*np.tile(448/np.asarray(size),2),0,448)
        diagnostic=None
        if mask is not None:
            packed=(RUN/'benchmark_masks'/mask).read_bytes()
            assert hashlib.sha256(packed).hexdigest()==mask
            diagnostic=np.unpackbits(np.frombuffer(packed,np.uint8),count=2*112*112).astype(bool).reshape(2,112,112)
        return object_points(scene(key),roi,diagnostic)

    rows=[]
    for i,(q,item) in enumerate(roster.items(),1):
        s=pair(item['image'],tuple(map(tuple,item['boxes'][:2])),tuple(item['size']),item['mask'])
        output,support=readout(s,item['category'])
        rows.append(dict(id=q,geometry=output,support=support))
        if i%96==0 or i==len(roster):
            print('geometry',i,'/',len(roster),round(time.monotonic()-started,1),'s',flush=True)
    jsonl(RUN/'geometry.jsonl',rows)
    write_json(RUN/'geometry_runtime.json',dict(seconds=time.monotonic()-started,model_calls=0,
        versions={p:importlib.metadata.version(p) for p in ('scikit-image','numpy','scipy')}))
    write_json(RUN/GEOMETRY_SEAL,dict(rows=len(rows),labels=False,input_seal=sha256(RUN/'input_seal.json'),
        native_seal=sha256(RUN/MODEL_SEAL),files={p:sha256(RUN/p) for p in ('geometry.jsonl','geometry_runtime.json')},
        code=sha256(STUDY/'object_pair_r4_geometry.py')))
    print('sealed geometry',len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','predict','geometry'))
    parser.add_argument('--model-cache',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':
        prepare()
    elif args.action=='predict':
        if args.model_cache is None:
            parser.error('--model-cache is required for predict')
        predict(args.model_cache)
    else:
        geometry()
