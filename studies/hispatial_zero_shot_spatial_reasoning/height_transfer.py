"""Seal region-given inputs and run frozen HiSpatial height evidence comparisons."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
from io import BytesIO
from importlib.metadata import version
import json
from pathlib import Path
import re
import time
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image

from download import DATA, ROOT, STUDY, download, read_json, sha256, write_json
from geometry_cohort import stable
from height_geometry import geometry
from height_measurements import metric
from object_region_plan import CONFIG as REGIONS_CONFIG, index_rows
from score import jsonl


CONFIG = STUDY / 'HEIGHT_TRANSFER.json'
SOURCE = DATA / 'spatialrgpt'
RUN = ROOT / 'outputs' / 'hispatial_height_transfer_20261122'
CATEGORIES = ('height_data','width_data')
COLORS = ('red','blue','green')
BGR_ON_RGB = ((255,0,0),(0,0,255),(0,255,0))
INPUT_COLUMNS = ('id','image_info','qa_info','text_q','bbox','image')
EVIDENCE_ARMS = ('prior_number_prompt','clipped_number_prompt','initial_state_prompt','clipped_state_prompt')


def fetch():
    meta = read_json(CONFIG)['dataset']
    repo, revision = meta['repository'], meta['revision']
    SOURCE.mkdir(parents=True,exist_ok=True,mode=0o700)
    url = f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/data'
    with urlopen(url,timeout=30) as response:
        assets = [e for e in json.load(response) if re.fullmatch(r'data/val-\d{5}-of-00003\.parquet',e['path'])]
    assert len(assets)==3

    def retrieve(entry):
        file = entry['path'].rsplit('/',1)[-1]
        size, digest = entry['size'],entry['lfs']['oid']
        download(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{entry["path"]}',SOURCE/file,size,digest)
        return dict(file=file,size=size,sha256=digest)

    with ThreadPoolExecutor(max_workers=3) as pool:
        assets=list(pool.map(retrieve,assets))
    write_json(SOURCE/'source.json',{**meta,'files':assets})


def source_files():
    manifest = read_json(SOURCE/'source.json')
    assert all(manifest[k]==v for k,v in read_json(CONFIG)['dataset'].items())
    files = []
    for item in manifest['files']:
        path=SOURCE/item['file']
        assert path.stat().st_size==item['size'] and sha256(path)==item['sha256']
        files.append(path)
    return files


def question_fields(row):
    qa = ast.literal_eval(row['qa_info'])
    if qa['type']!='quantitative' or qa['category'] not in CATEGORIES:
        return None
    refs = [f'the {name} (highlighted by a {color} box)' for name,color in zip(qa['class'],COLORS)]
    question = row['text_q']
    # The mirror supplies a question-only column with indexed region markers.
    tokens = [int(x) for x in re.findall(r'Region\s*\[(\d+)\]',question,flags=re.IGNORECASE)]
    assert len(tokens)==1 and 0<=tokens[0]<len(refs), (row['id'],tokens)
    first = tokens[0]
    prompt = '<image>\n '+re.sub(r'Region\s*\[(\d+)\]',lambda m:refs[int(m[1])],question,flags=re.IGNORECASE)
    bbox = ast.literal_eval(row['bbox'])
    assert len(bbox)>first and all(len(b)==4 and all(isinstance(v,int) for v in b) for b in bbox)
    info = ast.literal_eval(row['image_info'])
    return dict(id=row['id'],category=qa['category'],target=first,object=refs[first],boxes=bbox,prompt=prompt,
                source=info['dataset'],size=[info['width'],info['height']])


def prepare():
    import pyarrow.parquet as pq

    if (RUN/'input_seal.json').exists():
        raise FileExistsError(RUN/'input_seal.json')
    (RUN/'images').mkdir(parents=True,exist_ok=True,mode=0o700)
    paths = source_files()
    rows=[]
    total=0
    for file in paths:
        table=pq.ParquetFile(file)
        for batch in table.iter_batches(batch_size=16,columns=INPUT_COLUMNS):
            for record in batch.to_pylist():
                total+=1
                row=question_fields(record)
                if row is None:
                    continue
                image=record['image']['bytes']
                assert image is not None
                digest = hashlib.sha256(image).hexdigest()
                path=RUN/'images'/digest
                if not path.exists():
                    path.write_bytes(image)
                assert sha256(path)==digest
                row['image']=digest
                rows.append(row)
    assert len({x['id'] for x in rows})==len(rows)
    rows.sort(key=lambda x:(x['image'],x['boxes'],x['target'],x['id']))
    counts=Counter(x['category'] for x in rows)
    assert counts.keys()==set(CATEGORIES)
    jsonl(RUN/'inputs.jsonl',rows)
    write_json(RUN/'input_seal.json',dict(inputs=sha256(RUN/'inputs.jsonl'),protocol=sha256(CONFIG),
        input_columns=INPUT_COLUMNS,reference_answer_access=False,sources=sha256(SOURCE/'source.json'),
        regions=sha256(REGIONS_CONFIG),dataset_rows=total,counts=counts,frames=len({r['image'] for r in rows})))
    print('input cohort',dict(counts), 'images',len({r['image'] for r in rows}),flush=True)


def incoming(row, style, num, physical):
    result=row['prompt']
    if style.endswith('number_prompt'):
        note=f'Estimated complete height of {row["object"]}: {num:.3f} m.'
    else:
        note=(f'Object geometry for {row["object"]}:\n'
               f'Visible vertical extent: {physical["visible_height_m"]:.3f} m.\n'
               f'Estimated complete height: {num:.3f} m.\n'
               f'Bottom relative to the camera along estimated up: {physical["bottom_m"]:.3f} m.\n'
               'Geometric support: valid.')
    return result+'\n\nAdditional geometric evidence:\n'+note+'\nAnswer the original question.'


def infer(model_cache):
    import torch
    from skimage.segmentation import felzenszwalb
    from predict import BATCH_N, batch_query, cached_single, feature, release

    if (RUN/'prediction_seal.json').exists():
        raise FileExistsError(RUN/'prediction_seal.json')
    started=time.monotonic()
    locked=read_json(RUN/'input_seal.json')
    assert locked['inputs']==sha256(RUN/'inputs.jsonl') and locked['protocol']==sha256(CONFIG)
    assert locked['regions']==sha256(REGIONS_CONFIG) and not locked['reference_answer_access']
    cfg, proposals = read_json(CONFIG),read_json(REGIONS_CONFIG)
    roster=index_rows(RUN/'inputs.jsonl')
    assert len(roster)==sum(locked['counts'].values())
    cv2.setNumThreads(1)
    assert felzenszwalb(np.zeros((6,6,3)),channel_axis=-1).shape==(6,6)
    torch.manual_seed(20261122)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False
    model, moge, runtime = release(model_cache)

    @lru_cache(maxsize=12)
    def frame(key):
        path=RUN/'images'/key
        assert sha256(path)==key
        with Image.open(BytesIO(path.read_bytes())) as source:
            rgb=np.asarray(source.convert('RGB'))
        cache=RUN/'pointmaps'/f'{key}.npy'
        if cache.exists():
            xyz=torch.from_numpy(np.load(cache,allow_pickle=False))
        else:
            with torch.inference_mode():
                xyz=moge.apply_transform(rgb)
            partial=cache.with_suffix('.partial.npy')
            np.save(partial,xyz.numpy(),allow_pickle=False)
            partial.replace(cache)
        assert xyz.shape==(4,448,448)
        return rgb,xyz

    @lru_cache(maxsize=12)
    def model_input(key, boxes):
        rgb,xyz=frame(key)
        overlay=rgb.copy()
        for i,(x1,y1,x2,y2) in enumerate(boxes):
            color=BGR_ON_RGB[i] if i<len(BGR_ON_RGB) else (255,255,0)
            cv2.rectangle(overlay,(x1,y1),(x2,y2),color,2)
        resized=cv2.resize(overlay,(448,448))
        return resized,xyz,feature(model,resized,xyz)

    def infer_tasks(tasks, stage):
        tasks=sorted(tasks,key=lambda t:(t['image'],t['rects'],t['prompt']))
        partial={}
        for j in range(0,len(tasks),batch):
            block=tasks[j:j+batch]
            p=[model_input(t['image'],t['rects']) for t in block]
            response=batch_query(model,p,block) if len(block)>1 else [cached_single(model,p[0],block[0]['prompt'])]
            partial.update({(t['image'],t['rects'],t['prompt']):text for t,text in zip(block,response,strict=True)})
            if (j+len(block)) % 80 < len(block) or j+len(block)==len(tasks):
                print(stage,j+len(block),'/',len(tasks),round(time.monotonic()-started,1),'s',flush=True)
        return partial

    rect={q:tuple(map(tuple,row['boxes'])) for q,row in roster.items()}
    native={q:row['prompt'] for q,row in roster.items()}
    prior={q:cfg['prior_prompt'].format(object=row['object']) for q,row in roster.items()}
    alt={q:cfg['alternate_prior_prompt'].format(object=row['object']) for q,row in roster.items()}
    (RUN/'pointmaps').mkdir(exist_ok=True,mode=0o700)

    def tasks_for(prompt_sets, already):
        jobs={}
        for prompts in prompt_sets:
            for q,prompt in prompts.items():
                key=(roster[q]['image'],rect[q],prompt)
                if key not in already:
                    jobs[key]=dict(image=key[0],rects=key[1],prompt=key[2])
        return list(jobs.values())

    def response(q,prompt):
        return captured[roster[q]['image'],rect[q],prompt]

    probes=sorted(roster,key=lambda q:stable('height-parity/'+q))[:12]
    parity=0
    for j in range(0,len(probes),BATCH_N):
        ids=probes[j:j+BATCH_N]
        p=[model_input(roster[q]['image'],rect[q]) for q in ids]
        a=[cached_single(model,prepared,native[q]) for q,prepared in zip(ids,p,strict=True)]
        b=batch_query(model,p,[{'prompt':native[q]} for q in ids])
        parity+=sum(x==y for x,y in zip(a,b,strict=True))
    batch=BATCH_N if parity==len(probes) else 1
    print('generation parity',parity,'/',len(probes),'batch',batch,flush=True)
    first=tasks_for((native,prior,alt),{})
    captured=infer_tasks(first,'native and prior')
    start_evidence=time.monotonic()
    rows={}
    prompts={name:{} for name in EVIDENCE_ARMS}
    for q,row in roster.items():
        rgb,xyz=frame(row['image'])
        assert list(rgb.shape[1::-1])==row['size']
        boxes = np.array(row['boxes'],dtype=np.float64) * np.tile([448/rgb.shape[1],448/rgb.shape[0]],2)
        boxes=np.clip(boxes,0,448)
        value=metric(response(q,prior[q]),bare_meters=True)
        value_alt=metric(response(q,alt[q]),bare_meters=True)
        geom=geometry(cv2.resize(rgb,(448,448)),xyz.numpy(),boxes[row['target']],boxes,row['size'],value,proposals)
        rows[q] = dict(id=q,category=row['category'],source=row['source'],image=row['image'],
            geometry=geom,prior_m=value,alternate_prior_m=value_alt,
            native=response(q,native[q]),prior=response(q,prior[q]),alternate_prior=response(q,alt[q]))
        heights=dict(prior_number_prompt=value,clipped_number_prompt=geom.get('height_m'),
                     initial_state_prompt=geom.get('initial_height_m'),clipped_state_prompt=geom.get('height_m'))
        for arm,height in heights.items():
            if height is not None and height>0:
                prompts[arm][q]=incoming(row,arm,height,geom)
    jsonl(RUN/'geometry_stage.jsonl',rows.values())
    write_json(RUN/'pointmap_manifest.json',{key:sha256(RUN/'pointmaps'/f'{key}.npy')
                                           for key in sorted({r['image'] for r in roster.values()})})
    write_json(RUN/'geometry_seal.json',dict(rows=len(rows),geometry=sha256(RUN/'geometry_stage.jsonl'),
        code={name:sha256(STUDY/name) for name in ('height_transfer.py','height_geometry.py','height_measurements.py')},
        label_access=False,inputs=sha256(RUN/'input_seal.json'),pointmaps=sha256(RUN/'pointmap_manifest.json')))
    print('geometry',dict(Counter(row['geometry']['status'] for row in rows.values())),flush=True)
    second=tasks_for(prompts.values(),captured)
    captured.update(infer_tasks(second,'geometry QA'))
    for q,row in rows.items():
        row['evidence']={arm:dict(response=response(q,questions[q]) if q in questions else row['native'],
                                  used=q in questions,
                                  prompt_sha256=hashlib.sha256(questions[q].encode()).hexdigest() if q in questions else None)
                         for arm,questions in prompts.items()}
    jsonl(RUN/'predictions.jsonl',rows.values())
    write_json(RUN/'runtime.json',dict(model=runtime,parity_rows=len(probes),parity_equal=parity,batch=batch,
        geometry_packages={p:version(p) for p in ('scikit-image','numpy','scipy')},
        first_calls=len(first),evidence_calls=len(second),parity_calls=2*len(probes),frames=locked['frames'],
        seconds=time.monotonic()-started,evidence_stage_seconds=time.monotonic()-start_evidence,
        peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    assert not any(p.requires_grad for m in (model.model,moge.model) for p in m.parameters())
    write_json(RUN/'prediction_seal.json',dict(rows=len(rows),labels=False,training_steps=0,
        files={file:sha256(RUN/file) for file in ('input_seal.json','geometry_seal.json','predictions.jsonl','runtime.json')}))
    print('sealed',len(rows),'questions',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('fetch','prepare','predict'))
    parser.add_argument('--model-cache',type=Path)
    args=parser.parse_args()
    if args.action=='fetch':
        fetch()
    elif args.action=='prepare':
        prepare()
    else:
        if not args.model_cache:
            parser.error('--model-cache is required for predict')
        infer(args.model_cache)
