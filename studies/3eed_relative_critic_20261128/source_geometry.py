"""Collect frozen geometry reads from the immutable Waymo train/val rosters."""
from collections import Counter
from hashlib import file_digest
import json
import os
from pathlib import Path
import sys
from time import perf_counter


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=REPO.parent
RUN=ROOT/'runs/relative-critic-20261128'
SOURCE=ROOT/'runs/waymo-posttraining-20261128'
PRIOR=ROOT/'runs/four-paradigms-20261128'
WAYMO=ROOT/'assets/data/3eed_dataset/waymo'
UPSTREAM=ROOT/'assets/upstream-3eed'
MODEL=ROOT/'assets/models/Qwen3-VL-4B-Instruct/ebb281ec70b05090aa6165b016eac8ec08e71b17'
REG=REPO/'QWEN_CURRENT_METHOD.json'
BASELINE_SHA='3034cb6c0bde673a6972d3d60b0a5ca0863afd50f6a669c7f977feac4fbaf57e'


def digest(path):
    with Path(path).open('rb') as f:
        return file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path,data):
    path=Path(path)
    staging=path.with_suffix(path.suffix+'.tmp')
    staging.write_text(json.dumps(data,allow_nan=False,ensure_ascii=False,separators=(',',':'))+'\n')
    staging.replace(path)


def source_guard():
    """Reject target assets, label manifests and non-source run directories."""
    allowed={SOURCE/f for f in ('source-train.json','source-val.json','source-seal.json',
                                'source-frozen-2d.json','conditions-receipt.json')}
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,Path)):
            return
        path=Path(os.path.abspath(os.fsdecode(args[0])))
        if path.is_relative_to(ROOT/'runs'):
            if not (path in allowed or path.is_relative_to(SOURCE/'waymo-camera') or
                    path.is_relative_to(RUN/'source')):
                raise PermissionError(f'Source phase denied non-source run artifact: {path}')
        if (path.is_relative_to(ROOT/'assets/data') and not path.is_relative_to(WAYMO) or
                path.is_relative_to(ROOT/'assets/archives') or path.is_relative_to(ROOT/'assets/wildrefer-source') or
                path.name in ('drone_train.txt','drone_val.txt','quad_train.txt','quad_val.txt')):
            raise PermissionError(f'Source phase denied non-Waymo asset: {path}')
    sys.addaudithook(guard)


def runtime():
    sys.path.insert(0,str(REPO/'tools/3eed_recovery'))
    from run_qwen_3eed import check_geometry, registered_predictor, source_paths
    assert digest(REG)==BASELINE_SHA
    source_paths(UPSTREAM)
    check_geometry()
    return registered_predictor(read(REG))


def rosters():
    stamp=read(SOURCE/'source-seal.json')
    assert not stamp['target_read']
    result={}
    for part in ('train','val'):
        f=SOURCE/f'source-{part}.json'
        subset=UPSTREAM/f'data/splits/waymo_{part}.txt'
        assert digest(f)==stamp[part]['file'] and digest(subset)==stamp[part]['split']
        allowed=set(subset.read_text().split())
        population=read(f)
        assert len(population)==stamp[part]['n']
        for row in population:
            frame=row['frame']
            assert frame['target']=='waymo' and row['recording'] in allowed
            assert Path(frame['rgb']).is_relative_to(WAYMO) and Path(frame['points']).is_relative_to(WAYMO)
            frame['waymo_view']=('F','FL','FR','SL','SR')[frame['suffix']]
        result[part]=population
    assert not {r['recording'] for r in result['train']} & {r['recording'] for r in result['val']}
    assert not {r['rgb_sha'] for r in result['train']} & {r['rgb_sha'] for r in result['val']}
    return result,stamp


def decode():
    source_guard()
    runtime()
    import torch
    from run_full_qwen import batch_read,load_model

    out=RUN/'source'
    out.mkdir(parents=True,exist_ok=True)
    bank,pins=rosters()
    original=SOURCE/'source-frozen-2d.json'
    stopped=read(SOURCE/'conditions-receipt.json')
    old=read(original)
    assert digest(original) in stopped.values() or any(isinstance(v,dict) and digest(original) in v.values() for v in stopped.values())
    assert set(old)=={r['uid'] for r in bank['train']}
    processor,model,limit=load_model(MODEL)
    batches=Counter()
    start=perf_counter()
    made=Counter()
    with torch.inference_mode():
        for part,rows in bank.items():
            for task in ('2d','3d'):
                if part=='train' and task=='2d':
                    continue
                for i in range(0,len(rows),16):
                    batch=rows[i:i+16]
                    f=out/f'replies-{part}-{task}-{i:04d}.json'
                    if f.exists():
                        assert list(read(f))==[b['uid'] for b in batch]
                        continue
                    assert all(digest(b['frame']['rgb'])==b['rgb_sha'] for b in batch)
                    replies=batch_read(processor,model,limit,[b['frame'] for b in batch],task,batches)
                    write(f,dict(zip((b['uid'] for b in batch),replies,strict=True)))
                    made[task]+=len(replies)
                    if i%64==0 or i+16>=len(rows):
                        print('Source Qwen',part,task,min(i+16,len(rows)),len(rows),flush=True)
    response={part:{r['uid']:{} for r in members} for part,members in bank.items()}
    for uid,value in old.items():
        response['train'][uid]['2d']=value
    for path in sorted(out.glob('replies-*-*-*.json')):
        _,part,task,_=path.stem.split('-')
        for uid,value in read(path).items():
            response[part][uid][task]=value
    assert all(set(r)=={'2d','3d'} for g in response.values() for r in g.values())
    file=out/'frozen-source-private.json'
    write(file,response)
    write(out/'decode-seal.json',dict(payload=digest(file),source=pins,counts={k:len(v) for k,v in response.items()},
          reused_train_2d=len(old),reads_this_run=dict(made),target_read=False,qwen_updates=0,
          cache_2d=digest(original),seconds=perf_counter()-start))


if __name__=='__main__':
    decode()
