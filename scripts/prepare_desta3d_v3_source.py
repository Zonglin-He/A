"""Official full-source inventory and original HC1 extraction. CPU only."""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, shutil, subprocess, sys, tarfile, time, traceback, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/desta3d_v3/source_intake_v1'
BASE = ROOT/'downloads/dataset_intake_20260910'
VID = BASE/'annotations/datasets/VidSTG/annotations'
HC = BASE/'hcstvg/datasets/HC-STVG1/HC-STVG(v1-5660)'
DATA = ROOT/'data/desta3d_v3_source/hc1_train'
SEVEN = ROOT/'.cache/fullscale-7zip/usr/lib/7zip/7z'
RESERVE = 8*2**30
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def parent(name): return Path(name).stem.split('_',1)[1]
def validation_parents(parents):
    ranked=sorted(set(parents),key=lambda p:hashlib.sha256(('desta3d-v3-source-v1|20260928|'+p).encode()).hexdigest())
    return set(ranked[:math.ceil(.1*len(ranked))])
def index_zip(p):
    with zipfile.ZipFile(p) as z:
        return {Path(i.filename).stem:dict(archive=str(p),member=i.filename,bytes=i.file_size,crc32=i.CRC)
                for i in z.infolist() if i.filename.endswith('.mp4')}
def inventory():
    assert not (OUT/'MANIFEST.json').exists()
    paths=[VID/f'{s}_annotations.json' for s in ('train','val','test')]+[HC/'train.json',HC/'test.json']
    # Evaluation files: consume raw-parent identity only, never boxes/outcomes.
    train=read(paths[0]); forbidden={str(r['vid']) for p in paths[1:3] for r in read(p)}
    htrain=read(HC/'train.json'); htest=set(map(parent,read(HC/'test.json')))
    vp={str(r['vid']) for r in train}; hp=set(map(parent,htrain))
    assert not vp&forbidden and not hp&htest
    vh=validation_parents(vp); hh=validation_parents(hp)
    media={}
    archives=sorted((BASE/'datasets/VidOR').glob('training-video-part*.zip'))
    for p in archives:
        idx=index_zip(p);assert not media.keys()&idx.keys();media.update(idx)
    assert vp<=media.keys(),sorted(vp-media.keys())
    with zipfile.ZipFile(BASE/'datasets/VidOR/training-annotation.zip') as z:
        ann={Path(n).stem:n for n in z.namelist() if n.endswith('.json')}
    assert vp<=ann.keys()
    rows=[]
    for i,r in enumerate(train):
        for kind in ('captions','questions'):
            for j,q in enumerate(r[kind]):
                rows.append(dict(key=f'vid:{i}:{kind}:{j}',domain='Vid',parent=str(r['vid']),
                    split='validation' if str(r['vid']) in vh else 'train',annotation_index=i,query_kind=kind,
                    query_index=j,caption=q['description'],target_id=q['target_id'],media=media[str(r['vid'])],
                    trajectory_member=ann[str(r['vid'])]))
    for name,a in sorted(htrain.items()):
        rows.append(dict(key='hc1:'+name,domain='HC1',parent=parent(name),
            split='validation' if parent(name) in hh else 'train',annotation_key=name,caption=a['caption'],
            media=dict(path=str(DATA/name),original_archive=str(HC/'video(5660)/v1.zip'),verification='pending_original_archive_extraction')))
    counts={}
    for domain in ('Vid','HC1'):
        counts[domain]={}
        for split in ('train','validation'):
            part=[r for r in rows if r['domain']==domain and r['split']==split]
            counts[domain][split]=dict(queries=len(part),parents=len({r['parent'] for r in part}))
    h2=BASE/'hcstvg/datasets/HC-STVG2/HC-STVG/anno_v2'
    h2train=set(map(parent,read(h2/'train_v2.json')));h2val=set(map(parent,read(h2/'val_v2.json')))
    info=dict(status='metadata_complete_media_verification_pending',counts=counts,queries=len(rows),
       HC1_train_test_parent_overlap=0,Vid_train_official_val_test_parent_overlap=0,
       HC2_train_val_parent_overlap=len(h2train&h2val),HC1_train_HC2_val_parent_overlap=len(hp&h2val),
       split_namespace='desta3d-v3-source-v1|20260928|parent',validation_fraction=.1,
       historical_exposure='Not an unseen-development claim. Old HC2 evaluation parents overlapping HC1 train ineligible as independent target tests.',
       annotation_files={str(p):sha(p) for p in paths+[h2/'train_v2.json',h2/'val_v2.json']},
       media_inventory='ZIP central-directory membership only; not full decode or CRC validation',
       GPU_seconds=0,target_outcomes_used=False,benchmark_files_opened_for_identity_only=True)
    put(OUT/'MANIFEST.json',rows);put(OUT/'INVENTORY.json',info)
    put(OUT/'INVENTORY_SEAL.json',dict(pins={str(p):sha(p) for p in [Path(__file__),OUT/'MANIFEST.json',OUT/'INVENTORY.json']},time=time.time()))
    print(json.dumps(info,ensure_ascii=False),flush=True)
def extract_hc():
    assert (OUT/'INVENTORY_SEAL.json').exists() and not (OUT/'HC_MEDIA_COMPLETE.json').exists()
    assert not (OUT/'HC_EXTRACT_STARTED.json').exists(),'new version required after incomplete extraction'
    target=read(HC/'train.json');DATA.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(ROOT).free>RESERVE+39_000_000_000
    put(OUT/'HC_EXTRACT_STARTED.json',dict(pid=os.getpid(),time=time.time(),cap_bytes=39_000_000_000,reserve=RESERVE))
    t=time.monotonic();found={};total=0;status='failed'
    with (OUT/'HC_EXTRACT.stderr').open('xb') as log:
        proc=subprocess.Popen([str(SEVEN),'x','-so',str(HC/'video(5660)/v1.zip'),'v1.tgz'],stdout=subprocess.PIPE,stderr=log)
        try:
            with tarfile.open(fileobj=proc.stdout,mode='r|gz') as tar:
                for m in tar:
                    name=Path(m.name).name
                    if not m.isfile() or name not in target:continue
                    assert name not in found and total+m.size<=39_000_000_000
                    assert shutil.disk_usage(ROOT).free-m.size>RESERVE
                    dest=DATA/name;tmp=dest.with_suffix(dest.suffix+'.partial');h=hashlib.sha256();n=0
                    with tar.extractfile(m) as src,tmp.open('xb') as f:
                        for b in iter(lambda:src.read(8<<20),b''):
                            h.update(b);n+=len(b);f.write(b)
                    assert n==m.size;assert not dest.exists();tmp.rename(dest);total+=n
                    found[name]=dict(path=str(dest),member=m.name,bytes=n,sha256=h.hexdigest())
                    if len(found)%100==0:print('HC_ORIGINAL_EXTRACT',len(found),len(target),total,flush=True)
            while proc.stdout.read(8<<20):pass
            assert proc.wait()==0 and set(found)==set(target)
            put(OUT/'HC_MEDIA_COMPLETE.json',dict(members=found,original_outer_ZIP_CRC_verified=True,
                bytes=total,files=len(found),GPU_seconds=0,CPU_wall_seconds=time.monotonic()-t))
            status='completed'
        except BaseException as e:
            put(OUT/'HC_EXTRACT_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),completed_members=found));raise
        finally:
            proc.stdout.close()
            if proc.poll() is None:proc.terminate();proc.wait()
            put(OUT/'HC_EXTRACT_RECEIPT.json',dict(status=status,CPU_wall_seconds=time.monotonic()-t,GPU_seconds=0,
                bytes=total,files=len(found),free_bytes=shutil.disk_usage(ROOT).free))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['inventory','extract-hc']);a=ap.parse_args()
    inventory() if a.action=='inventory' else extract_hc()
