"""Download the one authorized grounding teacher from its pinned official HF repo."""
import concurrent.futures,hashlib,json,os,shutil,sys,time,traceback
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha
OUT=ROOT/'artifacts/ptd_teacher_qualification_v1';DEST=ROOT/'checkpoints/UniVG-R1'
REV='8c137311d12e55ea1796df38a2d285a706b8d5ab';REPO='GD-ML/UniVG-R1'

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    started=OUT/'DOWNLOAD_STARTED.json'
    if not started.exists():write(started,dict(time=time.time(),repo=REPO,revision=REV,budget_seconds=14400))
    clock=read(started)['time'];meta=read(OUT/'primary_sources/univg_model.json');assert meta['sha']==REV
    files=[f for f in meta['siblings'] if not f['rfilename'].startswith('.') and f['rfilename']!='training_args.bin']
    assert shutil.disk_usage(ROOT).free>sum(f['size'] for f in files)+25*2**30
    tasks=[]
    for f in files:
        name=f['rfilename'];dest=DEST/name;url=f'https://huggingface.co/{REPO}/resolve/{REV}/{name}'
        if dest.exists():
            assert dest.stat().st_size==f['size']
            if f.get('lfs'):assert sha(dest)==f['lfs']['sha256']
            continue
        if not f.get('lfs'):
            r=requests.get(url,timeout=60);r.raise_for_status();assert len(r.content)==f['size'];dest.write_bytes(r.content);continue
        temp=DEST/(name+'.partial')
        if not temp.exists():
            fd=os.open(temp,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600);os.ftruncate(fd,f['size']);os.close(fd)
        for a in range(0,f['size'],128<<20):tasks.append((f,url,temp,a,min(a+(128<<20),f['size'])-1))
    def fetch(t):
        f,url,temp,a,b=t;receipt=OUT/'download_ranges'/f'{temp.name}.{a}.json';done=read(receipt)['bytes'] if receipt.exists() else 0
        fd=os.open(temp,os.O_RDWR)
        try:
            for attempt in range(3):
                if done==b-a+1:return
                assert time.time()-clock<14400,'download wall budget'
                start=a+done
                try:
                    with requests.get(url+f'?qualification_range={start}',headers={'Range':f'bytes={start}-{b}'},stream=True,timeout=(25,60)) as r:
                        assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{b}/{f["size"]}','HTTP range mismatch'
                        for block in r.iter_content(4<<20):
                            if block:
                                assert done+len(block)<=b-a+1;assert os.pwrite(fd,block,a+done)==len(block);done+=len(block)
                                status(receipt,dict(bytes=done,start=a,end=b));assert time.time()-clock<14400
                    assert done==b-a+1;return
                except Exception as e:print('RANGE_RETRY',temp.name,a,attempt,type(e).__name__,flush=True)
            raise RuntimeError('bounded download retry exhausted')
        finally:os.close(fd)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures=[ex.submit(fetch,t) for t in tasks]
        while not all(f.done() for f in futures):
            done=sum(read(p)['bytes'] for p in (OUT/'download_ranges').glob('*.json'))
            status(OUT/'DOWNLOAD_STATUS.json',dict(state='running',pid=os.getpid(),bytes=done,total=sum(t[4]-t[3]+1 for t in tasks),seconds=time.time()-clock))
            concurrent.futures.wait(futures,timeout=15)
        for f in futures:f.result()
    verified=[]
    for f in files:
        dest=DEST/f['rfilename'];temp=DEST/(f['rfilename']+'.partial')
        if not dest.exists():
            assert sha(temp)==f['lfs']['sha256'];temp.rename(dest)
        verified.append(dict(name=f['rfilename'],bytes=dest.stat().st_size,sha256=sha(dest)))
    write(OUT/'DOWNLOAD_COMPLETE.json',dict(repo=REPO,revision=REV,files=verified,seconds=time.time()-clock,safe_weights_only=True))
    status(OUT/'DOWNLOAD_STATUS.json',dict(state='completed_verified',seconds=time.time()-clock))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        status(OUT/'DOWNLOAD_STATUS.json',dict(state='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
