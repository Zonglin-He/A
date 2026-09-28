"""Resume a pinned public checkpoint using validated HTTP ranges and LFS SHA256.

No GPU use. No mirrors, quantization, alternate weights, or implicit updates.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib,json,os,shutil,time,requests,traceback
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/desta3d_v3/external_privileged_opd_v1'
DEST=ROOT/'checkpoints/LLaVA-ST-Qwen2-7B'

def main():
    reg=json.loads((OUT/'MODEL_DOWNLOAD_REGISTRATION.json').read_text());started=time.time()
    rev=reg['revision'];repo=reg['repo'];chunk=32*2**20
    assert shutil.disk_usage(ROOT).free>sum(f['size'] for f in reg['files'])+8*2**30
    (OUT/'HTTP_DOWNLOAD_STARTED.json').write_text(json.dumps(dict(pid=os.getpid(),time=started,revision=rev,chunk_bytes=chunk,workers=8)))
    receipts=OUT/'download_ranges';receipts.mkdir(exist_ok=True)
    done=[]
    for f in reg['files']:
        target=DEST/f['path'];url=f'https://huggingface.co/{repo}/resolve/{rev}/{f["path"]}'
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.stat().st_size==f['size']:
            actual=hashlib.file_digest(target.open('rb'),'sha256').hexdigest()
            if not f['lfs_sha256'] or actual==f['lfs_sha256']:
                done.append({**f,'actual_sha256':actual});continue
        part=target.with_suffix(target.suffix+'.http-incomplete')
        fd=os.open(part,os.O_CREAT|os.O_RDWR,0o644);os.ftruncate(fd,f['size'])
        def fetch(start):
            end=min(start+chunk,f['size'])-1;rp=receipts/f'{f["path"]}.{start}.json'
            if rp.exists():
                old=json.loads(rp.read_text());data=os.pread(fd,end-start+1,start)
                if hashlib.sha256(data).hexdigest()==old['sha256']:return
            for attempt in range(4):
                try:
                    with requests.get(url+f'?download=true&offset={start}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,120)) as r:
                        r.raise_for_status()
                        if r.status_code==206:assert r.headers['Content-Range']==f'bytes {start}-{end}/{f["size"]}'
                        else:assert start==0 and end==f['size']-1 and r.status_code==200
                        count=0;h=hashlib.sha256()
                        for b in r.iter_content(2**20):
                            assert count+len(b)<=end-start+1
                            assert os.pwrite(fd,b,start+count)==len(b);count+=len(b);h.update(b)
                        assert count==end-start+1
                    rp.write_text(json.dumps(dict(start=start,end=end,sha256=h.hexdigest(),bytes=count)))
                    return
                except Exception:
                    if attempt==3:raise
                    time.sleep(1+attempt)
        try:
            starts=list(range(0,f['size'],chunk))
            with ThreadPoolExecutor(max_workers=8) as pool:
                for i,fut in enumerate(as_completed([pool.submit(fetch,s) for s in starts]),1):
                    fut.result()
                    (OUT/'DOWNLOAD_PROGRESS.json').write_text(json.dumps(dict(file=f['path'],chunks_complete=i,chunks_total=len(starts),time=time.time())))
                    if i%16==0:print(f['path'],i,'/',len(starts),flush=True)
            os.fsync(fd)
        finally:os.close(fd)
        actual=hashlib.file_digest(part.open('rb'),'sha256').hexdigest()
        if f['lfs_sha256']:assert actual==f['lfs_sha256'],f['path']
        os.replace(part,target);done.append({**f,'actual_sha256':actual});print('VERIFIED',f['path'],flush=True)
    (OUT/'MODEL_DOWNLOAD_COMPLETE.json').write_text(json.dumps(dict(seconds=time.time()-started,transport='official HTTPS ranges',files=done,revision=rev,all_lfs_hashes_match=True,GPU_used=False),indent=2)+'\n')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        (OUT/'HTTP_DOWNLOAD_FAILURE.json').write_text(json.dumps(dict(time=time.time(),error=repr(e)),indent=2));traceback.print_exc();raise
