"""Resumable, bounded parallel HTTP byte ranges; validates server range headers."""
import argparse, concurrent.futures, json, os, pathlib, time
import requests,threading

def download(url, output, workers=16, chunk=2*1024*1024):
    output=pathlib.Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    r=requests.get(url,headers={'Range':'bytes=0-0'},timeout=60);r.raise_for_status()
    if r.status_code!=206: raise RuntimeError('Server must honor byte range')
    total=int(r.headers['Content-Range'].split('/')[-1]);r.close()
    parts=output.parent/(output.name+'.parts2m');parts.mkdir(exist_ok=True)
    old=output.parent/(output.name+'.parts')
    if old.exists():
        for p in old.iterdir():
            if p.name.isdigit() and p.stat().st_size==8*1024*1024:
                data=p.read_bytes()
                for offset in range(0,len(data),chunk):
                    q=parts/str(int(p.name)+offset)
                    if not q.exists():q.write_bytes(data[offset:offset+chunk])
    local=threading.local()
    def fetch(start):
        if not hasattr(local,"session"):local.session=requests.Session()
        end=min(start+chunk,total)-1;p=parts/str(start)
        if p.exists() and p.stat().st_size==end-start+1:return
        for attempt in range(5):
            try:
                r=local.session.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,35))
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}' or len(r.content)!=end-start+1: raise RuntimeError('Range mismatch')
                t=p.with_suffix('.tmp');t.write_bytes(r.content);t.replace(p);return
            except Exception:
                if attempt==4:raise
                time.sleep(2**attempt)
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        for i,future in enumerate(concurrent.futures.as_completed([ex.submit(fetch,start) for start in range(0,total,chunk)])):
            future.result()
            if i%8==0:print(f'{output.name}: {min((i+1)*chunk,total)}/{total}',flush=True)
    with open(output.with_suffix(output.suffix+'.tmp'),'wb') as f:
        for start in range(0,total,chunk):f.write((parts/str(start)).read_bytes())
    output.with_suffix(output.suffix+'.tmp').replace(output)
    print('complete',output,total,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('output');p.add_argument('--workers',type=int,default=16);a=p.parse_args();download(a.url,a.output,a.workers)
