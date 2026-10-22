"""Bounded concurrent HTTP range transfer with exact server-range validation."""
import concurrent.futures,hashlib,json,os,threading,time
from pathlib import Path
import requests
from stvg.tube_runner import write

def transfer(repo,revision,files,target,status_path,workers=12,chunk_size=32*2**20):
    target=Path(target);target.mkdir(parents=True,exist_ok=True);work=target/'.verified_ranges';work.mkdir(exist_ok=True);state_path=work/'completed.json';state=json.loads(state_path.read_text())if state_path.exists()else{};lock=threading.Lock();jobs=[];fds={};done_bytes=sum(x.get('size',0)for x in files if(target/x['rfilename']).is_file()and(target/x['rfilename']).stat().st_size==x['size'])
    for f in files:
        name=f['rfilename'];dest=target/name
        if dest.exists()and dest.stat().st_size==f['size']:continue
        if f['size']<2**20:
            r=requests.get(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',timeout=90);r.raise_for_status();assert len(r.content)==f['size'];dest.write_bytes(r.content);done_bytes+=f['size'];continue
        part=work/(name+'.part');fd=os.open(part,os.O_RDWR|os.O_CREAT,0o600);os.ftruncate(fd,f['size']);fds[name]=(fd,part,dest)
        for start in range(0,f['size'],chunk_size):
            end=min(start+chunk_size,f['size'])-1;key=f'{name}:{start}-{end}'
            if key in state:
                content=os.pread(fd,end-start+1,start)
                if hashlib.sha256(content).hexdigest()==state[key]:done_bytes+=len(content);continue
            jobs.append((f,start,end,key))
    def chunk(job):
        nonlocal done_bytes
        f,start,end,key=job;name=f['rfilename'];n=end-start+1
        for attempt in range(4):
            try:
                url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true&range_start={start}'
                with requests.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},stream=True,timeout=(30,120))as r:
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{f["size"]}':raise RuntimeError(f'Invalid range response {r.status_code} {r.headers.get("Content-Range")}')
                    content=b''.join(r.iter_content(2**20));assert len(content)==n
                assert os.pwrite(fds[name][0],content,start)==n
                with lock:
                    state[key]=hashlib.sha256(content).hexdigest();done_bytes+=n;write(state_path,state);write(status_path,dict(status='downloading_verified_ranges',repo=repo,revision=revision,received_bytes=done_bytes,expected_bytes=sum(x['size']for x in files),completed_ranges=len(state),workers=workers,pid=os.getpid()));print('received_bytes',done_bytes,flush=True)
                return
            except Exception:
                if attempt==3:raise
                time.sleep(min(2**attempt,8))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers)as pool:
            for _ in pool.map(chunk,jobs):pass
        for name,(fd,part,dest)in fds.items():os.fsync(fd);os.close(fd);part.replace(dest)
    finally:
        for fd,_,_ in fds.values():
            try:os.close(fd)
            except OSError:pass
