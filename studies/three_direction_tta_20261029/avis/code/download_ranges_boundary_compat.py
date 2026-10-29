import requests,sys,pathlib,concurrent.futures,json,time,os,hashlib,threading
url,dest=sys.argv[1:];p=pathlib.Path(dest);meta=p.with_suffix(p.suffix+'.ranges.json');lock=threading.Lock();chunk=int(os.environ.get('DOWNLOAD_CHUNK_BYTES',1048576))
resolver=requests.Session();r=resolver.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=(30,30));r.raise_for_status();total=int(r.headers['Content-Range'].split('/')[-1]);direct_url=r.url;r.close();thread_local=threading.local()
if meta.exists():state=json.loads(meta.read_text());assert state['total']==total;chunk=state['chunk']
else:
 n=p.stat().st_size if p.exists() else 0;state={'total':total,'chunk':chunk,'done':list(range(n//chunk)),'url':url};meta.write_text(json.dumps(state))
fd=os.open(p,os.O_RDWR|os.O_CREAT);os.ftruncate(fd,total);tic=time.monotonic()
def task(i):
 a=i*chunk;b=min(a+chunk,total)-1;request_b=b+1 if b==0xffffffff and b+1<total else b
 for t in range(8):
  try:
   if not hasattr(thread_local,'session'):
    thread_local.session=requests.Session();thread_local.session.cookies.update(resolver.cookies)
   r=thread_local.session.get(direct_url,headers={'Range':f'bytes={a}-{request_b}'},stream=True,timeout=(30,60))
   if r.status_code==401:
    r.close();r=thread_local.session.get(url,headers={'Range':f'bytes={a}-{request_b}'},stream=True,timeout=(30,60))
   r.raise_for_status();assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {a}-{request_b}/')
   pos=a;received=0;overlap=b''
   for data in r.iter_content(256*1024):
    received+=len(data);use=data[:max(0,b+1-pos)]
    if use:os.pwrite(fd,use,pos);pos+=len(use)
    overlap+=data[len(use):]
   assert pos==b+1 and received==request_b-a+1
   if request_b!=b:
    assert len(overlap)==1
    if (i+1) in state['done']:assert overlap==os.pread(fd,1,b+1)
    print('UINT32_MAX end sentinel workaround verified',i,'overlap_bytes',len(overlap),flush=True)
   with lock:
    state['done'].append(i);meta.write_text(json.dumps(state));print(len(state['done']), (total+chunk-1)//chunk, round(time.monotonic()-tic,1),flush=True)
   return
  except Exception as e:print(i,t,type(e).__name__,str(e),flush=True);time.sleep(3)
 raise RuntimeError(f'chunk {i}')
priority_path=os.environ.get('DOWNLOAD_PRIORITY');priority=set(json.loads(pathlib.Path(priority_path).read_text())) if priority_path else set()
pending=[i for i in range((total+chunk-1)//chunk) if i not in state['done']];pending.sort(key=lambda i:(i not in priority,i))
try:
 with concurrent.futures.ThreadPoolExecutor(int(os.environ.get('DOWNLOAD_WORKERS',8))) as ex:list(ex.map(task,pending))
finally:os.close(fd)
sha=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();p.with_suffix(p.suffix+'.download.json').write_text(json.dumps({'state':'completed','bytes':total,'sha256':sha,'url':url},indent=2));print('complete',sha,flush=True)
