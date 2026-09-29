import urllib.request,pathlib,json,hashlib,time,concurrent.futures,os
root=pathlib.Path('checkpoints/Sa2VA-4B');rev='3fee777d49ee9276eac51ea3e5f9b69e81d09be6'
files=[('model-00001-of-00004.safetensors',4971473960,'6a9eb4ae0e47d9ddaf43a04e2b5771d789f0d19d75ea1d1ead30a7c343fc440b'),('model-00002-of-00004.safetensors',4932952216,'6ed4bb7708ea6ada2c1f2976f744fe0938efcfa3d127a8bd65f5d6a2cdc3b8ee'),('model-00003-of-00004.safetensors',4995688160,'f86024922b4eb082644a90a4befc21d5b6a9e69e47623b55ccec0754e1999cc2'),('model-00004-of-00004.safetensors',259328744,'b6f580096747943665a005a49daca589e2d603cff5d71c81e2a67f4ebb8f8868')]
# Pin and verify the official non-weight files as Git blobs before using remote code.
root.mkdir(parents=True,exist_ok=True)
meta=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/ByteDance/Sa2VA-4B/revision/{rev}?blobs=true',timeout=120));verified={}
for entry in meta['siblings']:
 if 'lfs' in entry:continue
 name=entry['rfilename'];dest=root/name;assert dest.resolve().is_relative_to(root.resolve())
 payload=dest.read_bytes() if dest.exists() else urllib.request.urlopen(f'https://huggingface.co/ByteDance/Sa2VA-4B/resolve/{rev}/{name}',timeout=120).read()
 gitsha=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest();assert gitsha==entry['blobId'],name
 if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
 verified[name]=dict(git_blob_sha1=gitsha,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
(root/'OFFICIAL_CODE_RECEIPT.json').write_text(json.dumps(dict(revision=rev,files=verified),indent=2))

chunk=32*1024**2;tasks=[];begin=time.monotonic();chunks=root/'range_chunks';chunks.mkdir(exist_ok=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
for name,size,expected in files:
 if (root/name).exists():assert digest(root/name)==expected;continue
 old=root/(name+'.partial');prefix=old.stat().st_size if old.exists() else 0
 for lo in range(prefix,size,chunk):tasks.append((name,lo,min(lo+chunk,size)-1))
def fetch(task):
 name,lo,hi=task;out=chunks/(name+f'.{lo}-{hi}');n=hi-lo+1
 if (root/name).exists():return 0
 if out.exists() and out.stat().st_size==n:return n
 for attempt in range(12):
  try:
   tmp=out.with_suffix(out.suffix+'.tmp');have=tmp.stat().st_size if tmp.exists() else 0
   assert 0<=have<=n
   if have==n:tmp.rename(out);return n
   start=lo+have
   url=f'https://huggingface.co/ByteDance/Sa2VA-4B/resolve/{rev}/{name}?range_start={start}&range_end={hi}'
   req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{hi}'})
   with urllib.request.urlopen(req,timeout=120) as inp,tmp.open('ab') as dst:
    assert inp.status==206 and inp.headers.get('Content-Range','').startswith(f'bytes {start}-{hi}/'),dict(inp.headers)
    while b:=inp.read(1024**2):dst.write(b)
   tmp=out.with_suffix(out.suffix+'.tmp');assert tmp.stat().st_size==n;tmp.rename(out);return n
  except Exception as e:
   print('RETRY',name,lo,attempt,type(e).__name__,str(e)[:160],flush=True);time.sleep(min(10,attempt+1))
 raise RuntimeError(task)
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
 total=0
 for i,f in enumerate(concurrent.futures.as_completed([pool.submit(fetch,t) for t in tasks])):
  total+=f.result()
  if i%8==0:print('PROGRESS',i+1,len(tasks),round(total/2**30,3),'GiB',round(time.monotonic()-begin,1),'seconds',flush=True)
receipts=[]
for name,size,expected in files:
 dest=root/name
 if not dest.exists():
  old=root/(name+'.partial');prefix=old.stat().st_size if old.exists() else 0;assembled=root/(name+'.assembling')
  with assembled.open('wb') as dst:
   paths=([old] if prefix else [])+[chunks/(name+f'.{lo}-{min(lo+chunk,size)-1}') for lo in range(prefix,size,chunk)]
   for p in paths:
    with p.open('rb') as src:
     for b in iter(lambda:src.read(8*1024**2),b''):dst.write(b)
  assert assembled.stat().st_size==size and digest(assembled)==expected,name;assembled.rename(dest)
 assert dest.stat().st_size==size and digest(dest)==expected,name
 receipts.append(dict(file=name,bytes=size,sha256=expected));print('VERIFIED',name,flush=True)
(root/'DOWNLOAD_RECEIPT.json').write_text(json.dumps(dict(repo='ByteDance/Sa2VA-4B',revision=rev,files=receipts,seconds=time.monotonic()-begin),indent=2))
