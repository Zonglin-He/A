"""Pinned public CLIP checkpoint acquisition, safe data only; no model execution.
No credentials or redirected signed URLs are printed or recorded.
"""
from pathlib import Path
import concurrent.futures,hashlib,json,os,requests,threading,time
ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets/clip-vit-base-patch32'
REVISION='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
TOTAL=605247071
EXPECTED='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
CHUNK=4*1024**2
URL=f'https://huggingface.co/openai/clip-vit-base-patch32/resolve/{REVISION}/pytorch_model.bin'

def main():
 ASSETS.mkdir(parents=True,exist_ok=True);target=ASSETS/'pytorch_model.bin';part=ASSETS/'pytorch_model.bin.partial';bitmap=ASSETS/'DOWNLOAD_BLOCKS.json';n=(TOTAL+CHUNK-1)//CHUNK
 if target.exists():
  h=hashlib.sha256()
  with target.open('rb') as f:
   for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
  assert target.stat().st_size==TOTAL and h.hexdigest()==EXPECTED,'Existing complete checkpoint hash mismatch'
  print('Existing complete official checkpoint SHA verified; no download.',flush=True);return
 done=set(json.loads(bitmap.read_text())['done']) if bitmap.exists() and part.exists() else set();fd=os.open(part,os.O_CREAT|os.O_RDWR,0o600);os.ftruncate(fd,TOTAL);lock=threading.Lock();tls=threading.local();started=time.monotonic()
 def fetch(i):
  if not hasattr(tls,'session'):tls.session=requests.Session()
  a=i*CHUNK;b=min(TOTAL,(i+1)*CHUNK)-1
  for attempt in range(5):
   try:
    r=tls.session.get(URL,params={'download':'true','r8block':str(i)},headers={'Range':f'bytes={a}-{b}'},timeout=(20,120));r.raise_for_status()
    assert r.status_code==206 and r.headers.get('content-range')==f'bytes {a}-{b}/{TOTAL}' and len(r.content)==b-a+1
    assert os.pwrite(fd,r.content,a)==len(r.content)
    with lock:
     done.add(i);tmp=bitmap.with_suffix('.tmp');tmp.write_text(json.dumps({'revision':REVISION,'total_bytes':TOTAL,'chunk_bytes':CHUNK,'done':sorted(done)})+'\n');tmp.replace(bitmap)
     if len(done)%10==0:print('verified_download_blocks',len(done),'of',n,flush=True)
    return
   except Exception as e:
    if attempt==4:raise RuntimeError(f'Block {i} failed: {type(e).__name__}') from None
    time.sleep(min(10,attempt+1))
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(fetch,[i for i in range(n) if i not in done]))
 finally:os.close(fd)
 h=hashlib.sha256()
 with part.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert h.hexdigest()==EXPECTED,'Official LFS SHA mismatch'
 part.replace(target)
 receipt={'state':'official_pinned_checkpoint_downloaded_SHA_verified','repo_id':'openai/clip-vit-base-patch32','revision':REVISION,'filename':target.name,'bytes':TOTAL,'sha256':h.hexdigest(),'official_LFS_sha256':EXPECTED,'complete_range_blocks':n,'range_blocks':n,'seconds':time.monotonic()-started,'no_model_or_GPU_execution':True}
 (ASSETS/'ACQUISITION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
