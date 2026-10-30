"""Anonymous official HF artifact; exact ranges, length and LFS SHA."""
import requests,pathlib,json,concurrent.futures,threading,hashlib,time,os
R=pathlib.Path(__file__).resolve().parents[1];D=R/'teacher';api=json.loads((D/'MODEL_API.json').read_text());item=next(x for x in api['siblings'] if x['rfilename']=='pytorch_model.bin');total=item['size'];expected=item['lfs']['sha256'];url=f"https://huggingface.co/laion/clap-htsat-unfused/resolve/{api['sha']}/pytorch_model.bin?download=true";dest=D/'pytorch_model.bin';bitmap=D/'download_bitmap.json';part=8*1024*1024
r=requests.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=60);r.raise_for_status();endpoint=r.url;r.close();done=set(json.loads(bitmap.read_text())) if bitmap.exists() else set();lock=threading.Lock();local=threading.local()
with dest.open('ab') as f:pass
fd=os.open(dest,os.O_WRONLY);os.ftruncate(fd,total)
def get(i):
 if not hasattr(local,'session'):local.session=requests.Session()
 lo=i*part;hi=min(total,(i+1)*part)-1
 for attempt in range(5):
  try:
   x=local.session.get(endpoint,headers={'Range':f'bytes={lo}-{hi}'},timeout=90);x.raise_for_status();assert x.status_code==206 and x.headers.get('Content-Range')==f'bytes {lo}-{hi}/{total}',(x.status_code,x.headers.get('Content-Range'));assert len(x.content)==hi-lo+1
   os.pwrite(fd,x.content,lo)
   with lock:done.add(i);temp=bitmap.with_suffix('.tmp');temp.write_text(json.dumps(sorted(done)));temp.replace(bitmap)
   return i
  except Exception as e:
   if attempt==4:raise RuntimeError(f'chunk{i} failed {type(e).__name__}') from None
   time.sleep(2*(attempt+1))
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for i in pool.map(get,[i for i in range((total+part-1)//part) if i not in done]):print('chunk',i,'complete',flush=True)
os.close(fd);actual=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest();assert actual==expected
(D/'DOWNLOAD_RECEIPT.json').write_text(json.dumps({'state':'completed','source_model':'laion/clap-htsat-unfused','revision':api['sha'],'bytes':total,'sha256':actual,'expected_LFS_sha256_match':True,'license':'Apache-2.0 model card','no_remote_code_executed':True},indent=2));print('COMPLETE',total,actual,flush=True)
