import io,requests,zipfile,json,sys,os,time
from pathlib import Path
class Remote(io.RawIOBase):
 def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.cache={}
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,off,whence=0):
  self.pos=off if whence==0 else self.pos+off if whence==1 else self.size+off
  return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  n=min(n,self.size-self.pos)
  if n<=0:return b''
  key=(self.pos,n)
  disk=Path('/home/wwww/new/crossplatform/downloads/ranges')/str(self.size)/f'{self.pos}_{n}.bin'
  if disk.exists():self.cache[key]=disk.read_bytes()
  if key not in self.cache:
   for attempt in range(4):
    try:
     r=requests.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},timeout=120);r.raise_for_status()
     if r.status_code!=206 or len(r.content)!=n:raise RuntimeError((r.status_code,len(r.content),n))
     self.cache[key]=r.content;disk.parent.mkdir(parents=True,exist_ok=True);disk.write_bytes(r.content);break
    except Exception:
     if attempt==3:raise
     time.sleep(2)
  self.pos+=n
  return self.cache[key]
if __name__=='__main__':
 which=sys.argv[1];size={'dataset':6724434217,'baseline':535349344}[which]
 name={'dataset':'3eed_dataset.zip','baseline':'baseline_ckpt_pred.zip'}[which]
 z=zipfile.ZipFile(Remote('https://huggingface.co/datasets/RRRong/3EED/resolve/main/'+name,size))
 infos=[{'name':a.filename,'size':a.file_size,'compressed':a.compress_size,'offset':a.header_offset,'type':a.compress_type} for a in z.infolist()]
 Path('/home/wwww/new/crossplatform/downloads/'+which+'_zip_index.json').write_text(json.dumps(infos,indent=2))
 print('files',len(infos),'uncompressed',sum(a['size'] for a in infos),flush=True)
 if len(sys.argv)>2:
  names=json.loads(Path(sys.argv[2]).read_text());out=Path(sys.argv[3]).resolve()
  for n in names:
   dest=(out/n).resolve()
   if not dest.is_relative_to(out):raise RuntimeError('Unsafe zip path')
   dest.parent.mkdir(exist_ok=True,parents=True)
   dest.write_bytes(z.read(n));print(n,dest.stat().st_size,flush=True)
